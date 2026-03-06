"""
BitWispr - push-to-talk speech-to-text with optional Discord auto-responder.
Runs fully in-process with shared Trillim LLM + Whisper components.
Press Right Ctrl+Right Alt to toggle recording on/off.

For Wayland: run with sudo or add user to input group:
    sudo usermod -aG input $USER
    (then log out and back in)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import heapq
import io
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal
from trillim import LLM, Whisper
from trillim.model_store import resolve_model_dir


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE pairs into environment without overriding existing vars."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


_load_env_file(Path(__file__).with_name(".env"))

# --- CONFIGURATION ---
MODEL_ID = os.environ.get("BITWISPR_MODEL_ID", "Trillim/BitNet-TRNQ")
ADAPTER_ID = os.environ.get("BITWISPR_ADAPTER_ID", "Trillim/BitNet-GenZ-LoRA-TRNQ")
WHISPER_MODEL = os.environ.get("BITWISPR_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.environ.get("BITWISPR_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.environ.get("BITWISPR_WHISPER_CPU_THREADS", "2"))
WHISPER_LANGUAGE = os.environ.get("BITWISPR_WHISPER_LANGUAGE", "en")
WHISPER_SAMPLE_RATE = 16000
LLM_TIMEOUT_SEC = float(os.environ.get("BITWISPR_LLM_TIMEOUT_SEC", "90"))
KEYBOARD_SCAN_INTERVAL_SEC = 60.0

DISCORD_AUTH_TOKEN = os.environ.get("DISCORD_AUTH_TOKEN", "").strip()
DISCORD_CHANNEL_IDS_RAW = os.environ.get("DISCORD_CHANNEL_IDS", "")
DISCORD_CHANNEL_IDS = [
    channel_id.strip()
    for chunk in DISCORD_CHANNEL_IDS_RAW.replace("\n", ",").split(",")
    for channel_id in [chunk]
    if channel_id.strip()
]
DISCORD_CONFIG_CHANNEL_ID = (
    os.environ.get("DISCORD_CONFIGURATION_CHANNEL_ID")
    or os.environ.get("DISCORD_CONFIG_CHANNEL_ID", "")
).strip()
DISCORD_POLL_INTERVAL_SEC = max(
    3.0, float(os.environ.get("DISCORD_POLL_INTERVAL_SEC", "900"))
)
DISCORD_FAST_POLL_SEC = max(1.0, float(os.environ.get("DISCORD_FAST_POLL_SEC", "5")))
DISCORD_FAST_WINDOW_SEC = max(
    DISCORD_FAST_POLL_SEC, float(os.environ.get("DISCORD_FAST_WINDOW_SEC", "60"))
)
DISCORD_BACKOFF_FACTOR = max(1.1, float(os.environ.get("DISCORD_BACKOFF_FACTOR", "2")))
DISCORD_BACKOFF_MAX_SEC = max(
    DISCORD_FAST_POLL_SEC,
    float(os.environ.get("DISCORD_BACKOFF_MAX_SEC", str(DISCORD_POLL_INTERVAL_SEC))),
)
DISCORD_POLL_JITTER_PCT = min(
    0.5, max(0.0, float(os.environ.get("DISCORD_POLL_JITTER_PCT", "0.1")))
)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return default


DISCORD_RESPONDER_ENABLED = _env_bool("DISCORD_RESPONDER_ENABLED", True)

# Auto-detect device sample rate
try:
    DEVICE_SAMPLE_RATE = int(sd.query_devices(kind="input")["default_samplerate"])
except Exception:
    DEVICE_SAMPLE_RATE = 44100

# Detect display server (check multiple env vars for robustness with sudo)
IS_WAYLAND = (
    os.environ.get("XDG_SESSION_TYPE") == "wayland"
    or os.environ.get("WAYLAND_DISPLAY") is not None
)

# Global state
recording = False
transcribing = False
audio_queue = queue.Queue()
typed_text = ""
transcribe_event = threading.Event()
worker_thread = None
runtime: BitWisprRuntime | None = None


class ContextWindowExceededError(RuntimeError):
    """Raised when prompt is larger than the model context window."""


class BitWisprRuntime:
    """Shared in-process runtime for Trillim LLM + Whisper."""

    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        whisper_model: str,
        whisper_compute_type: str,
        whisper_cpu_threads: int,
        llm_timeout_sec: float,
    ):
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.model_dir = resolve_model_dir(model_id)
        self.adapter_dir = resolve_model_dir(adapter_id) if adapter_id else None
        self.llm_timeout_sec = max(5.0, llm_timeout_sec)

        self.llm = LLM(
            model_dir=self.model_dir,
            adapter_dir=self.adapter_dir,
            num_threads=0,
        )
        self.whisper = Whisper(
            model_size=whisper_model,
            compute_type=whisper_compute_type,
            cpu_threads=whisper_cpu_threads,
        )

        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._start_event = threading.Event()
        self._started = False
        self._start_error: Exception | None = None
        self._chat_lock = threading.Lock()

    async def _async_start(self) -> None:
        await self.llm.start()
        await self.whisper.start()

    async def _async_stop(self) -> None:
        await self.whisper.stop()
        await self.llm.stop()

    def _loop_worker(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
            self._started = True
        except Exception as e:
            self._start_error = e
            try:
                self._loop.run_until_complete(self._async_stop())
            except Exception:
                pass
        finally:
            self._start_event.set()

        if not self._started:
            self._loop.close()
            return

        self._loop.run_forever()
        try:
            self._loop.run_until_complete(self._async_stop())
        finally:
            self._loop.close()

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._loop_worker,
            daemon=True,
            name="bitwispr-runtime",
        )
        self._thread.start()

        if not self._start_event.wait(timeout=600):
            raise RuntimeError("Timed out while initializing runtime")
        if self._start_error is not None:
            raise RuntimeError(str(self._start_error)) from self._start_error
        if not self._started:
            raise RuntimeError("Runtime did not start")

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._started and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        self._thread = None
        self._started = False

    def _run_coroutine(self, coro, timeout: float):
        if not self._started:
            raise RuntimeError("Runtime is not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as e:
            future.cancel()
            raise RuntimeError(f"Timed out after {timeout:.1f}s") from e

    async def _chat_async(self, messages: list[dict]) -> str:
        if self.llm.engine is None or self.llm.harness is None:
            raise RuntimeError("LLM engine is not available")

        prompt_messages = [
            {"role": item["role"], "content": item["content"]} for item in messages
        ]
        token_ids, _ = self.llm.harness._prepare_tokens(prompt_messages)
        max_ctx = self.llm.engine.arch_config.max_position_embeddings
        if len(token_ids) >= max_ctx:
            raise ContextWindowExceededError(
                f"Prompt length ({len(token_ids)} tokens) exceeds context window ({max_ctx})"
            )

        full_text = ""
        sentinels = [
            "[Spin-Jump-Spinning...]",
            "[Searching:",
            "[Synthesizing...]",
            "[Search unavailable]",
            "\n--- Step ",
            "[Search results]\n",
        ]
        async for chunk in self.llm.harness.run(prompt_messages):
            if any(chunk.startswith(sentinel) for sentinel in sentinels):
                continue
            full_text += chunk

        return full_text.strip()

    def chat(self, messages: list[dict]) -> str:
        with self._chat_lock:
            return self._run_coroutine(
                self._chat_async(messages), timeout=self.llm_timeout_sec
            )

    async def _transcribe_async(self, audio_bytes: bytes, language: str | None) -> str:
        if self.whisper.engine is None:
            raise RuntimeError("Whisper engine is not available")
        return (await self.whisper.engine.transcribe(audio_bytes, language=language)).strip()

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        return self._run_coroutine(
            self._transcribe_async(audio_bytes, language),
            timeout=120,
        )


@dataclass
class ChannelPollState:
    channel_id: str
    last_seen_id: str | None = None
    conversation: list[dict] = field(default_factory=list)
    next_poll_at: float = 0.0
    interval_sec: float = DISCORD_FAST_POLL_SEC
    fast_until: float = 0.0
    consecutive_idle_polls: int = 0


@dataclass
class PollOutcome:
    replied: bool = False
    rate_limited: bool = False
    retry_after_sec: float | None = None


class DiscordAutoResponder:
    """Poll Discord channels and auto-reply using in-process LLM runtime."""

    def __init__(
        self,
        auth_token: str,
        channel_ids: list[str],
        llm_reply_fn,
        config_channel_id: str | None = None,
        responder_enabled: bool = True,
        poll_interval_sec: float = 900.0,
        fast_poll_sec: float = 5.0,
        fast_window_sec: float = 300.0,
        backoff_factor: float = 2.0,
        poll_jitter_pct: float = 0.1,
    ):
        self.auth_token = auth_token
        self.responder_enabled = responder_enabled
        self.config_channel_id = (config_channel_id or "").strip() or None
        self.default_channel_ids = [
            channel_id
            for channel_id in channel_ids
            if channel_id != self.config_channel_id
        ]
        self.channel_ids = list(self.default_channel_ids)
        self.active_channel_ids = list(self.default_channel_ids)
        self.active_channel_id_set = set(self.default_channel_ids)
        self.llm_reply_fn = llm_reply_fn
        self.poll_interval_sec = max(3.0, poll_interval_sec)
        self.fast_poll_sec = max(1.0, fast_poll_sec)
        self.fast_window_sec = max(self.fast_poll_sec, fast_window_sec)
        self.backoff_factor = max(1.1, backoff_factor)
        self.poll_jitter_pct = min(0.5, max(0.0, poll_jitter_pct))
        self.self_user_id: str | None = None
        self.channel_states: dict[str, ChannelPollState] = {}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.channel_config_text: str = ""
        self.channel_config_message_id: str | None = None

    @property
    def enabled(self) -> bool:
        return bool(
            self.responder_enabled
            and self.auth_token
            and (self.default_channel_ids or self.config_channel_id)
        )

    def start(self) -> None:
        if not self.enabled or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        active_channels = ", ".join(self.active_channel_ids) if self.active_channel_ids else "(none)"
        print(
            "Discord auto-responder enabled "
            "(channels: "
            f"{active_channels}, fast={self.fast_poll_sec:.1f}s, "
            f"window={self.fast_window_sec:.0f}s, max_backoff={self.poll_interval_sec:.1f}s)"
        )
        if self.config_channel_id:
            print(
                "Discord rules channel enabled "
                f"(channel: {self.config_channel_id})"
            )

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=3)
            self.thread = None

    def _discord_request(
        self, method: str, path: str, payload: dict | None = None
    ) -> dict | list:
        url = f"https://discord.com/api/v9{path}"
        headers = {
            "Authorization": self.auth_token,
            "Accept": "*/*",
            "Origin": "https://discord.com",
            "Referer": "https://discord.com/channels/@me",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Discord/1.0.9165 "
                "Chrome/124.0.6367.243 Electron/30.2.0 Safari/537.36"
            ),
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        req = urllib.request.Request(url, method=method, headers=headers, data=data)
        with urllib.request.urlopen(req, timeout=20) as resp:
            return json.loads(resp.read().decode("utf-8"))

    @staticmethod
    def _http_error_details(error: urllib.error.HTTPError) -> tuple[str, float | None]:
        retry_after: float | None = None
        header_retry_after = error.headers.get("Retry-After")
        if header_retry_after:
            try:
                retry_after = float(header_retry_after)
            except ValueError:
                retry_after = None

        try:
            raw = error.read().decode("utf-8")
        except Exception:
            return str(error), retry_after

        try:
            payload = json.loads(raw)
            message = payload.get("message")
            code = payload.get("code")
            detail_field = payload.get("detail")
            if payload.get("retry_after") is not None:
                try:
                    retry_after = float(payload["retry_after"])
                except (TypeError, ValueError):
                    pass
            if message is not None and code is not None:
                return f"{error} | code={code} message={message}", retry_after
            if message is not None:
                return f"{error} | message={message}", retry_after
            if isinstance(detail_field, str):
                return f"{error} | detail={detail_field}", retry_after
            return f"{error} | body={raw}", retry_after
        except Exception:
            return f"{error} | body={raw}", retry_after

    def _llm_reply(self, messages: list[dict]) -> str:
        return self.llm_reply_fn(messages)[:1800]

    @staticmethod
    def _extract_channel_ids(text: str) -> list[str]:
        return re.findall(r"\b\d{15,25}\b", text)

    def _parse_active_channels(self, text: str) -> set[str] | None:
        raw = text.strip()
        if not raw:
            return set(self.default_channel_ids)

        active = set(self.active_channel_id_set)
        saw_directive = False
        for line in raw.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            lower = line_stripped.lower()
            ids = set(self._extract_channel_ids(line_stripped))
            if lower.startswith(("set", "only", "channels")):
                active = set(ids)
                saw_directive = True
            elif lower.startswith(("enable", "add")):
                active.update(ids)
                saw_directive = True
            elif lower.startswith(("disable", "remove")):
                active.difference_update(ids)
                saw_directive = True

        if saw_directive:
            return active

        ids = set(self._extract_channel_ids(raw))
        if not ids:
            return None
        cleaned = re.sub(r"\b\d{15,25}\b", "", raw)
        cleaned = re.sub(r"[\s,;|]+", "", cleaned)
        if cleaned:
            return None
        return ids

    def _set_active_channels(self, channels: set[str], source: str) -> None:
        channels.discard("")
        if self.config_channel_id:
            channels.discard(self.config_channel_id)

        new_set = set(channels)
        if new_set == self.active_channel_id_set:
            return

        removed = self.active_channel_id_set - new_set
        for channel_id in removed:
            self.channel_states.pop(channel_id, None)

        self.active_channel_id_set = new_set
        self.active_channel_ids = sorted(new_set)
        if self.active_channel_ids:
            print(
                "Discord active channels updated from "
                f"{source}: {', '.join(self.active_channel_ids)}"
            )
        else:
            print(f"Discord active channels updated from {source}: (none)")

    def _ensure_active_channels_scheduled(
        self, schedule: list[tuple[float, str]], now: float
    ) -> None:
        for idx, channel_id in enumerate(self.active_channel_ids):
            if channel_id in self.channel_states:
                continue
            state = ChannelPollState(
                channel_id=channel_id,
                next_poll_at=now + (idx * 0.4),  # Stagger startup reads.
                interval_sec=self.fast_poll_sec,
            )
            self.channel_states[channel_id] = state
            heapq.heappush(schedule, (state.next_poll_at, channel_id))

    def _refresh_dynamic_rules(self) -> None:
        if not self.config_channel_id:
            return

        try:
            messages = self._discord_request(
                "GET", f"/channels/{self.config_channel_id}/messages?limit=25"
            )
            if not isinstance(messages, list):
                return

            latest_message_id: str | None = None
            latest_rules_text = ""
            for msg in messages:
                author = msg.get("author", {}) or {}
                author_id = str(author.get("id", ""))
                if author_id == self.self_user_id or author.get("bot"):
                    continue
                content = (msg.get("content") or "").strip()
                if not content:
                    continue
                latest_message_id = str(msg.get("id", ""))
                latest_rules_text = content
                break

            if (
                latest_message_id == self.channel_config_message_id
                and latest_rules_text == self.channel_config_text
            ):
                return

            self.channel_config_message_id = latest_message_id
            self.channel_config_text = latest_rules_text
            parsed_channels = self._parse_active_channels(latest_rules_text)
            if parsed_channels is None:
                print(
                    "Discord config ignored from channel "
                    f"{self.config_channel_id}; unsupported format in "
                    f"message {self.channel_config_message_id}"
                )
                return

            source = (
                f"config channel {self.config_channel_id} "
                f"(message {self.channel_config_message_id})"
            )
            self._set_active_channels(parsed_channels, source=source)
        except urllib.error.HTTPError as e:
            detail, _ = self._http_error_details(e)
            print(f"Discord rules refresh failed: {detail}")
        except Exception as e:
            print(f"Discord rules refresh error: {e}")

    def _llm_reply_with_context_fallback(
        self, state: ChannelPollState, user_message: dict
    ) -> tuple[str, bool]:
        history = list(state.conversation)
        trimmed_messages = 0
        did_aggressive_trim = False

        while True:
            prompt_messages = [*history, user_message]
            try:
                reply = self._llm_reply(prompt_messages)
                if trimmed_messages > 0:
                    state.conversation = history
                    print(
                        f"Trimmed {trimmed_messages} history messages in channel "
                        f"{state.channel_id} to fit context window"
                    )
                return reply, True
            except ContextWindowExceededError:
                if not history:
                    print(
                        f"Context overflow on channel {state.channel_id}; "
                        "latest message is too large to fit."
                    )
                    return "", False

                if not did_aggressive_trim:
                    # First fallback: cut out 90% of old context and keep newest 10%.
                    keep_count = max(1, int(len(history) * 0.1))
                    if keep_count >= len(history):
                        keep_count = len(history) - 1
                    if keep_count > 0:
                        removed = len(history) - keep_count
                        history = history[-keep_count:]
                    else:
                        removed = len(history)
                        history = []
                    trimmed_messages += removed
                    did_aggressive_trim = True
                    continue

                # Second fallback: reset to just the latest message.
                trimmed_messages += len(history)
                history = []
                continue

    def _bootstrap_self_user(self) -> bool:
        try:
            me = self._discord_request("GET", "/users/@me")
            self.self_user_id = str(me.get("id", ""))
            username = me.get("username", "unknown")
            if self.self_user_id:
                print(f"Discord auto-responder authenticated as: {username}")
                return True
        except Exception as e:
            print(f"Discord auth bootstrap failed: {e}")
        return False

    def _run(self) -> None:
        # Authenticate first; skip polling if token is not valid yet.
        while not self.stop_event.is_set():
            if not self.self_user_id:
                self._bootstrap_self_user()
                self.stop_event.wait(5.0)
                continue
            break

        if self.stop_event.is_set():
            return

        now = time.monotonic()
        schedule: list[tuple[float, str]] = []
        self.channel_states = {}
        self._ensure_active_channels_scheduled(schedule, now)
        self._refresh_dynamic_rules()
        self._ensure_active_channels_scheduled(schedule, time.monotonic())

        while not self.stop_event.is_set():
            if not schedule:
                if self.stop_event.wait(self.fast_poll_sec):
                    break
                self._refresh_dynamic_rules()
                self._ensure_active_channels_scheduled(schedule, time.monotonic())
                continue

            due_at, channel_id = heapq.heappop(schedule)
            state = self.channel_states.get(channel_id)
            if state is None:
                continue
            if abs(due_at - state.next_poll_at) > 0.0001:
                continue

            wait_sec = max(0.0, due_at - time.monotonic())
            if self.stop_event.wait(wait_sec):
                break

            # Always refresh rules first before processing any channel poll wake.
            self._refresh_dynamic_rules()
            self._ensure_active_channels_scheduled(schedule, time.monotonic())
            if channel_id not in self.active_channel_id_set:
                continue
            outcome = self._process_channel(state)
            now = time.monotonic()
            self._schedule_next_poll(state, outcome, now)
            heapq.heappush(schedule, (state.next_poll_at, channel_id))

    def _jitter_delay(self, delay_sec: float) -> float:
        if self.poll_jitter_pct <= 0 or delay_sec <= self.fast_poll_sec:
            return delay_sec
        span = delay_sec * self.poll_jitter_pct
        return max(self.fast_poll_sec, delay_sec + random.uniform(-span, span))

    def _schedule_next_poll(
        self, state: ChannelPollState, outcome: PollOutcome, now: float
    ) -> None:
        if outcome.rate_limited:
            retry_after = outcome.retry_after_sec or self.fast_poll_sec
            state.next_poll_at = now + max(self.fast_poll_sec, retry_after)
            return

        if outcome.replied:
            # Reset fast window every time we successfully reply.
            state.fast_until = now + self.fast_window_sec
            state.interval_sec = self.fast_poll_sec
            state.consecutive_idle_polls = 0
            state.next_poll_at = now + self.fast_poll_sec
            return

        if now < state.fast_until:
            state.next_poll_at = now + self.fast_poll_sec
            return

        state.consecutive_idle_polls += 1
        state.interval_sec = min(
            self.poll_interval_sec,
            max(self.fast_poll_sec, state.interval_sec * self.backoff_factor),
        )
        state.next_poll_at = now + self._jitter_delay(state.interval_sec)

    def _process_channel(self, state: ChannelPollState) -> PollOutcome:
        channel_id = state.channel_id
        outcome = PollOutcome()
        try:
            messages = self._discord_request(
                "GET", f"/channels/{channel_id}/messages?limit=5"
            )
            if not isinstance(messages, list) or not messages:
                return outcome

            # Process oldest -> newest.
            messages.sort(key=lambda item: int(item.get("id", "0")))

            current_last_seen = state.last_seen_id
            if current_last_seen is None:
                state.last_seen_id = messages[-1]["id"]
                return outcome

            for msg in messages:
                msg_id = str(msg.get("id", "0"))
                if int(msg_id) <= int(current_last_seen):
                    continue

                state.last_seen_id = msg_id
                current_last_seen = msg_id
                author = msg.get("author", {}) or {}
                author_id = str(author.get("id", ""))
                if author_id == self.self_user_id or author.get("bot"):
                    continue

                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                user_message = {"role": "user", "content": content}
                try:
                    reply, should_track_user = self._llm_reply_with_context_fallback(
                        state, user_message
                    )
                    if not should_track_user:
                        continue
                    state.conversation.append(user_message)
                    if not reply:
                        continue
                    self._discord_request(
                        "POST",
                        f"/channels/{channel_id}/messages",
                        payload={"content": reply},
                    )
                    state.conversation.append({"role": "assistant", "content": reply})
                    outcome.replied = True
                except urllib.error.HTTPError as e:
                    detail, retry_after = self._http_error_details(e)
                    print(f"Discord reply failed for channel {channel_id}: {detail}")
                    if e.code == 429:
                        outcome.rate_limited = True
                        outcome.retry_after_sec = retry_after
                        break
                except Exception as e:
                    print(f"LLM/Discord handling error for channel {channel_id}: {e}")

        except urllib.error.HTTPError as e:
            if e.code == 429:
                detail, retry_after = self._http_error_details(e)
                print(f"Discord rate limited on channel {channel_id}: {detail}")
                outcome.rate_limited = True
                outcome.retry_after_sec = retry_after
            else:
                detail, _ = self._http_error_details(e)
                print(f"Discord polling HTTP error on channel {channel_id}: {detail}")
        except Exception as e:
            print(f"Discord polling error on channel {channel_id}: {e}")
        return outcome


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from original sample rate to target sample rate."""
    if orig_sr == target_sr:
        return audio_data

    num_samples = int(len(audio_data) * target_sr / orig_sr)
    resampled = signal.resample(audio_data, num_samples)
    return resampled.astype(np.float32)


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    """Convert float32 mono audio [-1, 1] into in-memory WAV bytes."""
    clipped = np.clip(audio_data, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


def transcribe_with_runtime(audio_data_16k: np.ndarray) -> str:
    """Transcribe WAV audio using in-process Whisper runtime."""
    if runtime is None:
        raise RuntimeError("Runtime is not initialized")

    audio_bytes = audio_to_wav_bytes(audio_data_16k, WHISPER_SAMPLE_RATE)
    return runtime.transcribe(audio_bytes, language=WHISPER_LANGUAGE)


def type_text(text: str):
    """Type text at cursor position. Works on both X11 and Wayland."""
    if not text:
        return

    if IS_WAYLAND:
        try:
            result = subprocess.run(
                ["ydotool", "type", "--", text],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(
                ["wtype", "--", text],
                capture_output=True,
                timeout=30,
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            pass

        print("⚠️  Could not type text. Install ydotool:")
        print("   sudo apt install ydotool")
        print(f"   Text was: {text}")
        return

    try:
        subprocess.run(
            ["xdotool", "type", "--clearmodifiers", "--", text],
            check=True,
            capture_output=True,
            timeout=5,
        )
    except FileNotFoundError:
        print("⚠️  xdotool not found. Install with: sudo apt install xdotool")
        print(f"   Text was: {text}")


def transcription_worker():
    """Persistent worker thread that waits for transcription requests."""
    global typed_text, transcribing

    while True:
        transcribe_event.wait()
        transcribe_event.clear()
        transcribing = True

        audio_buffer = []
        try:
            while True:
                chunk = audio_queue.get_nowait()
                audio_buffer.extend(chunk.tolist())
        except queue.Empty:
            pass

        if len(audio_buffer) > DEVICE_SAMPLE_RATE * 0.3:
            print("🔄 Transcribing...")
            final_audio = np.array(audio_buffer, dtype=np.float32)
            final_audio_16k = resample_audio(
                final_audio, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE
            )
            try:
                final_text = transcribe_with_runtime(final_audio_16k)
                if final_text:
                    print(f"✅ {final_text}")
                    type_text(final_text + " ")
                    typed_text = final_text
                else:
                    print("No speech detected.")
            except Exception as e:
                print(f"Transcription error: {e}")
        else:
            print("Recording too short.")

        transcribing = False


def audio_callback(indata, frames, time_info, status):
    """Callback for audio stream - adds audio to queue."""
    if recording:
        audio_queue.put(indata[:, 0].copy())


def start_recording():
    """Start recording audio."""
    global recording, typed_text, worker_thread

    if transcribing:
        print("⏳ Please wait, transcription in progress...")
        return False

    if worker_thread is None:
        worker_thread = threading.Thread(target=transcription_worker, daemon=True)
        worker_thread.start()

    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break

    typed_text = ""
    recording = True
    print("\n🎙️  Recording started... (Press Right Ctrl+Right Alt to stop)")
    return True


def stop_recording():
    """Stop recording and trigger transcription."""
    global recording
    recording = False
    print("⏹️  Recording stopped.\n")
    transcribe_event.set()


def run_with_evdev():
    """Use evdev for keyboard listening (works on Wayland with proper permissions)."""
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        print("❌ evdev not installed. Run: uv add evdev")
        sys.exit(1)

    stream = sd.InputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(DEVICE_SAMPLE_RATE * 0.2),
    )
    stream.start()

    print("Listening for Right Ctrl+Right Alt...\n")

    ctrl_keys = {ecodes.KEY_RIGHTCTRL}
    alt_keys = {ecodes.KEY_RIGHTALT}
    if hasattr(ecodes, "KEY_ALTGR"):
        alt_keys.add(ecodes.KEY_ALTGR)
    if hasattr(ecodes, "KEY_ISO_LEVEL3_SHIFT"):
        alt_keys.add(ecodes.KEY_ISO_LEVEL3_SHIFT)

    def toggle_recorder():
        if recording:
            print(f"\n{'=' * 40}\n⏹️  STOPPED RECORDING\n{'=' * 40}")
            stop_recording()
        else:
            if start_recording():
                print("=" * 40)

    try:
        from selectors import DefaultSelector, EVENT_READ

        selector = DefaultSelector()
        devices: dict[str, evdev.InputDevice] = {}
        state: dict[str, dict[str, bool]] = {}

        def is_keyboard_device(dev: evdev.InputDevice) -> bool:
            try:
                caps = dev.capabilities()
            except OSError:
                return False
            if ecodes.EV_KEY not in caps:
                return False
            key_caps = set(caps.get(ecodes.EV_KEY, []))
            required = {
                ecodes.KEY_A,
                ecodes.KEY_Z,
                ecodes.KEY_SPACE,
                ecodes.KEY_RIGHTCTRL,
                ecodes.KEY_RIGHTALT,
            }
            return bool(key_caps.intersection(required))

        def add_new_keyboards() -> None:
            for path in evdev.list_devices():
                if path in devices:
                    continue
                try:
                    dev = evdev.InputDevice(path)
                except OSError:
                    continue
                if not is_keyboard_device(dev):
                    continue
                try:
                    selector.register(dev, EVENT_READ)
                except Exception:
                    dev.close()
                    continue
                devices[path] = dev
                state[path] = {"ctrl": False, "alt": False, "latched": False}
                print(f"  + Keyboard: {dev.name} ({path})")

        def remove_keyboard(path: str) -> None:
            dev = devices.pop(path, None)
            state.pop(path, None)
            if dev is None:
                return
            try:
                selector.unregister(dev)
            except Exception:
                pass
            try:
                dev.close()
            except Exception:
                pass

        add_new_keyboards()
        if not devices:
            print("⚠️  No keyboard detected yet. Waiting for devices...")

        last_scan = time.monotonic()
        while True:
            now = time.monotonic()
            if now - last_scan >= KEYBOARD_SCAN_INTERVAL_SEC:
                add_new_keyboards()
                last_scan = now
            for key, _ in selector.select(timeout=1.0):
                device = key.fileobj
                path = getattr(device, "path", "")
                if not path or path not in state:
                    continue
                try:
                    events = device.read()
                except OSError:
                    print(f"  - Keyboard disconnected: {device.name} ({path})")
                    remove_keyboard(path)
                    continue
                for event in events:
                    if event.type != ecodes.EV_KEY:
                        continue

                    is_key_down = event.value > 0
                    if event.code in ctrl_keys:
                        state[path]["ctrl"] = is_key_down
                    elif event.code in alt_keys:
                        state[path]["alt"] = is_key_down
                    else:
                        continue

                    combo_down = state[path]["ctrl"] and state[path]["alt"]
                    if combo_down and event.value == 1 and not state[path]["latched"]:
                        toggle_recorder()
                        state[path]["latched"] = True
                    elif not combo_down:
                        state[path]["latched"] = False
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop()
        stream.close()


def run_with_pynput():
    from pynput import keyboard

    state = {"ctrl_r": False, "alt_r": False, "combo_latched": False}
    ctrl_keys = {keyboard.Key.ctrl_r}
    alt_keys = {
        keyboard.Key.alt_r,
        keyboard.Key.alt_gr,
    }

    def toggle_recorder():
        if recording:
            print(f"\n{'=' * 40}\n⏹️  STOPPED RECORDING\n{'=' * 40}")
            stop_recording()
        else:
            if start_recording():
                print("=" * 40)

    def update_state(key, is_pressed: bool):
        if key in ctrl_keys:
            state["ctrl_r"] = is_pressed
        elif key in alt_keys:
            state["alt_r"] = is_pressed
        else:
            return

        combo_down = state["ctrl_r"] and state["alt_r"]
        if combo_down and is_pressed and not state["combo_latched"]:
            toggle_recorder()
            state["combo_latched"] = True
        elif not combo_down:
            state["combo_latched"] = False

    def on_press(key):
        update_state(key, True)

    def on_release(key):
        update_state(key, False)

    print("Listening for Right Ctrl+Right Alt...\n")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nExiting...")


def main():
    global runtime

    print("=" * 55)
    print("BitWispr - Real-time Speech to Text")
    print("=" * 55)
    print("Architecture: in-process Trillim runtime (no local HTTP server)")
    print(f"Display server: {'Wayland' if IS_WAYLAND else 'X11'}")
    print(f"Audio sample rate: {DEVICE_SAMPLE_RATE} Hz")
    print(f"Model ID: {MODEL_ID}")
    print(f"Adapter ID: {ADAPTER_ID}")
    print(f"Whisper model: {WHISPER_MODEL}")

    try:
        runtime = BitWisprRuntime(
            model_id=MODEL_ID,
            adapter_id=ADAPTER_ID,
            whisper_model=WHISPER_MODEL,
            whisper_compute_type=WHISPER_COMPUTE_TYPE,
            whisper_cpu_threads=WHISPER_CPU_THREADS,
            llm_timeout_sec=LLM_TIMEOUT_SEC,
        )
        runtime.start()
        print(f"Resolved model path: {runtime.model_dir}")
        print(f"Resolved adapter path: {runtime.adapter_dir}")
    except Exception as e:
        print(f"❌ Failed to initialize Trillim runtime: {e}")
        sys.exit(1)

    discord_worker = DiscordAutoResponder(
        auth_token=DISCORD_AUTH_TOKEN,
        channel_ids=DISCORD_CHANNEL_IDS,
        llm_reply_fn=runtime.chat,
        config_channel_id=DISCORD_CONFIG_CHANNEL_ID,
        responder_enabled=DISCORD_RESPONDER_ENABLED,
        poll_interval_sec=DISCORD_BACKOFF_MAX_SEC,
        fast_poll_sec=DISCORD_FAST_POLL_SEC,
        fast_window_sec=DISCORD_FAST_WINDOW_SEC,
        backoff_factor=DISCORD_BACKOFF_FACTOR,
        poll_jitter_pct=DISCORD_POLL_JITTER_PCT,
    )

    if not DISCORD_RESPONDER_ENABLED:
        print("Discord auto-responder: disabled (DISCORD_RESPONDER_ENABLED=false)")
    elif discord_worker.enabled:
        print("Discord auto-responder: enabled")
    else:
        print("Discord auto-responder: disabled (.env not configured)")

    print("Hotkey: Right Ctrl+Right Alt (toggle recording on/off)")
    print("=" * 55 + "\n")

    discord_worker.start()
    try:
        if IS_WAYLAND:
            print("Using evdev for Wayland keyboard input...")
            print("(If this fails, run with sudo or add yourself to input group)\n")
            run_with_evdev()
        else:
            print("Using pynput for X11 keyboard input...\n")
            run_with_pynput()
    finally:
        discord_worker.stop()
        if runtime is not None:
            runtime.stop()
            runtime = None
        print("BitWispr stopped.")


if __name__ == "__main__":
    main()
