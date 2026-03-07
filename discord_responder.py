from __future__ import annotations

import heapq
import json
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from bitwispr_runtime import ContextWindowExceededError


@dataclass(order=True)
class PollJob:
    due_at: float
    channel_id: str = field(compare=False)


@dataclass
class ChannelState:
    channel_id: str
    initialized: bool = False
    cursor_id: str | None = None
    last_reply_id: str | None = None
    interval_sec: float = 5.0
    fast_until: float = 0.0
    scheduled_for: float = 0.0


@dataclass
class PollResult:
    replied: bool = False
    retry_after_sec: float | None = None


class DiscordAutoResponder:
    """Poll Discord channels and reply with one batched response per wake."""

    def __init__(
        self,
        auth_token: str,
        channel_ids: list[str],
        llm_reply_fn: Callable[[list[dict]], str],
        config_channel_id: str | None = None,
        responder_enabled: bool = True,
        poll_interval_sec: float = 900.0,
        fast_poll_sec: float = 5.0,
        fast_window_sec: float = 300.0,
        backoff_factor: float = 2.0,
        poll_jitter_pct: float = 0.1,
    ):
        self.auth_token = auth_token
        self.llm_reply_fn = llm_reply_fn
        self.responder_enabled = responder_enabled
        self.config_channel_id = (config_channel_id or "").strip() or None
        self.default_channel_ids = list(
            dict.fromkeys(
                channel_id
                for channel_id in channel_ids
                if channel_id and channel_id != self.config_channel_id
            )
        )
        self.active_channel_ids = list(self.default_channel_ids)
        self.active_channel_id_set = set(self.default_channel_ids)
        self.poll_interval_sec = max(3.0, poll_interval_sec)
        self.fast_poll_sec = max(1.0, fast_poll_sec)
        self.fast_window_sec = max(self.fast_poll_sec, fast_window_sec)
        self.backoff_factor = max(1.1, backoff_factor)
        self.poll_jitter_pct = min(0.5, max(0.0, poll_jitter_pct))

        self.self_user_id: str | None = None
        self.channel_states: dict[str, ChannelState] = {}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

        self.channel_config_text = ""
        self.channel_config_message_id: str | None = None
        self.runtime_responder_active = True

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
        channels = ", ".join(self.active_channel_ids) if self.active_channel_ids else "(none)"
        print(
            "Discord auto-responder enabled "
            f"(channels: {channels}, fast={self.fast_poll_sec:.1f}s, "
            f"window={self.fast_window_sec:.0f}s, max_backoff={self.poll_interval_sec:.1f}s)"
        )
        if self.config_channel_id:
            print(f"Discord rules channel enabled (channel: {self.config_channel_id})")

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
        except Exception:
            return f"{error} | body={raw}", retry_after

        if payload.get("retry_after") is not None:
            try:
                retry_after = float(payload["retry_after"])
            except (TypeError, ValueError):
                pass
        message = payload.get("message")
        code = payload.get("code")
        detail = payload.get("detail")
        if message is not None and code is not None:
            return f"{error} | code={code} message={message}", retry_after
        if message is not None:
            return f"{error} | message={message}", retry_after
        if isinstance(detail, str):
            return f"{error} | detail={detail}", retry_after
        return f"{error} | body={raw}", retry_after

    def _llm_reply(self, messages: list[dict]) -> str:
        return self.llm_reply_fn(messages)[:1800]

    @staticmethod
    def _message_id(message: dict) -> str:
        return str(message.get("id", "0"))

    @staticmethod
    def _id_value(message_id: str | None) -> int:
        try:
            return int(message_id or "0")
        except (TypeError, ValueError):
            return 0

    @staticmethod
    def _extract_channel_ids(text: str) -> list[str]:
        return re.findall(r"\b\d{15,25}\b", text)

    @staticmethod
    def _parse_responder_command(text: str) -> bool | None:
        command: bool | None = None
        for line in text.splitlines():
            token = re.sub(r"[^a-z]", "", line.strip().lower())
            if token == "start":
                command = True
            elif token == "stop":
                command = False
        return command

    def _parse_active_channels(self, text: str) -> set[str] | None:
        raw = text.strip()
        if not raw:
            return None

        active = set(self.active_channel_id_set)
        saw_directive = False
        content_lines: list[str] = []
        for raw_line in raw.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            if lower in {"start", "stop"}:
                continue
            content_lines.append(line)
            ids = set(self._extract_channel_ids(line))
            if lower.startswith(("set", "only", "channels")):
                active = ids
                saw_directive = True
            elif lower.startswith(("enable", "add")):
                active.update(ids)
                saw_directive = True
            elif lower.startswith(("disable", "remove")):
                active.difference_update(ids)
                saw_directive = True

        if saw_directive:
            return active
        if not content_lines:
            return None

        content = "\n".join(content_lines)
        ids = set(self._extract_channel_ids(content))
        if not ids:
            return None
        cleaned = re.sub(r"\b\d{15,25}\b", "", content)
        cleaned = re.sub(r"[\s,;|]+", "", cleaned)
        return ids if not cleaned else None

    def _set_active_channels(self, channels: set[str], source: str) -> None:
        channels.discard("")
        if self.config_channel_id:
            channels.discard(self.config_channel_id)

        new_set = set(channels)
        if new_set == self.active_channel_id_set:
            return

        for channel_id in self.active_channel_id_set - new_set:
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

    def _schedule(
        self, jobs: list[PollJob], state: ChannelState, due_at: float
    ) -> None:
        state.scheduled_for = due_at
        heapq.heappush(jobs, PollJob(due_at=due_at, channel_id=state.channel_id))

    def _ensure_active_channels(
        self, jobs: list[PollJob], now: float
    ) -> None:
        offset = 0
        for channel_id in self.active_channel_ids:
            if channel_id in self.channel_states:
                continue
            state = ChannelState(channel_id=channel_id, interval_sec=self.fast_poll_sec)
            self.channel_states[channel_id] = state
            self._schedule(jobs, state, now + (offset * 0.4))
            offset += 1

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
            for message in messages:
                author = message.get("author", {}) or {}
                author_id = str(author.get("id", ""))
                if author_id == self.self_user_id or author.get("bot"):
                    continue
                content = self._message_content(message)
                if not content:
                    continue
                latest_message_id = self._message_id(message)
                latest_rules_text = content
                break

            if (
                latest_message_id == self.channel_config_message_id
                and latest_rules_text == self.channel_config_text
            ):
                return

            self.channel_config_message_id = latest_message_id
            self.channel_config_text = latest_rules_text

            responder_command = self._parse_responder_command(latest_rules_text)
            if responder_command is not None:
                self.runtime_responder_active = responder_command
                status = "STARTED" if responder_command else "STOPPED"
                print(
                    "Discord responder "
                    f"{status} from config channel {self.config_channel_id} "
                    f"(message {self.channel_config_message_id})"
                )

            parsed_channels = self._parse_active_channels(latest_rules_text)
            if parsed_channels is None:
                if responder_command is None and latest_rules_text:
                    print(
                        "Discord config ignored from channel "
                        f"{self.config_channel_id}; unsupported format in "
                        f"message {self.channel_config_message_id}"
                    )
                return

            self._set_active_channels(
                parsed_channels,
                source=(
                    f"config channel {self.config_channel_id} "
                    f"(message {self.channel_config_message_id})"
                ),
            )
        except urllib.error.HTTPError as exc:
            detail, _ = self._http_error_details(exc)
            print(f"Discord rules refresh failed: {detail}")
        except Exception as exc:
            print(f"Discord rules refresh error: {exc}")

    def _bootstrap_self_user(self) -> bool:
        try:
            me = self._discord_request("GET", "/users/@me")
            self.self_user_id = str(me.get("id", ""))
            if self.self_user_id:
                print(
                    "Discord auto-responder authenticated as: "
                    f"{me.get('username', 'unknown')}"
                )
                return True
        except Exception as exc:
            print(f"Discord auth bootstrap failed: {exc}")
        return False

    def _jitter_delay(self, delay_sec: float) -> float:
        if self.poll_jitter_pct <= 0 or delay_sec <= self.fast_poll_sec:
            return delay_sec
        span = delay_sec * self.poll_jitter_pct
        return max(self.fast_poll_sec, delay_sec + random.uniform(-span, span))

    def _next_due(self, state: ChannelState, result: PollResult, now: float) -> float:
        if result.retry_after_sec is not None:
            return now + max(self.fast_poll_sec, result.retry_after_sec)
        if result.replied:
            state.interval_sec = self.fast_poll_sec
            state.fast_until = now + self.fast_window_sec
            return now + self.fast_poll_sec
        if now < state.fast_until:
            return now + self.fast_poll_sec
        state.interval_sec = min(
            self.poll_interval_sec,
            max(self.fast_poll_sec, state.interval_sec * self.backoff_factor),
        )
        return now + self._jitter_delay(state.interval_sec)

    def _run(self) -> None:
        while not self.stop_event.is_set():
            if self.self_user_id or self._bootstrap_self_user():
                break
            self.stop_event.wait(5.0)
        if self.stop_event.is_set():
            return

        jobs: list[PollJob] = []
        self.channel_states = {}
        now = time.monotonic()
        self._refresh_dynamic_rules()
        self._ensure_active_channels(jobs, now)

        while not self.stop_event.is_set():
            if not jobs:
                if self.stop_event.wait(self.fast_poll_sec):
                    break
                now = time.monotonic()
                self._refresh_dynamic_rules()
                self._ensure_active_channels(jobs, now)
                continue

            job = heapq.heappop(jobs)
            state = self.channel_states.get(job.channel_id)
            if state is None or abs(state.scheduled_for - job.due_at) > 0.0001:
                continue
            if self.stop_event.wait(max(0.0, job.due_at - time.monotonic())):
                break

            now = time.monotonic()
            self._refresh_dynamic_rules()
            self._ensure_active_channels(jobs, now)
            state = self.channel_states.get(job.channel_id)
            if state is None or job.channel_id not in self.active_channel_id_set:
                continue

            if not self.runtime_responder_active:
                self._schedule(jobs, state, now + self.fast_poll_sec)
                continue

            result = self._poll_channel(state)
            self._schedule(jobs, state, self._next_due(state, result, time.monotonic()))

    @staticmethod
    def _message_content(message: dict) -> str:
        return (message.get("content") or "").strip()

    def _is_self_message(self, message: dict) -> bool:
        author = message.get("author", {}) or {}
        return str(author.get("id", "")) == self.self_user_id

    def _is_human_message(self, message: dict) -> bool:
        author = message.get("author", {}) or {}
        return bool(self._message_content(message) and not author.get("bot") and not self._is_self_message(message))

    def _update_last_reply_id(self, state: ChannelState, recent: list[dict]) -> None:
        own_messages = [self._message_id(message) for message in recent if self._is_self_message(message)]
        if not own_messages:
            return
        newest_reply_id = own_messages[-1]
        if self._id_value(newest_reply_id) > self._id_value(state.last_reply_id):
            state.last_reply_id = newest_reply_id

    def _build_prompt(self, recent: list[dict], state: ChannelState) -> list[dict] | None:
        anchor_id = state.last_reply_id or state.cursor_id
        anchor_value = self._id_value(anchor_id)
        pending = [
            message
            for message in recent
            if self._id_value(self._message_id(message)) > anchor_value
            and self._is_human_message(message)
        ]
        if not pending:
            return None

        batch = "\n".join(self._message_content(message) for message in pending)
        return [{"role": "user", "content": batch}]

    def _generate_reply(self, prompt: list[dict], channel_id: str) -> str:
        working = list(prompt)
        trimmed = 0
        while working:
            try:
                reply = self._llm_reply(working)
                if trimmed:
                    print(
                        f"Trimmed {trimmed} prompt messages in channel "
                        f"{channel_id} to fit context window"
                    )
                return reply
            except ContextWindowExceededError:
                if len(working) == 1:
                    break
                trimmed += 1
                working = working[1:]

        print(
            f"Context overflow on channel {channel_id}; "
            "latest message batch is too large to fit."
        )
        return ""

    def _poll_channel(self, state: ChannelState) -> PollResult:
        result = PollResult()
        channel_id = state.channel_id
        try:
            recent = self._discord_request("GET", f"/channels/{channel_id}/messages?limit=5")
            if not isinstance(recent, list) or not recent:
                return result
            recent.sort(key=lambda message: self._id_value(self._message_id(message)))

            latest_id = self._message_id(recent[-1])
            self._update_last_reply_id(state, recent)
            if not state.initialized:
                state.initialized = True
                state.cursor_id = latest_id
                return result

            prompt = self._build_prompt(recent, state)
            if prompt is None:
                state.cursor_id = latest_id
                return result

            try:
                reply = self._generate_reply(prompt, channel_id)
            except Exception as exc:
                print(f"LLM/Discord handling error for channel {channel_id}: {exc}")
                state.cursor_id = latest_id
                return result

            if not reply:
                state.cursor_id = latest_id
                return result

            try:
                payload = self._discord_request(
                    "POST",
                    f"/channels/{channel_id}/messages",
                    payload={"content": reply},
                )
            except urllib.error.HTTPError as exc:
                detail, retry_after = self._http_error_details(exc)
                print(f"Discord reply failed for channel {channel_id}: {detail}")
                if exc.code == 429:
                    result.retry_after_sec = retry_after
                else:
                    state.cursor_id = latest_id
                return result

            state.cursor_id = latest_id
            if isinstance(payload, dict):
                posted_id = self._message_id(payload)
                if posted_id:
                    state.last_reply_id = posted_id
            result.replied = True
        except urllib.error.HTTPError as exc:
            detail, retry_after = self._http_error_details(exc)
            if exc.code == 429:
                print(f"Discord rate limited on channel {channel_id}: {detail}")
                result.retry_after_sec = retry_after
            else:
                print(f"Discord polling HTTP error on channel {channel_id}: {detail}")
        except Exception as exc:
            print(f"Discord polling error on channel {channel_id}: {exc}")
        return result
