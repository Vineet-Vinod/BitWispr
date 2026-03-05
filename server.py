"""
BitWispr Trillim server.
Runs LLM + Whisper on localhost:1111 with BitNet-TRNQ + GenZ adapter.
"""

import heapq
import json
import os
import random
import signal
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from trillim import LLM, Server, Whisper
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

HOST = "127.0.0.1"
PORT = 1111
MODEL_ID = os.environ.get("BITWISPR_MODEL_ID", "Trillim/BitNet-TRNQ")
ADAPTER_ID = os.environ.get("BITWISPR_ADAPTER_ID", "Trillim/BitNet-GenZ-LoRA-TRNQ")
WHISPER_MODEL = os.environ.get("BITWISPR_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.environ.get("BITWISPR_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.environ.get("BITWISPR_WHISPER_CPU_THREADS", "2"))
RESTART_DELAY_SEC = 2
DISCORD_AUTH_TOKEN = os.environ.get("DISCORD_AUTH_TOKEN", "").strip()
DISCORD_CHANNEL_IDS_RAW = os.environ.get("DISCORD_CHANNEL_IDS", "")
DISCORD_CHANNEL_IDS = [
    channel_id.strip()
    for chunk in DISCORD_CHANNEL_IDS_RAW.replace("\n", ",").split(",")
    for channel_id in [chunk]
    if channel_id.strip()
]
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

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


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
    """Poll Discord channels and auto-reply using local LLM endpoint."""

    def __init__(
        self,
        auth_token: str,
        channel_ids: list[str],
        llm_base_url: str,
        poll_interval_sec: float = 900.0,
        fast_poll_sec: float = 5.0,
        fast_window_sec: float = 300.0,
        backoff_factor: float = 2.0,
        poll_jitter_pct: float = 0.1,
    ):
        self.auth_token = auth_token
        self.channel_ids = channel_ids
        self.llm_base_url = llm_base_url.rstrip("/")
        self.poll_interval_sec = max(3.0, poll_interval_sec)
        self.fast_poll_sec = max(1.0, fast_poll_sec)
        self.fast_window_sec = max(self.fast_poll_sec, fast_window_sec)
        self.backoff_factor = max(1.1, backoff_factor)
        self.poll_jitter_pct = min(0.5, max(0.0, poll_jitter_pct))
        self.self_user_id: str | None = None
        self.channel_states: dict[str, ChannelPollState] = {}
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return bool(self.auth_token and self.channel_ids)

    def start(self) -> None:
        if not self.enabled or self.thread is not None:
            return
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        print(
            "Discord auto-responder enabled "
            "(channels: "
            f"{', '.join(self.channel_ids)}, fast={self.fast_poll_sec:.1f}s, "
            f"window={self.fast_window_sec:.0f}s, max_backoff={self.poll_interval_sec:.1f}s)"
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
        payload = {
            "model": "BitNet-TRNQ",
            "messages": messages,
        }

        req = urllib.request.Request(
            f"{self.llm_base_url}/v1/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))

        text = obj.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        return text[:1800]

    @staticmethod
    def _is_context_overflow_detail(detail: str) -> bool:
        detail_lower = detail.lower()
        return "exceeds context window" in detail_lower or "prompt length" in detail_lower

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
            except urllib.error.HTTPError as e:
                detail, _ = self._http_error_details(e)
                if e.code == 400 and self._is_context_overflow_detail(detail):
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

                    # Doc-aligned second fallback: reset to just the latest message.
                    trimmed_messages += len(history)
                    history = []
                    continue
                raise

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
        for idx, channel_id in enumerate(self.channel_ids):
            state = ChannelPollState(
                channel_id=channel_id,
                next_poll_at=now + (idx * 0.4),  # Stagger startup reads.
                interval_sec=self.fast_poll_sec,
            )
            self.channel_states[channel_id] = state
            heapq.heappush(schedule, (state.next_poll_at, channel_id))

        while not self.stop_event.is_set() and schedule:
            due_at, channel_id = heapq.heappop(schedule)
            state = self.channel_states[channel_id]
            if abs(due_at - state.next_poll_at) > 0.0001:
                continue

            wait_sec = max(0.0, due_at - time.monotonic())
            if self.stop_event.wait(wait_sec):
                break

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


def build_server() -> tuple[Server, str, str]:
    model_dir = resolve_model_dir(MODEL_ID)
    adapter_dir = resolve_model_dir(ADAPTER_ID)
    llm = LLM(model_dir=model_dir, adapter_dir=adapter_dir, num_threads=0)
    whisper = Whisper(
        model_size=WHISPER_MODEL,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
    )
    return Server(llm, whisper), model_dir, adapter_dir


def run_forever():
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)
    llm_base_url = f"http://{HOST}:{PORT}"
    discord_worker = DiscordAutoResponder(
        auth_token=DISCORD_AUTH_TOKEN,
        channel_ids=DISCORD_CHANNEL_IDS,
        llm_base_url=llm_base_url,
        poll_interval_sec=DISCORD_BACKOFF_MAX_SEC,
        fast_poll_sec=DISCORD_FAST_POLL_SEC,
        fast_window_sec=DISCORD_FAST_WINDOW_SEC,
        backoff_factor=DISCORD_BACKOFF_FACTOR,
        poll_jitter_pct=DISCORD_POLL_JITTER_PCT,
    )

    print("=" * 55)
    print("BitWispr Server")
    print("=" * 55)
    print(f"Host: {HOST}:{PORT}")
    print(f"Model ID: {MODEL_ID}")
    print(f"Adapter ID: {ADAPTER_ID}")
    print(f"Whisper model: {WHISPER_MODEL}")
    if discord_worker.enabled:
        print("Discord auto-responder: enabled")
    else:
        print("Discord auto-responder: disabled (.env not configured)")
    print("=" * 55)

    discord_worker.start()

    while not _stop_requested:
        try:
            server, model_dir, adapter_dir = build_server()
            print(f"Resolved model path: {model_dir}")
            print(f"Resolved adapter path: {adapter_dir}")
            print("Starting Trillim server...")
            server.run(host=HOST, port=PORT, log_level="info")
        except Exception as e:
            print(f"Server error: {e}")

        if _stop_requested:
            break

        print(f"Server exited. Restarting in {RESTART_DELAY_SEC}s...")
        time.sleep(RESTART_DELAY_SEC)

    discord_worker.stop()
    print("Server stopped.")


if __name__ == "__main__":
    run_forever()
