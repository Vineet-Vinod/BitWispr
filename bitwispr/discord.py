from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bitwispr.config import AppConfig, StateStore


def _snowflake_value(value: str | None) -> int:
    try:
        return int(value or "0")
    except ValueError:
        return 0


def _normalize_message_text(value: str) -> str:
    return " ".join(value.split())


@dataclass(slots=True)
class ChannelPollState:
    name: str
    channel_id: str
    last_seen_id: str | None = None
    next_poll_at: float = 0.0
    interval_sec: float = 10.0
    idle_polls: int = 0


@dataclass(slots=True)
class PollResult:
    replied: bool = False
    retry_after_sec: float | None = None


class DiscordWorker:
    def __init__(
        self,
        config: AppConfig,
        state_store: StateStore,
        *,
        llm_chat,
        list_voices,
    ):
        self.config = config
        self.state_store = state_store
        self._llm_chat = llm_chat
        self._list_voices = list_voices
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.self_user_id: str | None = None
        self.self_username = "assistant"
        self.control_last_seen_id: str | None = None
        self.channel_states: dict[str, ChannelPollState] = {}
        self.control_next_poll_at = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.config.discord_auth_token and self.config.control_channel_id)

    def start(self) -> None:
        if not self.enabled or self.thread is not None:
            return
        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="bitwispr-discord",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

    def _run(self) -> None:
        if not self._bootstrap_self_user():
            return

        print(
            "Discord control enabled "
            f"(channel: {self.config.control_channel_id}, poll: {self.config.control_poll_interval_sec:.0f}s)"
        )
        self.control_next_poll_at = time.monotonic()
        self._sync_channel_states(time.monotonic(), reset_schedule=True)

        while not self.stop_event.is_set():
            responder_active = self.state_store.snapshot().responder_active
            next_channel_at = (
                min(
                    (state.next_poll_at for state in self.channel_states.values()),
                    default=float("inf"),
                )
                if responder_active
                else float("inf")
            )
            due_at = min(self.control_next_poll_at, next_channel_at)
            wait_sec = max(0.0, due_at - time.monotonic())
            if self.stop_event.wait(wait_sec):
                break

            now = time.monotonic()
            if now >= self.control_next_poll_at:
                self._poll_control_channel()
                self.control_next_poll_at = now + self.config.control_poll_interval_sec

            self._sync_channel_states(time.monotonic())
            if not self.state_store.snapshot().responder_active:
                continue

            now = time.monotonic()
            due_states = sorted(
                (
                    state
                    for state in self.channel_states.values()
                    if state.next_poll_at <= now + 0.0001
                ),
                key=lambda state: (state.next_poll_at, state.name),
            )
            for state in due_states:
                result = self._poll_response_channel(state)
                self._schedule_next_poll(state, result, time.monotonic())

    def _bootstrap_self_user(self) -> bool:
        while not self.stop_event.is_set():
            try:
                payload = self._discord_request("GET", "/users/@me")
                self.self_user_id = str(payload.get("id", "")).strip() or None
                self.self_username = (
                    str(
                        payload.get("global_name")
                        or payload.get("username")
                        or self.self_username
                    ).strip()
                    or self.self_username
                )
                if self.self_user_id:
                    print(f"Discord authenticated as: {self.self_username}")
                    return True
            except Exception as exc:
                print(f"Discord auth failed: {exc}")
            if self.stop_event.wait(5.0):
                return False
        return False

    def _sync_channel_states(self, now: float, *, reset_schedule: bool = False) -> None:
        channels = self.state_store.snapshot().channels
        active_names = set(channels)

        for name in list(self.channel_states):
            if name not in active_names:
                del self.channel_states[name]

        for name, channel_id in channels.items():
            state = self.channel_states.get(name)
            if state is None:
                self.channel_states[name] = ChannelPollState(
                    name=name,
                    channel_id=channel_id,
                    next_poll_at=now,
                    interval_sec=self.config.responder_poll_interval_sec,
                )
                continue

            state.channel_id = channel_id
            if reset_schedule:
                state.interval_sec = self.config.responder_poll_interval_sec
                state.idle_polls = 0
                state.next_poll_at = now

    def _discord_request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict | list:
        url = f"https://discord.com/api/v9{path}"
        headers = {
            "Authorization": self.config.discord_auth_token,
            "Accept": "application/json",
            "Origin": "https://discord.com",
            "Referer": "https://discord.com/channels/@me",
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Discord/1.0 Safari/537.36"
            ),
        }
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")

        request_obj = urllib.request.Request(
            url,
            method=method,
            headers=headers,
            data=data,
        )
        with urllib.request.urlopen(request_obj, timeout=20) as response:
            return json.loads(response.read().decode("utf-8"))

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
        if message is not None and code is not None:
            return f"{error} | code={code} message={message}", retry_after
        if message is not None:
            return f"{error} | message={message}", retry_after
        return f"{error} | body={raw}", retry_after

    def _state_response(
        self,
        *,
        errors: list[str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        return self.state_store.payload(errors=errors, extra=extra)

    def _format_json_message(self, payload: dict[str, object]) -> str:
        pretty = json.dumps(payload, indent=2, sort_keys=True)
        wrapped = f"```json\n{pretty}\n```"
        if len(wrapped) <= 2000:
            return wrapped

        compact = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        wrapped = f"```json\n{compact}\n```"
        if len(wrapped) <= 2000:
            return wrapped

        return compact[:2000]

    def _post_message(self, channel_id: str, content: str) -> dict | list:
        return self._discord_request(
            "POST",
            f"/channels/{channel_id}/messages",
            payload={"content": content},
        )

    def _fetch_messages(self, channel_id: str) -> list[dict]:
        payload = self._discord_request(
            "GET",
            f"/channels/{channel_id}/messages?limit=15",
        )
        if isinstance(payload, list):
            payload.sort(key=lambda item: _snowflake_value(str(item.get("id", "0"))))
            return payload
        return []

    def _poll_control_channel(self) -> None:
        try:
            messages = self._fetch_messages(self.config.control_channel_id)
        except urllib.error.HTTPError as exc:
            detail, _ = self._http_error_details(exc)
            print(f"Discord control poll failed: {detail}")
            return
        except Exception as exc:
            print(f"Discord control poll error: {exc}")
            return

        if not messages:
            return

        if self.control_last_seen_id is None:
            self.control_last_seen_id = str(messages[-1].get("id", ""))
            return

        for message in messages:
            message_id = str(message.get("id", ""))
            if _snowflake_value(message_id) <= _snowflake_value(self.control_last_seen_id):
                continue

            self.control_last_seen_id = message_id
            content = str(message.get("content", "") or "").strip()
            if not content:
                continue

            for line in content.splitlines():
                command = line.strip()
                if not command:
                    continue
                response = self._apply_control_command(command)
                if response is None:
                    continue
                try:
                    self._post_message(
                        self.config.control_channel_id,
                        self._format_json_message(response),
                    )
                except urllib.error.HTTPError as exc:
                    detail, _ = self._http_error_details(exc)
                    print(f"Discord control reply failed: {detail}")
                except Exception as exc:
                    print(f"Discord control reply error: {exc}")

    def _apply_control_command(self, command: str) -> dict[str, object] | None:
        upper = command.upper()
        if upper == "HELP":
            return self._state_response(
                extra={
                    "help": [
                        "LIST VOICE - TTS list voices",
                        "SET VOICE <voice_id> - set the current voice",
                        "ADD CHANNEL <channel_id> <name> - add a response channel",
                        "LIST CHANNEL - list configured response channels",
                        "DELETE CHANNEL <name> - delete a configured channel by name",
                        "START - enable the Discord responder",
                        "STOP - disable the Discord responder",
                        "SET SPEED <float_value> - clamp to [0.25, 4.0] and set TTS speed",
                        "HELP - print this command list",
                    ]
                }
            )

        if upper in {"LIST VOICE", "LIST VOICES"}:
            try:
                voices = sorted(self._list_voices())
                return self._state_response(extra={"voices": voices})
            except Exception as exc:
                return self._state_response(errors=[str(exc)])

        if upper in {"LIST CHANNEL", "LIST CHANNELS"}:
            return self._state_response()

        if upper == "START":
            if self.state_store.set_responder_active(True):
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            return None

        if upper == "STOP":
            if self.state_store.set_responder_active(False):
                return self._state_response()
            return None

        if upper.startswith("SET VOICE "):
            voice = command[10:].strip()
            if not voice:
                return self._state_response(errors=["SET VOICE requires a voice id."])
            try:
                voices = sorted(self._list_voices())
            except Exception as exc:
                return self._state_response(errors=[str(exc)])
            if voice not in voices:
                return self._state_response(
                    errors=[f"unknown voice id: {voice}"],
                    extra={"voices": voices},
                )
            if self.state_store.set_voice(voice):
                return self._state_response()
            return None

        if upper.startswith("SET SPEED "):
            raw_value = command[10:].strip()
            try:
                speed = float(raw_value)
            except ValueError:
                return self._state_response(errors=[f"invalid speed: {raw_value}"])
            if self.state_store.set_speed(speed):
                return self._state_response()
            return None

        if upper.startswith("ADD CHANNEL "):
            remainder = command[12:].strip()
            parts = remainder.split(None, 1)
            if len(parts) != 2:
                return self._state_response(
                    errors=["ADD CHANNEL requires a channel id and a name."]
                )
            channel_id, name = parts
            changed, error = self.state_store.add_channel(channel_id, name)
            if error is not None:
                return self._state_response(errors=[error])
            if changed:
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            return None

        if upper.startswith("DELETE CHANNEL "):
            name = command[15:].strip()
            if self.state_store.delete_channel(name):
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            return None

        return None

    def _schedule_next_poll(
        self,
        state: ChannelPollState,
        result: PollResult,
        now: float,
    ) -> None:
        base = self.config.responder_poll_interval_sec
        if result.retry_after_sec is not None:
            state.next_poll_at = now + max(base, result.retry_after_sec)
            return

        if result.replied:
            state.interval_sec = base
            state.idle_polls = 0
            state.next_poll_at = now + base
            return

        state.idle_polls += 1
        if state.idle_polls <= self.config.responder_idle_polls_before_backoff:
            state.interval_sec = base
        else:
            state.interval_sec = min(
                self.config.responder_max_poll_interval_sec,
                max(base, state.interval_sec * self.config.responder_backoff_factor),
            )
        state.next_poll_at = now + state.interval_sec

    def _poll_response_channel(self, state: ChannelPollState) -> PollResult:
        result = PollResult()
        try:
            messages = self._fetch_messages(state.channel_id)
            if not messages:
                return result

            newest = messages[-1]
            newest_id = str(newest.get("id", ""))
            if state.last_seen_id is None:
                state.last_seen_id = newest_id
                return result

            if _snowflake_value(newest_id) <= _snowflake_value(state.last_seen_id):
                return result

            state.last_seen_id = newest_id
            newest_author = newest.get("author", {}) or {}
            if str(newest_author.get("id", "")) == self.self_user_id:
                return result

            latest_content = str(newest.get("content", "") or "").strip()
            if not latest_content:
                return result

            lines: list[str] = []
            for message in messages:
                author = message.get("author", {}) or {}
                username = (
                    str(author.get("global_name") or author.get("username") or "unknown")
                    .strip()
                    or "unknown"
                )
                content = str(message.get("content", "") or "").strip()
                if not content:
                    continue
                lines.append(f"{username}: {_normalize_message_text(content)}")

            if not lines:
                return result

            lines.append(f"{self.self_username}:")
            prompt = "\n".join(lines)
            reply = self._llm_chat([{"role": "user", "content": prompt}]).strip()
            if not reply:
                return result

            reply = reply[: self.config.llm_reply_max_chars].strip()
            if not reply:
                return result

            self._post_message(state.channel_id, reply)
            result.replied = True
            return result
        except urllib.error.HTTPError as exc:
            detail, retry_after = self._http_error_details(exc)
            if exc.code == 429:
                print(f"Discord rate limited on channel {state.name}: {detail}")
                result.retry_after_sec = retry_after
                return result
            print(f"Discord channel poll failed for {state.name}: {detail}")
            return result
        except Exception as exc:
            print(f"Discord channel poll error for {state.name}: {exc}")
            return result
