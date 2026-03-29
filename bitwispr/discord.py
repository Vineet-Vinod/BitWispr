from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from bitwispr.config import AppConfig, StateStore

logger = logging.getLogger(__name__)


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
            if not self.enabled:
                logger.info(
                    "Discord worker disabled (token configured=%s, control channel configured=%s)",
                    bool(self.config.discord_auth_token),
                    bool(self.config.control_channel_id),
                )
            return
        self.thread = threading.Thread(
            target=self._run_thread,
            daemon=True,
            name="bitwispr-discord",
        )
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=5)
            self.thread = None

    def _run_thread(self) -> None:
        try:
            self._run()
        except Exception:
            logger.exception("Discord worker thread crashed")

    def _run(self) -> None:
        if not self._bootstrap_self_user():
            logger.warning("Discord bootstrap did not complete; worker exiting")
            return

        logger.info(
            "Discord control enabled (channel=%s poll=%.0fs)",
            self.config.control_channel_id,
            self.config.control_poll_interval_sec,
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
                logger.debug("Polling control channel")
                self._poll_control_channel()
                self.control_next_poll_at = now + self.config.control_poll_interval_sec

            self._sync_channel_states(time.monotonic())
            if not self.state_store.snapshot().responder_active:
                logger.debug("Discord responder is stopped; skipping response-channel polls")
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
                logger.debug(
                    "Polling response channel %s (%s) interval=%.1fs idle_polls=%s",
                    state.name,
                    state.channel_id,
                    state.interval_sec,
                    state.idle_polls,
                )
                result = self._poll_response_channel(state)
                self._schedule_next_poll(state, result, time.monotonic())

    def _bootstrap_self_user(self) -> bool:
        while not self.stop_event.is_set():
            try:
                logger.info("Authenticating Discord REST client via /users/@me")
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
                    logger.info(
                        "Discord authenticated as %s (%s)",
                        self.self_username,
                        self.self_user_id,
                    )
                    return True
            except Exception:
                logger.exception("Discord auth bootstrap failed")
            if self.stop_event.wait(5.0):
                return False
        return False

    def _sync_channel_states(self, now: float, *, reset_schedule: bool = False) -> None:
        channels = self.state_store.snapshot().channels
        active_names = set(channels)

        for name in list(self.channel_states):
            if name not in active_names:
                logger.info("Removing Discord response channel %s", name)
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
                logger.info(
                    "Added Discord response channel %s -> %s",
                    name,
                    channel_id,
                )
                continue

            state.channel_id = channel_id
            if reset_schedule:
                state.interval_sec = self.config.responder_poll_interval_sec
                state.idle_polls = 0
                state.next_poll_at = now
                logger.debug(
                    "Reset poll schedule for channel %s -> %s",
                    name,
                    channel_id,
                )

    def _discord_request(
        self,
        method: str,
        path: str,
        payload: dict[str, object] | None = None,
    ) -> dict | list:
        url = f"https://discord.com/api/v9{path}"
        logger.debug("Discord request %s %s", method, path)
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
            body = response.read().decode("utf-8")
            logger.debug(
                "Discord response %s %s -> HTTP %s (%s bytes)",
                method,
                path,
                getattr(response, "status", "?"),
                len(body),
            )
            return json.loads(body)

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
        logger.info(
            "Posting Discord message to channel %s (%s chars)",
            channel_id,
            len(content),
        )
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
            logger.debug(
                "Fetched %s Discord messages from channel %s",
                len(payload),
                channel_id,
            )
            return payload
        logger.warning(
            "Discord message fetch for channel %s did not return a list: %s",
            channel_id,
            type(payload).__name__,
        )
        return []

    def _poll_control_channel(self) -> None:
        try:
            messages = self._fetch_messages(self.config.control_channel_id)
        except urllib.error.HTTPError as exc:
            detail, _ = self._http_error_details(exc)
            logger.error("Discord control poll failed: %s", detail)
            return
        except Exception:
            logger.exception("Discord control poll crashed")
            return

        if not messages:
            logger.debug("Control channel returned no messages")
            return

        if self.control_last_seen_id is None:
            self.control_last_seen_id = str(messages[-1].get("id", ""))
            logger.info(
                "Initialized control channel cursor at message %s",
                self.control_last_seen_id,
            )
            return

        for message in messages:
            message_id = str(message.get("id", ""))
            if _snowflake_value(message_id) <= _snowflake_value(self.control_last_seen_id):
                continue

            self.control_last_seen_id = message_id
            content = str(message.get("content", "") or "").strip()
            if not content:
                logger.debug("Skipping empty control message %s", message_id)
                continue

            for line in content.splitlines():
                command = line.strip()
                if not command:
                    continue
                logger.info("Processing control command from message %s: %r", message_id, command)
                response = self._apply_control_command(command)
                if response is None:
                    logger.debug("Ignoring non-command control line: %r", command)
                    continue
                try:
                    self._post_message(
                        self.config.control_channel_id,
                        self._format_json_message(response),
                    )
                except urllib.error.HTTPError as exc:
                    detail, _ = self._http_error_details(exc)
                    logger.error("Discord control reply failed: %s", detail)
                except Exception:
                    logger.exception("Discord control reply failed unexpectedly")

    def _apply_control_command(self, command: str) -> dict[str, object] | None:
        upper = command.upper()
        if upper == "HELP":
            logger.info("Control command matched HELP")
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
                logger.info("Control command matched LIST VOICE")
                voices = sorted(self._list_voices())
                return self._state_response(extra={"voices": voices})
            except Exception as exc:
                logger.exception("LIST VOICE failed")
                return self._state_response(errors=[str(exc)])

        if upper in {"LIST CHANNEL", "LIST CHANNELS"}:
            logger.info("Control command matched LIST CHANNEL")
            return self._state_response()

        if upper == "START":
            if self.state_store.set_responder_active(True):
                logger.info("Control command matched START; responder enabled")
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            logger.info("Control command START was a no-op; responder already enabled")
            return None

        if upper == "STOP":
            if self.state_store.set_responder_active(False):
                logger.info("Control command matched STOP; responder disabled")
                return self._state_response()
            logger.info("Control command STOP was a no-op; responder already disabled")
            return None

        if upper.startswith("SET VOICE "):
            voice = command[10:].strip()
            if not voice:
                logger.warning("SET VOICE missing voice id")
                return self._state_response(errors=["SET VOICE requires a voice id."])
            try:
                voices = sorted(self._list_voices())
            except Exception as exc:
                logger.exception("SET VOICE failed while listing voices")
                return self._state_response(errors=[str(exc)])
            if voice not in voices:
                logger.warning("SET VOICE requested unknown voice %s", voice)
                return self._state_response(
                    errors=[f"unknown voice id: {voice}"],
                    extra={"voices": voices},
                )
            if self.state_store.set_voice(voice):
                logger.info("Control command set voice to %s", voice)
                return self._state_response()
            logger.info("SET VOICE was a no-op; voice already %s", voice)
            return None

        if upper.startswith("SET SPEED "):
            raw_value = command[10:].strip()
            try:
                speed = float(raw_value)
            except ValueError:
                logger.warning("SET SPEED received invalid value %r", raw_value)
                return self._state_response(errors=[f"invalid speed: {raw_value}"])
            if self.state_store.set_speed(speed):
                logger.info("Control command set speed to %s", speed)
                return self._state_response()
            logger.info("SET SPEED was a no-op after clamping")
            return None

        if upper.startswith("ADD CHANNEL "):
            remainder = command[12:].strip()
            parts = remainder.split(None, 1)
            if len(parts) != 2:
                logger.warning("ADD CHANNEL missing channel id or name")
                return self._state_response(
                    errors=["ADD CHANNEL requires a channel id and a name."]
                )
            channel_id, name = parts
            changed, error = self.state_store.add_channel(channel_id, name)
            if error is not None:
                logger.warning("ADD CHANNEL failed: %s", error)
                return self._state_response(errors=[error])
            if changed:
                logger.info("Added response channel %s -> %s", name, channel_id)
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            return None

        if upper.startswith("DELETE CHANNEL "):
            name = command[15:].strip()
            if self.state_store.delete_channel(name):
                logger.info("Deleted response channel %s", name)
                self._sync_channel_states(time.monotonic(), reset_schedule=True)
                return self._state_response()
            logger.info("DELETE CHANNEL was a no-op; %s not found", name)
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
            logger.warning(
                "Rate limited on channel %s; next poll in %.1fs",
                state.name,
                max(base, result.retry_after_sec),
            )
            return

        if result.replied:
            state.interval_sec = base
            state.idle_polls = 0
            state.next_poll_at = now + base
            logger.info(
                "Reply sent in channel %s; reset poll interval to %.1fs",
                state.name,
                base,
            )
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
        logger.debug(
            "Scheduled next poll for %s in %.1fs (idle_polls=%s)",
            state.name,
            state.interval_sec,
            state.idle_polls,
        )

    def _poll_response_channel(self, state: ChannelPollState) -> PollResult:
        result = PollResult()
        try:
            messages = self._fetch_messages(state.channel_id)
            if not messages:
                logger.debug("No messages found in response channel %s", state.name)
                return result

            newest = messages[-1]
            newest_id = str(newest.get("id", ""))
            if state.last_seen_id is None:
                state.last_seen_id = newest_id
                logger.info(
                    "Initialized response channel cursor for %s at message %s",
                    state.name,
                    newest_id,
                )
                return result

            if _snowflake_value(newest_id) <= _snowflake_value(state.last_seen_id):
                logger.debug("No new messages in response channel %s", state.name)
                return result

            state.last_seen_id = newest_id
            newest_author = newest.get("author", {}) or {}
            if str(newest_author.get("id", "")) == self.self_user_id:
                logger.debug(
                    "Latest message in %s is from self; skipping response",
                    state.name,
                )
                return result

            latest_content = str(newest.get("content", "") or "").strip()
            if not latest_content:
                logger.debug(
                    "Latest message in %s has no content; skipping response",
                    state.name,
                )
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
                logger.debug("No textual messages to include in prompt for %s", state.name)
                return result

            lines.append(f"{self.self_username}:")
            prompt = "\n".join(lines)
            logger.info(
                "Generating reply for channel %s using %s messages (%s chars)",
                state.name,
                len(lines) - 1,
                len(prompt),
            )
            reply = self._llm_chat([{"role": "user", "content": prompt}]).strip()
            if not reply:
                logger.info("LLM returned an empty reply for channel %s", state.name)
                return result

            reply = reply[: self.config.llm_reply_max_chars].strip()
            if not reply:
                logger.info("Reply was empty after truncation for channel %s", state.name)
                return result

            self._post_message(state.channel_id, reply)
            result.replied = True
            logger.info(
                "Posted reply to channel %s (%s chars)",
                state.name,
                len(reply),
            )
            return result
        except urllib.error.HTTPError as exc:
            detail, retry_after = self._http_error_details(exc)
            if exc.code == 429:
                logger.warning("Discord rate limited on channel %s: %s", state.name, detail)
                result.retry_after_sec = retry_after
                return result
            logger.error("Discord channel poll failed for %s: %s", state.name, detail)
            return result
        except Exception:
            logger.exception("Discord channel poll crashed for %s", state.name)
            return result
