"""
BitWispr Trillim server.
Runs LLM + Whisper on localhost:1111 with BitNet-TRNQ + GenZ adapter.
"""

import json
import os
import signal
import threading
import time
import urllib.error
import urllib.request
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

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


class DiscordAutoResponder:
    """Poll Discord channels and auto-reply using local LLM endpoint."""

    def __init__(
        self,
        auth_token: str,
        channel_ids: list[str],
        llm_base_url: str,
        poll_interval_sec: float = 900.0,
    ):
        self.auth_token = auth_token
        self.channel_ids = channel_ids
        self.llm_base_url = llm_base_url.rstrip("/")
        self.poll_interval_sec = poll_interval_sec
        self.self_user_id: str | None = None
        self.last_seen_ids: dict[str, str] = {}
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
            f"(channels: {', '.join(self.channel_ids)}, poll={self.poll_interval_sec:.1f}s)"
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
    def _http_error_details(error: urllib.error.HTTPError) -> str:
        try:
            raw = error.read().decode("utf-8")
        except Exception:
            return str(error)

        try:
            payload = json.loads(raw)
            message = payload.get("message")
            code = payload.get("code")
            if message is not None and code is not None:
                return f"{error} | code={code} message={message}"
            if message is not None:
                return f"{error} | message={message}"
            return f"{error} | body={raw}"
        except Exception:
            return f"{error} | body={raw}"

    def _llm_reply(self, message_text: str, channel_id: str) -> str:
        payload = {
            "model": "BitNet-TRNQ",
            "messages": [
                {
                    "role": "user",
                    "content": (message_text),
                },
            ],
        }

        req = urllib.request.Request(
            f"{self.llm_base_url}/v1/chat/completions",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload).encode("utf-8"),
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            obj = json.loads(resp.read().decode("utf-8"))

        text = (
            obj.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
            .strip()
        )
        return text[:1800]

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
        while not self.stop_event.is_set():
            if not self.self_user_id:
                self._bootstrap_self_user()
                self.stop_event.wait(5.0) # Wait 5 seconds
                continue

            for channel_id in self.channel_ids:
                if self.stop_event.is_set():
                    break
                self._process_channel(channel_id)

            self.stop_event.wait(self.poll_interval_sec)

    def _process_channel(self, channel_id: str) -> None:
        try:
            messages = self._discord_request(
                "GET", f"/channels/{channel_id}/messages?limit=5"
            )
            if not isinstance(messages, list) or not messages:
                return

            # Process oldest -> newest.
            messages.sort(key=lambda item: int(item.get("id", "0")))

            current_last_seen = self.last_seen_ids.get(channel_id)
            if current_last_seen is None:
                self.last_seen_ids[channel_id] = messages[-1]["id"]
                return

            for msg in messages:
                msg_id = str(msg.get("id", "0"))
                if int(msg_id) <= int(current_last_seen):
                    continue

                self.last_seen_ids[channel_id] = msg_id
                author = msg.get("author", {}) or {}
                author_id = str(author.get("id", ""))
                if author_id == self.self_user_id or author.get("bot"):
                    continue

                content = (msg.get("content") or "").strip()
                if not content:
                    continue

                try:
                    reply = self._llm_reply(content, channel_id)
                    if not reply:
                        continue
                    self._discord_request(
                        "POST",
                        f"/channels/{channel_id}/messages",
                        payload={"content": reply},
                    )
                except urllib.error.HTTPError as e:
                    detail = self._http_error_details(e)
                    print(f"Discord reply failed for channel {channel_id}: {detail}")
                except Exception as e:
                    print(f"LLM/Discord handling error for channel {channel_id}: {e}")

        except urllib.error.HTTPError as e:
            if e.code == 429:
                retry_after = DISCORD_POLL_INTERVAL_SEC
                detail = self._http_error_details(e)
                print(f"Discord rate limited on channel {channel_id}: {detail}")
                print(f"Sleeping {retry_after:.2f}s")
                self.stop_event.wait(retry_after)
            else:
                detail = self._http_error_details(e)
                print(f"Discord polling HTTP error on channel {channel_id}: {detail}")
        except Exception as e:
            print(f"Discord polling error on channel {channel_id}: {e}")


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
        poll_interval_sec=DISCORD_POLL_INTERVAL_SEC,
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
