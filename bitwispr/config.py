from __future__ import annotations

import json
import logging
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


def load_env_file(path: Path) -> None:
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


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _default_state_path() -> Path:
    xdg_state_home = os.environ.get("XDG_STATE_HOME", "").strip()
    base = Path(xdg_state_home) if xdg_state_home else Path.home() / ".local" / "state"
    return base / "bitwispr" / "state.json"


@dataclass(slots=True)
class AppConfig:
    model_id: str
    adapter_id: str
    whisper_language: str | None
    min_recording_sec: float
    dictation_suffix: str
    default_voice: str
    default_speed: float
    selection_mode: str
    keyboard_scan_interval_sec: float
    input_sample_rate: int | None
    discord_auth_token: str
    control_channel_id: str
    state_path: Path
    log_level: str
    log_file: Path | None
    control_poll_interval_sec: float
    responder_poll_interval_sec: float
    responder_backoff_factor: float
    responder_max_poll_interval_sec: float
    responder_idle_polls_before_backoff: int
    llm_reply_max_chars: int

    @classmethod
    def from_env(cls, env_file: Path | None = None) -> AppConfig:
        if env_file is not None:
            load_env_file(env_file)

        selection_mode = _env_str("BITWISPR_SELECTION_MODE", "auto").lower()
        if selection_mode not in {"auto", "primary", "clipboard"}:
            raise ValueError(
                "BITWISPR_SELECTION_MODE must be one of: auto, primary, clipboard"
            )

        whisper_language_raw = os.environ.get("BITWISPR_WHISPER_LANGUAGE", "en")
        whisper_language = whisper_language_raw.strip() or None

        input_sample_rate_raw = os.environ.get("BITWISPR_AUDIO_SAMPLE_RATE", "").strip()
        input_sample_rate = int(input_sample_rate_raw) if input_sample_rate_raw else None

        return cls(
            model_id=_env_str("BITWISPR_MODEL_ID", "Trillim/BitNet-TRNQ"),
            adapter_id=_env_str(
                "BITWISPR_ADAPTER_ID",
                "Trillim/BitNet-GenZ-LoRA-TRNQ",
            ),
            whisper_language=whisper_language,
            min_recording_sec=max(0.1, _env_float("BITWISPR_MIN_RECORDING_SEC", 0.3)),
            dictation_suffix=os.environ.get("BITWISPR_DICTATION_SUFFIX", " "),
            default_voice=_env_str("BITWISPR_TTS_VOICE", "alba"),
            default_speed=max(0.25, min(4.0, _env_float("BITWISPR_TTS_SPEED", 1.0))),
            selection_mode=selection_mode,
            keyboard_scan_interval_sec=max(
                5.0,
                _env_float("BITWISPR_KEYBOARD_SCAN_INTERVAL_SEC", 60.0),
            ),
            input_sample_rate=input_sample_rate,
            discord_auth_token=os.environ.get("DISCORD_AUTH_TOKEN", "").strip(),
            control_channel_id=os.environ.get("CONTROL_CHANNEL", "").strip(),
            state_path=Path(
                os.environ.get("BITWISPR_STATE_PATH", str(_default_state_path()))
            ).expanduser(),
            log_level=_env_str("BITWISPR_LOG_LEVEL", "INFO"),
            log_file=(
                Path(os.environ["BITWISPR_LOG_FILE"]).expanduser()
                if os.environ.get("BITWISPR_LOG_FILE", "").strip()
                else None
            ),
            control_poll_interval_sec=max(
                5.0,
                _env_float("BITWISPR_CONTROL_POLL_INTERVAL_SEC", 30.0),
            ),
            responder_poll_interval_sec=max(
                1.0,
                _env_float("BITWISPR_RESPONDER_POLL_INTERVAL_SEC", 10.0),
            ),
            responder_backoff_factor=max(
                1.1,
                _env_float("BITWISPR_RESPONDER_BACKOFF_FACTOR", 2.0),
            ),
            responder_max_poll_interval_sec=max(
                60.0,
                _env_float("BITWISPR_RESPONDER_MAX_POLL_INTERVAL_SEC", 900.0),
            ),
            responder_idle_polls_before_backoff=max(
                0,
                _env_int("BITWISPR_RESPONDER_IDLE_POLLS", 6),
            ),
            llm_reply_max_chars=max(
                200,
                _env_int("BITWISPR_LLM_REPLY_MAX_CHARS", 1800),
            ),
        )


@dataclass(slots=True)
class MutableState:
    voice: str
    speed: float
    responder_active: bool = True
    channels: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "voice": self.voice,
            "speed": self.speed,
            "active": self.responder_active,
            "channels": dict(sorted(self.channels.items())),
        }


class StateStore:
    def __init__(self, path: Path, default_voice: str, default_speed: float):
        self.path = path
        self._lock = threading.Lock()
        self._state = self._load_state(default_voice, default_speed)

    @staticmethod
    def clamp_speed(value: float) -> float:
        return max(0.25, min(4.0, value))

    def _load_state(self, default_voice: str, default_speed: float) -> MutableState:
        default_state = MutableState(
            voice=default_voice,
            speed=self.clamp_speed(default_speed),
            responder_active=True,
            channels={},
        )
        if not self.path.exists():
            logger.info("State file %s does not exist; using defaults", self.path)
            return default_state

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(
                "Failed to load state file %s; using defaults: %s",
                self.path,
                exc,
            )
            return default_state

        if not isinstance(payload, dict):
            logger.warning(
                "State file %s does not contain a JSON object; using defaults",
                self.path,
            )
            return default_state

        channels_raw = payload.get("channels", {})
        channels: dict[str, str] = {}
        if isinstance(channels_raw, dict):
            for name, channel_id in channels_raw.items():
                name_text = str(name).strip()
                channel_text = str(channel_id).strip()
                if name_text and channel_text:
                    channels[name_text] = channel_text

        voice = str(payload.get("voice", default_voice)).strip() or default_voice

        try:
            speed = self.clamp_speed(float(payload.get("speed", default_speed)))
        except (TypeError, ValueError):
            speed = self.clamp_speed(default_speed)

        responder_active = bool(payload.get("active", True))

        return MutableState(
            voice=voice,
            speed=speed,
            responder_active=responder_active,
            channels=channels,
        )

    def _write_locked(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(self._state.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        tmp_path.replace(self.path)
        logger.debug("Persisted state to %s", self.path)

    def snapshot(self) -> MutableState:
        with self._lock:
            return MutableState(
                voice=self._state.voice,
                speed=self._state.speed,
                responder_active=self._state.responder_active,
                channels=dict(self._state.channels),
            )

    def payload(
        self,
        *,
        errors: list[str] | None = None,
        extra: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload = self.snapshot().to_dict()
        if errors:
            payload["errors"] = errors
        if extra:
            payload.update(extra)
        return payload

    def set_voice(self, voice: str) -> bool:
        voice = voice.strip()
        if not voice:
            return False
        with self._lock:
            if self._state.voice == voice:
                return False
            self._state.voice = voice
            self._write_locked()
            return True

    def set_speed(self, speed: float) -> bool:
        clamped = self.clamp_speed(speed)
        with self._lock:
            if self._state.speed == clamped:
                return False
            self._state.speed = clamped
            self._write_locked()
            return True

    def set_responder_active(self, active: bool) -> bool:
        with self._lock:
            if self._state.responder_active == active:
                return False
            self._state.responder_active = active
            self._write_locked()
            return True

    def add_channel(self, channel_id: str, name: str) -> tuple[bool, str | None]:
        channel_id = channel_id.strip()
        name = name.strip()
        if not channel_id or not name:
            return False, "ADD CHANNEL requires both a channel id and a name."

        with self._lock:
            if name in self._state.channels:
                return False, f"channel name already exists: {name}"
            if channel_id in self._state.channels.values():
                return False, f"channel id already exists: {channel_id}"
            self._state.channels[name] = channel_id
            self._write_locked()
            return True, None

    def delete_channel(self, name: str) -> bool:
        name = name.strip()
        if not name:
            return False
        with self._lock:
            if name not in self._state.channels:
                return False
            del self._state.channels[name]
            self._write_locked()
            return True
