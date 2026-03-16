from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines without overriding existing environment values."""
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


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return int(raw.strip())


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return float(raw.strip())


def _env_str(name: str, default: str) -> str:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip()
    return value if value else default


@dataclass(slots=True)
class AppConfig:
    whisper_model: str
    whisper_compute_type: str
    whisper_cpu_threads: int
    whisper_language: str | None
    whisper_timeout_sec: float
    min_recording_sec: float
    dictation_suffix: str
    tts_voice: str
    tts_speed: float
    tts_timeout_sec: float
    voices_dir: Path
    selection_mode: str
    keyboard_scan_interval_sec: float
    input_sample_rate: int | None

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
            whisper_model=_env_str("BITWISPR_WHISPER_MODEL", "base.en"),
            whisper_compute_type=_env_str(
                "BITWISPR_WHISPER_COMPUTE_TYPE",
                "int8",
            ),
            whisper_cpu_threads=_env_int("BITWISPR_WHISPER_CPU_THREADS", 2),
            whisper_language=whisper_language,
            whisper_timeout_sec=max(
                5.0,
                _env_float("BITWISPR_WHISPER_TIMEOUT_SEC", 120.0),
            ),
            min_recording_sec=max(
                0.1,
                _env_float("BITWISPR_MIN_RECORDING_SEC", 0.3),
            ),
            dictation_suffix=os.environ.get("BITWISPR_DICTATION_SUFFIX", " "),
            tts_voice=_env_str("BITWISPR_TTS_VOICE", "alba"),
            tts_speed=_env_float("BITWISPR_TTS_SPEED", 1.0),
            tts_timeout_sec=max(5.0, _env_float("BITWISPR_TTS_TIMEOUT_SEC", 60.0)),
            voices_dir=Path(
                os.environ.get(
                    "BITWISPR_TTS_VOICES_DIR",
                    str(Path.home() / ".trillim" / "voices"),
                )
            ).expanduser(),
            selection_mode=selection_mode,
            keyboard_scan_interval_sec=max(
                5.0,
                _env_float("BITWISPR_KEYBOARD_SCAN_INTERVAL_SEC", 60.0),
            ),
            input_sample_rate=input_sample_rate,
        )

