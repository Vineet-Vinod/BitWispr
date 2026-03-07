"""
BitWispr - push-to-talk speech-to-text with optional Discord auto-responder.
Runs fully in-process with shared Trillim LLM + Whisper components.
Press Right Ctrl+Right Alt to toggle recording on/off.

For Wayland: run with sudo or add user to input group:
    sudo usermod -aG input $USER
    (then log out and back in)
"""

from __future__ import annotations

import io
import os
import queue
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import numpy as np
import sounddevice as sd
from scipy import signal

from bitwispr_runtime import BitWisprRuntime
from discord_responder import DiscordAutoResponder


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
