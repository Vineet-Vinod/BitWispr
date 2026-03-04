"""
BitWispr client - push-to-talk speech-to-text via local Trillim server.
Press Right Ctrl+Right Alt to toggle recording on/off.

For Wayland: run with sudo or add user to input group:
    sudo usermod -aG input $USER
    (then log out and back in)
"""

import io
import json
import os
import queue
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
import sounddevice as sd
from scipy import signal

# --- CONFIGURATION ---
SERVER_BASE_URL = os.environ.get("BITWISPR_SERVER_URL", "http://127.0.0.1:1111").rstrip("/")
WHISPER_MODEL = "whisper-1"
WHISPER_LANGUAGE = "en"
WHISPER_SAMPLE_RATE = 16000
SERVER_TIMEOUT_SEC = 30
KEYBOARD_SCAN_INTERVAL_SEC = 60.0

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


def post_multipart(
    url: str,
    fields: dict[str, str],
    file_field: str,
    filename: str,
    file_bytes: bytes,
    content_type: str,
) -> dict:
    """Send multipart/form-data request and parse JSON response."""
    boundary = f"----bitwispr-{uuid.uuid4().hex}"
    body = bytearray()

    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="{file_field}"; '
            f'filename="{filename}"\r\n'
        ).encode("utf-8")
    )
    body.extend(f"Content-Type: {content_type}\r\n\r\n".encode("utf-8"))
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        data=bytes(body),
        method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=SERVER_TIMEOUT_SEC) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def check_server() -> None:
    """Fail fast if BitWispr server is not reachable."""
    try:
        payload = get_json(f"{SERVER_BASE_URL}/v1/models")
        loaded = payload.get("data", [])
        model_name = loaded[0]["id"] if loaded else "(no model loaded)"
        print(f"Connected to server: {SERVER_BASE_URL} | model: {model_name}")
    except urllib.error.URLError as e:
        print(f"❌ Cannot reach server at {SERVER_BASE_URL}: {e}")
        print("Start it with: uv run server.py")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Server check failed: {e}")
        sys.exit(1)


def transcribe_with_server(audio_data_16k: np.ndarray) -> str:
    """Send WAV audio to local Trillim endpoint and return text."""
    audio_bytes = audio_to_wav_bytes(audio_data_16k, WHISPER_SAMPLE_RATE)
    payload = post_multipart(
        f"{SERVER_BASE_URL}/v1/audio/transcriptions",
        fields={
            "model": WHISPER_MODEL,
            "language": WHISPER_LANGUAGE,
            "response_format": "json",
        },
        file_field="file",
        filename="recording.wav",
        file_bytes=audio_bytes,
        content_type="audio/wav",
    )
    return payload.get("text", "").strip()


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
                final_text = transcribe_with_server(final_audio_16k)
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
    print("=" * 55)
    print("BitWispr - Real-time Speech to Text")
    print("=" * 55)
    print(f"Display server: {'Wayland' if IS_WAYLAND else 'X11'}")
    print(f"Audio sample rate: {DEVICE_SAMPLE_RATE} Hz")
    check_server()
    print("Hotkey: Right Ctrl+Right Alt (toggle recording on/off)")
    print("=" * 55 + "\n")

    if IS_WAYLAND:
        print("Using evdev for Wayland keyboard input...")
        print("(If this fails, run with sudo or add yourself to input group)\n")
        run_with_evdev()
    else:
        print("Using pynput for X11 keyboard input...\n")
        run_with_pynput()


if __name__ == "__main__":
    main()
