"""
BitWispr - Speech to Text using Whisper
Press Shift+Tab to toggle recording on/off.

For Wayland: Run with sudo or add user to input group:
    sudo usermod -aG input $USER
    (then log out and back in)
"""

import os
import sys
import threading
import subprocess
import queue
import numpy as np
import sounddevice as sd
from scipy import signal
from faster_whisper import WhisperModel

# --- CONFIGURATION ---
MODEL_SIZE = "tiny.en"  # Options: tiny.en, base.en, small.en, medium.en
WHISPER_SAMPLE_RATE = 16000  # Whisper expects 16kHz

# Auto-detect device sample rate
try:
    DEVICE_SAMPLE_RATE = int(sd.query_devices(kind='input')['default_samplerate'])
except Exception:
    DEVICE_SAMPLE_RATE = 44100  # Fallback

# Detect display server (check multiple env vars for robustness with sudo)
IS_WAYLAND = (
    os.environ.get("XDG_SESSION_TYPE") == "wayland"
    or os.environ.get("WAYLAND_DISPLAY") is not None
)

print("=" * 55)
print("BitWispr - Real-time Speech to Text")
print("=" * 55)
print(f"Display server: {'Wayland' if IS_WAYLAND else 'X11'}")
print(f"Audio sample rate: {DEVICE_SAMPLE_RATE} Hz")
print(f"Loading Whisper model '{MODEL_SIZE}'... please wait.")

model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")
print("Model loaded successfully!")
print("Hotkey: Shift+Tab (toggle recording on/off)")
print("=" * 55 + "\n")

# Global state
recording = False
transcribing = False  # Track if transcription is in progress
audio_queue = queue.Queue()
typed_text = ""
transcribe_event = threading.Event()  # Signal to transcribe
worker_thread = None


def resample_audio(audio_data: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample audio from original sample rate to target sample rate."""
    if orig_sr == target_sr:
        return audio_data
    
    # Calculate the number of samples in the resampled audio
    num_samples = int(len(audio_data) * target_sr / orig_sr)
    resampled = signal.resample(audio_data, num_samples)
    return resampled.astype(np.float32)

def type_text(text: str):
    """Type text at cursor position. Works on both X11 and Wayland."""
    if not text:
        return

    if IS_WAYLAND:
        try:
            result = subprocess.run(
                ["ydotool", "type", "--", text],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            pass

        try:
            result = subprocess.run(
                ["wtype", "--", text],
                capture_output=True,
                timeout=5,
            )
            if result.returncode == 0:
                return
        except FileNotFoundError:
            pass

        print("⚠️  Could not type text. Install ydotool:")
        print("   sudo apt install ydotool")
        print(f"   Text was: {text}")

    else:
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
        # Wait for signal to transcribe
        transcribe_event.wait()
        transcribe_event.clear()
        
        transcribing = True

        # Collect all audio from queue
        audio_buffer = []
        try:
            while True:
                chunk = audio_queue.get_nowait()
                audio_buffer.extend(chunk.tolist())
        except queue.Empty:
            pass

        if len(audio_buffer) > DEVICE_SAMPLE_RATE * 0.3:  # At least 0.3s of audio
            print("🔄 Transcribing...")
            
            final_audio = np.array(audio_buffer, dtype=np.float32)
            final_audio_16k = resample_audio(final_audio, DEVICE_SAMPLE_RATE, WHISPER_SAMPLE_RATE)

            try:
                segments, _ = model.transcribe(
                    final_audio_16k,
                    beam_size=5,
                    vad_filter=True,
                )

                final_text = ""
                for segment in segments:
                    final_text += segment.text
                final_text = final_text.strip()

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
    # Ignore overflow warnings - they're common and don't affect quality much
    if recording:
        audio_queue.put(indata[:, 0].copy())

def start_recording():
    """Start recording audio."""
    global recording, typed_text, worker_thread

    # Don't start if transcription is in progress
    if transcribing:
        print("⏳ Please wait, transcription in progress...")
        return False

    # Start worker thread once (on first recording)
    if worker_thread is None:
        worker_thread = threading.Thread(target=transcription_worker, daemon=True)
        worker_thread.start()

    # Clear the audio queue
    while not audio_queue.empty():
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            break
    
    typed_text = ""
    recording = True

    print("\n🎙️  Recording started... (Press Shift+Tab to stop)")
    return True

def stop_recording():
    """Stop recording and trigger transcription."""
    global recording
    recording = False
    print("⏹️  Recording stopped.\n")
    
    # Signal the worker to transcribe
    transcribe_event.set()

def run_with_evdev():
    """Use evdev for keyboard listening (works on Wayland with proper permissions)."""
    try:
        import evdev
        from evdev import ecodes
    except ImportError:
        print("❌ evdev not installed. Run: uv add evdev")
        sys.exit(1)

    # Find actual keyboard devices (not touchpad, mouse, etc.)
    devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
    keyboards = []
    for d in devices:
        caps = d.capabilities()
        # Must have EV_KEY capability
        if ecodes.EV_KEY not in caps:
            continue
        # Check if it has actual keyboard keys (letters, not just buttons)
        key_caps = caps.get(ecodes.EV_KEY, [])
        # Look for letter keys (KEY_A = 30, KEY_Z = 44) or KEY_TAB (15)
        has_keyboard_keys = any(k in key_caps for k in [ecodes.KEY_A, ecodes.KEY_TAB, ecodes.KEY_LEFTSHIFT])
        if has_keyboard_keys:
            keyboards.append(d)

    if not keyboards:
        print("❌ No keyboard found. Make sure you have permission to access /dev/input/")
        print("   Run: sudo usermod -aG input $USER")
        print("   Then log out and back in.")
        sys.exit(1)

    print(f"Found {len(keyboards)} keyboard device(s):")
    for kbd in keyboards:
        print(f"  - {kbd.name}")

    # Track key states
    shift_pressed = False

    # Open audio stream with larger blocksize to prevent overflow
    stream = sd.InputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(DEVICE_SAMPLE_RATE * 0.2),  # 200ms blocks
    )
    stream.start()

    print("Listening for Shift+Tab... (press Ctrl+C to quit)\n")

    try:
        # Listen to all keyboards
        from selectors import DefaultSelector, EVENT_READ

        selector = DefaultSelector()
        for kbd in keyboards:
            selector.register(kbd, EVENT_READ)

        while True:
            for key, _ in selector.select():
                device = key.fileobj
                for event in device.read():
                    if event.type == ecodes.EV_KEY:
                        # Track Shift
                        if event.code in (ecodes.KEY_LEFTSHIFT, ecodes.KEY_RIGHTSHIFT):
                            shift_pressed = event.value != 0  # 1=press, 0=release, 2=hold

                        # Track Tab and trigger on Shift+Tab
                        if event.code == ecodes.KEY_TAB:
                            if event.value == 1:  # Key press (not hold/release)
                                if shift_pressed:
                                    # Toggle recording
                                    if recording:
                                        print("\n" + "=" * 40)
                                        print("⏹️  STOPPED RECORDING")
                                        print("=" * 40)
                                        stop_recording()
                                    else:
                                        if start_recording():
                                            print("=" * 40)

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop()
        stream.close()

def run_with_pynput():
    """Use pynput for keyboard listening (works on X11)."""
    from pynput import keyboard

    current_keys = set()
    combo_pressed = False

    # Open audio stream with larger blocksize to prevent overflow
    stream = sd.InputStream(
        samplerate=DEVICE_SAMPLE_RATE,
        channels=1,
        callback=audio_callback,
        blocksize=int(DEVICE_SAMPLE_RATE * 0.2),  # 200ms blocks
    )
    stream.start()

    def on_press(key):
        nonlocal combo_pressed
        current_keys.add(key)

        if keyboard.Key.shift in current_keys and keyboard.Key.tab in current_keys:
            if not combo_pressed:
                combo_pressed = True
                if recording:
                    print("\n" + "=" * 40)
                    print("⏹️  STOPPED RECORDING")
                    print("=" * 40)
                    stop_recording()
                else:
                    if start_recording():
                        print("=" * 40)

    def on_release(key):
        nonlocal combo_pressed
        try:
            current_keys.remove(key)
        except KeyError:
            pass
        if key == keyboard.Key.tab:
            combo_pressed = False

    print("Listening for Shift+Tab... (press Ctrl+C to quit)\n")

    try:
        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        stream.stop()
        stream.close()

def main():
    if IS_WAYLAND:
        print("Using evdev for Wayland keyboard input...")
        print("(If this fails, run with sudo or add yourself to input group)\n")
        run_with_evdev()
    else:
        print("Using pynput for X11 keyboard input...\n")
        run_with_pynput()


if __name__ == "__main__":
    main()
