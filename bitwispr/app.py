from __future__ import annotations

import concurrent.futures
import threading
import time
from pathlib import Path

from trillim import Runtime, TTS, Whisper

from bitwispr.audio import MicrophoneRecorder, SpeechPlayer, detect_input_sample_rate
from bitwispr.config import AppConfig
from bitwispr.platform import is_wayland_session, read_selected_text, type_text


class BitWisprApp:
    def __init__(self, config: AppConfig):
        self.config = config
        self.wayland = is_wayland_session()
        self.recorder = MicrophoneRecorder(
            sample_rate=config.input_sample_rate or detect_input_sample_rate()
        )
        self.runtime = Runtime(
            Whisper(
                model_size=config.whisper_model,
                compute_type=config.whisper_compute_type,
                cpu_threads=config.whisper_cpu_threads,
            ),
            TTS(
                voices_dir=config.voices_dir,
                default_voice=config.tts_voice,
                speed=config.tts_speed,
            ),
        )
        self.player: SpeechPlayer | None = None
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bitwispr-transcribe",
        )
        self._transcription_future: concurrent.futures.Future[None] | None = None
        self._shutdown = threading.Event()

    def run(self) -> None:
        self._print_startup_banner()
        self.runtime.start()
        self.recorder.start()
        self.player = SpeechPlayer(self.runtime, timeout_sec=self.config.tts_timeout_sec)

        voices = ", ".join(voice["voice_id"] for voice in self.runtime.tts.list_voices())
        print(f"Display server: {'Wayland' if self.wayland else 'X11'}")
        print(f"Input sample rate: {self.recorder.sample_rate} Hz")
        print(f"Whisper model: {self.config.whisper_model}")
        print(f"TTS voice: {self.config.tts_voice}")
        print(f"Available voices: {voices}")
        print("Hotkey: Right Ctrl + Right Alt toggles dictation")
        print("Hotkey: Right Ctrl + Right Shift reads the current selection")
        print()

        try:
            if self.wayland:
                self._run_with_evdev()
            else:
                self._run_with_pynput()
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()

        self.recorder.close()
        if self.player is not None:
            self.player.close()

        future = self._transcription_future
        if future is not None and not future.done():
            print("Waiting for the active transcription to finish...")
            try:
                future.result(timeout=self.config.whisper_timeout_sec)
            except Exception as exc:
                print(f"Stopping with an unfinished transcription: {exc}")

        self._executor.shutdown(wait=False, cancel_futures=True)
        self.runtime.stop()

    def toggle_dictation(self) -> None:
        if self.recorder.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def read_current_selection(self) -> None:
        if self.player is None:
            return

        text = read_selected_text(self.config.selection_mode, wayland=self.wayland)
        if not text:
            print(
                "No highlighted text found. On Wayland install `wl-clipboard`; "
                "on X11 install `xclip` or `xsel`."
            )
            return

        normalized = " ".join(text.split())
        if not normalized:
            print("The current selection is empty.")
            return

        voice = self.read_voice_for(normalized)
        print(f"Reading {len(normalized)} characters...")
        self.player.play(
            normalized,
            voice=voice,
            speed=self.config.tts_speed,
        )

    def read_voice_for(self, text: str) -> str | None:
        """Override this if you want custom per-selection voice routing."""
        return self.config.tts_voice

    def _start_recording(self) -> None:
        if self._transcription_in_progress():
            print("Please wait for the current transcription to finish.")
            return
        if not self.recorder.begin_recording():
            return
        print("Recording started. Press Right Ctrl + Right Alt again to stop.")

    def _stop_recording(self) -> None:
        audio = self.recorder.finish_recording()
        print("Recording stopped.")

        min_samples = int(self.recorder.sample_rate * self.config.min_recording_sec)
        if audio.size < min_samples:
            print("Recording was too short.")
            return

        print("Transcribing...")
        self._transcription_future = self._executor.submit(
            self._transcribe_and_type,
            audio,
        )

    def _transcription_in_progress(self) -> bool:
        future = self._transcription_future
        return future is not None and not future.done()

    def _transcribe_and_type(self, audio) -> None:
        try:
            text = self.runtime.whisper.transcribe_array(
                audio,
                sample_rate=self.recorder.sample_rate,
                language=self.config.whisper_language,
                timeout=self.config.whisper_timeout_sec,
            ).strip()
        except Exception as exc:
            print(f"Transcription failed: {exc}")
            return

        if not text:
            print("No speech detected.")
            return

        print(f"Transcribed: {text}")
        type_text(text + self.config.dictation_suffix, wayland=self.wayland)

    def _print_startup_banner(self) -> None:
        print("=" * 60)
        print("BitWispr")
        print("Local dictation + highlight-to-read")
        print("=" * 60)

    def _run_with_evdev(self) -> None:
        try:
            import evdev
            from evdev import ecodes
        except ImportError:
            raise RuntimeError("`evdev` is required for Wayland hotkeys")

        from selectors import DefaultSelector, EVENT_READ

        print("Using evdev for hotkeys.")

        ctrl_keys = {ecodes.KEY_RIGHTCTRL}
        alt_keys = {ecodes.KEY_RIGHTALT}
        shift_keys = {ecodes.KEY_RIGHTSHIFT}
        if hasattr(ecodes, "KEY_ALTGR"):
            alt_keys.add(ecodes.KEY_ALTGR)
        if hasattr(ecodes, "KEY_ISO_LEVEL3_SHIFT"):
            alt_keys.add(ecodes.KEY_ISO_LEVEL3_SHIFT)

        selector = DefaultSelector()
        devices: dict[str, evdev.InputDevice] = {}
        state: dict[str, dict[str, bool]] = {}

        def is_keyboard_device(device: evdev.InputDevice) -> bool:
            try:
                capabilities = device.capabilities()
            except OSError:
                return False
            if ecodes.EV_KEY not in capabilities:
                return False
            key_caps = set(capabilities.get(ecodes.EV_KEY, []))
            required = {
                ecodes.KEY_A,
                ecodes.KEY_SPACE,
                ecodes.KEY_RIGHTCTRL,
                ecodes.KEY_RIGHTALT,
                ecodes.KEY_RIGHTSHIFT,
            }
            return bool(key_caps.intersection(required))

        def add_new_keyboards() -> None:
            for path in evdev.list_devices():
                if path in devices:
                    continue
                try:
                    device = evdev.InputDevice(path)
                except OSError:
                    continue
                if not is_keyboard_device(device):
                    continue
                try:
                    selector.register(device, EVENT_READ)
                except Exception:
                    device.close()
                    continue
                devices[path] = device
                state[path] = {
                    "ctrl": False,
                    "alt": False,
                    "shift": False,
                    "dictation_latched": False,
                    "read_latched": False,
                }
                print(f"  + Keyboard: {device.name} ({path})")

        def remove_keyboard(path: str) -> None:
            device = devices.pop(path, None)
            state.pop(path, None)
            if device is None:
                return
            try:
                selector.unregister(device)
            except Exception:
                pass
            try:
                device.close()
            except Exception:
                pass

        def update_combos(device_state: dict[str, bool], is_key_down: bool) -> None:
            dictation_down = device_state["ctrl"] and device_state["alt"]
            read_down = device_state["ctrl"] and device_state["shift"]

            if dictation_down and is_key_down and not device_state["dictation_latched"]:
                self.toggle_dictation()
                device_state["dictation_latched"] = True
            elif not dictation_down:
                device_state["dictation_latched"] = False

            if read_down and is_key_down and not device_state["read_latched"]:
                self.read_current_selection()
                device_state["read_latched"] = True
            elif not read_down:
                device_state["read_latched"] = False

        add_new_keyboards()
        if not devices:
            print("No keyboard devices detected yet. Waiting for evdev devices...")

        last_scan = time.monotonic()
        while not self._shutdown.is_set():
            now = time.monotonic()
            if now - last_scan >= self.config.keyboard_scan_interval_sec:
                add_new_keyboards()
                last_scan = now

            for key, _ in selector.select(timeout=1.0):
                device = key.fileobj
                path = getattr(device, "path", "")
                if path not in state:
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
                    device_state = state[path]
                    if event.code in ctrl_keys:
                        device_state["ctrl"] = is_key_down
                    elif event.code in alt_keys:
                        device_state["alt"] = is_key_down
                    elif event.code in shift_keys:
                        device_state["shift"] = is_key_down
                    else:
                        continue

                    update_combos(device_state, event.value == 1)

    def _run_with_pynput(self) -> None:
        try:
            from pynput import keyboard
        except ImportError:
            raise RuntimeError("`pynput` is required for X11 hotkeys")

        print("Using pynput for hotkeys.")

        state = {
            "ctrl": False,
            "alt": False,
            "shift": False,
            "dictation_latched": False,
            "read_latched": False,
        }

        ctrl_keys = {keyboard.Key.ctrl_r}
        alt_keys = {keyboard.Key.alt_r}
        alt_gr = getattr(keyboard.Key, "alt_gr", None)
        if alt_gr is not None:
            alt_keys.add(alt_gr)
        shift_keys = {keyboard.Key.shift_r}

        def update_state(key, is_pressed: bool) -> None:
            if key in ctrl_keys:
                state["ctrl"] = is_pressed
            elif key in alt_keys:
                state["alt"] = is_pressed
            elif key in shift_keys:
                state["shift"] = is_pressed
            else:
                return

            dictation_down = state["ctrl"] and state["alt"]
            read_down = state["ctrl"] and state["shift"]

            if dictation_down and is_pressed and not state["dictation_latched"]:
                self.toggle_dictation()
                state["dictation_latched"] = True
            elif not dictation_down:
                state["dictation_latched"] = False

            if read_down and is_pressed and not state["read_latched"]:
                self.read_current_selection()
                state["read_latched"] = True
            elif not read_down:
                state["read_latched"] = False

        def on_press(key) -> None:
            update_state(key, True)

        def on_release(key) -> None:
            update_state(key, False)

        with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


def main() -> int:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        config = AppConfig.from_env(env_file=env_file)
        app = BitWisprApp(config)
        app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as exc:
        print(f"BitWispr failed to start: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
