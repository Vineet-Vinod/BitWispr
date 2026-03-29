from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

from trillim import LLM, Runtime, STT, TTS

from bitwispr.audio import MicrophoneRecorder, SpeechPlayer, audio_to_wav_bytes
from bitwispr.config import AppConfig, StateStore
from bitwispr.discord import DiscordWorker
from bitwispr.linux_platform import (
    is_wayland_session,
    read_selected_text,
    run_hotkey_loop,
    type_text,
)


class BitWisprApp:
    def __init__(self, config: AppConfig, runtime: Runtime, state_store: StateStore):
        self.config = config
        self.runtime = runtime
        self.state_store = state_store
        self.wayland = is_wayland_session()
        self.recorder = MicrophoneRecorder(sample_rate=config.input_sample_rate)
        self.player = SpeechPlayer(self.runtime.tts.synthesize_wav)
        self.discord = DiscordWorker(
            config,
            state_store,
            llm_chat=self.runtime.llm.chat,
            list_voices=self.runtime.tts.list_voices,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bitwispr-transcribe",
        )
        self._transcription_future: concurrent.futures.Future[None] | None = None
        self._shutdown = threading.Event()

    def run(self) -> None:
        self._print_startup_banner()
        self.recorder.start()
        self.discord.start()

        try:
            run_hotkey_loop(
                on_dictation=self.toggle_dictation,
                on_reader=self.read_current_selection,
                stop_event=self._shutdown,
                wayland=self.wayland,
                keyboard_scan_interval_sec=self.config.keyboard_scan_interval_sec,
            )
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        if self._shutdown.is_set():
            return
        self._shutdown.set()
        self.discord.stop()
        self.recorder.close()
        self.player.close()

        future = self._transcription_future
        if future is not None and not future.done():
            try:
                future.result(timeout=30)
            except Exception as exc:
                print(f"Stopping with an unfinished transcription: {exc}")

        self._executor.shutdown(wait=False, cancel_futures=True)

    def toggle_dictation(self) -> None:
        if self.recorder.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def read_current_selection(self) -> None:
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

        state = self.state_store.snapshot()
        print(f"Reading {len(normalized)} characters...")
        self.player.play(
            normalized,
            voice=state.voice,
            speed=state.speed,
        )

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
            wav_bytes = audio_to_wav_bytes(audio, self.recorder.sample_rate)
            text = self.runtime.stt.transcribe_bytes(
                wav_bytes,
                language=self.config.whisper_language,
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
        state = self.state_store.snapshot()
        print("=" * 60)
        print("BitWispr")
        print("Local dictation + reader + Discord responder")
        print("=" * 60)
        print(f"Display server: {'Wayland' if self.wayland else 'X11'}")
        print(f"Input sample rate: {self.recorder.sample_rate} Hz")
        print(f"Model ID: {self.config.model_id}")
        print(f"Adapter ID: {self.config.adapter_id}")
        print(f"TTS voice: {state.voice}")
        print(f"TTS speed: {state.speed}")
        if self.discord.enabled:
            print(f"Discord control channel: {self.config.control_channel_id}")
        else:
            print("Discord control: disabled (.env not configured)")
        print("Hotkey: Right Ctrl + Right Alt toggles dictation")
        print("Hotkey: Right Ctrl + Right Shift reads the current selection")
        print()


def main() -> int:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        config = AppConfig.from_env(env_file=env_file)
        state_store = StateStore(
            config.state_path,
            default_voice=config.default_voice,
            default_speed=config.default_speed,
        )
        state = state_store.snapshot()
        with Runtime(
            LLM(config.model_id, lora_dir=config.adapter_id),
            STT(),
            TTS(default_voice=state.voice, speed=state.speed),
        ) as runtime:
            app = BitWisprApp(config, runtime, state_store)
            app.run()
    except KeyboardInterrupt:
        print("\nExiting...")
        return 0
    except Exception as exc:
        print(f"BitWispr failed to start: {exc}")
        return 1
    return 0
