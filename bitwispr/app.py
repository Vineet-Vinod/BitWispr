from __future__ import annotations

import concurrent.futures
import logging
import threading
from pathlib import Path

from trillim import STT, TTS, Runtime

from bitwispr.audio import MicrophoneRecorder, SpeechPlayer, audio_to_wav_bytes
from bitwispr.config import AppConfig, StateStore, load_env_file
from bitwispr.discord import DiscordWorker
from bitwispr.llm_controller import LLMController
from bitwispr.linux_platform import (
    is_wayland_session,
    read_selected_text,
    run_hotkey_loop,
    type_text,
)
from bitwispr.logging_utils import configure_logging

logger = logging.getLogger(__name__)


class BitWisprApp:
    def __init__(
        self,
        config: AppConfig,
        runtime: Runtime,
        state_store: StateStore,
        llm_controller: LLMController,
    ):
        self.config = config
        self.runtime = runtime
        self.state_store = state_store
        self.llm_controller = llm_controller
        self.wayland = is_wayland_session()
        self.recorder = MicrophoneRecorder(sample_rate=config.input_sample_rate)
        self.player = SpeechPlayer(self.runtime.tts)
        self.discord = DiscordWorker(
            config,
            state_store,
            llm_chat=self.llm_controller.chat,
            llm_activate=self.llm_controller.activate,
            llm_deactivate=self.llm_controller.deactivate,
            llm_is_active=self.llm_controller.is_active,
            list_voices=self.runtime.tts.list_voices,
            on_speed_change=self.update_reader_speed,
        )
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="bitwispr-transcribe",
        )
        self._transcription_future: concurrent.futures.Future[None] | None = None
        self._shutdown = threading.Event()

    def run(self) -> None:
        self._print_startup_banner()
        logger.info("Starting microphone recorder")
        self.recorder.start()
        logger.info("Starting Discord worker")
        self.discord.start()

        try:
            logger.info("Entering hotkey loop")
            run_hotkey_loop(
                on_dictation=self.toggle_dictation,
                on_reader=self.read_current_selection,
                on_pause_toggle=self.toggle_reader_pause,
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
        logger.info("Shutting down BitWispr app")
        self.discord.stop()
        self.recorder.close()
        self.player.close()
        self.llm_controller.shutdown()

        future = self._transcription_future
        if future is not None and not future.done():
            try:
                future.result(timeout=30)
            except Exception as exc:
                logger.warning("Stopping with an unfinished transcription: %s", exc)

        self._executor.shutdown(wait=False, cancel_futures=True)

    def toggle_dictation(self) -> None:
        if self.recorder.recording:
            self._stop_recording()
        else:
            self._start_recording()

    def read_current_selection(self) -> None:
        text = read_selected_text(self.config.selection_mode, wayland=self.wayland)
        if not text:
            logger.warning(
                "No highlighted text found. On Wayland install `wl-clipboard`; "
                "on X11 install `xclip` or `xsel`."
            )
            return

        normalized = " ".join(text.split())
        if not normalized:
            logger.warning("The current selection is empty")
            return

        state = self.state_store.snapshot()
        logger.info(
            "Reading %s characters with voice=%s speed=%s",
            len(normalized),
            state.voice,
            state.speed,
        )
        if not self.player.play(
            normalized,
            voice=state.voice,
            speed=state.speed,
        ):
            logger.warning(
                "Reader play request was rejected because the text was empty"
            )

    def toggle_reader_pause(self) -> None:
        result = self.player.toggle_pause()
        if result == "paused":
            logger.info("Paused active reader playback")
        elif result == "resumed":
            logger.info("Resumed active reader playback")

    def update_reader_speed(self, speed: float) -> bool:
        if self.player.set_speed(speed):
            logger.info("Updated active reader playback speed to %s", speed)
            return True
        return False

    def _start_recording(self) -> None:
        if self._transcription_in_progress():
            logger.warning(
                "Ignoring dictation start while transcription is in progress"
            )
            return
        if not self.recorder.begin_recording():
            logger.warning(
                "Ignoring dictation start because recording is already active"
            )
            return
        logger.info("Recording started. Press Right Ctrl + Right Alt again to stop.")

    def _stop_recording(self) -> None:
        audio = self.recorder.finish_recording()
        logger.info("Recording stopped")

        min_samples = int(self.recorder.sample_rate * self.config.min_recording_sec)
        if audio.size < min_samples:
            logger.warning(
                "Recording was too short (%s samples; need at least %s)",
                audio.size,
                min_samples,
            )
            return

        logger.info("Submitting audio for transcription (%s samples)", audio.size)
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
            logger.info("Starting STT transcription (%s bytes WAV)", len(wav_bytes))
            with self.runtime.stt.open_session() as session:
                text = session.transcribe(
                    wav_bytes,
                    language=self.config.whisper_language,
                ).strip()
        except Exception:
            logger.exception("Transcription failed")
            return

        if not text:
            logger.info("No speech detected")
            return

        logger.info("Transcribed text: %r", text)
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
        print("LLM state: inactive on startup; use START from Discord control")
        print(f"TTS voice: {state.voice}")
        print(f"TTS speed: {state.speed}")
        if self.discord.enabled:
            print(f"Discord control channel: {self.config.control_channel_id}")
        else:
            print("Discord control: disabled (.env not configured)")
        print("Hotkey: Right Ctrl + Right Alt toggles dictation")
        print("Hotkey: Right Ctrl + Right Shift reads the current selection")
        print("Hotkey: Right Alt + P pauses or resumes active reader playback")
        print()


def main() -> int:
    env_file = Path(__file__).resolve().parents[1] / ".env"
    try:
        load_env_file(env_file)
        configure_logging()
        logger.info("Loaded environment from %s", env_file)
        config = AppConfig.from_env()
        configure_logging(level_name=config.log_level, log_file=config.log_file)
        logger.info("BitWispr configuration loaded")
        logger.info("Mutable state file: %s", config.state_path)
        state_store = StateStore(
            config.state_path,
            default_voice=config.default_voice,
            default_speed=config.default_speed,
        )
        state = state_store.snapshot()
        logger.info(
            "Starting Trillim runtime with voice=%s speed=%s; LLM starts inactive",
            state.voice,
            state.speed,
        )
        with Runtime(
            STT(),
            TTS(),
        ) as runtime:
            logger.info("Trillim runtime started")
            app = BitWisprApp(config, runtime, state_store, LLMController(config))
            app.run()
    except KeyboardInterrupt:
        logger.info("Exiting on keyboard interrupt")
        return 0
    except Exception:
        logger.exception("BitWispr failed to start")
        return 1
    return 0
