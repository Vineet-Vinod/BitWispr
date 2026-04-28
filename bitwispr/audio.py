from __future__ import annotations

import io
import logging
import queue
import threading
import wave
from collections import deque

import numpy as np
import sounddevice as sd

try:
    from trillim.components.tts._limits import PCM_CHANNELS, PCM_SAMPLE_RATE
except Exception:
    PCM_CHANNELS = 1
    PCM_SAMPLE_RATE = 24000

logger = logging.getLogger(__name__)

PCM_SAMPLE_WIDTH_BYTES = 2
PCM_FRAME_BYTES = PCM_CHANNELS * PCM_SAMPLE_WIDTH_BYTES
PLAYBACK_FRAME_SEC = 0.02
PAUSE_BUFFER_SEC = 0.35
MAX_BUFFER_SEC = 0.75


def detect_input_sample_rate() -> int:
    try:
        return int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:
        return 44100


def audio_to_wav_bytes(audio_data: np.ndarray, sample_rate: int) -> bytes:
    clipped = np.clip(audio_data, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm16.tobytes())
    return buffer.getvalue()


class MicrophoneRecorder:
    def __init__(self, sample_rate: int | None = None, block_seconds: float = 0.2):
        self.sample_rate = sample_rate or detect_input_sample_rate()
        self.blocksize = max(1, int(self.sample_rate * max(0.05, block_seconds)))
        self._chunks: list[np.ndarray] = []
        self._lock = threading.Lock()
        self._recording = False
        self._stream: sd.InputStream | None = None

    @property
    def recording(self) -> bool:
        with self._lock:
            return self._recording

    def start(self) -> None:
        if self._stream is not None:
            return
        logger.info(
            "Starting microphone input stream at %s Hz (blocksize=%s)",
            self.sample_rate,
            self.blocksize,
        )
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            blocksize=self.blocksize,
        )
        self._stream.start()

    def close(self) -> None:
        stream = self._stream
        self._stream = None
        if stream is None:
            return
        logger.info("Closing microphone input stream")
        stream.stop()
        stream.close()

    def begin_recording(self) -> bool:
        with self._lock:
            if self._recording:
                return False
            self._chunks = []
            self._recording = True
            return True

    def finish_recording(self) -> np.ndarray:
        with self._lock:
            self._recording = False
            chunks = self._chunks
            self._chunks = []

        if not chunks:
            return np.empty(0, dtype=np.float32)
        return np.concatenate(chunks)

    def _audio_callback(self, indata, frames, time_info, status) -> None:
        if status:
            logger.warning("Audio input status: %s", status)

        with self._lock:
            if not self._recording:
                return
            self._chunks.append(indata[:, 0].copy())


def _align_pcm_bytes(byte_count: int) -> int:
    aligned = byte_count - (byte_count % PCM_FRAME_BYTES)
    return max(PCM_FRAME_BYTES, aligned)


class _BufferedSpeechController:
    def __init__(
        self,
        *,
        generation: int,
        session,
        frame_bytes: int,
        pause_buffer_bytes: int,
        max_buffer_bytes: int,
    ):
        self.generation = generation
        self._session = session
        self._frame_bytes = frame_bytes
        self._pause_buffer_bytes = pause_buffer_bytes
        self._max_buffer_bytes = max_buffer_bytes
        self._condition = threading.Condition()
        self._buffer: deque[memoryview] = deque()
        self._buffer_bytes = 0
        self._pause_requested = False
        self._tts_paused = False
        self._tts_pause_pending = False
        self._cancelled = False
        self._generation_done = False
        self._error: Exception | None = None
        self._generator = threading.Thread(
            target=self._run_generator,
            daemon=True,
            name=f"bitwispr-tts-buffer-{generation}",
        )
        self._generator.start()

    def toggle_pause(self) -> str:
        with self._condition:
            self._sync_tts_state_locked()
            if not self._is_active_locked():
                return "noop"
            if self._pause_requested:
                self._pause_requested = False
                if self._tts_paused:
                    try:
                        self._session.resume()
                    except Exception:
                        logger.exception("Failed to resume buffered TTS generation")
                        self._pause_requested = True
                        return "noop"
                    self._tts_paused = False
                self._condition.notify_all()
                return "resumed"

            self._pause_requested = True
            if self._buffer_bytes >= self._pause_buffer_bytes:
                try:
                    self._request_tts_pause_locked()
                except Exception:
                    logger.exception("Failed to pause buffered TTS generation")
                    self._pause_requested = False
                    return "noop"
            self._condition.notify_all()
            return "paused"

    def set_speed(self, speed: float) -> bool:
        with self._condition:
            self._sync_tts_state_locked()
            if not self._is_active_locked():
                return False
            try:
                self._session.set_speed(speed)
            except Exception:
                logger.exception("Failed to update active TTS speed to %s", speed)
                return False
            return True

    def cancel(self) -> None:
        with self._condition:
            if self._cancelled:
                return
            self._cancelled = True
            self._pause_requested = False
            self._condition.notify_all()
        try:
            self._session.close()
        except Exception:
            logger.debug("Ignoring buffered TTS session close failure", exc_info=True)

    def close(self) -> None:
        self.cancel()
        self._generator.join(timeout=5)

    def read_frame(self) -> bytes | None:
        with self._condition:
            while True:
                self._sync_tts_state_locked()
                if self._cancelled:
                    return None
                if not self._pause_requested and self._buffer_bytes > 0:
                    frame = self._pop_frame_locked()
                    self._condition.notify_all()
                    return frame
                if self._generation_done and self._buffer_bytes == 0:
                    if self._error is not None:
                        raise self._error
                    return None
                self._condition.wait(timeout=self._poll_timeout_locked())

    def _run_generator(self) -> None:
        try:
            for chunk in self._session:
                if not chunk:
                    continue
                with self._condition:
                    while True:
                        self._sync_tts_state_locked()
                        if self._cancelled:
                            return
                        if self._pause_requested and self._tts_paused:
                            self._condition.wait(timeout=self._poll_timeout_locked())
                            continue
                        if (
                            not self._pause_requested
                            and self._buffer_bytes >= self._max_buffer_bytes
                        ):
                            self._condition.wait(timeout=self._poll_timeout_locked())
                            continue
                        break

                    self._buffer.append(memoryview(chunk))
                    self._buffer_bytes += len(chunk)
                    if self._buffer_bytes >= self._pause_buffer_bytes:
                        try:
                            self._request_tts_pause_locked()
                        except Exception as exc:
                            self._error = exc
                            self._generation_done = True
                            self._condition.notify_all()
                            return
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                if not self._cancelled:
                    self._error = exc
                self._generation_done = True
                self._condition.notify_all()
        else:
            with self._condition:
                self._generation_done = True
                self._condition.notify_all()

    def _is_active_locked(self) -> bool:
        return not self._cancelled and (
            self._buffer_bytes > 0 or not self._generation_done
        )

    def _request_tts_pause_locked(self) -> None:
        if (
            self._cancelled
            or self._generation_done
            or not self._pause_requested
            or self._tts_paused
            or self._tts_pause_pending
        ):
            return
        self._session.pause()
        self._tts_pause_pending = True

    def _sync_tts_state_locked(self) -> None:
        if not self._tts_pause_pending:
            return
        state = self._session_state_locked()
        if state != "paused":
            if state not in {None, "running"}:
                self._tts_pause_pending = False
                self._tts_paused = False
            return

        self._tts_pause_pending = False
        if self._cancelled:
            self._tts_paused = False
            return
        if not self._pause_requested:
            try:
                self._session.resume()
            except Exception:
                logger.debug(
                    "Ignoring buffered TTS auto-resume failure",
                    exc_info=True,
                )
            self._tts_paused = False
            return
        self._tts_paused = True

    def _session_state_locked(self) -> str | None:
        try:
            return self._session.state
        except Exception:
            logger.debug("Could not read TTS session state", exc_info=True)
            return None

    def _poll_timeout_locked(self) -> float | None:
        if self._tts_pause_pending:
            return 0.05
        return None

    def _pop_frame_locked(self) -> bytes:
        remaining = min(self._frame_bytes, self._buffer_bytes)
        parts: list[bytes] = []
        while remaining > 0:
            head = self._buffer[0]
            if len(head) <= remaining:
                parts.append(head.tobytes())
                remaining -= len(head)
                self._buffer_bytes -= len(head)
                self._buffer.popleft()
                continue
            parts.append(head[:remaining].tobytes())
            self._buffer[0] = head[remaining:]
            self._buffer_bytes -= remaining
            remaining = 0
        return b"".join(parts)


class _TrillimSpeechStream:
    def __init__(self, tts, text: str, *, voice: str | None, speed: float | None):
        self._session = tts.open_session(voice=voice, speed=speed)
        self._iterator = self._session.synthesize(text)

    @property
    def state(self) -> str:
        return self._session.state

    def __iter__(self):
        return self._iterator

    def pause(self) -> None:
        self._session.pause()

    def resume(self) -> None:
        self._session.resume()

    def set_speed(self, speed: float) -> None:
        self._session.set_speed(speed)

    def close(self) -> None:
        try:
            close = getattr(self._iterator, "close", None)
            if close is not None:
                close()
        finally:
            self._session.close()


class SpeechPlayer:
    def __init__(self, tts):
        self._tts = tts
        self._queue: queue.Queue[tuple[int, str, str | None, float | None] | None] = (
            queue.Queue()
        )
        self._lock = threading.Lock()
        self._generation = 0
        self._active_controller: _BufferedSpeechController | None = None
        bytes_per_second = PCM_SAMPLE_RATE * PCM_FRAME_BYTES
        self._frame_bytes = _align_pcm_bytes(int(bytes_per_second * PLAYBACK_FRAME_SEC))
        self._pause_buffer_bytes = _align_pcm_bytes(
            int(bytes_per_second * PAUSE_BUFFER_SEC)
        )
        self._max_buffer_bytes = max(
            self._pause_buffer_bytes,
            _align_pcm_bytes(int(bytes_per_second * MAX_BUFFER_SEC)),
        )
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="bitwispr-speaker",
        )
        self._thread.start()

    def play(self, text: str, *, voice: str | None, speed: float | None) -> bool:
        cleaned = " ".join(text.split())
        if not cleaned:
            return False

        with self._lock:
            self._generation += 1
            generation = self._generation
            active_controller = self._active_controller
        self._queue.put((generation, cleaned, voice, speed))
        if active_controller is not None:
            active_controller.cancel()
        return True

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            active_controller = self._active_controller
        if active_controller is not None:
            active_controller.cancel()

    def toggle_pause(self) -> str:
        with self._lock:
            controller = self._active_controller
            generation = self._generation
        if controller is None or controller.generation != generation:
            return "noop"
        return controller.toggle_pause()

    def set_speed(self, speed: float) -> bool:
        with self._lock:
            controller = self._active_controller
            generation = self._generation
        if controller is None or controller.generation != generation:
            return False
        return controller.set_speed(speed)

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, text, voice, speed = item
            with self._lock:
                if generation != self._generation:
                    continue

            stream: sd.RawOutputStream | None = None
            controller: _BufferedSpeechController | None = None
            try:
                logger.info(
                    "Starting TTS stream playback (%s chars, voice=%s, speed=%s)",
                    len(text),
                    voice,
                    speed,
                )
                controller = _BufferedSpeechController(
                    generation=generation,
                    session=_TrillimSpeechStream(
                        self._tts,
                        text,
                        voice=voice,
                        speed=speed,
                    ),
                    frame_bytes=self._frame_bytes,
                    pause_buffer_bytes=self._pause_buffer_bytes,
                    max_buffer_bytes=self._max_buffer_bytes,
                )
                with self._lock:
                    if generation != self._generation:
                        controller.cancel()
                        continue
                    self._active_controller = controller

                stream = sd.RawOutputStream(
                    samplerate=PCM_SAMPLE_RATE,
                    channels=PCM_CHANNELS,
                    dtype="int16",
                )
                stream.start()

                while True:
                    frame = controller.read_frame()
                    if frame is None:
                        break
                    if frame:
                        stream.write(frame)
                logger.info("Completed TTS buffered playback")
            except Exception as exc:
                with self._lock:
                    if generation == self._generation:
                        logger.exception("TTS streaming playback failed: %s", exc)
            finally:
                with self._lock:
                    if self._active_controller is controller:
                        self._active_controller = None
                if controller is not None:
                    controller.close()
                if stream is not None:
                    stream.stop()
                    stream.close()
