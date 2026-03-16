from __future__ import annotations

import queue
import threading

import numpy as np
import sounddevice as sd


def detect_input_sample_rate() -> int:
    try:
        return int(sd.query_devices(kind="input")["default_samplerate"])
    except Exception:
        return 44100


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
            print(f"Audio input status: {status}")

        with self._lock:
            if not self._recording:
                return
            self._chunks.append(indata[:, 0].copy())


class SpeechPlayer:
    def __init__(self, runtime, timeout_sec: float):
        self._runtime = runtime
        self._timeout_sec = timeout_sec
        self._queue: queue.Queue[tuple[int, object] | None] = queue.Queue()
        self._lock = threading.Lock()
        self._generation = 0
        self._current_session = None
        self._thread = threading.Thread(
            target=self._worker,
            daemon=True,
            name="bitwispr-speaker",
        )
        self._thread.start()

    def play(self, text: str, *, voice: str | None, speed: float | None) -> bool:
        cleaned = text.strip()
        if not cleaned:
            return False

        session = self._runtime.tts.speak(
            cleaned,
            voice=voice,
            speed=speed,
            timeout=self._timeout_sec,
            interrupt=True,
        )
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._current_session = session
        self._queue.put((generation, session))
        return True

    def stop(self) -> None:
        with self._lock:
            self._generation += 1
            session = self._current_session
        if session is None:
            return
        try:
            session.cancel()
        except Exception:
            pass

    def close(self) -> None:
        self.stop()
        self._queue.put(None)
        self._thread.join(timeout=5)

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            if item is None:
                return
            generation, session = item
            with self._lock:
                if generation != self._generation:
                    continue
            self._play_session(generation, session)

    def _play_session(self, generation: int, session) -> None:
        try:
            stream = sd.RawOutputStream(
                samplerate=self._runtime.tts.sample_rate,
                channels=1,
                dtype="int16",
            )
            stream.start()
        except Exception as exc:
            print(f"TTS playback could not start: {exc}")
            return

        try:
            for chunk in session:
                with self._lock:
                    if generation != self._generation:
                        break
                if chunk:
                    stream.write(chunk)
        except Exception as exc:
            with self._lock:
                current_generation = self._generation
            if generation == current_generation:
                print(f"TTS playback failed: {exc}")
        finally:
            stream.stop()
            stream.close()

