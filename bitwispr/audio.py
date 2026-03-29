from __future__ import annotations

import io
import queue
import threading
import wave

import numpy as np
import sounddevice as sd


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
    def __init__(self, synthesize_wav_fn):
        self._synthesize_wav_fn = synthesize_wav_fn
        self._queue: queue.Queue[tuple[int, str, str | None, float | None] | None] = (
            queue.Queue()
        )
        self._lock = threading.Lock()
        self._generation = 0
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
        self._queue.put((generation, cleaned, voice, speed))
        return True

    def stop(self) -> None:
        with self._lock:
            self._generation += 1

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

            try:
                wav_bytes = self._synthesize_wav_fn(text, voice=voice, speed=speed)
            except Exception as exc:
                with self._lock:
                    if generation == self._generation:
                        print(f"TTS synthesis failed: {exc}")
                continue

            with self._lock:
                if generation != self._generation:
                    continue
            self._play_wav(generation, wav_bytes)

    def _play_wav(self, generation: int, wav_bytes: bytes) -> None:
        stream: sd.RawOutputStream | None = None
        try:
            with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
                sampwidth = wav_file.getsampwidth()
                if sampwidth != 2:
                    raise RuntimeError(f"Unsupported TTS sample width: {sampwidth}")

                stream = sd.RawOutputStream(
                    samplerate=wav_file.getframerate(),
                    channels=wav_file.getnchannels(),
                    dtype="int16",
                )
                stream.start()

                while True:
                    chunk = wav_file.readframes(4096)
                    if not chunk:
                        break
                    with self._lock:
                        if generation != self._generation:
                            break
                    stream.write(chunk)
        except Exception as exc:
            with self._lock:
                if generation == self._generation:
                    print(f"TTS playback failed: {exc}")
        finally:
            if stream is not None:
                stream.stop()
                stream.close()
