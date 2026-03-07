from __future__ import annotations

import asyncio
import concurrent.futures
import threading

from trillim import LLM, Whisper
from trillim.model_store import resolve_model_dir


class ContextWindowExceededError(RuntimeError):
    """Raised when prompt is larger than the model context window."""


class BitWisprRuntime:
    """Shared in-process runtime for Trillim LLM + Whisper."""

    _CHAT_SENTINELS = (
        "[Spin-Jump-Spinning...]",
        "[Searching:",
        "[Synthesizing...]",
        "[Search unavailable]",
        "\n--- Step ",
        "[Search results]\n",
    )

    def __init__(
        self,
        model_id: str,
        adapter_id: str,
        whisper_model: str,
        whisper_compute_type: str,
        whisper_cpu_threads: int,
        llm_timeout_sec: float,
    ):
        self.model_id = model_id
        self.adapter_id = adapter_id
        self.model_dir = resolve_model_dir(model_id)
        self.adapter_dir = resolve_model_dir(adapter_id) if adapter_id else None
        self.llm_timeout_sec = max(5.0, llm_timeout_sec)

        self.llm = LLM(
            model_dir=self.model_dir,
            adapter_dir=self.adapter_dir,
            num_threads=0,
        )
        self.whisper = Whisper(
            model_size=whisper_model,
            compute_type=whisper_compute_type,
            cpu_threads=whisper_cpu_threads,
        )

        self._loop = asyncio.new_event_loop()
        self._thread: threading.Thread | None = None
        self._start_event = threading.Event()
        self._started = False
        self._start_error: Exception | None = None
        self._chat_lock = threading.Lock()

    async def _async_start(self) -> None:
        await self.llm.start()
        await self.whisper.start()

    async def _async_stop(self) -> None:
        await self.whisper.stop()
        await self.llm.stop()

    def _loop_worker(self) -> None:
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._async_start())
            self._started = True
        except Exception as exc:
            self._start_error = exc
            try:
                self._loop.run_until_complete(self._async_stop())
            except Exception:
                pass
        finally:
            self._start_event.set()

        if not self._started:
            self._loop.close()
            return

        self._loop.run_forever()
        try:
            self._loop.run_until_complete(self._async_stop())
        finally:
            self._loop.close()

    def start(self) -> None:
        if self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._loop_worker,
            daemon=True,
            name="bitwispr-runtime",
        )
        self._thread.start()

        if not self._start_event.wait(timeout=600):
            raise RuntimeError("Timed out while initializing runtime")
        if self._start_error is not None:
            raise RuntimeError(str(self._start_error)) from self._start_error
        if not self._started:
            raise RuntimeError("Runtime did not start")

    def stop(self) -> None:
        if self._thread is None:
            return
        if self._started and not self._loop.is_closed():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=30)
        self._thread = None
        self._started = False

    def _run_coroutine(self, coro, timeout: float):
        if not self._started:
            raise RuntimeError("Runtime is not started")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError as exc:
            future.cancel()
            raise RuntimeError(f"Timed out after {timeout:.1f}s") from exc

    async def _chat_async(self, messages: list[dict]) -> str:
        if self.llm.engine is None or self.llm.harness is None:
            raise RuntimeError("LLM engine is not available")

        prompt = [{"role": item["role"], "content": item["content"]} for item in messages]
        token_ids, _ = self.llm.harness._prepare_tokens(prompt)
        max_ctx = self.llm.engine.arch_config.max_position_embeddings
        if len(token_ids) >= max_ctx:
            raise ContextWindowExceededError(
                f"Prompt length ({len(token_ids)} tokens) exceeds context window ({max_ctx})"
            )

        chunks: list[str] = []
        async for chunk in self.llm.harness.run(prompt):
            if any(chunk.startswith(sentinel) for sentinel in self._CHAT_SENTINELS):
                continue
            chunks.append(chunk)
        return "".join(chunks).strip()

    def chat(self, messages: list[dict]) -> str:
        with self._chat_lock:
            return self._run_coroutine(
                self._chat_async(messages),
                timeout=self.llm_timeout_sec,
            )

    async def _transcribe_async(self, audio_bytes: bytes, language: str | None) -> str:
        if self.whisper.engine is None:
            raise RuntimeError("Whisper engine is not available")
        return (await self.whisper.engine.transcribe(audio_bytes, language=language)).strip()

    def transcribe(self, audio_bytes: bytes, language: str | None = None) -> str:
        return self._run_coroutine(
            self._transcribe_async(audio_bytes, language),
            timeout=120,
        )
