from __future__ import annotations

import logging
import threading

from trillim import LLM, Runtime

from bitwispr.config import AppConfig

logger = logging.getLogger(__name__)


class LLMController:
    def __init__(self, config: AppConfig):
        self.config = config
        self._lock = threading.Lock()
        self._runtime: Runtime | None = None

    def is_active(self) -> bool:
        with self._lock:
            return self._runtime is not None

    def activate(self) -> bool:
        with self._lock:
            if self._runtime is not None:
                return False

            logger.info(
                "Activating LLM runtime with model=%s adapter=%s",
                self.config.model_id,
                self.config.adapter_id,
            )
            runtime = Runtime(
                LLM(self.config.model_id, lora_dir=self.config.adapter_id)
            )
            try:
                runtime.start()
            except Exception:
                logger.exception("Failed to activate LLM runtime")
                try:
                    runtime.stop()
                except Exception:
                    logger.debug(
                        "Ignoring LLM runtime cleanup failure after startup error",
                        exc_info=True,
                    )
                raise

            self._runtime = runtime
            logger.info("LLM runtime activated")
            return True

    def deactivate(self) -> bool:
        with self._lock:
            runtime = self._runtime
            if runtime is None:
                return False

            logger.info("Deactivating LLM runtime")
            self._runtime = None
            runtime.stop()
            logger.info("LLM runtime deactivated")
            return True

    def chat(self, messages: list[dict[str, object]]) -> str:
        with self._lock:
            runtime = self._runtime
            if runtime is None:
                raise RuntimeError("LLM is inactive")
            if not messages:
                raise ValueError("LLM chat requires at least one message")

            *history, final_message = messages
            final_content = self._message_content(final_message)
            with runtime.llm.open_session() as session:
                for message in history:
                    session.append_message(
                        self._message_role(message),
                        self._message_content(message),
                    )
                return session.collect(final_content)

    def shutdown(self) -> None:
        try:
            self.deactivate()
        except Exception:
            logger.exception("LLM shutdown failed")

    @staticmethod
    def _message_role(message: dict[str, object]) -> str:
        role = message.get("role", "user")
        if not isinstance(role, str):
            return "user"
        return role

    @staticmethod
    def _message_content(message: dict[str, object]) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return str(content)
