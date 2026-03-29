from __future__ import annotations

import logging
import os
import sys
from pathlib import Path


def _coerce_level(name: str | None) -> int:
    if not name:
        return logging.INFO
    candidate = getattr(logging, name.strip().upper(), None)
    return candidate if isinstance(candidate, int) else logging.INFO


def configure_logging(
    *,
    level_name: str | None = None,
    log_file: str | Path | None = None,
) -> None:
    root = logging.getLogger()
    level = _coerce_level(level_name or os.environ.get("BITWISPR_LOG_LEVEL", "INFO"))

    formatter = logging.Formatter(
        fmt="%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handlers: list[logging.Handler] = []
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)

    resolved_log_file = log_file or os.environ.get("BITWISPR_LOG_FILE", "").strip()
    if resolved_log_file:
        path = Path(resolved_log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    root.handlers.clear()
    root.setLevel(level)
    for handler in handlers:
        root.addHandler(handler)

    logging.captureWarnings(True)
