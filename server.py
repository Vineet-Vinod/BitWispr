"""
BitWispr Trillim server.
Runs LLM + Whisper on localhost:1111 with BitNet-TRNQ + GenZ adapter.
"""

import os
import signal
import time

from trillim import LLM, Server, Whisper
from trillim.model_store import resolve_model_dir

HOST = "127.0.0.1"
PORT = 1111
MODEL_ID = os.environ.get("BITWISPR_MODEL_ID", "Trillim/BitNet-TRNQ")
ADAPTER_ID = os.environ.get("BITWISPR_ADAPTER_ID", "Trillim/BitNet-GenZ-LoRA-TRNQ")
WHISPER_MODEL = os.environ.get("BITWISPR_WHISPER_MODEL", "base.en")
WHISPER_COMPUTE_TYPE = os.environ.get("BITWISPR_WHISPER_COMPUTE_TYPE", "int8")
WHISPER_CPU_THREADS = int(os.environ.get("BITWISPR_WHISPER_CPU_THREADS", "2"))
RESTART_DELAY_SEC = 2

_stop_requested = False


def _handle_stop_signal(signum, frame):
    global _stop_requested
    _stop_requested = True


def build_server() -> tuple[Server, str, str]:
    model_dir = resolve_model_dir(MODEL_ID)
    adapter_dir = resolve_model_dir(ADAPTER_ID)
    llm = LLM(model_dir=model_dir, adapter_dir=adapter_dir, num_threads=0)
    whisper = Whisper(
        model_size=WHISPER_MODEL,
        compute_type=WHISPER_COMPUTE_TYPE,
        cpu_threads=WHISPER_CPU_THREADS,
    )
    return Server(llm, whisper), model_dir, adapter_dir


def run_forever():
    signal.signal(signal.SIGINT, _handle_stop_signal)
    signal.signal(signal.SIGTERM, _handle_stop_signal)

    print("=" * 55)
    print("BitWispr Server")
    print("=" * 55)
    print(f"Host: {HOST}:{PORT}")
    print(f"Model ID: {MODEL_ID}")
    print(f"Adapter ID: {ADAPTER_ID}")
    print(f"Whisper model: {WHISPER_MODEL}")
    print("=" * 55)

    while not _stop_requested:
        try:
            server, model_dir, adapter_dir = build_server()
            print(f"Resolved model path: {model_dir}")
            print(f"Resolved adapter path: {adapter_dir}")
            print("Starting Trillim server...")
            server.run(host=HOST, port=PORT, log_level="info")
        except Exception as e:
            print(f"Server error: {e}")

        if _stop_requested:
            break

        print(f"Server exited. Restarting in {RESTART_DELAY_SEC}s...")
        time.sleep(RESTART_DELAY_SEC)

    print("Server stopped.")


if __name__ == "__main__":
    run_forever()
