"""
Simple Python client for the local BitWispr Trillim server endpoints.
"""

import argparse
import json
import urllib.request
import uuid
from pathlib import Path


def get_json(url: str) -> dict:
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        method="POST",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_transcription(url: str, audio_path: Path, language: str = "en") -> dict:
    boundary = f"----bitwispr-{uuid.uuid4().hex}"
    file_bytes = audio_path.read_bytes()

    body = bytearray()
    fields = {
        "model": "whisper-1",
        "language": language,
        "response_format": "json",
    }
    for key, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            f'Content-Disposition: form-data; name="{key}"\r\n\r\n'.encode("utf-8")
        )
        body.extend(value.encode("utf-8"))
        body.extend(b"\r\n")

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(
        (
            f'Content-Disposition: form-data; name="file"; '
            f'filename="{audio_path.name}"\r\n'
        ).encode("utf-8")
    )
    body.extend(b"Content-Type: audio/wav\r\n\r\n")
    body.extend(file_bytes)
    body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(
        url,
        method="POST",
        data=bytes(body),
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    parser = argparse.ArgumentParser(description="Hit BitWispr server endpoints.")
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:1111",
        help="Trillim server base URL",
    )
    parser.add_argument(
        "--chat",
        default="Say hello in one sentence.",
        help="Message for /v1/chat/completions",
    )
    parser.add_argument(
        "--audio",
        type=Path,
        help="Optional WAV file for /v1/audio/transcriptions",
    )
    args = parser.parse_args()

    base = args.base_url.rstrip("/")

    print("GET /v1/models")
    models = get_json(f"{base}/v1/models")
    print(json.dumps(models, indent=2))

    print("\nPOST /v1/chat/completions")
    chat = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": "BitNet-TRNQ",
            "messages": [{"role": "user", "content": args.chat}],
        },
    )
    print(chat["choices"][0]["message"]["content"])

    if args.audio:
        print("\nPOST /v1/audio/transcriptions")
        tx = post_transcription(f"{base}/v1/audio/transcriptions", args.audio)
        print(tx.get("text", ""))


if __name__ == "__main__":
    main()
