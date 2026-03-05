# BitWispr

BitWispr now uses a local Trillim server for both:
- LLM (`BitNet-TRNQ` + `BitNet-GenZ-LoRA-TRNQ`)
- Whisper speech-to-text (`/v1/audio/transcriptions`)

The recorder client (`main.py`) captures audio and sends it to the local server on `http://127.0.0.1:1111`.

## Features

- Types transcribed text directly at your cursor
- Works on X11 and Wayland
- Uses local Trillim endpoints (no cloud dependency)
- Hotkey is **right-only** `Right Ctrl + Right Alt`

## Setup

### 1. Install system dependencies

```bash
# Audio library
sudo apt install portaudio19-dev

# For Wayland typing
sudo apt install ydotool

# For X11 typing
sudo apt install xdotool
```

### 2. Add yourself to input group (Wayland only)

```bash
sudo usermod -aG input $USER
```

Log out and back in after changing groups.

### 3. Install Python dependencies

```bash
uv sync
```

### 4. Configure Discord auto-responder (optional)

Edit `.env`:

```bash
DISCORD_AUTH_TOKEN=Bot <your_token_or_auth_header>
DISCORD_CHANNEL_IDS=123456789012345678,234567890123456789
DISCORD_FAST_POLL_SEC=5
DISCORD_FAST_WINDOW_SEC=300
DISCORD_BACKOFF_FACTOR=2
DISCORD_BACKOFF_MAX_SEC=900
```

`DISCORD_CHANNEL_IDS` is a comma-separated list of Discord channel IDs.
Polling is channel-specific: each channel backs off independently, and any channel with
a successful reply enters a fast mode (`DISCORD_FAST_POLL_SEC`) for
`DISCORD_FAST_WINDOW_SEC` seconds. Each additional successful reply resets that fast window.

## Autostart (always running)

Install user services so server + client start at login and auto-restart:

```bash
./scripts/install_autostart.sh
```

Check status:

```bash
systemctl --user status bitwispr-server.service bitwispr-client.service
```

Stop/remove autostart:

```bash
./scripts/uninstall_autostart.sh
```

## Run manually

### Start always-on Trillim server (localhost:1111)

```bash
uv run server.py
```

`server.py` runs with:
- model: `Trillim/BitNet-TRNQ`
- adapter: `Trillim/BitNet-GenZ-LoRA-TRNQ`
- components: `LLM + Whisper`
- host/port: `127.0.0.1:1111`
- optional Discord poller that auto-replies to non-self messages in configured channels

The script auto-restarts the server if it exits unexpectedly.

### Start recorder client

```bash
uv run main.py
```

- Press `Right Ctrl + Right Alt` to start recording
- Press `Right Ctrl + Right Alt` again to stop and transcribe
- Text is typed at the active cursor position

## Endpoint client script

Use the helper script to hit server endpoints directly:

```bash
uv run api_client.py --chat "Summarize this repo in one sentence."
```

Optional audio transcription test:

```bash
uv run api_client.py --audio /path/to/sample.wav
```

## Trillim docs in downloaded package

Trillim bundles docs as `.md` files inside the installed package:

```bash
find .venv/lib/python3.12/site-packages/trillim/docs -maxdepth 1 -type f -name "*.md" | sort
```

Important references:
- `.venv/lib/python3.12/site-packages/trillim/docs/server.md`
- `.venv/lib/python3.12/site-packages/trillim/docs/cli.md`
