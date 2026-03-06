# BitWispr

BitWispr is a single-process local app that shares one Trillim runtime for:
- LLM (`BitNet-TRNQ` + `BitNet-GenZ-LoRA-TRNQ`)
- Whisper speech-to-text

There is no local HTTP server/client split anymore. `main.py` instantiates `LLM` and `Whisper` directly and uses them in-process.

## Features

- Types transcribed text directly at your cursor
- Works on X11 and Wayland
- Uses local Trillim components (no cloud dependency)
- Hotkey is **right-only** `Right Ctrl + Right Alt`
- Optional Discord auto-responder using the same in-process LLM runtime

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
successful replies enters fast mode (`DISCORD_FAST_POLL_SEC`) for
`DISCORD_FAST_WINDOW_SEC` seconds. If channel context overflows, old turns are trimmed
and retried automatically.

## Autostart (always running)

Install a user service so BitWispr starts at login and auto-restarts:

```bash
./scripts/install_autostart.sh
```

Check status:

```bash
systemctl --user status bitwispr.service
```

Stop/remove autostart:

```bash
./scripts/uninstall_autostart.sh
```

## Run manually

```bash
uv run main.py
```

- Press `Right Ctrl + Right Alt` to start recording
- Press `Right Ctrl + Right Alt` again to stop and transcribe
- Text is typed at the active cursor position

## Trillim docs in downloaded package

Trillim bundles docs as `.md` files inside the installed package:

```bash
find .venv/lib/python3.12/site-packages/trillim/docs -maxdepth 1 -type f -name "*.md" | sort
```

Important references:
- `.venv/lib/python3.12/site-packages/trillim/docs/server.md`
- `.venv/lib/python3.12/site-packages/trillim/docs/cli.md`
