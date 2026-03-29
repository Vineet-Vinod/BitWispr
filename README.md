# BitWispr

BitWispr is one local background app with three features:

- dictation with `Right Ctrl + Right Alt`
- highlighted-text reader with `Right Ctrl + Right Shift`
- Discord responder controlled from a Discord channel

The app uses the Trillim Python SDK directly for:

- `STT()` for dictation
- `TTS()` for reader playback
- `LLM("Trillim/BitNet-TRNQ", lora_dir="Trillim/BitNet-GenZ-LoRA-TRNQ")` for Discord replies

Only OS integration and Discord REST polling are custom.

## Hotkeys

- `Right Ctrl + Right Alt`: toggle dictation
- `Right Ctrl + Right Shift`: read the current selection aloud

Dictation types the transcript into the active app. Reader uses the current saved
voice and speed.

## Discord Control

Set `CONTROL_CHANNEL` in `.env`. BitWispr polls that channel every 30 seconds and
applies new commands oldest to newest from the last 15 messages.

Supported commands:

- `LIST VOICE`
- `SET VOICE <voice_id>`
- `ADD CHANNEL <channel_id> <name>`
- `LIST CHANNEL`
- `DELETE CHANNEL <name>`
- `START`
- `STOP`
- `SET SPEED <float_value>`

Mutable state is persisted to a local JSON file, so voice, speed, active/stopped
status, and channel mappings survive restarts.

## Discord Responder

Configured response channels are polled with this schedule:

- start at 10 seconds
- stay at 10 seconds until 6 polls produce no reply
- then exponentially back off to 15 minutes
- any sent reply resets the poll interval to 10 seconds

For a reply, BitWispr fetches the last 15 messages and sends the model one prompt
formatted as:

```text
username: message
username: message
logged_in_user:
```

If the newest message was sent by the logged-in user, BitWispr does nothing.

## Linux Dependencies

```bash
# Python audio bindings
sudo apt install portaudio19-dev

# X11 typing + selection
sudo apt install xdotool xclip

# Optional X11 selection fallback
sudo apt install xsel

# Wayland selection
sudo apt install wl-clipboard

# Wayland typing
sudo apt install ydotool
# or
sudo apt install wtype
```

For Wayland global hotkeys, `evdev` access may require:

```bash
sudo usermod -aG input $USER
```

Log out and back in after changing groups.

## Setup

Install dependencies, create the user service, and start the app:

```bash
./scripts/install_autostart.sh
```

Remove the user service:

```bash
./scripts/uninstall_autostart.sh
```

Run manually:

```bash
uv sync
uv run main.py
```

## Configuration

BitWispr loads `.env` if present and respects these variables:

```bash
DISCORD_AUTH_TOKEN=
CONTROL_CHANNEL=
BITWISPR_MODEL_ID=Trillim/BitNet-TRNQ
BITWISPR_ADAPTER_ID=Trillim/BitNet-GenZ-LoRA-TRNQ
BITWISPR_WHISPER_LANGUAGE=en
BITWISPR_MIN_RECORDING_SEC=0.3
BITWISPR_DICTATION_SUFFIX=" "
BITWISPR_TTS_VOICE=alba
BITWISPR_TTS_SPEED=1.0
BITWISPR_SELECTION_MODE=auto
BITWISPR_KEYBOARD_SCAN_INTERVAL_SEC=60
BITWISPR_AUDIO_SAMPLE_RATE=
BITWISPR_STATE_PATH=~/.local/state/bitwispr/state.json
BITWISPR_CONTROL_POLL_INTERVAL_SEC=30
BITWISPR_RESPONDER_POLL_INTERVAL_SEC=10
BITWISPR_RESPONDER_IDLE_POLLS=6
BITWISPR_RESPONDER_BACKOFF_FACTOR=2
BITWISPR_RESPONDER_MAX_POLL_INTERVAL_SEC=900
BITWISPR_LLM_REPLY_MAX_CHARS=1800
```
