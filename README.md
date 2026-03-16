# BitWispr

BitWispr is one local background app with two desktop actions built on the public
Trillim SDK abstractions:

- `Runtime(Whisper(...), TTS(...))`
- local dictation using Whisper speech-to-text
- highlight-to-read using Trillim TTS

There is no Discord responder in this rebuild.

## Hotkeys

- `Right Ctrl + Right Alt`: start/stop dictation
- `Right Ctrl + Right Shift`: read the current highlighted text aloud

Dictation types the transcript into the active app. Highlight-to-read pulls text
from the current primary selection first and then falls back to the clipboard when
`BITWISPR_SELECTION_MODE=auto`.

## Linux dependencies

```bash
# Python audio bindings
sudo apt install portaudio19-dev

# X11 typing + selection
sudo apt install xdotool xclip

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

Run manually without installing the service:

```bash
uv sync
uv run main.py
```

## Configuration

BitWispr loads `.env` if present and respects these variables:

```bash
BITWISPR_WHISPER_MODEL=base.en
BITWISPR_WHISPER_COMPUTE_TYPE=int8
BITWISPR_WHISPER_CPU_THREADS=2
BITWISPR_WHISPER_LANGUAGE=en
BITWISPR_WHISPER_TIMEOUT_SEC=120
BITWISPR_MIN_RECORDING_SEC=0.3
BITWISPR_DICTATION_SUFFIX=" "
BITWISPR_TTS_VOICE=alba
BITWISPR_TTS_SPEED=1.0
BITWISPR_TTS_TIMEOUT_SEC=60
BITWISPR_TTS_VOICES_DIR=~/.trillim/voices
BITWISPR_SELECTION_MODE=auto
BITWISPR_KEYBOARD_SCAN_INTERVAL_SEC=60
BITWISPR_AUDIO_SAMPLE_RATE=
```

If you register custom Trillim voices later, point `BITWISPR_TTS_VOICE` at that
voice ID or change `read_voice_for()` in `bitwispr/app.py`.

## Trillim Docs

The bundled SDK docs are in:

```bash
.venv/lib/python3.12/site-packages/trillim/docs/
```

The relevant references for this app were:

- `components.md`
- `install-linux.md`
