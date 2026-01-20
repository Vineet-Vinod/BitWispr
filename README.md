# BitWispr

Real-time voice-to-text using OpenAI's Whisper. Press **Shift+Tab** to toggle recording, and text is transcribed and typed at your cursor when you're done recording.

## Features

- ⌨️ Types directly at cursor position
- 🐧 Works on both X11 and Wayland
- 🔒 Fully offline/local processing

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

### 2. Add yourself to the input group (Wayland only)

This allows reading keyboard events without sudo:

```bash
sudo usermod -aG input $USER
```

Then **log out and back in** for the group change to take effect.

### 3. Install Python dependencies

```bash
uv sync
```

## Usage

```bash
uv run main.py
```

- Press **Shift+Tab** to start recording
- Speak naturally (slower side is better)
- Press **Shift+Tab** again to stop
- Text is typed at your cursor in real-time as you speak

## Configuration

Edit `main.py` to change:

- `MODEL_SIZE`: `"tiny.en"` (fastest), `"base.en"`, `"small.en"`, `"medium.en"` (most accurate)
- For GPU acceleration, change `device="cpu"` to `device="cuda"`

## Troubleshooting

### "No keyboard found" error
Run with sudo or ensure you're in the `input` group:
```bash
groups  # Should show 'input'
```

### Text not typing on Wayland
Make sure `ydotool` is installed:
```bash
sudo apt install ydotool
```

### Permission denied errors
```bash
sudo uv run main.py  # Temporary workaround
```
