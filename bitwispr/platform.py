from __future__ import annotations

import os
import subprocess


def is_wayland_session() -> bool:
    return (
        os.environ.get("XDG_SESSION_TYPE") == "wayland"
        or os.environ.get("WAYLAND_DISPLAY") is not None
    )


def _run_text_command(command: list[str], timeout: float = 5.0) -> str | None:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None

    text = result.stdout.strip()
    return text or None


def read_selected_text(selection_mode: str, *, wayland: bool) -> str | None:
    modes = ["primary", "clipboard"] if selection_mode == "auto" else [selection_mode]

    for mode in modes:
        if wayland:
            commands = (
                [["wl-paste", "--primary", "--no-newline"]]
                if mode == "primary"
                else [["wl-paste", "--no-newline"]]
            )
        else:
            commands = (
                [
                    ["xclip", "-o", "-selection", "primary"],
                    ["xsel", "--primary", "--output"],
                ]
                if mode == "primary"
                else [
                    ["xclip", "-o", "-selection", "clipboard"],
                    ["xsel", "--clipboard", "--output"],
                ]
            )

        for command in commands:
            text = _run_text_command(command)
            if text:
                return text

    return None


def type_text(text: str, *, wayland: bool) -> bool:
    if not text:
        return True

    commands = (
        [
            ["ydotool", "type", "--", text],
            ["wtype", "--", text],
        ]
        if wayland
        else [["xdotool", "type", "--clearmodifiers", "--", text]]
    )

    for command in commands:
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                timeout=30,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            continue
        if result.returncode == 0:
            return True

    if wayland:
        print(
            "Could not type text. Install `ydotool` or `wtype` for Wayland output."
        )
    else:
        print("Could not type text. Install `xdotool` for X11 output.")
    print(f"Transcribed text was: {text}")
    return False

