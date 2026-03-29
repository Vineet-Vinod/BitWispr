from __future__ import annotations

import os
import subprocess
import threading
import time


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
        print("Could not type text. Install `ydotool` or `wtype` for Wayland output.")
    else:
        print("Could not type text. Install `xdotool` for X11 output.")
    print(f"Text was: {text}")
    return False


def run_hotkey_loop(
    *,
    on_dictation,
    on_reader,
    stop_event: threading.Event,
    wayland: bool,
    keyboard_scan_interval_sec: float,
) -> None:
    if wayland:
        _run_with_evdev(
            on_dictation=on_dictation,
            on_reader=on_reader,
            stop_event=stop_event,
            keyboard_scan_interval_sec=keyboard_scan_interval_sec,
        )
    else:
        _run_with_pynput(
            on_dictation=on_dictation,
            on_reader=on_reader,
            stop_event=stop_event,
        )


def _run_with_evdev(
    *,
    on_dictation,
    on_reader,
    stop_event: threading.Event,
    keyboard_scan_interval_sec: float,
) -> None:
    try:
        import evdev
        from evdev import ecodes
    except ImportError as exc:
        raise RuntimeError("`evdev` is required for Wayland hotkeys") from exc

    from selectors import DefaultSelector, EVENT_READ

    ctrl_keys = {ecodes.KEY_RIGHTCTRL}
    alt_keys = {ecodes.KEY_RIGHTALT}
    shift_keys = {ecodes.KEY_RIGHTSHIFT}
    if hasattr(ecodes, "KEY_ALTGR"):
        alt_keys.add(ecodes.KEY_ALTGR)
    if hasattr(ecodes, "KEY_ISO_LEVEL3_SHIFT"):
        alt_keys.add(ecodes.KEY_ISO_LEVEL3_SHIFT)

    selector = DefaultSelector()
    devices: dict[str, evdev.InputDevice] = {}
    state: dict[str, dict[str, bool]] = {}

    def is_keyboard_device(device: evdev.InputDevice) -> bool:
        try:
            capabilities = device.capabilities()
        except OSError:
            return False
        if ecodes.EV_KEY not in capabilities:
            return False
        key_caps = set(capabilities.get(ecodes.EV_KEY, []))
        required = {
            ecodes.KEY_A,
            ecodes.KEY_SPACE,
            ecodes.KEY_RIGHTCTRL,
            ecodes.KEY_RIGHTALT,
            ecodes.KEY_RIGHTSHIFT,
        }
        return bool(key_caps.intersection(required))

    def add_new_keyboards() -> None:
        for path in evdev.list_devices():
            if path in devices:
                continue
            try:
                device = evdev.InputDevice(path)
            except OSError:
                continue
            if not is_keyboard_device(device):
                continue
            try:
                selector.register(device, EVENT_READ)
            except Exception:
                device.close()
                continue
            devices[path] = device
            state[path] = {
                "ctrl": False,
                "alt": False,
                "shift": False,
                "dictation_latched": False,
                "read_latched": False,
            }
            print(f"  + Keyboard: {device.name} ({path})")

    def remove_keyboard(path: str) -> None:
        device = devices.pop(path, None)
        state.pop(path, None)
        if device is None:
            return
        try:
            selector.unregister(device)
        except Exception:
            pass
        try:
            device.close()
        except Exception:
            pass

    def update_combos(device_state: dict[str, bool], is_key_down: bool) -> None:
        dictation_down = device_state["ctrl"] and device_state["alt"]
        reader_down = device_state["ctrl"] and device_state["shift"]

        if dictation_down and is_key_down and not device_state["dictation_latched"]:
            on_dictation()
            device_state["dictation_latched"] = True
        elif not dictation_down:
            device_state["dictation_latched"] = False

        if reader_down and is_key_down and not device_state["read_latched"]:
            on_reader()
            device_state["read_latched"] = True
        elif not reader_down:
            device_state["read_latched"] = False

    add_new_keyboards()
    if not devices:
        print("No keyboard devices detected yet. Waiting for evdev devices...")

    last_scan = time.monotonic()
    while not stop_event.is_set():
        now = time.monotonic()
        if now - last_scan >= keyboard_scan_interval_sec:
            add_new_keyboards()
            last_scan = now

        for key, _ in selector.select(timeout=1.0):
            device = key.fileobj
            path = getattr(device, "path", "")
            if path not in state:
                continue
            try:
                events = device.read()
            except OSError:
                print(f"  - Keyboard disconnected: {device.name} ({path})")
                remove_keyboard(path)
                continue

            for event in events:
                if event.type != ecodes.EV_KEY:
                    continue

                is_key_down = event.value > 0
                device_state = state[path]
                if event.code in ctrl_keys:
                    device_state["ctrl"] = is_key_down
                elif event.code in alt_keys:
                    device_state["alt"] = is_key_down
                elif event.code in shift_keys:
                    device_state["shift"] = is_key_down
                else:
                    continue

                update_combos(device_state, event.value == 1)


def _run_with_pynput(*, on_dictation, on_reader, stop_event: threading.Event) -> None:
    try:
        from pynput import keyboard
    except ImportError as exc:
        raise RuntimeError("`pynput` is required for X11 hotkeys") from exc

    state = {
        "ctrl": False,
        "alt": False,
        "shift": False,
        "dictation_latched": False,
        "read_latched": False,
    }

    ctrl_keys = {keyboard.Key.ctrl_r}
    alt_keys = {keyboard.Key.alt_r}
    alt_gr = getattr(keyboard.Key, "alt_gr", None)
    if alt_gr is not None:
        alt_keys.add(alt_gr)
    shift_keys = {keyboard.Key.shift_r}

    def update_state(key, is_pressed: bool) -> None:
        if key in ctrl_keys:
            state["ctrl"] = is_pressed
        elif key in alt_keys:
            state["alt"] = is_pressed
        elif key in shift_keys:
            state["shift"] = is_pressed
        else:
            return

        dictation_down = state["ctrl"] and state["alt"]
        reader_down = state["ctrl"] and state["shift"]

        if dictation_down and is_pressed and not state["dictation_latched"]:
            on_dictation()
            state["dictation_latched"] = True
        elif not dictation_down:
            state["dictation_latched"] = False

        if reader_down and is_pressed and not state["read_latched"]:
            on_reader()
            state["read_latched"] = True
        elif not reader_down:
            state["read_latched"] = False

    def on_press(key) -> None:
        update_state(key, True)

    def on_release(key) -> None:
        update_state(key, False)

    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        while not stop_event.is_set():
            listener.join(0.5)
