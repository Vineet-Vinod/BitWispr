#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_PATH="${SYSTEMD_USER_DIR}/bitwispr.service"
UV_BIN="$(command -v uv)"

if [[ -z "${UV_BIN}" ]]; then
    echo "uv is required but was not found in PATH."
    exit 1
fi

cd "${REPO_DIR}"
"${UV_BIN}" sync

mkdir -p "${SYSTEMD_USER_DIR}"

cat > "${SERVICE_PATH}" <<EOF
[Unit]
Description=BitWispr local dictation and read-aloud daemon
After=graphical-session.target
Wants=graphical-session.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=${UV_BIN} run main.py
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now bitwispr.service

echo
echo "BitWispr is installed and running."
echo "Logs: journalctl --user -u bitwispr.service -f"
echo "X11 tools: xdotool + xclip (or xsel)"
echo "Wayland tools: wl-clipboard + ydotool (or wtype)"
echo "Wayland input capture may require: sudo usermod -aG input \$USER"

