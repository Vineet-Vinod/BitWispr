#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"
SERVICE_PATH="${SYSTEMD_USER_DIR}/bitwispr.service"

systemctl --user disable --now bitwispr.service || true
rm -f "${SERVICE_PATH}"
systemctl --user daemon-reload

echo "BitWispr autostart service removed."

