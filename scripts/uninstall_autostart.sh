#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

systemctl --user disable --now bitwispr.service || true
rm -f "${SYSTEMD_USER_DIR}/bitwispr.service"

systemctl --user daemon-reload

echo "BitWispr autostart services removed."
