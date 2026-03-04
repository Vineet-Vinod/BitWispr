#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

systemctl --user disable --now bitwispr-client.service || true
systemctl --user disable --now bitwispr-server.service || true

rm -f "${SYSTEMD_USER_DIR}/bitwispr-client.service"
rm -f "${SYSTEMD_USER_DIR}/bitwispr-server.service"

systemctl --user daemon-reload

echo "BitWispr autostart services removed."
