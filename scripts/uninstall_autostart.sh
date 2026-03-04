#!/usr/bin/env bash
set -euo pipefail

SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

# Remove legacy unit if present.
systemctl --user disable --now bitwisp.service || true

systemctl --user disable --now bitwispr-client.service || true
systemctl --user disable --now bitwispr-server.service || true

rm -f "${SYSTEMD_USER_DIR}/bitwisp.service"
rm -f "${SYSTEMD_USER_DIR}/bitwispr-client.service"
rm -f "${SYSTEMD_USER_DIR}/bitwispr-server.service"

systemctl --user daemon-reload
systemctl --user reset-failed bitwisp.service || true

echo "BitWispr autostart services removed."
