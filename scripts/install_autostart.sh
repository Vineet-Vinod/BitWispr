#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SYSTEMD_USER_DIR="${HOME}/.config/systemd/user"

mkdir -p "${SYSTEMD_USER_DIR}"

cat > "${SYSTEMD_USER_DIR}/bitwispr-server.service" <<EOF
[Unit]
Description=BitWispr Trillim Server
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/scripts/run_server.sh
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

cat > "${SYSTEMD_USER_DIR}/bitwispr-client.service" <<EOF
[Unit]
Description=BitWispr Recorder Client
After=bitwispr-server.service
Wants=bitwispr-server.service

[Service]
Type=simple
WorkingDirectory=${REPO_DIR}
ExecStart=${REPO_DIR}/scripts/run_client.sh
Restart=always
RestartSec=2
Environment=PYTHONUNBUFFERED=1

[Install]
WantedBy=default.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now bitwispr-server.service
systemctl --user enable --now bitwispr-client.service

echo
echo "BitWispr services are installed and started:"
systemctl --user --no-pager --full status bitwispr-server.service bitwispr-client.service || true
echo
echo "To make user services keep running after logout (optional), run:"
echo "  sudo loginctl enable-linger ${USER}"
