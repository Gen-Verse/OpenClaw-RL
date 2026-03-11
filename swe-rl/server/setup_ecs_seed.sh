#!/usr/bin/env bash
# Setup script for the Volcengine ECS "seed" instance.
# Run this ONCE on a fresh ECS to install Docker, swe_exec_server, and pull
# all SWE-Bench images. Then snapshot the ECS into a custom image.
#
# Prerequisites:
#   - A fresh Ubuntu 22.04 ECS instance with >= 1TB disk
#   - train.jsonl copied to ~/train.jsonl on this ECS
#   - This script + swe_exec_server.py + pull_swe_images.sh copied to ~/
#
# Usage:
#   bash setup_ecs_seed.sh

set -euo pipefail

echo "========================================"
echo "SWE ECS Seed Setup — $(date)"
echo "========================================"

# ── 1. Install Docker ─────────────────────────────────────────────────
if ! command -v docker &>/dev/null; then
    echo "[1/4] Installing Docker..."
    curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | apt-key add -
    add-apt-repository "deb [arch=amd64] https://mirrors.aliyun.com/docker-ce/linux/ubuntu $(lsb_release -cs) stable"
    apt-get update
    apt-get install -y docker-ce docker-ce-cli containerd.io
    systemctl enable docker
    systemctl start docker
    echo "Docker installed: $(docker --version)"
else
    echo "[1/4] Docker already installed: $(docker --version)"
fi

# ── 2. Install Python dependencies for swe_exec_server ────────────────
echo "[2/4] Installing Python dependencies..."
apt-get update -qq
apt-get install -y -qq python3 python3-pip > /dev/null
pip3 install flask --quiet

# ── 3. Install swe_exec_server as a systemd service ──────────────────
echo "[3/4] Setting up swe_exec_server..."
mkdir -p /opt/swe-exec-server

if [ -f ~/swe_exec_server.py ]; then
    cp ~/swe_exec_server.py /opt/swe-exec-server/server.py
elif [ -f "$(dirname "$0")/swe_exec_server.py" ]; then
    cp "$(dirname "$0")/swe_exec_server.py" /opt/swe-exec-server/server.py
else
    echo "ERROR: swe_exec_server.py not found. Copy it to ~/ first."
    exit 1
fi

cat > /etc/systemd/system/swe-exec-server.service <<'EOF'
[Unit]
Description=SWE Docker Exec Server
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/bin/python3 /opt/swe-exec-server/server.py --port 5000
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable swe-exec-server
systemctl start swe-exec-server

for i in {1..10}; do
    if curl -fsS http://localhost:5000/healthz >/dev/null 2>&1; then
        echo "swe_exec_server is running on :5000"
        break
    fi
    sleep 1
done

# ── 4. Pull SWE-Bench Docker images ──────────────────────────────────
TRAIN=${TRAIN:-${HOME}/train.jsonl}
if [ ! -f "${TRAIN}" ]; then
    echo "[4/4] SKIPPED: ${TRAIN} not found."
    echo "  Copy train.jsonl to ${TRAIN} and run:"
    echo "    TRAIN=${TRAIN} bash ~/pull_swe_images.sh"
else
    echo "[4/4] Pulling SWE-Bench Docker images from ${TRAIN}..."
    PULL_SCRIPT=""
    if [ -f ~/pull_swe_images.sh ]; then
        PULL_SCRIPT=~/pull_swe_images.sh
    elif [ -f "$(dirname "$0")/pull_swe_images.sh" ]; then
        PULL_SCRIPT="$(dirname "$0")/pull_swe_images.sh"
    fi

    if [ -n "${PULL_SCRIPT}" ]; then
        LOG_DIR=/var/log/swe-images TRAIN="${TRAIN}" bash "${PULL_SCRIPT}"
    else
        echo "  WARNING: pull_swe_images.sh not found, skipping image pull."
        echo "  Copy it and run manually."
    fi
fi

echo ""
echo "========================================"
echo "Setup complete — $(date)"
echo ""
echo "Next steps:"
echo "  1. Verify: curl http://localhost:5000/healthz"
echo "  2. Check images: curl http://localhost:5000/images | python3 -m json.tool | head"
echo "  3. Stop this ECS in Volcengine console"
echo "  4. Create a custom image from this ECS"
echo "  5. Use that image_id in your training script"
echo "========================================"
