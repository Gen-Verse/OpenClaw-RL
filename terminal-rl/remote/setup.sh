# System dependencies
if ! command -v docker &> /dev/null; then
  echo "Docker not found, installing..."
  sudo apt-get update
  sudo apt-get install -y docker.io
  sudo apt-get install -y docker-compose-v2
  sudo apt-get install -y docker-compose-plugin
  sudo apt-get install -y docker-compose
else
  echo "Docker found, skipping installation."
fi

# Modify docker to increase network address pool
DOCKER_DAEMON_CONFIG='/etc/docker/daemon.json'

# Backup existing daemon.json if it exists
if [ -f "$DOCKER_DAEMON_CONFIG" ]; then
    echo "Backing up existing Docker daemon configuration..."
    sudo cp "$DOCKER_DAEMON_CONFIG" "${DOCKER_DAEMON_CONFIG}.backup.$(date +%Y%m%d_%H%M%S)"
fi

# Create or update daemon.json with network pool settings
echo "Configuring Docker daemon..."
sudo tee "$DOCKER_DAEMON_CONFIG" > /dev/null <<EOF
{
  "default-address-pools": [
    {
      "base": "10.200.0.0/16",
      "size": 24
    }
  ]
}
EOF
# Restart Docker to apply changes
echo "Restarting Docker daemon..."
sudo systemctl restart docker
echo "Docker configuration complete!"

# install uv if not found
if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
else
  echo "uv found, skipping installation."
fi



# Setup Python env
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install terminal-bench
uv pip install camel-ai
uv pip install fastapi
uv pip install uvicorn
