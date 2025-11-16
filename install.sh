#!/bin/bash
# Bird-ID Automated Installer for Single Board Computers
# Zero-CLI appliance setup for Raspberry Pi, Orange Pi, etc.

set -e

INSTALL_DIR="/opt/bird-id"
SERVICE_NAME="bird-id"
WEB_PORT=8080
API_PORT=8000

echo "========================================="
echo "Bird-ID Appliance Installer"
echo "========================================="
echo ""

# Detect platform
if grep -qi "raspbian\|raspberry" /etc/os-release 2>/dev/null; then
    PLATFORM="Raspberry Pi"
elif grep -qi "debian" /etc/os-release 2>/dev/null; then
    PLATFORM="Debian-based SBC"
else
    PLATFORM="Unknown (proceeding anyway)"
fi

echo "Detected platform: $PLATFORM"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "ERROR: This installer must be run as root"
    echo "Please run: sudo $0"
    exit 1
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "[1/6] Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    rm get-docker.sh
    systemctl enable docker
    systemctl start docker
    echo "✓ Docker installed"
else
    echo "[1/6] Docker already installed"
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "[2/6] Installing Docker Compose..."
    apt-get update -qq
    apt-get install -y docker-compose
    echo "✓ Docker Compose installed"
else
    echo "[2/6] Docker Compose already installed"
fi

# Create installation directory
echo "[3/6] Setting up Bird-ID in $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR"
cp -r "$(dirname "$0")"/* "$INSTALL_DIR/"
chown -R root:root "$INSTALL_DIR"
chmod +x "$INSTALL_DIR/scripts/"*.sh

# Create systemd service for auto-start
echo "[4/6] Creating systemd service..."
cat > /etc/systemd/system/${SERVICE_NAME}.service <<'EOF'
[Unit]
Description=Bird-ID Classifier Service
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/bird-id
ExecStart=/usr/bin/docker-compose up -d birdid-tpu birdid-share
ExecStop=/usr/bin/docker-compose down
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable ${SERVICE_NAME}.service
echo "✓ Systemd service created"

# Check for EdgeTPU
echo "[5/6] Checking for Coral EdgeTPU..."
if lsusb | grep -qi "1a6e:089a\|18d1:9302"; then
    echo "✓ Coral EdgeTPU detected!"
    EDGETPU_DETECTED=1
    # Use TPU configuration by default
    sed -i 's/^#\?\s*birdid-tpu:/birdid-tpu:/' "$INSTALL_DIR/docker-compose.yaml"
    sed -i 's/^#\?\s*birdid-cpu:/#birdid-cpu:/' "$INSTALL_DIR/docker-compose.yaml"
else
    echo "⚠ No Coral EdgeTPU detected - will use CPU mode"
    EDGETPU_DETECTED=0
    # Use CPU configuration
    sed -i 's/^#\?\s*birdid-tpu:/#birdid-tpu:/' "$INSTALL_DIR/docker-compose.yaml"
    sed -i 's/^#\?\s*birdid-cpu:/birdid-cpu:/' "$INSTALL_DIR/docker-compose.yaml"
fi

# Set up configuration wizard service
echo "[6/6] Setting up first-run configuration wizard..."
mkdir -p /var/lib/bird-id
if [ ! -f /var/lib/bird-id/configured ]; then
    cat > /etc/systemd/system/bird-id-setup-wizard.service <<'EOF'
[Unit]
Description=Bird-ID First-Run Setup Wizard
After=network.target

[Service]
Type=simple
ExecStart=/opt/bird-id/scripts/setup_wizard.sh
Restart=no

[Install]
WantedBy=multi-user.target
EOF
    systemctl daemon-reload
    systemctl enable bird-id-setup-wizard.service
    echo "✓ Setup wizard enabled (will run on first boot)"
else
    echo "✓ System already configured"
fi

# Set hostname
if [ "$(hostname)" != "birdid" ]; then
    echo ""
    echo "Setting hostname to 'birdid' for easy access..."
    hostnamectl set-hostname birdid
    echo "✓ Hostname set to 'birdid'"
fi

echo ""
echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""

if [ ! -f /var/lib/bird-id/configured ]; then
    echo "NEXT STEPS:"
    echo "1. Reboot this device: sudo reboot"
    echo "2. After reboot, open your browser to:"
    echo "   http://birdid.local:$WEB_PORT"
    echo "3. Follow the setup wizard to configure your camera"
    echo ""
    echo "The setup wizard will:"
    echo "  - Guide you through camera configuration"
    echo "  - Auto-detect your Coral EdgeTPU (if connected)"
    echo "  - Start the Bird-ID service automatically"
else
    echo "Bird-ID is already configured!"
    echo "Access it at: http://birdid.local:$WEB_PORT"
    echo ""
    echo "To reconfigure, delete: /var/lib/bird-id/configured"
    echo "Then reboot to restart the setup wizard"
fi

echo ""
echo "Service management:"
echo "  Start:   sudo systemctl start $SERVICE_NAME"
echo "  Stop:    sudo systemctl stop $SERVICE_NAME"
echo "  Status:  sudo systemctl status $SERVICE_NAME"
echo "  Logs:    sudo journalctl -u $SERVICE_NAME -f"
echo ""
