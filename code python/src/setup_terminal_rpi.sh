#!/usr/bin/env bash
# =============================================================================
# Raspberry Pi Project Setup Script
# Run from inside the cloned repo on a fresh Raspberry Pi OS install.
#
# Usage:
#   git clone <your-repo-url>
#   cd <repo-folder>
#   chmod +x setup.sh && ./setup.sh
#
# Notes:
#   - Boots to console (no desktop) for best VLC performance via DRM
#   - Tailscale provides SSH access from anywhere
#   - For 1080p VLC on Pi 3B: always use --codec=mmal or --codec=v4l2 to enable
#     hardware decoding, otherwise the CPU will struggle
# =============================================================================

set -euo pipefail

# --- Config ------------------------------------------------------------------
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"  # wherever this script lives
VENV_DIR="$REPO_DIR/.venv"
USER="${SUDO_USER:-$(whoami)}"
HOME_DIR="/home/$USER"
# -----------------------------------------------------------------------------

echo "============================================="
echo " Raspberry Pi Project Setup"
echo " Repo: $REPO_DIR"
echo "============================================="

# --- 0. Sudo -----------------------------------------------------------------
# Refresh sudo credentials up-front so later sudo calls don't re-prompt.
# sudo -v

# --- 1. System packages -------------------------------------------------------
echo ""
echo "[1/4] Installing system packages..."

sudo apt-get update -y
sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    tmux \
    vlc

echo "  ✓ System packages installed"

# --- 2. Tailscale -------------------------------------------------------------
echo ""
echo "[2/4] Installing Tailscale..."

if command -v tailscale &>/dev/null; then
    echo "  Tailscale already installed, skipping"
else
    curl -fsSL https://tailscale.com/install.sh | sh
    sudo systemctl enable tailscaled
    sudo systemctl start tailscaled
    echo "  ✓ Tailscale installed and service enabled"
fi

# Bring Tailscale up with SSH enabled (will prompt for auth on first run)
sudo tailscale up --ssh
echo "  ✓ Tailscale is up (SSH enabled)"

# --- 3. Python venv + dependencies --------------------------------------------
echo ""
echo "[3/4] Setting up Python virtual environment..."

if [ -d "$VENV_DIR" ]; then
    echo "  Venv already exists, skipping creation"
else
    python3 -m venv "$VENV_DIR"
    echo "  ✓ Venv created"
fi

"$VENV_DIR/bin/pip" install --upgrade pip --quiet

if [ -f "$REPO_DIR/requirements.txt" ]; then
    "$VENV_DIR/bin/pip" install -r "$REPO_DIR/requirements.txt"
    echo "  ✓ Dependencies installed from requirements.txt"
else
    echo "  ⚠️  No requirements.txt found — skipping pip install"
fi

echo "  ✓ Venv ready at $VENV_DIR"

# --- 4. Console auto-login + autostart via tmux ------------------------------
echo ""
echo "[4/4] Configuring console auto-login and app autostart..."

# Boot to console with auto-login (B2 = CLI autologin)
# No desktop environment — VLC renders directly via DRM for best performance.
sudo raspi-config nonint do_boot_behaviour B2

# Ensure run.sh is executable
if [ -f "$REPO_DIR/run.sh" ]; then
    chmod +x "$REPO_DIR/run.sh"
    echo "  ✓ run.sh is executable"
else
    echo "  ⚠️  No run.sh found — create one to define how your app starts"
fi

# Add tmux autostart to .bashrc
BASHRC="$HOME_DIR/.bashrc"
MARKER="# >>> kiosk autostart >>>"
if ! grep -qF "$MARKER" "$BASHRC"; then
    cat >> "$BASHRC" << EOF

$MARKER
if [ -z "\$TMUX" ]; then
    tmux new-session -A -s kiosk "$REPO_DIR/run.sh"
fi
# <<< kiosk autostart <<<
EOF
    echo "  ✓ Autostart added to $BASHRC"
else
    echo "  Autostart already in $BASHRC, skipping"
fi

# --- Done ---------------------------------------------------------------------
echo ""
echo "============================================="
echo " Setup complete!"
echo ""
echo " What was configured:"
echo "   - System packages (python3, tmux, vlc)"
echo "   - Tailscale with SSH (already connected)"
echo "   - Python venv with dependencies"
echo "   - Console autologin (no desktop)"
echo "   - Tmux kiosk autostart in .bashrc"
echo ""
echo " Next steps:"
echo "   1. Reboot to apply console autologin + autostart: sudo reboot"
echo "   2. SSH in from anywhere via Tailscale: ssh $USER@$(hostname)"
echo "   3. To access the running app: tmux attach -t kiosk"
echo "   4. To detach without stopping: Ctrl+B then D"
echo "============================================="
