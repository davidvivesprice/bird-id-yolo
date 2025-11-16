#!/bin/bash
# Bird-ID Setup Script
# Double-click this file ONCE to install dependencies

cd "$(dirname "$0")"

echo "================================================"
echo "  Bird-ID Setup - Installing Dependencies"
echo "================================================"
echo ""

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found at venv/"
    echo "Please create it first or contact the other developer."
    exit 1
fi

echo "✓ Found virtual environment"
echo ""

# Activate venv and install
echo "Installing Python packages..."
source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ $? -eq 0 ]; then
    echo ""
    echo "================================================"
    echo "  ✅ Setup Complete!"
    echo "================================================"
    echo ""
    echo "You can now use:"
    echo "  • test-motion.command - Process a single video"
    echo "  • process-clips.command - Process all videos in clips folder"
    echo ""
else
    echo ""
    echo "❌ Installation failed. Check errors above."
    exit 1
fi

echo "Press any key to close..."
read -n 1
