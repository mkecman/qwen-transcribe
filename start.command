#!/bin/bash
# Starts Qwen Transcribe and opens it in your browser (macOS). Close this window to stop it.
cd "$(dirname "$0")"
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"  # ffmpeg from Homebrew
env/bin/python -m app "$@" || read -n 1 -p "Press any key to close..."
