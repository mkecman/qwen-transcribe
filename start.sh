#!/bin/bash
# Starts Qwen Transcribe and opens it in your browser (Linux). Close this window to stop it.
cd "$(dirname "$0")"
if [ -z "$QT_BACKEND" ]; then
  if ! env/bin/python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
    QT_BACKEND=cpu
    export QT_BACKEND
    echo "No usable NVIDIA GPU found - running on the CPU (set QT_BACKEND=cuda to force the GPU)."
  fi
fi
env/bin/python -m app "$@" || read -n 1 -p "Press any key to close..."
