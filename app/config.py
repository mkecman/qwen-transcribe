import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# "cuda" (Windows/Linux, NVIDIA GPU) or "mlx" (macOS, Apple Silicon)
BACKEND = os.environ.get("QT_BACKEND", "mlx" if sys.platform == "darwin" else "cuda")
MODEL_ID = "Qwen/Qwen3-ASR-1.7B"
MODEL_PATH = Path(os.environ.get("QT_MODEL_PATH", ROOT / "models" / "Qwen3-ASR-1.7B"))
# MLX only: small model that speeds up decoding without changing the text ("none" = off)
DRAFT_MODEL = os.environ.get("QT_DRAFT_MODEL", "moona3k/mlx-qwen3-asr-0.6b-4bit")
DRAFT_MODEL = "" if DRAFT_MODEL.lower() == "none" else DRAFT_MODEL
DATA_DIR = Path(os.environ.get("QT_DATA_DIR", ROOT / "data"))
DB_PATH = DATA_DIR / "transcribe.db"
AUDIO_DIR = DATA_DIR / "audio"
WEB_DIR = ROOT / "web"

HOST = os.environ.get("QT_HOST", "127.0.0.1")
PORT = int(os.environ.get("QT_PORT", "8765"))

_env_ffmpeg = ROOT / "env" / "Library" / "bin" / "ffmpeg.exe"
FFMPEG = str(_env_ffmpeg) if _env_ffmpeg.exists() else (shutil.which("ffmpeg") or "ffmpeg")

DEFAULT_SETTINGS = {
    "language": "English",        # a Qwen language name, or "auto"
    "context": "",                # names / jargon to bias recognition
    "pause_ms": 700,              # silence that ends a live segment
    "paragraph_after_s": 0,       # auto paragraph break after a pause this long (0 = off)
    "remove_fillers": True,       # drop "um", "uh", "erm"
    "keep_audio": True,
    "copy_on_stop": False,
    "noise_suppression": True,
    "auto_gain": True,
    "echo_cancellation": False,
    "theme": "system",
}
