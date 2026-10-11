# Qwen Transcribe

Local speech-to-text with [Qwen3-ASR-1.7B](https://huggingface.co/Qwen/Qwen3-ASR-1.7B): live dictation from any microphone, plus audio/video file transcription, with a searchable history. Everything runs on this machine; nothing is sent anywhere.

## Setup

You need Windows 10/11, an NVIDIA GPU with ~6 GB of free VRAM and a recent driver (tested on an RTX 5090), [Miniconda](https://docs.anaconda.com/miniconda/), and ~12 GB of disk. Run these in an Anaconda Prompt:

```bat
git clone https://github.com/mkecman/qwen-transcribe.git
cd qwen-transcribe

:: 1. Python 3.12 + ffmpeg, in a conda env inside the project folder
conda create -p .\env -c conda-forge --override-channels python=3.12 ffmpeg

:: 2. Python packages (PyTorch with CUDA 12.8, Qwen3-ASR, Silero VAD, FastAPI)
env\python.exe -m pip install -r requirements.txt

:: 3. The model (~4.4 GB) into models\
env\Scripts\hf.exe download Qwen/Qwen3-ASR-1.7B --local-dir models\Qwen3-ASR-1.7B
```

Then run `start.bat`.

## Start

Double-click **`start.bat`**. It loads the model (~10 s) and opens http://127.0.0.1:8765 in your browser. Close the console window to stop it.

## Use

- **Dictate**: pick a microphone in the bottom bar, then click **Record** (or press `Ctrl+Space`). Text appears as you speak; the grey tail is still being refined. Press `Enter` (or the ¶ button) to start a new paragraph. Stop with the same button or `Ctrl+Space`.
- **Continue** a session: open it from the history and press **Continue**. New text is appended as a new paragraph.
- **Files**: drag audio or video files anywhere onto the page, or use the upload button. Anything ffmpeg reads works (mp3, m4a, wav, ogg, flac, mp4, …). A 6-minute recording takes about 6 seconds.
- **Edit and copy**: transcripts are editable (autosaved). **Copy** puts the whole text on the clipboard. The **⋯** menu has Download .txt, Re-transcribe (from the saved audio), and Delete.
- **Settings** (sliders icon, bottom left): default language, a vocabulary list for names and jargon, filler-word cleanup, how long a pause finalizes text, auto-paragraphs, mic processing, whether to keep audio, and the theme.

## How it works

- `app/engine.py`: loads the model once and runs all GPU work on one worker thread, with priorities (live text first, file jobs last). Decoding replays a captured **CUDA graph** per token instead of launching ~2,000 small kernels. On Windows this made generation ~4–8× faster, at ~5–6 ms per token, with output identical to the reference implementation.
- `app/segmenter.py`: Silero VAD splits audio at pauses. File uploads use a batched pass that is 6× faster.
- `app/live.py`: live dictation over a WebSocket. Utterances are grouped into ~40–55 s windows. At each pause the window is re-read, with its earlier transcript forced as the start of the answer, so a sentence spanning a thinking pause comes out whole and only the new part is decoded.
- `app/files.py`: file jobs. Speech is grouped into 40–60 s chunks, cut at the longest pause.
- `app/textutil.py`: filler and stutter cleanup, and sentence repair where separately transcribed pieces meet.
- `web/`: the UI, plain HTML/CSS/JS with no build step. `worklet.js` captures 16 kHz PCM from the mic.

## Folders

| Folder | Contents |
|---|---|
| `env/` | conda environment (Python 3.12, PyTorch 2.9.1 CUDA 12.8, ffmpeg) |
| `models/` | model weights (4.4 GB) |
| `data/` | `transcribe.db` (SQLite: transcripts and settings) and `audio/` (saved recordings) |
| `.cache/` | pip, conda and Hugging Face download caches. Safe to delete |

Deleting the project folder removes everything. Nothing is installed outside it.

## Notes

- **Serbian is not supported** by Qwen3-ASR. The language list in the app is what the model supports (30 languages, plus auto-detect).
- The vocabulary setting is passed to the model as context. Keep it to a short list of terms: long text there can leak into the transcript.
- The server listens on `127.0.0.1` only, and rejects writes coming from other websites.
- Environment variables: `QT_PORT` (default 8765), `QT_DATA_DIR`, `QT_MODEL_PATH`.
- Linux should work too (`env/bin/python -m app`; it falls back to the system `ffmpeg`), but it's untested.

## Credits

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) by the Alibaba Qwen team (Apache-2.0): the model and the `qwen-asr` package. The prefix-rollback continuation follows its streaming implementation.
- [Silero VAD](https://github.com/snakers4/silero-vad) (MIT): speech detection.
- Icons adapted from [Lucide](https://lucide.dev) (ISC).
