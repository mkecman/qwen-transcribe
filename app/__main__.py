"""Run with:  env\\python.exe -m app   [--no-browser]"""
import logging
import os
import sys
import threading
import urllib.request
import webbrowser

# Everything is local: never reach out to the Hugging Face Hub at runtime.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import uvicorn  # noqa: E402

from . import config  # noqa: E402


def already_running(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/status", timeout=1) as r:
            return r.status == 200
    except Exception:
        return False


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)-7s %(message)s", datefmt="%H:%M:%S")
    url = f"http://127.0.0.1:{config.PORT}" if config.HOST in ("127.0.0.1", "0.0.0.0") else f"http://{config.HOST}:{config.PORT}"
    if already_running(url):
        # launched again (e.g. from the shortcut): just bring up the UI
        if "--no-browser" not in sys.argv:
            webbrowser.open(url)
        return
    print(f"\n  Qwen Transcribe  ->  {url}\n  (close this window to stop)\n", flush=True)
    if "--no-browser" not in sys.argv:
        threading.Timer(1.5, webbrowser.open, args=(url,)).start()
    uvicorn.run("app.main:app", host=config.HOST, port=config.PORT, log_level="warning", ws_max_size=1 << 20)


if __name__ == "__main__":
    main()
