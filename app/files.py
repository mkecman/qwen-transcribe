"""Transcription of uploaded files, and re-transcription of saved recordings."""
import asyncio
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from . import config, db, textutil
from .engine import PRIORITY_FILE, SAMPLE_RATE, engine
from .segmenter import Segment, Segmenter

log = logging.getLogger("files")

CHUNK_MIN_S, CHUNK_MAX_S = 40.0, 60.0  # utterances are grouped into chunks this long, cut at the longest pause


def decode(path: Path) -> np.ndarray:
    """Any audio/video file -> 16 kHz mono float32, via ffmpeg."""
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    proc = subprocess.run(
        [config.FFMPEG, "-nostdin", "-v", "error", "-i", str(path), "-vn", "-ac", "1",
         "-ar", str(SAMPLE_RATE), "-f", "f32le", "-"],
        capture_output=True, creationflags=flags,
    )
    if proc.returncode != 0:
        msg = proc.stderr.decode(errors="replace").strip().splitlines()
        raise RuntimeError(f"Could not read this file as audio. {msg[-1] if msg else ''}".strip())
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def chunk(segments: list[Segment], paragraph_after: float) -> list[tuple[bool, np.ndarray]]:
    """Group utterances into 40-60 s chunks, cutting at the longest pause in that range
    (long pauses are the likeliest sentence ends). Returns (starts_paragraph, audio) pairs."""
    gaps = [0.0] + [b.start_s - a.end_s for a, b in zip(segments, segments[1:])]
    durs = [s.audio.shape[0] / SAMPLE_RATE for s in segments]
    para = [paragraph_after > 0 and i > 0 and gaps[i] >= paragraph_after for i in range(len(segments))]
    out, i = [], 0
    while i < len(segments):
        j, total = i + 1, durs[i]
        while j < len(segments) and not para[j] and total + durs[j] <= CHUNK_MAX_S:
            total += durs[j]
            j += 1
        if j < len(segments) and not para[j]:
            # the chunk is full: move the cut back to the longest pause after CHUNK_MIN_S
            best, acc = j, 0.0
            for k in range(i + 1, j + 1):
                acc += durs[k - 1]
                if acc >= CHUNK_MIN_S and gaps[k] > gaps[best]:
                    best = k
            j = best
        out.append((para[i], np.concatenate([s.audio for s in segments[i:j]])))
        i = j
    return out


def prepare(path: Path, paragraph_after: float) -> tuple[float, list[tuple[bool, np.ndarray]]]:
    wav = decode(path)
    segments = Segmenter(pause_ms=500).segment_file(wav)
    return wav.shape[0] / SAMPLE_RATE, chunk(segments, paragraph_after)


async def transcribe_chunks(chunks: list[tuple[bool, np.ndarray]], language: Optional[str], context: str,
                            pieces: list[list], first_paragraph: bool = False):
    """Transcribe chunks in order, appending [starts_paragraph, raw_text] to `pieces`.

    Each chunk is decoded together with the previous chunk's audio, with the previous text
    (minus its final punctuation) forced as the start of the answer. The model then hears
    the speech continue and decides the punctuation at the seam itself, instead of ending
    every chunk with a period (and closing any open quote) because its audio stopped.
    Yields after each chunk.
    """
    prev_audio = None
    for j, (paragraph, audio) in enumerate(chunks):
        paragraph = paragraph or (j == 0 and first_paragraph)
        body = None
        if prev_audio is not None and not paragraph and language:
            stem, _ = textutil.split_tail(pieces[-1][1])
            if stem:
                _, _, raw = await engine.transcribe(
                    np.concatenate([prev_audio, audio]), language, context, PRIORITY_FILE,
                    continue_from=stem, new_s=audio.shape[0] / SAMPLE_RATE, rollback=0)
                if raw.startswith(stem):
                    lead, body = textutil.split_lead(raw[len(stem):])
                    pieces[-1][1] = stem + lead
        if body is None:
            detected, body, _ = await engine.transcribe(audio, language, context, PRIORITY_FILE)
            # continuation needs a fixed language; with auto-detect, keep what was detected
            language = language or detected or None
        pieces.append([paragraph, body.strip()])
        prev_audio = audio
        yield


def render(pieces: list[list], fillers: bool) -> str:
    text = ""
    for paragraph, raw in pieces:
        text = textutil.join(text, raw, paragraph)
    # cleaned as a whole, so stutters and quotes that cross chunk seams are handled too
    return textutil.repair_quotes(textutil.clean(text, fillers))


class FileJobs:
    def __init__(self):
        self.queue: asyncio.Queue = asyncio.Queue()
        self._task = None

    def start(self):
        self._task = asyncio.create_task(self._worker())

    def enqueue(self, sid: str):
        db.update_session(sid, status="queued", progress=0, error=None)
        self.queue.put_nowait(sid)

    async def _worker(self):
        while True:
            sid = await self.queue.get()
            try:
                await self._process(sid)
            except Exception as e:
                log.exception("File job failed")
                db.update_session(sid, status="error", error=str(e), progress=None)

    async def _process(self, sid: str):
        session = db.get_session(sid)
        if session is None:
            return
        while engine.status == "loading":
            await asyncio.sleep(0.5)
        if engine.status != "ready":
            raise RuntimeError(engine.error or "Model not available")

        settings = db.get_settings()
        language = None if session["language"] == "auto" else session["language"]
        context = (settings["context"] or "").strip()
        fillers = bool(settings["remove_fillers"])
        recordings = db.get_recordings(sid)
        if not recordings:
            raise RuntimeError("No saved audio for this session.")
        db.update_session(sid, status="processing", progress=0)

        parts, total = [], 0.0
        for rec in recordings:
            duration, chunks = await asyncio.to_thread(
                prepare, config.DATA_DIR / rec["path"], float(settings["paragraph_after_s"] or 0))
            if abs(duration - rec["duration"]) > 0.01:
                db.update_recording(rec["id"], duration)
            total += duration
            parts.append(chunks)
        db.update_session(sid, duration=total)

        count = sum(len(p) for p in parts) or 1
        pieces: list[list] = []  # [starts_paragraph, raw text] per chunk
        for i, chunks in enumerate(parts):
            # separate recordings (e.g. several live takes) become separate paragraphs
            async for _ in transcribe_chunks(chunks, language, context, pieces, first_paragraph=i > 0):
                db.update_session(sid, text=render(pieces, fillers), progress=len(pieces) / count)
        db.update_session(sid, text=render(pieces, fillers), status="done", progress=None)

        if not settings["keep_audio"] and session["kind"] == "file":
            delete_audio(sid)


def delete_audio(sid: str):
    db.delete_recordings(sid)
    shutil.rmtree(config.AUDIO_DIR / sid, ignore_errors=True)


jobs = FileJobs()
