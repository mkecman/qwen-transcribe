"""Live dictation over a WebSocket.

The browser streams 16 kHz int16 PCM. Silero VAD cuts it into utterances at pauses.
Consecutive utterances are collected into a *window* (up to ~40-55 s of speech). Each time
an utterance ends, the whole window is fed to the model again with the window's previous
transcript (minus its last few tokens) forced as the start of the answer: the model sees
all the audio for context, can fix the end of the previous phrase (so a sentence spanning
a thinking pause comes out as one sentence), and only decodes what's new. When a window
fills up (or a paragraph starts) it is closed, and once its last transcription lands its
text is locked. The next window's first decode also hears the previous window (its text
forced in minus the final punctuation), so the seam between windows is punctuated with
the speech on both sides of it, not as if the speech had stopped there.

While you speak, a preview of the window plus the utterance in progress is sent too;
the part that may still change is shown greyed out in the UI.
"""
import asyncio
import json
import logging
import shutil
import time

import numpy as np
import soundfile as sf
from fastapi import WebSocket, WebSocketDisconnect

from . import config, db, textutil
from .engine import PRIORITY_FINAL, PRIORITY_PARTIAL, SAMPLE_RATE, engine
from .serialize import full
from .segmenter import Segment, Segmenter, SpeechStart

log = logging.getLogger("live")

WINDOW_SOFT_S = 40.0    # after this much speech, the next pause of >= 1 s starts a new window
WINDOW_MAX_S = 55.0     # ...and at this length any pause does
PREVIEW_EVERY_S = 0.8   # preview cadence while speaking


class Window:
    def __init__(self, paragraph: bool):
        self.paragraph = paragraph          # starts a new paragraph
        self.segments: list[np.ndarray] = []
        self.text = ""                      # transcription of all finished utterances
        self.raw = ""                       # same, as the model said it (continuation prefix)
        self.text_n = 0                     # utterances covered by `text`
        self.text_s = 0.0                   # seconds of audio covered by `text`
        self.preview: str | None = None     # includes the utterance in progress
        self.preview_s = 0.0
        self.closed = False                 # no more utterances will be added
        self.ctx: "Window | None" = None    # previous window, heard by the first decode only

    @property
    def audio_s(self) -> float:
        return sum(s.shape[0] for s in self.segments) / SAMPLE_RATE

    @property
    def done(self) -> bool:
        return self.closed and self.text_n == len(self.segments)

    def shown(self) -> str:
        if self.preview is not None and self.preview_s > self.text_s:
            return self.preview
        return self.text


class LiveSession:
    def __init__(self, ws: WebSocket):
        self.ws = ws
        self.settings = db.get_settings()
        self.seg = Segmenter(pause_ms=int(self.settings["pause_ms"]))
        self.context = (self.settings["context"] or "").strip()
        self.fillers = bool(self.settings["remove_fillers"])
        self.paragraph_after = float(self.settings["paragraph_after_s"] or 0)
        self.language = None
        self.session_id = None
        self.created = False

        self.base = ""                      # session text before this take (may hold user edits)
        self.locked = ""                    # text of this take that can no longer change
        self.windows: list[Window] = []
        self.last_window: Window | None = None
        self.paragraph_next = False
        self.last_end_s = None
        self.was_speaking = False
        self.since_preview = 0
        self.preview_task: asyncio.Task | None = None
        self.finals: asyncio.Queue = asyncio.Queue()

        self.recorder = None
        self.recording_id = None
        self.connected = True
        self._send_lock = asyncio.Lock()

    # ---- protocol --------------------------------------------------------
    async def run(self):
        await self.ws.accept()
        try:
            start = await self.ws.receive_json()
        except WebSocketDisconnect:
            return
        if engine.status != "ready":
            await self.send({"type": "error", "message": "The model is still loading."})
            await self.ws.close()
            return
        self.begin(start)
        await self.send({"type": "started", "session": full(db.get_session(self.session_id))})
        worker = asyncio.create_task(self.final_worker())
        try:
            await self.receive_loop()
        except WebSocketDisconnect:
            self.connected = False
        except Exception:
            log.exception("Live session failed")
        finally:
            await self.finish(worker)

    async def receive_loop(self):
        while True:
            msg = await self.ws.receive()
            if msg["type"] == "websocket.disconnect":
                self.connected = False
                return
            if msg.get("bytes") is not None:
                self.on_audio(msg["bytes"])
            elif msg.get("text"):
                data = json.loads(msg["text"])
                if data.get("type") == "stop":
                    return
                if data.get("type") == "paragraph":
                    self.paragraph_next = True

    async def send(self, payload: dict):
        if not self.connected:
            return
        async with self._send_lock:
            try:
                await self.ws.send_json(payload)
            except Exception:
                self.connected = False

    def send_soon(self, payload: dict):
        asyncio.create_task(self.send(payload))

    # ---- lifecycle -------------------------------------------------------
    def begin(self, start: dict):
        lang = start.get("language") or self.settings["language"]
        self.language = None if lang == "auto" else lang
        session = db.get_session(start["session_id"]) if start.get("session_id") else None
        if session is None:
            session = db.create_session("live", lang, status="recording")
            self.created = True
        else:
            db.update_session(session["id"], status="recording")
        self.session_id = session["id"]
        self.base = session["text"] or ""  # continuing a session starts a new paragraph after it

        if self.settings["keep_audio"]:
            folder = config.AUDIO_DIR / self.session_id
            folder.mkdir(parents=True, exist_ok=True)
            path = folder / f"live-{time.strftime('%Y%m%d-%H%M%S')}.flac"
            self.recorder = sf.SoundFile(path, mode="w", samplerate=SAMPLE_RATE, channels=1,
                                         format="FLAC", subtype="PCM_16")
            self.recording_id = db.add_recording(self.session_id, str(path.relative_to(config.DATA_DIR)), 0)

    async def finish(self, worker: asyncio.Task):
        try:
            seg = self.seg.flush()
            if seg is not None:
                self.on_segment(seg)
            for w in self.windows:
                w.closed = True
            if self.windows:
                self.finals.put_nowait(self.windows[-1])
            self.finals.put_nowait(None)
            await worker
        except Exception:
            log.exception("Failed to finalize live session")
        finally:
            for w in self.windows:  # whatever made it, even after an error
                self.locked = textutil.join(self.locked, w.text, w.paragraph)
            self.windows.clear()
            text = self.with_base(self.polish(self.locked))
            recorded = self.seg.position_s
            if self.recorder is not None:
                self.recorder.close()
                db.update_recording(self.recording_id, recorded)
            if self.created and not text.strip():
                # nothing was said: don't leave an empty entry in the history
                db.delete_session(self.session_id)
                shutil.rmtree(config.AUDIO_DIR / self.session_id, ignore_errors=True)
                await self.send({"type": "stopped", "session": None})
            else:
                session = db.get_session(self.session_id)
                db.update_session(self.session_id, text=text, status="done",
                                  duration=(session["duration"] or 0) + recorded)
                await self.send({"type": "stopped", "session": full(db.get_session(self.session_id))})
            if self.connected:
                try:
                    await self.ws.close()
                except Exception:
                    pass

    # ---- audio -----------------------------------------------------------
    def current(self) -> Window | None:
        if self.windows and not self.windows[-1].closed:
            return self.windows[-1]
        return None

    def on_audio(self, data: bytes):
        pcm = np.frombuffer(data, dtype="<i2")
        if self.recorder is not None:
            self.recorder.write(pcm)
        x = pcm.astype(np.float32) / 32768.0
        ended = False
        for ev in self.seg.feed(x):
            if isinstance(ev, SpeechStart):
                self.on_speech_start(ev)
            else:
                self.on_segment(ev)
                ended = True

        speaking = self.seg.in_speech
        if self.was_speaking and not speaking and not ended:
            # utterance discarded as noise: drop its preview
            cur = self.current()
            if cur is not None and cur.preview is not None:
                cur.preview, cur.preview_s = None, 0.0
                self.send_soon(self.text_message())
        if speaking != self.was_speaking:
            self.was_speaking = speaking
            self.send_soon({"type": "vad", "speech": speaking})

        if speaking:
            self.since_preview += x.shape[0]
            idle = self.preview_task is None or self.preview_task.done()
            if self.since_preview >= PREVIEW_EVERY_S * SAMPLE_RATE and idle:
                self.since_preview = 0
                self.preview_task = asyncio.create_task(self.run_preview())

    def on_speech_start(self, ev: SpeechStart):
        gap = ev.start_s - self.last_end_s if self.last_end_s is not None else 0.0
        paragraph = self.paragraph_next or (
            self.paragraph_after > 0 and self.last_end_s is not None and gap >= self.paragraph_after)
        cur = self.current()
        if cur is not None and not cur.segments:
            cur.paragraph = cur.paragraph or paragraph  # reuse an empty window
        elif (cur is None or paragraph or cur.audio_s >= WINDOW_MAX_S
              or (cur.audio_s >= WINDOW_SOFT_S and gap >= 1.0)):
            if cur is not None:
                cur.closed = True
                self.finals.put_nowait(cur)
            self.new_window(paragraph)
        self.paragraph_next = False
        self.since_preview = 0

    def new_window(self, paragraph: bool) -> Window:
        w = Window(paragraph)
        prev = self.last_window
        if not paragraph and prev is not None and prev.segments:
            w.ctx = prev
        self.windows.append(w)
        self.last_window = w
        return w

    def on_segment(self, seg: Segment):
        self.last_end_s = seg.end_s
        cur = self.current()
        if cur is None:
            cur = self.new_window(self.paragraph_next)
            self.paragraph_next = False
        cur.segments.append(seg.audio)
        self.finals.put_nowait(cur)

    # ---- transcription ---------------------------------------------------
    async def final_worker(self):
        while True:
            w = await self.finals.get()
            if w is None:
                return
            n = len(w.segments)
            if n and w.text_n != n:
                audio = np.concatenate(w.segments[:n])
                audio_s = audio.shape[0] / SAMPLE_RATE
                try:
                    if not (w.text_n == 0 and await self.first_final_across_seam(w, audio)):
                        _, text, raw = await engine.transcribe(
                            audio, self.language, self.context, PRIORITY_FINAL,
                            continue_from=w.raw, new_s=audio_s - w.text_s)
                        w.text, w.raw = textutil.clean(text, self.fillers), raw
                except Exception:
                    log.exception("Transcription failed")
                    continue
                w.text_n, w.text_s = n, audio_s
            while self.windows and self.windows[0].done:
                done = self.windows.pop(0)
                self.locked = textutil.join(self.locked, done.text, done.paragraph)
            db.update_session(self.session_id, text=self.persisted_text())
            await self.send(self.text_message())

    async def first_final_across_seam(self, w: Window, audio: np.ndarray) -> bool:
        """Decode a window's first audio together with the previous window, forcing that
        window's text minus its final punctuation as the start of the answer. The model then
        hears the speech go on and decides how the previous window really ends (period,
        comma, nothing), instead of ending it as if the speech stopped there."""
        ctx, w.ctx = w.ctx, None
        if ctx is None or not self.language or ctx.text_n != len(ctx.segments):
            return False
        stem, _ = textutil.split_tail(ctx.raw)
        if not stem:
            return False
        _, _, raw = await engine.transcribe(
            np.concatenate(ctx.segments + [audio]), self.language, self.context, PRIORITY_FINAL,
            continue_from=stem, new_s=audio.shape[0] / SAMPLE_RATE, rollback=0)
        if not raw.startswith(stem):
            return False
        lead, body = textutil.split_lead(raw[len(stem):])
        if ctx in self.windows:
            ctx.raw = stem + lead
            ctx.text = textutil.clean(ctx.raw, self.fillers)
        else:  # already locked: it is the end of `locked`
            self.locked = textutil.split_tail(self.locked)[0] + lead
        w.raw, w.text = body, textutil.clean(body, self.fillers)
        return True

    async def run_preview(self):
        cur = self.current()
        live = self.seg.current_audio()
        if cur is None or live is None:
            return
        audio = np.concatenate(cur.segments + [live])
        covered = audio.shape[0] / SAMPLE_RATE
        try:
            _, text, _ = await engine.transcribe(
                audio, self.language, self.context, PRIORITY_PARTIAL,
                continue_from=cur.raw, new_s=covered - cur.text_s)
        except Exception:
            log.exception("Preview failed")
            return
        if self.current() is cur and covered > cur.preview_s:
            cur.preview, cur.preview_s = textutil.clean(text, self.fillers), covered
            await self.send(self.text_message())

    # ---- text ------------------------------------------------------------
    def polish(self, text: str) -> str:
        """Whole-text cleanup of this take: stutters and quotes that span windows."""
        return textutil.repair_quotes(textutil.clean(text, self.fillers))

    def with_base(self, text: str) -> str:
        return textutil.join(self.base, text, paragraph=True) if text else self.base

    def persisted_text(self) -> str:
        text = self.locked
        for w in self.windows:
            text = textutil.join(text, w.text, w.paragraph)
        return self.with_base(self.polish(text))

    def text_message(self) -> dict:
        """Everything to display: `stable` text, then a greyed-out `pending` tail."""
        text = self.locked
        for w in self.windows[:-1]:
            text = textutil.join(text, w.shown(), w.paragraph)
        stable = full = text
        if self.windows:
            last = self.windows[-1]
            shown = last.shown()
            full = textutil.join(text, shown, last.paragraph)
            k = textutil.stable_prefix(last.text, shown)
            stable = textutil.join(text, shown[:k], last.paragraph) if k else text
        if self.base:
            stable, full = (self.with_base(stable) if stable else self.base), self.with_base(full)
        return {"type": "text", "stable": stable, "pending": full[len(stable):]}


async def handle(ws: WebSocket):
    await LiveSession(ws).run()
