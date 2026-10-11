import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import config, db, live
from .engine import engine
from .files import delete_audio, jobs
from .serialize import full, summary


@asynccontextmanager
async def lifespan(app: FastAPI):
    torch.set_num_threads(1)  # Silero VAD is fastest single-threaded; the model runs on the GPU
    db.init()
    engine.start()
    jobs.start()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None)

ALLOWED_ORIGINS = {f"http://{h}:{config.PORT}" for h in ("127.0.0.1", "localhost", config.HOST)}


@app.middleware("http")
async def same_origin_only(request: Request, call_next):
    # Other websites open in the browser must not be able to change or delete transcripts.
    origin = request.headers.get("origin")
    if request.method not in ("GET", "HEAD") and origin and origin not in ALLOWED_ORIGINS:
        return JSONResponse({"detail": "Forbidden origin"}, status_code=403)
    return await call_next(request)


app.mount("/static", StaticFiles(directory=config.WEB_DIR), name="static")


@app.get("/")
async def index():
    return FileResponse(config.WEB_DIR / "index.html", headers={"Cache-Control": "no-cache"})


def get_or_404(sid: str) -> dict:
    s = db.get_session(sid)
    if s is None:
        raise HTTPException(404, "Session not found")
    return s


# ---- API ---------------------------------------------------------------
@app.get("/api/status")
async def status():
    return engine.info()


@app.get("/api/sessions")
async def list_sessions(q: str = ""):
    return [summary(s) for s in db.list_sessions(q.strip())]


@app.get("/api/sessions/{sid}")
async def get_session(sid: str):
    return full(get_or_404(sid))


class SessionPatch(BaseModel):
    title: Optional[str] = None
    text: Optional[str] = None


@app.patch("/api/sessions/{sid}")
async def patch_session(sid: str, patch: SessionPatch):
    s = get_or_404(sid)
    if s["status"] in ("recording", "queued", "processing") and patch.text is not None:
        raise HTTPException(409, "This session is still being transcribed")
    fields = {}
    if patch.title is not None:
        fields["title"] = patch.title.strip() or None
    if patch.text is not None:
        fields["text"] = patch.text
    db.update_session(sid, **fields)
    return full(db.get_session(sid))


@app.delete("/api/sessions/{sid}")
async def delete_session(sid: str):
    s = get_or_404(sid)
    if s["status"] == "recording":
        raise HTTPException(409, "Stop the recording first")
    db.delete_session(sid)
    shutil.rmtree(config.AUDIO_DIR / sid, ignore_errors=True)
    return {"ok": True}


@app.post("/api/sessions/{sid}/retranscribe")
async def retranscribe(sid: str):
    s = get_or_404(sid)
    if s["status"] in ("recording", "queued", "processing"):
        raise HTTPException(409, "This session is busy")
    if not db.get_recordings(sid):
        raise HTTPException(400, "No saved audio for this session")
    jobs.enqueue(sid)
    return full(db.get_session(sid))


@app.delete("/api/sessions/{sid}/audio")
async def delete_session_audio(sid: str):
    s = get_or_404(sid)
    if s["status"] in ("recording", "queued", "processing"):
        raise HTTPException(409, "This session is busy")
    delete_audio(sid)
    return full(db.get_session(sid))


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), language: str = Form("English")):
    name = Path(file.filename or "audio").name
    s = db.create_session("file", language, status="queued", source_name=name)
    folder = config.AUDIO_DIR / s["id"]
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / ("source" + Path(name).suffix.lower())
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out, length=1 << 20)
    db.add_recording(s["id"], str(dest.relative_to(config.DATA_DIR)), 0)
    jobs.enqueue(s["id"])
    return summary(db.get_session(s["id"]))


@app.get("/api/settings")
async def get_settings():
    return db.get_settings()


@app.put("/api/settings")
async def put_settings(values: dict):
    return db.save_settings(values)


@app.websocket("/ws/live")
async def ws_live(ws: WebSocket):
    if ws.headers.get("origin") not in ALLOWED_ORIGINS:
        await ws.close(code=1008)
        return
    await live.handle(ws)
