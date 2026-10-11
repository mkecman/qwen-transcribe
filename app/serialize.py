from . import db, textutil


def summary(s: dict) -> dict:
    text = s["text"] or ""
    return {
        "id": s["id"], "kind": s["kind"], "status": s["status"], "progress": s["progress"],
        "title": (s["title"] or textutil.auto_title(text) or s["source_name"]
                  or ("New recording" if s["kind"] == "live" else "Untitled")),
        "custom_title": bool(s["title"]), "snippet": " ".join(text.split())[:180],
        "words": len(text.split()), "duration": s["duration"], "language": s["language"],
        "source_name": s["source_name"], "created_at": s["created_at"], "updated_at": s["updated_at"],
    }


def full(s: dict) -> dict:
    return {**summary(s), "text": s["text"] or "", "error": s["error"],
            "has_audio": bool(db.get_recordings(s["id"]))}
