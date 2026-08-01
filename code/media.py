"""Stage 4: media understanding.

Voice notes: faster-whisper (CPU), transcription only, no training. Output
is plain transcript text, treated like any other text field downstream.

Images/posters: not processed here in a separate call — base64-encoded and
handed to the Stage 5 vision-capable OpenRouter call directly, to save
budget/requests. This module only resolves file paths and does the base64
encoding.
"""
import base64
from pathlib import Path

import pandas as pd

import config

_whisper_model = None


def get_whisper_model():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        _whisper_model = WhisperModel(config.WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    return _whisper_model


def transcribe_voice_note(file_path: Path) -> str:
    """Return plain transcript text for one audio file. Empty string on failure."""
    if not file_path.exists():
        return ""
    model = get_whisper_model()
    try:
        segments, _info = model.transcribe(str(file_path), beam_size=config.WHISPER_BEAM_SIZE)
        return " ".join(seg.text.strip() for seg in segments).strip()
    except Exception:
        return ""


def encode_image_base64(file_path: Path) -> str:
    """Return a base64 data URI for one image file. Empty string on failure."""
    if not file_path.exists():
        return ""
    ext = file_path.suffix.lstrip(".").lower() or "jpeg"
    mime = "jpeg" if ext == "jpg" else ext
    try:
        raw = file_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/{mime};base64,{b64}"
    except Exception:
        return ""


def resolve_media(data: dict, ctx: dict) -> dict:
    """Fill in media_transcript (voice) or media_data_uri (image) on a context.

    Mutates and returns ctx. No-op if media_type is None.
    """
    media_type = ctx.get("media_type")
    media_id = ctx.get("media_id")
    if media_type is None or media_id is None:
        return ctx

    if media_type == "voice":
        vn = data["voice_notes"]
        match = vn[vn["voice_note_id"] == media_id]
        if not match.empty:
            path = config.DATASET_DIR / match.iloc[0]["file_path"]
            ctx["media_transcript"] = transcribe_voice_note(path)
    elif media_type == "image":
        img = data["images"]
        match = img[img["image_id"] == media_id]
        if not match.empty:
            path = config.DATASET_DIR / match.iloc[0]["file_path"]
            ctx["media_data_uri"] = encode_image_base64(path)

    return ctx


if __name__ == "__main__":
    import data_loader

    data = data_loader.load_all()
    contexts = data_loader.build_all_contexts(data)

    voice_ctx = next(c for c in contexts if c.get("media_type") == "voice")
    resolve_media(data, voice_ctx)
    print(voice_ctx["message_id"], "transcript:", voice_ctx.get("media_transcript"))

    image_ctx = next(c for c in contexts if c.get("media_type") == "image")
    resolve_media(data, image_ctx)
    uri = image_ctx.get("media_data_uri", "")
    print(image_ctx["message_id"], "image_data_uri length:", len(uri), "prefix:", uri[:30])
