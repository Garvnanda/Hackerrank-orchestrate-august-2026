"""Stage 3: RAG-style evidence retrieval + near-duplicate detection.

Pretrained local sentence-embedding model, CPU inference, no training.
Embeds message_history.csv text and cosine-similarity searches it for each
incoming message. Scoped primarily toward the same sender/group/business but
not limited to it, so cross-sender pattern matches (e.g. the same scam
template from a different number) still surface.

Also handles voice-note transcripts and OCR'd image text the same way once
Stage 4 fills those in on the context dict (media_transcript key).
"""
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

import config
import media

_model = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _model


def _history_text(row: pd.Series, voice_notes: pd.DataFrame) -> str:
    text = row["message_text"] if pd.notna(row["message_text"]) else ""
    text = text.strip()
    if text:
        return text
    if row.get("media_type") == "voice" and pd.notna(row.get("media_id")):
        match = voice_notes[voice_notes["voice_note_id"] == row["media_id"]]
        if not match.empty:
            path = config.DATASET_DIR / match.iloc[0]["file_path"]
            return media.transcribe_voice_note(path)
    return ""


class EvidenceIndex:
    """Embeds message_history.csv once; answers top-k similarity queries."""

    def __init__(self, data: dict):
        history = data["message_history"].copy()
        # Historical voice notes ship with no transcript in message_history.csv
        # — without this, every incoming voice message would find zero
        # evidence candidates (nothing to match against) and get its
        # confidence permanently floored by the no-evidence cap regardless
        # of actual certainty. Transcribe them once here, same Whisper call
        # Stage 4 uses for incoming messages.
        history["_text"] = history.apply(lambda r: _history_text(r, data["voice_notes"]), axis=1)
        self.history = history[history["_text"].str.len() > 0].reset_index(drop=True)

        model = get_model()
        self.embeddings = model.encode(
            self.history["_text"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        self.events_idx = data["message_events"].set_index(["user_id", "message_id"])

    def query(self, ctx: dict, top_k: int = None) -> list:
        """Return up to top_k evidence candidates for one message context.

        Each candidate: message_id, similarity, sender/group/business match
        flags, and known message_events outcome for that history message's
        original recipient (weak signal for "was this kind of message
        engaged with or dismissed/muted/reported before").
        """
        top_k = top_k or config.RETRIEVAL_TOP_K
        # message_text is always empty for voice messages by schema design —
        # fall back to the Stage 4 transcript so voice queries aren't silently
        # starved of evidence candidates.
        query_text = (ctx.get("message_text") or ctx.get("media_transcript") or "").strip()
        if not query_text or len(self.history) == 0:
            return []

        model = get_model()
        query_vec = model.encode([query_text], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.embeddings @ query_vec

        # Boost same-sender/group/business matches slightly so identical
        # templates from the same source outrank loosely-related text from
        # elsewhere, without hard-filtering out cross-sender matches.
        boost = np.zeros(len(self.history))
        sender_id = ctx.get("sender_user_id")
        group_id = ctx.get("group_id")
        business_id = ctx.get("business_id")
        if sender_id is not None:
            boost += 0.03 * (self.history["sender_user_id"] == sender_id).to_numpy()
        if group_id is not None:
            boost += 0.02 * (self.history["group_id"] == group_id).to_numpy()
        if business_id is not None:
            boost += 0.02 * (self.history["business_id"] == business_id).to_numpy()

        scores = sims + boost
        top_idx = np.argsort(-scores)[:top_k]

        results = []
        for i in top_idx:
            row = self.history.iloc[i]
            key = (row["user_id"], row["message_id"])
            outcome = None
            if key in self.events_idx.index:
                ev = self.events_idx.loc[key]
                outcome = {
                    "message_opened": bool(ev["message_opened"]),
                    "message_replied": bool(ev["message_replied"]),
                    "notification_dismissed": bool(ev["notification_dismissed"]),
                    "muted_after_message": bool(ev["muted_after_message"]),
                    "message_reported": bool(ev["message_reported"]),
                }
            results.append(
                {
                    "message_id": row["message_id"],
                    "similarity": float(sims[i]),
                    "text": row["_text"],
                    "same_sender": bool(sender_id is not None and row["sender_user_id"] == sender_id),
                    "same_group": bool(group_id is not None and row["group_id"] == group_id),
                    "same_business": bool(business_id is not None and row["business_id"] == business_id),
                    "outcome": outcome,
                }
            )
        return results

    def is_near_duplicate(self, ctx: dict, threshold: float = 0.93) -> bool:
        """True if the top match is near-identical text (repetition signal)."""
        candidates = self.query(ctx, top_k=1)
        return bool(candidates) and candidates[0]["similarity"] >= threshold


if __name__ == "__main__":
    import data_loader

    data = data_loader.load_all()
    index = EvidenceIndex(data)
    contexts = data_loader.build_all_contexts(data)

    for ctx in contexts:
        if ctx["message_id"] in ("msg_029", "msg_030", "msg_040", "msg_090"):
            candidates = index.query(ctx, top_k=3)
            print(ctx["message_id"], "->", [(c["message_id"], round(c["similarity"], 3)) for c in candidates])
            print("  near_dup:", index.is_near_duplicate(ctx))
