"""Stage 8: batch runner. Runs every stage over all of messages.csv -> output.csv.

Per-row error isolation: a failed row falls back to the Stage 6 safe default
(digest/unknown, low confidence) rather than crashing the whole run.
"""
import sys
import time
import traceback

import pandas as pd

import config
import data_loader
import guardrails
import llm_router
import media
import propensity
import retrieval

OUTPUT_COLUMNS = ["message_id", "action", "message_type", "reason", "confidence", "evidence_message_ids"]

SAFE_FALLBACK_ROW = {
    "action": "digest",
    "message_type": "unknown",
    "reason": "Row-level processing error — fell back to a safe default rather than failing the batch.",
    "confidence": 0.3,
    "evidence_message_ids": [],
}


def _format_evidence(ids: list) -> str:
    return ";".join(ids) if ids else "none"


def process_message(ctx: dict, data: dict, pipeline, index: retrieval.EvidenceIndex, few_shot: str) -> dict:
    media.resolve_media(data, ctx)
    prop_score = propensity.predict_proba(pipeline, ctx)
    evidence_candidates = index.query(ctx)

    decision = llm_router.get_decision(ctx, prop_score, evidence_candidates, few_shot)

    if llm_router.needs_escalation(ctx, decision):
        escalated = llm_router.get_decision(ctx, prop_score, evidence_candidates, few_shot, escalate=True)
        if escalated["confidence"] >= decision["confidence"]:
            decision = escalated

    return guardrails.apply_guardrails(ctx, evidence_candidates, decision)


def run(limit: int = None) -> pd.DataFrame:
    print("Loading data...", file=sys.stderr)
    data = data_loader.load_all()
    data["sample_messages"] = pd.read_csv(config.DATASET_DIR / "sample_messages.csv")
    few_shot = llm_router._load_few_shot_examples(data)

    print("Training propensity scorer...", file=sys.stderr)
    pipeline, auc = propensity.train(data)
    print(f"  holdout AUC={auc}", file=sys.stderr)

    print("Building evidence index...", file=sys.stderr)
    index = retrieval.EvidenceIndex(data)

    contexts = data_loader.build_all_contexts(data)
    if limit:
        contexts = contexts[:limit]

    rows = []
    total = len(contexts)
    for i, ctx in enumerate(contexts, start=1):
        message_id = ctx["message_id"]
        try:
            result = process_message(ctx, data, pipeline, index, few_shot)
        except Exception:
            print(f"[{i}/{total}] {message_id} FAILED, using safe fallback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            result = {"message_id": message_id, **SAFE_FALLBACK_ROW}

        rows.append(
            {
                "message_id": result["message_id"],
                "action": result["action"],
                "message_type": result["message_type"],
                "reason": result["reason"],
                "confidence": result["confidence"],
                "evidence_message_ids": _format_evidence(result["evidence_message_ids"]),
            }
        )
        print(f"[{i}/{total}] {message_id} -> {result['action']}/{result['message_type']} (conf={result['confidence']})", file=sys.stderr)

    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    df = run(limit=limit)
    df.to_csv(config.OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} rows to {config.OUTPUT_CSV}", file=sys.stderr)


if __name__ == "__main__":
    main()
