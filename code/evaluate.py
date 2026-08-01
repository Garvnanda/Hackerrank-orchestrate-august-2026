"""Stage 9: evaluation workflow.

sample_messages.csv is the only labeled data available (a small format/style
reference set per the problem statement, not a real training or test set).
This script runs the full pipeline on those rows — ignoring their given
labels as input, exactly like a real messages.csv row — and diffs the
predictions against the given labels. This is a sanity check on pipeline
behavior, not a real accuracy number: 12-ish labeled rows is far too small
to generalize from, and the eval set for real grading is hidden.
"""
import sys

import pandas as pd

import config
import data_loader
import propensity
import retrieval
import run_pipeline


def load_sample_contexts(data: dict) -> list:
    samples = pd.read_csv(config.DATASET_DIR / "sample_messages.csv")
    label_cols = ["action", "message_type", "reason", "confidence", "evidence_message_ids"]
    labels = samples.set_index("message_id")[label_cols].to_dict(orient="index")

    input_cols = [c for c in samples.columns if c not in label_cols]
    contexts = [data_loader.build_context(data, row) for _, row in samples[input_cols].iterrows()]
    return contexts, labels


def run_eval():
    print("Loading data...", file=sys.stderr)
    data = data_loader.load_all()
    data["sample_messages"] = pd.read_csv(config.DATASET_DIR / "sample_messages.csv")
    from llm_router import _load_few_shot_examples

    few_shot = _load_few_shot_examples(data)

    print("Training propensity scorer...", file=sys.stderr)
    pipeline, auc = propensity.train(data)

    print("Building evidence index...", file=sys.stderr)
    index = retrieval.EvidenceIndex(data)

    contexts, labels = load_sample_contexts(data)

    action_correct = 0
    type_correct = 0
    rows = []
    for ctx in contexts:
        mid = ctx["message_id"]
        expected = labels[mid]
        predicted = run_pipeline.process_message(ctx, data, pipeline, index, few_shot)

        action_match = predicted["action"] == expected["action"]
        type_match = predicted["message_type"] == expected["message_type"]
        action_correct += int(action_match)
        type_correct += int(type_match)

        rows.append(
            {
                "message_id": mid,
                "expected_action": expected["action"],
                "predicted_action": predicted["action"],
                "action_match": action_match,
                "expected_type": expected["message_type"],
                "predicted_type": predicted["message_type"],
                "type_match": type_match,
                "predicted_reason": predicted["reason"],
                "predicted_confidence": predicted["confidence"],
                "predicted_evidence": ";".join(predicted["evidence_message_ids"]) or "none",
            }
        )

    n = len(contexts)
    print(f"\n--- Results over {n} labeled sample_messages.csv rows ---")
    print(f"action accuracy:       {action_correct}/{n} = {action_correct / n:.2f}")
    print(f"message_type accuracy: {type_correct}/{n} = {type_correct / n:.2f}")
    print("\n(This is a sanity check on a tiny format-reference set, not a real accuracy metric.)")

    df = pd.DataFrame(rows)
    pd.set_option("display.max_colwidth", 60)
    pd.set_option("display.width", 200)
    print("\n--- Row-level detail (spot-check reason/evidence quality manually) ---")
    print(df.to_string(index=False))

    return df


if __name__ == "__main__":
    run_eval()
