"""Stage 2: engagement propensity scorer.

Trained locally on message_history.csv + message_events.csv (weak
supervision: message_opened is the label — dismissed/muted/reported rows
have message_opened=0). Not trained on eval labels, not trained on
messages.csv. Runs once, in-memory, before the main batch pass; inference
per message during Stage 5 prompt assembly.

message_type is deliberately excluded as a feature: it isn't a real column
in message_history.csv, only in the small sample_messages.csv reference set.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import data_loader

CATEGORICAL_FEATURES = [
    "conversation_type",
    "group_type",
    "user_role_in_group",
    "business_category",
    "user_business_relationship",
]
NUMERIC_FEATURES = [
    "forwarded_count",
    "hour_of_day",
    "is_weekend",
    "has_media",
    "text_length",
    "sender_is_group_admin",
    "group_muted_by_user",
    "business_verified",
    "domain_mismatch",
    "user_opted_out_of_promotions",
    "messages_opened_30d",
    "messages_replied_30d",
    "notifications_dismissed_30d",
    "messages_reported_30d",
]
ALL_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def context_to_features(ctx: dict) -> dict:
    """Flatten a build_context() dict into a flat feature row.

    Missing group/business context (message not that conversation_type)
    fills in as None/0 defaults so every row has the same feature set.
    """
    created_at = pd.to_datetime(ctx["created_at"])
    group = ctx.get("group", {})
    business = ctx.get("business", {})
    profile = ctx.get("user_profile", {})

    return {
        "conversation_type": ctx["conversation_type"],
        "group_type": group.get("group_type"),
        "user_role_in_group": group.get("user_role_in_group"),
        "business_category": business.get("business_category"),
        "user_business_relationship": business.get("user_business_relationship"),
        "forwarded_count": ctx["forwarded_count"],
        "hour_of_day": created_at.hour,
        "is_weekend": int(created_at.dayofweek >= 5),
        "has_media": int(ctx["media_type"] is not None),
        "text_length": len(ctx["message_text"]),
        "sender_is_group_admin": int(ctx.get("sender_is_group_admin", False)),
        "group_muted_by_user": int(group.get("group_muted_by_user", False)),
        "business_verified": int(business.get("business_verified", False)),
        "domain_mismatch": int(business.get("domain_mismatch", False)),
        "user_opted_out_of_promotions": int(business.get("user_opted_out_of_promotions", False)),
        "messages_opened_30d": profile.get("messages_opened_30d", 0),
        "messages_replied_30d": profile.get("messages_replied_30d", 0),
        "notifications_dismissed_30d": profile.get("notifications_dismissed_30d", 0),
        "messages_reported_30d": profile.get("messages_reported_30d", 0),
    }


def _build_training_frame(data: dict) -> pd.DataFrame:
    history = data["message_history"]
    events = data["message_events"]

    rows = []
    events_idx = events.set_index(["user_id", "message_id"])
    for _, msg_row in history.iterrows():
        key = (msg_row["user_id"], msg_row["message_id"])
        if key not in events_idx.index:
            continue
        ctx = data_loader.build_context(data, msg_row)
        features = context_to_features(ctx)
        features["label"] = int(events_idx.loc[key, "message_opened"])
        rows.append(features)

    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]
    )


def train(data: dict) -> tuple:
    """Train the propensity scorer. Returns (fitted_pipeline, holdout_auc)."""
    frame = _build_training_frame(data)
    X = frame[ALL_FEATURES]
    y = frame["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    holdout_auc = None
    if y_test.nunique() > 1:
        proba = pipeline.predict_proba(X_test)[:, 1]
        holdout_auc = roc_auc_score(y_test, proba)

    # Refit on all data for the final model used in inference.
    pipeline.fit(X, y)
    return pipeline, holdout_auc


def predict_proba(pipeline: Pipeline, ctx: dict) -> float:
    """Predict engagement probability for one message context."""
    features = context_to_features(ctx)
    row = pd.DataFrame([{k: features[k] for k in ALL_FEATURES}])
    return float(pipeline.predict_proba(row)[0, 1])


if __name__ == "__main__":
    data = data_loader.load_all()
    pipeline, auc = train(data)
    print(f"trained on {len(_build_training_frame(data))} history rows, holdout AUC={auc}")

    contexts = data_loader.build_all_contexts(data)
    for ctx in contexts[:5]:
        p = predict_proba(pipeline, ctx)
        print(ctx["message_id"], "engagement_proba=", round(p, 3))
