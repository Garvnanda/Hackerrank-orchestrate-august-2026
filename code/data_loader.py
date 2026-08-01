"""Stage 0-1: load all dataset CSVs and assemble per-message context.

Pure pandas, no model involved. Output of build_context() is a plain dict
per message_id, ready to feed into propensity scoring / retrieval / prompting.
"""
import pandas as pd

import config


def load_all() -> dict:
    """Load every dataset CSV into a dict of DataFrames keyed by name."""
    d = {}
    d["messages"] = pd.read_csv(config.DATASET_DIR / "messages.csv")
    d["users"] = pd.read_csv(config.DATASET_DIR / "users.csv")
    d["groups"] = pd.read_csv(config.DATASET_DIR / "groups.csv")
    d["group_members"] = pd.read_csv(config.DATASET_DIR / "group_members.csv")
    d["business_accounts"] = pd.read_csv(config.DATASET_DIR / "business_accounts.csv")
    d["user_business_history"] = pd.read_csv(config.DATASET_DIR / "user_business_history.csv")
    d["message_history"] = pd.read_csv(config.DATASET_DIR / "message_history.csv")
    d["message_events"] = pd.read_csv(config.DATASET_DIR / "message_events.csv")
    d["images"] = pd.read_csv(config.DATASET_DIR / "images.csv")
    d["voice_notes"] = pd.read_csv(config.DATASET_DIR / "voice_notes.csv")
    d["daily_notification_summary"] = pd.read_csv(config.DATASET_DIR / "daily_notification_summary.csv")

    # Index lookups used repeatedly during context assembly.
    d["users_by_id"] = d["users"].set_index("user_id")
    d["groups_by_id"] = d["groups"].set_index("group_id")
    d["business_by_id"] = d["business_accounts"].set_index("business_id")

    return d


def _user_profile(data: dict, user_id: str) -> dict:
    if user_id not in data["users_by_id"].index:
        return {}
    row = data["users_by_id"].loc[user_id]
    return {
        "do_not_disturb_window": row["do_not_disturb_window"],
        "messages_opened_30d": int(row["messages_opened_30d"]),
        "messages_replied_30d": int(row["messages_replied_30d"]),
        "notifications_dismissed_30d": int(row["notifications_dismissed_30d"]),
        "messages_reported_30d": int(row["messages_reported_30d"]),
    }


def _group_context(data: dict, group_id: str, user_id: str) -> dict:
    ctx = {}
    if group_id in data["groups_by_id"].index:
        g = data["groups_by_id"].loc[group_id]
        ctx["group_name"] = g["group_name"]
        ctx["group_type"] = g["group_type"]
        ctx["member_count"] = int(g["member_count"])
        ctx["admin_count"] = int(g["admin_count"])
        ctx["group_messages_30d"] = int(g["messages_30d"])

    gm = data["group_members"]
    match = gm[(gm["group_id"] == group_id) & (gm["user_id"] == user_id)]
    if not match.empty:
        m = match.iloc[0]
        ctx["user_role_in_group"] = m["role"]
        ctx["user_messages_sent_30d"] = int(m["messages_sent_30d"])
        ctx["user_messages_read_30d"] = int(m["messages_read_30d"])
        ctx["user_replies_sent_30d"] = int(m["replies_sent_30d"])
        ctx["user_dismissed_30d_in_group"] = int(m["notifications_dismissed_30d"])
        ctx["group_muted_by_user"] = bool(m["group_muted_by_user"])
    return ctx


def _sender_is_group_admin(data: dict, group_id: str, sender_user_id: str) -> bool:
    gm = data["group_members"]
    match = gm[(gm["group_id"] == group_id) & (gm["user_id"] == sender_user_id)]
    if match.empty:
        return False
    return match.iloc[0]["role"] == "admin"


def _business_context(data: dict, business_id: str, user_id: str) -> dict:
    ctx = {}
    if business_id in data["business_by_id"].index:
        b = data["business_by_id"].loc[business_id]
        ctx["business_display_name"] = b["display_name"]
        ctx["business_brand_name"] = b["brand_name"]
        ctx["business_category"] = b["category"]
        ctx["business_verified"] = bool(b["verified"])
        ctx["official_domain"] = b["official_domain"] if pd.notna(b["official_domain"]) else None
        ctx["domain_used_by_sender"] = b["domain_used_by_sender"] if pd.notna(b["domain_used_by_sender"]) else None
        ctx["domain_mismatch"] = (
            ctx["official_domain"] is not None
            and ctx["domain_used_by_sender"] is not None
            and ctx["official_domain"] != ctx["domain_used_by_sender"]
        )
        ctx["account_age_days"] = int(b["account_age_days"])
        ctx["business_messages_sent_30d"] = int(b["messages_sent_30d"])
        ctx["business_user_reports_30d"] = int(b["user_reports_30d"])
        ctx["domain_used_by_sender_age_days"] = int(b["domain_used_by_sender_age_days"])

    ubh = data["user_business_history"]
    match = ubh[(ubh["user_id"] == user_id) & (ubh["business_id"] == business_id)]
    if not match.empty:
        h = match.iloc[0]
        ctx["user_business_relationship"] = h["why_user_knows_account"]
        ctx["user_allows_promotions"] = bool(h["allows_promotions"])
        ctx["user_opted_out_of_promotions"] = pd.notna(h["promotions_opted_out_at"])
        ctx["user_activity_count_180d"] = int(h["activity_count_180d"])
        ctx["user_messages_opened_30d_this_business"] = int(h["messages_opened_30d"])
        ctx["user_messages_dismissed_30d_this_business"] = int(h["messages_dismissed_30d"])
        ctx["user_messages_replied_30d_this_business"] = int(h["messages_replied_30d"])
    else:
        ctx["user_business_relationship"] = None
    return ctx


def build_context(data: dict, message_row: pd.Series) -> dict:
    """Assemble full deterministic context for one row of messages.csv."""
    user_id = message_row["user_id"]
    conversation_type = message_row["conversation_type"]

    ctx = {
        "message_id": message_row["message_id"],
        "user_id": user_id,
        "conversation_type": conversation_type,
        "created_at": message_row["created_at"],
        "message_text": message_row["message_text"] if pd.notna(message_row["message_text"]) else "",
        "media_type": message_row["media_type"] if pd.notna(message_row["media_type"]) else None,
        "media_id": message_row["media_id"] if pd.notna(message_row["media_id"]) else None,
        "forwarded_count": int(message_row["forwarded_count"]),
        "user_profile": _user_profile(data, user_id),
    }

    if conversation_type == "group" and pd.notna(message_row["group_id"]):
        group_id = message_row["group_id"]
        sender_user_id = message_row["sender_user_id"]
        ctx["group_id"] = group_id
        ctx["sender_user_id"] = sender_user_id
        ctx["group"] = _group_context(data, group_id, user_id)
        ctx["sender_is_group_admin"] = _sender_is_group_admin(data, group_id, sender_user_id)

    elif conversation_type == "business" and pd.notna(message_row["business_id"]):
        business_id = message_row["business_id"]
        ctx["business_id"] = business_id
        ctx["business"] = _business_context(data, business_id, user_id)

    elif conversation_type == "personal":
        sender_user_id = message_row["sender_user_id"]
        ctx["sender_user_id"] = sender_user_id

    return ctx


def build_all_contexts(data: dict) -> list:
    """Build context dicts for every row in messages.csv, in file order."""
    return [build_context(data, row) for _, row in data["messages"].iterrows()]


if __name__ == "__main__":
    data = load_all()
    contexts = build_all_contexts(data)
    print(f"loaded {len(data['messages'])} messages, built {len(contexts)} contexts")
    print(contexts[0])
