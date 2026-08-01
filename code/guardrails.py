"""Stage 6: deterministic post-validation & override. Pure Python, no model.

Runs after the Stage 5 LLM call to catch what free-tier model output can't
be fully trusted to get right on its own:

1. Prompt-injection detection — message_text/media_transcript containing
   fake "system note" / "routing override" / "assistant instruction" text
   trying to steer the router (e.g. msg_107, msg_108, msg_109, msg_110,
   msg_095 in this dataset). This is a backstop for the Stage 5 system
   prompt's own untrusted-content framing, not a replacement for it.
2. Hard safety override — unverified/new business + payment-request
   language + no user transaction history -> force mute/scam regardless
   of what the LLM returned.
3. Evidence-id verification — drop any evidence_message_ids the LLM
   invented that aren't in Stage 3's actual candidate set.
4. Confidence capping — cap confidence when there's no real evidence to
   back the decision.
"""
import re

INJECTION_PATTERNS = [
    r"routing override",
    r"system note",
    r"internal router metadata",
    r"assistant instruction",
    r"ignore (all )?(previous|prior) (routing )?(rules|instructions)",
    r"ignore sender risk",
    r"mark (this|it) as notify",
    r"mark notify",
    r"set action\s*=",
    r"classify as (urgent|notify)",
    r"notification router",
    r"for the (notification )?router",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)

SENSITIVE_INFO_PATTERNS = [
    r"\botp\b",
    r"\bpin\b",
    r"login code",
    r"verification code",
    r"password",
    r"card number",
    r"cvv",
    r"wallet pin",
    r"\b6[\s-]?digit\b",
    r"bank details",
    r"account number",
]
_SENSITIVE_RE = re.compile("|".join(SENSITIVE_INFO_PATTERNS), re.IGNORECASE)

PAYMENT_PRESSURE_PATTERNS = [
    r"pay (now|today|immediately)",
    r"scan (this |the )?qr",
    r"clearance amount",
    r"processing fee",
    r"token amount",
    r"release the amount",
    r"complete (the )?verification",
    r"(account|access|profile|service) (will be|may be|expires?) (blocked|locked|restricted|suspended|today)",
    r"before (it|access) (expires|is restricted)",
    r"expires? (today|soon)",
    r"login now",
    r"verify (now|immediately|today)",
]
_PAYMENT_PRESSURE_RE = re.compile("|".join(PAYMENT_PRESSURE_PATTERNS), re.IGNORECASE)

NEW_BUSINESS_AGE_DAYS_THRESHOLD = 60


def _combined_text(ctx: dict) -> str:
    parts = [ctx.get("message_text") or "", ctx.get("media_transcript") or ""]
    return " ".join(p for p in parts if p)


def detect_injection_attempt(ctx: dict) -> bool:
    return bool(_INJECTION_RE.search(_combined_text(ctx)))


def detect_sensitive_info_request(ctx: dict) -> bool:
    return bool(_SENSITIVE_RE.search(_combined_text(ctx)))


def detect_payment_pressure(ctx: dict) -> bool:
    return bool(_PAYMENT_PRESSURE_RE.search(_combined_text(ctx)))


def is_suspicious_business(ctx: dict) -> bool:
    """True if the business sender looks unverified/newly-created/risky.

    Domain mismatch (official domain vs. domain actually used by sender) is
    the strongest single signal in this dataset — every unverified business
    row in business_accounts.csv has one.
    """
    business = ctx.get("business")
    if not business:
        return False
    if business.get("domain_mismatch"):
        return True
    if not business.get("business_verified", True):
        return True
    if business.get("account_age_days", 10_000) < NEW_BUSINESS_AGE_DAYS_THRESHOLD:
        return True
    return False


def has_user_transaction_history(ctx: dict) -> bool:
    business = ctx.get("business")
    if not business:
        return False
    return business.get("user_business_relationship") is not None


def should_force_scam_mute(ctx: dict) -> tuple:
    """Hard safety override. Returns (should_override, reason) or (False, None).

    Fires when ANY of:
    - injection attempt detected + sensitive-info request in the same message
      (the exact msg_107/108/109/110/095 pattern in this dataset)
    - unverified/new/domain-mismatched business + payment-pressure language +
      no prior user-business relationship
    - sensitive-info request + payment-pressure language from a personal/group
      sender with no verified business backing it at all (classic OTP-phishing
      shape, e.g. msg_091, msg_110)
    """
    injection = detect_injection_attempt(ctx)
    sensitive = detect_sensitive_info_request(ctx)
    pressure = detect_payment_pressure(ctx)

    if injection and (sensitive or pressure):
        return True, "Message tries to instruct the router directly while requesting sensitive info or applying payment pressure; treated as scam regardless of its claimed framing."

    if ctx.get("conversation_type") == "business":
        if is_suspicious_business(ctx) and pressure and not has_user_transaction_history(ctx):
            return True, "Sender is an unverified or newly created business using payment-pressure language, with no prior transaction history for this user."

    if sensitive and pressure and ctx.get("conversation_type") != "business":
        return True, "Message requests OTP/PIN/verification details under urgency pressure, a classic phishing pattern."

    return False, None


def verify_evidence_ids(claimed_ids: list, candidate_ids: set) -> list:
    """Drop any evidence_message_ids not present in Stage 3's candidate set."""
    return [mid for mid in claimed_ids if mid in candidate_ids]


def apply_guardrails(ctx: dict, evidence_candidates: list, llm_output: dict) -> dict:
    """Apply all Stage 6 rules to one LLM decision. Returns the final row dict.

    llm_output expected keys: action, message_type, reason, confidence,
    evidence_message_ids (list of str).
    """
    candidate_ids = {c["message_id"] for c in evidence_candidates}
    verified_ids = verify_evidence_ids(llm_output.get("evidence_message_ids", []), candidate_ids)

    result = {
        "message_id": ctx["message_id"],
        "action": llm_output.get("action", "digest"),
        "message_type": llm_output.get("message_type", "unknown"),
        "reason": llm_output.get("reason", ""),
        "confidence": float(llm_output.get("confidence", 0.5)),
        "evidence_message_ids": verified_ids,
    }

    force_mute, override_reason = should_force_scam_mute(ctx)

    if not verified_ids:
        result["confidence"] = min(result["confidence"], 0.5)

    if force_mute:
        # Hard safety override: certainty comes from the deterministic rule,
        # not from retrieval evidence, so it must not be undone by the
        # no-evidence cap above.
        result["action"] = "mute"
        result["message_type"] = "scam"
        result["reason"] = override_reason
        result["confidence"] = max(result["confidence"], 0.85)

    result["confidence"] = round(min(max(result["confidence"], 0.0), 1.0), 2)
    return result


if __name__ == "__main__":
    import data_loader

    data = data_loader.load_all()
    contexts = data_loader.build_all_contexts(data)

    injection_ids = {"msg_107", "msg_108", "msg_109", "msg_110", "msg_095"}
    for ctx in contexts:
        if ctx["message_id"] in injection_ids:
            override, reason = should_force_scam_mute(ctx)
            print(ctx["message_id"], "force_override=", override, "|", reason)

    print("---")
    for mid in ("msg_091", "msg_030", "msg_016", "msg_003"):
        ctx = next(c for c in contexts if c["message_id"] == mid)
        override, reason = should_force_scam_mute(ctx)
        print(mid, "force_override=", override, "|", reason)
