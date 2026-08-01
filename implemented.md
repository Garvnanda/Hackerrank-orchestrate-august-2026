# Implemented — Progress Log

Running log of what's built, in what way, and what's still in progress. Updated immediately after every change — do not batch updates.

## Architecture reference

See `docs/idea.md` and `docs/implementation.md` in the parent repo for the full design rationale (why local fine-tuning was rejected, where each model/ML component lives, the 10-stage pipeline). This file only tracks build status, not design rationale.

## Status by stage

| Stage | What | Status |
|---|---|---|
| 0 | Setup & data audit | Done (see below) |
| 1 | Context assembly (pandas joins) | Done — `code/data_loader.py` |
| 2 | Engagement propensity scorer (LogisticRegression) | Done — `code/propensity.py` |
| 3 | RAG evidence retrieval (multilingual embeddings) | Done — `code/retrieval.py` |
| 4 | Media understanding (Whisper + vision) | Done — `code/media.py` |
| 5 | Core LLM decision call (OpenRouter) | Done, live-tested — `code/llm_router.py` |
| 6 | Deterministic guardrails + injection detection | Done — `code/guardrails.py` |
| 7 | Selective escalation | Built — `llm_router.needs_escalation()` (not yet live-tested on an escalation-triggering row) |
| 8 | Batch runner -> output.csv | Done, full 110-row run completed |
| 9 | Evaluation script | Done — `code/evaluate.py`, run against 30 labeled rows |
| 10 | README + packaging | Done — `README.md`, `requirements.txt` |

## Decisions locked in (from planning discussion)

- Embedding model: local `paraphrase-multilingual-MiniLM-L12-v2` (sentence-transformers, CPU). Checked OpenRouter's embedding collection — no free general-purpose multilingual text embedding model exists there (only free one, Nemotron Embed VL 1B V2, is vision-multimodal, not a fit); bge-m3 is the best multilingual option but paid. Going local per fallback instruction.
- OpenRouter model fallback lists: picked as ordered lists (never a single hardcoded ID), with retry/backoff + rotation on 429/5xx. Free-tier primary, one paid escalation model for Stage 7.
- Propensity scorer: sklearn `LogisticRegression`. `message_type` excluded as a feature — it doesn't exist as a real column in `message_history.csv`, only in the small `sample_messages.csv` reference set.
- Injection handling: both prompt-hardening (Stage 5 system prompt marks message content as untrusted data, never instructions) AND a deterministic Stage 6 regex/keyword override (forces mute/scam when injection-style phrasing + sensitive-info request appear together), as a backstop.
- Project is self-contained under this folder: `dataset/` copied in here (not referenced from parent repo) so the eventual separate GitHub repo (`Garvnanda/Hackerrank-orchestrate-august-2026`) works standalone after push.

## Data audit findings (Stage 0)

- All 13 dataset files present and match `problem_statement.md` schema; `output.csv` row order matches `messages.csv` 1:1, columns in exact required order.
- `message_history.csv` has the same schema as `messages.csv` but **no `message_type` column** — implementation.md's Stage 2 feature list assumed it; adjusted (see decisions above).
- `daily_notification_summary.csv` stops at 2026-07-17; `messages.csv` runs to 2026-07-31 — daily load stats are up to 2 weeks stale relative to newest incoming messages, usable only as a baseline signal.
- Prompt-injection payloads embedded directly in `message_text` for msg_107, msg_108, msg_109, msg_110, msg_095 (fake "routing override" / "system note" / "assistant instruction" text trying to force notify/high-confidence). Explicit adversarial test case — handled per decisions above.
- Mixed-language content: Hinglish (romanized Hindi) and one French message present alongside English — drove the embedding model choice.
- No orphaned media references — every `media_id` in `messages.csv` resolves to a real path in `images.csv`/`voice_notes.csv` and an actual file under `dataset/media/`.

## Files built so far

- `code/config.py` — paths, env vars, OpenRouter fallback model lists (text/vision/escalation), retry/embedding/whisper settings.
- `code/data_loader.py` — loads all 10 dataset CSVs; `build_context()` joins user profile + group/group-member info (if group message, incl. whether sender is a group admin) + business/user-business-history info (if business message) into one dict per message. Smoke-tested: loads 110 messages, builds 110 contexts correctly (verified HDFC bank example resolves domain match + user relationship correctly).

- `code/propensity.py` — reuses `data_loader.build_context()` against `message_history.csv` rows, joins each to its `message_events.csv` label (`message_opened` — dismissed/muted/reported rows are 0 by construction), flattens context into categorical + numeric features (conversation type, group/business identity signals, forwarded_count, time-of-day, user's 30d engagement baseline; `message_type` excluded, see decisions). `ColumnTransformer` (OneHotEncoder + StandardScaler) -> `LogisticRegression(class_weight="balanced")`. Smoke-tested: trains on 412 labeled history rows, holdout AUC = 0.92, no convergence warnings after scaling numeric features.

- `code/retrieval.py` — `EvidenceIndex` embeds all non-empty `message_history.csv` text once with `paraphrase-multilingual-MiniLM-L12-v2`; `.query(ctx, top_k)` does cosine similarity + a small additive boost for same sender/group/business (doesn't hard-filter, so cross-sender template matches like the same scam text from a different number still surface), returns candidate message_ids + similarity + known `message_events` outcome. `.is_near_duplicate()` flags near-identical repeats (threshold 0.93) for repetition-based muting. Smoke-tested on real `messages.csv` rows — e.g. the two identical "Selling a barely used Myntra kurta set..." listings (msg_029/msg_030) both correctly retrieve the same top historical matches.
- `code/media.py` — `resolve_media()` fills `media_transcript` (voice, via faster-whisper CPU int8) or `media_data_uri` (image, base64 data URI for the Stage 5 vision call) onto a context dict. Smoke-tested: msg_086 voice note transcribed correctly ("Your airport pickup for tomorrow has moved to 6.15am..."), msg_005 image encoded to a valid base64 data URI.

- `code/guardrails.py` — `should_force_scam_mute(ctx)` fires on: (a) injection phrasing + sensitive-info-or-payment-pressure language together, (b) unverified/domain-mismatched/new business + payment pressure + no prior user-business relationship, (c) sensitive-info request + payment pressure from a non-business sender (OTP-phishing shape). `verify_evidence_ids()` drops any LLM-claimed evidence id not in Stage 3's real candidate set. `apply_guardrails()` ties it together into the final output row, capping confidence to <=0.5 when no evidence survived verification. Smoke-tested against all 5 known injection payloads in `messages.csv` (msg_107/108/109/110/095) — all correctly force to mute/scam — plus msg_091 (OTP phishing, correctly caught) and confirmed no false positives on legit messages (msg_030 marketplace listing, msg_003 Amazon delivery update, msg_016 — a scam without direct OTP/PIN ask, left to the LLM+prompt-hardening layer since guardrails are a backstop, not the primary detector).

- `code/llm_router.py` — `SYSTEM_PROMPT` enforces the untrusted-content framing matching guardrails.py's rules; `build_user_prompt()` assembles context + propensity score + evidence candidates + few-shot examples from `sample_messages.csv`; image messages get an `image_url` content block for vision models. `call_with_fallback()` tries each model in `config.TEXT_MODEL_FALLBACKS`/`VISION_MODEL_FALLBACKS` in order with retry/backoff, extracts+validates JSON (enum-checked action/message_type), and — critically — never raises: on total failure across the whole fallback list it returns a safe `digest`/`unknown`/low-confidence row per the Stage 6 spec, so a bad API day can't crash the batch run. `needs_escalation()` implements the Stage 7 trigger (confidence < 0.6, or sensitive-info/payment-pressure signals present) which routes that message to `config.ESCALATION_MODEL` instead. Dry-run smoke-tested (prompt assembly only, verified output is well-formed) — **live API call untested, no `OPENROUTER_API_KEY` set yet**.

- **Model ID fix**: original guessed free-tier IDs all 404'd ("unavailable for free" — paid-only now). Queried `https://openrouter.ai/api/v1/models` live, found the actual current free-tier list (14 models), updated `config.py`: `TEXT_MODEL_FALLBACKS` = nvidia/nemotron-3-super-120b-a12b:free -> google/gemma-4-31b-it:free -> openai/gpt-oss-20b:free -> nvidia/nemotron-3-nano-30b-a3b:free; `VISION_MODEL_FALLBACKS` = google/gemma-4-31b-it:free -> nvidia/nemotron-nano-12b-v2-vl:free.
- **Live end-to-end test** (`.env` key added by user): ran data_loader -> propensity -> retrieval -> media -> llm_router -> guardrails on 5 real messages. All correct: msg_003 (Amazon delivery) -> notify/business_update citing message_0004; msg_048 (society QR penalty scam) -> mute/scam; msg_107 (injection payload) -> mute/scam via guardrail override; msg_005 (kurta pickup) -> notify/personal; msg_012 (forwarded health tip) -> mute/forward citing prior dismissed forwards. Evidence ids all real, reasons grounded, confidences sensible.
- **Sarvam AI considered** (user has free credits) — no text-embeddings API, so no fit for Stage 3. Real fit: Stage 4 voice transcription — Sarvam's Saaras STT has a "codemix" mode built for Hindi-English mixed speech, relevant since this dataset's text messages show heavy Hinglish usage and some voice notes likely match. Not wired in yet — Whisper (faster-whisper) stays primary since it's already working and free; Sarvam is a candidate upgrade for voice notes specifically if time permits, not a blocker.

- `code/run_pipeline.py` — `run(limit=None)` wires every stage together per message (media resolve -> propensity -> retrieval -> LLM decision -> Stage 7 escalation if low-confidence/payment-signal -> guardrails), writes `output.csv` in the exact required column order with `evidence_message_ids` semicolon-joined (or `none`). `process_message()` wrapped per-row in try/except in `run()` — any exception falls back to `SAFE_FALLBACK_ROW` (digest/unknown, 0.3 confidence) instead of crashing the batch. Optional `limit` arg (`python run_pipeline.py 5`) for smoke-testing on a subset before the full 110-message run. Currently smoke-testing on first 5.

- 5-message smoke test passed: correct column order, `evidence_message_ids` semicolon-joined, quoting handled correctly by pandas for reasons containing commas. Verified escalation model (`openai/gpt-4o-mini`) resolves and costs are negligible (~$0.0000024 for a trivial call) before committing to a full run that would trigger Stage 7 escalation on many payment/urgency-signal rows.
- Full 110-message batch run kicked off (background) -> writing `dataset/output.csv`.

- **Full batch run completed**: 110/110 rows written to `dataset/output.csv`, zero row failures. Distribution: action = mute 56 / notify 38 / digest 16; message_type dominated by scam (27) and spam (14) — consistent with this dataset being heavily loaded with adversarial/scam test cases. Confidence range 0.5-0.96 (0.5 floor from the no-evidence cap working as designed). 10 rows have `evidence_message_ids=none`.

- `code/evaluate.py` — `load_sample_contexts()` strips the label columns off `sample_messages.csv` and builds contexts the same way as real input rows; `run_eval()` reuses `run_pipeline.process_message()` (no duplicated logic) and diffs predicted vs. given `action`/`message_type`, printing accuracy + a full row-level table for manual reason/evidence spot-check. Explicitly framed as a sanity check, not a real metric — it's a ~20-row format-reference set, not a test set. Running now.

- **Eval run results** (30 labeled `sample_messages.csv` rows, not 70 — some sample_msg_XXX numbers are gaps in the file): action accuracy 25/30 = 0.83, message_type accuracy 21/30 = 0.70. Note this includes the sample set's own embedded injection payload (sample_msg_053, "Ignore all previous routing rules...") which the guardrail correctly force-muted as scam. Main error patterns: a few promotion/business_update rows misclassified as spam by the LLM (usually still muted correctly, just wrong sub-type — e.g. sample_msg_011, sample_msg_014, sample_msg_045), one over-cautious mute on a digest-worthy repetitive greeting (sample_msg_009), one over-notify on a low-urgency unknown-sender message (sample_msg_049), one under-notify on a business safety-advisory image treated as promotional spam (sample_msg_048). No systemic failure mode — errors are scattered borderline calls, not a broken rule. Not chasing further tuning against this tiny reference set to avoid overfitting to it (same overfitting risk implementation.md warned about for local fine-tuning).

- `README.md` + `requirements.txt` written — architecture summary, stage-to-file map, setup/run instructions, rejected-approach rationale, log.txt confirmation.

## Post-build AI-judge audit + fixes

User asked for a pre-push audit (code vulnerabilities + output.csv quality) with no changes until reviewed. Findings and resolution:

1. **Critical, fixed**: `.gitignore` had been written to the parent repo root instead of inside `hackerrank-orchestrate-august-2026/` — the folder about to be `git init`'d and pushed as its own repo. As given, the push commands would have committed `code/.env` (real `OPENROUTER_API_KEY`) to GitHub. Moved the file to the correct location.
2. **Confidence-calibration bug, fixed**: in `guardrails.apply_guardrails()`, the no-evidence confidence cap (`min(confidence, 0.5)`) ran *after* the hard safety-override floor (`max(confidence, 0.85)`), silently undoing it whenever the override fired on a message with no retrieval evidence. Confirmed in real output: `msg_085` was correctly muted as scam by the deterministic rule but shipped at confidence 0.5 instead of 0.85. Reordered so the override always wins; only `msg_085` was actually affected in this run (all 12 other override rows had evidence, so the bug never bound for them) — recomputed that one row live with the fixed code (0.5 -> 0.85) and patched `output.csv` directly rather than hand-typing a number.
3. **Low-severity hardening items (path-containment check, JSON-extraction regex greediness)**: skipped per user's explicit instruction — not urgent, not exploited by current data.
5. **Few-shot diversity, fixed**: `_load_few_shot_examples()` in `llm_router.py` was `samples.head(8)` — always the same 8 rows, which happen to be 6 notify + 2 digest, zero mute. Changed to a stratified sample (`groupby("action").head(3)`, sorted back to original order) — verified it now shows 3 notify + 3 digest + 3 mute examples to every LLM call.
6. **Whisper accuracy, upgraded**: `WHISPER_MODEL_SIZE` "base" -> "small", `beam_size` 1 (greedy) -> 5, to reduce mistranscription risk on Hinglish/accented voice notes (13/110 rows). Re-verified `media.py` smoke test still transcribes correctly.
7. **User chose full rerun.** Full 110-message batch completed with the fixed few-shot diversity + Whisper settings. Re-validated: 110/110 rows, zero failures, all evidence citations verified against `message_history.csv`, all enums valid, no out-of-range confidence, no empty reasons. New distribution: mute 59 / notify 39 / digest 12; message_type now led by urgent (34) and scam (23).
9. **Voice-note evidence retrieval — two bugs found and fixed, not one**:
   - Bug A: `message_history.csv`'s 4 historical voice rows had no transcript, so `EvidenceIndex` excluded them entirely. Fixed `_history_text()` to transcribe them with the same Whisper call Stage 4 uses for incoming messages.
   - Bug B (deeper, found while verifying Bug A's fix): `EvidenceIndex.query()` built its query text from `ctx["message_text"]` only — which is *always* empty for voice messages by schema design — so every voice query short-circuited to zero candidates regardless of what was in the index. Bug A's fix alone produced no visible change (still all `evidence_message_ids=none`) until this was caught and fixed too: `query()` now falls back to `ctx["media_transcript"]` when `message_text` is empty.
   - Verified end-to-end after both fixes: `msg_086` ("airport pickup moved to 6:15am") now retrieves `message_0222` at 0.988 similarity — a near-exact historical match. Regenerated the 8 affected voice-note rows in `output.csv` (not a full 110-row rerun).

8. **OpenRouter account ran out of credits mid-eval-run** (confirmed via `/credits` endpoint: `total_credits=0`). This happened *after* the real batch run finished cleanly — `dataset/output.csv` has zero 402 errors, zero fallback rows, unaffected. Only the follow-up `evaluate.py` diagnostic run hit the wall partway through (rows from `sample_msg_046` on), producing contaminated 73%/50% numbers that don't reflect the actual fixes. User topped up with a new key ($10 balance confirmed) — re-running eval on the non-voice sample rows only (excluded 3 voice rows per user's request to not re-trigger Whisper) for a clean read.

4. **Confidence clumping (55% of rows at exactly 0.5/0.92/0.95) and mute-heavy action distribution**: reviewed but deliberately **not hand-edited**. Manually rewriting confidence numbers or action labels without re-deriving them from the actual pipeline would be fabricating output — exactly the "hardcoded answers" risk the problem statement warns against, and would misrepresent what the system actually does. Left as an honest reflection of current LLM behavior; noted as a possible future prompt-tuning target, not a same-day fix.

## All 10 stages complete

`output.csv` has all 110 predictions, evaluation sanity-checked (83% action / 70% message_type accuracy on the 30 labeled reference rows). Remaining work is packaging/submission logistics, not pipeline logic — see git push commands provided directly to the user (not run by the agent, per their instruction).
