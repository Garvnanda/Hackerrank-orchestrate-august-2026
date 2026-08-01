# Message Notification Router — HackerRank Orchestrate (August 2026)

A personalized routing system for WhatsApp messages: for every incoming message it decides `notify` / `digest` / `mute`, with a `message_type`, a grounded `reason`, a `confidence`, and cited `evidence_message_ids`.

## Architecture, in one paragraph

Preprocessing and retrieval do the personalization heavy-lifting deterministically and cheaply — user/group/business context joins, an engagement-propensity score, and historical evidence candidates — so the LLM's job narrows to synthesis and explanation rather than raw judgment from scratch. A hosted LLM (OpenRouter, free-tier primary with a paid escalation tier for ambiguous/risky cases) makes the final call, fed by those grounded signals instead of guesswork. Deterministic guardrails run after the LLM call to catch what a free-tier model can't be fully trusted to get right alone — most importantly, prompt-injection payloads embedded directly in message text trying to steer the router itself.

**No component here is fine-tuned.** The only trained model is a small logistic regression (seconds on CPU, weak-supervised from real historical engagement data, not from eval labels). Everything else is either a pretrained model used as-is (embeddings, Whisper, the LLM calls) or pure deterministic code.

## Why this design (rejected alternatives)

- **Local fine-tuned LLM**: rejected. No GPU headroom for it, and critically, no real labeled training set exists (`sample_messages.csv` is a ~30-row format-reference set, not training data) — fine-tuning on it would just be overfitting to a handful of examples, exactly what "don't hardcode to the dataset" warns against.
- **Standalone trained classifier for the final action**: rejected. Even with more data, a classifier only covers 2 of the 4 graded dimensions (`action`, `message_type`) — it can't produce a grounded `reason` or real `evidence_message_ids`. Would need an LLM bolted on anyway.

See `implemented.md` for the full stage-by-stage build log and every design decision made along the way, including the specific data-audit findings that shaped this build (e.g. `message_history.csv` has no `message_type` column, so the propensity scorer's feature set had to skip it).

## Pipeline stages -> code

| Stage | What | File |
|---|---|---|
| 0-1 | Data loading + per-message context assembly (user/group/business joins) | `code/data_loader.py` |
| 2 | Engagement propensity scorer (LogisticRegression, trained on `message_history.csv` + `message_events.csv`) | `code/propensity.py` |
| 3 | RAG-style evidence retrieval (multilingual sentence embeddings + cosine similarity, near-duplicate detection) | `code/retrieval.py` |
| 4 | Media understanding (faster-whisper voice transcription; image base64 encoding for the vision call) | `code/media.py` |
| 5 | Core LLM routing decision (OpenRouter, JSON-structured, few-shot from `sample_messages.csv`) | `code/llm_router.py` |
| 6 | Deterministic guardrails (injection detection, scam override, evidence verification, confidence capping) | `code/guardrails.py` |
| 7 | Selective escalation to a paid model on low confidence / payment-risk signals | `code/llm_router.py` (`needs_escalation`) |
| 8 | Batch runner -> `dataset/output.csv` | `code/run_pipeline.py` |
| 9 | Evaluation against the labeled sample rows | `code/evaluate.py` |

## Notable design decisions

- **Embedding model**: local `paraphrase-multilingual-MiniLM-L12-v2` (CPU, free). The dataset mixes English, Hinglish (romanized Hindi), and French — checked OpenRouter's hosted embedding models first and found no free general-purpose multilingual text-embedding option there, so this stays local.
- **OpenRouter model lists never hardcode a single ID** — free-tier models rotate without notice. `config.py`'s `TEXT_MODEL_FALLBACKS` / `VISION_MODEL_FALLBACKS` are ordered lists, retried with backoff, and were verified against OpenRouter's live `/models` endpoint (the first guessed IDs had already been retired to paid-only — worth re-checking if these start 404ing again).
- **Prompt-injection handling is two-layered**: the Stage 5 system prompt explicitly frames message content as untrusted data, never instructions; `guardrails.py` adds a deterministic backstop that force-mutes messages combining injection-style phrasing ("system note", "routing override", "assistant instruction", etc.) with sensitive-info requests or payment pressure — this dataset contains several real adversarial test rows of exactly this shape (`msg_107`, `msg_108`, `msg_109`, `msg_110`, `msg_095`), all correctly caught.
- **`evidence_message_ids` are always verified** against the actual Stage 3 retrieval candidate set before being written out — the LLM cannot cite an id it invented.
- **Every row is error-isolated** in the batch runner — a failure on one message falls back to a safe `digest`/`unknown`/low-confidence row instead of crashing the run.

## Setup

```bash
cd code
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r ../requirements.txt
cp .env.example .env        # then fill in OPENROUTER_API_KEY
```

## Run

```bash
cd code
python run_pipeline.py            # full run over dataset/messages.csv -> dataset/output.csv
python run_pipeline.py 5           # smoke-test on the first 5 rows only
python evaluate.py                 # sanity-check against the labeled sample_messages.csv rows
```

Each stage's module is also independently runnable for debugging (`python data_loader.py`, `python propensity.py`, `python retrieval.py`, `python media.py`, `python guardrails.py`, `python llm_router.py`).

## Requirements met

- Runnable from the terminal, reads all input from `dataset/`.
- No organizer-only files or hardcoded labels — the only trained component (the propensity scorer) is weak-supervised from real historical engagement data, never from the graded eval labels.
- Deterministic where it matters: guardrails, evidence verification, and confidence capping are pure code; LLM calls run at temperature 0.
- Secrets read only from environment variables via `.env` (gitignored, never committed).

## Chat transcript / logging

Per this repo's `AGENTS.md`, every conversation turn during development was logged to `%USERPROFILE%\hackerrank_orchestrate_august26\log.txt` (outside the repo, never committed). That log is the submission's chat transcript.
