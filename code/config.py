"""Central config: paths, env vars, model fallback lists."""
import os
from pathlib import Path

from dotenv import load_dotenv

CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent

load_dotenv(CODE_DIR / ".env")
DATASET_DIR = PROJECT_DIR / "dataset"
MEDIA_DIR = DATASET_DIR / "media"
OUTPUT_CSV = DATASET_DIR / "output.csv"

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Free-tier text models rotate without notice — never call a single hardcoded ID.
# Ordered fallback: first is tried first, subsequent ones used on 429/5xx/empty response.
# Verified live against https://openrouter.ai/api/v1/models on 2026-08-01 — re-check
# that endpoint if these start 404ing again (free-tier lineup rotates).
TEXT_MODEL_FALLBACKS = [
    "nvidia/nemotron-3-super-120b-a12b:free",
    "google/gemma-4-31b-it:free",
    "openai/gpt-oss-20b:free",
    "nvidia/nemotron-3-nano-30b-a3b:free",
]

VISION_MODEL_FALLBACKS = [
    "google/gemma-4-31b-it:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
]

# Paid escalation model for Stage 7 (low-confidence / payment-scam-signal messages only).
ESCALATION_MODEL = "openai/gpt-4o-mini"

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES_PER_MODEL = 2
RETRY_BACKOFF_SECONDS = 3

EMBEDDING_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
WHISPER_MODEL_SIZE = "small"
WHISPER_BEAM_SIZE = 5

LLM_TEMPERATURE = 0
RETRIEVAL_TOP_K = 5
