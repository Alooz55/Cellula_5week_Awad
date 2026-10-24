"""Central configuration (read from environment / .env)."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", BASE_DIR / "storage"))
STORAGE_DIR.mkdir(parents=True, exist_ok=True)

# ---- LLM (same free providers as the previous project) ----
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "openrouter").lower()   # openrouter | ollama
LLM_MODEL = os.environ.get("LLM_MODEL", "")                           # blank = provider default
OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "").strip().strip('"').strip("'")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
LLM_TIMEOUT_SEC = float(os.environ.get("LLM_TIMEOUT_SEC", "60"))

# ---- Speech-to-text (Faster-Whisper) ----
WHISPER_MODEL = os.environ.get("WHISPER_MODEL", "small")          # tiny|base|small|medium|large-v3
WHISPER_DEVICE = os.environ.get("WHISPER_DEVICE", "auto")         # auto|cpu|cuda
WHISPER_COMPUTE_TYPE = os.environ.get("WHISPER_COMPUTE_TYPE", "int8")
MAX_AUDIO_MB = int(os.environ.get("MAX_AUDIO_MB", "25"))

# ---- SQL execution ----
MAX_ROWS = int(os.environ.get("MAX_ROWS", "1000"))
SQL_TIMEOUT_SEC = float(os.environ.get("SQL_TIMEOUT_SEC", "5"))
MAX_SQL_RETRIES = int(os.environ.get("MAX_SQL_RETRIES", "2"))
MAX_UPLOAD_MB = int(os.environ.get("MAX_UPLOAD_MB", "50"))
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "4"))
