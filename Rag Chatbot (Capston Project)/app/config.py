"""
Central configuration for the RAG chatbot.
Everything here can be overridden with environment variables so the
same codebase works locally and on Render without code changes.
"""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------------------
# Data source: the quarterly report to load into the knowledge base.
# Can be:
#   - a URL to a PDF   (e.g. https://.../report.pdf)
#   - a URL to an HTML filing (e.g. an SEC EDGAR .htm 10-Q)
#   - a local path to a PDF placed in data/source/ (set LOCAL_SOURCE_PATH)
# Default: Amazon's Q1 2026 10-Q as filed with the SEC.
# ---------------------------------------------------------------------------
SOURCE_URL = os.getenv(
    "SOURCE_URL",
    "https://www.sec.gov/Archives/edgar/data/0001018724/000101872426000014/amzn-20260331.htm",
)

# If set, this local file is used INSTEAD of downloading SOURCE_URL.
# Point it at a PDF you've dropped into data/source/, e.g. "data/source/my_report.pdf"
LOCAL_SOURCE_PATH = os.getenv("LOCAL_SOURCE_PATH", "")

# A human-friendly label shown to users / used in citations
SOURCE_LABEL = os.getenv("SOURCE_LABEL", "Amazon.com, Inc. Form 10-Q (Quarter Ended March 31, 2026)")

# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))       # characters per chunk
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))  # overlap between chunks

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = int(os.getenv("TOP_K", "5"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Generation (Google Gemini)
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))

# ---------------------------------------------------------------------------
# Storage paths
# ---------------------------------------------------------------------------
INDEX_DIR = BASE_DIR / "data" / "index"
FAISS_INDEX_PATH = INDEX_DIR / "index.faiss"
CHUNKS_PATH = INDEX_DIR / "chunks.json"
META_PATH = INDEX_DIR / "meta.json"

# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------
PORT = int(os.getenv("PORT", "8000"))
