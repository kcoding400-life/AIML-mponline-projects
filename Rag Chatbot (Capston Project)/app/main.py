import json
import logging
import traceback

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import config
from app.ingest import build_index

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("amazon-rag-chatbot")

app = FastAPI(title="Amazon Quarterly Report RAG Chatbot")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: list[ChatMessage] = []


@app.on_event("startup")
def ensure_index_ready():
    """Build the vector index on first boot if it doesn't already exist on disk.
    This makes the service self-healing across Render's ephemeral filesystem,
    even if the build-time `python -m app.ingest` step was skipped."""
    try:
        build_index(force=False)
    except Exception:
        # Don't crash the whole server if the source is briefly unreachable;
        # /api/chat will raise a clear error and /api/health will report it.
        logger.error("Index build failed at startup:\n%s", traceback.format_exc())


@app.get("/api/health")
def health():
    ok = config.FAISS_INDEX_PATH.exists() and config.CHUNKS_PATH.exists()
    meta = {}
    if config.META_PATH.exists():
        meta = json.loads(config.META_PATH.read_text())
    return {
        "status": "ok" if ok else "index_missing",
        "index_ready": ok,
        "has_api_key": bool(config.GEMINI_API_KEY),
        "meta": meta,
    }


@app.post("/api/chat")
def chat(req: ChatRequest):
    from app import rag  # lazy import: avoids loading the embedder before index exists

    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    try:
        history = [m.dict() for m in req.history]
        result = rag.generate_answer(req.message.strip(), history=history)
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception:
        logger.error("Chat request failed:\n%s", traceback.format_exc())
        raise HTTPException(status_code=500, detail="Something went wrong generating the answer.")


@app.post("/api/reindex")
def reindex():
    """Force-rebuild the index (e.g. after changing SOURCE_URL). Consider
    protecting this endpoint before exposing it publicly in production."""
    try:
        build_index(force=True)
        return {"status": "rebuilt"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index_page():
    return FileResponse("static/index.html")
