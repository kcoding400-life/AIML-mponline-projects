"""
Core RAG logic: load the index, retrieve relevant chunks for a query,
and call Claude to generate a grounded answer.
"""
import json
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types as genai_types

from app import config

SYSTEM_PROMPT = """You are a financial research assistant that answers questions \
about {source_label} using ONLY the context excerpts provided below.

Rules:
- Base your answer strictly on the provided context. Do not use outside knowledge \
about Amazon or make up figures.
- If the context does not contain the answer, say so clearly instead of guessing.
- When you cite a number or fact, mention which excerpt it came from, e.g. "(Excerpt 2)".
- Be concise and precise. Use bullet points or short paragraphs for readability.
- This is a financial filing; do not give investment advice or opinions on whether \
to buy/sell the stock -- stick to reporting what the filing says.
"""


@lru_cache(maxsize=1)
def _load_index():
    import faiss

    if not config.FAISS_INDEX_PATH.exists():
        raise RuntimeError(
            "Vector index not found. Build it first with: python -m app.ingest"
        )
    index = faiss.read_index(str(config.FAISS_INDEX_PATH))
    chunks = json.loads(config.CHUNKS_PATH.read_text())
    meta = json.loads(config.META_PATH.read_text()) if config.META_PATH.exists() else {}
    return index, chunks, meta


@lru_cache(maxsize=1)
def _load_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBEDDING_MODEL_NAME)


def retrieve(query: str, top_k: int = None):
    """Return a list of {rank, score, text} for the chunks most relevant to query."""
    top_k = top_k or config.TOP_K
    index, chunks, _meta = _load_index()
    embedder = _load_embedder()

    query_vec = embedder.encode([query], normalize_embeddings=True)
    query_vec = np.asarray(query_vec, dtype="float32")

    scores, indices = index.search(query_vec, min(top_k, len(chunks)))
    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:
            continue
        results.append({"rank": rank, "score": float(score), "text": chunks[idx]})
    return results


def _build_context_block(retrieved: list[dict]) -> str:
    parts = []
    for r in retrieved:
        parts.append(f"[Excerpt {r['rank']} | relevance {r['score']:.2f}]\n{r['text']}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, history: list[dict] | None = None) -> dict:
    """
    history: optional list of {"role": "user"|"assistant", "content": str}
             representing prior turns (not including the current query).
    """
    if not config.GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your environment / Render service settings."
        )

    retrieved = retrieve(query)
    _index, _chunks, meta = _load_index()
    source_label = meta.get("source_label", config.SOURCE_LABEL)
    context_block = _build_context_block(retrieved)

    system = SYSTEM_PROMPT.format(source_label=source_label)

    # Gemini uses "model" instead of "assistant" for the model's turns, and
    # expects each turn as a Content object with a list of Parts.
    contents = []
    for turn in history or []:
        role = "model" if turn.get("role") == "assistant" else "user"
        contents.append(
            genai_types.Content(role=role, parts=[genai_types.Part.from_text(text=turn["content"])])
        )

    user_turn = (
        f"Context excerpts from {source_label}:\n\n{context_block}\n\n"
        f"---\n\nQuestion: {query}"
    )
    contents.append(genai_types.Content(role="user", parts=[genai_types.Part.from_text(text=user_turn)]))

    client = genai.Client(api_key=config.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=config.GEMINI_MODEL,
        contents=contents,
        config=genai_types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=config.MAX_TOKENS,
        ),
    )

    answer_text = response.text or ""

    return {
        "answer": answer_text,
        "sources": [
            {"rank": r["rank"], "score": round(r["score"], 3), "excerpt": r["text"][:280]}
            for r in retrieved
        ],
    }
