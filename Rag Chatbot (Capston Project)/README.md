# Amazon Quarterly Report RAG Chatbot

A retrieval-augmented generation (RAG) chatbot that answers questions about
Amazon's quarterly report (10-Q), grounded in the actual filing text — with
citations back to the source excerpts it used.

**Stack:** FastAPI · sentence-transformers (local embeddings) · FAISS (vector
search) · Gemini (Google AI) for answer generation · vanilla JS chat UI.

```
User question
   │
   ▼
Embed query (MiniLM) ──► FAISS similarity search ──► top-k report excerpts
   │                                                        │
   └───────────────────────► Gemini (with excerpts as context) ──► grounded answer
```

## Project layout

```
amazon-rag-chatbot/
├── app/
│   ├── config.py      # all settings, env-var driven
│   ├── ingest.py       # download/parse report → chunk → embed → FAISS index
│   ├── rag.py          # retrieval + Gemini call
│   └── main.py         # FastAPI app (serves API + chat UI)
├── static/              # chat UI (HTML/CSS/JS)
├── data/
│   ├── source/          # drop a local PDF here if not using a URL
│   └── index/           # generated FAISS index + chunks (gitignored)
├── requirements.txt
├── render.yaml           # Render Blueprint (one-click infra-as-code deploy)
└── .env.example
```

## 1. Local setup

```bash
cd amazon-rag-chatbot
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and set GEMINI_API_KEY=AIza...
```

Build the index (downloads the default Amazon 10-Q filing, chunks it, and
embeds it — takes ~1-2 minutes the first time while the embedding model
downloads):

```bash
python -m app.ingest
```

Run the server:

```bash
uvicorn app.main:app --reload
```

Open **http://localhost:8000** and start asking questions, e.g.:
- "What were total net sales this quarter and how does that compare to last year?"
- "How did the AWS segment perform?"
- "What risk factors does the filing mention?"
- "Summarize the cash flow from operating activities."

## 2. Using your own PDF instead

You have two options — no code changes needed either way:

**Option A — point at any PDF/HTML URL:**
```bash
export SOURCE_URL="https://example.com/some-quarterly-report.pdf"
export SOURCE_LABEL="Company X Q2 2026 10-Q"
python -m app.ingest --force
```

**Option B — use a local file:**
```bash
# put your file at data/source/my_report.pdf, then:
export LOCAL_SOURCE_PATH="data/source/my_report.pdf"
python -m app.ingest --force
```

The default source is Amazon's Form 10-Q for the quarter ended March 31,
2026, as filed with the SEC:
`https://www.sec.gov/Archives/edgar/data/0001018724/000101872426000014/amzn-20260331.htm`

You can also just `POST /api/reindex` on a running server to rebuild without
restarting (handy after changing `SOURCE_URL`) — consider putting this
endpoint behind auth before exposing it publicly.

## 3. Deploying to Render

### Option A — Blueprint (recommended, one click)

1. Push this project to a GitHub repo.
2. In Render: **New +** → **Blueprint** → connect the repo. Render will read
   `render.yaml` and provision the web service automatically.
3. When prompted, set the secret env var **`GEMINI_API_KEY`** (get one at
   https://aistudio.google.com/apikey).
4. Deploy. Render will run `pip install -r requirements.txt && python -m app.ingest`
   as the build step (this downloads the filing and builds the FAISS index),
   then start the server with `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.
5. Once live, visit the service URL — the chat UI is served at `/`.

### Option B — Manual web service

1. **New +** → **Web Service** → connect your repo.
2. Runtime: **Python 3**
3. Build Command: `pip install -r requirements.txt && python -m app.ingest`
4. Start Command: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variable `GEMINI_API_KEY` (and optionally `SOURCE_URL`,
   `SOURCE_LABEL`, etc. — see `.env.example`).
6. Health check path: `/api/health`

### A note on instance size

`sentence-transformers` pulls in PyTorch, which is comfortably fine on
Render's **Standard** plan (2 GB RAM) but can be tight on the free/Starter
tier (512 MB). If you want to run on the smallest tier, either:
- upgrade to Standard for reliability, or
- swap the embedding step in `app/rag.py` / `app/ingest.py` for a lighter
  retrieval method (e.g. TF-IDF via scikit-learn, or an embeddings API like
  Voyage AI/OpenAI) to drop the torch dependency entirely.

### Resilience across restarts

The FastAPI app also rebuilds the index automatically on startup
(`ensure_index_ready` in `app/main.py`) if it's missing — so even if Render's
disk resets between deploys, the very first request after a cold start will
trigger a rebuild rather than erroring out permanently.

## 4. API reference

| Endpoint       | Method | Description                                      |
|----------------|--------|---------------------------------------------------|
| `/`            | GET    | Chat UI                                            |
| `/api/health`  | GET    | Index/API-key readiness check                      |
| `/api/chat`    | POST   | `{ "message": str, "history": [{role,content}] }` → `{ answer, sources }` |
| `/api/reindex` | POST   | Force-rebuild the index from the current source    |

## 5. Customizing

- **Chunking:** tune `CHUNK_SIZE` / `CHUNK_OVERLAP` in `.env`.
- **Retrieval depth:** tune `TOP_K`.
- **Model:** change `GEMINI_MODEL` (e.g. to a faster/cheaper or higher-capability model).
- **System prompt / tone:** edit `SYSTEM_PROMPT` in `app/rag.py`.
- **Multiple documents:** the current ingest pipeline handles one source at a
  time; to index several filings, extend `ingest.py` to loop over a list of
  URLs/paths and tag each chunk with its source name before embedding.

## Disclaimer

This tool summarizes and answers questions about a public SEC filing. It is for informational purposes only and is not investment advice.