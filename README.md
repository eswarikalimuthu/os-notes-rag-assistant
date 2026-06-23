# OS Notes RAG Assistant (Python backend)

A real Retrieval-Augmented Generation pipeline over a small Operating Systems notes
corpus, using:
- **sentence-transformers** (`all-MiniLM-L6-v2`) for real embeddings
- **ChromaDB** as the vector database
- **Anthropic Claude API** for grounded answer generation
- **FastAPI** to expose it as an HTTP API (plus a terminal CLI for quick testing)

## 1. Requirements
- Python 3.9+
- An Anthropic API key (https://console.anthropic.com -> Settings -> API Keys)
- Internet access (to download the embedding model on first run, and to call the API)

## 2. Setup

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your API key
cp .env.example .env
# then edit .env and paste your real key
```

## 3. Build the vector database (run once, and again whenever /data changes)

```bash
python ingest.py
```

This reads every `.txt` file in `/data`, embeds it with sentence-transformers, and
stores the vectors in a local persistent ChromaDB at `./chroma_db`. The first run will
download the embedding model (~90MB) — this needs internet access.

## 3.5 Want this 100% free, no API key at all?

Everything except the final answer-generation step is already free and local
(embeddings + ChromaDB run on your machine). To make generation free too, swap
the Claude API for a local LLM via **Ollama**:

```bash
# 1. Install Ollama: https://ollama.com/download
# 2. Pull a small free model
ollama pull llama3.2        # or `ollama pull phi3` on a lower-RAM laptop
```

Then in `cli.py` and `app.py`, comment out `import rag` and uncomment
`import rag_free as rag`. That's it — no `.env`, no API key, no internet
needed after the model is pulled. `rag_free.py` does the exact same retrieval,
just calls your local Ollama model instead of the Anthropic API for the answer.

Tradeoff: local models like Llama 3.2 are noticeably less capable than Claude,
so answers may be rougher — fine for a free demo, but mention this tradeoff
explicitly in your project report; examiners respect that kind of honesty.



```bash
python cli.py
```

Example:
```
Ask> what is a deadlock
--- Retrieved chunks ---
  OS-03_deadlocks             (Deadlocks)  distance=0.31
  ...
--- Answer ---
A deadlock occurs when... [OS-03_deadlocks]
```

## 5. Run as a real API (for your resume/demo)

```bash
uvicorn app:app --reload
```

Then either:
- Open **http://127.0.0.1:8000/docs** for an interactive Swagger UI, or
- Call it directly:

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "explain LRU page replacement", "k": 3}'
```

## 6. Project structure

```
os-rag-project/
├── data/              # the "database" - one .txt file per OS topic chunk
├── ingest.py           # embeds chunks and builds the ChromaDB vector store
├── rag.py              # core retrieval + generation logic (used by app.py and cli.py)
├── app.py               # FastAPI server exposing POST /ask
├── cli.py                # terminal demo, no server needed
├── requirements.txt
├── .env.example
└── chroma_db/          # created automatically by ingest.py (vector store on disk)
```

## 7. How it actually works (for your viva/interview)

1. **Indexing (ingest.py)**: each note chunk -> sentence-transformers embedding (384-dim
   vector capturing meaning) -> stored in ChromaDB with its id and title as metadata.
2. **Retrieval (rag.py: retrieve())**: your question is embedded with the *same* model,
   then ChromaDB does a nearest-neighbour search to find the most semantically similar
   chunks (lower distance = more relevant).
3. **Generation (rag.py: generate_answer())**: the retrieved chunks are inserted into a
   prompt sent to Claude, with a system prompt instructing it to answer only from that
   context and cite chunk ids — this is what grounds the answer and reduces hallucination.
4. **API (app.py)**: wraps the pipeline in a FastAPI POST endpoint so it could be plugged
   into a real frontend, a Slack bot, etc.

## 8. Extending this for a stronger resume project
- Swap `/data` for your actual course PDFs (use a PDF parser like `pypdf` and chunk by
  paragraph instead of by file).
- Add an evaluation script that runs a fixed set of test questions and checks whether
  the answer cites the expected chunk id.
- Add a simple web frontend (Streamlit or a small React app) that calls `/ask`.
- Track retrieval quality with `distance` thresholds — if all chunks have a high distance,
  have the bot say "I don't have information on that" instead of guessing.
