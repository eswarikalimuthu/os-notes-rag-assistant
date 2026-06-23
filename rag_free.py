"""
rag_free.py
-----------
100% cost-free version of rag.py — uses a LOCAL LLM via Ollama instead of the
Anthropic API for generation. Retrieval (sentence-transformers + ChromaDB) is
unchanged, since that was already free and local.

Setup (one-time):
1. Install Ollama: https://ollama.com/download
2. Pull a small free model:
     ollama pull llama3.2
   (llama3.2 is ~2GB and runs fine on most laptops with 8GB+ RAM. If your
   laptop is low on RAM, try `ollama pull phi3` instead, which is smaller.)
3. Make sure Ollama is running (it usually starts automatically after install,
   or run `ollama serve` manually).
4. No API key, no .env file, no internet needed once the model is pulled.

Run exactly like rag.py (swap the import in app.py / cli.py to use this file).
"""
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
import os
import requests
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "os_notes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2"  # change to "phi3" if your machine has limited RAM

SYSTEM_PROMPT = (
    "You are a study assistant answering questions about Operating Systems "
    "using ONLY the provided context chunks. Cite the chunk id(s) you used "
    "(e.g. [OS-03_deadlocks]). If the context does not contain the answer, "
    "say so clearly instead of guessing. Keep answers concise and exam-friendly."
)

_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _chroma_client.get_collection(COLLECTION_NAME)


def retrieve(query: str, k: int = 3):
    query_embedding = _embed_model.encode([query]).tolist()[0]
    results = _collection.query(query_embeddings=[query_embedding], n_results=k)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "text": results["documents"][0][i],
            "distance": results["distances"][0][i],
        })
    return chunks


def generate_answer(query: str, chunks: list) -> str:
    """Same idea as rag.py, but calls a local Ollama model instead of the Anthropic API."""
    context = "\n\n".join(f"[{c['id']}] {c['title']}: {c['text']}" for c in chunks)
    prompt = f"{SYSTEM_PROMPT}\n\nContext:\n{context}\n\nQuestion: {query}\n\nAnswer:"

    response = requests.post(OLLAMA_URL, json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
    })
    response.raise_for_status()
    return response.json()["response"].strip()


def ask(query: str, k: int = 3):
    chunks = retrieve(query, k=k)
    answer = generate_answer(query, chunks)
    return {
        "query": query,
        "answer": answer,
        "sources": [{"id": c["id"], "title": c["title"], "distance": c["distance"]} for c in chunks],
    }
