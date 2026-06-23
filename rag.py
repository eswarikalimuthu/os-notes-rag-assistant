"""
rag.py
------
Core retrieval-augmented generation logic, shared by app.py (API) and cli.py (terminal demo).

Retrieval: embed the user's question with the same sentence-transformers model
           used in ingest.py, then query ChromaDB for the top-k most similar chunks.
Generation: pass those chunks as context to Claude via the Anthropic API, with a
            system prompt instructing it to answer only from the given context.
"""

import os
import chromadb
from sentence_transformers import SentenceTransformer
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "os_notes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = (
    "You are a study assistant answering questions about Operating Systems "
    "using ONLY the provided context chunks. Cite the chunk id(s) you used "
    "(e.g. [OS-03_deadlocks]). If the context does not contain the answer, "
    "say so clearly instead of guessing. Keep answers concise and exam-friendly."
)

# Loaded once at import time so repeated calls (e.g. from the API) are fast
_embed_model = SentenceTransformer(EMBED_MODEL_NAME)
_chroma_client = chromadb.PersistentClient(path=DB_DIR)
_collection = _chroma_client.get_collection(COLLECTION_NAME)
_anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY from environment


def retrieve(query: str, k: int = 3):
    """Embed the query and return the top-k most similar chunks from ChromaDB."""
    query_embedding = _embed_model.encode([query]).tolist()[0]
    results = _collection.query(query_embeddings=[query_embedding], n_results=k)

    chunks = []
    for i in range(len(results["ids"][0])):
        chunks.append({
            "id": results["ids"][0][i],
            "title": results["metadatas"][0][i]["title"],
            "text": results["documents"][0][i],
            # ChromaDB returns squared L2 distance by default; smaller = more similar.
            # Converted to a 0-1-ish similarity score just for display purposes.
            "distance": results["distances"][0][i],
        })
    return chunks


def generate_answer(query: str, chunks: list) -> str:
    """Send the retrieved chunks + question to Claude and return the answer text."""
    context = "\n\n".join(f"[{c['id']}] {c['title']}: {c['text']}" for c in chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {query}"

    response = _anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return "".join(block.text for block in response.content if block.type == "text")


def ask(query: str, k: int = 3):
    """Full RAG pipeline: retrieve relevant chunks, then generate a grounded answer."""
    chunks = retrieve(query, k=k)
    answer = generate_answer(query, chunks)
    return {
        "query": query,
        "answer": answer,
        "sources": [{"id": c["id"], "title": c["title"], "distance": c["distance"]} for c in chunks],
    }
