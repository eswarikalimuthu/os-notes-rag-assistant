"""
ingest.py
----------
Reads all .txt files in /data, embeds them using a real sentence-transformers
model (all-MiniLM-L6-v2), and stores them in a persistent ChromaDB collection.

Run this once (or whenever you update /data) before starting the API:
    python ingest.py
"""

import os
import glob
import chromadb
from sentence_transformers import SentenceTransformer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "os_notes"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, fast, runs on CPU


def load_documents():
    """Each .txt file in /data is treated as one chunk (small notes -> 1 file = 1 chunk).
    For larger documents you would split each file into multiple smaller chunks here."""
    docs = []
    for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
        chunk_id = os.path.splitext(os.path.basename(path))[0]  # e.g. OS-01_process_vs_thread
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        title = text.split("\n")[0].replace("Title:", "").strip()
        docs.append({"id": chunk_id, "title": title, "text": text})
    return docs


def main():
    print(f"Loading documents from {DATA_DIR} ...")
    docs = load_documents()
    print(f"Found {len(docs)} chunks.")

    print(f"Loading embedding model '{EMBED_MODEL_NAME}' (downloads on first run)...")
    model = SentenceTransformer(EMBED_MODEL_NAME)

    print("Computing embeddings...")
    texts = [d["text"] for d in docs]
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    print(f"Writing to persistent ChromaDB at {DB_DIR} ...")
    client = chromadb.PersistentClient(path=DB_DIR)

    # Recreate the collection fresh each time ingest.py runs
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[d["id"] for d in docs],
        embeddings=embeddings,
        documents=[d["text"] for d in docs],
        metadatas=[{"title": d["title"]} for d in docs],
    )

    print(f"Done. Stored {collection.count()} chunks in collection '{COLLECTION_NAME}'.")


if __name__ == "__main__":
    main()
