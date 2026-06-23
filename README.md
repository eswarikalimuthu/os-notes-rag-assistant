# OS Notes RAG Assistant

A Retrieval-Augmented Generation (RAG) application that answers Operating Systems questions using a custom knowledge base, semantic search, and a local Large Language Model (LLM).

The system retrieves relevant Operating Systems notes from a vector database and generates grounded responses with source citations, improving answer accuracy and reliability.

---

## Project Overview

Traditional LLMs can generate inaccurate or hallucinated responses when answering domain-specific questions. This project addresses that challenge using Retrieval-Augmented Generation (RAG).

When a user submits a query:

1. The query is converted into vector embeddings.
2. ChromaDB retrieves the most relevant Operating Systems notes.
3. The retrieved content is provided to the language model.
4. The model generates a response based on the retrieved context.
5. Sources and retrieval scores are displayed alongside the answer.

---

## Features

* Semantic search using Sentence Transformers
* Retrieval-Augmented Generation (RAG)
* ChromaDB vector database
* Local LLM integration using Ollama
* Streamlit web interface
* FastAPI backend API
* Source citations
* Retrieval queue visualization
* Local execution without external API costs

---

## System Architecture

```text
User Query
    ↓
Sentence Transformer Embeddings
    ↓
ChromaDB Vector Database
    ↓
Top-K Relevant Chunks Retrieval
    ↓
Ollama Local LLM
    ↓
Grounded Answer with Citations
```

---

## Technology Stack

### Frontend

* Streamlit

### Backend

* FastAPI
* Python

### Vector Database

* ChromaDB

### Embeddings

* Sentence Transformers

### Language Model

* Ollama

### Additional Libraries

* Requests
* Pydantic
* Uvicorn

---

## Project Structure

```text
os-notes-rag-assistant/
│
├── app.py
├── frontend.py
├── cli.py
├── rag.py
├── rag_free.py
├── ingest.py
├── requirements.txt
├── README.md
├── data/
├── chroma_db/
└── .gitignore
```

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/eswarikalimuthu/os-notes-rag-assistant.git
cd os-notes-rag-assistant
```

### Create a Virtual Environment

```bash
python -m venv venv
```

### Activate the Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Build the Vector Database

```bash
python ingest.py
```

This command loads the Operating Systems notes and stores their embeddings in ChromaDB.

---

## Run the Backend

```bash
uvicorn app:app --reload
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Run the Frontend

```bash
streamlit run frontend.py
```

Application URL:

```text
http://localhost:8501
```

---

## Demonstration

Add your project screenshot to the repository as:

```text
demo.png
```

Then include:

```markdown
![OS Notes RAG Assistant](demo.png)
```

---

## Sample Query

Question:

```text
What is a deadlock?
```

Answer:

```text
A deadlock occurs when two or more processes are each waiting indefinitely for a resource held by another process in the same set, so none of them can proceed.
```

Retrieved Sources:

* OS-03 Deadlocks
* OS-04 Process Synchronization
* OS-01 Process vs Thread

---

## Learning Outcomes

This project provided practical experience in:

* Retrieval-Augmented Generation (RAG)
* Vector Databases
* Embedding Models
* Semantic Search
* Prompt Engineering
* FastAPI Development
* Streamlit Application Development
* Local LLM Deployment using Ollama

---

## Future Enhancements

* PDF upload and indexing
* Conversational memory
* Hybrid Search (Keyword + Vector)
* Multi-document support
* User authentication
* Cloud deployment

---

## license
This project is for educational and learning purpose only
[OS Notes RAG Assistant](sc-1.png)
[OS Notes RAG Assistant](sc-2.png)
[OS Notes RAG Assistant](sc-3.png)
[OS Notes RAG Assistant](sc-4.png)
[OS Notes RAG Assistant](sc-5.png)
[OS Notes RAG Assistant](sc-6.png)


GitHub: https://github.com/eswarikalimuthu
