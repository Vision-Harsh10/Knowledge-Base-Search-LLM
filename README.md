# Knowledge Base Search (RAG) with Groq + LangChain

This project demonstrates a simple **Retrieval-Augmented Generation (RAG)** pipeline on top of a PDF document.  
It loads a PDF, embeds its pages, retrieves the most relevant chunks for a user query, and then uses a Groq-hosted LLM to generate an answer grounded in the retrieved context.

---

## What I built (so far)

### 1) Environment setup + API key loading
- Loads `GROQ_API_KEY` from a local `.env` file using `python-dotenv`.
- Initializes a Groq client / LangChain Groq chat model.

### 2) Document ingestion (PDF)
- Loads a PDF (example: `pdfs/mlschool.pdf`) using `PyPDFLoader`.
- Each page is converted into a LangChain `Document`.

### 3) Embeddings + Vector Store (Indexing)
- Creates embeddings using **HuggingFace sentence-transformers**:
  - `sentence-transformers/all-MiniLM-L6-v2`
- Stores vectors in an in-memory vector database:
  - `DocArrayInMemorySearch`

### 4) Retrieval
- Builds a retriever from the vector store (`k=4`).
- Tests retrieval to confirm relevant pages are being returned for a query.

### 5) RAG Chain (Retriever → Prompt → Groq LLM)
- Uses a prompt template that forces answers to be based on the retrieved context.
- Full chain:
  - `question → retriever → context`
  - `context + question → prompt`
  - `prompt → Groq LLM`
  - output parsing with `StrOutputParser`

### 6) Batch Q&A
- Runs multiple questions in a loop and prints answers.

---

## Tech Stack / Libraries Used

- **Python**
- **Groq**: LLM inference
  - `groq`
  - `langchain-groq`
- **LangChain** (RAG pipeline)
  - `langchain-core`
  - `langchain-community`
  - `langchain-text-splitters` (recommended even if not used yet)
- **PDF loading**
  - `pypdf` (used by `PyPDFLoader` internally)
- **Embeddings**
  - `sentence-transformers`
  - `transformers`
  - `torch` (required by sentence-transformers)
- **Vector store**
  - `docarray`
- **Environment variables**
  - `python-dotenv`

---

## Setup

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies
```bash
pip install -U pip

pip install \
  groq python-dotenv \
  langchain-core langchain-community langchain-text-splitters langchain-groq \
  docarray \
  sentence-transformers \
  pypdf
```

### 3) Add your Groq API key
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_key_here
```

> Important: `.env` is ignored by git and should never be committed.

---

## Run
Open and run the notebook:
- `notebook.ipynb`

---

## Next Improvements
- Add text chunking (`RecursiveCharacterTextSplitter`) instead of per-page chunks for better retrieval quality.
- Add source citations (show which page(s) were used).
- Add a FastAPI backend endpoint for `/ingest` and `/query` for production usage.
