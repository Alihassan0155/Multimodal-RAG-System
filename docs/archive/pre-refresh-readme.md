# Multimodal RAG System

A production-style **Multimodal Retrieval-Augmented Generation (RAG)** project focused on PDF understanding.  
It ingests documents, chunks content, stores vector embeddings in ChromaDB, and supports retrieval workflows that can be extended with reranking and LLM-based answer generation.

## Features

- PDF ingestion pipeline for document processing
- Chunking and embedding generation
- Persistent vector storage with ChromaDB
- CLI workflow for:
  - `ingest` a PDF
  - checking `stats`
  - `delete` by `doc_id`
  - running `query`
- Clean project structure for extension into full hybrid retrieval + reranking

## Tech Stack

- **Language:** Python
- **Vector DB:** ChromaDB
- **Embeddings:** Sentence Transformers
- **Document Processing:** PyMuPDF, pdfplumber, Camelot (tables)
- **Retrieval Utilities:** rank-bm25, tiktoken, LangChain text splitters
- **Environment:** python-dotenv

## Project Structure

```text
Multimodal-rag/
├── main.py
├── requirements.txt
├── .gitignore
├── src/
│   └── ... pipeline modules ...
├── data/
│   └── ... input/processed files ...
└── chroma_db/
```

## Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/Alihassan0155/multimodal-rag-system.git
   cd multimodal-rag-system
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   # Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables:

   - Create a `.env` file in the project root
   - Add any required keys/config values used by your local setup

## Usage

Run commands from the project root:

```bash
python main.py ingest path/to/document.pdf
python main.py stats
python main.py delete <doc_id>
python main.py query "Your question here"
```

## Current Status

- Core ingest + retrieval flow is implemented
- Query workflow is operational and structured for iterative improvement
- Ready for extension with:
  - hybrid retrieval (sparse + dense)
  - reranking
  - final grounded answer generation with an LLM

## Why This Project

This repository demonstrates practical RAG engineering skills:

- building modular data pipelines
- handling real-world PDFs (including table-heavy files)
- designing retrieval systems for accuracy and scalability
- preparing a codebase for production-grade iteration

## Author

**Ali Hassan**  
GitHub: [@Alihassan0155](https://github.com/Alihassan0155)
