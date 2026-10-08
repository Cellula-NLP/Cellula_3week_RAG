# Task 0: BigBird Model Research
*Breaking the Quadratic Barrier in Transformers*

- **The Problem:** BERT's dense self-attention creates an \(N \times N\) matrix (\(O(N^2)\)), capping sequences at 512 tokens.
- **The Innovation:** BigBird combines **Global**, **Local**, and **Random** sparse attention to achieve linear \(O(N)\) complexity.
- **The Impact:** Enables processing of sequences up to 4096 tokens without losing global context.
- **Rule of Thumb:** Use BigBird for > 1024 tokens; use BERT for shorter sequences (especially 512–1024 tokens).

---

# Task 1: Personal RAG System

A Retrieval-Augmented Generation (RAG) system that builds an intelligent 
knowledge base from a person's professional information.

## Architecture
- **Data**: `data/` folder with LinkedIn, CV (PDF), GitHub, certifications, portfolio
- **Processing**: `rag_utils.py` — multi-format loading, chunking, embedding
- **Pipeline**: `rag_chain.py` — prompt engineering + LCEL RAG chain
- **Demo**: `rag_notebook.ipynb` — end-to-end walkthrough

## Setup
1. Create `.env` with `OPENROUTER_API_KEY=your_key`
2. `pip install -r requirements.txt`
3. Open `rag_notebook.ipynb` and run all cells.

## Features
- Multi-format document ingestion (TXT, PDF, MD)
- Chunking with overlap for context preservation
- HuggingFace embeddings + ChromaDB vector store
- Source citation in every answer
- Hallucination guardrails