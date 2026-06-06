# VernaSolver

> A textbook-grounded AI study assistant for Indian students — ask questions in Marathi, Hindi, or English and get answers drawn strictly from your prescribed textbook, with verifiable page citations and inline diagrams from the original page.

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-vector_store-FF6B6B)](https://www.trychroma.com/)
[![Claude](https://img.shields.io/badge/Anthropic-Claude_Sonnet_4.6-D97757)](https://www.anthropic.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## ✨ Why VernaSolver?

General-purpose AI chatbots (ChatGPT, Gemini) suffer from three serious limitations in an academic context:

- ❌ Their answers come from broad internet training data — **not** from the student's prescribed textbook.
- ❌ They confidently hallucinate facts, equations, and citations.
- ❌ They are predominantly English-first and offer weak regional-language support.

**VernaSolver** closes this gap. It is a Retrieval-Augmented Generation (RAG) system that constrains a state-of-the-art LLM to a closed corpus of student-uploaded textbooks, citing the exact page of every claim and reproducing diagrams from the original page.

---

## 🎯 Features

- 📚 **Textbook-grounded answering** — answers come only from your uploaded PDFs.
- 📄 **Page-level citations** — every claim is traceable, click any source to see the original paragraph.
- 📊 **Inline diagrams** — pages with figures are detected at ingest time and rendered as PNG previews next to the answer.
- 🧠 **Two-stage retrieval** — dense vector search + cross-encoder reranking for high precision.
- 💡 **Auto-extracted Key Point summary card** at the top of every answer.
- 🌐 **Streaming UI** — answers appear token-by-token, like Claude.ai.
- 💬 **Conversation memory** — short follow-ups ("what about its advantages?") work naturally.
- 🔁 **LLM redundancy** — Anthropic Claude as primary, OpenAI GPT-4o-mini as automatic fallback.
- 🧮 **No math hallucinations** — strict system prompt forbids re-derivation; equations are quoted from the book.
- 🖥️ **CLI + Web** — same functionality through `python chatbot.py` or `localhost:8000`.

---

## 🏗️ Architecture

```
       Student
          │
   ┌──────┴──────┐
   │  Web UI/CLI │
   └──────┬──────┘
          ▼
   ┌─────────────────────────────────────────┐
   │   FastAPI Server (server.py)            │
   │   • /api/chat (SSE streaming)           │
   │   • /api/ingest (background job)        │
   │   • /api/page/{book_id}/{n}.png         │
   └────────┬─────────────────────────┬──────┘
            │                          │
   ┌────────▼────────┐       ┌─────────▼─────────┐
   │ Query Pipeline  │       │   LLM Layer       │
   │ (query.py)      │       │   (llm.py)        │
   │  1. Embed query │       │   Claude → OpenAI │
   │  2. ChromaDB    │       │   (fallback)      │
   │  3. Rerank      │       │   Streaming via   │
   │  4. Top-5       │       │   SSE             │
   └────────┬────────┘       └───────────────────┘
            │
   ┌────────▼─────────────────────────────────┐
   │   Persistence                            │
   │   • ChromaDB (vectors + metadata)        │
   │   • books_registry.json                  │
   │   • books/{book_id}.pdf                  │
   │   • static/book_images/ (page cache)     │
   └──────────────────────────────────────────┘
```

See [docs/architecture.md](docs/architecture.md) for the full diagram and data flow.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.13 |
| Backend | FastAPI + Uvicorn |
| PDF Processing | PyMuPDF (`fitz`) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Reranker | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Vector Store | ChromaDB (persistent local) |
| LLM | Anthropic Claude Sonnet 4.6 + OpenAI GPT-4o-mini (fallback) |
| CLI | Click |
| Frontend | Vanilla HTML / CSS / JavaScript + marked.js |
| Streaming | Server-Sent Events (SSE) |

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An Anthropic and/or OpenAI API key

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/VernaSolver.git
cd VernaSolver

# Install dependencies
pip install -r requirements.txt

# Set up API keys
cp .env.example .env
# Edit .env and add your API keys
```

### Run the Web App

```bash
uvicorn server:app --reload --port 8000
```

Open **http://localhost:8000** in your browser.

1. Visit `/admin` to upload a textbook PDF.
2. Pick a subject and book on the main page.
3. Start asking questions.

### Or use the CLI

```bash
# Ingest a book
python chatbot.py ingest "path/to/textbook.pdf" -s "Software Engineering" -t "Software Engineering" -a "Roger Pressman"

# List books
python chatbot.py books

# Start an interactive Q&A session
python chatbot.py ask -s "Software Engineering"
```

---

## 📁 Project Structure

```
VernaSolver/
├── chatbot.py           # CLI entry point
├── server.py            # FastAPI web server
├── ingest.py            # PDF → chunks → embeddings pipeline
├── query.py             # Retrieval + reranking + query contextualization
├── llm.py               # Claude/OpenAI streaming abstraction
├── registry.py          # Book registry (JSON-backed)
├── config.py            # Paths and constants
├── static/
│   ├── index.html       # Chat UI
│   └── admin.html       # Book management UI
├── books/               # Ingested PDFs (gitignored)
├── db/                  # ChromaDB index (gitignored)
└── requirements.txt
```

---

## 🔬 How It Works

**Ingestion.** PyMuPDF extracts text page-by-page. A `page_has_visuals()` heuristic flags pages containing diagrams (embedded images > 80×80 px or > 20 vector drawings). Text is split into paragraph-aware semantic chunks (~400 words, 40-word overlap), embedded with MiniLM, and stored in ChromaDB with full metadata.

**Querying.** The query is contextualized using recent conversation history (so "what about its advantages?" expands using the previous question). It is embedded and ChromaDB returns the top 12 candidates, which are reranked by a cross-encoder to the top 5. These chunks are formatted into a context block and streamed to Claude with a strict system prompt that forbids outside knowledge and requires verbatim quoting of mathematical content.

**Streaming.** Tokens flow from Claude → producer thread → `asyncio.Queue` → FastAPI async generator → Server-Sent Events → browser `fetch().body.getReader()` → in-place DOM text-node updates (O(1) per token, rAF-throttled scroll for smoothness).

---

## 👥 Team

This project was developed as part of an internship.

| Name | Role |
|---|---|
| **Pritam Bairagi** | Backend & RAG pipeline, LLM integration, system architecture |
| **Shwetaki Killedar** | Frontend, UI/UX design, streaming chat interface |
| **Preksha Pokharna** | PDF processing, ingestion pipeline, testing |

---

## 📝 License

MIT — see [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

- [Anthropic](https://www.anthropic.com/) for Claude.
- [ChromaDB](https://www.trychroma.com/) for the local vector store.
- [sentence-transformers](https://www.sbert.net/) for embedding and reranker models.
- [PyMuPDF](https://pymupdf.readthedocs.io/) for everything PDF-related.

---

## 📚 References

- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*, NeurIPS 2020
- Karpukhin et al., *Dense Passage Retrieval for Open-Domain QA*, EMNLP 2020
- Nogueira & Cho, *Passage Re-ranking with BERT*, 2019
- Reimers & Gurevych, *Sentence-BERT*, EMNLP 2019
