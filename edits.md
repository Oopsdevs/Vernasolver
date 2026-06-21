# VernaSolver — Project Overview & Edit Log

---

# Part A — What is VernaSolver?

## In one line
VernaSolver is an AI study assistant where students upload their textbooks (PDFs),
ask questions in natural language, and get answers drawn **strictly from the chosen
book** — every answer backed by a verifiable page-number citation.

## The problem it solves
Generic chatbots (ChatGPT, etc.) answer from their training data, not from the
student's prescribed textbook. They invent citations, ignore the syllabus, and
usually only work in English. VernaSolver fixes all three: answers are sourced from
the uploaded book, cited to the exact page, and available in English, Hindi, and
Marathi.

## How it works (RAG pipeline)
VernaSolver is a **Retrieval-Augmented Generation (RAG)** system with three phases:

1. **Ingestion** — A PDF is uploaded. Text is extracted page-by-page with **PyMuPDF**,
   split into overlapping ~400-word chunks, embedded with a local
   **sentence-transformers (all-MiniLM-L6-v2)** model, and stored in **ChromaDB**
   (a local vector database) along with metadata (title, author, subject, page).

2. **Query** — The student's question is embedded with the same model; a cosine
   similarity search over ChromaDB returns the most relevant chunks. A cross-encoder
   (`ms-marco-MiniLM-L-6-v2`) reranks them. Follow-up questions are rewritten using
   conversation history before searching.

3. **Generation** — The retrieved chunks are passed to the LLM with a strict
   "answer only from these excerpts" prompt. The answer is streamed back token by
   token over **Server-Sent Events (SSE)**, with page citations attached.

## Key features
- **Grounded answers** — the LLM is forbidden from using outside knowledge; if the
  answer isn't in the book, it says so (no hallucinations).
- **Page-level citations** — every answer shows its source page; the PDF page can be
  rendered as an image with the cited passage highlighted.
- **Subject & book scoping** — query a single book or all books in a subject.
- **Multi-turn conversation** — remembers context for follow-up questions.
- **Multilingual** — answers in English, Hindi, or Marathi.
- **Small-talk fast path** — greetings skip the vector search entirely.
- **ELI5 mode** — "Explain Like I'm in 6th Grade" simplified answers.
- **Quiz & flashcard generation** — auto-generate a 5-question MCQ quiz or 8
  flashcards from any topic in the book.
- **User accounts** — email + password (PBKDF2-hashed) with session cookies.
- **Dual interface** — a CLI tool (`chatbot.py`) for rapid testing and a streaming
  web app for end users.

## Tech stack
| Layer | Technology |
|---|---|
| Backend framework | FastAPI (Python) |
| PDF parsing | PyMuPDF (fitz) |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 (local, CPU) |
| Reranking | cross-encoder / ms-marco-MiniLM-L-6-v2 |
| Vector database | ChromaDB (local, persistent) |
| Primary LLM | Anthropic Claude (`claude-opus-4-8`) |
| Fallback LLM | OpenAI GPT-4o-mini |
| Streaming | Server-Sent Events (SSE) |
| Auth | SQLite + PBKDF2-SHA256 + session tokens |
| Frontend | Vanilla HTML/CSS/JavaScript (no build step) |
| CLI | Click (Python) |

## Project structure
| File | Purpose |
|---|---|
| `server.py` | FastAPI app — all API endpoints, chat streaming, ingest jobs, auth |
| `chatbot.py` | Command-line interface (ingest / books / ask / remove) |
| `ingest.py` | PDF → text → chunks → embeddings → ChromaDB |
| `query.py` | Vector search, reranking, query contextualization, context formatting |
| `llm.py` | LLM calls (Claude + OpenAI), prompts, quiz/flashcard/small-talk logic |
| `registry.py` | JSON-backed book registry (titles, subjects, paths) |
| `users.py` | User accounts: signup, signin, sessions (SQLite + PBKDF2) |
| `config.py` | Paths, chunk sizes, model names, retrieval constants |
| `static/` | Frontend — `landing.html`, `index.html` (app), `admin.html` |
| `books/` | Uploaded PDF files (gitignored) |
| `db/` | ChromaDB store + registry + users.db (gitignored) |

---

# Part B — Edit Log

A summary of all changes made to the project, grouped by topic.

---

## 1. Setup & Environment

### `SETUP.bat` (added)
One-click Windows installer that:
- Checks for Python 3.11; downloads and installs it silently if missing
- Deletes any old `venv` and clears the pip cache (avoids stale/broken wheels)
- Creates a fresh virtual environment
- Installs pinned, known-good versions: `torch==2.2.2` (CPU), `transformers==4.44.2`,
  `sentence-transformers==3.0.1`, plus all other dependencies
- Verifies every import works
- Generates `start_server.bat` for one-click launch

### Python version requirement
- **Python 3.11** is required. Python 3.13/3.14 break PyTorch and the ML stack
  (DLL errors, pydantic conflicts, "PyTorch >= 2.4 required" mismatches).

### Common install fixes documented during the session
- **`uvicorn` not recognized** → run `python -m uvicorn server:app --reload --port 8000`
- **`field_validator` ImportError (chromadb)** → old pydantic v1; run `pip install --upgrade "pydantic>=2"`
- **torch `WinError 1114` / `c10.dll` failed** → install Visual C++ Runtime
  (https://aka.ms/vs/17/release/vc_redist.x64.exe), then reinstall torch CPU build
- **"Numpy is not available" on book upload** → NumPy 2.x incompatibility;
  run `pip install "numpy<2" --force-reinstall`

### Start command
```
uvicorn server:app --reload --port 8000
```
Then open http://localhost:8000

---

## 2. Model Configuration (`llm.py`, `.env`, `.env.example`)

### Switched default model: Sonnet → Opus
- Replaced all 5 hardcoded `claude-sonnet-4-6` references with a single
  `CLAUDE_MODEL` constant.
- Default model is now **`claude-opus-4-8`** (higher answer quality).

### Made the model configurable via `.env`
- `llm.py`:
  ```python
  CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "").strip() or "claude-opus-4-8"
  ```
- Added to `.env` and `.env.example`:
  ```
  # Claude model to use. Opus = best quality, Sonnet = faster/cheaper.
  CLAUDE_MODEL=claude-opus-4-8
  ```
- Switch models by editing `.env` (no code changes); both the web server and the
  CLI (`chatbot.py`) read it automatically. Restart the server to apply.

> Note: This setup uses a proxy (`ANTHROPIC_BASE_URL`). If the proxy lacks the
> chosen model, calls fall back to GPT-4o-mini.

---

## 3. Quiz / Flashcards JSON Fix (`llm.py`)

### Problem
After switching to Opus, quiz/flashcard generation failed with
`No JSON object found` — the model returned a prose explanation instead of JSON.

### Fix (in `_structured()`)
- **Assistant prefill** — seed the assistant reply with `{` so the model is forced
  to continue a JSON object instead of writing prose. The leading `{` is
  re-attached to the streamed result before parsing.
- **Stronger instruction** — the user message now explicitly requires the reply to
  start with `{` and end with `}`, with no preamble or markdown fences.
- **OpenAI fallback** — kept clean (no prefill); continues to use its native
  `response_format: json_object`.
- Applies to both quiz and flashcards (shared function).

---

## 4. Design Redesign — Added then Reverted

### Added (commit `ac90f85`) — later reverted
- A full "Booklight" visual redesign of `landing.html`, `index.html`, `admin.html`
  (new type stack, shared design tokens, custom SVG art, aurora accents).

### Reverted (commit `3ea98f6`)
- Restored all three pages to their pre-redesign state (from commit `4269349`).

---

## 5. Cloud Deployment (Render) — Added then Reverted

### Added (commits `56548b3`, `1e44f92`) — later reverted
- `render.yaml` (free web service, Python 3.11, binds `$PORT`)
- `DEPLOY.md` (step-by-step Render guide; free URL, no domain needed)
- Pinned `requirements.txt`; un-ignored and committed `books/` + `db/`

### Reverted (commit `3ea98f6`)
- Removed `render.yaml` and `DEPLOY.md`
- Restored original `.gitignore` and `requirements.txt`
- Un-tracked `books/` and `db/` from git **without deleting them from disk**
  (they're back to being gitignored, as before)

> Note: `books/` and `db/` still exist in earlier git history; only current
> tracking was removed.

---

## 6. Git / Repo

- Set git identity: `Prathamesh` / `prathamhere09@gmail.com`
- Remote updated to: `https://github.com/Oopsdevs/vernasolver.git`
  (use `git remote set-url origin <url>` when "origin already exists")
- Render rejects a `pythonVersion:` field on a service — set Python via the
  `PYTHON_VERSION` env var instead.

---

## 7. Team Details (saved to memory)

VernaSolver — 3-person CSE internship project (1 June – 1 July 2026):

| Member | Enrollment No. | Role | ~% |
|---|---|---|---|
| Pritam Bairagi | ADT24SOCB0846 | Backend, RAG, LLM, architecture | ~45% |
| Shwetaki Killedar | ADT24SOCB1158 | Frontend, streaming UI, UX, voice | ~30% |
| Preksha Pokharna | ADT24SOCB0839 | PDF processing, ingestion, testing, docs | ~25% |

---

## Commit history (this session)

```
3ea98f6  Revert Booklight redesign and Render deployment setup
1e44f92  Fix render.yaml: drop invalid pythonVersion field
56548b3  Add Render deployment config and commit ingested data
ac90f85  Redesign all pages with the Booklight design system
4269349  Add SETUP.bat — one-click installer for all dependencies
6fa03ee  Rebrand to VernaSolver with full feature overhaul
```

> Uncommitted at time of writing: `llm.py`, `.env`, `.env.example` (model config
> + JSON fix). Commit when ready.
