import asyncio
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

import registry
from config import BOOKS_DIR, IMAGES_DIR, MAX_HISTORY_TURNS
from ingest import ingest_pdf, remove_book
from llm import determine_model, stream_answer
from query import contextualize_query, format_context, search

app = FastAPI(title="BookBot")

sessions: dict[str, list[dict]] = {}   # session_id -> message history
jobs: dict[str, dict] = {}             # job_id -> ingest progress

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/admin")
def admin_page():
    return FileResponse("static/admin.html")


# ── Subjects & Books ──────────────────────────────────────────────────────────

@app.get("/api/subjects")
def get_subjects():
    return {"subjects": registry.all_subjects()}

@app.get("/api/books")
def get_books(subject: str | None = None):
    books = registry.by_subject(subject) if subject else registry.load()
    return {"books": books}

@app.get("/api/page/{book_id}/{page_num}.png")
def render_page(book_id: str, page_num: int):
    """Render a PDF page as PNG on demand; cached to static/book_images/."""
    book = registry.find(book_id)
    if not book or not book.get("pdf_path"):
        raise HTTPException(404, "Book or PDF not found — re-ingest to enable diagrams")

    pdf_path = book["pdf_path"]
    if not Path(pdf_path).exists():
        raise HTTPException(404, "PDF file missing — re-ingest to enable diagrams")

    cache_dir = IMAGES_DIR / book_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"p{page_num}.png"

    if not cache_path.exists():
        import fitz
        doc = fitz.open(pdf_path)
        if not (1 <= page_num <= len(doc)):
            raise HTTPException(404, "Page out of range")
        pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
        pix.save(str(cache_path))
        doc.close()

    return FileResponse(str(cache_path), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


@app.delete("/api/books/{book_id}")
def delete_book(book_id: str):
    if not registry.find(book_id):
        raise HTTPException(status_code=404, detail="Book not found")
    remove_book(book_id)
    return {"success": True}


# ── Ingest ────────────────────────────────────────────────────────────────────

def _run_ingest(job_id: str, pdf_path: str, subject: str, title: str, author: str):
    try:
        jobs[job_id]["message"] = "Extracting pages and building chunks..."
        ingest_pdf(pdf_path, subject, title, author)
        jobs[job_id] = {"status": "done", "message": f"'{title}' ingested successfully."}
    except Exception as e:
        jobs[job_id] = {"status": "error", "message": str(e)}

@app.post("/api/ingest")
async def start_ingest(
    background_tasks: BackgroundTasks,
    file: UploadFile,
    subject: str = Form(...),
    title: str = Form(...),
    author: str = Form(...),
):
    content = await file.read()
    pdf_path = BOOKS_DIR / file.filename
    with open(pdf_path, "wb") as f:
        f.write(content)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "running", "message": "File received, starting ingestion..."}
    background_tasks.add_task(_run_ingest, job_id, str(pdf_path), subject, title, author)
    return {"job_id": job_id}

@app.get("/api/ingest/{job_id}")
def ingest_status(job_id: str):
    return jobs.get(job_id, {"status": "not_found", "message": "Job not found"})


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str
    subject: str
    book_id: str | None = None

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.get(session_id, [])

    search_query = contextualize_query(req.question, history)
    chunks = search(search_query, req.subject, book_id=req.book_id)

    seen: set[str] = set()
    sources = []
    for c in chunks:
        key = f"{c['title']}|{c['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": c["title"],
                "author": c["author"],
                "page": c["page"],
                "book_id": c["book_id"],
                "snippet": c["text"],
                "has_visuals": c.get("has_visuals", False),
            })

    context = format_context(chunks) if chunks else ""
    model = determine_model()

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    async def event_stream():
        if not chunks:
            yield _sse({"type": "token", "content": "I couldn't find relevant content for that question. Try rephrasing or selecting a different scope."})
            yield _sse({"type": "done", "sources": [], "session_id": session_id, "model": "N/A"})
            return

        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for token in stream_answer(context, req.question, history=history):
                    loop.call_soon_threadsafe(q.put_nowait, ("tok", token))
            except Exception as e:
                loop.call_soon_threadsafe(q.put_nowait, ("err", str(e)))
            loop.call_soon_threadsafe(q.put_nowait, ("end", None))

        loop.run_in_executor(None, produce)

        full: list[str] = []
        while True:
            kind, val = await q.get()
            if kind == "tok":
                full.append(val)
                yield _sse({"type": "token", "content": val})
            elif kind == "err":
                yield _sse({"type": "error", "message": val})
                return
            else:
                break

        answer = "".join(full)
        new_history = history + [
            {"role": "user", "content": req.question},
            {"role": "assistant", "content": answer},
        ]
        if len(new_history) > MAX_HISTORY_TURNS * 2:
            new_history = new_history[-(MAX_HISTORY_TURNS * 2):]
        sessions[session_id] = new_history

        yield _sse({"type": "done", "sources": sources, "session_id": session_id, "model": model})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"success": True}
