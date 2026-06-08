import asyncio
import hashlib
import json
import uuid
from pathlib import Path

from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, Request, Response, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

import registry
import users
from config import BOOKS_DIR, IMAGES_DIR, MAX_HISTORY_TURNS
from ingest import ingest_pdf, remove_book
from llm import determine_model, generate_flashcards, generate_quiz, is_small_talk, stream_answer, stream_smalltalk
from query import contextualize_query, format_context, search

app = FastAPI(title="VernaSolver")

sessions: dict[str, list[dict]] = {}   # session_id -> message history
jobs: dict[str, dict] = {}             # job_id -> ingest progress

app.mount("/static", StaticFiles(directory="static"), name="static")


# ── Pages ─────────────────────────────────────────────────────────────────────

@app.get("/")
def landing():
    return FileResponse("static/landing.html")

@app.get("/app")
def app_page():
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

def _open_pdf_for(book_id: str):
    book = registry.find(book_id)
    if not book or not book.get("pdf_path"):
        raise HTTPException(404, "Book or PDF not found — re-ingest the book")
    pdf_path = book["pdf_path"]
    if not Path(pdf_path).exists():
        raise HTTPException(404, "PDF file missing — re-ingest the book")
    import fitz
    return fitz.open(pdf_path)


def _highlight_snippet(page, snippet: str) -> None:
    """Find phrases from the snippet on the page and add yellow highlight annotations."""
    words = snippet.split()
    if len(words) < 3:
        return

    phrase_len, step = 6, 4
    seen_rects: set[tuple] = set()

    for i in range(0, max(1, len(words) - phrase_len + 1), step):
        phrase = " ".join(words[i:i + phrase_len])
        if len(phrase) < 12:
            continue
        try:
            for rect in page.search_for(phrase, quads=False):
                key = (round(rect.x0, 1), round(rect.y0, 1), round(rect.x1, 1), round(rect.y1, 1))
                if key in seen_rects:
                    continue
                seen_rects.add(key)
                page.add_highlight_annot(rect)
        except Exception:
            pass


@app.get("/api/page/{book_id}/{page_num}.png")
def render_page(book_id: str, page_num: int):
    """Render a PDF page as PNG on demand (no highlight). Cached."""
    cache_dir = IMAGES_DIR / book_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"p{page_num}.png"

    if not cache_path.exists():
        import fitz
        doc = _open_pdf_for(book_id)
        if not (1 <= page_num <= len(doc)):
            doc.close()
            raise HTTPException(404, "Page out of range")
        pix = doc[page_num - 1].get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
        pix.save(str(cache_path))
        doc.close()

    return FileResponse(str(cache_path), media_type="image/png",
                        headers={"Cache-Control": "public, max-age=86400"})


class HighlightRequest(BaseModel):
    snippet: str


@app.post("/api/page-highlighted/{book_id}/{page_num}")
def render_page_highlighted(book_id: str, page_num: int, req: HighlightRequest):
    """Render a PDF page as PNG with the snippet text highlighted. Cached per (page, snippet hash)."""
    snippet = req.snippet or ""
    snippet_hash = hashlib.md5(snippet.encode("utf-8")).hexdigest()[:12]

    cache_dir = IMAGES_DIR / book_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"p{page_num}_h{snippet_hash}.png"

    if not cache_path.exists():
        import fitz
        doc = _open_pdf_for(book_id)
        if not (1 <= page_num <= len(doc)):
            doc.close()
            raise HTTPException(404, "Page out of range")
        page = doc[page_num - 1]
        _highlight_snippet(page, snippet)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), colorspace=fitz.csRGB)
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


# ── Auth ──────────────────────────────────────────────────────────────────────

SESSION_COOKIE = "vs_session"
SESSION_MAX_AGE = 30 * 24 * 3600   # 30 days

class SignUpRequest(BaseModel):
    email: str
    name: str
    password: str

class SignInRequest(BaseModel):
    email: str
    password: str


def _set_session_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        SESSION_COOKIE, token, max_age=SESSION_MAX_AGE,
        httponly=True, samesite="lax", path="/",
    )


@app.post("/api/auth/signup")
def auth_signup(req: SignUpRequest, response: Response):
    try:
        user = users.create_user(req.email, req.name, req.password)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception:
        raise HTTPException(409, "An account with that email already exists")
    token = users.create_session(user["id"])
    _set_session_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/signin")
def auth_signin(req: SignInRequest, response: Response):
    user = users.authenticate(req.email, req.password)
    if not user:
        raise HTTPException(401, "Invalid email or password")
    token = users.create_session(user["id"])
    _set_session_cookie(response, token)
    return {"user": user}


@app.post("/api/auth/signout")
def auth_signout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    users.delete_session(token)
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"success": True}


@app.get("/api/auth/me")
def auth_me(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    user = users.get_user_by_token(token) if token else None
    return {"user": user}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    session_id: str | None = None
    question: str
    subject: str
    book_id: str | None = None
    history: list[dict] | None = None  # if client passes its own history, use it directly
    eli5: bool = False                  # explain-like-I'm-in-6th-grade mode


class StudyRequest(BaseModel):
    topic: str
    subject: str
    book_id: str | None = None

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = req.history if req.history is not None else sessions.get(session_id, [])
    model = determine_model()

    def _sse(payload: dict) -> str:
        return f"data: {json.dumps(payload)}\n\n"

    # ── Small-talk fast path — no book search, no citations ───────────────
    if is_small_talk(req.question):
        async def smalltalk_stream():
            q: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def produce():
                try:
                    for token in stream_smalltalk(req.question, history=history):
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

            yield _sse({"type": "done", "sources": [], "session_id": session_id, "model": model})

        return StreamingResponse(
            smalltalk_stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

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

    # Wrap stream_answer so we can pass eli5 through to the producer below
    def _stream_with_mode():
        yield from stream_answer(context, req.question, history=history, eli5=req.eli5)

    async def event_stream():
        if not chunks:
            yield _sse({"type": "token", "content": "I couldn't find relevant content for that question. Try rephrasing or selecting a different scope."})
            yield _sse({"type": "done", "sources": [], "session_id": session_id, "model": "N/A"})
            return

        q: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def produce():
            try:
                for token in _stream_with_mode():
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


# ── Quiz / Flashcards ────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict:
    """Strip optional ```json fences and parse robustly."""
    import json as _json
    import re as _re
    if not text or not text.strip():
        raise ValueError("LLM returned an empty response")
    cleaned = text.strip()
    cleaned = _re.sub(r"^```(?:json)?\s*", "", cleaned, flags=_re.IGNORECASE)
    cleaned = _re.sub(r"\s*```\s*$", "", cleaned)
    cleaned = cleaned.strip()
    # Extract the outermost JSON object
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError(f"No JSON object found. Response starts with: {cleaned[:160]!r}")
    return _json.loads(cleaned[start:end + 1])


def _gather_study_context(topic: str, subject: str, book_id: str | None) -> tuple[str, list[dict]]:
    """Shared helper: pull top chunks for the topic and return (context, sources)."""
    chunks = search(topic, subject, book_id=book_id)
    if not chunks:
        raise HTTPException(404, "No relevant content found for that topic in the selected book(s).")
    sources = []
    seen: set[str] = set()
    for c in chunks:
        key = f"{c['title']}|{c['page']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "title": c["title"], "author": c["author"], "page": c["page"],
                "book_id": c["book_id"], "snippet": c["text"],
                "has_visuals": c.get("has_visuals", False),
            })
    return format_context(chunks), sources


@app.post("/api/quiz")
def make_quiz(req: StudyRequest):
    context, sources = _gather_study_context(req.topic, req.subject, req.book_id)
    raw = generate_quiz(context, req.topic)
    try:
        data = _parse_json_response(raw)
    except Exception as e:
        print(f"[quiz parse error] raw response: {raw!r}")
        raise HTTPException(500, f"Quiz generation failed: {e}")
    return {"quiz": data, "sources": sources, "model": determine_model()}


@app.post("/api/flashcards")
def make_flashcards(req: StudyRequest):
    context, sources = _gather_study_context(req.topic, req.subject, req.book_id)
    raw = generate_flashcards(context, req.topic)
    try:
        data = _parse_json_response(raw)
    except Exception as e:
        print(f"[flashcards parse error] raw response: {raw!r}")
        raise HTTPException(500, f"Flashcards generation failed: {e}")
    return {"flashcards": data, "sources": sources, "model": determine_model()}

@app.delete("/api/sessions/{session_id}")
def clear_session(session_id: str):
    sessions.pop(session_id, None)
    return {"success": True}
