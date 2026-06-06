import re
import shutil
from pathlib import Path

import chromadb
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer

import registry
from config import BOOKS_DIR, CHUNK_OVERLAP, CHUNK_SIZE, DB_DIR, EMBED_MODEL, MIN_CHUNK_WORDS


def get_db():
    return chromadb.PersistentClient(path=str(DB_DIR))


def get_collection(db=None):
    if db is None:
        db = get_db()
    return db.get_or_create_collection("books", metadata={"hnsw:space": "cosine"})


_embedder = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        print(f"Loading embedding model '{EMBED_MODEL}' (downloads ~90 MB on first run)...")
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def page_has_visuals(page) -> bool:
    """Return True if the page likely contains a diagram, figure, or chart."""
    for img in page.get_images(full=True):
        if len(img) >= 4 and img[2] > 80 and img[3] > 80:
            return True
    # Vector drawings (UML, flowcharts, etc.) show up as paths
    if len(page.get_drawings()) > 20:
        return True
    return False


def chunk_text(text: str) -> list[str]:
    # Split on paragraph/section boundaries rather than blind word count.
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\n(?=[A-Z0-9•\-])", text) if p.strip()]

    chunks: list[str] = []
    current: list[str] = []

    for para in paragraphs:
        para_words = para.split()
        if not para_words:
            continue

        if current and len(current) + len(para_words) > CHUNK_SIZE:
            if len(current) >= MIN_CHUNK_WORDS:
                chunks.append(" ".join(current))
            # Carry overlap into next chunk so context isn't lost at boundaries.
            current = current[-CHUNK_OVERLAP:] + para_words
        else:
            current.extend(para_words)

    if len(current) >= MIN_CHUNK_WORDS:
        chunks.append(" ".join(current))

    return chunks


def ingest_pdf(pdf_path: str, subject: str, title: str, author: str) -> None:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    book_id = registry.make_book_id(subject, author, title)

    if registry.find(book_id):
        print(f"'{title}' is already ingested. Use `remove` first to re-ingest.")
        return

    # Keep a permanent copy of the PDF so we can render pages on demand later.
    stored_pdf = BOOKS_DIR / f"{book_id}.pdf"
    if path.resolve() != stored_pdf.resolve():
        shutil.copy2(path, stored_pdf)

    print(f"Opening: {path.name}")
    doc = fitz.open(str(stored_pdf))
    total_pages = len(doc)
    print(f"Extracting text from {total_pages} pages...")

    ids, documents, embeddings, metadatas = [], [], [], []
    embedder = get_embedder()

    for page_num in range(total_pages):
        page = doc[page_num]
        page_text = page.get_text("text").strip()
        has_visuals = page_has_visuals(page)
        if not page_text:
            continue
        for chunk_idx, chunk in enumerate(chunk_text(page_text)):
            chunk_id = f"{book_id}__p{page_num + 1}__c{chunk_idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "book_id": book_id,
                    "subject": subject,
                    "title": title,
                    "author": author,
                    "page": page_num + 1,
                    "has_visuals": has_visuals,
                }
            )

    doc.close()

    print(f"Embedding {len(ids)} chunks...")
    collection = get_collection()
    batch_size = 64
    for i in range(0, len(ids), batch_size):
        sl = slice(i, i + batch_size)
        batch_embeddings = embedder.encode(documents[sl], show_progress_bar=False).tolist()
        collection.add(
            ids=ids[sl],
            documents=documents[sl],
            embeddings=batch_embeddings,
            metadatas=metadatas[sl],
        )
        done = min(i + batch_size, len(ids))
        print(f"  {done}/{len(ids)} chunks stored...", end="\r")

    print()
    registry.add(
        {
            "book_id": book_id,
            "subject": subject,
            "title": title,
            "author": author,
            "pages": total_pages,
            "chunks": len(ids),
            "pdf_path": str(stored_pdf),
        }
    )
    print(f"Done! '{title}' by {author} — {len(ids)} chunks from {total_pages} pages.")


def remove_book(book_id: str) -> None:
    collection = get_collection()
    collection.delete(where={"book_id": book_id})
    registry.remove(book_id)
    print(f"Removed book '{book_id}' from the knowledge base.")
