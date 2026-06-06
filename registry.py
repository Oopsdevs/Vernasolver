import json
import re
from config import REGISTRY_FILE


def slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def make_book_id(subject: str, author: str, title: str) -> str:
    return f"{slugify(subject)}__{slugify(author)}__{slugify(title)}"


def load() -> list[dict]:
    if not REGISTRY_FILE.exists():
        return []
    with open(REGISTRY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save(books: list[dict]) -> None:
    with open(REGISTRY_FILE, "w", encoding="utf-8") as f:
        json.dump(books, f, indent=2)


def find(book_id: str) -> dict | None:
    return next((b for b in load() if b["book_id"] == book_id), None)


def add(entry: dict) -> None:
    books = load()
    books = [b for b in books if b["book_id"] != entry["book_id"]]
    books.append(entry)
    save(books)


def remove(book_id: str) -> bool:
    books = load()
    filtered = [b for b in books if b["book_id"] != book_id]
    if len(filtered) == len(books):
        return False
    save(filtered)
    return True


def by_subject(subject: str) -> list[dict]:
    return [b for b in load() if b["subject"].lower() == subject.lower()]


def all_subjects() -> list[str]:
    seen = set()
    result = []
    for b in load():
        s = b["subject"]
        if s not in seen:
            seen.add(s)
            result.append(s)
    return sorted(result)
