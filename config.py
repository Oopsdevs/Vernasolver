from pathlib import Path

BASE_DIR = Path(__file__).parent
BOOKS_DIR = BASE_DIR / "books"
DB_DIR = BASE_DIR / "db"
REGISTRY_FILE = DB_DIR / "books_registry.json"

CHUNK_SIZE = 400      # max words per chunk
CHUNK_OVERLAP = 40    # word overlap carried into next chunk
MIN_CHUNK_WORDS = 30

TOP_K_RETRIEVE = 12   # candidates fetched from vector DB
TOP_K_RERANK = 5      # kept after reranking (sent to LLM)

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

MAX_HISTORY_TURNS = 6  # message pairs kept in conversation context

IMAGES_DIR = BASE_DIR / "static" / "book_images"

BOOKS_DIR.mkdir(exist_ok=True)
DB_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
