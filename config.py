import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

RAW_BASES = [ROOT, ROOT / "hadith-json-main"]
RAW_BASES = [base for base in RAW_BASES if base.exists()]

RAW_SOURCE_DIRS = []
for base in RAW_BASES:
    RAW_SOURCE_DIRS.extend([
        base / "db" / "by_book" / "forties",
        base / "db" / "by_book" / "other_books",
        base / "db" / "by_book" / "the_9_books",
        base / "db" / "fatwas",
        base / "quran",
    ])

RAW_SOURCE_DIRS = [path for path in RAW_SOURCE_DIRS if path.exists()]
PROCESSED_DIR = ROOT / "processed"
PROCESSED_DIRS = {
    "hadith": PROCESSED_DIR / "hadith",
    "fatwas": PROCESSED_DIR / "fatwas",
    "quran": PROCESSED_DIR / "quran",
    "tafsir": PROCESSED_DIR / "tafsir",
}

PROCESSED_SOURCE_DIRS = list(PROCESSED_DIRS.values())

OUTPUT_PATHS = {
    "hadith": PROCESSED_DIRS["hadith"] / "hadith_unified.jsonl",
    "fatwas": PROCESSED_DIRS["fatwas"] / "fatwas_unified.jsonl",
    "quran": PROCESSED_DIRS["quran"] / "quran_unified.jsonl",
    "tafsir": PROCESSED_DIRS["tafsir"] / "tafsir_unified.jsonl",
}

INDEX_DIR = ROOT / "indexes"
LOG_DIR = ROOT / "logs"
INDEX_METADATA_FILE = INDEX_DIR / "metadata.json"
EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
FAISS_INDEX_FILE = INDEX_DIR / "faiss.index"
BM25_CACHE_FILE = INDEX_DIR / "bm25.pkl"

MODEL_NAME = "intfloat/multilingual-e5-base"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

DEBUG = False
SUBMISSION_MODE = str(os.getenv("SUBMISSION_MODE", "true")).lower() in {"1", "true", "yes", "on"}
ENABLE_RERANKER = str(os.getenv("ENABLE_RERANKER", "false")).lower() in {"1", "true", "yes", "on"} and not SUBMISSION_MODE
ENABLE_LLM_QUERY_OPTIMIZATION = str(os.getenv("ENABLE_LLM_QUERY_OPTIMIZATION", "false")).lower() in {"1", "true", "yes", "on"} and not SUBMISSION_MODE
ENABLE_LLM_VERIFICATION = str(os.getenv("ENABLE_LLM_VERIFICATION", "false")).lower() in {"1", "true", "yes", "on"} and not SUBMISSION_MODE
TOP_K = 5 if SUBMISSION_MODE else 10
DEFAULT_TOP_K = TOP_K
BATCH_SIZE = 64
MIN_BATCH_SIZE = 8
MAX_MEMORY_BYTES = 2_000_000_000
BM25_WEIGHT = 0.4
SEMANTIC_WEIGHT = 0.6
RE_RANK_TOP_K = 20
HYBRID_TOP_K = 50
PRIORITY_BOOST = 0.15
WEAK_SCORE_THRESHOLD = 0.15
CONCISE_OUTPUT = SUBMISSION_MODE

TEXT_SOURCE_PRIORITY = {
    "hukm": ["fatwa", "hadith"],
    "tafsir": ["tafsir", "quran"],
    "hadith": ["hadith"],
}

ARABIC_TERMINAL_MODELS = ["arabic_reshaper", "bidi.algorithm"]
