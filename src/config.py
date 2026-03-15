import os
from pathlib import Path
from dotenv import load_dotenv

# Loading environment variables from the project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


# ===============================
# LLM provider configuration
# ===============================

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

# Groq provider settings
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# Ollama settings (used as a local fallback provider)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")


# ===============================
# Embedding model configuration
# ===============================

EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2",
)


# ===============================
# Chunking configuration
# ===============================

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))


# ===============================
# Retrieval configuration
# ===============================

TOP_K = int(os.getenv("TOP_K", 5))

# Limiting how many chunks are allowed inside the prompt context
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", 4))


# ===============================
# Safety controls
# ===============================

MIN_RETRIEVAL_RESULTS = int(os.getenv("MIN_RETRIEVAL_RESULTS", 1))

ENABLE_LOW_SUPPORT_WARNING = (
    os.getenv("ENABLE_LOW_SUPPORT_WARNING", "true").lower() == "true"
)


# ===============================
# Project directories
# ===============================

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

# Ensuring the artifacts directory exists before writing index files
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


# ===============================
# Index storage locations
# ===============================

FAISS_INDEX_FILE = ARTIFACTS_DIR / "faiss_index.bin"
CHUNK_METADATA_FILE = ARTIFACTS_DIR / "chunk_metadata.json"
CITATION_METADATA_FILE = ARTIFACTS_DIR / "citation_metadata.json"