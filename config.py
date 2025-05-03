import os

# Model configurations
MODELS_DIR = "models"
EMBEDDING_MODEL = os.path.join(MODELS_DIR, "ruri-large")
RERANKER_MODEL = os.path.join(MODELS_DIR, "ruri-reranker-large")
LLM_MODEL = os.path.join(MODELS_DIR, "Qwen-VL")

# ChromaDB settings
CHROMA_PERSIST_DIR = "chroma_db"
CHROMA_COLLECTION_NAME = "japanese_documents"

# RAG settings
TOP_K_RETRIEVAL = 5
TOP_N_RERANK = 3

# Cache settings
CACHE_DIR = "cache" 