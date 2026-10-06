"""Vector stores. Importing this package registers every backend."""

from rag_chatbot.stores import chroma, faiss, memory, pgvector, qdrant  # noqa: F401  (registration)
from rag_chatbot.stores.base import STORES, Hit, VectorStore, build_store
from rag_chatbot.stores.docstore import SQLiteDocStore

__all__ = ["STORES", "Hit", "SQLiteDocStore", "VectorStore", "build_store"]
