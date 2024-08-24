import os

from qdrant_client import QdrantClient

QDRANT_URL = os.environ.get("QDRANT_URL", "http://localhost:6333")
qdrant_db = QdrantClient(url=QDRANT_URL, timeout=60)
