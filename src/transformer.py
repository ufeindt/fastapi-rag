from sentence_transformers import SentenceTransformer

TRANSFORMER_MODEL = "sentence-transformers/multi-qa-mpnet-base-dot-v1"

transformer = SentenceTransformer(TRANSFORMER_MODEL)
