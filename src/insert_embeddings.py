import json
import os
import uuid

from db import qdrant_client
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm


def load_embeddings_and_transformer():
    data_dir = "../data"
    for filename in os.listdir(data_dir):
        with open(os.path.join(data_dir, filename)) as f:
            embedding_data = json.load(f)

        if qdrant_client.collection_exists(embedding_data["collection_name"]):
            qdrant_client.delete_collection(embedding_data["collection_name"])

        qdrant_client.create_collection(
            collection_name=embedding_data["collection_name"],
            vectors_config=VectorParams(
                size=embedding_data["vector_size"], distance=Distance.COSINE
            ),
        )
        points = [
            PointStruct(id=str(uuid.uuid4()), **embedding_item)
            for embedding_item in embedding_data["data"]
        ]
        for k in tqdm(range(0, len(points), 200)):
            qdrant_client.upsert(
                collection_name=embedding_data["collection_name"],
                points=points[k : k + 200],
            )


if __name__ == "__main__":
    load_embeddings_and_transformer()
