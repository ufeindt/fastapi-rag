import json
import os

import typer
from db import qdrant_db
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm import tqdm
from typing_extensions import Annotated

app = typer.Typer()


def main(data_dir: Annotated[str, typer.Argument(envvar="DATA_DIR")]) -> None:
    data_dir = "../data"
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue

        with open(os.path.join(data_dir, filename)) as f:
            embedding_data = json.load(f)

        if qdrant_db.collection_exists(embedding_data["collection_name"]):
            qdrant_db.delete_collection(embedding_data["collection_name"])

        qdrant_db.create_collection(
            collection_name=embedding_data["collection_name"],
            vectors_config=VectorParams(
                size=embedding_data["vector_size"], distance=Distance.COSINE
            ),
        )
        points = [
            PointStruct(**embedding_item) for embedding_item in embedding_data["data"]
        ]
        for k in tqdm(range(0, len(points), 200)):
            qdrant_db.upsert(
                collection_name=embedding_data["collection_name"],
                points=points[k : k + 200],
            )


if __name__ == "__main__":
    typer.run(main)
