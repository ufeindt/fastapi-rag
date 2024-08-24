import json
import os
import uuid
from hashlib import md5
from typing import Generator, Tuple

import typer
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from transformer import TRANSFORMER_MODEL
from typing_extensions import Annotated

app = typer.Typer()


def get_markdown_file_contents(
    directory: str,
) -> Generator[Tuple[str, str], None, None]:
    markdown_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".md"):
                markdown_files.append(os.path.join(root, file))

    for markdown_file in markdown_files:
        with open(markdown_file, "r") as f:
            yield markdown_file, f.read()


def split_recursive(text: str, max_len: int, separator: str) -> list[str]:
    if len(text) < max_len:
        return [text]
    else:
        half_len = len(text) // 2
        right_text = text[half_len:]
        left_text = text[:half_len]

        try:
            left_index = left_text.rindex(separator)
        except ValueError:
            left_index = None

        try:
            right_index = right_text.index(separator)
        except ValueError:
            right_index = None

        # No split found.
        if not left_index and not right_index:
            return [text]

        # Pick index closer to the middle of the original text.
        if not right_index or (
            left_index and (len(left_text) - left_index) < right_index
        ):
            split_left = left_text[:left_index]
            split_right = left_text[left_index:] + right_text
        else:
            split_left = left_text + right_text[:right_index]
            split_right = right_text[right_index:]

        # Prevent tiny splits.
        if len(split_left) < 2 or len(split_right) < 2:
            return [text]

        return split_recursive(
            split_left.strip(), max_len, separator
        ) + split_recursive(split_right.strip(), max_len, separator)


def split_markdown(text: str, max_len: int) -> list[str]:
    separators = [f"\n{"#"*k} " for k in range(1, 7)] + ["\n", ""]

    split_text = [text]
    for separator in separators:
        split_text = [
            item
            for old_split in split_text
            for item in split_recursive(old_split, max_len, separator)
        ]

    return split_text


def main(
    data_dir: Annotated[
        str, typer.Argument(envvar="DATA_DIR", help="Path to data directory.")
    ],
    subdirectory: Annotated[
        str, typer.Argument(help="Subdirectory of DATA_DIR to parse.")
    ] = "dnd-5e-srd",
    cuda: Annotated[
        bool, typer.Option(help="Use CUDA when computing embeddings.")
    ] = False,
) -> None:
    if cuda:
        transformer = SentenceTransformer(TRANSFORMER_MODEL, device="cuda")
    else:
        transformer = SentenceTransformer(TRANSFORMER_MODEL)

    embeddings = {
        "collection_name": subdirectory,
        "vector_size": transformer.get_sentence_embedding_dimension(),
        "data": [],
    }

    directory = os.path.join(data_dir, subdirectory)
    for filename, text in get_markdown_file_contents(directory):
        split_text = split_markdown(text, 1024)
        for k, item in enumerate(split_text):
            embeddings["data"].append(
                {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "filename": filename,
                        "file_part": k + 1,
                        "file_parts": len(split_text),
                        "md5": md5(item.encode("utf-8")).hexdigest(),
                        "text": item,
                    },
                }
            )

    for split_data in tqdm(embeddings["data"]):
        split_data["vector"] = transformer.encode(
            split_data["payload"]["text"]
        ).tolist()

    with open(os.path.join(data_dir, f"{subdirectory}-embeddings.json"), "w") as f:
        json.dump(embeddings, f)


if __name__ == "__main__":
    typer.run(main)
