import os
import shutil
import tarfile
import tempfile

import requests
import typer
from typing_extensions import Annotated

app = typer.Typer()


def download_dnd_5e_srd(url: str, tempfile_path: str):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(tempfile_path, "wb") as f:
            f.write(response.raw.read())


def extract_files(tempfile_path: str, target_dir: str):
    with tarfile.open(tempfile_path) as tar:
        for filename in tar.getnames():
            if not filename.endswith(".md"):
                continue

            file_split = filename.split("/")

            if len(file_split) < 3:
                continue
            if "(Alt)" in file_split[1]:
                continue

            tar.extract(filename, target_dir)

    # The subdirectories will be inside "DND.SRD.Wiki-0.5.1"
    out_dir = os.path.join(target_dir, "DND.SRD.Wiki-0.5.1")
    for subdir in os.listdir(out_dir):
        if os.path.exists(os.path.join(target_dir, subdir)):
            shutil.rmtree(os.path.join(target_dir, subdir))
        os.replace(os.path.join(out_dir, subdir), os.path.join(target_dir, subdir))
    os.rmdir(out_dir)


def main(data_dir: Annotated[str, typer.Argument(envvar="DATA_DIR")]) -> None:
    url = "https://github.com/OldManUmby/DND.SRD.Wiki/archive/refs/tags/v0.5.1.tar.gz"
    filename = "dnd-5e-srd-0.5.1.tar.gz"
    tempdir = tempfile.mkdtemp()
    tempfile_path = os.path.join(tempdir, filename)
    srd_dir = os.path.join(data_dir, "dnd-5e-srd")

    download_dnd_5e_srd(url, tempfile_path)
    extract_files(tempfile_path, srd_dir)


if __name__ == "__main__":
    typer.run(main)
