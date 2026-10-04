from __future__ import annotations
import hashlib
import subprocess
import tarfile
import urllib.request
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def pull_openslr(resource: str, name: str, dest) -> tuple[Path, str]:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.tar.gz"
    urllib.request.urlretrieve(f"https://www.openslr.org/resources/{resource}/{name}.tar.gz", archive)
    checksum = sha256_file(archive)
    with tarfile.open(archive) as tar:
        tar.extractall(dest, filter="data")
    return dest, checksum


def pull_huggingface(repo_id: str, revision: str, dest) -> tuple[Path, str, str]:
    from huggingface_hub import HfApi, snapshot_download

    info = HfApi().dataset_info(repo_id, revision=revision)
    licence = ""
    if info.card_data is not None:
        raw = getattr(info.card_data, "license", None)
        licence = raw[0] if isinstance(raw, list) and raw else (raw or "")
    path = snapshot_download(repo_id=repo_id, repo_type="dataset", revision=info.sha, local_dir=str(dest))
    return Path(path), info.sha, licence


def pull_kaggle(slug: str, dest) -> Path:
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"], check=True)
    return dest
