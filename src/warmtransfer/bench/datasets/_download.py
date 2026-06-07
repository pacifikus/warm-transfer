"""Download and cache raw datasets in ``~/.warmtransfer/data``."""

from __future__ import annotations

import zipfile
from pathlib import Path

import requests

CACHE_ROOT = Path.home() / ".warmtransfer" / "data"


def cache_dir(name: str) -> Path:
    d = CACHE_ROOT / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def download(url: str, dest: Path, *, timeout: int = 120) -> Path:
    """Download ``url`` to ``dest`` (if not already present). Returns the path."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=1 << 16):
            f.write(chunk)
    tmp.rename(dest)
    return dest


def unzip(archive: Path, dest_dir: Path) -> Path:
    """Extract zip into ``dest_dir`` (if not already extracted). Returns ``dest_dir``."""
    marker = dest_dir / ".unzipped"
    if marker.exists():
        return dest_dir
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest_dir)
    marker.touch()
    return dest_dir
