from __future__ import annotations
import hashlib
import json
from math import gcd
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

TARGET_SR = 16000


def load_corpus(corpus_dir, name: str):
    corpus_dir = Path(corpus_dir)
    meta = json.loads((corpus_dir / f"{name}.meta.json").read_text(encoding="utf-8"))
    rows = [json.loads(l) for l in (corpus_dir / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    roots = {ds: rec["root"] for ds, rec in meta["datasets"].items()}
    return meta, rows, roots


def split_by_speaker(rows: list[dict], val_pct: int):
    train, val = [], []
    for r in rows:
        key = r.get("speaker_id") or r["sample_id"]
        bucket = int(hashlib.sha256(key.encode("utf-8")).hexdigest(), 16) % 100
        (val if bucket < val_pct else train).append(r)
    return train, val


def read_audio(path: Path) -> np.ndarray:
    x, sr = sf.read(str(path), dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != TARGET_SR:
        g = gcd(sr, TARGET_SR)
        x = resample_poly(x, TARGET_SR // g, sr // g).astype("float32")
    return x


class SpeechDataset(Dataset):
    def __init__(self, rows, roots, tokenizer, max_seconds: float = 30.0):
        self.rows = [r for r in rows if r["duration"] <= max_seconds]
        self.roots = roots
        self.tok = tokenizer

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        wave = read_audio(Path(self.roots[r["dataset_id"]]) / r["path"])
        return torch.from_numpy(wave), torch.tensor(self.tok.encode(r["transcript"]), dtype=torch.long), r["transcript"]


def collate(batch):
    waves, targets, texts = zip(*batch)
    wave_lengths = torch.tensor([len(w) for w in waves])
    padded = torch.zeros(len(waves), int(wave_lengths.max()))
    for i, w in enumerate(waves):
        padded[i, : len(w)] = w
    return padded, wave_lengths, torch.cat(targets), torch.tensor([len(t) for t in targets]), list(texts)
