from __future__ import annotations
import io
import json
import re
from math import gcd
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
DISALLOWED = re.compile(r"[^A-Z' ]")


def clean_transcript(text) -> str | None:
    cleaned = (text or "").upper().replace("’", "'").replace("`", "'")
    if DISALLOWED.search(cleaned):
        return None
    cleaned = " ".join(cleaned.split())
    return cleaned or None


def decode_audio(data: bytes):
    wave, rate = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    wave = wave.mean(axis=1)
    if rate != TARGET_SR:
        g = gcd(rate, TARGET_SR)
        wave = resample_poly(wave, TARGET_SR // g, rate // g).astype("float32")
    return wave


def convert_parquet(parquet_path, out_dir, audio_col: str, text_col: str, id_col: str, speaker_col: str | None = None, shard_tag: str = "", batch_size: int = 64) -> tuple[int, int]:
    out_dir = Path(out_dir)
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    columns = [audio_col, text_col, id_col] + ([speaker_col] if speaker_col else [])
    kept = skipped = 0
    with (out_dir / "metadata.jsonl").open("a", encoding="utf-8") as meta:
        for batch in pq.ParquetFile(str(parquet_path)).iter_batches(batch_size=batch_size, columns=columns):
            for row in batch.to_pylist():
                text = clean_transcript(row[text_col])
                audio = row[audio_col]
                if text is None or not audio or not audio.get("bytes"):
                    skipped += 1
                    continue
                try:
                    wave = decode_audio(audio["bytes"])
                except Exception:
                    skipped += 1
                    continue
                stem = re.sub(r"[^A-Za-z0-9_-]", "_", f"{shard_tag}{row[id_col]}")
                rel = f"audio/{stem}.flac"
                sf.write(str(out_dir / rel), wave, TARGET_SR, format="FLAC", subtype="PCM_16")
                entry = {"path": rel, "transcript": text}
                if speaker_col and row.get(speaker_col):
                    entry["speaker_id"] = str(row[speaker_col])
                meta.write(json.dumps(entry, ensure_ascii=False) + "\n")
                kept += 1
    return kept, skipped
