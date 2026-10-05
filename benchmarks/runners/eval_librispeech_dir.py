import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from benchmarks.metrics.wer import compute_metrics


def collect(root: Path, limit: int, stride: int):
    items = []
    for trans in sorted(root.rglob("*.trans.txt")):
        for line in trans.read_text(encoding="utf-8").splitlines():
            utt_id, _, text = line.partition(" ")
            flac = trans.parent / f"{utt_id}.flac"
            if flac.exists():
                items.append((flac, text))
    items = items[::stride]
    return items[:limit] if limit else items


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--cli", default="target/release/auralis-bench-cli.exe")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default="")
    a = ap.parse_args(argv)

    items = collect(Path(a.root), a.limit, a.stride)
    with tempfile.TemporaryDirectory() as tmp:
        wavs, refs, seconds = [], [], 0.0
        for i, (flac, text) in enumerate(items):
            audio, sr = sf.read(str(flac), dtype="float32")
            wav = Path(tmp) / f"{i}.wav"
            sf.write(str(wav), audio, sr, subtype="FLOAT")
            wavs.append(str(wav))
            refs.append(text)
            seconds += len(audio) / sr
        listing = Path(tmp) / "list.txt"
        listing.write_text("\n".join(wavs), encoding="utf-8")

        start = time.time()
        proc = subprocess.run([str(Path(a.cli).resolve()), "--model", str(Path(a.model).resolve()), "--batch", str(listing)], capture_output=True, text=True, encoding="utf-8", check=True)
        elapsed = time.time() - start

    hyps = {}
    for line in proc.stdout.splitlines():
        path, _, text = line.partition("\t")
        hyps[path] = text
    ordered = [hyps.get(w, "") for w in wavs]
    metrics = compute_metrics(refs, ordered, elapsed, seconds)
    result = {"model": a.model, "root": a.root, "clips": metrics.num_clips, "audio_hours": round(seconds / 3600, 3), "wer": metrics.wer, "cer": metrics.cer, "rtf_including_load": metrics.rtf}
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
