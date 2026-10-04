from __future__ import annotations
import csv
import json
from pathlib import Path
from ..provenance import SampleRecord
from .base import probe


def read_local(root: Path, language: str, dataset_id: str, score: bool) -> list[SampleRecord]:
    meta = root / "metadata.jsonl"
    out = []
    for line in meta.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rel = row["path"]
        duration, rate, quality = probe(root / rel, score)
        out.append(SampleRecord(
            sample_id=f"{dataset_id}-{Path(rel).stem}",
            dataset_id=dataset_id,
            path=rel,
            transcript=row["transcript"],
            language=row.get("language", language),
            duration=duration,
            speaker_id=row.get("speaker_id", ""),
            quality=quality,
            sample_rate=rate,
        ))
    return out


def read_librispeech(root: Path, language: str, dataset_id: str, score: bool) -> list[SampleRecord]:
    out = []
    for trans in sorted(root.rglob("*.trans.txt")):
        chapter = trans.parent
        speaker = chapter.parent.name
        for line in trans.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            utt, text = line.split(" ", 1)
            audio = chapter / f"{utt}.flac"
            if not audio.exists():
                continue
            duration, rate, quality = probe(audio, score)
            out.append(SampleRecord(
                sample_id=f"{dataset_id}-{utt}",
                dataset_id=dataset_id,
                path=audio.relative_to(root).as_posix(),
                transcript=text.strip().lower(),
                language=language,
                duration=duration,
                speaker_id=speaker,
                quality=quality,
                sample_rate=rate,
            ))
    return out


def read_commonvoice(root: Path, language: str, dataset_id: str, score: bool, table: str = "validated.tsv") -> list[SampleRecord]:
    durations = {}
    dur_file = root / "clip_durations.tsv"
    if dur_file.exists():
        with dur_file.open(encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                durations[row["clip"]] = float(row["duration[ms]"]) / 1000.0
    out = []
    with (root / table).open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            rel = "clips/" + row["path"]
            clip = root / rel
            if not clip.exists():
                continue
            if row["path"] in durations and not score:
                duration, rate, quality = durations[row["path"]], 16000, 1.0
            else:
                duration, rate, quality = probe(clip, score)
            out.append(SampleRecord(
                sample_id=f"{dataset_id}-{Path(row['path']).stem}",
                dataset_id=dataset_id,
                path=rel,
                transcript=row["sentence"].strip(),
                language=language,
                duration=duration,
                speaker_id=row.get("client_id", "")[:16],
                quality=quality,
                sample_rate=rate,
            ))
    return out


def read_fleurs(root: Path, language: str, dataset_id: str, score: bool, split: str = "train") -> list[SampleRecord]:
    out = []
    with (root / f"{split}.tsv").open(encoding="utf-8", newline="") as f:
        for row in csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            file_name, transcript, num_samples = row[1], row[3], int(row[-2]) if row[-2].isdigit() else 0
            rel = f"audio/{split}/{file_name}"
            clip = root / rel
            if not clip.exists():
                continue
            if score or num_samples == 0:
                duration, rate, quality = probe(clip, score)
            else:
                duration, rate, quality = num_samples / 16000.0, 16000, 1.0
            out.append(SampleRecord(
                sample_id=f"{dataset_id}-{Path(file_name).stem}",
                dataset_id=dataset_id,
                path=rel,
                transcript=transcript.strip(),
                language=language,
                duration=duration,
                speaker_id="",
                quality=quality,
                sample_rate=rate,
            ))
    return out


READERS = {
    "local": read_local,
    "librispeech": read_librispeech,
    "commonvoice": read_commonvoice,
    "fleurs": read_fleurs,
}
