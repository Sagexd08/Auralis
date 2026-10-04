from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import soundfile as sf
from ..provenance import DatasetRecord, LicenseState, SampleRecord

KNOWN_LICENCES = {
    "cc0-1.0": (LicenseState.ELIGIBLE, True, True),
    "cc-by-4.0": (LicenseState.ELIGIBLE, True, True),
    "cc-by-3.0": (LicenseState.ELIGIBLE, True, True),
    "cc-by-sa-4.0": (LicenseState.ELIGIBLE, True, True),
    "mit": (LicenseState.ELIGIBLE, True, True),
    "apache-2.0": (LicenseState.ELIGIBLE, True, True),
    "cc-by-nc-4.0": (LicenseState.NONCOMMERCIAL_ONLY, False, False),
    "cc-by-nc-sa-4.0": (LicenseState.NONCOMMERCIAL_ONLY, False, False),
}


def classify_licence(spdx: str) -> tuple[LicenseState, bool, bool]:
    return KNOWN_LICENCES.get(spdx.strip().lower(), (LicenseState.UNKNOWN, False, False))


def build_record(dataset_id, source, revision, licence, root, languages, hours, checksum="") -> DatasetRecord:
    state, commercial, redistribute = classify_licence(licence)
    return DatasetRecord(
        dataset_id=dataset_id,
        source=source,
        revision=revision,
        license=licence,
        state=state,
        training_allowed=state != LicenseState.UNKNOWN,
        commercial_allowed=commercial,
        redistribution_allowed=redistribute,
        languages=sorted(languages),
        hours=round(hours, 4),
        downloaded_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        checksum=checksum,
        root=str(Path(root).resolve().as_posix()),
    )


def score_audio(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    x = np.asarray(x, dtype=np.float64)
    score = 1.0
    if np.mean(np.abs(x) >= 0.999) > 0.005:
        score *= 0.5
    if float(np.sqrt(np.mean(x * x))) < 1e-3:
        score *= 0.2
    return score


def probe(path: Path, score: bool) -> tuple[float, int, float]:
    info = sf.info(str(path))
    quality = 1.0
    if score:
        data, _ = sf.read(str(path), dtype="float32", always_2d=False)
        if data.ndim > 1:
            data = data.mean(axis=1)
        quality = score_audio(data)
    return info.frames / info.samplerate, info.samplerate, quality


def write_manifest(manifest_dir, dataset_id: str, samples: list[SampleRecord]) -> Path:
    manifest_dir = Path(manifest_dir)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    out = manifest_dir / f"{dataset_id}.jsonl"
    rows = sorted(samples, key=lambda s: s.sample_id)
    out.write_text("".join(json.dumps(asdict(s), sort_keys=True, ensure_ascii=False) + "\n" for s in rows), encoding="utf-8")
    return out
