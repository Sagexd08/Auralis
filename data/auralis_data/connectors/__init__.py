from __future__ import annotations
from pathlib import Path
from ..provenance import DatasetRecord
from .base import build_record, classify_licence, score_audio, write_manifest
from .readers import READERS

DEFAULT_LICENCES = {"librispeech": "CC-BY-4.0", "commonvoice": "CC0-1.0", "fleurs": "CC-BY-4.0"}


def import_dataset(kind, root, dataset_id, language, revision, manifest_dir, licence="", score=False, source=None, checksum=""):
    if kind not in READERS:
        raise ValueError(f"unknown importer {kind!r}; choose from {sorted(READERS)}")
    root = Path(root)
    licence = licence or DEFAULT_LICENCES.get(kind, "")
    samples = READERS[kind](root, language, dataset_id, score)
    hours = sum(s.duration for s in samples) / 3600
    record = build_record(dataset_id, source or kind, revision, licence, root, {s.language for s in samples} or {language}, hours, checksum)
    write_manifest(manifest_dir, dataset_id, samples)
    return record, samples


__all__ = ["import_dataset", "build_record", "classify_licence", "score_audio", "write_manifest", "READERS", "DEFAULT_LICENCES"]
