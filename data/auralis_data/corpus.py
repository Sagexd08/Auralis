from __future__ import annotations
import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from .gate import check_dataset
from .provenance import SampleRecord
from .registry import Registry

@dataclass
class CorpusConfig:
    name: str
    datasets: list[str]
    commercial: bool = False
    min_duration: float = 0.5
    max_duration: float = 30.0
    min_quality: float = 0.8
    language_weights: dict[str, float] = field(default_factory=dict)
    seed: int = 0

class LicenceGateError(RuntimeError):
    pass

def _load(manifest_dir: Path, dataset_id: str):
    for line in (manifest_dir / f"{dataset_id}.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield SampleRecord(**json.loads(line))

def build_corpus(cfg: CorpusConfig, registry: Registry, manifest_dir, out_dir):
    manifest_dir, out_dir = Path(manifest_dir), Path(out_dir)
    refusals = [d.reason for d in (check_dataset(registry.get(ds), cfg.commercial) for ds in cfg.datasets) if not d.allowed]
    if refusals:
        raise LicenceGateError("corpus refused by licence gate: " + "; ".join(refusals))

    seen, kept = set(), []
    for ds in sorted(cfg.datasets):
        for s in _load(manifest_dir, ds):
            if not (cfg.min_duration <= s.duration <= cfg.max_duration) or s.quality < cfg.min_quality:
                continue
            key = (s.language, " ".join(s.transcript.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            kept.append(s)
    kept.sort(key=lambda s: (s.dataset_id, s.sample_id))

    if cfg.language_weights:
        rng = random.Random(cfg.seed)
        by_lang: dict[str, list] = {}
        for s in kept:
            by_lang.setdefault(s.language, []).append(s)
        cap = min(len(v) / cfg.language_weights.get(lang, 1.0) for lang, v in by_lang.items())
        kept = []
        for lang in sorted(by_lang):
            v = by_lang[lang]
            rng.shuffle(v)
            kept += sorted(v[: int(cap * cfg.language_weights.get(lang, 1.0))], key=lambda s: (s.dataset_id, s.sample_id))

    lines = [json.dumps(s.__dict__, sort_keys=True, ensure_ascii=False) for s in kept]
    body = "\n".join(lines) + ("\n" if lines else "")
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{cfg.name}.jsonl").write_text(body, encoding="utf-8")
    meta = {"name": cfg.name, "sha256": digest, "samples": len(kept), "hours": round(sum(s.duration for s in kept) / 3600, 4),
            "datasets": {d: registry.get(d).to_json() for d in sorted(cfg.datasets)},
            "commercial": cfg.commercial, "seed": cfg.seed}
    (out_dir / f"{cfg.name}.meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return meta
