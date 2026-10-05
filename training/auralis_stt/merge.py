from __future__ import annotations
import argparse
import json
import shutil
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file


def average_checkpoints(checkpoint_dirs, out_dir, weights=None) -> Path:
    dirs = [Path(d) for d in checkpoint_dirs]
    if len(dirs) < 2:
        raise ValueError("need at least two checkpoints to merge")
    weights = list(weights) if weights else [1.0] * len(dirs)
    if len(weights) != len(dirs) or min(weights) <= 0:
        raise ValueError("weights must be positive, one per checkpoint")
    total = sum(weights)

    configs = [json.loads((d / "config.json").read_text(encoding="utf-8")) for d in dirs]
    if any(c != configs[0] for c in configs[1:]):
        raise ValueError("checkpoints have different model configs")
    tokenizers = [(d / "tokenizer.json").read_text(encoding="utf-8") for d in dirs]
    if any(t != tokenizers[0] for t in tokenizers[1:]):
        raise ValueError("checkpoints have different tokenizers")

    states = [load_file(str(d / "model.safetensors")) for d in dirs]
    keys = set(states[0])
    if any(set(s) != keys for s in states[1:]):
        raise ValueError("checkpoints have different parameters")

    merged = {}
    for key in sorted(keys):
        reference = states[0][key]
        if reference.is_floating_point():
            acc = sum(s[key].to(torch.float32) * (w / total) for s, w in zip(states, weights))
            merged[key] = acc.to(reference.dtype).contiguous()
        else:
            merged[key] = reference.contiguous()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(merged, str(out_dir / "model.safetensors"))
    shutil.copyfile(dirs[0] / "config.json", out_dir / "config.json")
    shutil.copyfile(dirs[0] / "tokenizer.json", out_dir / "tokenizer.json")
    (out_dir / "merge.json").write_text(json.dumps({"sources": [str(d) for d in dirs], "weights": weights}, indent=2), encoding="utf-8")
    return out_dir


def main(argv=None):
    ap = argparse.ArgumentParser(prog="auralis-merge")
    ap.add_argument("out_dir")
    ap.add_argument("checkpoints", nargs="+")
    ap.add_argument("--weights", default="")
    a = ap.parse_args(argv)
    weights = [float(x) for x in a.weights.split(",")] if a.weights else None
    print(average_checkpoints(a.checkpoints, a.out_dir, weights))


if __name__ == "__main__":
    main()
