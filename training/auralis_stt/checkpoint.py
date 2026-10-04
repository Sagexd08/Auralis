from __future__ import annotations
import json
import subprocess
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file
from .model import AuralisSTT, ModelConfig
from .tokenizer import Tokenizer


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def save_checkpoint(out_dir, step, model, tokenizer, optimizer, scheduler, metrics, data_manifest, seed, hardware, train_config):
    d = Path(out_dir) / f"auralis-stt-step-{step}"
    d.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps(model.cfg.to_dict(), indent=2), encoding="utf-8")
    tokenizer.save(d / "tokenizer.json")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "step": step}, d / "trainer_state.pt")
    (d / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (d / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (d / "train_config.json").write_text(json.dumps(train_config, indent=2), encoding="utf-8")
    (d / "git_commit.txt").write_text(git_commit(), encoding="utf-8")
    (d / "run.json").write_text(json.dumps({"seed": seed, "hardware": hardware, "step": step}, indent=2), encoding="utf-8")
    return d


def load_model(checkpoint_dir, device="cpu"):
    d = Path(checkpoint_dir)
    cfg = ModelConfig(**json.loads((d / "config.json").read_text(encoding="utf-8")))
    model = AuralisSTT(cfg)
    model.load_state_dict(load_file(str(d / "model.safetensors")))
    return model.to(device).eval(), Tokenizer.load(d / "tokenizer.json")


def latest_checkpoint(out_dir):
    cands = sorted(Path(out_dir).glob("auralis-stt-step-*"), key=lambda p: int(p.name.rsplit("-", 1)[1]))
    return cands[-1] if cands else None
