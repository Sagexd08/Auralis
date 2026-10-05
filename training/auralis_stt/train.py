from __future__ import annotations
import argparse
import json
import math
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from .checkpoint import latest_checkpoint, load_model, save_checkpoint
from .ctc import ctc_loss, greedy_decode
from .dataset import SpeechDataset, collate, load_corpus, split_by_speaker
from .features import LogMel, normalize_features
from .metrics import cer, wer
from .model import AuralisSTT, ModelConfig
from .tokenizer import Tokenizer


@dataclass
class TrainConfig:
    corpus_dir: str
    corpus_name: str
    out_dir: str
    vocab_size: int = 256
    batch_size: int = 8
    grad_accum: int = 1
    lr: float = 1e-3
    warmup_steps: int = 200
    max_steps: int = 2000
    eval_every: int = 500
    val_pct: int = 5
    seed: int = 1004
    max_seconds: float = 20.0
    num_workers: int = 0
    amp: bool = True
    grad_clip: float = 5.0
    spec_augment: bool = False
    time_budget: float = 0.0
    init_from: str = ""
    init_from: str = ""
    model: dict = field(default_factory=dict)

    @staticmethod
    def load(path) -> "TrainConfig":
        return TrainConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def spec_augment(feats: torch.Tensor, frames: torch.Tensor, freq_width: int = 27, freq_masks: int = 2, time_width: int = 40, time_masks: int = 2) -> torch.Tensor:
    out = feats.clone()
    n_mels = out.shape[1]
    for b in range(out.shape[0]):
        length = int(frames[b])
        for _ in range(freq_masks):
            w = random.randint(0, min(freq_width, n_mels))
            f0 = random.randint(0, n_mels - w)
            out[b, f0:f0 + w, :] = 0
        for _ in range(time_masks):
            w = random.randint(0, min(time_width, max(1, length // 5)))
            t0 = random.randint(0, max(0, length - w))
            out[b, :, t0:t0 + w] = 0
    return out


def lr_at(step: int, cfg: TrainConfig, elapsed: float = 0.0) -> float:
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    if cfg.time_budget > 0:
        progress = max(progress, elapsed / cfg.time_budget)
    return cfg.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(progress, 1.0))))


@torch.no_grad()
def evaluate(model, tokenizer, loader, logmel, device, language_of=None):
    model.eval()
    refs, hyps = [], []
    for waves, wave_lengths, _, _, texts in loader:
        waves, wave_lengths = waves.to(device), wave_lengths.to(device)
        feats = logmel(waves)
        frames = torch.tensor([logmel.num_frames(int(n)) for n in wave_lengths.tolist()], device=device)
        log_probs, out_lengths = model(normalize_features(feats, frames), frames)
        for ids in greedy_decode(log_probs, out_lengths, tokenizer.blank_id):
            hyps.append(tokenizer.decode(ids))
        refs += texts
    model.train()
    return {"wer": wer(refs, hyps), "cer": cer(refs, hyps), "samples": len(refs)}, refs, hyps


def train(cfg: TrainConfig, resume: bool = False):
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta, rows, roots = load_corpus(cfg.corpus_dir, cfg.corpus_name)
    train_rows, val_rows = split_by_speaker(rows, cfg.val_pct)
    if resume and latest_checkpoint(out_dir):
        ckpt = latest_checkpoint(out_dir)
        model, tokenizer = load_model(ckpt, device)
        model.train()
    elif cfg.init_from:
        ckpt = None
        model, tokenizer = load_model(cfg.init_from, device)
        model.train()
    elif cfg.init_from:
        ckpt = None
        model, tokenizer = load_model(cfg.init_from, device)
        model.train()
    else:
        ckpt = None
        tokenizer = Tokenizer.train([r["transcript"] for r in train_rows], cfg.vocab_size)
        model = AuralisSTT(ModelConfig(vocab_size=len(tokenizer), **cfg.model)).to(device)
    logmel = LogMel(n_mels=model.cfg.n_mels).to(device)

    train_ds = SpeechDataset(train_rows, roots, tokenizer, cfg.max_seconds)
    val_ds = SpeechDataset(val_rows, roots, tokenizer, cfg.max_seconds)
    g = torch.Generator().manual_seed(cfg.seed)
    train_loader = DataLoader(train_ds, cfg.batch_size, shuffle=True, collate_fn=collate, num_workers=cfg.num_workers, generator=g, drop_last=len(train_ds) >= cfg.batch_size)
    val_loader = DataLoader(val_ds, cfg.batch_size, shuffle=False, collate_fn=collate, num_workers=cfg.num_workers)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=1e-2)
    t0 = time.time()
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, cfg, time.time() - t0) / cfg.lr)
    step = 0
    if ckpt is not None:
        state = torch.load(ckpt / "trainer_state.pt", map_location=device)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        step = state["step"]
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hardware = {"device": device.type, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(), "torch": torch.__version__}
    log = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    history = []
    micro = 0
    final = False
    model.train()
    while step < cfg.max_steps and not final:
        for waves, wave_lengths, targets, target_lengths, _ in train_loader:
            waves, wave_lengths = waves.to(device), wave_lengths.to(device)
            with torch.no_grad():
                feats = logmel(waves)
                frames = torch.tensor([logmel.num_frames(int(n)) for n in wave_lengths.tolist()], device=device)
                feats = normalize_features(feats, frames)
                if cfg.spec_augment:
                    feats = spec_augment(feats, frames)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                log_probs, out_lengths = model(feats, frames)
            loss = ctc_loss(log_probs.float(), targets.to(device), out_lengths, target_lengths.to(device), tokenizer.blank_id) / cfg.grad_accum
            scaler.scale(loss).backward()
            micro += 1
            if micro % cfg.grad_accum:
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            step += 1
            final = step >= cfg.max_steps or (cfg.time_budget > 0 and time.time() - t0 >= cfg.time_budget)
            row = {"step": step, "loss": loss.item() * cfg.grad_accum, "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - t0, 1)}
            if device.type == "cuda":
                row["peak_vram_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if step % cfg.eval_every == 0 or final:
                metrics, _, _ = evaluate(model, tokenizer, val_loader, logmel, device)
                row.update({f"val_{k}": v for k, v in metrics.items()})
            log.write(json.dumps(row) + "\n")
            log.flush()
            history.append(row)
            if "val_wer" in row:
                save_checkpoint(out_dir, step, model, tokenizer, optimizer, scheduler, {"history_tail": history[-5:], **row},
                                {"corpus": meta, "train_samples": len(train_ds), "val_samples": len(val_ds)}, cfg.seed, hardware, asdict(cfg))
            if final:
                break
    log.close()
    return model, tokenizer, history


def main(argv=None):
    ap = argparse.ArgumentParser(prog="auralis-train")
    ap.add_argument("config")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args(argv)
    _, _, history = train(TrainConfig.load(a.config), a.resume)
    last = history[-1]
    print(json.dumps(last))


if __name__ == "__main__":
    main()
