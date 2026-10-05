from __future__ import annotations
import argparse
import json
import platform
import queue
import random
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from .checkpoint import load_model, save_checkpoint
from .ctc import ctc_loss
from .dataset import SpeechDataset, collate, load_corpus
from .features import LogMel, normalize_features
from .train import TrainConfig, evaluate, lr_at, seed_everything, spec_augment

SAMPLE_RATE = 16000


@dataclass
class StreamConfig:
    repo: str
    shards: list
    init_from: str
    out_dir: str
    audio_col: str = "audio"
    text_col: str = "text"
    batch_size: int = 8
    grad_accum: int = 8
    lr: float = 2e-4
    warmup_steps: int = 100
    max_steps: int = 3000
    passes_per_group: int = 8
    min_seconds: float = 1.0
    max_seconds: float = 20.0
    eval_every: int = 500
    eval_clips: int = 200
    eval_corpus_dir: str = ""
    eval_corpus_name: str = ""
    spec_augment: bool = True
    amp: bool = True
    grad_clip: float = 5.0
    seed: int = 1004
    prefetch: int = 1
    meta: dict = field(default_factory=dict)

    @staticmethod
    def load(path) -> "StreamConfig":
        return StreamConfig(**json.loads(Path(path).read_text(encoding="utf-8")))


def hf_opener(repo: str):
    from huggingface_hub import HfFileSystem

    fs = HfFileSystem()
    return lambda shard: fs.open(f"datasets/{repo}/{shard}", "rb")


def decode_group(rows, audio_col, text_col, tokenizer, min_seconds, max_seconds):
    from auralis_data.connectors.parquet_audio import clean_transcript, decode_audio

    items = []
    for row in rows:
        text = clean_transcript(row[text_col])
        audio = row[audio_col]
        if text is None or not audio or not audio.get("bytes"):
            continue
        try:
            wave = decode_audio(audio["bytes"])
        except Exception:
            continue
        seconds = len(wave) / SAMPLE_RATE
        if not (min_seconds <= seconds <= max_seconds):
            continue
        ids = tokenizer.encode(text)
        if not ids or tokenizer.unk_id in ids or len(ids) > len(wave) // 160 // 4 - 2:
            continue
        items.append(((np.clip(wave, -1.0, 1.0) * 32767).astype(np.int16), ids))
    return items


def group_stream(open_file, shards, audio_col, text_col, tokenizer, min_seconds, max_seconds, retries: int = 3):
    for shard in shards:
        try:
            handle = open_file(shard)
            parquet = pq.ParquetFile(handle)
        except Exception as error:
            print(f"skipping shard {shard}: {error}", flush=True)
            continue
        for index in range(parquet.num_row_groups):
            table = None
            for attempt in range(retries):
                try:
                    table = parquet.read_row_group(index, columns=[audio_col, text_col])
                    break
                except Exception as error:
                    print(f"retry {attempt + 1} for {shard} group {index}: {error}", flush=True)
                    time.sleep(5)
            if table is None:
                continue
            yield shard, index, decode_group(table.to_pylist(), audio_col, text_col, tokenizer, min_seconds, max_seconds)
        try:
            handle.close()
        except Exception:
            pass


class Prefetcher:
    def __init__(self, iterator, depth: int = 1):
        self.queue = queue.Queue(maxsize=max(1, depth))
        self.thread = threading.Thread(target=self.fill, args=(iterator,), daemon=True)
        self.thread.start()

    def fill(self, iterator):
        try:
            for item in iterator:
                self.queue.put(item)
        finally:
            self.queue.put(None)

    def __iter__(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            yield item


def make_batches(items, batch_size: int, rng: random.Random):
    order = list(range(len(items)))
    rng.shuffle(order)
    window = batch_size * 16
    batches = []
    for start in range(0, len(order), window):
        part = sorted(order[start:start + window], key=lambda j: len(items[j][0]))
        batches += [part[k:k + batch_size] for k in range(0, len(part), batch_size)]
    rng.shuffle(batches)
    return batches


def to_tensors(items, batch):
    waves = [torch.from_numpy(items[j][0].astype(np.float32) / 32768.0) for j in batch]
    lengths = torch.tensor([len(w) for w in waves])
    padded = torch.zeros(len(waves), int(lengths.max()))
    for i, w in enumerate(waves):
        padded[i, : len(w)] = w
    targets = [torch.tensor(items[j][1], dtype=torch.long) for j in batch]
    return padded, lengths, torch.cat(targets), torch.tensor([len(t) for t in targets])


def build_eval(cfg: StreamConfig, tokenizer):
    if not cfg.eval_corpus_dir:
        return None
    _, rows, roots = load_corpus(cfg.eval_corpus_dir, cfg.eval_corpus_name)
    stride = max(1, len(rows) // max(1, cfg.eval_clips))
    dataset = SpeechDataset(rows[::stride][: cfg.eval_clips], roots, tokenizer, 35.0)
    return DataLoader(dataset, 8, shuffle=False, collate_fn=collate)


def run(cfg: StreamConfig, groups):
    seed_everything(cfg.seed)
    rng = random.Random(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(cfg.init_from, device)
    model.train()
    logmel = LogMel(n_mels=model.cfg.n_mels).to(device)
    eval_loader = build_eval(cfg, tokenizer)
    schedule = TrainConfig(corpus_dir="", corpus_name="", out_dir=str(out_dir), lr=cfg.lr, warmup_steps=cfg.warmup_steps, max_steps=cfg.max_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.98), weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: lr_at(s, schedule) / cfg.lr)
    use_amp = cfg.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    hardware = {"device": device.type, "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else platform.processor(), "torch": torch.__version__}
    log = (out_dir / "train_log.jsonl").open("a", encoding="utf-8")
    history = []
    started = time.time()

    def score():
        if eval_loader is None:
            return {}
        metrics, _, _ = evaluate(model, tokenizer, eval_loader, logmel, device)
        return {f"val_{k}": v for k, v in metrics.items()}

    def checkpoint(step, row):
        save_checkpoint(out_dir, step, model, tokenizer, optimizer, scheduler, {"history_tail": history[-5:], **row},
                        {"stream": {"repo": cfg.repo, "shards": cfg.shards}}, cfg.seed, hardware, asdict(cfg))

    baseline = score()
    if baseline:
        print(json.dumps({"step": 0, **baseline}), flush=True)
        log.write(json.dumps({"step": 0, **baseline}) + "\n")

    step = 0
    micro = 0
    clips_seen = 0
    for shard, index, items in groups:
        if not items:
            continue
        print(f"group {shard}#{index}: {len(items)} clips, {sum(len(i[0]) for i in items) / SAMPLE_RATE / 3600:.2f} h", flush=True)
        for _ in range(cfg.passes_per_group):
            for batch in make_batches(items, cfg.batch_size, rng):
                waves, wave_lengths, targets, target_lengths = to_tensors(items, batch)
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
                clips_seen += len(batch)
                if micro % cfg.grad_accum:
                    continue
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                step += 1
                final = step >= cfg.max_steps
                row = {"step": step, "loss": loss.item() * cfg.grad_accum, "lr": scheduler.get_last_lr()[0], "seconds": round(time.time() - started, 1), "clips_seen": clips_seen}
                if step % cfg.eval_every == 0 or final:
                    row.update(score())
                    checkpoint(step, row)
                    model.train()
                log.write(json.dumps(row) + "\n")
                log.flush()
                history.append(row)
                if step % 25 == 0 or final or "val_wer" in row:
                    print(json.dumps(row), flush=True)
                if final:
                    break
            if step >= cfg.max_steps:
                break
        if step >= cfg.max_steps:
            break
    log.close()
    return model, tokenizer, history


def main(argv=None):
    ap = argparse.ArgumentParser(prog="auralis-train-stream")
    ap.add_argument("config")
    a = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))
    cfg = StreamConfig.load(a.config)
    tokenizer = load_model(cfg.init_from, "cpu")[1]
    stream = group_stream(hf_opener(cfg.repo), cfg.shards, cfg.audio_col, cfg.text_col, tokenizer, cfg.min_seconds, cfg.max_seconds)
    _, _, history = run(cfg, Prefetcher(stream, cfg.prefetch))
    print(json.dumps(history[-1]) if history else "no training steps ran")


if __name__ == "__main__":
    main()
