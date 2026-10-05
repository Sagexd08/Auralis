import io
import json
import random
import sys
from pathlib import Path
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from safetensors.torch import save_file
from auralis_stt import AuralisSTT, ModelConfig, Tokenizer
from auralis_stt.checkpoint import latest_checkpoint
from auralis_stt.train_stream import Prefetcher, StreamConfig, group_stream, make_batches, run

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "data"))

WORDS = ["hello", "world", "good", "morning", "test", "speech"]


def wav_bytes(seconds, freq):
    t = np.arange(int(16000 * seconds)) / 16000
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * freq * t)).astype("float32"), 16000, format="WAV")
    return buffer.getvalue()


def make_checkpoint(path, tokenizer):
    torch.manual_seed(0)
    cfg = ModelConfig(vocab_size=len(tokenizer), d_model=16, n_layers=1, n_heads=2, conv_kernel=7, dropout=0.0)
    model = AuralisSTT(cfg)
    path.mkdir(parents=True)
    save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path / "model.safetensors"))
    (path / "config.json").write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    tokenizer.save(path / "tokenizer.json")


def make_shard(path, rows=24):
    audio = [{"bytes": wav_bytes(1.5 + 0.1 * (i % 5), 300 + 40 * (i % 6)), "path": str(i)} for i in range(rows)]
    text = [" ".join(WORDS[(i + k) % len(WORDS)].lower() for k in range(3)) for i in range(rows)]
    pq.write_table(pa.table({"audio": audio, "text": text}), path, row_group_size=12)


def test_group_stream_decodes_row_groups_and_filters(tmp_path):
    tok = Tokenizer.train([" ".join(WORDS)], 60)
    shard = tmp_path / "s.parquet"
    make_shard(shard)
    groups = list(group_stream(lambda s: open(s, "rb"), [str(shard)], "audio", "text", tok, 1.0, 20.0))
    assert len(groups) == 2
    assert all(len(items) == 12 for _, _, items in groups)
    wave, ids = groups[0][2][0]
    assert wave.dtype == np.int16 and len(ids) > 0
    short_only = list(group_stream(lambda s: open(s, "rb"), [str(shard)], "audio", "text", tok, 5.0, 20.0))
    assert [len(items) for _, _, items in short_only] == [0, 0]


def test_make_batches_covers_every_item_once():
    items = [(np.zeros(100 + i, dtype=np.int16), [1]) for i in range(37)]
    batches = make_batches(items, 8, random.Random(0))
    assert sorted(j for b in batches for j in b) == list(range(37))


def test_streaming_run_trains_checkpoints_and_stops_at_max_steps(tmp_path):
    tok = Tokenizer.train([" ".join(WORDS)], 60)
    make_checkpoint(tmp_path / "init", tok)
    shard = tmp_path / "s.parquet"
    make_shard(shard)
    cfg = StreamConfig(repo="x", shards=[str(shard)], init_from=str(tmp_path / "init"), out_dir=str(tmp_path / "out"),
                       batch_size=4, grad_accum=2, lr=1e-3, warmup_steps=2, max_steps=5, passes_per_group=3, eval_every=5, amp=False)
    stream = group_stream(lambda s: open(s, "rb"), cfg.shards, cfg.audio_col, cfg.text_col, tok, 1.0, 20.0)
    _, _, history = run(cfg, Prefetcher(stream, 1))
    assert [h["step"] for h in history] == [1, 2, 3, 4, 5]
    assert all(np.isfinite(h["loss"]) for h in history)
    assert latest_checkpoint(tmp_path / "out").name == "auralis-stt-step-5"
