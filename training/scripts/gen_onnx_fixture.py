import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "data"))

from auralis_data import CorpusConfig, Registry, build_corpus
from auralis_data.connectors import import_dataset
from auralis_stt import LogMel
from auralis_stt.checkpoint import latest_checkpoint, load_model
from auralis_stt.ctc import greedy_decode
from auralis_stt.export import export_checkpoint
from auralis_stt.features import normalize_features
from auralis_stt.train import TrainConfig, train

OUT = ROOT.parent / "runtime" / "core" / "tests" / "fixtures" / "auralis_tiny"
LETTERS = "abcdef"
FREQS = {c: 300 + 260 * i for i, c in enumerate(LETTERS)}


def utterance(rng, letters):
    parts = []
    for c in letters:
        t = np.arange(int(0.3 * 16000)) / 16000
        parts.append(0.3 * np.sin(2 * np.pi * FREQS[c] * t) + 0.01 * rng.standard_normal(len(t)))
        parts.append(np.zeros(int(0.1 * 16000)))
    return np.concatenate(parts).astype("float32")


def build_source(root, n=240):
    rng = np.random.default_rng(7)
    rows = []
    (root / "wav").mkdir(parents=True)
    for k in range(n):
        seq = [LETTERS[i] for i in rng.integers(0, len(LETTERS), size=int(rng.integers(3, 6)))]
        sf.write(str(root / "wav" / f"u{k}.wav"), utterance(rng, seq), 16000)
        rows.append({"path": f"wav/u{k}.wav", "transcript": " ".join(seq), "speaker_id": f"s{k % 20}"})
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def main():
    work = Path(tempfile.mkdtemp())
    build_source(work / "src")
    rec, _ = import_dataset("local", work / "src", "synth", "en", "1", work / "manifests", licence="CC0-1.0")
    reg = Registry(work / "reg.jsonl")
    reg.register(rec)
    build_corpus(CorpusConfig(name="synth", datasets=["synth"], min_duration=0.5, min_quality=0.5), reg, work / "manifests", work / "corpus")
    cfg = TrainConfig(
        corpus_dir=str(work / "corpus"), corpus_name="synth", out_dir=str(work / "run"), vocab_size=16,
        batch_size=16, lr=3e-3, warmup_steps=20, max_steps=260, eval_every=130, val_pct=10, amp=False,
        model={"d_model": 48, "n_layers": 2, "n_heads": 2, "dropout": 0.0, "conv_kernel": 7},
    )
    torch.set_num_threads(4)
    _, _, history = train(cfg)
    final = [h for h in history if "val_wer" in h][-1]
    print("validation WER", final["val_wer"])
    assert final["val_wer"] < 0.2

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    export_checkpoint(latest_checkpoint(work / "run"), OUT, "auralis", ["en"])

    model, tok = load_model(latest_checkpoint(work / "run"))
    rng = np.random.default_rng(99)
    wave = utterance(rng, list("bdfac"))
    tensor = torch.from_numpy(wave).unsqueeze(0)
    logmel = LogMel(n_mels=model.cfg.n_mels)
    frames = torch.tensor([logmel.num_frames(len(wave))])
    with torch.no_grad():
        feats = normalize_features(logmel(tensor), frames)
        log_probs, out_len = model(feats, frames)
    text = tok.decode(greedy_decode(log_probs, out_len, tok.blank_id)[0])
    print("expected text:", text)
    wave.astype("<f4").tofile(OUT / "sample.f32")
    log_probs[0].numpy().astype("<f4").tofile(OUT / "expected_log_probs.f32")
    (OUT / "expected.json").write_text(
        json.dumps({"text": text, "steps": int(log_probs.shape[1]), "vocab": int(log_probs.shape[2]), "samples": len(wave)}, indent=2),
        encoding="utf-8",
    )
    print("wrote", OUT)


if __name__ == "__main__":
    main()
