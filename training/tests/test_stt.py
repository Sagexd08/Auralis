import json
import numpy as np
import pytest
import soundfile as sf
import torch
import torch.nn.functional as F
from auralis_frontend import log_mel
from auralis_stt import AuralisSTT, LogMel, ModelConfig, Tokenizer, cer, ctc_loss, greedy_decode, wer
from auralis_stt.checkpoint import latest_checkpoint, load_model
from auralis_stt.ctc import ctc_neg_log_likelihood
from auralis_stt.model import subsampled_length

CORPUS = [
    "hello world this is a test of the tokenizer",
    "नमस्ते दुनिया कल मेरी मीटिंग है",
    "আজ আমাকে রিপোর্ট শেষ করতে হবে",
    "明日の3時にクライアントとの打ち合わせがあります",
    "please deploy this to production tomorrow morning",
]


def test_tokenizer_roundtrip_all_launch_scripts():
    tok = Tokenizer.train(CORPUS, 120)
    for text in CORPUS:
        assert tok.decode(tok.encode(text)) == text
    assert tok.blank_id == 0


def test_tokenizer_is_deterministic_and_persists(tmp_path):
    a, b = Tokenizer.train(CORPUS, 90), Tokenizer.train(CORPUS, 90)
    assert a.vocab == b.vocab and a.merges == b.merges
    a.save(tmp_path / "t.json")
    c = Tokenizer.load(tmp_path / "t.json")
    assert c.encode(CORPUS[0]) == a.encode(CORPUS[0])


def test_tokenizer_unknown_character_maps_to_unk():
    tok = Tokenizer.train(["abc abc"], 20)
    assert tok.unk_id in tok.encode("abz")


def test_japanese_needs_no_whitespace():
    tok = Tokenizer.train([CORPUS[3]] * 3, 80)
    assert tok.decode(tok.encode(CORPUS[3])) == CORPUS[3]


def test_wer_cer_known_values():
    assert wer(["the cat sat"], ["the cat sat"]) == 0.0
    assert wer(["the cat sat"], ["the bat sat down"]) == pytest.approx(2 / 3)
    assert cer(["abcd"], ["abxd"]) == pytest.approx(0.25)


def test_greedy_ctc_collapse_rule():
    lp = torch.full((1, 8, 4), -10.0)
    for t, s in enumerate([0, 1, 1, 0, 2, 2, 0, 3]):
        lp[0, t, s] = 0.0
    assert greedy_decode(lp, torch.tensor([8]), blank=0) == [[1, 2, 3]]
    rep = torch.full((1, 5, 3), -10.0)
    for t, s in enumerate([1, 0, 1, 1, 2]):
        rep[0, t, s] = 0.0
    assert greedy_decode(rep, torch.tensor([5]), blank=0) == [[1, 1, 2]]


def test_own_ctc_forward_matches_torch():
    rng = np.random.default_rng(0)
    T, V, target = 12, 5, [1, 2, 2, 3]
    logits = rng.standard_normal((T, V))
    lp = logits - np.logaddexp.reduce(logits, axis=1, keepdims=True)
    mine = ctc_neg_log_likelihood(lp, target)
    ref = F.ctc_loss(torch.tensor(lp).unsqueeze(1), torch.tensor([target]), torch.tensor([T]), torch.tensor([len(target)]), reduction="sum")
    assert mine == pytest.approx(ref.item(), rel=1e-6)


def test_torch_log_mel_matches_numpy_reference():
    rng = np.random.default_rng(1)
    t = np.arange(16000) / 16000
    x = 0.3 * np.sin(2 * np.pi * 300 * t) + 0.02 * rng.standard_normal(16000)
    ref = log_mel(x)
    got = LogMel()(torch.tensor(x, dtype=torch.float32).unsqueeze(0))[0].numpy()
    assert got.shape == ref.shape
    assert np.abs(got - ref).max() < 5e-3


def test_model_shapes_and_length_formula():
    cfg = ModelConfig(vocab_size=30, d_model=32, n_layers=2, n_heads=2)
    model = AuralisSTT(cfg).eval()
    feats = torch.randn(2, 80, 101)
    lengths = torch.tensor([101, 60])
    lp, out_len = model(feats, lengths)
    assert lp.shape == (2, 26, 30)
    assert out_len.tolist() == subsampled_length(lengths).tolist() == [26, 15]
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 26), atol=1e-4)


def make_synthetic_corpus(root, n=240):
    rng = np.random.default_rng(7)
    letters = "abcdef"
    freqs = {c: 300 + 260 * i for i, c in enumerate(letters)}
    rows = []
    (root / "wav").mkdir(parents=True)
    for k in range(n):
        seq = [letters[i] for i in rng.integers(0, len(letters), size=int(rng.integers(3, 6)))]
        parts = []
        for c in seq:
            t = np.arange(int(0.3 * 16000)) / 16000
            parts.append(0.3 * np.sin(2 * np.pi * freqs[c] * t) + 0.01 * rng.standard_normal(len(t)))
            parts.append(np.zeros(int(0.1 * 16000)))
        sf.write(str(root / "wav" / f"u{k}.wav"), np.concatenate(parts).astype("float32"), 16000)
        rows.append({"path": f"wav/u{k}.wav", "transcript": " ".join(seq), "speaker_id": f"s{k % 20}"})
    (root / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def test_end_to_end_overfit_checkpoint_and_resume(tmp_path):
    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2] / "data"))
    from auralis_data import CorpusConfig, Registry, build_corpus
    from auralis_data.connectors import import_dataset
    from auralis_stt.train import TrainConfig, train

    src = tmp_path / "src"
    make_synthetic_corpus(src)
    rec, _ = import_dataset("local", src, "synth", "en", "1", tmp_path / "manifests", licence="CC0-1.0")
    reg = Registry(tmp_path / "reg.jsonl")
    reg.register(rec)
    build_corpus(CorpusConfig(name="synth", datasets=["synth"], min_duration=0.5, min_quality=0.5), reg, tmp_path / "manifests", tmp_path / "corpus")

    cfg = TrainConfig(
        corpus_dir=str(tmp_path / "corpus"), corpus_name="synth", out_dir=str(tmp_path / "run"), vocab_size=16,
        batch_size=16, lr=3e-3, warmup_steps=20, max_steps=220, eval_every=110, val_pct=10, amp=False,
        model={"d_model": 48, "n_layers": 2, "n_heads": 2, "dropout": 0.0, "conv_kernel": 7},
    )
    torch.set_num_threads(4)
    _, _, history = train(cfg)
    assert history[-1]["loss"] < history[0]["loss"]
    final = [h for h in history if "val_wer" in h][-1]
    assert final["val_wer"] < 0.2

    ckpt = latest_checkpoint(tmp_path / "run")
    for name in ("model.safetensors", "config.json", "tokenizer.json", "trainer_state.pt", "metrics.json", "data_manifest.json", "git_commit.txt"):
        assert (ckpt / name).exists()
    model, tok = load_model(ckpt)
    assert len(tok) == model.cfg.vocab_size

    cfg.max_steps = 240
    cfg.eval_every = 20
    _, _, more = train(cfg, resume=True)
    assert more[0]["step"] == 221
