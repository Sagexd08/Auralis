import json
import numpy as np
import pytest
import soundfile as sf
from auralis_data import LicenseState, Registry, CorpusConfig, build_corpus, check_dataset
from auralis_data.connectors import import_dataset, score_audio, classify_licence


def tone(path, seconds=2.0, amp=0.3):
    t = np.arange(int(16000 * seconds)) / 16000
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), (amp * np.sin(2 * np.pi * 220 * t)).astype("float32"), 16000)


def test_librispeech_layout(tmp_path):
    chap = tmp_path / "19" / "198"
    for k in range(3):
        tone(chap / f"19-198-000{k}.flac")
    (chap / "19-198.trans.txt").write_text("\n".join(f"19-198-000{k} HELLO WORLD {k}" for k in range(3)), encoding="utf-8")
    rec, samples = import_dataset("librispeech", tmp_path, "ls", "en", "dev-clean", tmp_path / "m", score=True)
    assert len(samples) == 3 and samples[0].transcript.startswith("hello world")
    assert samples[0].speaker_id == "19" and samples[0].duration == pytest.approx(2.0)
    assert rec.state == LicenseState.ELIGIBLE and rec.training_allowed and rec.commercial_allowed


def test_commonvoice_layout(tmp_path):
    for k in range(2):
        tone(tmp_path / "clips" / f"clip{k}.wav")
    (tmp_path / "validated.tsv").write_text(
        "client_id\tpath\tsentence\nabc\tclip0.wav\tnamaste duniya\nabc\tclip1.wav\tkal meeting hai\n", encoding="utf-8")
    rec, samples = import_dataset("commonvoice", tmp_path, "cv-hi", "hi", "27.0", tmp_path / "m")
    assert [s.transcript for s in samples] == ["namaste duniya", "kal meeting hai"]
    assert rec.license == "CC0-1.0" and rec.languages == ["hi"]


def test_fleurs_layout(tmp_path):
    tone(tmp_path / "audio" / "train" / "a.wav")
    (tmp_path / "train.tsv").write_text("1\ta.wav\tRaw text\ttranscript text\tchars\t32000\tMALE\n", encoding="utf-8")
    rec, samples = import_dataset("fleurs", tmp_path, "fl-ja", "ja", "main", tmp_path / "m")
    assert samples[0].transcript == "transcript text" and samples[0].duration == pytest.approx(2.0)


def test_local_unknown_licence_is_refused_by_the_gate(tmp_path):
    tone(tmp_path / "x.wav")
    (tmp_path / "metadata.jsonl").write_text(json.dumps({"path": "x.wav", "transcript": "hi"}) + "\n", encoding="utf-8")
    rec, _ = import_dataset("local", tmp_path, "mine", "en", "1", tmp_path / "m", licence="")
    assert rec.state == LicenseState.UNKNOWN
    assert not check_dataset(rec).allowed


def test_licence_table():
    assert classify_licence("CC-BY-NC-4.0")[0] == LicenseState.NONCOMMERCIAL_ONLY
    assert classify_licence("made-up")[0] == LicenseState.UNKNOWN


def test_quality_scoring():
    good = 0.3 * np.sin(np.linspace(0, 100, 16000))
    assert score_audio(good) == 1.0
    assert score_audio(np.zeros(16000)) < 0.5
    assert score_audio(np.sign(good)) < 1.0


def test_import_then_build_is_deterministic(tmp_path):
    tone(tmp_path / "x.wav")
    (tmp_path / "metadata.jsonl").write_text(json.dumps({"path": "x.wav", "transcript": "hello there"}) + "\n", encoding="utf-8")
    rec, _ = import_dataset("local", tmp_path, "mine", "en", "1", tmp_path / "m", licence="CC-BY-4.0")
    reg = Registry(tmp_path / "reg.jsonl")
    reg.register(rec)
    cfg = CorpusConfig(name="c", datasets=["mine"], min_duration=0.5)
    a = build_corpus(cfg, reg, tmp_path / "m", tmp_path / "o1")
    b = build_corpus(cfg, reg, tmp_path / "m", tmp_path / "o2")
    assert a["sha256"] == b["sha256"] and a["samples"] == 1
