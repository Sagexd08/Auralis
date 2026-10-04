import json
import pytest
from auralis_data import (DatasetRecord, LicenseState, Registry, CorpusConfig, LicenceGateError,
                          build_corpus, check_dataset)

def rec(i, state, **kw):
    base = dict(dataset_id=i, source="local", revision="1", license="CC-BY-4.0", state=state, training_allowed=True,
                commercial_allowed=state in (LicenseState.ELIGIBLE, LicenseState.REDISTRIBUTABLE), languages=["en"])
    base.update(kw)
    return DatasetRecord(**base)

def samples(ds, n, lang="en", dur=5.0):
    return [dict(sample_id=f"{ds}-{lang}-{k}", dataset_id=ds, path=f"{ds}/{k}.wav", transcript=f"{ds} {lang} sentence {k}",
                 language=lang, duration=dur, quality=0.9) for k in range(n)]

def write(dirp, ds, rows):
    (dirp / f"{ds}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

def test_unknown_licence_is_rejected():
    assert not check_dataset(rec("x", LicenseState.UNKNOWN)).allowed
    assert not check_dataset(rec("x", LicenseState.REJECTED)).allowed

def test_missing_licence_or_revision_is_rejected():
    assert not check_dataset(rec("x", LicenseState.ELIGIBLE, license="")).allowed
    assert not check_dataset(rec("x", LicenseState.ELIGIBLE, revision="")).allowed

def test_training_not_allowed_is_rejected():
    assert not check_dataset(rec("x", LicenseState.ELIGIBLE, training_allowed=False)).allowed

def test_noncommercial_excluded_when_commercial():
    r = rec("x", LicenseState.NONCOMMERCIAL_ONLY)
    assert check_dataset(r, commercial=False).allowed
    assert not check_dataset(r, commercial=True).allowed

def test_training_only_excluded_when_commercial():
    r = rec("x", LicenseState.TRAINING_ONLY)
    assert check_dataset(r).allowed and not check_dataset(r, commercial=True).allowed

def test_unregistered_dataset_is_an_error(tmp_path):
    with pytest.raises(KeyError):
        Registry(tmp_path / "r.jsonl").get("ghost")

def test_registry_roundtrip(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(rec("a", LicenseState.ELIGIBLE))
    reg.save()
    assert Registry(tmp_path / "r.jsonl").get("a").state == LicenseState.ELIGIBLE

def test_corpus_refuses_unknown_dataset(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(rec("good", LicenseState.ELIGIBLE))
    reg.register(rec("bad", LicenseState.UNKNOWN))
    write(tmp_path, "good", samples("good", 3))
    write(tmp_path, "bad", samples("bad", 3))
    with pytest.raises(LicenceGateError):
        build_corpus(CorpusConfig("c", ["good", "bad"]), reg, tmp_path, tmp_path / "out")
    assert not (tmp_path / "out" / "c.jsonl").exists()

def test_corpus_is_reproducible_and_filters(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(rec("a", LicenseState.ELIGIBLE))
    base = samples("a", 5)
    rows = base + [dict(base[0], sample_id="dup"),
                   dict(base[0], sample_id="short", duration=0.1, transcript="s"),
                   dict(base[0], sample_id="bad-q", quality=0.1, transcript="q")]
    write(tmp_path, "a", rows)
    m1 = build_corpus(CorpusConfig("c", ["a"]), reg, tmp_path, tmp_path / "o1")
    m2 = build_corpus(CorpusConfig("c", ["a"]), reg, tmp_path, tmp_path / "o2")
    assert m1["sha256"] == m2["sha256"] and m1["samples"] == 5

def test_language_weights_balance(tmp_path):
    reg = Registry(tmp_path / "r.jsonl")
    reg.register(rec("a", LicenseState.ELIGIBLE))
    write(tmp_path, "a", samples("a", 40, "en") + samples("a", 10, "hi"))
    build_corpus(CorpusConfig("c", ["a"], language_weights={"en": 1.0, "hi": 1.0}), reg, tmp_path, tmp_path / "o")
    rows = [json.loads(line) for line in (tmp_path / "o" / "c.jsonl").read_text().splitlines()]
    assert sum(r["language"] == "en" for r in rows) == sum(r["language"] == "hi" for r in rows) == 10
