import io
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
from auralis_data.connectors import import_dataset
from auralis_data.connectors.base import classify_licence
from auralis_data.connectors.parquet_audio import clean_transcript, convert_parquet
from auralis_data.provenance import LicenseState


def flac_bytes(rate, seconds=1.0):
    t = np.arange(int(rate * seconds)) / rate
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * 440 * t)).astype("float32"), rate, format="FLAC")
    return buffer.getvalue()


def test_clean_transcript_uppercases_and_rejects_unsupported_characters():
    assert clean_transcript("don't  stop") == "DON'T STOP"
    assert clean_transcript("it’s fine") == "IT'S FINE"
    assert clean_transcript("call 911") is None
    assert clean_transcript("hello, world") is None
    assert clean_transcript("   ") is None
    assert clean_transcript(None) is None


def test_convert_parquet_writes_resampled_flac_and_importable_metadata(tmp_path):
    table = pa.table({
        "id": ["a", "b", "c", "d"],
        "audio": [{"bytes": flac_bytes(8000), "path": "x"}, {"bytes": flac_bytes(16000), "path": "y"}, {"bytes": b"not audio", "path": "z"}, {"bytes": flac_bytes(16000), "path": "w"}],
        "text": ["hello there", "good morning", "broken audio", "has 4 digits"],
        "speaker": ["s1", "s2", "s3", ""],
    })
    shard = tmp_path / "shard.parquet"
    pq.write_table(table, shard)
    out = tmp_path / "out"
    kept, skipped = convert_parquet(shard, out, "audio", "text", "id", "speaker", shard_tag="t0-")
    assert (kept, skipped) == (2, 2)
    rows = [json.loads(line) for line in (out / "metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [r["transcript"] for r in rows] == ["HELLO THERE", "GOOD MORNING"]
    assert rows[0]["speaker_id"] == "s1"
    wave, rate = sf.read(str(out / rows[0]["path"]))
    assert rate == 16000 and abs(len(wave) - 16000) < 20
    record, samples = import_dataset("local", out, "demo", "en", "1", tmp_path / "manifests", licence="cc-by-2.0")
    assert len(samples) == 2 and record.state == LicenseState.ELIGIBLE


def test_cc_by_2_licences_are_eligible():
    assert classify_licence("CC-BY-2.0")[0] == LicenseState.ELIGIBLE
    assert classify_licence("cc-by-2.5")[0] == LicenseState.ELIGIBLE
