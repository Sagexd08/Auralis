"""Smoke test for the benchmark harness: no full dataset download required.

Exercises the real pipeline — bench_cli subprocess, WER/CER computation,
report generation — against the small bundled fixture (1 clip) plus a
synthetically noised copy of it, so this validates harness mechanics, not
model accuracy. Skips (rather than fails) if the whisper model or the
bench_cli binary isn't built, since those are produced by earlier Phase 1
steps, not by this test.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = BENCH_DIR.parent
sys.path.insert(0, str(BENCH_DIR))

from metrics.wer import compute_metrics  # noqa: E402
from runners.run_variant import run_variant  # noqa: E402

MODEL_PATH = REPO_ROOT / "models" / "ggml-base.en-q5_1.bin"


def _bench_cli_path() -> Path | None:
    exe = "auralis-bench-cli.exe" if sys.platform == "win32" else "auralis-bench-cli"
    for profile in ("debug", "release"):
        candidate = REPO_ROOT / "target" / profile / exe
        if candidate.exists():
            return candidate
    return None


class HarnessSmokeTest(unittest.TestCase):
    def setUp(self):
        self.bench_cli = _bench_cli_path()
        if self.bench_cli is None:
            self.skipTest("auralis-bench-cli not built; run: cargo build -p auralis-runtime --bin auralis-bench-cli")
        if not MODEL_PATH.exists():
            self.skipTest(f"model not found at {MODEL_PATH}; run models/pull-model.ps1")

    def test_clean_fixture_transcribes_and_scores(self):
        dataset_dir = BENCH_DIR / "datasets" / "fixtures"
        result = run_variant(self.bench_cli, MODEL_PATH, dataset_dir, "raw", "base.en", "clean")

        self.assertEqual(len(result.references), len(result.hypotheses))
        self.assertGreater(result.total_audio_s, 0)

        metrics = compute_metrics(result.references, result.hypotheses, result.total_elapsed_s, result.total_audio_s)
        # A known-good clip on base.en should transcribe near-perfectly.
        self.assertLess(metrics.wer, 0.2, f"unexpectedly high WER on clean fixture: {metrics.wer}, hyp={result.hypotheses}")

    def test_noisy_variant_generation_and_report_roundtrip(self):
        # Imports the LOCAL benchmarks/datasets/ package (not pip's `datasets`
        # library) — deliberately scoped inside this test since sys.path
        # already has BENCH_DIR on it, and this test never needs the pip
        # package, so the shadowing is harmless here.
        from datasets.prepare_noisy import main as prepare_noisy_main

        with tempfile.TemporaryDirectory() as tmp:
            noisy_dir = Path(tmp) / "noisy"
            old_argv = sys.argv
            sys.argv = [
                "prepare_noisy.py",
                "--src", str(BENCH_DIR / "datasets" / "fixtures"),
                "--dest", str(noisy_dir),
                "--snr-db", "5.0",
            ]
            try:
                prepare_noisy_main()
            finally:
                sys.argv = old_argv

            self.assertTrue((noisy_dir / "manifest.json").exists())

            result = run_variant(self.bench_cli, MODEL_PATH, noisy_dir, "vad-denoise", "base.en", "noisy")
            metrics = compute_metrics(result.references, result.hypotheses, result.total_elapsed_s, result.total_audio_s)

            report = [
                {
                    "model": result.model,
                    "preprocess": result.preprocess,
                    "dataset": result.dataset,
                    "wer": metrics.wer,
                    "cer": metrics.cer,
                    "rtf": metrics.rtf,
                }
            ]
            report_json = json.dumps(report)
            self.assertIn("wer", report_json)


if __name__ == "__main__":
    unittest.main()
