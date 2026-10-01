#!/usr/bin/env python3
"""Single reproducible entrypoint for the Auralis benchmark harness.

Runs every (model size) x (raw | vad-denoise) x (clean | noisy) combination,
computes WER/CER/RTF for each, and writes a report. No SOTA claims — only
measured numbers plus the exact commands used (PRD §27).

Usage:
    python benchmarks/run_all.py [--dataset-dir DIR] [--snr-db 5.0]

By default this runs against the small bundled fixture set in
`benchmarks/datasets/fixtures/` (one clip) rather than downloading the full
LibriSpeech subset — run `datasets/prepare_librispeech.py` first and pass
`--dataset-dir` to benchmark against real data.
"""
import argparse
import json
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from metrics.wer import compute_metrics
from runners.run_variant import run_variant

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_DIR = Path(__file__).resolve().parent

MODELS = {
    "tiny.en": REPO_ROOT / "models" / "ggml-tiny.en-q5_1.bin",
    "base.en": REPO_ROOT / "models" / "ggml-base.en-q5_1.bin",
    "small.en": REPO_ROOT / "models" / "ggml-small.en-q5_1.bin",
}

PREPROCESS_VARIANTS = ["raw", "vad-denoise"]


def bench_cli_path() -> Path:
    exe = "auralis-bench-cli.exe" if sys.platform == "win32" else "auralis-bench-cli"
    for profile in ("debug", "release"):
        candidate = REPO_ROOT / "target" / profile / exe
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "auralis-bench-cli not built. Run: cargo build -p auralis-runtime --bin auralis-bench-cli"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", type=Path, default=BENCH_DIR / "datasets" / "fixtures")
    parser.add_argument("--snr-db", type=float, default=5.0)
    parser.add_argument("--report-name", default=None)
    args = parser.parse_args()

    cli = bench_cli_path()

    with tempfile.TemporaryDirectory() as tmp:
        noisy_dir = Path(tmp) / "noisy"
        subprocess.run(
            [
                sys.executable,
                str(BENCH_DIR / "datasets" / "prepare_noisy.py"),
                "--src", str(args.dataset_dir),
                "--dest", str(noisy_dir),
                "--snr-db", str(args.snr_db),
            ],
            check=True,
        )

        datasets = {"clean": args.dataset_dir, "noisy": noisy_dir}

        rows = []
        for model_name, model_path in MODELS.items():
            if not model_path.exists():
                print(f"skipping {model_name}: not found at {model_path} (run models/pull-model.ps1 or download it)")
                continue
            for preprocess in PREPROCESS_VARIANTS:
                for dataset_name, dataset_dir in datasets.items():
                    result = run_variant(cli, model_path, dataset_dir, preprocess, model_name, dataset_name)
                    metrics = compute_metrics(
                        result.references, result.hypotheses, result.total_elapsed_s, result.total_audio_s
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "preprocess": preprocess,
                            "dataset": dataset_name,
                            "wer": metrics.wer,
                            "cer": metrics.cer,
                            "rtf": metrics.rtf,
                            "num_clips": metrics.num_clips,
                        }
                    )
                    print(f"{model_name:10s} {preprocess:12s} {dataset_name:6s}  WER={metrics.wer:.3f} CER={metrics.cer:.3f} RTF={metrics.rtf:.3f}")

    write_report(rows, args.report_name)


def write_report(rows: list[dict], report_name: str | None) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = report_name or timestamp
    reports_dir = BENCH_DIR / "reports"
    reports_dir.mkdir(exist_ok=True)

    json_path = reports_dir / f"{name}.json"
    json_path.write_text(json.dumps(rows, indent=2))

    lines = [
        f"# Auralis benchmark report ({timestamp})",
        "",
        "| Model | Preprocess | Dataset | WER | CER | RTF | Clips |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['model']} | {r['preprocess']} | {r['dataset']} | "
            f"{r['wer']:.3f} | {r['cer']:.3f} | {r['rtf']:.3f} | {r['num_clips']} |"
        )
    md_path = reports_dir / f"{name}.md"
    md_path.write_text("\n".join(lines) + "\n")

    print(f"\nReport written to {json_path} and {md_path}")


if __name__ == "__main__":
    main()
