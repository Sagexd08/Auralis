# Auralis benchmark harness

Measures whether the audio-intelligence layer (VAD + RNNoise) actually helps
transcription accuracy, rather than assuming it does — per the PRD's "a
denoiser that produces better-looking audio but worse STT is a failed
component." No SOTA claims, only measured numbers with the commands used.

## Quick start

```bash
pip install -r benchmarks/requirements.txt
cargo build -p auralis-runtime --bin auralis-bench-cli
python benchmarks/run_all.py
```

By default this runs against the one bundled fixture clip in
`datasets/fixtures/` (a real whisper.cpp sample with a known transcript),
synthetically noised at 5dB SNR — enough to prove the harness works, not a
meaningful accuracy measurement. For a real benchmark pass:

```bash
python benchmarks/datasets/prepare_librispeech.py --dest benchmarks/datasets/librispeech_clean --count 50
python benchmarks/run_all.py --dataset-dir benchmarks/datasets/librispeech_clean
```

This pulls real audio from the Hugging Face Hub — not run automatically.

## What it measures

For every `(model size) x (raw | vad-denoise) x (clean | noisy)` combination
with a model present in `models/`: WER, CER (via `jiwer`), and RTF
(wall-clock / audio duration). Reports land in `reports/` as paired
`.json`/`.md` files (gitignored — regenerate locally).

## Layout

- `datasets/prepare_librispeech.py` — pulls a small `test-clean` slice via HF `datasets` (streaming, on demand)
- `datasets/prepare_noisy.py` — mixes a clean dataset with procedurally generated noise at a target SNR (no external noise corpus needed)
- `datasets/fixtures/` — bundled 1-clip smoke-test set, no download required
- `runners/run_variant.py` — runs `auralis-bench-cli` across a dataset for one model/preprocessing variant
- `metrics/wer.py` — WER/CER/RTF computation
- `run_all.py` — single reproducible entrypoint
- `tests/test_harness_smoke.py` — exercises the real pipeline against the bundled fixture; skips (not fails) if the model/binary aren't built yet

Run the smoke test directly:

```bash
python -m unittest benchmarks.tests.test_harness_smoke -v
```
