# Auralis Phase 1 — Vertical Slice Design

Status: Approved
Date: 2026-10-01
Source PRD: `auralis_prd_end_to_end.md` (repo root)

## Purpose

The full Auralis PRD describes a multi-year, multi-discipline system: a custom-trained
STT model family, a distributed self-hosted GPU compute fabric, desktop/mobile clients,
and a public website/benchmark leaderboard. None of that is achievable end-to-end in a
single build session, and §53 of the PRD itself warns against building the compute
fabric, dashboards, or mobile apps before the core transcription loop is good.

Phase 1, as scoped here, is **not** §45's "train a custom architecture" phase. It is the
vertical slice from §53:

```
MIC → VAD → DENOISER → STT → TEXT CLEANUP → KEYBOARD INSERTION
```

built as a real, runnable, daily-usable desktop dictation tool, wrapping an existing
open STT model rather than training a new one, plus a small reproducible benchmark
harness that measures whether the audio-intelligence layer actually helps (per §30: "a
denoiser that produces better-looking audio but worse STT is a failed component").

## Decisions locked in during brainstorming

| Decision | Choice | Why |
|---|---|---|
| Phase scope | Vertical slice (§53), not §45's custom-model training | Only realistically buildable/runnable outcome this session |
| Desktop stack | Tauri 2 + Rust, from day one | Matches PRD §14 target architecture; user chose this over a Python prototype |
| STT model | whisper.cpp via `whisper-rs` bindings, GGUF quantized weights | No Python runtime dependency; mature Rust bindings; CUDA feature available |
| GPU | NVIDIA GPU available — build with the `cuda` feature | User confirmed CUDA-capable hardware |
| VAD | `webrtc-vad` (GMM-based, no model file) | Mature, cheap, no extra runtime dependency |
| Denoiser | RNNoise via `nnnoiseless` (pure Rust) | No training required, proven for fan/keyboard/room noise |
| Text cleanup | Local rules only (punctuation, normalization, correction heuristics) | Matches local-first/privacy-by-default principle (§39, §51); no cloud LLM dependency in Phase 1 |
| Hotkey mode | Push-to-talk only | Simplest reliable mode; toggle/continuous are future work |
| Benchmark scope | Two datasets (clean + synthetically noised LibriSpeech subset), compare model sizes × VAD+denoise on/off | Small enough to run and reproduce; large enough to show whether audio-intelligence layer helps |
| Repo scope | Minimal: only what Phase 1 builds gets real directories; full §37 layout lives in `docs/ROADMAP.md` | Avoids empty stub clutter implying unbuilt subsystems exist |
| Platform | Windows-only for Phase 1 | Matches the dev environment; pipeline sits behind a platform trait for later macOS/Linux backends |

## Architecture

```
C:\Desktop\Auralis\
├── apps/desktop/                  # Tauri 2 + Rust desktop app
│   ├── src-tauri/                 # Rust backend: hotkey, tray, pipeline orchestration
│   │   ├── src/
│   │   └── Cargo.toml
│   └── src/                       # Minimal frontend: status window
├── runtime/core/                  # auralis-runtime: reusable Rust lib crate
│   ├── src/
│   │   ├── audio.rs               # mic capture (cpal)
│   │   ├── vad.rs                 # webrtc-vad wrapper
│   │   ├── denoise.rs             # RNNoise wrapper (nnnoiseless)
│   │   ├── stt.rs                 # whisper.cpp wrapper (whisper-rs)
│   │   ├── text.rs                # punctuation/normalization/correction rules
│   │   ├── pipeline.rs            # orchestrates mic → ... → text
│   │   └── lib.rs
│   └── Cargo.toml
├── benchmarks/                    # Python harness (separate toolchain, per PRD §36)
│   ├── datasets/                  # download/prepare scripts
│   ├── runners/                   # run pipeline variants, collect metrics
│   ├── metrics/                   # WER/CER/RTF calculation (jiwer)
│   ├── reports/                   # output markdown/JSON reports
│   └── requirements.txt
├── models/                        # downloaded GGUF weights (gitignored) + pull script
├── scripts/                       # install / model-pull scripts
├── docs/
│   ├── ROADMAP.md                 # full §37 target monorepo layout + phase plan
│   └── superpowers/specs/         # this file
├── auralis_prd_end_to_end.md      # original PRD (already present)
├── Cargo.toml                     # Rust workspace (members: apps/desktop/src-tauri, runtime/core)
├── README.md
└── .gitignore
```

The Rust workspace has two members so pipeline logic lives in a reusable library
(`runtime/core`), not locked inside the Tauri app — matches PRD §51 ("don't couple the
model to the product").

## Data flow (push-to-talk)

```
Hold Ctrl+Space
  → cpal mic capture (16kHz mono) buffers while held
On release:
  → webrtc-vad trims leading/trailing silence from the held buffer
  → RNNoise (nnnoiseless) denoises the trimmed audio
  → whisper.cpp (whisper-rs, GGUF base.en, cuda feature) transcribes
  → local rule-based text cleanup: punctuation restoration, number/date
     normalization, capitalization, "actually change X to Y" correction detection
  → inject into previously-focused window via Windows SendInput, clipboard-paste fallback
```

Because push-to-talk explicitly brackets start/stop, VAD's Phase 1 job is trimming
silence within the held segment — continuous always-listening segmentation (PRD §6
"Continuous" mode) is out of scope.

## Desktop app specifics

- Global hotkey via the `global-hotkey` crate; tray icon + a small status window
  showing Listening/Processing/Idle and the last transcript.
- Settings (model choice, hotkey binding) persisted to a local JSON config in the
  Tauri app config dir.
- Windows-only; pipeline code sits behind a platform trait so macOS/Linux
  input-injection backends can be added later without touching pipeline logic.
- Default model: `base.en` GGUF, q5_1 quantization — swappable to tiny/small via config.

## Benchmark harness (Python)

- `benchmarks/datasets/prepare_librispeech.py`: pulls a small `test-clean` subset via
  Hugging Face `datasets`.
- `benchmarks/datasets/prepare_noisy.py`: synthetically mixes the same clips with a
  small bundled noise-clip set at controlled SNR (no registration-gated corpora).
- `benchmarks/runners/run_variant.py`: invokes a `auralis-bench-cli` binary (raw-file
  transcribe mode exposed from `runtime/core`) across model size (tiny/base/small) ×
  preprocessing (raw vs VAD+denoise) × dataset (clean vs noisy).
- `benchmarks/metrics/wer.py`: WER/CER via `jiwer`, RTF from wall-clock/audio-duration.
- `benchmarks/reports/`: markdown + JSON report table.
- `benchmarks/run_all.py`: single reproducible entrypoint. No SOTA claims — only
  measured numbers with the exact commands used (PRD §27).

## Error handling

- No mic / no input device → surfaced in tray UI, pipeline does not start.
- Missing model file → prompt to run the model-pull script (or confirm-then-download).
- STT failure/timeout → fails loud; no text is inserted; status shows "transcription
  failed" rather than silently inserting wrong text.
- Keyboard injection blocked by a protected window → falls back to clipboard copy +
  notification ("copied to clipboard, paste manually").

## Testing

- Rust: unit tests for text-cleanup rules (number normalization, correction-detection
  regex), VAD classification on synthetic silence/tone fixtures, and a pipeline
  integration test against a bundled short WAV fixture asserting a non-empty transcript.
- Python: benchmark harness smoke test on a 2-clip fixture set (no full dataset
  download required), asserting WER computation and report generation work.
- Manual: the desktop app is built and run live on this machine to confirm
  push-to-talk dictation actually works end-to-end (native app — not browser-testable).

## Known risk

Building `whisper-rs` with the `cuda` feature on Windows requires the CUDA toolkit, a
matching cuDNN, CMake, and MSVC Build Tools; Tauri needs Node/npm and (for a bundled
installer) WiX. The first implementation step verifies what's already present on this
machine and surfaces any gaps rather than assuming they exist.

## Explicitly out of scope for Phase 1

- Training any custom model architecture (PRD §45 Phase 1 proper, §8–10)
- Cloud/self-hosted compute fabric, scheduler, node agent (§16–18)
- Mobile clients (§7 Phase 7)
- Public website, Hugging Face release, observability stack (§23, §31, §34)
- Continuous/toggle dictation modes, context-aware vocabulary (§6, §13)
- Cloud LLM-based text polish (§39 "cloud mode must be explicit" — deferred entirely)
