# Auralis

Push-to-talk dictation for Windows. Hold **Ctrl+Space**, speak, release — cleaned-up
transcribed text is inserted into whatever application currently has focus.

Auralis runs entirely on-device: no cloud STT, no cloud LLM cleanup, no telemetry.

## How it works

```
MIC → VAD → DENOISER → STT → TEXT CLEANUP → KEYBOARD INSERTION
```

- **Mic capture** — [cpal](https://github.com/RustAudio/cpal), default input device
- **VAD** — [webrtc-vad](https://github.com/valenzuela/webrtc-vad) trims leading/trailing silence
- **Denoise** — RNNoise via [nnnoiseless](https://github.com/jneem/nnnoiseless) (pure Rust)
- **STT** — [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [whisper-rs](https://github.com/tazz4843/whisper-rs), quantized `base.en` GGUF weights, optional CUDA acceleration
- **Text cleanup** — local rules only: capitalization, terminal punctuation, and a simple spoken-correction heuristic ("actually, change X to Y")
- **Insertion** — simulated keystrokes via [enigo](https://github.com/enigo-rs/enigo), with a clipboard fallback

This is the Phase 1 vertical slice described in
[`docs/superpowers/specs/2026-10-01-phase1-vertical-slice-design.md`](docs/superpowers/specs/2026-10-01-phase1-vertical-slice-design.md):
a real, daily-usable dictation tool built on an existing open STT model, not a
custom-trained model or the full distributed system described in the PRD
(`auralis_prd_end_to_end.md`).

## Repo layout

```
runtime/core/         auralis-runtime — reusable Rust library (capture, VAD, denoise, STT, text cleanup, pipeline)
apps/desktop/          Tauri 2 desktop app — tray-only, global hotkey, keystroke injection
benchmarks/            Python WER/CER/RTF harness — does VAD+denoise actually help? (see benchmarks/README.md)
models/                downloaded GGUF model weights (gitignored) + pull-model.ps1
docs/superpowers/      design doc and implementation plan for this phase
```

The Rust workspace has two members so pipeline logic lives in `runtime/core`
independent of the Tauri app shell — see the design doc for the full rationale.

## Status

Phase 1 is implemented task-by-task per
[`docs/superpowers/plans/2026-10-01-push-to-talk-dictation.md`](docs/superpowers/plans/2026-10-01-push-to-talk-dictation.md).
Current progress:

- [x] Workspace skeleton
- [x] Resampling utility (rubato)
- [x] Mic capture (cpal)
- [x] VAD-based silence trimming
- [x] RNNoise denoising
- [x] STT via whisper.cpp
- [x] Text cleanup rules
- [x] Pipeline orchestration
- [ ] CUDA acceleration (deferred — no CUDA Toolkit on this machine; CPU build works)
- [x] Tauri desktop app scaffold
- [x] Global hotkey, pipeline wiring, keyboard injection
- [ ] Manual end-to-end verification (needs a human with a mic — see plan doc)

## Building

Requires: Rust (stable, `x86_64-pc-windows-msvc`), Node.js + npm (for the Tauri
frontend), and a CUDA Toolkit install if building with GPU acceleration.

```powershell
# Rust library + tests
cargo build -p auralis-runtime
cargo test -p auralis-runtime

# Download the whisper.cpp model
powershell -File models/pull-model.ps1

# Desktop app (once Task 10+ land)
cd apps/desktop
npm install
npm run tauri dev
```

## Benchmark harness (Phase 2)

Measures whether VAD+denoise actually helps STT accuracy rather than assuming
it does — WER/CER/RTF across model size x raw/vad-denoise x clean/noisy. See
[`benchmarks/README.md`](benchmarks/README.md).

## Out of scope for Phase 1

- Toggle/continuous dictation modes — push-to-talk only
- Tray icon UI polish, settings persistence, model-selection UI
- macOS/Linux support
- Cloud LLM text polish
- True in-place correction of already-inserted text
