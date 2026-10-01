<p align="center">
  <img src=".github/banner.png" alt="Auralis — real-time push-to-talk dictation" width="100%" />
</p>

<p align="center">
  <a href="https://github.com/Sagexd08/Auralis/actions/workflows/ci.yml">
    <img src="https://github.com/Sagexd08/Auralis/actions/workflows/ci.yml/badge.svg" alt="CI" />
  </a>
</p>

Push-to-talk dictation for Windows. Hold **Ctrl+Space**, speak, release — cleaned-up
transcribed text is inserted into whatever application currently has focus. Or tap
**Ctrl+Shift+Space** for hands-free continuous dictation.

Auralis runs entirely on-device: no cloud STT, no cloud LLM cleanup, no telemetry.

## How it works

```
MIC → VAD → DENOISER → STT → TEXT CLEANUP → KEYBOARD INSERTION
```

- **Mic capture** — [cpal](https://github.com/RustAudio/cpal), default input device
- **VAD** — [webrtc-vad](https://github.com/valenzuela/webrtc-vad) trims leading/trailing silence
- **Denoise** — RNNoise via [nnnoiseless](https://github.com/jneem/nnnoiseless) (pure Rust)
- **STT** — [whisper.cpp](https://github.com/ggerganov/whisper.cpp) via [whisper-rs](https://github.com/tazz4843/whisper-rs), quantized GGUF weights, beam search with temperature fallback, optional CUDA acceleration
- **Text cleanup** — local rules only: capitalization, terminal punctuation, and a spoken-correction heuristic ("actually, change X to Y")
- **Insertion** — simulated keystrokes via [enigo](https://github.com/enigo-rs/enigo), with a clipboard fallback. A spoken correction backspaces the text Auralis just typed and replaces it in place

### Accuracy knobs

The single biggest lever is model size. `base.en` (~59 MB) is the fast default;
`small.en` (~190 MB) is noticeably better on proper nouns and accented speech at
roughly 3x the decode cost:

```powershell
.\models\pull-model.ps1 small.en
```

Any downloaded model appears in the tray **Settings → Model** picker. Decoding
uses beam search (width 5) with whisper.cpp's temperature-fallback thresholds and
`no_context`, which together suppress the repetition loops that greedy decoding
with cross-utterance context is prone to.

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
- [x] Manual end-to-end verification (live mic, push-to-talk and continuous)

**Phase 2 — measurement**

- [x] WER/CER/RTF benchmark harness (`benchmarks/`) with a `auralis-bench-cli` runner

**Phase 3 — dictation UX**

- [x] Toggle/continuous dictation via VAD stream segmentation (Ctrl+Shift+Space)
- [x] Tray icon, settings window, persisted config (hotkeys, model, mic device)
- [x] In-place spoken correction — backspaces what Auralis typed and replaces it

**Phase 4 — shipping**

- [x] GitHub Actions CI (runtime tests + Linux Tauri build)
- [x] Brand assets (app icon, banner)
- [x] STT accuracy pass (beam search, no_context, temperature fallback)
- [ ] Signed Windows installer / release automation

## Building

Requires: Rust (stable, `x86_64-pc-windows-msvc`), Node.js + npm (for the Tauri
frontend), CMake, and a `libclang` compatible with `bindgen` 0.69 (needed to build
`whisper-rs-sys`, which compiles whisper.cpp from source). A CUDA Toolkit is only
needed for GPU acceleration.

**`libclang` note:** a too-new LLVM/clang install (e.g. 20+) produces broken bindgen
output for `whisper-rs-sys` — opaque structs with only an `_address` field, failing
with dozens of `no field ... on type whisper_full_params` errors. If you hit that,
install a compatible libclang and point `LIBCLANG_PATH` at it, e.g.:

```powershell
pip install --user libclang
setx LIBCLANG_PATH "%APPDATA%\Python\Python313\site-packages\clang\native"
# open a new terminal so the env var takes effect, then build
```

```powershell
# Rust library + tests
cargo build -p auralis-runtime
cargo test -p auralis-runtime

# Download the whisper.cpp model
powershell -File models/pull-model.ps1

# Desktop app
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
