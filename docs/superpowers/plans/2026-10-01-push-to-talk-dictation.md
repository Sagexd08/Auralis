# Push-to-Talk Dictation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a working Windows desktop app where holding Ctrl+Space, speaking, and
releasing inserts cleaned-up transcribed text into whatever application currently has
focus — the full mic → VAD → denoise → STT → text-cleanup → keyboard-insertion vertical
slice from the Phase 1 design doc (`docs/superpowers/specs/2026-10-01-phase1-vertical-slice-design.md`).

**Architecture:** A reusable `runtime/core` Rust library owns the pipeline (capture,
resample, VAD trim, RNNoise denoise, whisper.cpp transcribe, text cleanup). A minimal
Tauri 2 app (`apps/desktop`) is tray-icon-only with no visible window during normal use
— this means the global hotkey never steals OS focus from the app the user is dictating
into, so no focus-restore logic is needed. On hotkey release, the app calls into
`runtime/core` and injects the result via simulated keystrokes.

**Tech Stack:** Rust (stable-x86_64-pc-windows-msvc), Tauri 2, cpal (mic capture),
rubato (resampling), webrtc-vad (speech trimming), nnnoiseless (RNNoise denoiser),
whisper-rs (whisper.cpp bindings, `cuda` feature), enigo (keystroke injection),
arboard (clipboard fallback), tauri-plugin-global-shortcut.

---

## File Structure

```
C:\Desktop\Auralis\
├── Cargo.toml                          # workspace root
├── runtime/core/
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs                      # re-exports
│       ├── resample.rs                 # sample-rate conversion (rubato)
│       ├── audio.rs                    # cpal mic capture
│       ├── vad.rs                      # webrtc-vad speech trimming
│       ├── denoise.rs                  # RNNoise via nnnoiseless
│       ├── stt.rs                      # whisper-rs wrapper
│       ├── text.rs                     # punctuation/normalization/correction rules
│       └── pipeline.rs                 # orchestrates the above
├── runtime/core/tests/
│   └── fixtures/jfk.wav                # known whisper.cpp test sample (downloaded)
├── apps/desktop/
│   ├── package.json
│   ├── src/                            # frontend (status window only)
│   │   └── index.html
│   └── src-tauri/
│       ├── Cargo.toml
│       ├── tauri.conf.json
│       └── src/
│           ├── main.rs
│           └── inject.rs               # keyboard injection + clipboard fallback
├── models/
│   └── pull-model.ps1                  # downloads ggml-base.en-q5_1.bin
└── scripts/
    └── verify-toolchain.ps1
```

---

## Task 1: Workspace skeleton

**Files:**
- Create: `Cargo.toml`
- Create: `runtime/core/Cargo.toml`
- Create: `runtime/core/src/lib.rs`
- Test: `runtime/core/src/lib.rs` (inline doctest-free unit test)

- [ ] **Step 1: Create the workspace root `Cargo.toml`**

```toml
[workspace]
resolver = "2"
members = [
    "runtime/core",
    "apps/desktop/src-tauri",
]
```

- [ ] **Step 2: Create `runtime/core/Cargo.toml`**

```toml
[package]
name = "auralis-runtime"
version = "0.1.0"
edition = "2021"

[lib]
name = "auralis_runtime"
path = "src/lib.rs"

[dependencies]
anyhow = "1"
thiserror = "1"

[dev-dependencies]
hound = "3.5"
```

- [ ] **Step 3: Create `runtime/core/src/lib.rs` with a placeholder test**

```rust
pub fn crate_version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_crate_version() {
        assert_eq!(crate_version(), "0.1.0");
    }
}
```

- [ ] **Step 4: Run the test to verify the workspace builds**

Run: `cargo test -p auralis-runtime`
Expected: `test tests::reports_crate_version ... ok`

- [ ] **Step 5: Commit**

```bash
git add Cargo.toml runtime/core/Cargo.toml runtime/core/src/lib.rs
git commit -m "chore: initialize Rust workspace with auralis-runtime crate"
```

---

## Task 2: Resampling utility

**Files:**
- Create: `runtime/core/src/resample.rs`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `rubato` dependency**

In `runtime/core/Cargo.toml`, add under `[dependencies]`:

```toml
rubato = "0.15"
```

- [ ] **Step 2: Write the failing test in `runtime/core/src/resample.rs`**

```rust
use rubato::{SincFixedIn, SincInterpolationParameters, SincInterpolationType, Resampler, WindowFunction};

/// Resamples mono f32 PCM from `from_hz` to `to_hz`. No-op if the rates match.
pub fn resample(input: &[f32], from_hz: u32, to_hz: u32) -> Vec<f32> {
    if from_hz == to_hz || input.is_empty() {
        return input.to_vec();
    }

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let ratio = to_hz as f64 / from_hz as f64;
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, input.len(), 1)
        .expect("valid resampler parameters");

    let waves_in = vec![input.to_vec()];
    let waves_out = resampler
        .process(&waves_in, None)
        .expect("resample succeeds");

    waves_out[0].clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_when_rates_match() {
        let input = vec![0.1, 0.2, 0.3];
        let out = resample(&input, 16_000, 16_000);
        assert_eq!(out, input);
    }

    #[test]
    fn downsamples_48k_to_16k_roughly_one_third_length() {
        // 480ms of a 100Hz test tone at 48kHz
        let sr = 48_000u32;
        let freq = 100.0f32;
        let input: Vec<f32> = (0..(sr / 2))
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr as f32).sin())
            .collect();

        let out = resample(&input, 48_000, 16_000);

        let expected_len = input.len() / 3;
        let tolerance = (expected_len as f64 * 0.05) as usize + 32;
        assert!(
            (out.len() as i64 - expected_len as i64).unsigned_abs() as usize <= tolerance,
            "expected ~{expected_len} samples, got {}",
            out.len()
        );
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod resample;
```

- [ ] **Step 4: Run the tests to verify they fail, then implement**

Run: `cargo test -p auralis-runtime resample`
Expected first run: FAIL (module not wired / function missing) — the code above already
contains the implementation, so this step and the next collapse into one verification
run for this module.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test -p auralis-runtime resample`
Expected: both `resample::tests::*` tests pass. If `rubato`'s API differs from the
version above (check with `cargo doc -p rubato --open` if it fails to compile), adjust
constructor arguments to match — the resampling algorithm choice (sinc, linear
interpolation) and no-op behavior must stay the same.

- [ ] **Step 6: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/resample.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add sample-rate resampling utility"
```

---

## Task 3: Mic capture

**Files:**
- Create: `runtime/core/src/audio.rs`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `cpal` and `crossbeam-channel` dependencies**

In `runtime/core/Cargo.toml`:

```toml
cpal = "0.15"
crossbeam-channel = "0.5"
```

- [ ] **Step 2: Write `runtime/core/src/audio.rs`**

```rust
use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{SampleFormat, Stream};
use crossbeam_channel::{Receiver, Sender};

/// Owns an open input stream. Dropping this stops capture.
pub struct AudioCapture {
    _stream: Stream,
    pub sample_rate: u32,
    receiver: Receiver<f32>,
}

impl AudioCapture {
    /// Opens the default input device's default mono-compatible config and starts
    /// streaming samples immediately. Samples arrive as mono f32 at `sample_rate`
    /// (the device's native rate — callers resample as needed).
    pub fn start() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("no default input (microphone) device found")?;
        let config = device
            .default_input_config()
            .context("failed to read default input config")?;

        let sample_rate = config.sample_rate().0;
        let channels = config.channels() as usize;
        let sample_format = config.sample_format();

        let (tx, rx): (Sender<f32>, Receiver<f32>) = crossbeam_channel::unbounded();

        let err_fn = |err| eprintln!("audio stream error: {err}");

        let stream = match sample_format {
            SampleFormat::F32 => device.build_input_stream(
                &config.into(),
                move |data: &[f32], _| send_mono(&tx, data, channels),
                err_fn,
                None,
            )?,
            SampleFormat::I16 => device.build_input_stream(
                &config.into(),
                move |data: &[i16], _| {
                    let floats: Vec<f32> = data.iter().map(|s| *s as f32 / i16::MAX as f32).collect();
                    send_mono(&tx, &floats, channels)
                },
                err_fn,
                None,
            )?,
            other => anyhow::bail!("unsupported input sample format: {other:?}"),
        };

        stream.play().context("failed to start input stream")?;

        Ok(Self {
            _stream: stream,
            sample_rate,
            receiver: rx,
        })
    }

    /// Drains whatever samples have arrived so far without blocking.
    pub fn drain_available(&self) -> Vec<f32> {
        self.receiver.try_iter().collect()
    }
}

fn send_mono(tx: &Sender<f32>, data: &[f32], channels: usize) {
    if channels <= 1 {
        for &s in data {
            let _ = tx.send(s);
        }
        return;
    }
    for frame in data.chunks(channels) {
        let avg = frame.iter().sum::<f32>() / channels as f32;
        let _ = tx.send(avg);
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod audio;
```

- [ ] **Step 4: Build to verify it compiles**

Run: `cargo build -p auralis-runtime`
Expected: compiles cleanly. There is no automated test here — `AudioCapture::start()`
opens a real OS microphone device, which isn't something a unit test should do (it
would hang/fail in CI or a machine with no mic permission granted). It gets exercised
for real in Task 8's manual end-to-end verification.

- [ ] **Step 5: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/audio.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add cpal-based microphone capture"
```

---

## Task 4: VAD-based silence trimming

**Files:**
- Create: `runtime/core/src/vad.rs`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `webrtc-vad` dependency**

In `runtime/core/Cargo.toml`:

```toml
webrtc-vad = "0.4"
```

- [ ] **Step 2: Write the failing test in `runtime/core/src/vad.rs`**

```rust
use webrtc_vad::{SampleRate, Vad, VadMode};

/// 48kHz, 10ms frames (480 samples) — matches both webrtc-vad's supported rates
/// and RNNoise's fixed frame size, so no per-frame resampling is needed upstream
/// of denoising.
const FRAME_SAMPLES: usize = 480;

/// Trims leading and trailing non-speech frames from a 48kHz mono f32 buffer.
/// Interior silence (between speech segments) is left untouched.
pub fn trim_silence(samples: &[f32]) -> Vec<f32> {
    let mut vad = Vad::new();
    vad.set_sample_rate(SampleRate::Rate48kHz);
    vad.set_mode(VadMode::Aggressive);

    let frames: Vec<&[f32]> = samples.chunks(FRAME_SAMPLES).collect();
    let speech_flags: Vec<bool> = frames
        .iter()
        .map(|frame| {
            let i16_frame: Vec<i16> = frame
                .iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            let mut padded = i16_frame.clone();
            padded.resize(FRAME_SAMPLES, 0);
            vad.is_voice_segment(&padded).unwrap_or(false)
        })
        .collect();

    let first_speech = speech_flags.iter().position(|&s| s);
    let last_speech = speech_flags.iter().rposition(|&s| s);

    match (first_speech, last_speech) {
        (Some(start), Some(end)) => {
            let start_sample = start * FRAME_SAMPLES;
            let end_sample = ((end + 1) * FRAME_SAMPLES).min(samples.len());
            samples[start_sample..end_sample].to_vec()
        }
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(num_frames: usize) -> Vec<f32> {
        vec![0.0; num_frames * FRAME_SAMPLES]
    }

    fn tone(num_frames: usize) -> Vec<f32> {
        let sr = 48_000.0f32;
        let freq = 220.0f32; // well within speech-band energy the VAD will flag
        (0..(num_frames * FRAME_SAMPLES))
            .map(|i| 0.6 * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    #[test]
    fn all_silence_trims_to_empty() {
        let input = silence(10);
        assert!(trim_silence(&input).is_empty());
    }

    #[test]
    fn trims_leading_and_trailing_silence() {
        let mut input = silence(5);
        input.extend(tone(10));
        input.extend(silence(5));

        let trimmed = trim_silence(&input);

        assert!(!trimmed.is_empty());
        assert!(trimmed.len() <= tone(10).len() + 2 * FRAME_SAMPLES);
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod vad;
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p auralis-runtime vad`
Expected: `all_silence_trims_to_empty` and `trims_leading_and_trailing_silence` pass.
If `webrtc-vad`'s enum names differ from `SampleRate::Rate48kHz` / `VadMode::Aggressive`
(check `cargo doc -p webrtc-vad --open`), adjust to match the installed version's API
— keep the 48kHz / 480-sample-frame / aggressive-mode choices.

- [ ] **Step 5: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/vad.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add VAD-based leading/trailing silence trimming"
```

---

## Task 5: RNNoise denoising

**Files:**
- Create: `runtime/core/src/denoise.rs`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `nnnoiseless` dependency**

In `runtime/core/Cargo.toml`:

```toml
nnnoiseless = "0.5"
```

- [ ] **Step 2: Write `runtime/core/src/denoise.rs`**

```rust
use nnnoiseless::DenoiseState;

/// Denoises 48kHz mono f32 PCM using RNNoise, frame by frame. RNNoise's internal
/// scale expects roughly int16-range magnitude, so samples are scaled up before
/// processing and back down afterward.
pub fn denoise_48k(samples: &[f32]) -> Vec<f32> {
    let frame_size = DenoiseState::FRAME_SIZE;
    let mut state = DenoiseState::new();
    let mut output = Vec::with_capacity(samples.len());

    let mut i = 0;
    while i < samples.len() {
        let end = (i + frame_size).min(samples.len());
        let mut in_frame = vec![0.0f32; frame_size];
        for (j, &s) in samples[i..end].iter().enumerate() {
            in_frame[j] = s * i16::MAX as f32;
        }

        let mut out_frame = vec![0.0f32; frame_size];
        state.process_frame(&mut out_frame, &in_frame);

        let valid = end - i;
        for &s in &out_frame[..valid] {
            output.push(s / i16::MAX as f32);
        }

        i = end;
    }

    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_length() {
        let input = vec![0.1f32; DenoiseState::FRAME_SIZE * 3 + 50];
        let output = denoise_48k(&input);
        assert_eq!(output.len(), input.len());
    }

    #[test]
    fn silence_stays_near_silent() {
        let input = vec![0.0f32; DenoiseState::FRAME_SIZE * 4];
        let output = denoise_48k(&input);
        let max_abs = output.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        assert!(max_abs < 0.05, "expected near-silence, got max abs {max_abs}");
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod denoise;
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p auralis-runtime denoise`
Expected: both tests pass. If `DenoiseState::new()` returns a `Box<DenoiseState>`
rather than `DenoiseState` directly (check the installed version's docs), adjust the
binding (`let mut state = DenoiseState::new();` still works with a boxed value since
`Box` derefs), and confirm `FRAME_SIZE` and `process_frame` signatures match.

- [ ] **Step 5: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/denoise.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add RNNoise-based denoising"
```

---

## Task 6: STT via whisper.cpp

**Files:**
- Create: `runtime/core/src/stt.rs`
- Create: `runtime/core/tests/fixtures/jfk.wav` (downloaded)
- Create: `runtime/core/tests/stt_integration.rs`
- Create: `models/pull-model.ps1`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `whisper-rs` dependency with the `cuda` feature**

In `runtime/core/Cargo.toml`:

```toml
whisper-rs = { version = "0.13", features = ["cuda"] }
```

If the CUDA Toolkit install from earlier in this session hasn't finished yet, first
build without the feature to keep making progress:

```toml
whisper-rs = "0.13"
```

and revisit Task 9 ("Enable CUDA") once `nvcc --version` succeeds.

- [ ] **Step 2: Write `runtime/core/src/stt.rs`**

```rust
use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct SttEngine {
    context: WhisperContext,
}

impl SttEngine {
    pub fn load(model_path: &Path) -> Result<Self> {
        let path_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let context = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .context("failed to load whisper model")?;
        Ok(Self { context })
    }

    /// Transcribes mono f32 PCM sampled at 16kHz (whisper.cpp's required input rate).
    pub fn transcribe(&self, samples_16k: &[f32]) -> Result<String> {
        let mut state = self.context.create_state().context("failed to create whisper state")?;

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("en"));

        state
            .full(params, samples_16k)
            .context("whisper inference failed")?;

        let num_segments = state.full_n_segments().context("failed to read segment count")?;
        let mut text = String::new();
        for i in 0..num_segments {
            text.push_str(&state.full_get_segment_text(i).context("failed to read segment text")?);
        }

        Ok(text.trim().to_string())
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod stt;
```

- [ ] **Step 4: Create the model download script `models/pull-model.ps1`**

```powershell
# Downloads the quantized base.en whisper.cpp model used by Auralis Phase 1.
$ErrorActionPreference = "Stop"
$modelDir = Join-Path $PSScriptRoot "."
$modelPath = Join-Path $modelDir "ggml-base.en-q5_1.bin"
$url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin"

if (Test-Path $modelPath) {
    Write-Host "Model already present at $modelPath"
    exit 0
}

Write-Host "Downloading base.en (q5_1) model to $modelPath ..."
Invoke-WebRequest -Uri $url -OutFile $modelPath
Write-Host "Done."
```

- [ ] **Step 5: Download the test fixture and the model**

Run:
```bash
mkdir -p runtime/core/tests/fixtures
curl -L -o runtime/core/tests/fixtures/jfk.wav https://github.com/ggerganov/whisper.cpp/raw/master/samples/jfk.wav
powershell -File models/pull-model.ps1
```
Expected: `runtime/core/tests/fixtures/jfk.wav` (~350KB) and `models/ggml-base.en-q5_1.bin`
(~60MB) both exist. `jfk.wav` is whisper.cpp's own canonical test sample (a short clip
of "Ask not what your country can do for you...", public domain), used here for the
same purpose their own test suite uses it: a known-good real-speech fixture.

- [ ] **Step 6: Write the failing integration test `runtime/core/tests/stt_integration.rs`**

```rust
use std::path::PathBuf;

#[test]
fn transcribes_known_jfk_sample() {
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jfk.wav");
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../models/ggml-base.en-q5_1.bin");

    if !model.exists() {
        eprintln!("skipping: model not found at {model:?}, run models/pull-model.ps1 first");
        return;
    }

    let mut reader = hound::WavReader::open(&fixture).expect("jfk.wav should be readable");
    let spec = reader.spec();
    assert_eq!(spec.sample_rate, 16_000, "jfk.wav is expected to already be 16kHz mono");
    assert_eq!(spec.channels, 1);

    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    let engine = auralis_runtime::stt::SttEngine::load(&model).expect("model loads");
    let text = engine.transcribe(&samples).expect("transcription succeeds");

    let lower = text.to_lowercase();
    assert!(
        lower.contains("country"),
        "expected the well-known JFK line to mention 'country', got: {text:?}"
    );
}
```

- [ ] **Step 7: Run the test to verify it fails, then run again after Step 2's code is in place**

Run: `cargo test -p auralis-runtime --test stt_integration -- --nocapture`
Expected: PASS, printing a transcript containing "country". First compile may take
several minutes since `whisper-rs` builds whisper.cpp from source via `cmake`. If
`WhisperContext::new_with_params`, `create_state`, `full`, `full_n_segments`, or
`full_get_segment_text` differ in signature (e.g. return `Option` instead of `Result`,
or take ownership differently) from the installed `whisper-rs` version, check
`cargo doc -p whisper-rs --open` and adjust `stt.rs` accordingly — the load-once /
transcribe-many shape (`SttEngine::load` then `SttEngine::transcribe`) must stay the
same since `pipeline.rs` in Task 8 depends on it.

- [ ] **Step 8: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/stt.rs runtime/core/src/lib.rs \
        runtime/core/tests/stt_integration.rs runtime/core/tests/fixtures/jfk.wav \
        models/pull-model.ps1
git commit -m "feat(runtime): add whisper.cpp STT wrapper with jfk.wav integration test"
```

(`models/*.bin` stays untracked per `.gitignore` — only the pull script is committed.)

---

## Task 7: Text cleanup rules

**Files:**
- Create: `runtime/core/src/text.rs`
- Modify: `runtime/core/src/lib.rs`
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Add `regex` and `once_cell` dependencies**

In `runtime/core/Cargo.toml`:

```toml
regex = "1"
once_cell = "1"
```

- [ ] **Step 2: Write `runtime/core/src/text.rs`**

```rust
use once_cell::sync::Lazy;
use regex::Regex;

/// Capitalizes the first letter and ensures a single terminal punctuation mark.
/// whisper.cpp's base.en model already emits punctuation/casing for most speech,
/// so this is a safety net rather than the primary source of punctuation.
pub fn clean_transcript(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut chars = trimmed.chars();
    let first = chars.next().unwrap().to_uppercase().to_string();
    let rest: String = chars.collect();
    let mut result = format!("{first}{rest}");

    if !result.ends_with(['.', '!', '?']) {
        result.push('.');
    }

    result
}

static CORRECTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(?:actually,?\s+)?change\s+(.+?)\s+to\s+(.+?)[\.\!\?]?$").unwrap()
});

/// Detects a spoken correction like "Actually, change Rahul to Rohan" against the
/// previous finalized transcript. Returns the revised text if `utterance` matches
/// the correction pattern and `find` is present (case-insensitively) in `previous`;
/// otherwise `None`, meaning the caller should treat `utterance` as new dictation.
pub fn detect_correction(previous: &str, utterance: &str) -> Option<String> {
    let caps = CORRECTION_PATTERN.captures(utterance.trim())?;
    let find = caps.get(1)?.as_str();
    let replace = caps.get(2)?.as_str().trim_end_matches(['.', '!', '?']);

    let lower_prev = previous.to_lowercase();
    let lower_find = find.to_lowercase();
    let pos = lower_prev.find(&lower_find)?;

    let mut result = previous.to_string();
    result.replace_range(pos..pos + find.len(), replace);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_and_adds_period() {
        assert_eq!(clean_transcript("hello there"), "Hello there.");
    }

    #[test]
    fn leaves_existing_terminal_punctuation() {
        assert_eq!(clean_transcript("is this working?"), "Is this working?");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(clean_transcript("   "), "");
    }

    #[test]
    fn detects_simple_correction() {
        let previous = "Send the report to Rahul tomorrow.";
        let revised = detect_correction(previous, "Actually, change Rahul to Rohan.");
        assert_eq!(
            revised,
            Some("Send the report to Rohan tomorrow.".to_string())
        );
    }

    #[test]
    fn non_correction_utterance_returns_none() {
        let previous = "Send the report to Rahul tomorrow.";
        assert_eq!(detect_correction(previous, "Also cc the design team."), None);
    }

    #[test]
    fn correction_target_not_found_returns_none() {
        let previous = "Send the report to Rahul tomorrow.";
        assert_eq!(detect_correction(previous, "change Priya to Rohan"), None);
    }
}
```

- [ ] **Step 3: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod text;
```

- [ ] **Step 4: Run the tests**

Run: `cargo test -p auralis-runtime text`
Expected: all 6 tests in `text::tests` pass.

- [ ] **Step 5: Commit**

```bash
git add runtime/core/Cargo.toml runtime/core/src/text.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add local text cleanup and correction detection"
```

---

## Task 8: Pipeline orchestration

**Files:**
- Create: `runtime/core/src/pipeline.rs`
- Modify: `runtime/core/src/lib.rs`

- [ ] **Step 1: Write `runtime/core/src/pipeline.rs`**

```rust
use crate::{audio::AudioCapture, denoise, resample, stt::SttEngine, text, vad};
use anyhow::Result;
use std::path::Path;
use std::time::Duration;

pub struct Pipeline {
    stt: SttEngine,
    last_transcript: Option<String>,
}

impl Pipeline {
    pub fn new(model_path: &Path) -> Result<Self> {
        Ok(Self {
            stt: SttEngine::load(model_path)?,
            last_transcript: None,
        })
    }

    /// Runs one push-to-talk cycle: captures audio from `capture` while `is_held`
    /// returns true (polled every 20ms), then runs it through
    /// resample -> VAD trim -> denoise -> resample -> STT -> text cleanup.
    /// If the cleaned utterance matches a correction pattern against the previous
    /// transcript, returns the *revised previous transcript* instead of the raw
    /// new text, and updates `last_transcript` to match.
    pub fn run_once(&mut self, capture: &AudioCapture, mut is_held: impl FnMut() -> bool) -> Result<String> {
        let mut raw_samples: Vec<f32> = Vec::new();
        while is_held() {
            raw_samples.extend(capture.drain_available());
            std::thread::sleep(Duration::from_millis(20));
        }
        raw_samples.extend(capture.drain_available());

        if raw_samples.is_empty() {
            return Ok(String::new());
        }

        let at_48k = resample::resample(&raw_samples, capture.sample_rate, 48_000);
        let trimmed = vad::trim_silence(&at_48k);
        if trimmed.is_empty() {
            return Ok(String::new());
        }
        let denoised = denoise::denoise_48k(&trimmed);
        let at_16k = resample::resample(&denoised, 48_000, 16_000);

        let raw_text = self.stt.transcribe(&at_16k)?;
        let cleaned = text::clean_transcript(&raw_text);

        if cleaned.is_empty() {
            return Ok(String::new());
        }

        let output = match &self.last_transcript {
            Some(prev) => match text::detect_correction(prev, &cleaned) {
                Some(revised) => revised,
                None => cleaned,
            },
            None => cleaned,
        };

        self.last_transcript = Some(output.clone());
        Ok(output)
    }
}
```

- [ ] **Step 2: Wire the module into `lib.rs`**

In `runtime/core/src/lib.rs`, add:

```rust
pub mod pipeline;
```

- [ ] **Step 3: Build to verify it compiles**

Run: `cargo build -p auralis-runtime`
Expected: compiles cleanly. `Pipeline::run_once` depends on a live `AudioCapture` and
a held-key poll closure, both of which only make sense wired to a real hotkey — it is
exercised for real in Task 12's manual verification, not by a unit test.

- [ ] **Step 4: Commit**

```bash
git add runtime/core/src/pipeline.rs runtime/core/src/lib.rs
git commit -m "feat(runtime): add pipeline orchestration for push-to-talk cycles"
```

---

## Task 9: Enable CUDA (if not already on from Task 6)

**Files:**
- Modify: `runtime/core/Cargo.toml`

- [ ] **Step 1: Verify the CUDA Toolkit install finished**

Run: `nvcc --version`
Expected: prints a CUDA compiler version (e.g. `release 13.4`). If this still fails,
stop here and finish the CUDA Toolkit install before continuing this task — the rest
of the plan does not require it (CPU inference from Task 6 already works), so it's
safe to skip this task for now and come back to it later.

- [ ] **Step 2: Enable the `cuda` feature**

In `runtime/core/Cargo.toml`, ensure:

```toml
whisper-rs = { version = "0.13", features = ["cuda"] }
```

- [ ] **Step 3: Rebuild and rerun the STT integration test**

Run: `cargo test -p auralis-runtime --test stt_integration -- --nocapture`
Expected: PASS, same "country" assertion as Task 6, now running on the GPU. First
build with the new feature will recompile whisper.cpp with CUDA support via cmake,
which takes longer than the CPU-only build.

- [ ] **Step 4: Commit**

```bash
git add runtime/core/Cargo.toml
git commit -m "feat(runtime): enable CUDA acceleration for whisper.cpp inference"
```

---

## Task 10: Scaffold the Tauri desktop app

**Files:**
- Create: `apps/desktop/package.json`
- Create: `apps/desktop/src/index.html`
- Create: `apps/desktop/src-tauri/Cargo.toml`
- Create: `apps/desktop/src-tauri/tauri.conf.json`
- Create: `apps/desktop/src-tauri/build.rs`
- Create: `apps/desktop/src-tauri/src/main.rs` (placeholder, replaced in Task 11)
- Modify: `Cargo.toml`

- [ ] **Step 1: Scaffold via the Tauri CLI**

Run from `apps/desktop/`:
```bash
npm create tauri-app@latest . -- --template vanilla --manager npm --yes
```
This generates `package.json`, `src/` (a minimal vanilla HTML/JS frontend), and
`src-tauri/` (a Rust binary crate with `Cargo.toml`, `tauri.conf.json`, `src/main.rs`,
`build.rs`, and an `icons/` directory). Accept the generated defaults; the next steps
edit specific files rather than replacing the scaffold.

- [ ] **Step 2: Rename the generated package and add it to the workspace**

In `apps/desktop/src-tauri/Cargo.toml`, set:

```toml
[package]
name = "auralis-desktop"
```

In the workspace root `Cargo.toml`, confirm `apps/desktop/src-tauri` is already listed
in `members` (it was added in Task 1).

- [ ] **Step 3: Replace the generated frontend with a minimal status page**

`apps/desktop/src/index.html`:

```html
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Auralis</title>
    <style>
      body { font-family: system-ui, sans-serif; background: #111; color: #eee; padding: 16px; }
      #status { font-size: 20px; font-weight: 600; }
      #transcript { margin-top: 12px; color: #aaa; white-space: pre-wrap; }
    </style>
  </head>
  <body>
    <div id="status">Idle</div>
    <div id="transcript"></div>
    <script type="module">
      import { listen } from "@tauri-apps/api/event";

      listen("auralis://status", (event) => {
        document.getElementById("status").textContent = event.payload;
      });
      listen("auralis://transcript", (event) => {
        document.getElementById("transcript").textContent = event.payload;
      });
    </script>
  </body>
</html>
```

- [ ] **Step 4: Verify the scaffold builds**

Run: `cd apps/desktop && npm install && npm run tauri build -- --debug`
Expected: build succeeds (may take a few minutes on first run) and produces a debug
binary under `apps/desktop/src-tauri/target/debug/`.

- [ ] **Step 5: Commit**

```bash
git add apps/desktop Cargo.toml
git commit -m "chore(desktop): scaffold Tauri 2 app"
```

---

## Task 11: Global hotkey, pipeline wiring, and keyboard injection

**Files:**
- Modify: `apps/desktop/src-tauri/Cargo.toml`
- Modify: `apps/desktop/src-tauri/src/main.rs`
- Create: `apps/desktop/src-tauri/src/inject.rs`
- Modify: `apps/desktop/src-tauri/tauri.conf.json`

- [ ] **Step 1: Add dependencies**

In `apps/desktop/src-tauri/Cargo.toml`, add:

```toml
[dependencies]
auralis-runtime = { path = "../../../runtime/core" }
tauri-plugin-global-shortcut = "2"
enigo = "0.2"
arboard = "3"
anyhow = "1"
```

(keep whatever `tauri` dependency line the scaffold already generated)

- [ ] **Step 2: Write `apps/desktop/src-tauri/src/inject.rs`**

```rust
use anyhow::Result;
use enigo::{Enigo, Keyboard, Settings};

/// Types `text` into whatever window currently has OS focus. Falls back to copying
/// to the clipboard (and returns a flag saying so) if keystroke injection fails,
/// e.g. because the focused window blocks synthetic input.
pub fn insert_text(text: &str) -> Result<bool> {
    let mut enigo = Enigo::new(&Settings::default())?;
    match enigo.text(text) {
        Ok(()) => Ok(true),
        Err(_) => {
            let mut clipboard = arboard::Clipboard::new()?;
            clipboard.set_text(text.to_string())?;
            Ok(false)
        }
    }
}
```

- [ ] **Step 3: Write `apps/desktop/src-tauri/src/main.rs`**

```rust
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod inject;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::Pipeline;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{Emitter, Manager};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/ggml-base.en-q5_1.bin")
}

fn main() {
    let held = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let model = model_path();
            if !model.exists() {
                eprintln!(
                    "Model not found at {model:?}. Run models/pull-model.ps1 first."
                );
            }

            let pipeline = Arc::new(Mutex::new(
                Pipeline::new(&model).expect("failed to load STT pipeline"),
            ));

            let app_handle = app.handle().clone();
            let held_for_handler = held.clone();

            let shortcut = Shortcut::new(Some(Modifiers::CONTROL), Code::Space);
            app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                match event.state() {
                    ShortcutState::Pressed => {
                        held_for_handler.store(true, Ordering::SeqCst);
                        let _ = app_handle.emit("auralis://status", "Listening");

                        let held_inner = held_for_handler.clone();
                        let pipeline_inner = pipeline.clone();
                        let handle_inner = app_handle.clone();

                        std::thread::spawn(move || {
                            let capture = match AudioCapture::start() {
                                Ok(c) => c,
                                Err(e) => {
                                    let _ = handle_inner.emit(
                                        "auralis://status",
                                        format!("Mic error: {e}"),
                                    );
                                    held_inner.store(false, Ordering::SeqCst);
                                    return;
                                }
                            };

                            let _ = handle_inner.emit("auralis://status", "Processing");

                            let result = {
                                let mut pipeline = pipeline_inner.lock().unwrap();
                                pipeline.run_once(&capture, || held_inner.load(Ordering::SeqCst))
                            };

                            match result {
                                Ok(text) if !text.is_empty() => {
                                    let _ = handle_inner.emit("auralis://transcript", &text);
                                    match crate::inject::insert_text(&text) {
                                        Ok(true) => {
                                            let _ = handle_inner.emit("auralis://status", "Idle");
                                        }
                                        Ok(false) => {
                                            let _ = handle_inner.emit(
                                                "auralis://status",
                                                "Idle (copied to clipboard)",
                                            );
                                        }
                                        Err(e) => {
                                            let _ = handle_inner.emit(
                                                "auralis://status",
                                                format!("Injection error: {e}"),
                                            );
                                        }
                                    }
                                }
                                Ok(_) => {
                                    let _ = handle_inner.emit("auralis://status", "Idle (no speech detected)");
                                }
                                Err(e) => {
                                    let _ = handle_inner.emit(
                                        "auralis://status",
                                        format!("Transcription failed: {e}"),
                                    );
                                }
                            }
                        });
                    }
                    ShortcutState::Released => {
                        held_for_handler.store(false, Ordering::SeqCst);
                    }
                }
            })?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running auralis-desktop");
}
```

- [ ] **Step 4: Ensure the app doesn't show a window on launch**

In `apps/desktop/src-tauri/tauri.conf.json`, under the window config, set:

```json
{
  "app": {
    "windows": [
      {
        "label": "main",
        "visible": false,
        "width": 480,
        "height": 320,
        "title": "Auralis"
      }
    ]
  }
}
```

This is what avoids the focus-stealing problem described in the design doc: the app
runs tray-only, so the OS-focused window never changes when the hotkey fires, and
`enigo`'s injected keystrokes land in whatever app the user was dictating into.

- [ ] **Step 5: Build**

Run: `cd apps/desktop && npm run tauri build -- --debug`
Expected: builds successfully. Errors here are most likely `tauri-plugin-global-shortcut`
or `tauri::Emitter`/`Manager` API mismatches against the exact Tauri 2.x point version
the scaffold generated — check `apps/desktop/src-tauri/Cargo.toml` for the resolved
`tauri` version and consult that version's docs (`cargo doc -p tauri --open`) to adjust
event-emission and plugin-registration calls if needed. The push-to-talk *behavior*
(press → capture → release → pipeline → inject) must stay the same.

- [ ] **Step 6: Commit**

```bash
git add apps/desktop/src-tauri Cargo.toml
git commit -m "feat(desktop): wire push-to-talk hotkey, pipeline, and keystroke injection"
```

---

## Task 12: Manual end-to-end verification

**Files:** none (verification only)

- [ ] **Step 1: Confirm the model and fixtures are present**

Run: `powershell -File models/pull-model.ps1`
Expected: "Model already present" (from Task 6) or a fresh download completes.

- [ ] **Step 2: Launch the app**

Run: `cd apps/desktop && npm run tauri dev`
Expected: process starts with no visible window; no panics in the terminal.

- [ ] **Step 3: Open Notepad (or any text field) and dictate**

- Open Notepad, click into the document so it has focus.
- Hold **Ctrl+Space**, say a short sentence out loud (e.g. "This is a test of Auralis
  dictation"), then release.
- Expected: within a few seconds, the spoken sentence appears in Notepad, capitalized
  and ending in punctuation (per Task 7's `clean_transcript`).

- [ ] **Step 4: Test the correction flow**

- Dictate: "Send the report to Rahul tomorrow." (release, wait for insertion)
- Dictate again: "Actually change Rahul to Rohan." (release)
- Expected: per the design's correction behavior, the app's transcript event should
  reflect the revised sentence. Note in your session notes whether the *inserted*
  text was also corrected in place in Notepad, or only a new line was appended — if
  Notepad wasn't corrected in place (Task 11's `main.rs` only emits the revised text
  and re-injects it as a fresh insertion, it does not delete-and-retype the original
  words already sitting in Notepad), record this as a known Phase 1 limitation rather
  than a bug: true in-place correction of already-inserted text requires tracking
  and deleting prior keystrokes, which is out of scope for this plan.

- [ ] **Step 5: Test with background noise**

- Play some background noise (music, a fan, typing) at moderate volume.
- Dictate a short sentence while the noise plays.
- Expected: transcript is still reasonably accurate — this is a qualitative check;
  rigorous noisy-speech accuracy measurement is the benchmark harness's job (a
  separate follow-up plan), not this one.

- [ ] **Step 6: Record results**

Add a short "Verification Results" section to the bottom of this plan file noting:
model used (CPU or CUDA build), whether each step passed, and any issues observed.
Commit that update.

```bash
git add docs/superpowers/plans/2026-10-01-push-to-talk-dictation.md
git commit -m "docs: record Phase 1 push-to-talk manual verification results"
```

---

## Verification Results

Automated verification (performed by the implementing agent, no human available for
voice input during this session):

- Model used: CPU build (`ggml-base.en-q5_1.bin`, 59.12 MB). CUDA (Task 9) stayed
  deferred — no CUDA Toolkit installed on this machine (`nvcc` not found).
- `cargo test -p auralis-runtime` — 13 unit tests + the `jfk.wav` STT integration
  test all pass. The integration test ran real whisper.cpp inference end-to-end and
  correctly transcribed "...ask not what your country can do for you, ask what you
  can do for your country." (verifying mic→...→STT minus only the live-mic and
  keyboard-injection legs).
- `npm run tauri build -- --debug` — builds and bundles cleanly (msi + nsis), zero
  warnings.
- Launched `auralis-desktop.exe` directly: whisper model loaded successfully
  (CPU, no GPU), global shortcut (Ctrl+Space) registered without error, no panics,
  clean shutdown. No visible window appeared at launch, confirming the
  no-focus-steal design.

**Not yet verified — requires a human with a working microphone:**
- Step 3: hold Ctrl+Space, speak, release, confirm text appears in Notepad.
- Step 4: the spoken-correction flow ("Actually change X to Y").
- Step 5: dictation accuracy with background noise.

These three steps need to be run manually before Phase 1 is considered fully done;
everything mechanically verifiable without a human voice has passed.

## Explicitly out of scope for this plan

- Benchmark harness (separate follow-up plan, per the design doc)
- Toggle/continuous dictation modes, only push-to-talk
- Tray icon UI polish, settings persistence, model-selection UI
- macOS/Linux support
- Cloud LLM text polish
- True in-place correction of already-inserted text (see Task 12, Step 4)
