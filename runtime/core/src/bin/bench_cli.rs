//! Benchmark CLI: transcribes a single WAV file with a chosen preprocessing
//! variant and prints the transcript to stdout. Invoked by the Python
//! benchmark harness (`benchmarks/runners/run_variant.py`), which measures
//! wall-clock time around the subprocess call itself and reads audio
//! duration independently — this binary's only job is "file in, transcript
//! out" so the harness can vary model/preprocessing/dataset without any
//! Rust changes.
use anyhow::{Context, Result};
use auralis_runtime::{denoise, quality, resample, stt::SttEngine, vad};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preprocess {
    /// Resample straight to 16kHz, no VAD trimming or denoising.
    Raw,
    /// Resample to 48kHz, VAD-trim silence, RNNoise-denoise, resample to 16kHz —
    /// the same chain the live desktop pipeline uses.
    VadDenoise,
}

#[derive(Parser, Debug)]
struct Args {
    /// Path to a GGUF whisper.cpp model file.
    #[arg(long)]
    model: PathBuf,

    /// Path to a WAV file (any sample rate; mono or multi-channel).
    #[arg(long)]
    input: PathBuf,

    /// Preprocessing variant to apply before transcription.
    #[arg(long, value_enum, default_value_t = Preprocess::Raw)]
    preprocess: Preprocess,

    /// Also report input audio quality (SNR, speech ratio, RMS, clipping) on
    /// stderr. Kept off stdout so the harness still reads exactly one line of
    /// transcript there.
    #[arg(long)]
    quality: bool,
}

fn read_wav_mono_f32(path: &PathBuf) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::open(path)
        .with_context(|| format!("failed to open WAV file {path:?}"))?;
    let spec = reader.spec();

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read f32 samples")?,
        hound::SampleFormat::Int => reader
            .samples::<i32>()
            .map(|s| s.map(|v| v as f32 / (1i64 << (spec.bits_per_sample - 1)) as f32))
            .collect::<Result<Vec<_>, _>>()
            .context("failed to read integer samples")?,
    };

    let mono: Vec<f32> = if spec.channels <= 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };

    Ok((mono, spec.sample_rate))
}

fn main() -> Result<()> {
    let args = Args::parse();

    let (samples, sample_rate) = read_wav_mono_f32(&args.input)?;

    if args.quality {
        let at_48k = resample::resample(&samples, sample_rate, 48_000);
        eprintln!("quality: {}", quality::analyze_48k(&at_48k).summary());
    }

    let samples_16k = match args.preprocess {
        Preprocess::Raw => resample::resample(&samples, sample_rate, 16_000),
        Preprocess::VadDenoise => {
            let at_48k = resample::resample(&samples, sample_rate, 48_000);
            let trimmed = vad::trim_silence(&at_48k);
            let denoised = denoise::denoise_48k(&trimmed);
            resample::resample(&denoised, 48_000, 16_000)
        }
    };

    let engine = SttEngine::load(&args.model)?;
    let transcript = engine.transcribe(&samples_16k)?;

    println!("{transcript}");
    Ok(())
}
