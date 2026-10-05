use anyhow::{Context, Result};
use auralis_runtime::{denoise, quality, resample, stt::{join_segments, load_engine}, vad};
use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Preprocess {
    Raw,
    VadDenoise,
}

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long)]
    input: Option<PathBuf>,

    #[arg(long)]
    batch: Option<PathBuf>,

    #[arg(long, value_enum, default_value_t = Preprocess::Raw)]
    preprocess: Preprocess,

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

fn prepare(path: &PathBuf, preprocess: Preprocess, report_quality: bool) -> Result<Vec<f32>> {
    let (samples, sample_rate) = read_wav_mono_f32(path)?;

    if report_quality {
        let at_48k = resample::resample(&samples, sample_rate, 48_000);
        eprintln!("quality: {}", quality::analyze_48k(&at_48k).summary());
    }

    Ok(match preprocess {
        Preprocess::Raw => resample::resample(&samples, sample_rate, 16_000),
        Preprocess::VadDenoise => {
            let at_48k = resample::resample(&samples, sample_rate, 48_000);
            let trimmed = vad::trim_silence(&at_48k);
            let denoised = denoise::denoise_48k(&trimmed);
            resample::resample(&denoised, 48_000, 16_000)
        }
    })
}

fn main() -> Result<()> {
    let args = Args::parse();
    let engine = load_engine(&args.model)?;

    if let Some(list) = &args.batch {
        let text = std::fs::read_to_string(list).with_context(|| format!("failed to read batch list {list:?}"))?;
        for line in text.lines().map(str::trim).filter(|l| !l.is_empty()) {
            let path = PathBuf::from(line);
            let samples_16k = prepare(&path, args.preprocess, false)?;
            let transcript = join_segments(&engine.segments(&samples_16k, None)?);
            println!("{line}	{transcript}");
        }
        return Ok(());
    }

    let input = args.input.context("pass --input <wav> or --batch <list>")?;
    let samples_16k = prepare(&input, args.preprocess, args.quality)?;
    let transcript = join_segments(&engine.segments(&samples_16k, None)?);

    println!("{transcript}");
    Ok(())
}
