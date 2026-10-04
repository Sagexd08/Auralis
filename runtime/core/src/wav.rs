use anyhow::{Context, Result};
use std::io::Read;

pub fn decode_wav_mono_f32(source: impl Read) -> Result<(Vec<f32>, u32)> {
    let mut reader = hound::WavReader::new(source).context("not a readable WAV file")?;
    let spec = reader.spec();
    if spec.channels == 0 || spec.bits_per_sample == 0 || spec.bits_per_sample > 32 {
        anyhow::bail!("unsupported WAV format: {spec:?}");
    }

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

    let mono = if spec.channels == 1 {
        samples
    } else {
        samples
            .chunks(spec.channels as usize)
            .map(|frame| frame.iter().sum::<f32>() / spec.channels as f32)
            .collect()
    };
    Ok((mono, spec.sample_rate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn wav_bytes(channels: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            for s in samples {
                writer.write_sample(*s).unwrap();
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[test]
    fn decodes_mono_and_scales_to_unit_range() {
        let (samples, rate) = decode_wav_mono_f32(Cursor::new(wav_bytes(1, &[0, i16::MAX, i16::MIN]))).unwrap();
        assert_eq!(rate, 16_000);
        assert_eq!(samples.len(), 3);
        assert!((samples[1] - 1.0).abs() < 0.001);
        assert!((samples[2] + 1.0).abs() < 0.001);
    }

    #[test]
    fn averages_stereo_to_mono() {
        let (samples, _) = decode_wav_mono_f32(Cursor::new(wav_bytes(2, &[16384, -16384, 16384, 16384]))).unwrap();
        assert_eq!(samples.len(), 2);
        assert!(samples[0].abs() < 0.001);
        assert!((samples[1] - 0.5).abs() < 0.001);
    }

    #[test]
    fn rejects_non_wav_input() {
        assert!(decode_wav_mono_f32(Cursor::new(b"definitely not audio".to_vec())).is_err());
    }
}
