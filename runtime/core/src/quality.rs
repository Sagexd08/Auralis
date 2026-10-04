use crate::vad::{self, FRAME_SAMPLES};

const CLIP_THRESHOLD: f32 = 0.999;

#[derive(Debug, Clone, PartialEq)]
pub struct AudioQuality {
    pub snr_db: Option<f32>,
    pub speech_probability: f32,
    pub clipping: bool,
    pub rms: f32,
}

impl AudioQuality {
    pub fn summary(&self) -> String {
        let snr = match self.snr_db {
            Some(db) => format!("{db:.1}dB"),
            None => "n/a".to_string(),
        };
        format!(
            "snr={snr} speech={:.0}% rms={:.4}{}",
            self.speech_probability * 100.0,
            self.rms,
            if self.clipping { " CLIPPING" } else { "" }
        )
    }
}

pub fn analyze_48k(samples: &[f32]) -> AudioQuality {
    if samples.is_empty() {
        return AudioQuality {
            snr_db: None,
            speech_probability: 0.0,
            clipping: false,
            rms: 0.0,
        };
    }

    let clipping = samples.iter().any(|s| s.abs() >= CLIP_THRESHOLD);
    let rms = (samples.iter().map(|s| s * s).sum::<f32>() / samples.len() as f32).sqrt();

    let flags = vad::speech_flags_48k(samples);
    let frames: Vec<&[f32]> = samples.chunks(FRAME_SAMPLES).collect();

    let mut speech_power = 0.0f64;
    let mut speech_samples = 0usize;
    let mut noise_power = 0.0f64;
    let mut noise_samples = 0usize;

    for (frame, &is_speech) in frames.iter().zip(flags.iter()) {
        let power: f64 = frame.iter().map(|s| (*s as f64) * (*s as f64)).sum();
        if is_speech {
            speech_power += power;
            speech_samples += frame.len();
        } else {
            noise_power += power;
            noise_samples += frame.len();
        }
    }

    let speech_probability = if flags.is_empty() {
        0.0
    } else {
        flags.iter().filter(|&&f| f).count() as f32 / flags.len() as f32
    };

    let snr_db = if speech_samples == 0 || noise_samples == 0 {
        None
    } else {
        let speech_mean = speech_power / speech_samples as f64;
        let noise_mean = noise_power / noise_samples as f64;
        if noise_mean <= 0.0 || speech_mean <= 0.0 {
            None
        } else {
            Some((10.0 * (speech_mean / noise_mean).log10()) as f32)
        }
    };

    AudioQuality {
        snr_db,
        speech_probability,
        clipping,
        rms,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(num_frames: usize) -> Vec<f32> {
        vec![0.0; num_frames * FRAME_SAMPLES]
    }

    fn tone(num_frames: usize, amplitude: f32) -> Vec<f32> {
        let sr = 48_000.0f32;
        let freq = 220.0f32;
        (0..(num_frames * FRAME_SAMPLES))
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect()
    }

    fn noise(num_frames: usize, amplitude: f32) -> Vec<f32> {
        let mut state = 0x2545_F491_4F6C_DD1Du64;
        (0..(num_frames * FRAME_SAMPLES))
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32 / (u32::MAX >> 1) as f32) - 1.0;
                unit * amplitude
            })
            .collect()
    }

    #[test]
    fn empty_input_reports_nothing_measured() {
        let q = analyze_48k(&[]);
        assert_eq!(q.snr_db, None);
        assert_eq!(q.speech_probability, 0.0);
        assert!(!q.clipping);
        assert_eq!(q.rms, 0.0);
    }

    #[test]
    fn digital_silence_has_no_snr_and_no_speech() {
        let q = analyze_48k(&silence(20));
        assert_eq!(q.snr_db, None, "no speech frames means no honest SNR");
        assert_eq!(q.speech_probability, 0.0);
        assert_eq!(q.rms, 0.0);
    }

    #[test]
    fn detects_clipping_at_full_scale() {
        let mut samples = tone(5, 0.5);
        samples[1000] = 1.0;
        assert!(analyze_48k(&samples).clipping);
    }

    #[test]
    fn does_not_report_clipping_for_healthy_levels() {
        assert!(!analyze_48k(&tone(5, 0.5)).clipping);
    }

    #[test]
    fn rms_of_full_scale_square_wave_is_one() {
        let samples: Vec<f32> = (0..FRAME_SAMPLES * 4)
            .map(|i| if i % 2 == 0 { 0.5 } else { -0.5 })
            .collect();
        let q = analyze_48k(&samples);
        assert!((q.rms - 0.5).abs() < 1e-5, "expected rms 0.5, got {}", q.rms);
    }

    #[test]
    fn speech_over_silence_yields_a_positive_snr() {
        let mut samples = silence(20);
        samples.extend(noise(20, 0.0005));
        let mid = samples.len();
        samples.extend(tone(20, 0.6));
        assert!(mid > 0);

        let q = analyze_48k(&samples);
        assert!(q.speech_probability > 0.0, "expected the tone to register as speech");
        let snr = q.snr_db.expect("both speech and noise frames present");
        assert!(snr > 0.0, "loud speech over a quiet floor should be positive SNR, got {snr}");
    }

    #[test]
    fn louder_speech_at_the_same_noise_floor_raises_snr() {
        let build = |amplitude: f32| {
            let mut s = noise(20, 0.001);
            s.extend(tone(20, amplitude));
            s
        };

        let quiet = analyze_48k(&build(0.1)).snr_db.expect("quiet case has both frame kinds");
        let loud = analyze_48k(&build(0.8)).snr_db.expect("loud case has both frame kinds");
        assert!(loud > quiet, "expected louder speech to raise SNR: loud={loud} quiet={quiet}");
    }

    #[test]
    fn summary_renders_unmeasurable_snr_as_not_available() {
        let q = analyze_48k(&silence(10));
        assert!(q.summary().contains("snr=n/a"), "got {}", q.summary());
    }
}
