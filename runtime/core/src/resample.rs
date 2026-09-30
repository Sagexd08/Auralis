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
