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
