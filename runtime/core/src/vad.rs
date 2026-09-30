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
        // webrtc-vad has built-in hangover: after a speech segment ends, it keeps
        // flagging a handful of subsequent frames as "speech" for a few frames
        // (~8 frames / 80ms observed empirically) before its internal state
        // settles back to silence. The leading/trailing silence blocks here are
        // sized well beyond that hangover window so the trimmed result still
        // demonstrates genuine trimming on both ends, and the tolerance below
        // accounts for the hangover rather than assuming an exact cut at the
        // tone boundary.
        const HANGOVER_FRAMES: usize = 12;
        let mut input = silence(20);
        input.extend(tone(10));
        input.extend(silence(20));

        let trimmed = trim_silence(&input);

        assert!(!trimmed.is_empty());
        // Must retain all real speech.
        assert!(trimmed.len() >= tone(10).len());
        // Must not retain more than the speech plus a bounded hangover margin.
        assert!(trimmed.len() <= tone(10).len() + HANGOVER_FRAMES * FRAME_SAMPLES);
        // Must be meaningfully shorter than the untrimmed input (i.e. trimming
        // actually happened on both ends).
        assert!(trimmed.len() < input.len());
    }
}
