use webrtc_vad::{SampleRate, Vad, VadMode};

/// 48kHz, 10ms frames (480 samples) — matches both webrtc-vad's supported rates
/// and RNNoise's fixed frame size, so no per-frame resampling is needed upstream
/// of denoising.
const FRAME_SAMPLES: usize = 480;
const FRAME_MS: usize = 10;

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

/// Detects utterance boundaries in a continuous stream of 48kHz frames, for
/// continuous/toggle dictation mode (as opposed to `trim_silence`, which
/// trims a single already-complete held buffer). Feed it one 480-sample
/// frame at a time; once real speech has been seen and then enough trailing
/// silence follows, `feed` returns `true` exactly once, signaling the caller
/// to flush the accumulated audio as one utterance and call `reset` before
/// continuing to feed frames for the next one.
pub struct StreamSegmenter {
    vad: Vad,
    speech_started: bool,
    silence_run: usize,
    silence_frames_to_end: usize,
}

impl StreamSegmenter {
    /// `trailing_silence_ms` is how much silence after speech ends an
    /// utterance — e.g. 700ms is a natural end-of-sentence pause without
    /// being so short it splits a normal mid-sentence breath.
    pub fn new(trailing_silence_ms: u32) -> Self {
        let mut vad = Vad::new();
        vad.set_sample_rate(SampleRate::Rate48kHz);
        vad.set_mode(VadMode::Aggressive);

        Self {
            vad,
            speech_started: false,
            silence_run: 0,
            silence_frames_to_end: trailing_silence_ms as usize / FRAME_MS,
        }
    }

    /// Feed exactly one 480-sample (10ms) 48kHz mono f32 frame. Returns
    /// `true` when this frame completes an utterance boundary.
    pub fn feed(&mut self, frame: &[f32]) -> bool {
        let i16_frame: Vec<i16> = frame
            .iter()
            .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
            .collect();
        let mut padded = i16_frame;
        padded.resize(FRAME_SAMPLES, 0);
        let is_speech = self.vad.is_voice_segment(&padded).unwrap_or(false);

        if is_speech {
            self.speech_started = true;
            self.silence_run = 0;
            false
        } else if self.speech_started {
            self.silence_run += 1;
            self.silence_run >= self.silence_frames_to_end
        } else {
            false
        }
    }

    /// Call after `feed` returns `true` and the flushed segment has been
    /// handed off, to start cleanly tracking the next utterance.
    pub fn reset(&mut self) {
        let mut vad = Vad::new();
        vad.set_sample_rate(SampleRate::Rate48kHz);
        vad.set_mode(VadMode::Aggressive);
        self.vad = vad;
        self.speech_started = false;
        self.silence_run = 0;
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

    fn feed_frames(segmenter: &mut StreamSegmenter, samples: &[f32]) -> usize {
        let mut boundaries = 0;
        for frame in samples.chunks(FRAME_SAMPLES) {
            if frame.len() == FRAME_SAMPLES && segmenter.feed(frame) {
                boundaries += 1;
                segmenter.reset();
            }
        }
        boundaries
    }

    #[test]
    fn segmenter_stays_quiet_through_pure_silence() {
        let mut segmenter = StreamSegmenter::new(700);
        assert_eq!(feed_frames(&mut segmenter, &silence(50)), 0);
    }

    #[test]
    fn segmenter_fires_once_after_speech_then_enough_trailing_silence() {
        let mut segmenter = StreamSegmenter::new(700);
        // 700ms trailing silence threshold = 70 frames.
        let mut stream = tone(10);
        stream.extend(silence(80));

        assert_eq!(feed_frames(&mut segmenter, &stream), 1);
    }

    #[test]
    fn segmenter_does_not_fire_on_short_pause_mid_speech() {
        let mut segmenter = StreamSegmenter::new(700);
        // A short (<700ms) pause between two spoken words shouldn't split the utterance.
        let mut stream = tone(10);
        stream.extend(silence(20)); // 200ms pause
        stream.extend(tone(10));

        assert_eq!(feed_frames(&mut segmenter, &stream), 0);
    }

    #[test]
    fn segmenter_detects_two_separate_utterances() {
        let mut segmenter = StreamSegmenter::new(700);
        let mut stream = tone(10);
        stream.extend(silence(80)); // boundary 1
        stream.extend(tone(10));
        stream.extend(silence(80)); // boundary 2

        assert_eq!(feed_frames(&mut segmenter, &stream), 2);
    }
}
