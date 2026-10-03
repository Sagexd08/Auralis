use webrtc_vad::{SampleRate, Vad, VadMode};

pub const FRAME_SAMPLES: usize = 480;
const FRAME_MS: usize = 10;

fn new_vad() -> Vad {
    let mut vad = Vad::new();
    vad.set_sample_rate(SampleRate::Rate48kHz);
    vad.set_mode(VadMode::Aggressive);
    vad
}

fn to_vad_frame(frame: &[f32]) -> Vec<i16> {
    let mut padded: Vec<i16> = frame
        .iter()
        .map(|&s| (s.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    padded.resize(FRAME_SAMPLES, 0);
    padded
}

pub fn speech_flags_48k(samples: &[f32]) -> Vec<bool> {
    let mut vad = new_vad();
    samples
        .chunks(FRAME_SAMPLES)
        .map(|frame| vad.is_voice_segment(&to_vad_frame(frame)).unwrap_or(false))
        .collect()
}

pub fn trim_silence(samples: &[f32]) -> Vec<f32> {
    let speech_flags = speech_flags_48k(samples);

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

pub struct StreamSegmenter {
    vad: Vad,
    speech_started: bool,
    silence_run: usize,
    silence_frames_to_end: usize,
}

impl StreamSegmenter {
    pub fn new(trailing_silence_ms: u32) -> Self {
        Self {
            vad: new_vad(),
            speech_started: false,
            silence_run: 0,
            silence_frames_to_end: trailing_silence_ms as usize / FRAME_MS,
        }
    }

    pub fn feed(&mut self, frame: &[f32]) -> bool {
        let is_speech = self
            .vad
            .is_voice_segment(&to_vad_frame(frame))
            .unwrap_or(false);

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

    pub fn reset(&mut self) {
        self.vad = new_vad();
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
        let freq = 220.0f32;
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
        const HANGOVER_FRAMES: usize = 12;
        let mut input = silence(20);
        input.extend(tone(10));
        input.extend(silence(20));

        let trimmed = trim_silence(&input);

        assert!(!trimmed.is_empty());
        assert!(trimmed.len() >= tone(10).len());
        assert!(trimmed.len() <= tone(10).len() + HANGOVER_FRAMES * FRAME_SAMPLES);
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
        let mut stream = tone(10);
        stream.extend(silence(80));

        assert_eq!(feed_frames(&mut segmenter, &stream), 1);
    }

    #[test]
    fn segmenter_does_not_fire_on_short_pause_mid_speech() {
        let mut segmenter = StreamSegmenter::new(700);
        let mut stream = tone(10);
        stream.extend(silence(20));
        stream.extend(tone(10));

        assert_eq!(feed_frames(&mut segmenter, &stream), 0);
    }

    #[test]
    fn segmenter_detects_two_separate_utterances() {
        let mut segmenter = StreamSegmenter::new(700);
        let mut stream = tone(10);
        stream.extend(silence(80));
        stream.extend(tone(10));
        stream.extend(silence(80));

        assert_eq!(feed_frames(&mut segmenter, &stream), 2);
    }
}
