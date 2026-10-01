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
