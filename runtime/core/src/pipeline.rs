use crate::{audio::AudioCapture, denoise, quality, resample, stt::SttEngine, text, vad};
use crate::text::{CleanupMode, CleanupModeHandle};
use crate::personalize::PersonalizationHandle;
use anyhow::Result;
use log::{debug, warn};
use std::path::Path;
use std::time::{Duration, Instant};

const TRAILING_SILENCE_MS: u32 = 700;
const CLASSIFY_CHUNK_MS: u64 = 300;
const VAD_FRAME_SAMPLES_48K: usize = 480;
pub const CORRECTION_WINDOW: Duration = Duration::from_secs(45);

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Transcript {
    Empty,
    Fresh(String),
    Correction(String),
}

pub struct Pipeline {
    stt: SttEngine,
    last_transcript: Option<String>,
    last_transcript_at: Option<Instant>,
    cleanup_mode: CleanupModeHandle,
    personalization: PersonalizationHandle,
}

impl Pipeline {
    pub fn new(model_path: &Path) -> Result<Self> {
        Self::with_cleanup_mode(model_path, CleanupModeHandle::default())
    }

    pub fn with_cleanup_mode(model_path: &Path, cleanup_mode: CleanupModeHandle) -> Result<Self> {
        Self::with_options(model_path, cleanup_mode, PersonalizationHandle::default())
    }

    pub fn with_options(
        model_path: &Path,
        cleanup_mode: CleanupModeHandle,
        personalization: PersonalizationHandle,
    ) -> Result<Self> {
        Ok(Self {
            stt: SttEngine::load(model_path)?,
            last_transcript: None,
            last_transcript_at: None,
            cleanup_mode,
            personalization,
        })
    }

    pub fn cleanup_mode(&self) -> CleanupMode {
        self.cleanup_mode.get()
    }

    pub fn record_while(capture: &AudioCapture, mut is_held: impl FnMut() -> bool) -> Vec<f32> {
        let mut raw_samples: Vec<f32> = Vec::new();
        while is_held() {
            raw_samples.extend(capture.drain_available());
            std::thread::sleep(Duration::from_millis(20));
        }
        raw_samples.extend(capture.drain_available());
        raw_samples
    }

    pub fn transcribe_recording(&mut self, raw_samples: &[f32], native_rate: u32) -> Result<Transcript> {
        self.process_utterance(raw_samples, native_rate)
    }

    pub fn run_once(&mut self, capture: &AudioCapture, is_held: impl FnMut() -> bool) -> Result<Transcript> {
        let raw_samples = Self::record_while(capture, is_held);
        self.process_utterance(&raw_samples, capture.sample_rate)
    }

    pub fn run_continuous(
        &mut self,
        capture: &AudioCapture,
        mut should_continue: impl FnMut() -> bool,
        mut on_utterance: impl FnMut(Result<Transcript>),
    ) {
        let mut segmenter = vad::StreamSegmenter::new(TRAILING_SILENCE_MS);
        let mut utterance_buf: Vec<f32> = Vec::new();
        let mut classify_native: Vec<f32> = Vec::new();
        let mut pending_48k: Vec<f32> = Vec::new();

        let classify_chunk_native_len = (capture.sample_rate as u64 * CLASSIFY_CHUNK_MS / 1000) as usize;

        while should_continue() {
            let chunk = capture.drain_available();
            if chunk.is_empty() {
                std::thread::sleep(Duration::from_millis(20));
                continue;
            }

            utterance_buf.extend_from_slice(&chunk);
            classify_native.extend_from_slice(&chunk);

            if classify_native.len() < classify_chunk_native_len {
                continue;
            }

            let batch: Vec<f32> = std::mem::take(&mut classify_native);
            pending_48k.extend(resample::resample(&batch, capture.sample_rate, 48_000));

            let mut boundary_hit = false;
            while pending_48k.len() >= VAD_FRAME_SAMPLES_48K {
                let frame: Vec<f32> = pending_48k.drain(..VAD_FRAME_SAMPLES_48K).collect();
                if segmenter.feed(&frame) {
                    boundary_hit = true;
                    break;
                }
            }

            if boundary_hit {
                debug!("[continuous] utterance boundary detected, flushing {} native samples", utterance_buf.len());
                let result = self.process_utterance(&utterance_buf, capture.sample_rate);
                on_utterance(result);
                utterance_buf.clear();
                classify_native.clear();
                pending_48k.clear();
                segmenter.reset();
            }
        }

        if !utterance_buf.is_empty() {
            debug!("[continuous] stopped mid-utterance, flushing {} remaining native samples", utterance_buf.len());
            let result = self.process_utterance(&utterance_buf, capture.sample_rate);
            on_utterance(result);
        }
    }

    fn process_utterance(&mut self, raw_samples: &[f32], native_rate: u32) -> Result<Transcript> {
        debug!(
            "captured {} samples @ {}Hz ({:.2}s)",
            raw_samples.len(),
            native_rate,
            raw_samples.len() as f32 / native_rate as f32
        );

        if raw_samples.is_empty() {
            return Ok(Transcript::Empty);
        }

        let at_48k = resample::resample(raw_samples, native_rate, 48_000);
        debug!("resampled to 48k: {} samples ({:.2}s)", at_48k.len(), at_48k.len() as f32 / 48_000.0);

        let input_quality = quality::analyze_48k(&at_48k);
        debug!("input audio quality: {}", input_quality.summary());
        if input_quality.clipping {
            warn!("input audio is clipping — lower the microphone gain for better accuracy");
        }

        let trimmed = vad::trim_silence(&at_48k);
        debug!("after VAD trim: {} samples ({:.2}s)", trimmed.len(), trimmed.len() as f32 / 48_000.0);
        if trimmed.is_empty() {
            debug!("VAD trimmed everything to silence — no speech detected, aborting");
            return Ok(Transcript::Empty);
        }
        let denoised = denoise::denoise_48k(&trimmed);
        debug!("after denoise: {} samples", denoised.len());

        let at_16k = resample::resample(&denoised, 48_000, 16_000);
        debug!("resampled to 16k for STT: {} samples ({:.2}s)", at_16k.len(), at_16k.len() as f32 / 16_000.0);

        let stt_start = Instant::now();
        let raw_text = self.stt.transcribe(&at_16k)?;
        debug!("STT took {:.2}s, raw output: {raw_text:?}", stt_start.elapsed().as_secs_f32());

        let mode = self.cleanup_mode.get();
        let personalization = self.personalization.get();
        let cleaned = match personalization.snippet_for(&raw_text).filter(|_| mode != CleanupMode::Raw) {
            Some(snippet) => snippet,
            None => personalization.finish(&text::clean_transcript_with(&raw_text, mode), mode),
        };
        if cleaned.is_empty() {
            return Ok(Transcript::Empty);
        }

        if Self::is_stale(self.last_transcript_at, Instant::now()) {
            self.last_transcript = None;
        }
        let transcript = Self::classify_output(&mut self.last_transcript, cleaned);
        self.last_transcript_at = Some(Instant::now());
        Ok(transcript)
    }

    fn is_stale(last_at: Option<Instant>, now: Instant) -> bool {
        last_at.is_some_and(|at| now.duration_since(at) > CORRECTION_WINDOW)
    }

    fn classify_output(prev: &mut Option<String>, cleaned: String) -> Transcript {
        let transcript = match prev.as_deref().and_then(|p| text::detect_correction(p, &cleaned)) {
            Some(revised) => Transcript::Correction(revised),
            None => Transcript::Fresh(cleaned),
        };

        *prev = Some(match &transcript {
            Transcript::Fresh(t) | Transcript::Correction(t) => t.clone(),
            Transcript::Empty => unreachable!("cleaned was checked non-empty above"),
        });

        transcript
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classify_chunk_native_len_is_sane_at_common_sample_rates() {
        for sample_rate in [16_000u32, 44_100, 48_000] {
            let len = (sample_rate as u64 * CLASSIFY_CHUNK_MS / 1000) as usize;
            assert!(len > 0, "classify chunk length must be non-zero at {sample_rate}Hz");
        }
    }

    #[test]
    fn corrections_expire_after_the_window() {
        let now = Instant::now();
        assert!(!Pipeline::is_stale(None, now));
        assert!(!Pipeline::is_stale(Some(now), now));
        assert!(!Pipeline::is_stale(Some(now), now + CORRECTION_WINDOW));
        assert!(Pipeline::is_stale(Some(now), now + CORRECTION_WINDOW + Duration::from_secs(1)));
    }

    #[test]
    fn classify_output_first_utterance_is_always_fresh() {
        let mut prev = None;
        let out = Pipeline::classify_output(&mut prev, "Send the report to Rahul tomorrow.".to_string());
        assert_eq!(out, Transcript::Fresh("Send the report to Rahul tomorrow.".to_string()));
        assert_eq!(prev.as_deref(), Some("Send the report to Rahul tomorrow."));
    }

    #[test]
    fn classify_output_detects_correction_against_previous() {
        let mut prev = Some("Send the report to Rahul tomorrow.".to_string());
        let out = Pipeline::classify_output(&mut prev, "Actually, change Rahul to Rohan.".to_string());
        assert_eq!(out, Transcript::Correction("Send the report to Rohan tomorrow.".to_string()));
        assert_eq!(prev.as_deref(), Some("Send the report to Rohan tomorrow."));
    }

    #[test]
    fn classify_output_non_correction_is_fresh_and_replaces_prev() {
        let mut prev = Some("Send the report to Rahul tomorrow.".to_string());
        let out = Pipeline::classify_output(&mut prev, "Also cc the design team.".to_string());
        assert_eq!(out, Transcript::Fresh("Also cc the design team.".to_string()));
        assert_eq!(prev.as_deref(), Some("Also cc the design team."));
    }

    #[test]
    fn classify_output_chains_two_corrections() {
        let mut prev = Some("Send the report to Rahul tomorrow.".to_string());
        let first = Pipeline::classify_output(&mut prev, "change Rahul to Rohan".to_string());
        assert_eq!(first, Transcript::Correction("Send the report to Rohan tomorrow.".to_string()));
        let second = Pipeline::classify_output(&mut prev, "change tomorrow to Friday".to_string());
        assert_eq!(second, Transcript::Correction("Send the report to Rohan Friday.".to_string()));
    }
}
