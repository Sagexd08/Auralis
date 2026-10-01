use crate::{audio::AudioCapture, denoise, resample, stt::SttEngine, text, vad};
use anyhow::Result;
use log::debug;
use std::path::Path;
use std::time::{Duration, Instant};

/// How much trailing silence ends an utterance in continuous mode — long
/// enough that a normal mid-sentence breath doesn't split it, short enough
/// that sentence boundaries feel responsive.
const TRAILING_SILENCE_MS: u32 = 700;
/// How much native-rate audio to batch up before resampling it to 48kHz for
/// VAD classification in continuous mode. VAD timing precision only needs to
/// be good to within a fraction of `TRAILING_SILENCE_MS`, so batching avoids
/// re-initializing the sinc resampler on every tiny ~10-20ms capture chunk.
const CLASSIFY_CHUNK_MS: u64 = 300;
const VAD_FRAME_SAMPLES_48K: usize = 480;

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
    /// returns true (polled every 20ms), then transcribes the whole held buffer.
    pub fn run_once(&mut self, capture: &AudioCapture, mut is_held: impl FnMut() -> bool) -> Result<String> {
        let mut raw_samples: Vec<f32> = Vec::new();
        while is_held() {
            raw_samples.extend(capture.drain_available());
            std::thread::sleep(Duration::from_millis(20));
        }
        raw_samples.extend(capture.drain_available());

        self.process_utterance(&raw_samples, capture.sample_rate)
    }

    /// Runs continuous/toggle dictation: keeps capturing and auto-segmenting
    /// speech via VAD while `should_continue` returns true, calling
    /// `on_utterance` with each finalized result as soon as that utterance's
    /// trailing silence is detected — so callers can insert text
    /// incrementally instead of waiting for the whole session to end. Flushes
    /// one final in-progress utterance (if any) once `should_continue`
    /// returns false, then returns.
    pub fn run_continuous(
        &mut self,
        capture: &AudioCapture,
        mut should_continue: impl FnMut() -> bool,
        mut on_utterance: impl FnMut(Result<String>),
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

            let batch: Vec<f32> = classify_native.drain(..).collect();
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

    /// Resample -> VAD trim -> denoise -> resample -> STT -> text cleanup,
    /// shared by both push-to-talk and continuous mode.
    fn process_utterance(&mut self, raw_samples: &[f32], native_rate: u32) -> Result<String> {
        debug!(
            "captured {} samples @ {}Hz ({:.2}s)",
            raw_samples.len(),
            native_rate,
            raw_samples.len() as f32 / native_rate as f32
        );

        if raw_samples.is_empty() {
            return Ok(String::new());
        }

        let at_48k = resample::resample(raw_samples, native_rate, 48_000);
        debug!("resampled to 48k: {} samples ({:.2}s)", at_48k.len(), at_48k.len() as f32 / 48_000.0);

        let trimmed = vad::trim_silence(&at_48k);
        debug!("after VAD trim: {} samples ({:.2}s)", trimmed.len(), trimmed.len() as f32 / 48_000.0);
        if trimmed.is_empty() {
            debug!("VAD trimmed everything to silence — no speech detected, aborting");
            return Ok(String::new());
        }
        let denoised = denoise::denoise_48k(&trimmed);
        debug!("after denoise: {} samples", denoised.len());

        let at_16k = resample::resample(&denoised, 48_000, 16_000);
        debug!("resampled to 16k for STT: {} samples ({:.2}s)", at_16k.len(), at_16k.len() as f32 / 16_000.0);

        let stt_start = Instant::now();
        let raw_text = self.stt.transcribe(&at_16k)?;
        debug!("STT took {:.2}s, raw output: {raw_text:?}", stt_start.elapsed().as_secs_f32());

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
}
