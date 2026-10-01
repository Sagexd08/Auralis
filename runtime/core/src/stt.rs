use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

pub struct SttEngine {
    context: WhisperContext,
}

impl SttEngine {
    pub fn load(model_path: &Path) -> Result<Self> {
        let path_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let context = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .context("failed to load whisper model")?;
        Ok(Self { context })
    }

    /// Transcribes mono f32 PCM sampled at 16kHz (whisper.cpp's required input rate).
    pub fn transcribe(&self, samples_16k: &[f32]) -> Result<String> {
        let mut state = self.context.create_state().context("failed to create whisper state")?;

        // whisper.cpp defaults to min(4, hardware_concurrency) threads when unset,
        // which leaves most cores idle on anything beyond a 4-core machine. Use
        // up to 8 (diminishing returns past that for whisper.cpp's ggml threading).
        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8);

        let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
        params.set_n_threads(n_threads);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("en"));

        state
            .full(params, samples_16k)
            .context("whisper inference failed")?;

        let num_segments = state.full_n_segments().context("failed to read segment count")?;
        let mut text = String::new();
        for i in 0..num_segments {
            text.push_str(&state.full_get_segment_text(i).context("failed to read segment text")?);
        }

        Ok(text.trim().to_string())
    }
}
