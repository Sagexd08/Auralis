use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const BEAM_SIZE: i32 = 5;

pub struct SttEngine {
    context: WhisperContext,
}

impl SttEngine {
    pub fn load(model_path: &Path) -> Result<Self> {
        static LOG_HOOK: std::sync::Once = std::sync::Once::new();
        LOG_HOOK.call_once(whisper_rs::install_whisper_log_trampoline);

        let path_str = model_path
            .to_str()
            .context("model path is not valid UTF-8")?;
        let context = WhisperContext::new_with_params(path_str, WhisperContextParameters::default())
            .context("failed to load whisper model")?;
        Ok(Self { context })
    }

    pub fn transcribe(&self, samples_16k: &[f32]) -> Result<String> {
        let mut state = self.context.create_state().context("failed to create whisper state")?;

        let n_threads = std::thread::available_parallelism()
            .map(|n| n.get() as i32)
            .unwrap_or(4)
            .min(8);

        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: BEAM_SIZE,
            patience: 0.0,
        });
        params.set_n_threads(n_threads);
        params.set_print_progress(false);
        params.set_print_special(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);
        params.set_language(Some("en"));

        params.set_no_context(true);

        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);

        params.set_suppress_blank(true);
        params.set_suppress_non_speech_tokens(true);
        params.set_no_speech_thold(0.6);

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
