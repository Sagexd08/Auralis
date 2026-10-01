use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

/// Beam width for decoding. 5 is whisper.cpp's own default for beam search and
/// the point where WER gains flatten out against the added decode cost.
const BEAM_SIZE: i32 = 5;

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

        // Beam search beats greedy decoding on word error rate; the extra cost
        // lands on decode, which is a small fraction of total time for the
        // short (<30s) utterances dictation produces.
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

        // Each utterance is independent dictation, so carrying the previous
        // window's decoded text forward as a prompt only invites the model to
        // continue a sentence that already ended — the cause of the
        // "who is expected? who is expected?" style repetition loops.
        params.set_no_context(true);

        // Temperature fallback: retry a window at increasing temperature when
        // the greedy/beam result looks degenerate (low average logprob or high
        // token entropy, i.e. a repetition loop). These thresholds are
        // whisper.cpp's own upstream defaults.
        params.set_temperature(0.0);
        params.set_temperature_inc(0.2);
        params.set_entropy_thold(2.4);
        params.set_logprob_thold(-1.0);

        // Dictation wants words, not transcribed room noise: suppress the
        // "(door closes)" / "[BLANK_AUDIO]" / "♪" class of tokens, and raise
        // the bar for calling a near-silent window speech at all.
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
