use anyhow::{Context, Result};
use std::path::Path;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const BEAM_SIZE: i32 = 5;

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub start_s: f64,
    pub end_s: f64,
    pub text: String,
}

pub trait Speech: Send + Sync {
    fn segments(&self, samples_16k: &[f32], language: Option<&str>) -> Result<Vec<Segment>>;
    fn languages(&self) -> Vec<String>;
    fn engine_name(&self) -> &'static str;
}

pub fn load_engine(model_path: &Path) -> Result<Box<dyn Speech>> {
    let ext = model_path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default();
    match ext.as_str() {
        "onnx" => Ok(Box::new(crate::auralis::AuralisEngine::load(model_path)?)),
        _ => Ok(Box::new(SttEngine::load(model_path)?)),
    }
}

pub fn join_segments(segments: &[Segment]) -> String {
    segments.iter().map(|s| s.text.as_str()).collect::<String>().trim().to_string()
}

pub struct SttEngine {
    context: WhisperContext,
    english_only: bool,
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
        let english_only = model_path
            .file_name()
            .map(|n| n.to_string_lossy().contains(".en"))
            .unwrap_or(false);
        Ok(Self { context, english_only })
    }

    pub fn transcribe(&self, samples_16k: &[f32]) -> Result<String> {
        Ok(join_segments(&self.segments(samples_16k, None)?))
    }
}

impl Speech for SttEngine {
    fn languages(&self) -> Vec<String> {
        if self.english_only {
            vec!["en".to_string()]
        } else {
            vec!["auto".to_string()]
        }
    }

    fn engine_name(&self) -> &'static str {
        "whisper.cpp"
    }

    fn segments(&self, samples_16k: &[f32], language: Option<&str>) -> Result<Vec<Segment>> {
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
        let lang = if self.english_only { "en" } else { language.unwrap_or("auto") };
        params.set_language(Some(lang));

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
        let mut out = Vec::new();
        for i in 0..num_segments {
            let text = state.full_get_segment_text(i).context("failed to read segment text")?;
            let t0 = state.full_get_segment_t0(i).context("failed to read segment start")?;
            let t1 = state.full_get_segment_t1(i).context("failed to read segment end")?;
            out.push(Segment { start_s: t0 as f64 / 100.0, end_s: t1 as f64 / 100.0, text });
        }
        Ok(out)
    }
}
