use crate::frontend::Frontend;
use crate::stt::{Segment, Speech};
use anyhow::{bail, Context, Result};
use serde_json::Value;
use std::path::{Path, PathBuf};
use tract_onnx::prelude::*;

type Plan = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

const SPECIAL_PREFIXES: [&str; 2] = ["<blank>", "<unk>"];
const WORD_MARK: char = '\u{2581}';

pub struct AuralisEngine {
    plan: Plan,
    frontend: Frontend,
    vocab: Vec<String>,
    blank: usize,
    languages: Vec<String>,
    n_mels: usize,
    chunk_seconds: f64,
}

fn sibling(model_path: &Path, suffix: &str) -> PathBuf {
    let stem = model_path.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
    model_path.with_file_name(format!("{stem}{suffix}"))
}

fn load_vocab(model_path: &Path) -> Result<Vec<String>> {
    let named = sibling(model_path, ".tokenizer.json");
    let generic = model_path.with_file_name("tokenizer.json");
    let path = if named.exists() { named } else { generic };
    let raw = std::fs::read_to_string(&path).with_context(|| format!("missing tokenizer next to the model: {}", path.display()))?;
    let json: Value = serde_json::from_str(&raw).context("tokenizer.json is not valid JSON")?;
    json["vocab"]
        .as_array()
        .context("tokenizer.json has no vocab")?
        .iter()
        .map(|v| v.as_str().map(str::to_string).context("vocab entries must be strings"))
        .collect()
}

fn is_special(token: &str) -> bool {
    SPECIAL_PREFIXES.contains(&token) || (token.starts_with("<LANG_") && token.ends_with('>'))
}

impl AuralisEngine {
    pub fn load(model_path: &Path) -> Result<Self> {
        let vocab = load_vocab(model_path)?;
        let blank = vocab.iter().position(|t| t == "<blank>").context("tokenizer has no <blank> token")?;

        let meta: Value = std::fs::read_to_string(sibling(model_path, ".json"))
            .ok()
            .and_then(|raw| serde_json::from_str(&raw).ok())
            .unwrap_or(Value::Null);
        let n_mels = meta["n_mels"].as_u64().unwrap_or(80) as usize;
        let languages = meta["languages"]
            .as_array()
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect::<Vec<_>>())
            .filter(|l| !l.is_empty())
            .unwrap_or_else(|| vec!["auto".to_string()]);

        let mut model = tract_onnx::onnx()
            .model_for_path(model_path)
            .with_context(|| format!("failed to read ONNX model {}", model_path.display()))?;
        let frames = model.symbols.sym("frames");
        model.set_input_fact(0, f32::fact([1.to_dim(), n_mels.to_dim(), frames.to_dim()]).into())?;
        model.set_input_fact(1, i64::fact([1]).into())?;
        let plan = model
            .into_optimized()
            .context("failed to optimise the model")?
            .into_runnable()
            .context("failed to build the model plan")?;

        Ok(Self { plan, frontend: Frontend::new(16_000, 512, 400, 160, n_mels), vocab, blank, languages, n_mels, chunk_seconds: 60.0 })
    }

    pub fn set_chunk_seconds(&mut self, seconds: f64) {
        self.chunk_seconds = seconds.max(1.0);
    }

    pub fn decode_ids(&self, ids: &[usize]) -> String {
        let joined: String = ids
            .iter()
            .filter_map(|&i| self.vocab.get(i))
            .filter(|t| !is_special(t))
            .map(String::as_str)
            .collect();
        joined.replace(WORD_MARK, " ").trim().to_string()
    }

    pub fn log_probs(&self, samples_16k: &[f32]) -> Result<(Vec<f32>, usize, usize)> {
        let wave: Vec<f64> = samples_16k.iter().map(|&x| x as f64).collect();
        if wave.len() < 2 {
            bail!("audio too short");
        }
        let logmel = self.frontend.log_mel(&wave);
        let frames = logmel[0].len();
        let count = (self.n_mels * frames) as f64;
        let mean = logmel.iter().flatten().sum::<f64>() / count;
        let var = logmel.iter().flatten().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (count - 1.0).max(1.0);
        let std = var.sqrt();

        let features = tract_ndarray::Array3::from_shape_fn((1, self.n_mels, frames), |(_, m, t)| ((logmel[m][t] - mean) / (std + 1e-5)) as f32);
        let lengths = tract_ndarray::arr1(&[frames as i64]);
        let outputs = self
            .plan
            .run(tvec!(features.into_tensor().into(), lengths.into_tensor().into()))
            .context("model inference failed")?;
        let view = outputs[0].to_array_view::<f32>()?;
        let shape = view.shape().to_vec();
        Ok((view.iter().copied().collect(), shape[1], shape[2]))
    }

    pub fn greedy_ids(&self, flat: &[f32], steps: usize, vocab: usize) -> Vec<usize> {
        let mut ids = Vec::new();
        let mut previous = usize::MAX;
        for t in 0..steps {
            let row = &flat[t * vocab..(t + 1) * vocab];
            let best = row.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(self.blank);
            if best != previous && best != self.blank {
                ids.push(best);
            }
            previous = best;
        }
        ids
    }
}

impl Speech for AuralisEngine {
    fn segments(&self, samples_16k: &[f32], _language: Option<&str>) -> Result<Vec<Segment>> {
        let chunk = (self.chunk_seconds * 16_000.0) as usize;
        let mut segments = Vec::new();
        for (index, window) in samples_16k.chunks(chunk).enumerate() {
            if window.len() < 400 {
                continue;
            }
            let (flat, steps, vocab) = self.log_probs(window)?;
            let text = self.decode_ids(&self.greedy_ids(&flat, steps, vocab));
            if text.is_empty() {
                continue;
            }
            let start = (index * chunk) as f64 / 16_000.0;
            segments.push(Segment { start_s: start, end_s: start + window.len() as f64 / 16_000.0, text: format!(" {text}") });
        }
        Ok(segments)
    }

    fn languages(&self) -> Vec<String> {
        self.languages.clone()
    }

    fn engine_name(&self) -> &'static str {
        "auralis-onnx"
    }
}
