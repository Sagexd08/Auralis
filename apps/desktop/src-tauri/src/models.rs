use anyhow::{bail, Context, Result};
use serde::Serialize;
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Manager};

/// A whisper.cpp model Auralis knows how to download. Only catalog entries can
/// be fetched, so a model name from the UI can never turn into an arbitrary
/// URL or a path outside the models directory.
pub struct ModelSpec {
    pub file: &'static str,
    pub label: &'static str,
    pub size_mb: u32,
}

pub const CATALOG: &[ModelSpec] = &[
    ModelSpec { file: "ggml-base.en-q5_1.bin", label: "base.en — fast (default)", size_mb: 59 },
    ModelSpec { file: "ggml-small.en-q5_1.bin", label: "small.en — more accurate, ~3x slower", size_mb: 190 },
    ModelSpec { file: "ggml-medium.en-q5_1.bin", label: "medium.en — most accurate, needs a fast CPU/GPU", size_mb: 539 },
];

/// whisper.cpp ggml files start with the little-endian magic of "ggml".
const GGML_MAGIC: &[u8; 4] = b"lmgg";
/// Anything smaller than this is an error page or a truncated download.
const MIN_MODEL_BYTES: u64 = 10 * 1024 * 1024;

#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub file: String,
    pub label: String,
    pub size_mb: Option<u32>,
    pub installed: bool,
}

#[derive(Serialize, Clone)]
pub struct DownloadProgress {
    pub file: String,
    pub downloaded: u64,
    pub total: Option<u64>,
}

/// Where downloaded models live: the per-user app data dir, so an installed
/// build works without write access to Program Files.
pub fn models_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_local_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("models")
}

/// In debug builds, also look in the repo's `models/` folder so
/// `models/pull-model.ps1` keeps working during development.
fn dev_models_dir() -> Option<PathBuf> {
    if cfg!(debug_assertions) {
        Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models"))
    } else {
        None
    }
}

fn is_plain_filename(file: &str) -> bool {
    !file.is_empty() && !file.contains(['/', '\\']) && !file.contains("..")
}

/// Resolves a model filename to an existing file, or `None` if not installed.
pub fn find(app: &AppHandle, file: &str) -> Option<PathBuf> {
    if !is_plain_filename(file) {
        return None;
    }
    std::iter::once(models_dir(app))
        .chain(dev_models_dir())
        .map(|dir| dir.join(file))
        .find(|p| p.is_file())
}

fn installed_files(app: &AppHandle) -> Vec<String> {
    let mut names: Vec<String> = std::iter::once(models_dir(app))
        .chain(dev_models_dir())
        .filter_map(|dir| fs::read_dir(dir).ok())
        .flat_map(|entries| entries.flatten())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|name| name.ends_with(".bin"))
        .collect();
    names.sort();
    names.dedup();
    names
}

/// Catalog models (installed or not) plus any other `.bin` the user dropped in.
pub fn list(app: &AppHandle) -> Vec<ModelInfo> {
    let installed = installed_files(app);
    let mut out: Vec<ModelInfo> = CATALOG
        .iter()
        .map(|spec| ModelInfo {
            file: spec.file.to_string(),
            label: spec.label.to_string(),
            size_mb: Some(spec.size_mb),
            installed: installed.iter().any(|f| f == spec.file),
        })
        .collect();
    for file in installed {
        if !CATALOG.iter().any(|s| s.file == file) {
            out.push(ModelInfo { label: file.clone(), file, size_mb: None, installed: true });
        }
    }
    out
}

/// Downloads a catalog model into [`models_dir`], reporting progress. Writes to
/// a `.part` file and only renames it into place after the size and ggml magic
/// check out, so an interrupted download never leaves a half-model that looks
/// installed.
pub fn download(app: &AppHandle, file: &str, mut on_progress: impl FnMut(DownloadProgress)) -> Result<PathBuf> {
    let spec = CATALOG
        .iter()
        .find(|s| s.file == file)
        .with_context(|| format!("{file:?} is not a downloadable model"))?;

    let dir = models_dir(app);
    fs::create_dir_all(&dir).context("failed to create models directory")?;
    let dest = dir.join(spec.file);
    let part = dir.join(format!("{}.part", spec.file));

    let url = format!("https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{}", spec.file);
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(15))
        .timeout_read(Duration::from_secs(30))
        .build();
    let response = agent.get(&url).call().with_context(|| format!("failed to fetch {url}"))?;
    let total: Option<u64> = response.header("Content-Length").and_then(|v| v.parse().ok());

    let result = (|| -> Result<()> {
        let mut reader = response.into_reader();
        let mut out = File::create(&part).context("failed to create temp file")?;
        let mut buf = vec![0u8; 64 * 1024];
        let mut downloaded = 0u64;
        let mut last_report = Instant::now() - Duration::from_secs(1);
        loop {
            let n = reader.read(&mut buf).context("download interrupted")?;
            if n == 0 {
                break;
            }
            out.write_all(&buf[..n])?;
            downloaded += n as u64;
            if last_report.elapsed() >= Duration::from_millis(250) {
                on_progress(DownloadProgress { file: spec.file.to_string(), downloaded, total });
                last_report = Instant::now();
            }
        }
        out.flush()?;
        drop(out);

        if let Some(total) = total {
            if downloaded != total {
                bail!("download truncated: got {downloaded} of {total} bytes");
            }
        }
        if downloaded < MIN_MODEL_BYTES {
            bail!("downloaded file is only {downloaded} bytes — not a model");
        }
        let mut magic = [0u8; 4];
        File::open(&part)?.read_exact(&mut magic)?;
        if &magic != GGML_MAGIC {
            bail!("downloaded file is not a ggml model");
        }
        on_progress(DownloadProgress { file: spec.file.to_string(), downloaded, total });
        Ok(())
    })();

    match result {
        Ok(()) => {
            fs::rename(&part, &dest).context("failed to move model into place")?;
            Ok(dest)
        }
        Err(e) => {
            let _ = fs::remove_file(&part);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_path_traversal_in_model_names() {
        assert!(!is_plain_filename("../secret.bin"));
        assert!(!is_plain_filename("a/b.bin"));
        assert!(!is_plain_filename("a\\b.bin"));
        assert!(!is_plain_filename(""));
        assert!(is_plain_filename("ggml-base.en-q5_1.bin"));
    }

    #[test]
    fn default_config_model_is_in_the_catalog() {
        let default = crate::config::AppConfig::default().model_file;
        assert!(CATALOG.iter().any(|s| s.file == default));
    }
}
