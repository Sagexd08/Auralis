use serde::Serialize;
use std::fs;
use std::path::PathBuf;
use tauri::{AppHandle, Manager};

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


#[derive(Serialize, Clone)]
pub struct ModelInfo {
    pub file: String,
    pub label: String,
    pub size_mb: Option<u32>,
    pub installed: bool,
}

pub fn models_dir(app: &AppHandle) -> PathBuf {
    app.path()
        .app_local_data_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join("models")
}

/// Models shipped inside the installer (read-only, next to the executable).
fn bundled_models_dir(app: &AppHandle) -> Option<PathBuf> {
    app.path().resource_dir().ok().map(|d| d.join("models"))
}

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

pub fn find(app: &AppHandle, file: &str) -> Option<PathBuf> {
    if !is_plain_filename(file) {
        return None;
    }
    std::iter::once(models_dir(app))
        .chain(bundled_models_dir(app))
        .chain(dev_models_dir())
        .map(|dir| dir.join(file))
        .find(|p| p.is_file())
}

fn installed_files(app: &AppHandle) -> Vec<String> {
    let mut names: Vec<String> = std::iter::once(models_dir(app))
        .chain(bundled_models_dir(app))
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

pub fn missing_message(app: &AppHandle, file: &str) -> String {
    format!("Model not found: place {file} in {}", models_dir(app).display())
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
