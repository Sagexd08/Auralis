use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

/// Persisted user settings: hotkey bindings, which model to load, and which
/// mic device to capture from. Stored as JSON in the Tauri app config dir.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    /// e.g. "Ctrl+Space" — parsed by `hotkey::parse`.
    pub push_to_talk_hotkey: String,
    /// e.g. "Ctrl+Shift+Space".
    pub toggle_hotkey: String,
    /// Filename only (relative to `models/`), e.g. "ggml-base.en-q5_1.bin".
    pub model_file: String,
    /// `None` means use the system default input device.
    pub mic_device: Option<String>,
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            push_to_talk_hotkey: "Ctrl+Space".to_string(),
            toggle_hotkey: "Ctrl+Shift+Space".to_string(),
            model_file: "ggml-base.en-q5_1.bin".to_string(),
            mic_device: None,
        }
    }
}

impl AppConfig {
    fn file_path(config_dir: &Path) -> PathBuf {
        config_dir.join("config.json")
    }

    /// Loads the config file, falling back to defaults if it's missing or
    /// unparseable (e.g. from an older/incompatible version).
    pub fn load(config_dir: &Path) -> Self {
        let path = Self::file_path(config_dir);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, config_dir: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(config_dir)?;
        let path = Self::file_path(config_dir);
        fs::write(path, serde_json::to_string_pretty(self)?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_file_falls_back_to_default() {
        let dir = std::env::temp_dir().join(format!("auralis-config-test-{}", std::process::id()));
        let config = AppConfig::load(&dir);
        assert_eq!(config.push_to_talk_hotkey, "Ctrl+Space");
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = std::env::temp_dir().join(format!("auralis-config-roundtrip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut config = AppConfig::default();
        config.model_file = "ggml-tiny.en-q5_1.bin".to_string();
        config.mic_device = Some("Test Mic".to_string());
        config.save(&dir).expect("save succeeds");

        let loaded = AppConfig::load(&dir);
        assert_eq!(loaded.model_file, "ggml-tiny.en-q5_1.bin");
        assert_eq!(loaded.mic_device, Some("Test Mic".to_string()));

        let _ = fs::remove_dir_all(&dir);
    }
}
