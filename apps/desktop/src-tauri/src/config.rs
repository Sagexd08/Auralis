use auralis_runtime::personalize::Personalization;
use auralis_runtime::text::CleanupMode;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AppConfig {
    pub push_to_talk_hotkey: String,
    pub toggle_hotkey: String,
    pub model_file: String,
    pub mic_device: Option<String>,
    pub cleanup_mode: String,
    pub dictionary: String,
    pub snippets: String,
    #[serde(default = "spoken_commands_on")]
    pub spoken_commands: bool,
    #[serde(default = "existing_install_is_onboarded")]
    pub onboarded: bool,
}

fn existing_install_is_onboarded() -> bool {
    true
}

fn spoken_commands_on() -> bool {
    true
}

impl Default for AppConfig {
    fn default() -> Self {
        Self {
            push_to_talk_hotkey: "Ctrl+Space".to_string(),
            toggle_hotkey: "Ctrl+Shift+Space".to_string(),
            model_file: "ggml-base.en-q5_1.bin".to_string(),
            mic_device: None,
            cleanup_mode: CleanupMode::default().as_str().to_string(),
            dictionary: String::new(),
            snippets: String::new(),
            spoken_commands: true,
            onboarded: false,
        }
    }
}

impl AppConfig {
    fn file_path(config_dir: &Path) -> PathBuf {
        config_dir.join("config.json")
    }

    pub fn load(config_dir: &Path) -> Self {
        let path = Self::file_path(config_dir);
        match fs::read_to_string(&path) {
            Ok(contents) => serde_json::from_str(&contents).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn personalization(&self) -> Personalization {
        Personalization::from_settings(&self.dictionary, &self.snippets, self.spoken_commands)
    }

    pub fn cleanup_mode(&self) -> CleanupMode {
        CleanupMode::parse(&self.cleanup_mode).unwrap_or_default()
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
    fn fresh_install_is_not_onboarded_but_an_existing_config_file_is() {
        assert!(!AppConfig::default().onboarded);
        let old: AppConfig = serde_json::from_str(r#"{"push_to_talk_hotkey":"Ctrl+Space"}"#).unwrap();
        assert!(old.onboarded, "configs saved before the welcome screen existed must not show it");
    }

    #[test]
    fn old_configs_get_spoken_commands_on_and_empty_lists() {
        let old: AppConfig = serde_json::from_str(r#"{"cleanup_mode":"clean"}"#).unwrap();
        assert!(old.spoken_commands);
        assert!(old.personalization().dictionary.is_empty());
    }

    #[test]
    fn personalization_is_built_from_settings_text() {
        let mut config = AppConfig::default();
        config.dictionary = "post gres = PostgreSQL".to_string();
        config.snippets = "my link = https://example.com".to_string();
        config.spoken_commands = false;
        let p = config.personalization();
        assert_eq!(p.dictionary.len(), 1);
        assert_eq!(p.snippets.len(), 1);
        assert!(!p.spoken_commands);
    }

    #[test]
    fn default_cleanup_mode_is_clean() {
        assert_eq!(AppConfig::default().cleanup_mode(), CleanupMode::Clean);
    }

    #[test]
    fn unknown_cleanup_mode_falls_back_to_default() {
        let mut config = AppConfig::default();
        config.cleanup_mode = "wat".to_string();
        assert_eq!(config.cleanup_mode(), CleanupMode::Clean);
    }

    #[test]
    fn round_trips_through_save_and_load() {
        let dir = std::env::temp_dir().join(format!("auralis-config-roundtrip-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);

        let mut config = AppConfig::default();
        config.model_file = "ggml-tiny.en-q5_1.bin".to_string();
        config.mic_device = Some("Test Mic".to_string());
        config.cleanup_mode = "polished".to_string();
        config.save(&dir).expect("save succeeds");

        let loaded = AppConfig::load(&dir);
        assert_eq!(loaded.model_file, "ggml-tiny.en-q5_1.bin");
        assert_eq!(loaded.mic_device, Some("Test Mic".to_string()));
        assert_eq!(loaded.cleanup_mode(), CleanupMode::Polished);

        let _ = fs::remove_dir_all(&dir);
    }
}
