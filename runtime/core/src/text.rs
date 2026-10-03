use once_cell::sync::Lazy;
use regex::Regex;
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CleanupMode {
    Raw,
    #[default]
    Clean,
    Polished,
    Developer,
}

impl CleanupMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_lowercase().as_str() {
            "raw" => Some(Self::Raw),
            "clean" => Some(Self::Clean),
            "polished" => Some(Self::Polished),
            "developer" | "dev" => Some(Self::Developer),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Raw => "raw",
            Self::Clean => "clean",
            Self::Polished => "polished",
            Self::Developer => "developer",
        }
    }

    pub fn all() -> [Self; 4] {
        [Self::Raw, Self::Clean, Self::Polished, Self::Developer]
    }
}

#[derive(Debug, Clone)]
pub struct CleanupModeHandle(Arc<Mutex<CleanupMode>>);

impl CleanupModeHandle {
    pub fn new(mode: CleanupMode) -> Self {
        Self(Arc::new(Mutex::new(mode)))
    }

    pub fn get(&self) -> CleanupMode {
        *self.0.lock().expect("cleanup mode mutex poisoned")
    }

    pub fn set(&self, mode: CleanupMode) {
        *self.0.lock().expect("cleanup mode mutex poisoned") = mode;
    }
}

impl Default for CleanupModeHandle {
    fn default() -> Self {
        Self::new(CleanupMode::default())
    }
}

static FILLER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:uh+|um+|uhm|erm|er|ah|hmm+|mm+)\b,?\s*").unwrap());

static WHITESPACE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

static DOUBLED_COMMA: Lazy<Regex> = Lazy::new(|| Regex::new(r",(?:\s*,)+").unwrap());
static SPACE_BEFORE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([,.!?;:])").unwrap());

pub fn clean_transcript(raw: &str) -> String {
    clean_transcript_with(raw, CleanupMode::Clean)
}

pub fn clean_transcript_with(raw: &str, mode: CleanupMode) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if mode == CleanupMode::Raw {
        return trimmed.to_string();
    }

    let mut result = trimmed.to_string();

    if matches!(mode, CleanupMode::Polished | CleanupMode::Developer) {
        result = FILLER_PATTERN.replace_all(&result, "").to_string();
        result = DOUBLED_COMMA.replace_all(&result, ",").to_string();
        result = SPACE_BEFORE_PUNCT.replace_all(&result, "$1").to_string();
        result = result.trim_start_matches([',', ' ']).to_string();
    }

    result = WHITESPACE_RUN.replace_all(&result, " ").trim().to_string();
    if result.is_empty() {
        return String::new();
    }

    if matches!(mode, CleanupMode::Clean | CleanupMode::Polished) {
        let mut chars = result.chars();
        let first = chars.next().unwrap().to_uppercase().to_string();
        let rest: String = chars.collect();
        result = format!("{first}{rest}");

        if !result.ends_with(['.', '!', '?']) {
            result.push('.');
        }
    }

    result
}

static CORRECTION_PATTERN: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"(?i)^(?:actually,?\s+)?change\s+(.+?)\s+to\s+(.+?)[\.\!\?]?$").unwrap()
});

pub fn detect_correction(previous: &str, utterance: &str) -> Option<String> {
    let caps = CORRECTION_PATTERN.captures(utterance.trim())?;
    let find = caps.get(1)?.as_str();
    let replace = caps.get(2)?.as_str().trim_end_matches(['.', '!', '?']);

    let lower_prev = previous.to_lowercase();
    let lower_find = find.to_lowercase();
    let pos = lower_prev.find(&lower_find)?;

    let mut result = previous.to_string();
    result.replace_range(pos..pos + find.len(), replace);
    Some(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capitalizes_and_adds_period() {
        assert_eq!(clean_transcript("hello there"), "Hello there.");
    }

    #[test]
    fn leaves_existing_terminal_punctuation() {
        assert_eq!(clean_transcript("is this working?"), "Is this working?");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(clean_transcript("   "), "");
    }

    #[test]
    fn clean_collapses_whitespace_runs() {
        assert_eq!(clean_transcript("hello   there\n\nfriend"), "Hello there friend.");
    }

    #[test]
    fn raw_mode_changes_nothing_but_surrounding_whitespace() {
        let raw = "  uh, hello   there  ";
        assert_eq!(clean_transcript_with(raw, CleanupMode::Raw), "uh, hello   there");
    }

    #[test]
    fn clean_mode_keeps_fillers() {
        assert_eq!(clean_transcript_with("uh, hello there", CleanupMode::Clean), "Uh, hello there.");
    }

    #[test]
    fn polished_mode_strips_leading_filler_then_capitalizes() {
        assert_eq!(
            clean_transcript_with("um, we need to deploy this tomorrow", CleanupMode::Polished),
            "We need to deploy this tomorrow."
        );
    }

    #[test]
    fn polished_mode_strips_interior_filler_without_doubling_commas() {
        assert_eq!(
            clean_transcript_with("I was, uh, thinking about it", CleanupMode::Polished),
            "I was, thinking about it."
        );
    }

    #[test]
    fn polished_mode_strips_repeated_and_lengthened_fillers() {
        assert_eq!(
            clean_transcript_with("uhhh um so we ship it", CleanupMode::Polished),
            "So we ship it."
        );
    }

    #[test]
    fn polished_mode_leaves_ordinary_words_that_look_like_fillers_alone() {
        assert_eq!(
            clean_transcript_with("summary of the error ahead", CleanupMode::Polished),
            "Summary of the error ahead."
        );
    }

    #[test]
    fn developer_mode_adds_no_capital_and_no_terminal_period() {
        assert_eq!(
            clean_transcript_with("um, cargo test --locked", CleanupMode::Developer),
            "cargo test --locked"
        );
    }

    #[test]
    fn developer_mode_preserves_punctuation_the_speaker_dictated() {
        assert_eq!(
            clean_transcript_with("git commit -m \"fix it.\"", CleanupMode::Developer),
            "git commit -m \"fix it.\""
        );
    }

    #[test]
    fn all_filler_input_cleans_to_empty() {
        assert_eq!(clean_transcript_with("uh um hmm", CleanupMode::Polished), "");
    }

    #[test]
    fn mode_tags_round_trip() {
        for mode in CleanupMode::all() {
            assert_eq!(CleanupMode::parse(mode.as_str()), Some(mode));
        }
        assert_eq!(CleanupMode::parse("DEV"), Some(CleanupMode::Developer));
        assert_eq!(CleanupMode::parse("nonsense"), None);
    }

    #[test]
    fn handle_shares_mode_between_clones() {
        let a = CleanupModeHandle::new(CleanupMode::Clean);
        let b = a.clone();
        assert_eq!(b.get(), CleanupMode::Clean);
        b.set(CleanupMode::Developer);
        assert_eq!(a.get(), CleanupMode::Developer, "clones must see each other's writes");
    }

    #[test]
    fn handle_defaults_to_clean() {
        assert_eq!(CleanupModeHandle::default().get(), CleanupMode::Clean);
    }

    #[test]
    fn detects_simple_correction() {
        let previous = "Send the report to Rahul tomorrow.";
        let revised = detect_correction(previous, "Actually, change Rahul to Rohan.");
        assert_eq!(
            revised,
            Some("Send the report to Rohan tomorrow.".to_string())
        );
    }

    #[test]
    fn non_correction_utterance_returns_none() {
        let previous = "Send the report to Rahul tomorrow.";
        assert_eq!(detect_correction(previous, "Also cc the design team."), None);
    }

    #[test]
    fn correction_target_not_found_returns_none() {
        let previous = "Send the report to Rahul tomorrow.";
        assert_eq!(detect_correction(previous, "change Priya to Rohan"), None);
    }
}
