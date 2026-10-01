use once_cell::sync::Lazy;
use regex::Regex;

/// How much the text layer is allowed to rewrite what was actually said.
///
/// PRD §23 requires an explicit raw mode so a transcript can always be
/// inspected without the cleanup rules in the way — every other mode layers
/// rules on top of that baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CleanupMode {
    /// Exactly what the model emitted, trimmed and nothing else: no casing,
    /// punctuation, filler removal or whitespace collapsing.
    Raw,
    /// Leading capital, single terminal punctuation mark, collapsed runs of
    /// whitespace. The default, and what Phase 1 shipped.
    #[default]
    Clean,
    /// `Clean`, plus removal of spoken disfluencies ("uh", "um", ...) — for
    /// prose headed somewhere it will be read, like an email or a document.
    Polished,
    /// Disfluency removal and whitespace collapsing, but no capitalization
    /// and no invented terminal punctuation — dictating a shell command or an
    /// identifier should not acquire a trailing period.
    Developer,
}

impl CleanupMode {
    /// Parses the lowercase tag used in the persisted desktop config and in
    /// the benchmark CLI, so the same spelling works in both.
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

    /// Every mode, in the order the settings UI should list them.
    pub fn all() -> [Self; 4] {
        [Self::Raw, Self::Clean, Self::Polished, Self::Developer]
    }
}

/// Spoken disfluencies, matched standalone (never inside a word) along with a
/// comma whisper.cpp tends to emit after them. Deliberately conservative:
/// only tokens that are not also ordinary English words, so `Polished` can't
/// quietly eat meaning. Words like "like" and "so" are left alone for exactly
/// that reason.
static FILLER_PATTERN: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(?:uh+|um+|uhm|erm|er|ah|hmm+|mm+)\b,?\s*").unwrap());

static WHITESPACE_RUN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+").unwrap());

/// Leftovers from removing a filler that sat between two commas, e.g.
/// "I was, uh, thinking" -> "I was, , thinking" -> "I was, thinking".
static DOUBLED_COMMA: Lazy<Regex> = Lazy::new(|| Regex::new(r",(?:\s*,)+").unwrap());
static SPACE_BEFORE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"\s+([,.!?;:])").unwrap());

/// Capitalizes the first letter and ensures a single terminal punctuation mark.
/// whisper.cpp's base.en model already emits punctuation/casing for most speech,
/// so this is a safety net rather than the primary source of punctuation.
///
/// Equivalent to [`clean_transcript_with`] in [`CleanupMode::Clean`].
pub fn clean_transcript(raw: &str) -> String {
    clean_transcript_with(raw, CleanupMode::Clean)
}

/// Applies `mode`'s cleanup rules to a raw transcript.
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
        // Removing a leading filler can leave the sentence starting on its
        // punctuation, e.g. "Um, hello" -> ", hello".
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

/// Detects a spoken correction like "Actually, change Rahul to Rohan" against the
/// previous finalized transcript. Returns the revised text if `utterance` matches
/// the correction pattern and `find` is present (case-insensitively) in `previous`;
/// otherwise `None`, meaning the caller should treat `utterance` as new dictation.
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
        // Only Polished/Developer strip disfluencies — Clean is a formatting
        // pass, not an editorial one.
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
        // "summary" contains "um", "ahead" starts with "ah" and "error" starts
        // with "er" — none is a standalone filler token, so all must survive.
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
