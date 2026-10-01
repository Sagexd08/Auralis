use once_cell::sync::Lazy;
use regex::Regex;

/// Capitalizes the first letter and ensures a single terminal punctuation mark.
/// whisper.cpp's base.en model already emits punctuation/casing for most speech,
/// so this is a safety net rather than the primary source of punctuation.
pub fn clean_transcript(raw: &str) -> String {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return String::new();
    }

    let mut chars = trimmed.chars();
    let first = chars.next().unwrap().to_uppercase().to_string();
    let rest: String = chars.collect();
    let mut result = format!("{first}{rest}");

    if !result.ends_with(['.', '!', '?']) {
        result.push('.');
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
