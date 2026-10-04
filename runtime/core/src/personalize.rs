use crate::text::CleanupMode;
use once_cell::sync::Lazy;
use regex::{NoExpand, Regex};
use std::sync::{Arc, Mutex};

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Personalization {
    pub dictionary: Vec<(String, String)>,
    pub snippets: Vec<(String, String)>,
    pub spoken_commands: bool,
}

#[derive(Debug, Clone, Default)]
pub struct PersonalizationHandle(Arc<Mutex<Personalization>>);

impl PersonalizationHandle {
    pub fn new(value: Personalization) -> Self {
        Self(Arc::new(Mutex::new(value)))
    }

    pub fn get(&self) -> Personalization {
        self.0.lock().expect("personalization mutex poisoned").clone()
    }

    pub fn set(&self, value: Personalization) {
        *self.0.lock().expect("personalization mutex poisoned") = value;
    }
}

pub fn parse_pairs(text: &str) -> Vec<(String, String)> {
    text.lines()
        .filter_map(|line| {
            let (from, to) = line.split_once("=>").or_else(|| line.split_once('='))?;
            let (from, to) = (from.trim(), to.trim().replace("\\n", "\n"));
            (!from.is_empty() && !to.is_empty()).then(|| (from.to_string(), to))
        })
        .collect()
}

fn normalize_trigger(text: &str) -> String {
    text.trim()
        .trim_end_matches(['.', '!', '?', ','])
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

impl Personalization {
    pub fn from_settings(dictionary: &str, snippets: &str, spoken_commands: bool) -> Self {
        Self { dictionary: parse_pairs(dictionary), snippets: parse_pairs(snippets), spoken_commands }
    }

    pub fn snippet_for(&self, utterance: &str) -> Option<String> {
        let heard = normalize_trigger(utterance);
        if heard.is_empty() {
            return None;
        }
        self.snippets.iter().find(|(trigger, _)| normalize_trigger(trigger) == heard).map(|(_, body)| body.clone())
    }

    pub fn apply_dictionary(&self, text: &str) -> String {
        let mut out = text.to_string();
        for (from, to) in &self.dictionary {
            let escaped = regex::escape(from);
            let starts_word = from.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_');
            let ends_word = from.chars().last().is_some_and(|c| c.is_alphanumeric() || c == '_');
            let pattern = format!(
                "(?i){}{}{}",
                if starts_word { r"\b" } else { "" },
                escaped,
                if ends_word { r"\b" } else { "" }
            );
            if let Ok(re) = Regex::new(&pattern) {
                out = re.replace_all(&out, NoExpand(to)).to_string();
            }
        }
        out
    }

    pub fn finish(&self, cleaned: &str, mode: CleanupMode) -> String {
        if mode == CleanupMode::Raw {
            return cleaned.to_string();
        }
        let mut out = cleaned.to_string();
        if self.spoken_commands {
            out = apply_spoken_commands(&out, mode);
        }
        self.apply_dictionary(&out)
    }
}

static PARAGRAPH: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)[ \t]*\bnew paragraph\b[.,!?]?[ \t]*").unwrap());
static NEWLINE: Lazy<Regex> = Lazy::new(|| Regex::new(r"(?i)[ \t]*\b(?:new line|newline)\b[.,!?]?[ \t]*").unwrap());
static CASE_COMMAND: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?i)\b(camel|snake|pascal|kebab) case ((?:[a-z0-9]+ ?)+)").unwrap());
static AFTER_BREAK: Lazy<Regex> = Lazy::new(|| Regex::new(r"([.!?]\s+|\n)(\p{Ll})").unwrap());
static SPACES_BEFORE_PUNCT: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ \t]+([,.!?;:)])").unwrap());
static SPACE_AFTER_OPEN: Lazy<Regex> = Lazy::new(|| Regex::new(r"\(\s+").unwrap());

const PUNCTUATION: [(&str, &str, &str, &str); 9] = [
    (r"question mark", "?", r"[ \t]*", "[.,!?]?"),
    (r"exclamation (?:mark|point)", "!", r"[ \t]*", "[.,!?]?"),
    (r"full stop", ".", r"[ \t]*", "[.,!?]?"),
    (r"period", ".", r"[ \t]*", "[.,!?]?"),
    (r"comma", ",", r"[ \t]*", "[.,!?]?"),
    (r"semicolon", ";", r"[ \t]*", "[.,!?]?"),
    (r"colon", ":", r"[ \t]*", "[.,!?]?"),
    (r"open (?:parenthesis|bracket)", "(", "", ""),
    (r"close (?:parenthesis|bracket)", ")", r"[ \t]*", ""),
];

static PUNCTUATION_RES: Lazy<Vec<(Regex, &'static str)>> = Lazy::new(|| {
    PUNCTUATION
        .iter()
        .map(|(words, mark, lead, trail)| (Regex::new(&format!(r"(?i){lead}\b{words}\b{trail}")).unwrap(), *mark))
        .collect()
});

fn join_case(style: &str, words: &str) -> String {
    let parts: Vec<String> = words.split_whitespace().map(str::to_lowercase).collect();
    let cap = |w: &String| {
        let mut c = w.chars();
        c.next().map(|f| f.to_uppercase().collect::<String>() + c.as_str()).unwrap_or_default()
    };
    match style {
        "snake" => parts.join("_"),
        "kebab" => parts.join("-"),
        "pascal" => parts.iter().map(cap).collect(),
        _ => parts
            .iter()
            .enumerate()
            .map(|(i, w)| if i == 0 { w.clone() } else { cap(w) })
            .collect(),
    }
}

pub fn apply_spoken_commands(text: &str, mode: CleanupMode) -> String {
    let mut out = text.to_string();
    if mode == CleanupMode::Developer {
        out = CASE_COMMAND
            .replace_all(&out, |caps: &regex::Captures| {
                let tail = if caps[2].ends_with(' ') { " " } else { "" };
                format!("{}{}", join_case(&caps[1].to_lowercase(), &caps[2]), tail)
            })
            .to_string();
    }
    out = PARAGRAPH.replace_all(&out, NoExpand("\n\n")).to_string();
    out = NEWLINE.replace_all(&out, NoExpand("\n")).to_string();
    for (re, mark) in PUNCTUATION_RES.iter() {
        out = re.replace_all(&out, NoExpand(*mark)).to_string();
    }
    out = SPACES_BEFORE_PUNCT.replace_all(&out, "$1").to_string();
    out = SPACE_AFTER_OPEN.replace_all(&out, "(").to_string();
    for dup in ["..", ",,", "?.", "!."] {
        while out.contains(dup) {
            out = out.replace(dup, &dup[..1]);
        }
    }
    if matches!(mode, CleanupMode::Clean | CleanupMode::Polished) {
        out = AFTER_BREAK
            .replace_all(&out, |caps: &regex::Captures| format!("{}{}", &caps[1], caps[2].to_uppercase()))
            .to_string();
    }
    out.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(dictionary: &str, snippets: &str, commands: bool) -> Personalization {
        Personalization::from_settings(dictionary, snippets, commands)
    }

    #[test]
    fn parses_both_arrow_styles_and_skips_junk() {
        let pairs = parse_pairs("post gres = PostgreSQL\nkubernetes => Kubernetes\n\nbroken line\n= nothing\nlink = a\\nb");
        assert_eq!(pairs[0], ("post gres".into(), "PostgreSQL".into()));
        assert_eq!(pairs[1], ("kubernetes".into(), "Kubernetes".into()));
        assert_eq!(pairs[2], ("link".into(), "a\nb".into()));
        assert_eq!(pairs.len(), 3);
    }

    #[test]
    fn dictionary_replaces_whole_words_case_insensitively() {
        let pz = p("post gres = PostgreSQL\ncuber netties = Kubernetes", "", false);
        assert_eq!(pz.apply_dictionary("Deploy the Post Gres migration."), "Deploy the PostgreSQL migration.");
        assert_eq!(pz.apply_dictionary("compost gresham"), "compost gresham");
    }

    #[test]
    fn dictionary_replacement_text_is_literal() {
        let pz = p("price = $1 total", "", false);
        assert_eq!(pz.apply_dictionary("the price"), "the $1 total");
    }

    #[test]
    fn snippet_matches_the_whole_utterance_only() {
        let pz = p("", "my scheduling link = https://cal.example/me", false);
        assert_eq!(pz.snippet_for("My scheduling link."), Some("https://cal.example/me".into()));
        assert_eq!(pz.snippet_for("send my scheduling link to Rahul"), None);
        assert_eq!(pz.snippet_for(""), None);
    }

    #[test]
    fn spoken_punctuation_and_paragraphs() {
        let out = apply_spoken_commands("Hello comma world new paragraph second part question mark", CleanupMode::Clean);
        assert_eq!(out, "Hello, world\n\nSecond part?");
    }

    #[test]
    fn parentheses_and_trailing_period_are_not_doubled() {
        let out = apply_spoken_commands("See open parenthesis the notes close parenthesis period.", CleanupMode::Clean);
        assert_eq!(out, "See (the notes).");
    }

    #[test]
    fn developer_case_commands() {
        assert_eq!(apply_spoken_commands("call camel case get user by id", CleanupMode::Developer), "call getUserById");
        assert_eq!(apply_spoken_commands("snake case max retry count", CleanupMode::Developer), "max_retry_count");
        assert_eq!(apply_spoken_commands("pascal case user service", CleanupMode::Developer), "UserService");
        assert_eq!(apply_spoken_commands("kebab case build output", CleanupMode::Developer), "build-output");
    }

    #[test]
    fn case_commands_are_ignored_outside_developer_mode() {
        assert_eq!(apply_spoken_commands("camel case is a style", CleanupMode::Clean), "camel case is a style");
    }

    #[test]
    fn raw_mode_is_never_rewritten() {
        let pz = p("a = b", "", true);
        assert_eq!(pz.finish("a comma a", CleanupMode::Raw), "a comma a");
    }

    #[test]
    fn finish_applies_commands_then_dictionary() {
        let pz = p("post gres = PostgreSQL", "", true);
        assert_eq!(pz.finish("Use post gres comma please.", CleanupMode::Clean), "Use PostgreSQL, please.");
    }

    #[test]
    fn commands_can_be_switched_off() {
        let pz = p("", "", false);
        assert_eq!(pz.finish("Hello comma world.", CleanupMode::Clean), "Hello comma world.");
    }
}
