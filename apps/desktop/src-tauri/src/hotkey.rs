use anyhow::{bail, Context, Result};
use tauri_plugin_global_shortcut::{Code, Modifiers, Shortcut};

/// Parses a human-typed hotkey string like "Ctrl+Shift+Space" into a
/// `Shortcut`. Case-insensitive, `+`-separated, modifiers in any order.
/// Used both at startup (loading saved config) and when the settings UI
/// saves a rebind, so the same validation applies in both places.
pub fn parse(s: &str) -> Result<Shortcut> {
    let mut modifiers = Modifiers::empty();
    let mut code = None;

    for part in s.split('+') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        match part.to_lowercase().as_str() {
            "ctrl" | "control" => modifiers |= Modifiers::CONTROL,
            "shift" => modifiers |= Modifiers::SHIFT,
            "alt" => modifiers |= Modifiers::ALT,
            "super" | "meta" | "win" | "cmd" => modifiers |= Modifiers::SUPER,
            other => {
                if code.is_some() {
                    bail!("hotkey {s:?} specifies more than one non-modifier key");
                }
                code = Some(parse_code(other).with_context(|| format!("in hotkey {s:?}"))?);
            }
        }
    }

    let code = code.with_context(|| format!("hotkey {s:?} has no key (only modifiers)"))?;
    Ok(Shortcut::new(Some(modifiers), code))
}

fn parse_code(key: &str) -> Result<Code> {
    if let Some(c) = match key {
        "space" => Some(Code::Space),
        "enter" | "return" => Some(Code::Enter),
        "escape" | "esc" => Some(Code::Escape),
        "tab" => Some(Code::Tab),
        "backspace" => Some(Code::Backspace),
        "up" => Some(Code::ArrowUp),
        "down" => Some(Code::ArrowDown),
        "left" => Some(Code::ArrowLeft),
        "right" => Some(Code::ArrowRight),
        _ => None,
    } {
        return Ok(c);
    }

    if key.len() == 1 {
        let ch = key.chars().next().unwrap();
        if ch.is_ascii_alphabetic() {
            return letter_code(ch.to_ascii_uppercase());
        }
        if ch.is_ascii_digit() {
            return digit_code(ch);
        }
    }

    if let Some(n) = key.strip_prefix('f') {
        if let Ok(n) = n.parse::<u8>() {
            if let Some(c) = f_key_code(n) {
                return Ok(c);
            }
        }
    }

    bail!("unrecognized key {key:?} (supported: letters, digits, space, enter, escape, tab, backspace, arrows, f1-f12)")
}

fn letter_code(ch: char) -> Result<Code> {
    Ok(match ch {
        'A' => Code::KeyA, 'B' => Code::KeyB, 'C' => Code::KeyC, 'D' => Code::KeyD,
        'E' => Code::KeyE, 'F' => Code::KeyF, 'G' => Code::KeyG, 'H' => Code::KeyH,
        'I' => Code::KeyI, 'J' => Code::KeyJ, 'K' => Code::KeyK, 'L' => Code::KeyL,
        'M' => Code::KeyM, 'N' => Code::KeyN, 'O' => Code::KeyO, 'P' => Code::KeyP,
        'Q' => Code::KeyQ, 'R' => Code::KeyR, 'S' => Code::KeyS, 'T' => Code::KeyT,
        'U' => Code::KeyU, 'V' => Code::KeyV, 'W' => Code::KeyW, 'X' => Code::KeyX,
        'Y' => Code::KeyY, 'Z' => Code::KeyZ,
        _ => bail!("not a letter: {ch}"),
    })
}

fn digit_code(ch: char) -> Result<Code> {
    Ok(match ch {
        '0' => Code::Digit0, '1' => Code::Digit1, '2' => Code::Digit2, '3' => Code::Digit3,
        '4' => Code::Digit4, '5' => Code::Digit5, '6' => Code::Digit6, '7' => Code::Digit7,
        '8' => Code::Digit8, '9' => Code::Digit9,
        _ => bail!("not a digit: {ch}"),
    })
}

fn f_key_code(n: u8) -> Option<Code> {
    Some(match n {
        1 => Code::F1, 2 => Code::F2, 3 => Code::F3, 4 => Code::F4,
        5 => Code::F5, 6 => Code::F6, 7 => Code::F7, 8 => Code::F8,
        9 => Code::F9, 10 => Code::F10, 11 => Code::F11, 12 => Code::F12,
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_push_to_talk_default() {
        let shortcut = parse("Ctrl+Space").unwrap();
        assert_eq!(shortcut, Shortcut::new(Some(Modifiers::CONTROL), Code::Space));
    }

    #[test]
    fn parses_toggle_default_order_independent_and_case_insensitive() {
        let a = parse("Ctrl+Shift+Space").unwrap();
        let b = parse("shift+ctrl+SPACE").unwrap();
        assert_eq!(a, b);
        assert_eq!(a, Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space));
    }

    #[test]
    fn parses_letter_and_function_keys() {
        assert_eq!(parse("Alt+D").unwrap(), Shortcut::new(Some(Modifiers::ALT), Code::KeyD));
        assert_eq!(parse("F9").unwrap(), Shortcut::new(None, Code::F9));
    }

    #[test]
    fn rejects_no_key() {
        assert!(parse("Ctrl+Shift").is_err());
    }

    #[test]
    fn rejects_unrecognized_key() {
        assert!(parse("Ctrl+Banana").is_err());
    }

    #[test]
    fn rejects_two_non_modifier_keys() {
        assert!(parse("Space+Enter").is_err());
    }
}
