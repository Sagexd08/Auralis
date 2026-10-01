use anyhow::Result;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

/// Types `text` into whatever window currently has OS focus. Falls back to copying
/// to the clipboard (and returns a flag saying so) if keystroke injection fails,
/// e.g. because the focused window blocks synthetic input.
pub fn insert_text(text: &str) -> Result<bool> {
    let mut enigo = Enigo::new(&Settings::default())?;
    match enigo.text(text) {
        Ok(()) => Ok(true),
        Err(_) => {
            let mut clipboard = arboard::Clipboard::new()?;
            clipboard.set_text(text.to_string())?;
            Ok(false)
        }
    }
}

/// Undoes a previous in-place insertion for a spoken correction: presses
/// Backspace `undo_chars` times (deleting exactly what this app typed for the
/// previous utterance, assuming the cursor hasn't moved since) then types
/// `replacement` in its place. Falls back to clipboard, same as `insert_text`,
/// if keystroke injection fails partway through.
pub fn replace_text(undo_chars: usize, replacement: &str) -> Result<bool> {
    let mut enigo = Enigo::new(&Settings::default())?;
    for _ in 0..undo_chars {
        enigo.key(Key::Backspace, Direction::Click)?;
    }
    match enigo.text(replacement) {
        Ok(()) => Ok(true),
        Err(_) => {
            let mut clipboard = arboard::Clipboard::new()?;
            clipboard.set_text(replacement.to_string())?;
            Ok(false)
        }
    }
}
