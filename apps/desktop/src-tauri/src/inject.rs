use anyhow::Result;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};

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
