use anyhow::Result;
use enigo::{Enigo, Keyboard, Settings};

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
