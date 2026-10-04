use anyhow::Result;
use enigo::{Direction, Enigo, Key, Keyboard, Settings};
use log::warn;
use std::thread::sleep;
use std::time::Duration;

fn type_or_paste(enigo: &mut Enigo, text: &str) -> Result<bool> {
    match enigo.text(text) {
        Ok(()) => return Ok(true),
        Err(e) => warn!("direct typing failed ({e}); falling back to paste"),
    }

    let mut clipboard = arboard::Clipboard::new()?;
    let previous = clipboard.get_text().ok();
    clipboard.set_text(text.to_string())?;
    sleep(Duration::from_millis(40));

    let pasted = enigo
        .key(Key::Control, Direction::Press)
        .and_then(|_| enigo.key(Key::Unicode('v'), Direction::Click))
        .and_then(|_| enigo.key(Key::Control, Direction::Release));

    match pasted {
        Ok(()) => {
            sleep(Duration::from_millis(180));
            if let Some(previous) = previous {
                let _ = clipboard.set_text(previous);
            }
            Ok(true)
        }
        Err(e) => {
            warn!("paste fallback failed ({e}); text left on the clipboard");
            let _ = enigo.key(Key::Control, Direction::Release);
            Ok(false)
        }
    }
}

pub fn insert_text(text: &str) -> Result<bool> {
    let mut enigo = Enigo::new(&Settings::default())?;
    type_or_paste(&mut enigo, text)
}

pub fn replace_text(undo_chars: usize, replacement: &str) -> Result<bool> {
    let mut enigo = Enigo::new(&Settings::default())?;
    for _ in 0..undo_chars {
        enigo.key(Key::Backspace, Direction::Click)?;
    }
    type_or_paste(&mut enigo, replacement)
}
