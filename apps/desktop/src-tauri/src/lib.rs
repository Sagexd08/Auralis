mod config;
mod hotkey;
mod inject;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::{Pipeline, Transcript};
use auralis_runtime::text::{CleanupMode, CleanupModeHandle};
use config::AppConfig;
use log::{error, info, warn};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

/// Shared mutable state, reachable from hotkey handlers and Tauri commands
/// alike via `app.state::<AppState>()`.
struct AppState {
    config: Mutex<AppConfig>,
    config_dir: PathBuf,
    pipeline: Arc<Mutex<Option<Pipeline>>>,
    held: Arc<AtomicBool>,
    continuous_active: Arc<AtomicBool>,
    /// What this app most recently typed via keystroke injection, or `None`
    /// if nothing was typed (clipboard fallback, injection error, or nothing
    /// inserted yet). Lets a spoken correction backspace exactly what it
    /// typed and retype in place, instead of appending the corrected
    /// sentence after it.
    last_insertion: Arc<Mutex<Option<LastInsertion>>>,
    /// Read by the pipeline on every utterance. Held here as well as in the
    /// pipeline so a settings save can change it without taking the pipeline
    /// lock, which continuous dictation holds for a whole session.
    cleanup_mode: CleanupModeHandle,
}

/// Bookkeeping for one completed keystroke insertion.
#[derive(Debug, Clone, PartialEq, Eq)]
struct LastInsertion {
    /// The separator typed immediately before the utterance text: `" "`
    /// between consecutive utterances, `""` for the first one. A correction
    /// has to retype this, because the backspaces that undo the insertion
    /// delete the separator too.
    separator: String,
    /// Characters in separator + text, i.e. exactly how many backspaces undo
    /// this insertion.
    total_chars: usize,
}

/// What to type next, and what to remember once it lands. Pure bookkeeping,
/// kept separate from the `AppHandle`-bound apply step so the character
/// arithmetic a correction depends on is unit-testable without a running
/// Tauri app.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Insertion {
    /// Exact text to type, separator included.
    text: String,
    /// Backspaces to send first; 0 for a fresh insertion.
    undo_chars: usize,
    /// What to record as the new `last_insertion` on success.
    record: LastInsertion,
}

/// Plans a fresh insertion. Consecutive utterances (as continuous mode
/// produces) get a leading space so words don't run together; the very first
/// insertion, and the one after a clipboard fallback left nothing typed,
/// don't.
fn plan_fresh(previous: Option<&LastInsertion>, text: &str) -> Insertion {
    let separator = if previous.is_some() { " " } else { "" };
    let full = format!("{separator}{text}");
    Insertion {
        undo_chars: 0,
        record: LastInsertion {
            separator: separator.to_string(),
            total_chars: full.chars().count(),
        },
        text: full,
    }
}

/// Plans a spoken correction: undo the whole previous insertion, then retype
/// it with `replacement` as the body. The separator is carried over
/// deliberately — undoing `previous.total_chars` deletes it, so retyping
/// without it would run the correction into the utterance before it.
fn plan_correction(previous: &LastInsertion, replacement: &str) -> Insertion {
    let full = format!("{}{}", previous.separator, replacement);
    Insertion {
        undo_chars: previous.total_chars,
        record: LastInsertion {
            separator: previous.separator.clone(),
            total_chars: full.chars().count(),
        },
        text: full,
    }
}

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}

/// Emits a transcript event and applies it (inserting fresh text, or
/// replacing the previously-inserted text in place for a spoken correction),
/// updating the status event either way. Shared by push-to-talk and
/// continuous mode so both report status identically.
fn handle_transcript_result(handle: &AppHandle, last_insertion: &Mutex<Option<LastInsertion>>, result: anyhow::Result<Transcript>) {
    match result {
        Ok(Transcript::Fresh(text)) => {
            let _ = handle.emit("auralis://transcript", &text);
            let plan = plan_fresh(last_insertion.lock().unwrap().as_ref(), &text);
            apply(handle, last_insertion, plan, "Idle", "Idle (copied to clipboard)");
        }
        Ok(Transcript::Correction(replacement)) => {
            let _ = handle.emit("auralis://transcript", &replacement);
            let previous = last_insertion.lock().unwrap().take();
            match previous {
                Some(previous) => {
                    let plan = plan_correction(&previous, &replacement);
                    apply(handle, last_insertion, plan, "Idle (corrected)", "Idle (correction copied to clipboard)");
                }
                None => {
                    warn!("correction detected but nothing was tracked as typed; inserting fresh instead");
                    let plan = plan_fresh(None, &replacement);
                    apply(handle, last_insertion, plan, "Idle", "Idle (copied to clipboard)");
                }
            }
        }
        Ok(Transcript::Empty) => {
            let _ = handle.emit("auralis://status", "Idle (no speech detected)");
        }
        Err(e) => {
            error!("transcription failed: {e}");
            let _ = handle.emit("auralis://status", format!("Transcription failed: {e}"));
        }
    }
}

/// Carries out `plan` via keystroke injection (backspacing first if it is a
/// correction), records what landed so a later correction can undo exactly
/// that much, and reports status. A clipboard fallback or an outright failure
/// clears the record, since nothing was typed for a correction to undo.
fn apply(
    handle: &AppHandle,
    last_insertion: &Mutex<Option<LastInsertion>>,
    plan: Insertion,
    ok_status: &str,
    clipboard_status: &str,
) {
    let typed = if plan.undo_chars > 0 {
        crate::inject::replace_text(plan.undo_chars, &plan.text)
    } else {
        crate::inject::insert_text(&plan.text)
    };

    match typed {
        Ok(true) => {
            *last_insertion.lock().unwrap() = Some(plan.record);
            let _ = handle.emit("auralis://status", ok_status);
        }
        Ok(false) => {
            *last_insertion.lock().unwrap() = None;
            let _ = handle.emit("auralis://status", clipboard_status);
        }
        Err(e) => {
            error!("keystroke injection failed: {e}");
            *last_insertion.lock().unwrap() = None;
            let _ = handle.emit("auralis://status", format!("Injection error: {e}"));
        }
    }
}

fn handle_push_to_talk_event(app: &AppHandle, state: &AppState, event: tauri_plugin_global_shortcut::ShortcutEvent) {
    match event.state() {
        ShortcutState::Pressed => {
            state.held.store(true, Ordering::SeqCst);
            let _ = app.emit("auralis://status", "Listening");

            let held = state.held.clone();
            let pipeline = state.pipeline.clone();
            let last_insertion = state.last_insertion.clone();
            let mic_device = state.config.lock().unwrap().mic_device.clone();
            let handle = app.clone();

            std::thread::spawn(move || {
                let capture = match AudioCapture::start_with_device(mic_device.as_deref()) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("failed to open microphone: {e}");
                        let _ = handle.emit("auralis://status", format!("Mic error: {e}"));
                        held.store(false, Ordering::SeqCst);
                        return;
                    }
                };

                let _ = handle.emit("auralis://status", "Processing");

                let mut pipeline_guard = pipeline.lock().unwrap();
                let Some(pipeline) = pipeline_guard.as_mut() else {
                    let _ = handle.emit("auralis://status", "Model not loaded — run models/pull-model.ps1");
                    held.store(false, Ordering::SeqCst);
                    return;
                };
                let result = pipeline.run_once(&capture, || held.load(Ordering::SeqCst));
                handle_transcript_result(&handle, &last_insertion, result);
            });
        }
        ShortcutState::Released => {
            state.held.store(false, Ordering::SeqCst);
        }
    }
}

fn handle_toggle_event(app: &AppHandle, state: &AppState, event: tauri_plugin_global_shortcut::ShortcutEvent) {
    if event.state() != ShortcutState::Pressed {
        return;
    }

    let was_active = state
        .continuous_active
        .swap(!state.continuous_active.load(Ordering::SeqCst), Ordering::SeqCst);
    let now_active = !was_active;

    if !now_active {
        // Flag flip alone stops the running thread's loop (it polls this
        // same flag); nothing else to do here.
        return;
    }

    info!("continuous mode started");
    let _ = app.emit("auralis://status", "Listening (continuous)");

    let continuous_active = state.continuous_active.clone();
    let pipeline = state.pipeline.clone();
    let last_insertion = state.last_insertion.clone();
    let mic_device = state.config.lock().unwrap().mic_device.clone();
    let handle = app.clone();

    std::thread::spawn(move || {
        let capture = match AudioCapture::start_with_device(mic_device.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                error!("failed to open microphone: {e}");
                let _ = handle.emit("auralis://status", format!("Mic error: {e}"));
                continuous_active.store(false, Ordering::SeqCst);
                return;
            }
        };

        let mut pipeline_guard = pipeline.lock().unwrap();
        let Some(pipeline) = pipeline_guard.as_mut() else {
            let _ = handle.emit("auralis://status", "Model not loaded — run models/pull-model.ps1");
            continuous_active.store(false, Ordering::SeqCst);
            return;
        };

        pipeline.run_continuous(
            &capture,
            || continuous_active.load(Ordering::SeqCst),
            |result| handle_transcript_result(&handle, &last_insertion, result),
        );

        info!("continuous mode stopped");
        let _ = handle.emit("auralis://status", "Idle");
    });
}

/// Parses and registers both hotkeys from the current config. Callers must
/// `unregister_all()` first if re-registering after a rebind.
fn register_hotkeys(app: &AppHandle) -> anyhow::Result<()> {
    let config = app.state::<AppState>().config.lock().unwrap().clone();

    let ptt_shortcut = hotkey::parse(&config.push_to_talk_hotkey)?;
    let app_for_ptt = app.clone();
    app.global_shortcut().on_shortcut(ptt_shortcut, move |app, _shortcut, event| {
        handle_push_to_talk_event(app, &app_for_ptt.state::<AppState>(), event);
    })?;

    let toggle_shortcut = hotkey::parse(&config.toggle_hotkey)?;
    let app_for_toggle = app.clone();
    app.global_shortcut().on_shortcut(toggle_shortcut, move |app, _shortcut, event| {
        handle_toggle_event(app, &app_for_toggle.state::<AppState>(), event);
    })?;

    Ok(())
}

fn show_settings_window(app: &AppHandle) {
    if let Some(window) = app.get_webview_window("settings") {
        let _ = window.show();
        let _ = window.set_focus();
        return;
    }

    if let Err(e) = WebviewWindowBuilder::new(app, "settings", WebviewUrl::App("settings.html".into()))
        .title("Auralis Settings")
        .inner_size(420.0, 460.0)
        .resizable(false)
        .build()
    {
        error!("failed to open settings window: {e}");
    }
}

#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> AppConfig {
    state.config.lock().unwrap().clone()
}

#[tauri::command]
fn list_models() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(models_dir())
        .map(|entries| {
            entries
                .flatten()
                .filter_map(|e| e.file_name().into_string().ok())
                .filter(|name| name.ends_with(".bin"))
                .collect()
        })
        .unwrap_or_default();
    names.sort();
    names
}

/// The cleanup modes this build supports, for the settings dropdown — so the
/// UI can't offer a mode `save_config` would then reject.
#[tauri::command]
fn list_cleanup_modes() -> Vec<String> {
    CleanupMode::all().iter().map(|m| m.as_str().to_string()).collect()
}

#[tauri::command]
fn list_mic_devices() -> Vec<String> {
    auralis_runtime::audio::list_input_device_names().unwrap_or_default()
}

#[tauri::command]
fn save_config(app: AppHandle, state: tauri::State<AppState>, new_config: AppConfig) -> Result<(), String> {
    hotkey::parse(&new_config.push_to_talk_hotkey).map_err(|e| e.to_string())?;
    hotkey::parse(&new_config.toggle_hotkey).map_err(|e| e.to_string())?;

    let cleanup_mode = CleanupMode::parse(&new_config.cleanup_mode)
        .ok_or_else(|| format!("unknown text cleanup mode {:?}", new_config.cleanup_mode))?;

    let (model_changed, hotkeys_changed) = {
        let current = state.config.lock().unwrap();
        (
            current.model_file != new_config.model_file,
            current.push_to_talk_hotkey != new_config.push_to_talk_hotkey
                || current.toggle_hotkey != new_config.toggle_hotkey,
        )
    };

    new_config.save(&state.config_dir).map_err(|e| e.to_string())?;

    // Takes effect on the next utterance, including mid-session, and needs
    // neither a model reload nor the pipeline lock.
    state.cleanup_mode.set(cleanup_mode);

    if model_changed {
        let model_path = models_dir().join(&new_config.model_file);
        match Pipeline::with_cleanup_mode(&model_path, state.cleanup_mode.clone()) {
            Ok(p) => *state.pipeline.lock().unwrap() = Some(p),
            Err(e) => return Err(format!("failed to load model {}: {e}", new_config.model_file)),
        }
    }

    *state.config.lock().unwrap() = new_config;

    if hotkeys_changed {
        let _ = app.global_shortcut().unregister_all();
        register_hotkeys(&app).map_err(|e| e.to_string())?;
    }

    Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // RUST_LOG=auralis_runtime=debug,auralis_desktop_lib=debug for verbose
    // per-stage pipeline tracing; defaults to warnings/errors only.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![get_config, save_config, list_models, list_mic_devices, list_cleanup_modes])
        .setup(|app| {
            let config_dir = app.path().app_config_dir().unwrap_or_else(|_| PathBuf::from("."));
            let config = AppConfig::load(&config_dir);

            let model_path = models_dir().join(&config.model_file);
            if !model_path.exists() {
                warn!("Model not found at {model_path:?}. Run models/pull-model.ps1 first.");
            }
            // A missing/corrupt model shouldn't crash this tray-only app with no
            // visible window and no explanation — load it lazily and report a
            // clear status if a hotkey is pressed before it's available, instead
            // of panicking the whole process in setup().
            let cleanup_mode = CleanupModeHandle::new(config.cleanup_mode());
            let pipeline = Pipeline::with_cleanup_mode(&model_path, cleanup_mode.clone())
                .map_err(|e| error!("failed to load STT pipeline: {e:#}"))
                .ok();

            app.manage(AppState {
                config: Mutex::new(config),
                config_dir,
                pipeline: Arc::new(Mutex::new(pipeline)),
                held: Arc::new(AtomicBool::new(false)),
                continuous_active: Arc::new(AtomicBool::new(false)),
                last_insertion: Arc::new(Mutex::new(None)),
                cleanup_mode,
            });

            register_hotkeys(app.handle())?;

            let settings_item = MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&settings_item, &quit_item])?;

            TrayIconBuilder::new()
                .icon(app.default_window_icon().cloned().unwrap())
                .menu(&menu)
                .tooltip("Auralis")
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "settings" => show_settings_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running auralis-desktop");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replays a plan against a buffer the way the OS would: backspaces pop
    /// characters off the end, then the plan's text is typed.
    fn type_into(buffer: &mut String, plan: &Insertion) {
        for _ in 0..plan.undo_chars {
            buffer.pop();
        }
        buffer.push_str(&plan.text);
    }

    #[test]
    fn first_insertion_has_no_leading_separator() {
        let plan = plan_fresh(None, "Hello there.");
        assert_eq!(plan.text, "Hello there.");
        assert_eq!(plan.undo_chars, 0);
        assert_eq!(plan.record.separator, "");
        assert_eq!(plan.record.total_chars, 12);
    }

    #[test]
    fn later_insertions_are_separated_by_a_space() {
        let first = plan_fresh(None, "Hello there.");
        let second = plan_fresh(Some(&first.record), "How are you?");
        assert_eq!(second.text, " How are you?");
        assert_eq!(second.record.separator, " ");
        // The separator counts toward the undo length, since it was typed.
        assert_eq!(second.record.total_chars, 13);
    }

    #[test]
    fn correction_undoes_exactly_what_was_typed() {
        let first = plan_fresh(None, "Send the report to Rahul tomorrow.");
        let correction = plan_correction(&first.record, "Send the report to Rohan tomorrow.");
        assert_eq!(correction.undo_chars, first.text.chars().count());
        assert_eq!(correction.text, "Send the report to Rohan tomorrow.");
    }

    #[test]
    fn correction_retypes_the_separator_it_backspaced_over() {
        // Regression: a correction used to backspace the previous insertion in
        // full — separator included — but retype only the replacement body,
        // collapsing the space between two utterances in continuous mode.
        let first = plan_fresh(None, "Hello there.");
        let second = plan_fresh(Some(&first.record), "Send the report to Rahul tomorrow.");
        let correction = plan_correction(&second.record, "Send the report to Rohan tomorrow.");

        assert_eq!(correction.text, " Send the report to Rohan tomorrow.");

        let mut buffer = String::new();
        type_into(&mut buffer, &first);
        type_into(&mut buffer, &second);
        type_into(&mut buffer, &correction);
        assert_eq!(buffer, "Hello there. Send the report to Rohan tomorrow.");
    }

    #[test]
    fn chained_corrections_keep_the_separator_each_time() {
        let first = plan_fresh(None, "Hello there.");
        let second = plan_fresh(Some(&first.record), "Send it to Rahul tomorrow.");
        let one = plan_correction(&second.record, "Send it to Rohan tomorrow.");
        let two = plan_correction(&one.record, "Send it to Rohan Friday.");

        let mut buffer = String::new();
        type_into(&mut buffer, &first);
        type_into(&mut buffer, &second);
        type_into(&mut buffer, &one);
        type_into(&mut buffer, &two);
        assert_eq!(buffer, "Hello there. Send it to Rohan Friday.");
    }

    #[test]
    fn a_fresh_utterance_after_a_correction_is_still_separated() {
        let first = plan_fresh(None, "Send it to Rahul.");
        let correction = plan_correction(&first.record, "Send it to Rohan.");
        let next = plan_fresh(Some(&correction.record), "Also cc the design team.");

        let mut buffer = String::new();
        type_into(&mut buffer, &first);
        type_into(&mut buffer, &correction);
        type_into(&mut buffer, &next);
        assert_eq!(buffer, "Send it to Rohan. Also cc the design team.");
    }

    #[test]
    fn undo_length_counts_characters_not_bytes() {
        // Backspace deletes one character, so a multi-byte transcript must be
        // counted in chars — byte length would over-delete into earlier text.
        let plan = plan_fresh(None, "Déjà vu — naïve café.");
        assert_eq!(plan.record.total_chars, plan.text.chars().count());
        assert!(plan.record.total_chars < plan.text.len(), "fixture should be multi-byte");

        let mut buffer = String::from("Keep this. ");
        let follow_up = plan_fresh(Some(&plan.record), "Déjà vu — naïve café.");
        type_into(&mut buffer, &follow_up);
        let correction = plan_correction(&follow_up.record, "Deja vu.");
        type_into(&mut buffer, &correction);
        assert_eq!(buffer, "Keep this.  Deja vu.");
    }
}
