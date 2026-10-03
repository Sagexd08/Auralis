mod config;
mod hotkey;
mod inject;
mod models;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::{Pipeline, Transcript, CORRECTION_WINDOW};
use auralis_runtime::text::{CleanupMode, CleanupModeHandle};
use config::AppConfig;
use log::{error, info, warn};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::TrayIconBuilder;
use tauri::{AppHandle, Emitter, Manager, PhysicalPosition, WebviewUrl, WebviewWindowBuilder};
use tauri_plugin_global_shortcut::{GlobalShortcutExt, ShortcutState};

/// Shared mutable state, reachable from hotkey handlers and Tauri commands
/// alike via `app.state::<AppState>()`.
struct AppState {
    config: Mutex<AppConfig>,
    config_dir: PathBuf,
    pipeline: Arc<Mutex<Option<Pipeline>>>,
    held: Arc<AtomicBool>,
    /// True from a push-to-talk press until its recording ends. Rejects the
    /// OS key-repeat presses that arrive while the key is held, and stops
    /// push-to-talk and continuous mode from capturing at the same time.
    ptt_active: Arc<AtomicBool>,
    continuous_active: Arc<AtomicBool>,
    downloading: Arc<AtomicBool>,
    /// Bumped on every status change so a delayed "hide the overlay" only
    /// fires if nothing newer has been shown since.
    status_gen: Arc<AtomicU64>,
    /// Set when startup couldn't register the hotkeys; shown in Settings.
    startup_warning: Mutex<Option<String>>,
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

/// Locks `m`, recovering the data if another thread panicked while holding it
/// rather than turning one failed dictation into a permanently dead app.
fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
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
    /// When it landed. A correction or a separating space only makes sense
    /// shortly after, before the user has likely clicked elsewhere.
    at: Instant,
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
/// insertion, one after a clipboard fallback left nothing typed, and one long
/// after the previous (the cursor has probably moved) don't.
fn plan_fresh(previous: Option<&LastInsertion>, text: &str) -> Insertion {
    let recent = previous.is_some_and(|p| p.at.elapsed() <= CORRECTION_WINDOW);
    let separator = if recent { " " } else { "" };
    let full = format!("{separator}{text}");
    Insertion {
        undo_chars: 0,
        record: LastInsertion {
            separator: separator.to_string(),
            total_chars: full.chars().count(),
            at: Instant::now(),
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
            at: Instant::now(),
        },
        text: full,
    }
}

/// Shows `text` everywhere the user can see status: the overlay pill, the tray
/// tooltip, and the `auralis://status` event. Progress states keep the overlay
/// up; "Idle" hides it; anything else (a result or an error) shows briefly.
fn set_status(app: &AppHandle, text: impl Into<String>) {
    let text: String = text.into();
    let _ = app.emit("auralis://status", &text);
    if let Some(tray) = app.tray_by_id("main") {
        let _ = tray.set_tooltip(Some(format!("Auralis — {text}")));
    }

    let Some(window) = app.get_webview_window("main") else { return };
    let state = app.state::<AppState>();
    let generation = state.status_gen.fetch_add(1, Ordering::SeqCst) + 1;

    if text == "Idle" {
        let _ = window.hide();
        return;
    }
    let _ = window.show();

    let sticky = ["Listening", "Processing", "Downloading", "Loading"]
        .iter()
        .any(|p| text.starts_with(p));
    if !sticky {
        let gen_ref = state.status_gen.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(3500));
            if gen_ref.load(Ordering::SeqCst) == generation {
                let _ = window.hide();
            }
        });
    }
}

/// Emits a transcript event and applies it (inserting fresh text, or
/// replacing the previously-inserted text in place for a spoken correction),
/// updating the status either way. Shared by push-to-talk and continuous mode
/// so both report status identically.
fn handle_transcript_result(handle: &AppHandle, last_insertion: &Mutex<Option<LastInsertion>>, result: anyhow::Result<Transcript>) {
    match result {
        Ok(Transcript::Fresh(text)) => {
            let _ = handle.emit("auralis://transcript", &text);
            let plan = plan_fresh(lock(last_insertion).as_ref(), &text);
            apply(handle, last_insertion, plan, "Idle", "Copied to clipboard");
        }
        Ok(Transcript::Correction(replacement)) => {
            let _ = handle.emit("auralis://transcript", &replacement);
            let previous = lock(last_insertion).take();
            match previous.filter(|p| p.at.elapsed() <= CORRECTION_WINDOW) {
                Some(previous) => {
                    let plan = plan_correction(&previous, &replacement);
                    apply(handle, last_insertion, plan, "Corrected", "Correction copied to clipboard");
                }
                None => {
                    warn!("correction detected but nothing recent was tracked as typed; inserting fresh instead");
                    let plan = plan_fresh(None, &replacement);
                    apply(handle, last_insertion, plan, "Idle", "Copied to clipboard");
                }
            }
        }
        Ok(Transcript::Empty) => set_status(handle, "No speech detected"),
        Err(e) => {
            error!("transcription failed: {e}");
            set_status(handle, format!("Transcription failed: {e}"));
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
            *lock(last_insertion) = Some(plan.record);
            set_status(handle, ok_status);
        }
        Ok(false) => {
            *lock(last_insertion) = None;
            set_status(handle, clipboard_status);
        }
        Err(e) => {
            error!("keystroke injection failed: {e}");
            *lock(last_insertion) = None;
            set_status(handle, format!("Injection error: {e}"));
        }
    }
}

const MODEL_NOT_READY: &str = "Model not ready — open Settings to download one";

fn handle_push_to_talk_event(app: &AppHandle, state: &AppState, event: tauri_plugin_global_shortcut::ShortcutEvent) {
    match event.state() {
        ShortcutState::Pressed => {
            // Ignore OS key-repeat while held, a press during continuous
            // dictation, and a press while another recording is still open.
            if state.continuous_active.load(Ordering::SeqCst) || state.ptt_active.swap(true, Ordering::SeqCst) {
                return;
            }
            state.held.store(true, Ordering::SeqCst);
            set_status(app, "Listening");

            let held = state.held.clone();
            let ptt_active = state.ptt_active.clone();
            let pipeline = state.pipeline.clone();
            let last_insertion = state.last_insertion.clone();
            let mic_device = lock(&state.config).mic_device.clone();
            let handle = app.clone();

            std::thread::spawn(move || {
                let end_recording = || {
                    held.store(false, Ordering::SeqCst);
                    ptt_active.store(false, Ordering::SeqCst);
                };

                if lock(&pipeline).is_none() {
                    end_recording();
                    set_status(&handle, MODEL_NOT_READY);
                    return;
                }

                let capture = match AudioCapture::start_with_device(mic_device.as_deref()) {
                    Ok(c) => c,
                    Err(e) => {
                        error!("failed to open microphone: {e}");
                        end_recording();
                        set_status(&handle, format!("Mic error: {e}"));
                        return;
                    }
                };

                // Record without holding the pipeline lock, so a still-running
                // transcription never makes the user's speech wait.
                let samples = Pipeline::record_while(&capture, || held.load(Ordering::SeqCst));
                let sample_rate = capture.sample_rate;
                drop(capture);
                end_recording();

                set_status(&handle, "Processing");
                let result = match lock(&pipeline).as_mut() {
                    Some(p) => p.transcribe_recording(&samples, sample_rate),
                    None => Err(anyhow::anyhow!("model was unloaded")),
                };
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
    if state.ptt_active.load(Ordering::SeqCst) {
        return;
    }

    let was_active = state.continuous_active.fetch_xor(true, Ordering::SeqCst);
    if was_active {
        // Flag flip alone stops the running thread's loop (it polls this
        // same flag); nothing else to do here.
        return;
    }

    info!("continuous mode started");
    set_status(app, "Listening (continuous)");

    let continuous_active = state.continuous_active.clone();
    let pipeline = state.pipeline.clone();
    let last_insertion = state.last_insertion.clone();
    let mic_device = lock(&state.config).mic_device.clone();
    let handle = app.clone();

    std::thread::spawn(move || {
        let mut pipeline_guard = lock(&pipeline);
        let Some(pipeline) = pipeline_guard.as_mut() else {
            continuous_active.store(false, Ordering::SeqCst);
            set_status(&handle, MODEL_NOT_READY);
            return;
        };

        let capture = match AudioCapture::start_with_device(mic_device.as_deref()) {
            Ok(c) => c,
            Err(e) => {
                error!("failed to open microphone: {e}");
                continuous_active.store(false, Ordering::SeqCst);
                set_status(&handle, format!("Mic error: {e}"));
                return;
            }
        };

        pipeline.run_continuous(
            &capture,
            || continuous_active.load(Ordering::SeqCst),
            |result| handle_transcript_result(&handle, &last_insertion, result),
        );

        info!("continuous mode stopped");
        set_status(&handle, "Idle");
    });
}

/// Parses and registers both hotkeys from `config`. Callers must
/// `unregister_all()` first if re-registering after a rebind.
fn register_hotkeys(app: &AppHandle, config: &AppConfig) -> anyhow::Result<()> {
    let ptt_shortcut = hotkey::parse(&config.push_to_talk_hotkey)?;
    let toggle_shortcut = hotkey::parse(&config.toggle_hotkey)?;
    if ptt_shortcut == toggle_shortcut {
        anyhow::bail!("push-to-talk and toggle can't use the same hotkey");
    }

    let app_for_ptt = app.clone();
    app.global_shortcut().on_shortcut(ptt_shortcut, move |app, _shortcut, event| {
        handle_push_to_talk_event(app, &app_for_ptt.state::<AppState>(), event);
    })?;

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
        .inner_size(440.0, 640.0)
        .resizable(false)
        .build()
    {
        error!("failed to open settings window: {e}");
    }
}

/// Downloads `file` unless a download is already running, mirroring progress
/// to the overlay and to the settings window.
fn download_with_status(app: &AppHandle, file: &str) -> anyhow::Result<PathBuf> {
    let state = app.state::<AppState>();
    if state.downloading.swap(true, Ordering::SeqCst) {
        anyhow::bail!("a model download is already in progress");
    }
    let result = models::download(app, file, |progress| {
        if let Some(total) = progress.total.filter(|t| *t > 0) {
            set_status(app, format!("Downloading model… {}%", progress.downloaded * 100 / total));
        } else {
            set_status(app, "Downloading model…");
        }
        let _ = app.emit("auralis://model-progress", &progress);
    });
    state.downloading.store(false, Ordering::SeqCst);
    result
}

/// Makes sure a pipeline is loaded for the configured model, downloading it
/// first if it's a known model that isn't on disk yet (the first-run path for
/// an installed app, which ships without model weights). Blocking.
fn ensure_pipeline(app: &AppHandle) -> anyhow::Result<()> {
    let state = app.state::<AppState>();
    if lock(&state.pipeline).is_some() {
        return Ok(());
    }

    let model_file = lock(&state.config).model_file.clone();
    let path = match models::find(app, &model_file) {
        Some(p) => p,
        None => download_with_status(app, &model_file)?,
    };

    set_status(app, "Loading model…");
    let pipeline = Pipeline::with_cleanup_mode(&path, state.cleanup_mode.clone())?;
    *lock(&state.pipeline) = Some(pipeline);
    set_status(app, "Idle");
    Ok(())
}

#[tauri::command]
fn get_config(state: tauri::State<AppState>) -> AppConfig {
    lock(&state.config).clone()
}

#[tauri::command]
fn get_startup_warning(state: tauri::State<AppState>) -> Option<String> {
    lock(&state.startup_warning).clone()
}

#[tauri::command]
fn list_models(app: AppHandle) -> Vec<models::ModelInfo> {
    models::list(&app)
}

/// Downloads a catalog model; progress arrives as `auralis://model-progress`
/// events. If no model was loaded yet, loads the configured one afterwards.
#[tauri::command]
async fn download_model(app: AppHandle, file: String) -> Result<(), String> {
    let app_for_job = app.clone();
    tauri::async_runtime::spawn_blocking(move || -> anyhow::Result<()> {
        download_with_status(&app_for_job, &file)?;
        ensure_pipeline(&app_for_job)
    })
    .await
    .map_err(|e| e.to_string())?
    .map_err(|e| {
        set_status(&app, "Idle");
        format!("{e:#}")
    })?;
    set_status(&app, "Idle");
    Ok(())
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

/// Applies and persists settings. Everything that can fail (hotkey parsing,
/// loading the new model, registering the new hotkeys) happens before the live
/// state is replaced, and a failed rebind restores the previous hotkeys, so an
/// invalid save never leaves the app without working shortcuts or a model.
#[tauri::command]
async fn save_config(app: AppHandle, new_config: AppConfig) -> Result<(), String> {
    let state = app.state::<AppState>();

    let ptt = hotkey::parse(&new_config.push_to_talk_hotkey).map_err(|e| e.to_string())?;
    let toggle = hotkey::parse(&new_config.toggle_hotkey).map_err(|e| e.to_string())?;
    if ptt == toggle {
        return Err("push-to-talk and toggle can't use the same hotkey".to_string());
    }
    let cleanup_mode = CleanupMode::parse(&new_config.cleanup_mode)
        .ok_or_else(|| format!("unknown text cleanup mode {:?}", new_config.cleanup_mode))?;

    if state.continuous_active.load(Ordering::SeqCst) || state.ptt_active.load(Ordering::SeqCst) {
        return Err("Stop dictation before changing settings.".to_string());
    }

    let old_config = lock(&state.config).clone();
    let model_changed = old_config.model_file != new_config.model_file;
    let hotkeys_changed = old_config.push_to_talk_hotkey != new_config.push_to_talk_hotkey
        || old_config.toggle_hotkey != new_config.toggle_hotkey;

    let new_pipeline = if model_changed {
        let path = models::find(&app, &new_config.model_file)
            .ok_or_else(|| format!("model {} isn't downloaded yet", new_config.model_file))?;
        let mode = state.cleanup_mode.clone();
        let loaded = tauri::async_runtime::spawn_blocking(move || Pipeline::with_cleanup_mode(&path, mode))
            .await
            .map_err(|e| e.to_string())?
            .map_err(|e| format!("failed to load model {}: {e:#}", new_config.model_file))?;
        Some(loaded)
    } else {
        None
    };

    if hotkeys_changed {
        let _ = app.global_shortcut().unregister_all();
        if let Err(e) = register_hotkeys(&app, &new_config) {
            let _ = app.global_shortcut().unregister_all();
            let _ = register_hotkeys(&app, &old_config);
            return Err(format!("couldn't register those hotkeys (another app may own them): {e}"));
        }
        *lock(&state.startup_warning) = None;
    }

    if let Some(pipeline) = new_pipeline {
        *lock(&state.pipeline) = Some(pipeline);
    }
    state.cleanup_mode.set(cleanup_mode);
    *lock(&state.config) = new_config.clone();

    new_config
        .save(&state.config_dir)
        .map_err(|e| format!("applied, but couldn't be saved for next launch: {e}"))
}

/// Parks the status overlay at the bottom-centre of the primary monitor and
/// makes it click-through, so it never intercepts input or competes with the
/// application the user is dictating into.
fn place_overlay(app: &tauri::App) {
    let Some(window) = app.get_webview_window("main") else { return };
    let _ = window.set_ignore_cursor_events(true);
    if let (Ok(Some(monitor)), Ok(size)) = (window.primary_monitor(), window.outer_size()) {
        let m_size = monitor.size();
        let m_pos = monitor.position();
        let margin = (96.0 * monitor.scale_factor()) as i32;
        let x = m_pos.x + (m_size.width as i32 - size.width as i32) / 2;
        let y = m_pos.y + m_size.height as i32 - size.height as i32 - margin;
        let _ = window.set_position(PhysicalPosition::new(x, y));
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    tauri::Builder::default()
        // Must be first: a second launch would otherwise fight the first for
        // the global hotkeys and the microphone. It just opens Settings.
        .plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| show_settings_window(app)))
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .invoke_handler(tauri::generate_handler![
            get_config,
            save_config,
            list_models,
            download_model,
            list_mic_devices,
            list_cleanup_modes,
            get_startup_warning
        ])
        .setup(|app| {
            let config_dir = app.path().app_config_dir().unwrap_or_else(|_| PathBuf::from("."));
            let config = AppConfig::load(&config_dir);
            let cleanup_mode = CleanupModeHandle::new(config.cleanup_mode());

            app.manage(AppState {
                config: Mutex::new(config.clone()),
                config_dir,
                pipeline: Arc::new(Mutex::new(None)),
                held: Arc::new(AtomicBool::new(false)),
                ptt_active: Arc::new(AtomicBool::new(false)),
                continuous_active: Arc::new(AtomicBool::new(false)),
                downloading: Arc::new(AtomicBool::new(false)),
                status_gen: Arc::new(AtomicU64::new(0)),
                startup_warning: Mutex::new(None),
                last_insertion: Arc::new(Mutex::new(None)),
                cleanup_mode,
            });

            let settings_item = MenuItem::with_id(app, "settings", "Settings...", true, None::<&str>)?;
            let quit_item = MenuItem::with_id(app, "quit", "Quit", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&settings_item, &quit_item])?;

            TrayIconBuilder::with_id("main")
                .icon(app.default_window_icon().cloned().unwrap())
                .menu(&menu)
                .tooltip("Auralis")
                .on_menu_event(|app, event| match event.id().as_ref() {
                    "settings" => show_settings_window(app),
                    "quit" => app.exit(0),
                    _ => {}
                })
                .build(app)?;

            place_overlay(app);

            // A hotkey another program already owns (Ctrl+Space is an input-
            // method toggle on some setups) must not kill a tray-only app with
            // no window to explain why: surface it in Settings instead.
            if let Err(e) = register_hotkeys(app.handle(), &config) {
                error!("failed to register hotkeys: {e:#}");
                *lock(&app.state::<AppState>().startup_warning) = Some(format!(
                    "Couldn't register your hotkeys ({e}). Another app may be using them — pick different ones below."
                ));
                show_settings_window(app.handle());
            }

            // Loading (and, on first run, downloading) the model takes seconds
            // to minutes, so it must not block startup.
            let handle = app.handle().clone();
            std::thread::spawn(move || {
                if let Err(e) = ensure_pipeline(&handle) {
                    error!("failed to prepare the STT model: {e:#}");
                    set_status(&handle, format!("Model unavailable: {e:#}"));
                }
            });

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
