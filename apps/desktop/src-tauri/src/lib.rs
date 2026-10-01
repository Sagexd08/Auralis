mod config;
mod hotkey;
mod inject;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::Pipeline;
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
}

fn models_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../models")
}

/// Emits a transcript event and inserts it via keystroke injection (falling
/// back to clipboard), updating the status event either way. Shared by
/// push-to-talk and continuous mode so both report status identically.
fn handle_transcript_result(handle: &AppHandle, result: anyhow::Result<String>) {
    match result {
        Ok(text) if !text.is_empty() => {
            let _ = handle.emit("auralis://transcript", &text);
            match crate::inject::insert_text(&text) {
                Ok(true) => {
                    let _ = handle.emit("auralis://status", "Idle");
                }
                Ok(false) => {
                    let _ = handle.emit("auralis://status", "Idle (copied to clipboard)");
                }
                Err(e) => {
                    error!("keystroke injection failed: {e}");
                    let _ = handle.emit("auralis://status", format!("Injection error: {e}"));
                }
            }
        }
        Ok(_) => {
            let _ = handle.emit("auralis://status", "Idle (no speech detected)");
        }
        Err(e) => {
            error!("transcription failed: {e}");
            let _ = handle.emit("auralis://status", format!("Transcription failed: {e}"));
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
                handle_transcript_result(&handle, result);
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
            |result| handle_transcript_result(&handle, result),
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

#[tauri::command]
fn list_mic_devices() -> Vec<String> {
    auralis_runtime::audio::list_input_device_names().unwrap_or_default()
}

#[tauri::command]
fn save_config(app: AppHandle, state: tauri::State<AppState>, new_config: AppConfig) -> Result<(), String> {
    hotkey::parse(&new_config.push_to_talk_hotkey).map_err(|e| e.to_string())?;
    hotkey::parse(&new_config.toggle_hotkey).map_err(|e| e.to_string())?;

    let (model_changed, hotkeys_changed) = {
        let current = state.config.lock().unwrap();
        (
            current.model_file != new_config.model_file,
            current.push_to_talk_hotkey != new_config.push_to_talk_hotkey
                || current.toggle_hotkey != new_config.toggle_hotkey,
        )
    };

    new_config.save(&state.config_dir).map_err(|e| e.to_string())?;

    if model_changed {
        let model_path = models_dir().join(&new_config.model_file);
        match Pipeline::new(&model_path) {
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
        .invoke_handler(tauri::generate_handler![get_config, save_config, list_models, list_mic_devices])
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
            let pipeline = Pipeline::new(&model_path)
                .map_err(|e| error!("failed to load STT pipeline: {e:#}"))
                .ok();

            app.manage(AppState {
                config: Mutex::new(config),
                config_dir,
                pipeline: Arc::new(Mutex::new(pipeline)),
                held: Arc::new(AtomicBool::new(false)),
                continuous_active: Arc::new(AtomicBool::new(false)),
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
