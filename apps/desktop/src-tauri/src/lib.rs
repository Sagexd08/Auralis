mod inject;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::Pipeline;
use log::{error, info, warn};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::{AppHandle, Emitter};
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/ggml-base.en-q5_1.bin")
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // RUST_LOG=auralis_runtime=debug,auralis_desktop_lib=debug for verbose
    // per-stage pipeline tracing; defaults to warnings/errors only.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let held = Arc::new(AtomicBool::new(false));
    let continuous_active = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let model = model_path();
            if !model.exists() {
                warn!("Model not found at {model:?}. Run models/pull-model.ps1 first.");
            }

            // A missing/corrupt model shouldn't crash this tray-only app with no
            // visible window and no explanation — load it lazily and report a
            // clear status if a hotkey is pressed before it's available, instead
            // of panicking the whole process in setup().
            let pipeline: Arc<Mutex<Option<Pipeline>>> = Arc::new(Mutex::new(
                Pipeline::new(&model)
                    .map_err(|e| error!("failed to load STT pipeline: {e:#}"))
                    .ok(),
            ));

            let app_handle = app.handle().clone();

            // Push-to-talk: hold Ctrl+Space, speak, release.
            {
                let held_for_handler = held.clone();
                let pipeline = pipeline.clone();
                let app_handle = app_handle.clone();

                let shortcut = Shortcut::new(Some(Modifiers::CONTROL), Code::Space);
                app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                    match event.state() {
                        ShortcutState::Pressed => {
                            held_for_handler.store(true, Ordering::SeqCst);
                            let _ = app_handle.emit("auralis://status", "Listening");

                            let held_inner = held_for_handler.clone();
                            let pipeline_inner = pipeline.clone();
                            let handle_inner = app_handle.clone();

                            std::thread::spawn(move || {
                                let capture = match AudioCapture::start() {
                                    Ok(c) => c,
                                    Err(e) => {
                                        error!("failed to open microphone: {e}");
                                        let _ = handle_inner.emit("auralis://status", format!("Mic error: {e}"));
                                        held_inner.store(false, Ordering::SeqCst);
                                        return;
                                    }
                                };

                                let _ = handle_inner.emit("auralis://status", "Processing");

                                let mut pipeline_guard = pipeline_inner.lock().unwrap();
                                let Some(pipeline) = pipeline_guard.as_mut() else {
                                    let _ = handle_inner.emit("auralis://status", "Model not loaded — run models/pull-model.ps1");
                                    held_inner.store(false, Ordering::SeqCst);
                                    return;
                                };
                                let result = pipeline.run_once(&capture, || held_inner.load(Ordering::SeqCst));
                                handle_transcript_result(&handle_inner, result);
                            });
                        }
                        ShortcutState::Released => {
                            held_for_handler.store(false, Ordering::SeqCst);
                        }
                    }
                })?;
            }

            // Toggle/continuous mode: tap Ctrl+Shift+Space to start listening
            // continuously (auto-segmenting speech via VAD), tap again to stop.
            {
                let continuous_for_handler = continuous_active.clone();
                let pipeline = pipeline.clone();
                let app_handle = app_handle.clone();

                let shortcut = Shortcut::new(Some(Modifiers::CONTROL | Modifiers::SHIFT), Code::Space);
                app.global_shortcut().on_shortcut(shortcut, move |_app, _shortcut, event| {
                    if event.state() != ShortcutState::Pressed {
                        return;
                    }

                    let was_active = continuous_for_handler.swap(
                        !continuous_for_handler.load(Ordering::SeqCst),
                        Ordering::SeqCst,
                    );
                    let now_active = !was_active;

                    if !now_active {
                        // Flag flip alone stops the running thread's loop (it polls
                        // this same flag); nothing else to do here.
                        return;
                    }

                    info!("continuous mode started");
                    let _ = app_handle.emit("auralis://status", "Listening (continuous)");

                    let continuous_inner = continuous_for_handler.clone();
                    let pipeline_inner = pipeline.clone();
                    let handle_inner = app_handle.clone();

                    std::thread::spawn(move || {
                        let capture = match AudioCapture::start() {
                            Ok(c) => c,
                            Err(e) => {
                                error!("failed to open microphone: {e}");
                                let _ = handle_inner.emit("auralis://status", format!("Mic error: {e}"));
                                continuous_inner.store(false, Ordering::SeqCst);
                                return;
                            }
                        };

                        let mut pipeline_guard = pipeline_inner.lock().unwrap();
                        let Some(pipeline) = pipeline_guard.as_mut() else {
                            let _ = handle_inner.emit("auralis://status", "Model not loaded — run models/pull-model.ps1");
                            continuous_inner.store(false, Ordering::SeqCst);
                            return;
                        };

                        pipeline.run_continuous(
                            &capture,
                            || continuous_inner.load(Ordering::SeqCst),
                            |result| handle_transcript_result(&handle_inner, result),
                        );

                        info!("continuous mode stopped");
                        let _ = handle_inner.emit("auralis://status", "Idle");
                    });
                })?;
            }

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running auralis-desktop");
}
