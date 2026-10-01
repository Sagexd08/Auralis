mod inject;

use auralis_runtime::audio::AudioCapture;
use auralis_runtime::pipeline::Pipeline;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use tauri::Emitter;
use tauri_plugin_global_shortcut::{Code, GlobalShortcutExt, Modifiers, Shortcut, ShortcutState};

fn model_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../models/ggml-base.en-q5_1.bin")
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let held = Arc::new(AtomicBool::new(false));

    tauri::Builder::default()
        .plugin(tauri_plugin_global_shortcut::Builder::new().build())
        .setup(move |app| {
            let model = model_path();
            if !model.exists() {
                eprintln!(
                    "Model not found at {model:?}. Run models/pull-model.ps1 first."
                );
            }

            let pipeline = Arc::new(Mutex::new(
                Pipeline::new(&model).expect("failed to load STT pipeline"),
            ));

            let app_handle = app.handle().clone();
            let held_for_handler = held.clone();

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
                                    let _ = handle_inner.emit(
                                        "auralis://status",
                                        format!("Mic error: {e}"),
                                    );
                                    held_inner.store(false, Ordering::SeqCst);
                                    return;
                                }
                            };

                            let _ = handle_inner.emit("auralis://status", "Processing");

                            let result = {
                                let mut pipeline = pipeline_inner.lock().unwrap();
                                pipeline.run_once(&capture, || held_inner.load(Ordering::SeqCst))
                            };

                            match result {
                                Ok(text) if !text.is_empty() => {
                                    let _ = handle_inner.emit("auralis://transcript", &text);
                                    match crate::inject::insert_text(&text) {
                                        Ok(true) => {
                                            let _ = handle_inner.emit("auralis://status", "Idle");
                                        }
                                        Ok(false) => {
                                            let _ = handle_inner.emit(
                                                "auralis://status",
                                                "Idle (copied to clipboard)",
                                            );
                                        }
                                        Err(e) => {
                                            let _ = handle_inner.emit(
                                                "auralis://status",
                                                format!("Injection error: {e}"),
                                            );
                                        }
                                    }
                                }
                                Ok(_) => {
                                    let _ = handle_inner.emit("auralis://status", "Idle (no speech detected)");
                                }
                                Err(e) => {
                                    let _ = handle_inner.emit(
                                        "auralis://status",
                                        format!("Transcription failed: {e}"),
                                    );
                                }
                            }
                        });
                    }
                    ShortcutState::Released => {
                        held_for_handler.store(false, Ordering::SeqCst);
                    }
                }
            })?;

            Ok(())
        })
        .run(tauri::generate_context!())
        .expect("error while running auralis-desktop");
}
