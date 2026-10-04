use anyhow::{Context, Result};
use auralis_runtime::stt::{join_segments, Segment, SttEngine, Speech};
use auralis_runtime::text::{self, CleanupMode};
use auralis_runtime::{denoise, resample, vad, wav};
use clap::Parser;
use serde_json::{json, Value};
use std::io::Read;
use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};
use tungstenite::handshake::derive_accept_key;
use tungstenite::protocol::Role;
use tungstenite::{Message, WebSocket};

const MAX_BODY_BYTES: u64 = 50 * 1024 * 1024;
const MAX_CONNECTIONS: usize = 32;
const MAX_STREAM_SECONDS: f64 = 300.0;
const PARTIAL_EVERY_SECONDS: f64 = 0.6;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    #[arg(long, default_value_t = 8787)]
    port: u16,

    #[arg(long = "allow-origin")]
    allow_origin: Vec<String>,
}

#[derive(Default)]
struct Metrics {
    requests: AtomicU64,
    errors: AtomicU64,
    audio_millis: AtomicU64,
    processing_micros: AtomicU64,
    stream_sessions: AtomicU64,
    stream_partials: AtomicU64,
    active: AtomicUsize,
}

impl Metrics {
    fn render(&self, model_name: &str) -> String {
        format!(
            "# HELP auralis_requests_total Transcription requests received.\n\
             # TYPE auralis_requests_total counter\n\
             auralis_requests_total {}\n\
             # HELP auralis_errors_total Transcription requests that failed.\n\
             # TYPE auralis_errors_total counter\n\
             auralis_errors_total {}\n\
             # HELP auralis_audio_seconds_total Seconds of audio transcribed.\n\
             # TYPE auralis_audio_seconds_total counter\n\
             auralis_audio_seconds_total {:.3}\n\
             # HELP auralis_processing_seconds_total Seconds spent transcribing.\n\
             # TYPE auralis_processing_seconds_total counter\n\
             auralis_processing_seconds_total {:.3}\n\
             # HELP auralis_stream_sessions_total WebSocket streaming sessions opened.\n\
             # TYPE auralis_stream_sessions_total counter\n\
             auralis_stream_sessions_total {}\n\
             # HELP auralis_stream_partials_total Partial transcripts sent over WebSocket.\n\
             # TYPE auralis_stream_partials_total counter\n\
             auralis_stream_partials_total {}\n\
             # HELP auralis_active_connections Connections being served now.\n\
             # TYPE auralis_active_connections gauge\n\
             auralis_active_connections {}\n\
             # HELP auralis_model_info Loaded model.\n\
             # TYPE auralis_model_info gauge\n\
             auralis_model_info{{model=\"{}\"}} 1\n",
            self.requests.load(Ordering::Relaxed),
            self.errors.load(Ordering::Relaxed),
            self.audio_millis.load(Ordering::Relaxed) as f64 / 1000.0,
            self.processing_micros.load(Ordering::Relaxed) as f64 / 1_000_000.0,
            self.stream_sessions.load(Ordering::Relaxed),
            self.stream_partials.load(Ordering::Relaxed),
            self.active.load(Ordering::Relaxed),
            model_name.replace('"', ""),
        )
    }
}

struct App {
    engine: Box<dyn Speech>,
    gate: Mutex<()>,
    model_name: String,
    model_dir: PathBuf,
    metrics: Metrics,
    allowed_origins: Vec<String>,
    loopback_only: bool,
}

type Reply = Response<std::io::Cursor<Vec<u8>>>;

fn text_response(body: String) -> Reply {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"text/plain; version=0.0.4"[..]).unwrap();
    Response::from_string(body).with_header(header)
}

fn json_response(status: u16, body: Value) -> Reply {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::from_string(body.to_string()).with_status_code(status).with_header(header)
}

fn query_param<'a>(url: &'a str, key: &str) -> Option<&'a str> {
    url.split_once('?')?
        .1
        .split('&')
        .find_map(|pair| pair.split_once('=').filter(|(k, _)| *k == key).map(|(_, v)| v))
}

fn header_value(request: &Request, name: &str) -> Option<String> {
    request
        .headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

fn is_loopback_host(host: &str) -> bool {
    let name = host.trim_start_matches('[').split(']').next().unwrap_or(host);
    let name = name.split(':').next().unwrap_or(name);
    matches!(name, "localhost" | "127.0.0.1" | "::1" | "")
}

fn origin_allowed(app: &App, origin: &str) -> bool {
    if app.allowed_origins.iter().any(|o| o == origin) {
        return true;
    }
    let rest = origin.strip_prefix("http://").or_else(|| origin.strip_prefix("https://"));
    rest.is_some_and(is_loopback_host)
}

fn with_cors(mut response: Reply, origin: Option<&str>) -> Reply {
    if let Some(origin) = origin {
        for (name, value) in [
            ("Access-Control-Allow-Origin", origin),
            ("Vary", "Origin"),
            ("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            ("Access-Control-Allow-Headers", "Content-Type"),
        ] {
            if let Ok(h) = Header::from_bytes(name.as_bytes(), value.as_bytes()) {
                response = response.with_header(h);
            }
        }
    }
    response
}

struct Prepared {
    samples_16k: Vec<f32>,
}

fn prepare(samples: &[f32], rate: u32, trim: bool) -> Prepared {
    let at_48k = resample::resample(samples, rate, 48_000);
    let voiced = if trim { vad::trim_silence(&at_48k) } else { at_48k };
    if voiced.is_empty() {
        return Prepared { samples_16k: Vec::new() };
    }
    let denoised = denoise::denoise_48k(&voiced);
    Prepared { samples_16k: resample::resample(&denoised, 48_000, 16_000) }
}

fn recognize(app: &App, samples: &[f32], rate: u32, trim: bool, language: Option<&str>) -> Result<Vec<Segment>> {
    let prepared = prepare(samples, rate, trim);
    if prepared.samples_16k.is_empty() {
        return Ok(Vec::new());
    }
    let _turn = app.gate.lock().unwrap_or_else(|e| e.into_inner());
    app.engine.segments(&prepared.samples_16k, language)
}

fn list_models(app: &App) -> Value {
    let mut ids: Vec<String> = std::fs::read_dir(&app.model_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.ends_with(".bin") || n.ends_with(".onnx"))
        .collect();
    if !ids.contains(&app.model_name) {
        ids.push(app.model_name.clone());
    }
    ids.sort();
    let data: Vec<Value> = ids
        .into_iter()
        .map(|id| {
            let active = id == app.model_name;
            json!({
                "id": id,
                "object": "model",
                "active": active,
                "engine": if active { Some(app.engine.engine_name()) } else { None },
                "languages": if active { app.engine.languages() } else { Vec::new() },
            })
        })
        .collect();
    json!({ "object": "list", "data": data })
}

fn transcriptions(app: &App, request: &mut Request, url: &str) -> Reply {
    let mode = match query_param(url, "cleanup") {
        None => CleanupMode::default(),
        Some(v) => match CleanupMode::parse(v) {
            Some(m) => m,
            None => return json_response(400, json!({ "error": format!("unknown cleanup mode {v:?}") })),
        },
    };
    let want_segments = query_param(url, "timestamps") == Some("true");
    let language = query_param(url, "language").filter(|l| !l.is_empty());
    if let Some(l) = language {
        let supported = app.engine.languages();
        if !supported.iter().any(|s| s == l || s == "auto") {
            return json_response(400, json!({ "error": format!("language {l:?} is not supported by this model"), "supported": supported }));
        }
    }

    let mut body = Vec::new();
    if let Err(e) = request.as_reader().take(MAX_BODY_BYTES + 1).read_to_end(&mut body) {
        return json_response(400, json!({ "error": format!("failed to read body: {e}") }));
    }
    if body.len() as u64 > MAX_BODY_BYTES {
        return json_response(413, json!({ "error": "audio larger than 50 MB" }));
    }

    let (samples, rate) = match wav::decode_wav_mono_f32(std::io::Cursor::new(body)) {
        Ok(decoded) => decoded,
        Err(e) => return json_response(400, json!({ "error": format!("{e:#}") })),
    };
    app.metrics.requests.fetch_add(1, Ordering::Relaxed);

    let started = Instant::now();
    let result = recognize(app, &samples, rate, !want_segments, language);
    let audio_seconds = samples.len() as f64 / rate as f64;
    app.metrics.audio_millis.fetch_add((audio_seconds * 1000.0) as u64, Ordering::Relaxed);
    app.metrics.processing_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
    match result {
        Ok(segments) => {
            let raw = join_segments(&segments);
            let text = text::clean_transcript_with(&raw, mode);
            let processing = started.elapsed().as_secs_f64();
            let mut out = json!({
                "text": text,
                "raw_text": raw,
                "duration_s": audio_seconds,
                "processing_s": processing,
                "rtf": if audio_seconds > 0.0 { processing / audio_seconds } else { 0.0 },
                "model": app.model_name,
                "language": language.unwrap_or(app.engine.languages().first().map(String::as_str).unwrap_or("auto")),
            });
            if want_segments {
                out["segments"] = json!(segments
                    .iter()
                    .map(|s| json!({ "start": s.start_s, "end": s.end_s, "text": s.text.trim() }))
                    .collect::<Vec<_>>());
            }
            json_response(200, out)
        }
        Err(e) => {
            app.metrics.errors.fetch_add(1, Ordering::Relaxed);
            json_response(500, json!({ "error": format!("{e:#}") }))
        }
    }
}

fn pcm16_to_f32(bytes: &[u8]) -> impl Iterator<Item = f32> + '_ {
    bytes.chunks_exact(2).map(|b| i16::from_le_bytes([b[0], b[1]]) as f32 / 32768.0)
}

fn send(ws: &mut WebSocket<Box<dyn tiny_http::ReadWrite + Send>>, value: Value) -> bool {
    ws.send(Message::Text(value.to_string())).is_ok()
}

fn stream_session(app: Arc<App>, request: Request) {
    let key = header_value(&request, "Sec-WebSocket-Key");
    let version = header_value(&request, "Sec-WebSocket-Version");
    let upgrade = header_value(&request, "Upgrade").unwrap_or_default().to_lowercase();
    let Some(key) = key.filter(|_| upgrade == "websocket" && version.as_deref() == Some("13")) else {
        let _ = request.respond(json_response(400, json!({ "error": "expected a WebSocket upgrade (version 13)" })));
        return;
    };
    let accept = derive_accept_key(key.as_bytes());
    let handshake = Response::new_empty(StatusCode(101))
        .with_header(Header::from_bytes(&b"Sec-WebSocket-Accept"[..], accept.as_bytes()).unwrap());
    let stream = request.upgrade("websocket", handshake);
    let mut ws = WebSocket::from_raw_socket(stream, Role::Server, None);
    app.metrics.stream_sessions.fetch_add(1, Ordering::Relaxed);

    let mut rate: u32 = 16_000;
    let mut mode = CleanupMode::default();
    let mut language: Option<String> = None;
    let mut samples: Vec<f32> = Vec::new();
    let mut decoded_len = 0usize;
    let mut last_decode_s = 0.0f64;
    let mut finished = false;

    while !finished {
        let message = match ws.read() {
            Ok(m) => m,
            Err(_) => break,
        };
        match message {
            Message::Binary(bytes) => {
                samples.extend(pcm16_to_f32(&bytes));
                if samples.len() as f64 / rate as f64 > MAX_STREAM_SECONDS {
                    send(&mut ws, json!({ "type": "error", "error": "stream longer than 5 minutes" }));
                    break;
                }
                let fresh = (samples.len() - decoded_len) as f64 / rate as f64;
                if fresh >= PARTIAL_EVERY_SECONDS.max(last_decode_s * 1.5) {
                    decoded_len = samples.len();
                    let started = Instant::now();
                    match recognize(&app, &samples, rate, true, language.as_deref()) {
                        Ok(segments) => {
                            last_decode_s = started.elapsed().as_secs_f64();
                            app.metrics.stream_partials.fetch_add(1, Ordering::Relaxed);
                            let ok = send(
                                &mut ws,
                                json!({
                                    "type": "partial",
                                    "text": text::clean_transcript_with(&join_segments(&segments), mode),
                                    "audio_s": samples.len() as f64 / rate as f64,
                                    "latency_ms": started.elapsed().as_millis() as u64,
                                }),
                            );
                            if !ok {
                                break;
                            }
                        }
                        Err(e) => {
                            send(&mut ws, json!({ "type": "error", "error": format!("{e:#}") }));
                            break;
                        }
                    }
                }
            }
            Message::Text(raw) => {
                let Ok(v) = serde_json::from_str::<Value>(&raw) else {
                    send(&mut ws, json!({ "type": "error", "error": "messages must be JSON" }));
                    continue;
                };
                match v.get("type").and_then(Value::as_str) {
                    Some("start") => {
                        if let Some(r) = v.get("sample_rate").and_then(Value::as_u64) {
                            if (8_000..=96_000).contains(&r) {
                                rate = r as u32;
                            }
                        }
                        if let Some(c) = v.get("cleanup").and_then(Value::as_str).and_then(CleanupMode::parse) {
                            mode = c;
                        }
                        language = v.get("language").and_then(Value::as_str).filter(|l| !l.is_empty()).map(str::to_string);
                        send(&mut ws, json!({ "type": "ready", "model": app.model_name, "sample_rate": rate }));
                    }
                    Some("stop") => finished = true,
                    _ => {
                        send(&mut ws, json!({ "type": "error", "error": "unknown message type" }));
                    }
                }
            }
            Message::Close(_) => break,
            _ => {}
        }
    }

    if finished {
        let started = Instant::now();
        let audio_seconds = samples.len() as f64 / rate as f64;
        match recognize(&app, &samples, rate, true, language.as_deref()) {
            Ok(segments) => {
                let processing = started.elapsed().as_secs_f64();
                app.metrics.requests.fetch_add(1, Ordering::Relaxed);
                app.metrics.audio_millis.fetch_add((audio_seconds * 1000.0) as u64, Ordering::Relaxed);
                app.metrics.processing_micros.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
                send(
                    &mut ws,
                    json!({
                        "type": "final",
                        "text": text::clean_transcript_with(&join_segments(&segments), mode),
                        "raw_text": join_segments(&segments),
                        "duration_s": audio_seconds,
                        "processing_s": processing,
                        "rtf": if audio_seconds > 0.0 { processing / audio_seconds } else { 0.0 },
                    }),
                );
            }
            Err(e) => {
                app.metrics.errors.fetch_add(1, Ordering::Relaxed);
                send(&mut ws, json!({ "type": "error", "error": format!("{e:#}") }));
            }
        }
        let _ = ws.close(None);
        let _ = ws.flush();
    }
}

fn route(app: Arc<App>, mut request: Request) {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();
    let origin = header_value(&request, "Origin");

    if app.loopback_only {
        if let Some(host) = header_value(&request, "Host") {
            if !is_loopback_host(&host) {
                let _ = request.respond(json_response(403, json!({ "error": "unexpected Host header" })));
                return;
            }
        }
    }
    if let Some(o) = origin.as_deref() {
        if !origin_allowed(&app, o) {
            let _ = request.respond(json_response(
                403,
                json!({ "error": "origin not allowed; start the server with --allow-origin to permit it" }),
            ));
            return;
        }
    }
    let cors = origin.as_deref();

    if *request.method() == Method::Options {
        let _ = request.respond(with_cors(Response::from_string("").with_status_code(204), cors));
        return;
    }

    if *request.method() == Method::Get && path == "/ws/v1/transcribe" {
        stream_session(app, request);
        return;
    }

    let response = match (request.method(), path.as_str()) {
        (Method::Get, "/healthz") => json_response(200, json!({ "status": "ok", "model": app.model_name, "engine": app.engine.engine_name() })),
        (Method::Get, "/metrics") => text_response(app.metrics.render(&app.model_name)),
        (Method::Get, "/v1/models") => json_response(200, list_models(&app)),
        (Method::Post, "/v1/transcriptions") => transcriptions(&app, &mut request, &url),
        (_, "/healthz") | (_, "/metrics") | (_, "/v1/models") | (_, "/v1/transcriptions") | (_, "/ws/v1/transcribe") => {
            json_response(405, json!({ "error": "method not allowed" }))
        }
        _ => json_response(404, json!({ "error": "not found" })),
    };
    if let Err(e) = request.respond(with_cors(response, cors)) {
        eprintln!("failed to send response: {e}");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let engine: Box<dyn Speech> = Box::new(SttEngine::load(&args.model)?);
    let model_name = args.model.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let model_dir = args.model.parent().map(PathBuf::from).unwrap_or_default();

    let addr = format!("{}:{}", args.bind, args.port);
    let listener = TcpListener::bind(&addr).with_context(|| format!("failed to bind {addr}"))?;
    let server = Server::from_listener(listener, None).map_err(|e| anyhow::anyhow!("failed to start server: {e}"))?;
    println!("auralis-server listening on http://{addr}");

    let app = Arc::new(App {
        engine,
        gate: Mutex::new(()),
        model_name,
        model_dir,
        metrics: Metrics::default(),
        allowed_origins: args.allow_origin,
        loopback_only: is_loopback_host(&args.bind),
    });

    for request in server.incoming_requests() {
        if app.metrics.active.load(Ordering::Relaxed) >= MAX_CONNECTIONS {
            let _ = request.respond(json_response(503, json!({ "error": "too many connections" })));
            continue;
        }
        let app = Arc::clone(&app);
        app.metrics.active.fetch_add(1, Ordering::Relaxed);
        std::thread::spawn(move || {
            let counter = Arc::clone(&app);
            route(app, request);
            counter.metrics.active.fetch_sub(1, Ordering::Relaxed);
        });
    }
    Ok(())
}
