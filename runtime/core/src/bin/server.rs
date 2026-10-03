use anyhow::{Context, Result};
use auralis_runtime::text::{self, CleanupMode};
use auralis_runtime::{denoise, resample, stt::SttEngine, vad, wav};
use clap::Parser;
use serde_json::json;
use std::io::Read;
use std::path::PathBuf;
use std::time::Instant;
use tiny_http::{Header, Method, Request, Response, Server};

const MAX_BODY_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    model: PathBuf,

    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    #[arg(long, default_value_t = 8787)]
    port: u16,
}

fn json_response(status: u16, body: serde_json::Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(header)
}

fn query_param<'a>(url: &'a str, key: &str) -> Option<&'a str> {
    let query = url.split_once('?')?.1;
    query
        .split('&')
        .filter_map(|pair| pair.split_once('='))
        .find(|(k, _)| *k == key)
        .map(|(_, v)| v)
}

fn transcribe(engine: &SttEngine, samples: &[f32], rate: u32, mode: CleanupMode) -> Result<String> {
    let at_48k = resample::resample(samples, rate, 48_000);
    let trimmed = vad::trim_silence(&at_48k);
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    let denoised = denoise::denoise_48k(&trimmed);
    let at_16k = resample::resample(&denoised, 48_000, 16_000);
    let raw = engine.transcribe(&at_16k)?;
    Ok(text::clean_transcript_with(&raw, mode))
}

fn handle(engine: &SttEngine, model_name: &str, request: &mut Request) -> Response<std::io::Cursor<Vec<u8>>> {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("");

    match (request.method(), path) {
        (Method::Get, "/healthz") => json_response(200, json!({ "status": "ok", "model": model_name })),
        (Method::Post, "/v1/transcriptions") => {
            let mode = match query_param(&url, "cleanup") {
                None => CleanupMode::default(),
                Some(v) => match CleanupMode::parse(v) {
                    Some(m) => m,
                    None => return json_response(400, json!({ "error": format!("unknown cleanup mode {v:?}") })),
                },
            };

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

            let started = Instant::now();
            match transcribe(engine, &samples, rate, mode) {
                Ok(text) => json_response(
                    200,
                    json!({
                        "text": text,
                        "duration_s": samples.len() as f64 / rate as f64,
                        "processing_s": started.elapsed().as_secs_f64(),
                        "model": model_name,
                    }),
                ),
                Err(e) => json_response(500, json!({ "error": format!("{e:#}") })),
            }
        }
        (_, "/healthz") | (_, "/v1/transcriptions") => json_response(405, json!({ "error": "method not allowed" })),
        _ => json_response(404, json!({ "error": "not found" })),
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let engine = SttEngine::load(&args.model)?;
    let model_name = args
        .model
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();

    let addr = format!("{}:{}", args.bind, args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    println!("auralis-server listening on http://{addr}");

    for mut request in server.incoming_requests() {
        let response = handle(&engine, &model_name, &mut request);
        if let Err(e) = request.respond(response) {
            eprintln!("failed to send response: {e}");
        }
    }
    Ok::<(), anyhow::Error>(()).context("server stopped")
}
