use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};
use tungstenite::{connect, Message};

struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

fn start(port: u16, extra: &[&str]) -> Option<Guard> {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models/ggml-base.en-q5_1.bin");
    if !model.exists() {
        eprintln!("skipping: model not found at {model:?}");
        return None;
    }
    let mut child = Command::new(env!("CARGO_BIN_EXE_auralis-server"))
        .args(["--model", model.to_str().unwrap(), "--port", &port.to_string()])
        .args(extra)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut banner = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    assert!(banner.contains("listening"), "unexpected banner: {banner:?}");
    Some(Guard(child))
}

fn exchange(port: u16, request: &[u8]) -> (u16, String, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.write_all(request).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text.split_whitespace().nth(1).unwrap().parse().unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap_or((&text, ""));
    (status, head.to_string(), body.to_string())
}

fn http(port: u16, request: &str) -> (u16, String, String) {
    exchange(port, request.as_bytes())
}

fn post_wav(port: u16, query: &str, wav: &[u8]) -> (u16, String) {
    let mut request = format!(
        "POST /v1/transcriptions{query} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        wav.len()
    )
    .into_bytes();
    request.extend_from_slice(wav);
    let (status, _, body) = exchange(port, &request);
    (status, body)
}

fn jfk_wav() -> Vec<u8> {
    std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/jfk.wav")).unwrap()
}

#[test]
fn models_timestamps_language_and_origin_rules() {
    let Some(_guard) = start(18801, &["--allow-origin", "https://auralis-speak.vercel.app"]) else { return };
    let port = 18801;

    let (status, _, body) = http(port, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    let models: serde_json::Value = serde_json::from_str(&body).unwrap();
    let active = models["data"].as_array().unwrap().iter().find(|m| m["active"] == true).unwrap();
    assert_eq!(active["engine"], "whisper.cpp");
    assert_eq!(active["languages"][0], "en");

    let (status, body) = post_wav(port, "?timestamps=true", &jfk_wav());
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    let segments = v["segments"].as_array().unwrap();
    assert!(!segments.is_empty());
    assert!(segments[0]["end"].as_f64().unwrap() > segments[0]["start"].as_f64().unwrap());
    assert!(v["rtf"].as_f64().unwrap() > 0.0);
    assert!(v["text"].as_str().unwrap().to_lowercase().contains("country"));

    let (status, body) = post_wav(port, "?language=fr", &jfk_wav());
    assert_eq!(status, 400);
    assert!(body.contains("not supported"));

    let (status, _, _) = http(port, "GET /healthz HTTP/1.1\r\nHost: evil.example\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 403, "a non-loopback Host header must be refused");

    let (status, _, _) = http(port, "GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 403);

    let (status, head, _) = http(
        port,
        "GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://auralis-speak.vercel.app\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 200);
    assert!(head.to_lowercase().contains("access-control-allow-origin: https://auralis-speak.vercel.app"));

    let (status, head, _) = http(
        port,
        "OPTIONS /v1/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://auralis-speak.vercel.app\r\nConnection: close\r\n\r\n",
    );
    assert_eq!(status, 204);
    assert!(head.to_lowercase().contains("access-control-allow-methods"));
}

#[test]
fn websocket_streams_partials_then_a_final_transcript() {
    let Some(_guard) = start(18802, &[]) else { return };

    let reader = hound::WavReader::new(std::io::Cursor::new(jfk_wav())).unwrap();
    assert_eq!(reader.spec().sample_rate, 16_000);
    let pcm: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();

    let (mut ws, response) = connect("ws://127.0.0.1:18802/ws/v1/transcribe").unwrap();
    assert_eq!(response.status().as_u16(), 101);
    ws.send(Message::Text(r#"{"type":"start","sample_rate":16000,"cleanup":"clean"}"#.into())).unwrap();
    let ready: serde_json::Value = match ws.read().unwrap() {
        Message::Text(t) => serde_json::from_str(&t).unwrap(),
        other => panic!("unexpected {other:?}"),
    };
    assert_eq!(ready["type"], "ready");

    for chunk in pcm.chunks(4000) {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        ws.send(Message::Binary(bytes)).unwrap();
    }
    ws.send(Message::Text(r#"{"type":"stop"}"#.into())).unwrap();

    let mut partials = 0;
    let final_message = loop {
        match ws.read().unwrap() {
            Message::Text(t) => {
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                match v["type"].as_str().unwrap() {
                    "partial" => partials += 1,
                    "final" => break v,
                    "error" => panic!("server error: {v}"),
                    _ => {}
                }
            }
            Message::Close(_) => panic!("closed before final"),
            _ => {}
        }
    };
    assert!(partials >= 1, "expected at least one partial before the final");
    assert!(final_message["text"].as_str().unwrap().to_lowercase().contains("country"), "{final_message}");
    assert!(final_message["duration_s"].as_f64().unwrap() > 5.0);
}

#[test]
fn serves_the_auralis_onnx_model_through_the_same_endpoints() {
    let model = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/auralis_tiny/auralis.onnx");
    let port = 18804;
    let mut child = Command::new(env!("CARGO_BIN_EXE_auralis-server"))
        .args(["--model", model.to_str().unwrap(), "--port", &port.to_string()])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut banner = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    let _guard = Guard(child);
    assert!(banner.contains("listening"), "unexpected banner: {banner:?}");

    let (status, _, body) = http(port, "GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    assert!(body.contains("auralis-onnx"), "{body}");

    let (status, _, body) = http(port, "GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    let models: serde_json::Value = serde_json::from_str(&body).unwrap();
    let active = models["data"].as_array().unwrap().iter().find(|m| m["active"] == true).unwrap();
    assert_eq!(active["engine"], "auralis-onnx");
    assert_eq!(active["languages"][0], "en");

    let raw: Vec<f32> = std::fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/auralis_tiny/sample.f32"))
        .unwrap()
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut wav = Vec::new();
    {
        let spec = hound::WavSpec { channels: 1, sample_rate: 16_000, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
        let mut writer = hound::WavWriter::new(std::io::Cursor::new(&mut wav), spec).unwrap();
        for s in raw {
            writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
        }
        writer.finalize().unwrap();
    }
    let (status, body) = post_wav(port, "?timestamps=true", &wav);
    assert_eq!(status, 200, "{body}");
    let v: serde_json::Value = serde_json::from_str(&body).unwrap();
    assert!(v["model"].as_str().unwrap().ends_with("auralis.onnx"));
    assert!(v["rtf"].as_f64().unwrap() > 0.0);
}

#[test]
fn websocket_refuses_a_foreign_origin() {
    let Some(_guard) = start(18803, &[]) else { return };
    let (status, _, _) = http(
        18803,
        "GET /ws/v1/transcribe HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: https://evil.example\r\nUpgrade: websocket\r\nConnection: Upgrade\r\nSec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\nSec-WebSocket-Version: 13\r\n\r\n",
    );
    assert_eq!(status, 403);
}
