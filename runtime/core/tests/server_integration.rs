use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::{Child, Command, Stdio};

struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

fn http(port: u16, request: &[u8]) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.write_all(request).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

#[test]
fn serves_health_errors_and_transcriptions() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let model = root.join("../../models/ggml-base.en-q5_1.bin");
    if !model.exists() {
        eprintln!("skipping: model not found at {model:?}");
        return;
    }

    let port = 18787;
    let mut child = Command::new(env!("CARGO_BIN_EXE_auralis-server"))
        .args(["--model", model.to_str().unwrap(), "--port", &port.to_string()])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut ready = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut ready).unwrap();
    let _guard = Guard(child);
    assert!(ready.contains("listening"), "unexpected banner: {ready:?}");

    let (status, body) = http(port, b"GET /healthz HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    assert!(body.contains("\"ok\""));

    let (status, _) = http(port, b"GET /nope HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 404);

    let garbage = b"POST /v1/transcriptions HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 4\r\nConnection: close\r\n\r\nnope";
    let (status, _) = http(port, garbage);
    assert_eq!(status, 400);

    let wav = std::fs::read(root.join("tests/fixtures/jfk.wav")).unwrap();
    let mut request = format!(
        "POST /v1/transcriptions?cleanup=clean HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: audio/wav\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        wav.len()
    )
    .into_bytes();
    request.extend_from_slice(&wav);
    let (status, body) = http(port, &request);
    assert_eq!(status, 200, "body: {body}");
    assert!(body.to_lowercase().contains("country"), "body: {body}");
}
