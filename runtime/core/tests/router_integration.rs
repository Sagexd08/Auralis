use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use tiny_http::{Header, Response, Server};

struct Guard(Child);
impl Drop for Guard {
    fn drop(&mut self) {
        let _ = self.0.kill();
    }
}

struct StubNode {
    server: Option<Arc<Server>>,
    port: u16,
    thread: Option<JoinHandle<()>>,
}

impl StubNode {
    fn start(name: &'static str) -> Self {
        let server = Arc::new(Server::http("127.0.0.1:0").unwrap());
        let port = server.server_addr().to_ip().unwrap().port();
        let worker = Arc::clone(&server);
        let thread = thread::spawn(move || {
            for request in worker.incoming_requests() {
                let body = if request.url() == "/healthz" {
                    format!("{{\"status\":\"ok\",\"model\":\"{name}\"}}")
                } else {
                    format!("{{\"text\":\"from {name}\"}}")
                };
                let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
                let _ = request.respond(Response::from_string(body).with_header(header));
            }
        });
        StubNode { server: Some(server), port, thread: Some(thread) }
    }

    fn stop(&mut self) {
        if let Some(server) = self.server.take() {
            server.unblock();
        }
        if let Some(t) = self.thread.take() {
            let _ = t.join();
        }
    }
}

fn free_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}

fn http(port: u16, request: &str) -> (u16, String) {
    let mut stream = TcpStream::connect(("127.0.0.1", port)).unwrap();
    stream.write_all(request.as_bytes()).unwrap();
    let mut raw = Vec::new();
    stream.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8_lossy(&raw).to_string();
    let status = text.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("").to_string();
    (status, body)
}

fn transcribe(port: u16) -> (u16, String) {
    http(
        port,
        "POST /v1/transcriptions?cleanup=clean HTTP/1.1\r\nHost: x\r\nContent-Length: 4\r\nConnection: close\r\n\r\nRIFF",
    )
}

#[test]
fn balances_across_nodes_and_fails_over() {
    let mut a = StubNode::start("node-a");
    let mut b = StubNode::start("node-b");
    let port = free_port();

    let mut child = Command::new(env!("CARGO_BIN_EXE_auralis-router"))
        .args([
            "--port",
            &port.to_string(),
            "--health-interval",
            "1",
            "--node",
            &format!("http://127.0.0.1:{}", a.port),
            "--node",
            &format!("http://127.0.0.1:{}", b.port),
        ])
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut banner = String::new();
    BufReader::new(child.stdout.take().unwrap()).read_line(&mut banner).unwrap();
    let _guard = Guard(child);
    assert!(banner.contains("listening"), "unexpected banner: {banner:?}");

    let (status, body) = http(port, "GET /healthz HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    assert!(body.contains("\"healthy_nodes\":2"), "{body}");

    let mut seen = std::collections::HashSet::new();
    for _ in 0..4 {
        let (status, body) = transcribe(port);
        assert_eq!(status, 200, "{body}");
        seen.insert(body);
    }
    assert!(seen.contains("{\"text\":\"from node-a\"}") && seen.contains("{\"text\":\"from node-b\"}"), "{seen:?}");

    let (status, body) = http(port, "GET /v1/nodes HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 200);
    assert!(body.contains("node-a") && body.contains("node-b"), "{body}");

    let (_, metrics) = http(port, "GET /metrics HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert!(metrics.contains("auralis_router_routed_total 4"), "{metrics}");

    a.stop();
    for _ in 0..3 {
        let (status, body) = transcribe(port);
        assert_eq!(status, 200, "failover should reach node-b: {body}");
        assert!(body.contains("node-b"), "{body}");
    }

    b.stop();
    let (status, _) = transcribe(port);
    assert_eq!(status, 503);

    let (status, _) = http(port, "GET /nope HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n");
    assert_eq!(status, 404);
}
