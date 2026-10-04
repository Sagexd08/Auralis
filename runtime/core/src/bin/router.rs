use anyhow::Result;
use clap::Parser;
use serde_json::{json, Value};
use std::io::Read;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tiny_http::{Header, Method, Request, Response, Server};

const MAX_BODY_BYTES: u64 = 50 * 1024 * 1024;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long = "node", required = true)]
    nodes: Vec<String>,

    #[arg(long, default_value = "127.0.0.1")]
    bind: String,

    #[arg(long, default_value_t = 8780)]
    port: u16,

    #[arg(long, default_value_t = 5)]
    health_interval: u64,
}

#[derive(Clone, Debug)]
struct Node {
    url: String,
    healthy: bool,
    model: String,
    inflight: u32,
    served: u64,
    failures: u64,
    last_latency_ms: u64,
}

impl Node {
    fn new(url: &str) -> Self {
        Node {
            url: url.trim_end_matches('/').to_string(),
            healthy: false,
            model: String::new(),
            inflight: 0,
            served: 0,
            failures: 0,
            last_latency_ms: 0,
        }
    }
}

fn pick_node(nodes: &[Node], skip: &[usize]) -> Option<usize> {
    nodes
        .iter()
        .enumerate()
        .filter(|(i, n)| n.healthy && !skip.contains(i))
        .min_by_key(|(_, n)| (n.inflight, n.served))
        .map(|(i, _)| i)
}

#[derive(Default)]
struct Counters {
    routed: AtomicU64,
    rejected: AtomicU64,
}

type Shared = Arc<Mutex<Vec<Node>>>;

fn json_response(status: u16, body: Value) -> Response<std::io::Cursor<Vec<u8>>> {
    let header = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
    Response::from_string(body.to_string())
        .with_status_code(status)
        .with_header(header)
}

fn check_health(url: &str) -> Option<String> {
    let resp = ureq::get(&format!("{url}/healthz")).timeout(Duration::from_secs(2)).call().ok()?;
    let body: Value = resp.into_json().ok()?;
    Some(body.get("model").and_then(Value::as_str).unwrap_or_default().to_string())
}

fn health_pass(nodes: &Shared) {
    let urls: Vec<String> = nodes.lock().unwrap().iter().map(|n| n.url.clone()).collect();
    for (i, url) in urls.iter().enumerate() {
        let started = Instant::now();
        let result = check_health(url);
        let mut guard = nodes.lock().unwrap();
        let node = &mut guard[i];
        node.healthy = result.is_some();
        if let Some(model) = result {
            node.model = model;
            node.last_latency_ms = started.elapsed().as_millis() as u64;
        }
    }
}

fn health_loop(nodes: Shared, interval: Duration) {
    loop {
        thread::sleep(interval);
        health_pass(&nodes);
    }
}

fn forward(url: &str, query: &str, body: &[u8]) -> Result<(u16, Vec<u8>), String> {
    let target = if query.is_empty() {
        format!("{url}/v1/transcriptions")
    } else {
        format!("{url}/v1/transcriptions?{query}")
    };
    let agent = ureq::AgentBuilder::new()
        .timeout_connect(Duration::from_secs(3))
        .timeout(Duration::from_secs(300))
        .build();
    let outcome = agent
        .post(&target)
        .set("Content-Type", "audio/wav")
        .send_bytes(body);
    let response = match outcome {
        Ok(r) => r,
        Err(ureq::Error::Status(_, r)) => r,
        Err(e) => return Err(e.to_string()),
    };
    let status = response.status();
    let mut out = Vec::new();
    response.into_reader().take(MAX_BODY_BYTES).read_to_end(&mut out).map_err(|e| e.to_string())?;
    Ok((status, out))
}

fn route_transcription(nodes: &Shared, counters: &Counters, url: &str, body: &[u8]) -> Response<std::io::Cursor<Vec<u8>>> {
    let query = url.split_once('?').map(|(_, q)| q).unwrap_or("");
    let mut tried: Vec<usize> = Vec::new();

    loop {
        let chosen = {
            let mut guard = nodes.lock().unwrap();
            let Some(i) = pick_node(&guard, &tried) else { break };
            guard[i].inflight += 1;
            (i, guard[i].url.clone())
        };
        let (index, node_url) = chosen;
        let result = forward(&node_url, query, body);

        let mut guard = nodes.lock().unwrap();
        guard[index].inflight -= 1;
        match result {
            Ok((status, payload)) => {
                guard[index].served += 1;
                counters.routed.fetch_add(1, Ordering::Relaxed);
                let content_type = Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..]).unwrap();
                let node_header = Header::from_bytes(&b"X-Auralis-Node"[..], node_url.as_bytes()).unwrap();
                return Response::from_data(payload)
                    .with_status_code(status)
                    .with_header(content_type)
                    .with_header(node_header);
            }
            Err(_) => {
                guard[index].failures += 1;
                guard[index].healthy = false;
                tried.push(index);
            }
        }
    }

    counters.rejected.fetch_add(1, Ordering::Relaxed);
    json_response(503, json!({ "error": "no healthy node available" }))
}

fn metrics_text(nodes: &Shared, counters: &Counters) -> String {
    let guard = nodes.lock().unwrap();
    let mut out = String::new();
    out.push_str("# TYPE auralis_router_routed_total counter\n");
    out.push_str(&format!("auralis_router_routed_total {}\n", counters.routed.load(Ordering::Relaxed)));
    out.push_str("# TYPE auralis_router_rejected_total counter\n");
    out.push_str(&format!("auralis_router_rejected_total {}\n", counters.rejected.load(Ordering::Relaxed)));
    out.push_str("# TYPE auralis_router_node_up gauge\n");
    for n in guard.iter() {
        out.push_str(&format!("auralis_router_node_up{{node=\"{}\"}} {}\n", n.url, u8::from(n.healthy)));
    }
    out.push_str("# TYPE auralis_router_node_inflight gauge\n");
    for n in guard.iter() {
        out.push_str(&format!("auralis_router_node_inflight{{node=\"{}\"}} {}\n", n.url, n.inflight));
    }
    out.push_str("# TYPE auralis_router_node_served_total counter\n");
    for n in guard.iter() {
        out.push_str(&format!("auralis_router_node_served_total{{node=\"{}\"}} {}\n", n.url, n.served));
    }
    out
}

fn handle(nodes: &Shared, counters: &Counters, mut request: Request) {
    let url = request.url().to_string();
    let path = url.split('?').next().unwrap_or("").to_string();

    let response = match (request.method().clone(), path.as_str()) {
        (Method::Get, "/healthz") => {
            let guard = nodes.lock().unwrap();
            let healthy = guard.iter().filter(|n| n.healthy).count();
            let status = if healthy > 0 { 200 } else { 503 };
            json_response(
                status,
                json!({ "status": if healthy > 0 { "ok" } else { "degraded" }, "healthy_nodes": healthy, "total_nodes": guard.len() }),
            )
        }
        (Method::Get, "/v1/nodes") => {
            let guard = nodes.lock().unwrap();
            let list: Vec<Value> = guard
                .iter()
                .map(|n| {
                    json!({
                        "url": n.url, "healthy": n.healthy, "model": n.model, "inflight": n.inflight,
                        "served": n.served, "failures": n.failures, "health_latency_ms": n.last_latency_ms,
                    })
                })
                .collect();
            json_response(200, json!({ "nodes": list }))
        }
        (Method::Get, "/metrics") => {
            let header = Header::from_bytes(&b"Content-Type"[..], &b"text/plain; version=0.0.4"[..]).unwrap();
            Response::from_string(metrics_text(nodes, counters)).with_header(header)
        }
        (Method::Post, "/v1/transcriptions") => {
            let mut body = Vec::new();
            match request.as_reader().take(MAX_BODY_BYTES + 1).read_to_end(&mut body) {
                Err(e) => json_response(400, json!({ "error": format!("failed to read body: {e}") })),
                Ok(_) if body.len() as u64 > MAX_BODY_BYTES => json_response(413, json!({ "error": "audio larger than 50 MB" })),
                Ok(_) => route_transcription(nodes, counters, &url, &body),
            }
        }
        (_, "/healthz") | (_, "/v1/nodes") | (_, "/metrics") | (_, "/v1/transcriptions") => {
            json_response(405, json!({ "error": "method not allowed" }))
        }
        _ => json_response(404, json!({ "error": "not found" })),
    };

    if let Err(e) = request.respond(response) {
        eprintln!("failed to send response: {e}");
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    let nodes: Shared = Arc::new(Mutex::new(args.nodes.iter().map(|u| Node::new(u)).collect()));

    health_pass(&nodes);
    {
        let nodes = Arc::clone(&nodes);
        let interval = Duration::from_secs(args.health_interval.max(1));
        thread::spawn(move || health_loop(nodes, interval));
    }

    let addr = format!("{}:{}", args.bind, args.port);
    let server = Server::http(&addr).map_err(|e| anyhow::anyhow!("failed to bind {addr}: {e}"))?;
    println!("auralis-router listening on http://{addr} with {} node(s)", args.nodes.len());

    let counters = Arc::new(Counters::default());
    for request in server.incoming_requests() {
        let nodes = Arc::clone(&nodes);
        let counters = Arc::clone(&counters);
        thread::spawn(move || handle(&nodes, &counters, request));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(healthy: bool, inflight: u32, served: u64) -> Node {
        Node { healthy, inflight, served, ..Node::new("http://n") }
    }

    #[test]
    fn picks_least_inflight_healthy_node() {
        let nodes = vec![node(true, 2, 0), node(true, 0, 9), node(false, 0, 0)];
        assert_eq!(pick_node(&nodes, &[]), Some(1));
    }

    #[test]
    fn breaks_ties_by_fewest_served() {
        let nodes = vec![node(true, 0, 5), node(true, 0, 2)];
        assert_eq!(pick_node(&nodes, &[]), Some(1));
    }

    #[test]
    fn skips_tried_and_unhealthy_nodes() {
        let nodes = vec![node(true, 0, 0), node(true, 1, 0), node(false, 0, 0)];
        assert_eq!(pick_node(&nodes, &[0]), Some(1));
        assert_eq!(pick_node(&nodes, &[0, 1]), None);
    }

    #[test]
    fn trims_trailing_slash_from_node_url() {
        assert_eq!(Node::new("http://a:1/").url, "http://a:1");
    }
}
