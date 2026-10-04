# Infrastructure

What ships today is a small, honest compute fabric: transcription nodes, a router in front of
them, Prometheus metrics, and containers. It is not a general scheduler, and it does not yet
include a node agent, GPU-aware placement, NATS, Kubernetes manifests or a model registry.

```
client ── POST /v1/transcriptions ──► auralis-router ──► auralis-server (node A)
                                          │         └──► auralis-server (node B)
                                          │
                       /metrics ◄── Prometheus ◄── Grafana
```

## Components

| Binary | Purpose |
|---|---|
| `auralis-server` | One model behind `POST /v1/transcriptions`, `GET /healthz`, `GET /metrics`. |
| `auralis-router` | Health-checks nodes and forwards each request to the healthy node with the fewest requests in flight. Retries the next node when one fails. Exposes `GET /v1/nodes`, `GET /healthz`, `GET /metrics`. |

Routing policy: least in-flight, ties broken by fewest served. A node that fails a request is
marked unhealthy until its next successful `/healthz` check (every 5 s by default). With no
healthy node the router answers `503`.

`auralis-server` handles one request at a time, so the router's balancing is what gives you
parallelism: run more nodes, not more threads.

## Run it

```bash
# 1. a model: copy your own ggml file to models/ggml-base.en-q5_1.bin

# 2. two nodes and a router
docker compose up --build

# 3. use it
curl -X POST "http://127.0.0.1:8780/v1/transcriptions?cleanup=clean" --data-binary @speech.wav
curl http://127.0.0.1:8780/v1/nodes
```

With dashboards:

```bash
export GF_SECURITY_ADMIN_PASSWORD="$(openssl rand -hex 12)"
docker compose --profile observability up --build
# Prometheus http://127.0.0.1:9090   Grafana http://127.0.0.1:3000
```

Without Docker, run the binaries directly:

```bash
cargo run --release -p auralis-runtime --bin auralis-server -- --model models/ggml-base.en-q5_1.bin --port 8787
cargo run --release -p auralis-runtime --bin auralis-router -- --node http://127.0.0.1:8787
```

## Security notes

- Nodes and the router have no authentication. Compose publishes only the router, Prometheus and
  Grafana, and only on `127.0.0.1`. Put a reverse proxy with auth and TLS in front before exposing
  anything to a network.
- Nodes accept WAV uploads up to 50 MB.

## Metrics

Node: `auralis_requests_total`, `auralis_errors_total`, `auralis_audio_seconds_total`,
`auralis_processing_seconds_total`, `auralis_model_info`.

Router: `auralis_router_routed_total`, `auralis_router_rejected_total`,
`auralis_router_node_up`, `auralis_router_node_inflight`, `auralis_router_node_served_total`.

Real-time factor is `processing_seconds / audio_seconds`, as charted in the bundled dashboard.

## Release and deploy

- `ci.yml`: tests, clippy, desktop build, benchmark smoke test, site markup check, and a Docker
  build of both images plus `docker compose config`.
- `release.yml`: a `v*` tag builds the Windows NSIS installer, writes `SHA256SUMS.txt` and
  publishes both to GitHub Releases. Hyphenated tags are pre-releases.
- The site (`apps/web`) deploys through the Vercel Git integration on merge to `main`.

## Not built yet

Node agent and registration, GPU/VRAM-aware scheduling, streaming (WebSocket) transcription,
NATS control plane, Kubernetes/Helm, model registry, mTLS between nodes, a custom-trained model
family, and mobile clients. The PRD describes all of these; none exist in this repository.
