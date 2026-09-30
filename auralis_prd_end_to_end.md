# Auralis — Open Speech Intelligence & Compute Fabric
## End-to-End Product Requirements Document + Technical Stack

**Working name:** Auralis  
**Category:** Open-source Speech AI / STT / Edge Inference / Self-hosted AI Compute  
**Primary outputs:** Custom STT model family, desktop dictation client, mobile runtime, local/cloud inference runtime, self-hosted GPU cluster, model registry, benchmark suite, observability control plane, public website, Hugging Face release.

---

# 1. Product Thesis

Auralis is not another wrapper around an existing speech API.

It is an end-to-end, deploy-anywhere speech stack:

**microphone → audio intelligence → custom STT → text intelligence → system-wide text insertion**

with a second layer that makes compute portable:

**local machine → local GPU → local cluster → cloud GPU**

The user should be able to press a keyboard shortcut anywhere on their computer, speak naturally, and have polished text appear in the active application.

The same model family should run:

- offline on a phone
- offline on a laptop/desktop
- on a local GPU
- on a self-hosted multi-GPU cluster
- in a public/private cloud

The project should be benchmark-first. The claim is not “best at everything”; the measurable objective is to become state-of-the-art on selected public benchmarks and real-world stress tests while offering an unusually strong accuracy/latency/memory/cost tradeoff.

Hugging Face's current Open ASR Leaderboard evaluates accuracy across multiple datasets and now includes multilingual, long-form, Hindi and Indian-English tracks. That makes reproducible benchmarking a first-class product feature, not an afterthought.

---

# 2. Problem

Current speech systems commonly force a choice between:

1. high accuracy but expensive/cloud-only inference,
2. fast inference but weaker transcription,
3. local inference but large memory/compute requirements,
4. clean transcripts but weak handling of noisy or conversational speech,
5. model access without a polished system-wide input experience,
6. powerful inference without a simple self-hosting path.

Auralis aims to combine these into one open stack.

---

# 3. Target Users

### Primary

- ML engineers
- AI researchers
- developers building voice products
- open-source contributors
- privacy-conscious users
- developers who want system-wide dictation
- students/researchers building speech systems
- teams that need self-hosted speech inference

### Secondary

- call-center/meeting tooling developers
- accessibility applications
- local AI enthusiasts
- edge-device developers
- organizations with sensitive audio

---

# 4. Product Objectives

## P0 objectives

### O1 — Competitive STT

Build a custom STT model family that can compete with strong open models such as Whisper variants, NVIDIA Parakeet and Canary.

### O2 — Real-time dictation

Create a WhisperFlow-style system-wide keyboard experience:

**hotkey → speak → streaming transcript → final polished text → active application**

### O3 — Local-first inference

Auralis should work without the internet where the target model/hardware permits it.

### O4 — Portable deployment

The same model should be deployable on:

- CPU
- consumer GPU
- cloud GPU
- mobile hardware

### O5 — Self-hosted compute fabric

Build an Auralis node agent + scheduler so users can turn their own machines into an inference cluster.

### O6 — Reproducible benchmark

Publish model weights, benchmark scripts, model cards, inference code and documented evaluation methodology on Hugging Face/GitHub.

---

# 5. Non-Goals

Do not attempt in v1 to:

- recreate AWS,
- build a general-purpose Kubernetes replacement,
- support every mobile chipset,
- train a multi-billion-parameter foundation model from zero immediately,
- claim universal SOTA without evidence,
- build a full general-purpose LLM platform,
- build a complete Grafana clone.

The infrastructure should remain speech-specialized.

---

# 6. Core Product Experience

## Desktop

Global hotkey:

`Ctrl/Cmd + Space`

Modes:

### Push-to-talk

Hold hotkey → speak → release → final text inserted.

### Toggle

Press once → speak → press again → finalize.

### Continuous

Automatic VAD starts/stops speech capture.

### Correction

Example:

User:
> “Send the report to Rahul tomorrow.”

Then:
> “Actually, change Rahul to Rohan.”

The text layer recognizes the second utterance as a correction/edit rather than blindly appending it.

---

# 7. System Architecture

```text
                           AURALIS
                              │
          ┌───────────────────┼───────────────────┐
          │                   │                   │
       MODEL                 AUDIO             RUNTIME
      RESEARCH              INTELLIGENCE       PLATFORM
          │                   │                   │
   ┌──────┼──────┐      ┌─────┼─────┐      ┌──────┼──────┐
   │      │      │      │     │     │      │      │      │
  STT   VAD   Text    Denoise AEC  VAD   Local  Edge   Cloud
 Model        Layer                       CPU/GPU Mobile
   │                                      
   └───────────────────┬───────────────────────┘
                       │
                 PIPELINE RUNTIME
                       │
            ┌──────────┴──────────┐
            │                     │
       Streaming Mode         Offline Mode
            │                     │
       low latency             max accuracy
            │                     │
            └──────────┬──────────┘
                       │
                 COMPUTE ROUTER
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
       Laptop       Local GPU    Cloud GPU
          │            │            │
          └────────────┼────────────┘
                       │
                 CONTROL PLANE
                       │
       ┌───────────────┼────────────────┐
       │               │                │
   Scheduler       Model Registry    Observability
       │               │                │
   Node Agent       HF + S3        Metrics/Traces/Logs
```

---

# 8. STT Model Strategy

## 8.1 Model family

Create three target variants:

### Auralis Nano

Target:
- edge/mobile
- low memory
- INT8/INT4
- offline
- low power

### Auralis Base

Target:
- laptops/desktops
- consumer GPUs
- good accuracy/latency balance

### Auralis Large

Target:
- cloud/local high-end GPUs
- maximum accuracy
- long-form/offline transcription

The exact parameter counts should be decided after the first benchmark cycle rather than arbitrarily fixed.

---

# 9. Model Architecture

Use an experimentation-driven architecture rather than cloning Whisper.

## Candidate A — Streaming model

**Conformer/FastConformer encoder + RNNT/TDT/CTC-style decoding**

Purpose:
- streaming
- low latency
- efficient inference

## Candidate B — High-accuracy model

**Conformer/Transformer audio encoder + lightweight language-aware decoder**

Purpose:
- maximum accuracy
- long-form speech
- contextual decoding

Current open-ASR research suggests a tradeoff: CTC/TDT-style decoders are attractive for speed, while Conformer encoders paired with stronger language-model decoders can perform very well on accuracy-focused tasks.

## Candidate C — Hybrid

The recommended product path is:

```text
audio
  ↓
shared encoder
  ├──────────────→ streaming RNNT/TDT head
  │
  └──────────────→ high-accuracy decoder
```

This lets one model family serve both interactive dictation and offline transcription.

---

# 10. Whisper-Inspired Feature Set

Borrow the *capabilities*, not the implementation.

Auralis should support:

- multilingual ASR
- automatic language identification
- timestamps
- segment timestamps
- optional word timestamps
- punctuation
- capitalization
- long-form transcription
- speech translation as a future capability
- task conditioning
- robust noisy-audio handling

Whisper is a useful conceptual benchmark because it is a general-purpose multilingual speech model with transcription, translation and language-identification capabilities.

---

# 11. Audio Intelligence Layer

This should be a separate subsystem.

```text
Audio
 ↓
Sample-rate normalization
 ↓
Channel normalization
 ↓
VAD
 ↓
SNR estimation
 ↓
Noise classifier
 ↓
Denoiser
 ↓
Dereverberation
 ↓
Echo cancellation
 ↓
STT
```

## Components

### VAD

Detect:
- speech
- silence
- music
- background noise

### Denoiser

Handle:
- fans
- traffic
- keyboard noise
- room noise
- café noise
- air conditioners
- microphone hiss

### Dereverberation

Improve:
- rooms
- conference rooms
- far-field microphones

### Echo cancellation

For:
- calls
- speaker playback
- meetings

### Audio quality analyzer

Output:

```json
{
  "snr_db": 17.4,
  "speech_probability": 0.96,
  "clipping": false,
  "reverb_score": 0.22,
  "noise_class": "room"
}
```

The runtime can dynamically select preprocessing strength.

---

# 12. Text Intelligence Layer

Raw ASR should not be the only output.

Pipeline:

```text
Raw transcript
      ↓
Boundary detection
      ↓
Punctuation/capitalization
      ↓
Text normalization
      ↓
Number/date/currency normalization
      ↓
Grammar cleanup
      ↓
Context-aware correction
      ↓
Final transcript
```

Examples:

Raw:

> “uh we need to deploy this to prod tomorrow”

Final:

> “We need to deploy this to production tomorrow.”

This layer must have an explicit **raw mode** so users can disable rewriting.

---

# 13. Context-Aware Dictation

Optional local context sources:

- active application
- document title
- selected text
- coding language
- clipboard context
- user dictionary
- custom vocabulary

Examples:

In VS Code:
> “create a postgres migration”

The system should preserve developer terminology.

In Gmail:
> “best regards soh...”

The output should be natural prose.

Do not silently upload application contents. Context sharing must be explicit and configurable.

---

# 14. Desktop Application

## Recommended stack

**Tauri 2 + Rust core**

Why:
- low memory footprint
- native system integration
- Rust is useful for audio/runtime/networking
- can share core components across platforms

### Core modules

```text
auralis-desktop/
├── hotkey
├── microphone
├── audio-buffer
├── vad
├── denoiser
├── inference-client
├── local-runtime
├── cloud-router
├── text-processor
├── keyboard-injector
├── clipboard
├── settings
├── telemetry
└── updater
```

## Keyboard injection

Use platform-native APIs:

- Windows: native input APIs
- macOS: Accessibility/input APIs
- Linux: X11/Wayland-compatible strategy with permission-aware fallbacks

Provide clipboard insertion as a universal fallback.

---

# 15. Local Inference Runtime

Create an `auralis-runtime` independent from the UI.

```text
auralis-runtime
├── model-loader
├── scheduler
├── audio-pipeline
├── tokenizer
├── decoder
├── quantization
├── memory-manager
├── device-discovery
└── streaming-engine
```

Supported execution backends:

### CPU

ONNX Runtime / native CPU kernels

### NVIDIA GPU

TensorRT / CUDA

### General desktop

ONNX Runtime

### Apple

Core ML where appropriate

### Android/iOS

ONNX Runtime Mobile and platform-specific execution paths

ONNX Runtime officially supports iOS and Android deployment, while Apple Core ML can leverage CPU, GPU and Neural Engine resources for on-device inference.

---

# 16. Cloud / Self-Hosted Compute Fabric

This should NOT be a general-purpose cloud.

It is a speech inference fabric.

## Node

Every compute machine runs:

`auralis-agent`

The agent reports:

```json
{
  "node_id": "gpu-01",
  "cpu": "AMD Ryzen",
  "ram_gb": 64,
  "gpu": "RTX 4090",
  "vram_gb": 24,
  "driver": "xxx",
  "models": ["auralis-base"],
  "latency_ms": 42,
  "status": "healthy"
}
```

## Scheduler

Inputs:

```text
request:
model = base
mode = streaming
latency_target = 150ms
privacy = local_preferred
```

Scheduler chooses a node.

Decision dimensions:

- latency
- GPU availability
- VRAM
- queue length
- model locality
- power/battery
- estimated cost
- privacy policy
- network latency

---

# 17. Deployment Modes

## Local-only

```text
Desktop → Local Runtime → Local Model
```

## Hybrid

```text
Desktop
  ↓
Local VAD/Denoise
  ↓
Router
  ├── local GPU
  └── cloud fallback
```

## Self-hosted cluster

```text
                    Control Plane
                         │
        ┌────────────────┼────────────────┐
        │                │                │
     GPU Node 1      GPU Node 2      GPU Node 3
        │                │                │
     RTX 4090         A100             H100
```

## Public cloud

Run identical containers in Kubernetes.

---

# 18. Kubernetes Layer

Use:

- Kubernetes for production cluster orchestration
- K3s for lightweight self-hosted/local clusters
- NVIDIA device plugin for GPU resources
- containerd
- Helm
- Argo CD later for GitOps

Kubernetes has stable support for scheduling NVIDIA/AMD GPUs via device plugins.

The Auralis scheduler remains application-specific; Kubernetes handles infrastructure placement while Auralis decides speech workload policy.

---

# 19. Inference Serving

Use **NVIDIA Triton Inference Server** for GPU-serving where it helps.

Triton supports:

- PyTorch
- ONNX
- TensorRT
- concurrent execution
- dynamic batching
- ensembles
- HTTP
- gRPC
- streaming workloads
- GPU/latency/throughput metrics

Architecture:

```text
Auralis Gateway
      ↓
Auralis Scheduler
      ↓
Triton
 ├── Auralis Nano
 ├── Auralis Base
 ├── Auralis Large
 ├── Denoiser
 └── Text Processor
```

Use Triton ensembles only where the pipeline benefits from server-side composition.

---

# 20. GPU Optimization

Training:

- PyTorch
- CUDA
- BF16
- FSDP/DistributedDataParallel where needed
- gradient checkpointing
- mixed precision
- experiment tracking

Inference:

- ONNX
- TensorRT
- FP16/BF16
- INT8
- INT4 where quality allows
- CUDA Graphs where beneficial
- kernel fusion
- batching for offline workloads
- no batching or micro-batching for interactive streams

TensorRT converts trained models into optimized GPU-specific engines and supports modern reduced-precision modes.

Benchmark every optimization. Never assume a theoretical speedup is an actual speedup.

---

# 21. Messaging / Control Plane

Use **NATS + JetStream**.

Subjects:

```text
speech.request
speech.assigned
speech.started
speech.partial
speech.completed
node.heartbeat
node.health
model.loaded
model.unloaded
benchmark.started
benchmark.completed
```

JetStream provides persistence/replay and at-least-once delivery semantics.

Use gRPC for direct inference and NATS for control-plane events.

---

# 22. Storage

## PostgreSQL

Store:

- users
- devices
- nodes
- deployments
- models
- model versions
- benchmark runs
- API keys
- routing policies
- usage

## Redis

Use for:

- ephemeral session state
- rate limits
- short-lived routing data
- caching

## S3-compatible object storage

Use MinIO for self-hosted deployments.

Buckets:

```text
models/
datasets/
audio-temp/
benchmark-results/
artifacts/
logs/
```

Do not retain user audio by default.

---

# 23. Observability

Do not rebuild Grafana.

Build an Auralis control dashboard on top of standard telemetry.

Use:

- OpenTelemetry
- Prometheus
- Grafana
- Loki
- Tempo

OpenTelemetry provides vendor-neutral telemetry for traces, metrics and logs.

Dashboard sections:

### Fleet

```text
Nodes: 5
Healthy: 4
Busy: 3
Offline: 1
GPU VRAM: 71%
```

### Inference

```text
Requests/min
p50 latency
p95 latency
p99 latency
RTF
tokens/sec
audio-sec/sec
queue depth
```

### Model

```text
Model version
GPU memory
load time
requests
error rate
```

### Audio

```text
mean SNR
noise classes
VAD ratios
denoiser usage
```

### Cost

```text
GPU-hours
audio minutes
cost/minute
local vs cloud percentage
```

---

# 24. Benchmark System

This is one of the most important modules.

Repository:

```text
benchmarks/
├── datasets
├── adapters
├── metrics
├── runners
├── reports
└── leaderboard
```

Evaluate against strong current baselines rather than random Hub models.

Baseline family:

- OpenAI Whisper variants
- NVIDIA Parakeet variants
- NVIDIA Canary
- other leading open ASR models available at evaluation time

Hugging Face's Open ASR Leaderboard publishes reproducible evaluation code and evaluates across diverse datasets. It should be used as a reference for evaluation design.

---

# 25. Metrics

## Accuracy

- WER
- CER
- normalized WER
- punctuation-aware WER
- number accuracy
- proper-noun accuracy
- timestamp error

## Streaming

- first-partial latency
- final latency
- endpointing latency
- streaming RTF

## Efficiency

- VRAM
- RAM
- model size
- CPU %
- GPU %
- energy
- audio seconds processed per second
- requests per GPU

## Product quality

- insertion latency
- correction rate
- hallucination rate
- dropped audio %
- transcription stability
- partial transcript churn

---

# 26. Benchmark Tracks

### Track A — Clean speech

Standard English datasets.

### Track B — Conversational

Meetings, conversations, spontaneous speech.

### Track C — Long-form

30+ minute recordings.

### Track D — Noise

Traffic, café, fan, keyboard, room noise.

### Track E — Streaming

Latency-focused evaluation.

### Track F — Multilingual

Multiple languages.

### Track G — Code-switching

Mixed-language speech.

### Track H — Edge

CPU/mobile performance.

### Track I — Efficiency

Accuracy vs memory vs throughput.

---

# 27. SOTA Definition

Never write:

> “Best STT model in the world”

unless an independent, reproducible evaluation establishes it.

Use explicit claims:

> “Best WER on Benchmark X.”

> “Lowest latency among models tested under Y hardware configuration.”

> “Highest accuracy/VRAM efficiency on our evaluation suite.”

The model card should include the exact command/configuration used to produce the result.

---

# 28. Data Pipeline

```text
Raw Dataset
    ↓
License/provenance checker
    ↓
Audio validation
    ↓
Resampling
    ↓
Silence/VAD filtering
    ↓
Language ID
    ↓
Transcript normalization
    ↓
Duplicate detection
    ↓
Quality scoring
    ↓
Train/Validation/Test split
    ↓
Augmentation
    ↓
Training shards
```

Track:

- dataset source
- license
- speaker metadata where legally available
- language
- duration
- sample rate
- quality
- augmentation history

Never include data that cannot legally be redistributed in the public training package.

---

# 29. Training Strategy

## Phase 1

Fine-tune a strong open encoder/model to validate the data + pipeline.

## Phase 2

Train your own architecture.

## Phase 3

Hard-example mining:

```text
model predictions
      ↓
identify difficult samples
      ↓
classify error
      ↓
reweight/retrain
```

Error categories:

- names
- numbers
- accents
- code-switching
- noise
- fast speech
- overlapping speakers

## Phase 4

Distillation:

Teacher:
large model

Student:
small model

Objective:

```text
student accuracy ≈ teacher
student compute << teacher
```

## Phase 5

Quantization.

## Phase 6

Hardware-specific optimization.

---

# 30. Speech Enhancement Training

Train/evaluate denoising separately.

Dataset generation:

```text
clean speech
   +
noise library
   +
reverb simulation
   +
room impulse response
   ↓
synthetic noisy speech
```

Measure whether denoising actually improves downstream WER.

A denoiser that produces better-looking audio but worse STT is a failed component.

---

# 31. Model Hub Structure

Recommended Hugging Face organization:

```text
auralis-ai/
├── auralis-nano
├── auralis-base
├── auralis-large
├── auralis-vad
├── auralis-denoiser
├── auralis-normalizer
└── auralis-benchmarks
```

Model cards should include:

- intended use
- architecture
- datasets
- licensing
- hardware tested
- WER/CER
- RTF
- known failure cases
- limitations
- safety/privacy notes
- reproducibility commands

---

# 32. Public API

## REST

```http
POST /v1/transcriptions
```

Request:

```json
{
  "model": "auralis-base",
  "language": "auto",
  "timestamps": true,
  "stream": true
}
```

## WebSocket

```text
/ws/transcribe
```

Events:

```json
{
  "type": "partial",
  "text": "we need to"
}
```

```json
{
  "type": "final",
  "text": "We need to deploy this tomorrow."
}
```

## Python SDK

```python
from auralis import Client

client = Client("http://localhost:8000")

result = client.transcribe(
    "meeting.wav",
    model="auralis-base"
)

print(result.text)
```

## CLI

```bash
auralis transcribe meeting.wav
auralis listen
auralis benchmark
auralis node start
auralis model pull auralis-base
auralis deploy
```

---

# 33. Website Product Requirements

## Visual direction

Premium technical infrastructure.

Avoid:
- crypto aesthetics
- excessive gradients
- fake 3D
- generic AI stock imagery
- giant marketing walls of text

Use:

- dark/neutral background
- dense but readable technical UI
- monospace data elements
- subtle animated signal/audio visualization
- benchmark charts
- live inference demo
- architectural diagrams
- strong typography

---

# 34. Website Pages

## `/`

Hero:

> **Speech infrastructure you can run anywhere.**

Subhead:

> Auralis is an open speech stack built for real-time transcription, edge inference and self-hosted AI compute.

CTAs:

**Try the demo**  
**View on Hugging Face**  
**GitHub**

Hero visual:

Live waveform → denoiser → STT → text.

---

## `/playground`

Interactive demo:

- microphone input
- upload audio
- streaming transcription
- model selector
- denoiser toggle
- timestamps
- latency
- RTF
- confidence
- transcript comparison

Show:

```text
Auralis
Whisper
Parakeet
```

side-by-side where legally and technically practical.

---

## `/benchmarks`

Public leaderboard.

Filters:

- model
- dataset
- language
- latency
- VRAM
- device

Charts:

### Accuracy vs speed

### Accuracy vs memory

### Throughput vs cost

### Streaming latency

Every result links to reproducibility details.

---

## `/models`

Cards:

```text
Auralis Nano
Edge / Mobile
Fast

Auralis Base
Desktop
Balanced

Auralis Large
Cloud
Maximum accuracy
```

Each model includes:

- params
- disk size
- VRAM
- supported runtimes
- languages
- benchmark results

---

## `/cloud`

Explain the self-hosted compute fabric.

Visual:

```text
YOUR MACHINES
     ↓
AURALIS AGENT
     ↓
CONTROL PLANE
     ↓
GPU SCHEDULER
     ↓
INFERENCE
```

CTA:

> Turn your machines into an AI speech cluster.

---

## `/desktop`

Show:

- global hotkey
- system-wide dictation
- local mode
- cloud mode
- hybrid mode
- context-aware formatting

Download buttons.

---

## `/mobile`

Show:

- offline transcription
- low-power mode
- on-device processing
- model download
- privacy

---

## `/docs`

Sections:

```text
Getting Started
Install
Desktop
CLI
Python SDK
API
Streaming
Local inference
Self-hosting
GPU cluster
Models
Benchmarks
Training
Contributing
```

---

## `/architecture`

Interactive system diagram.

Click components:

- STT
- denoiser
- runtime
- scheduler
- Triton
- storage
- observability

---

## `/research`

Research papers/technical reports.

Every release gets:

```text
Model
Dataset
Training
Results
Ablations
Failure analysis
```

---

# 35. Website Technical Stack

## Frontend

- Next.js
- TypeScript
- Tailwind CSS
- shadcn/ui
- Framer Motion only where meaningful
- Web Audio API for demo visualizations

## Charts

- Apache ECharts or Recharts

## Docs

- MDX
- Next.js-based docs system

## Hosting

- Cloudflare for DNS/CDN/WAF
- Vercel for initial website deployment

The inference infrastructure remains separate from the marketing site.

---

# 36. Backend Stack

### API gateway

Rust or Go.

Recommendation:

**Go for control-plane APIs**  
**Rust for audio/runtime/client-side infrastructure**

### ML services

Python:

- PyTorch
- torchaudio
- Hugging Face Transformers where appropriate
- NeMo where useful for baseline comparisons
- custom training framework

### Database

PostgreSQL.

### Queue / events

NATS JetStream.

### Cache

Redis.

### Storage

MinIO/S3.

### Inference

Triton + TensorRT + ONNX Runtime.

### Observability

OpenTelemetry + Prometheus + Grafana + Loki + Tempo.

---

# 37. Monorepo

```text
auralis/
├── apps/
│   ├── web/
│   ├── desktop/
│   ├── mobile/
│   └── playground/
│
├── models/
│   ├── auralis-stt/
│   ├── auralis-vad/
│   ├── auralis-denoiser/
│   └── auralis-text/
│
├── runtime/
│   ├── core/
│   ├── onnx/
│   ├── tensorrt/
│   ├── cpu/
│   └── mobile/
│
├── cloud/
│   ├── api/
│   ├── scheduler/
│   ├── node-agent/
│   ├── gateway/
│   └── registry/
│
├── infra/
│   ├── docker/
│   ├── k8s/
│   ├── helm/
│   ├── terraform/
│   └── observability/
│
├── benchmarks/
├── datasets/
├── training/
├── docs/
├── scripts/
└── README.md
```

---

# 38. API Architecture

```text
Desktop/Mobile/SDK
        ↓
   API Gateway
        ↓
 Authentication
        ↓
 Session Service
        ↓
 Speech Router
        ↓
 Scheduler
        ↓
 Node Agent / Triton
        ↓
 Model
        ↓
 Postprocessing
        ↓
 Response
```

Streaming:

```text
client
  ⇅ WebSocket/gRPC
gateway
  ⇅
stream session
  ⇅
inference node
```

---

# 39. Security / Privacy

Default:

**audio is processed locally where possible.**

Cloud mode must be explicit.

Requirements:

- TLS everywhere
- mTLS between nodes
- signed model manifests
- API keys
- short-lived tokens
- encrypted object storage
- no persistent raw audio by default
- configurable retention
- audit logs
- per-node authorization
- model checksum verification

For the desktop client:

- clear microphone indicator
- explicit permissions
- privacy mode
- cloud/local indicator

---

# 40. Failure Handling

### No GPU

Fallback to CPU Nano.

### Local model missing

Download from model registry if allowed.

### Cloud unreachable

Remain local.

### Model OOM

Scheduler routes to smaller model/node.

### Poor audio

Return quality signal and optionally reprocess.

### Network slow

Switch to local inference if policy permits.

---

# 41. Intelligent Model Routing

Routing policy:

```text
if offline:
    local_best_available()

elif privacy == local_only:
    local_best_available()

elif latency_target < threshold:
    local_streaming_model()

elif accuracy == maximum:
    route_to_large_model()

elif local_gpu_available:
    route_local()

else:
    route_cloud()
```

Later optimize routing with learned policies.

---

# 42. Model Versioning

Every model has:

```text
model name
architecture version
weights version
tokenizer version
runtime version
benchmark version
```

Example:

`auralis-base-v0.3.2`

A production deployment pins all versions.

---

# 43. CI/CD

GitHub Actions:

```text
push
 ↓
lint
 ↓
unit tests
 ↓
integration tests
 ↓
audio pipeline tests
 ↓
model smoke test
 ↓
benchmark smoke test
 ↓
build Docker
 ↓
push image
 ↓
deploy staging
 ↓
health checks
 ↓
production
```

Model CI should run a small deterministic test set for regression detection.

---

# 44. Testing

## Unit

- tokenizer
- VAD
- decoder
- scheduler
- node agent

## Audio

- sample rates
- stereo/mono
- clipped audio
- silence
- noisy audio

## Model

- known transcript fixtures
- regression benchmark

## System

- node failure
- GPU failure
- model download failure
- network partition
- fallback routing

## Desktop

- global shortcut
- permissions
- text insertion
- sleep/wake
- microphone changes

---

# 45. Development Plan

## Phase 0 — Research / baseline

Deliver:

- benchmark harness
- Whisper baseline
- Parakeet baseline
- Canary baseline
- 3–5 public datasets
- local inference prototype

Success:
reproducible numbers.

## Phase 1 — Custom model

Deliver:

- Auralis prototype architecture
- training pipeline
- first checkpoints
- benchmark regression

Success:
competitive accuracy on selected track.

## Phase 2 — Audio intelligence

Deliver:

- VAD
- denoiser
- audio quality estimator

Success:
measurable improvement on noisy-speech benchmark.

## Phase 3 — Streaming

Deliver:

- streaming decoder
- partial transcripts
- endpointing
- low-latency runtime

Success:
interactive dictation feels immediate.

## Phase 4 — Desktop

Deliver:

- Tauri app
- global hotkey
- system-wide insertion
- local inference

Success:
use it as daily dictation software.

## Phase 5 — Compute fabric

Deliver:

- node agent
- scheduler
- local GPU routing
- cloud fallback

Success:
two machines can share inference workloads.

## Phase 6 — Observability

Deliver:

- metrics
- traces
- logs
- GPU dashboard
- benchmark dashboard

Success:
see every inference path end-to-end.

## Phase 7 — Mobile

Deliver:

- Android
- iOS
- quantized Nano model

Success:
offline transcription on supported devices.

## Phase 8 — Public release

Deliver:

- Hugging Face models
- benchmark report
- GitHub
- docs
- website
- demo
- model cards
- reproducible commands

---

# 46. MVP Definition

MVP is NOT the whole cloud.

The first convincing release is:

```text
Auralis Base
+
VAD
+
Denoiser
+
Streaming
+
Desktop hotkey
+
Local GPU/CPU
+
Hugging Face
+
Benchmark dashboard
```

Then add the distributed cloud.

This prevents infrastructure from hiding a weak model.

---

# 47. Launch Criteria

Auralis v1 is release-ready when:

### Model

- reproducible public evaluation
- no undisclosed benchmark cherry-picking
- competitive WER/CER
- strong noisy-speech behavior
- stable long-form behavior

### Runtime

- local CPU fallback
- NVIDIA GPU acceleration
- streaming
- memory limits

### Desktop

- reliable hotkey
- text insertion
- permissions
- local/cloud toggle

### Infra

- Docker Compose
- one-command local deployment
- basic Kubernetes deployment
- node registration
- model routing

### Website

- live demo
- benchmark page
- docs
- model links
- GitHub
- download/install instructions

---

# 48. First Repository Release

Tag:

`v0.1.0`

Repository headline:

> **Auralis — Open Speech Intelligence, Built to Run Anywhere.**

README sections:

1. Demo
2. What it is
3. Architecture
4. Quickstart
5. Models
6. Benchmarks
7. Desktop
8. Self-hosting
9. Training
10. Roadmap

---

# 49. One-command Developer Experience

Local:

```bash
git clone https://github.com/<you>/auralis
cd auralis
./scripts/install.sh
auralis doctor
auralis pull auralis-nano
auralis listen
```

Self-host:

```bash
docker compose up -d
auralis node register
auralis model pull auralis-base
auralis cluster status
```

Cloud:

```bash
auralis deploy \
  --model auralis-large \
  --gpu auto
```

The UX should hide infrastructure complexity while exposing it to advanced users.

---

# 50. Recommended Initial Technology Stack

| Layer | Choice |
|---|---|
| Model training | PyTorch + CUDA |
| Audio | torchaudio + custom Rust audio core |
| STT architecture | Conformer/FastConformer experiments + RNNT/TDT + high-accuracy decoder track |
| VAD | custom/evaluated VAD component |
| Denoising | custom neural enhancement |
| Desktop | Tauri 2 + Rust |
| Mobile | native Swift/Kotlin + shared Rust runtime where useful |
| Local inference | ONNX Runtime |
| NVIDIA acceleration | TensorRT |
| Production serving | Triton |
| API | Go |
| Runtime | Rust |
| Messaging | NATS JetStream |
| DB | PostgreSQL |
| Cache | Redis |
| Object store | MinIO/S3 |
| Containers | Docker |
| Orchestration | K3s/Kubernetes |
| GPU scheduling | Kubernetes device plugins + Auralis scheduler |
| Metrics | Prometheus |
| Telemetry | OpenTelemetry |
| Dashboards | Grafana + custom Auralis UI |
| Logs | Loki |
| Traces | Tempo |
| CI/CD | GitHub Actions |
| Registry | GHCR + Hugging Face |
| Website | Next.js + TypeScript |
| Styling | Tailwind + shadcn/ui |
| Website hosting | Vercel + Cloudflare |
| Docs | Next.js/MDX |

---

# 51. Architecture Principle

The project should have one hard rule:

**Do not couple the model to the product.**

The model should be usable independently.

The desktop app should be usable with another model.

The runtime should support third-party models.

The scheduler should route any supported speech model.

That means the project becomes an **open speech infrastructure layer**, not just a proprietary application.

---

# 52. The Final Portfolio Story

The finished portfolio project should demonstrate five disciplines simultaneously:

### ML research

You designed, trained, evaluated and improved a real speech model.

### Audio engineering

You built VAD, denoising, enhancement and robust preprocessing.

### Systems engineering

You created a portable inference runtime.

### Distributed systems

You created node agents, scheduling, routing and cloud deployment.

### Product engineering

You shipped desktop/mobile voice input and a polished developer website.

The strongest final description is:

> **Auralis is an open, deploy-anywhere speech intelligence stack combining a custom STT model family, neural audio enhancement, real-time system-wide dictation, edge runtimes, GPU-aware scheduling, self-hosted inference clusters, observability, and reproducible public benchmarks.**

That is the project.

---

# 53. Critical Constraint

The hardest part is the model.

Do NOT spend the first month building:
- Kubernetes
- dashboards
- mobile apps
- fancy website
- cloud billing

while the transcription model remains mediocre.

Build the vertical slice first:

```text
MIC
 ↓
VAD
 ↓
DENOISER
 ↓
YOUR STT
 ↓
TEXT CLEANUP
 ↓
KEYBOARD INSERTION
```

Run it every day.

Benchmark it against strong baselines.

Only once that loop is genuinely good should the compute fabric become the next major layer.

---

# 54. Definition of “Crazy Project”

The finished system should let a user do this:

```text
1. Install Auralis.
2. Press Ctrl/Cmd + Space.
3. Speak.
4. Get live text in any app.
5. Turn off the network.
6. Keep transcribing locally.
7. Plug in a second GPU machine.
8. Register it as an Auralis node.
9. Let Auralis route workloads automatically.
10. Open the dashboard and see every node/model/request.
11. Deploy the same model to a cloud GPU.
12. Download the model from Hugging Face.
13. Reproduce the benchmark.
14. Run the complete stack on their own hardware.
```

That is the north-star experience.

