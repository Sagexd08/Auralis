# Auralis build plan: from dictation app to own models

Status: proposal. Derived from the internal product PRD; this file records only what is planned, not what exists.

## Where things stand

Shipped today: a Windows push-to-talk dictation app on whisper.cpp with an on-device pipeline (capture, VAD, RNNoise, STT, text cleanup, keystroke insertion), a local HTTP API, a router and a benchmark harness. The speech model is **Whisper base.en**, not an Auralis model.

The plan below is the path to replacing it. It is ordered so each milestone is useful on its own and can be verified before the next starts.

## Ground rules

- **From scratch means random initialisation**: own tokenizer, own frontend, own encoder/decoder, own training loop. Whisper, Parakeet and Canary are benchmark references, not starting weights.
- **No third-party weights in Auralis model releases.** Whisper is MIT-licensed, but it stays out of anything published as an Auralis model (including the `models-v1` release). It lives beside the foundry as a baseline that competes with Auralis; Auralis does not depend on it.
- **Licence gate before data**: no dataset enters training without a recorded licence and training-use status. Unknown means rejected.
- **Scoped claims only**: results are stated as "WER X on dataset Y at configuration Z", never as universal superiority.
- **Hardware reality**: the development machine (RTX 3050, 6 GB VRAM, 24 GB RAM) can train and debug tiny/small models and run inference. Base- and Large-class training needs rented or borrowed GPUs; the code must not assume otherwise.
- **Hindi, Bengali, Japanese and English** are the launch languages and are in every evaluation from M1 onward.

## Repository layout (new)

```
models/
  auralis/            published Auralis models only
    auralis-tiny/
    auralis-small/
    auralis-base/
benchmarks/
  baselines/
    whisper/          adapter.py, config.yaml, README.md; weights fetched at evaluation time, never committed
training/
  audio_frontend/   reference DSP: wav, resample, frames, windows, dft, stft, mel, log_mel
  tokenizer/        BPE/Unigram training + tests
  models/           encoder, CTC head, decoder
  train/            loop, checkpoints, configs
data/
  provenance/       manifest schema, licence states, registry
  connectors/       huggingface, kaggle, commonvoice, openslr, local, s3
  normalization/    16 kHz mono PCM, transcript canonicalisation
benchmarks/         existing harness; add Auralis model adapter
```

The foundry is one directed graph, and no pretrained checkpoint enters it:

```
data sources -> Data Factory -> corpus version -> tokenizer -> Auralis frontend
  -> Auralis model -> training -> evaluation -> hard-example mining -> next run
  -> model registry -> Hugging Face
```

Whisper sits beside the graph. The benchmark runs the same held-out data through the same preprocessing contract for Auralis and Whisper and reports WER, CER, latency, RTF, VRAM and per-language score.

Python for research, Rust for the shipped runtime. A research component is only ported to Rust after its numbers are fixed.

## M0: Audio frontend (about 2 weeks)

Build `training/audio_frontend` by hand: WAV read, resample, framing, Hann/Hamming/Blackman windows, DFT, STFT, mel scale, triangular filter bank, log-mel.

- **Tests**: compare numerically with `torch.stft` and `librosa.feature.melspectrogram` on fixed fixtures (tolerance recorded); property tests for Nyquist, window sums, filter-bank coverage; silence, clipping and 8/16/48 kHz inputs.
- **Output**: `WAV -> 80-band log-mel` at 25 ms window, 10 ms hop, 16 kHz, with a plot script.
- **Port**: after the numbers match, port to `runtime/core` and add a Rust-vs-Python parity test.
- **Exit**: Rust output matches the Python reference to a documented numeric tolerance (stated per stage, enforced in CI), not just a spectrogram that looks similar; the frontend runs faster than real time on CPU.

## M1: Data and licence tooling (about 3 weeks, parallel with M0)

Build `data/` so every later training run is reproducible and defensible.

1. **Provenance registry**: JSON schema for dataset and per-sample lineage (source, revision, licence, download date, checksum, `training_allowed`, `commercial_allowed`, `redistribution_allowed`). States: `UNKNOWN`, `REJECTED`, `NONCOMMERCIAL_ONLY`, `TRAINING_ONLY`, `REDISTRIBUTABLE`, `ELIGIBLE`. `UNKNOWN` can never enter the default corpus.
2. **Connectors, in order**: local folder, OpenSLR (LibriSpeech), Common Voice (hi, bn, ja, en), FLEURS, Hugging Face Hub (revision-pinned), Kaggle. Each pull writes its provenance record. Fetching *training data* here is separate from the app, which never downloads models.
3. **Normalisation**: mono, 16 kHz, PCM, canonical transcripts per language (Unicode normalisation; Japanese without whitespace assumptions).
4. **Quality and dedup**: file/transcript hash, near-duplicate text, speaker-level split to prevent leakage, SNR/clipping/duration filters, a 0-1 quality score.
5. **Mixing and shards**: temperature-sampled language mixture from a config; WebDataset shards.
6. **CLI**: `auralis data {search,inspect,pull,verify,normalize,dedupe,score,mix,shard}`.

- **Exit**: one command rebuilds a named corpus from manifests, and a unit test proves an `UNKNOWN`-licence dataset is rejected.

## M2: Tiny STT (about 4 weeks, after M0)

`log-mel -> conv subsampling -> small Transformer/Conformer encoder -> CTC`, with an Auralis-trained tokenizer.

- **Step 1 (pipeline proof)**: train on LibriSpeech clean-100 only. Goal is not quality, it is proving loss decreases, greedy CTC decoding works and WER is computed correctly (implement WER by dynamic programming and cross-check).
- **Step 2 (launch languages)**: expand English, Hindi, Bengali and Japanese with FLEURS and Common Voice; shared multilingual tokenizer with language tokens.
- **Constraints**: model sized for the 6 GB GPU (mixed precision, gradient checkpointing); every run records git commit, seed, dataset revisions and config.
- **Exit**: reproducible WER/CER per language on held-out FLEURS and Common Voice test sets, and a documented comparison against Whisper base on the same sets. A weak result is acceptable; an unmeasured one is not.

## M3 onward (not yet scheduled)

Streaming (causal/chunked encoder, partial stability), Conformer upgrade, denoiser and VAD of our own, quantisation (INT8/INT4), runtime integration so the desktop app can load an Auralis checkpoint, then data scale-up (100 h to 1k to 10k hours of eligible audio), hard-example mining, and TTS research. Compute fabric and mobile come after a model exists that is worth serving. Each is gated on the previous milestone's results.

## Decisions needed before M2 starts

1. **Compute**: where do larger runs happen (rented GPUs, a university cluster, friends' machines)? Budget and a rough hour count.
2. **Data you own**: is there recorded Hindi/Bengali/Japanese speech with consent that can be added?
3. **Commercial intent**: if Auralis may ship commercially, `NONCOMMERCIAL_ONLY` datasets are excluded from the default corpus.
4. **Installer model**: Whisper is not published as an Auralis model, so the installer cannot rely on a `models-v1` release containing it. Until an Auralis model exists, either the installer ships without a model (the user supplies their own) or it bundles a model you explicitly choose. PR #11 and the release workflow need to follow this decision.

## Risks

- **Data volume and quality** will limit accuracy long before architecture does; budget most effort there.
- **Low-resource languages** (Bengali especially) have far less licensed audio than English; expect uneven results and report them per language.
- **A 6 GB GPU** caps experiment size; plan for iteration on small models and rented time for confirmation runs.
- **Scope creep**: TTS, mobile and clusters are explicitly deferred until M2 produces numbers.
