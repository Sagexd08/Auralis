# Contributing to Auralis

Auralis is open source under the [MIT license](LICENSE). Contributions of every size are welcome: bug reports, docs, benchmarks, and code.

## Where things live

| Path | What it is |
| --- | --- |
| `runtime/core/` | `auralis-runtime`: capture, VAD, denoise, STT, text cleanup, pipeline. Also the `auralis-server` HTTP API and `auralis-bench-cli`. |
| `apps/desktop/` | Tauri 2 tray app: hotkeys, keystroke injection, settings, model downloads. |
| `apps/web/` | The static landing page. No build step. |
| `benchmarks/` | Python WER/CER/RTF harness. |

## Set up

You need Rust (stable, `x86_64-pc-windows-msvc` on Windows), Node.js 20+, CMake, and a `libclang` that works with bindgen 0.69. See the "libclang note" in the [README](README.md#building) if `whisper-rs-sys` fails to build.

```powershell
powershell -File models/pull-model.ps1
cargo test -p auralis-runtime -p auralis-desktop
cd apps/desktop
npm install
npm run tauri dev
```

Tests that need a model skip themselves when `models/ggml-base.en-q5_1.bin` is missing.

## Before you open a pull request

- `cargo test -p auralis-runtime -p auralis-desktop` passes.
- `cargo clippy -p auralis-runtime --all-targets -- -D warnings` is clean.
- New behaviour has a test. Pure logic belongs in a function that can be tested without a microphone, a model, or a running app.
- Accuracy or latency changes include numbers from the harness in `benchmarks/`, before and after. See `benchmarks/README.md`.
- Keep changes focused. One concern per pull request.
- The codebase carries no source comments. Prefer clear names and small functions, and put explanations in the pull request description or in `docs/`.

## Good first areas

- Text cleanup rules and modes (`runtime/core/src/text.rs`).
- macOS and Linux keystroke injection and hotkeys.
- Benchmark datasets and reproducible reports.
- Settings UI polish.
- Documentation and translations.

## Reporting bugs

Open an issue with your OS version, the model file in use, what you did, what you expected, and what happened. Run with `RUST_LOG=auralis_runtime=debug,auralis_desktop_lib=debug` for per-stage logs.

For security issues, see [SECURITY.md](SECURITY.md) rather than opening a public issue.

## Releasing (maintainers)

Pushing a tag such as `v0.2.0` runs `.github/workflows/release.yml`, which builds the Windows NSIS installer and attaches it to a GitHub Release. The landing page download button reads the latest release, so no site change is needed.

To sign the installer, add two repository secrets: `WINDOWS_CERTIFICATE` (a base64-encoded `.pfx`) and `WINDOWS_CERTIFICATE_PASSWORD`. Without them the installer is built unsigned and Windows SmartScreen will warn on it.

The workflow installs a pip-provided `libclang` because the LLVM preinstalled on the runner is too new for bindgen 0.69.

## Conduct

Be respectful. See [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).
