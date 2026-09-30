# Downloads the quantized base.en whisper.cpp model used by Auralis Phase 1.
$ErrorActionPreference = "Stop"
$modelDir = Join-Path $PSScriptRoot "."
$modelPath = Join-Path $modelDir "ggml-base.en-q5_1.bin"
$url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en-q5_1.bin"

if (Test-Path $modelPath) {
    Write-Host "Model already present at $modelPath"
    exit 0
}

Write-Host "Downloading base.en (q5_1) model to $modelPath ..."
Invoke-WebRequest -Uri $url -OutFile $modelPath
Write-Host "Done."
