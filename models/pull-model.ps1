param(
    [ValidateSet("base.en", "small.en", "medium.en")]
    [string]$Model = "base.en"
)

$ErrorActionPreference = "Stop"
$fileName = "ggml-$Model-q5_1.bin"
$modelPath = Join-Path $PSScriptRoot $fileName
$url = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/$fileName"

if (Test-Path $modelPath) {
    Write-Host "Model already present at $modelPath"
    exit 0
}

Write-Host "Downloading $Model (q5_1) model to $modelPath ..."
Invoke-WebRequest -Uri $url -OutFile $modelPath
Write-Host "Done. Pick it in Auralis tray > Settings > Model."
