# Downloads a quantized whisper.cpp model for Auralis.
#
#   .\pull-model.ps1            # base.en  (~59 MB)  — fastest, default
#   .\pull-model.ps1 small.en   # small.en (~190 MB) — noticeably better on
#                               #   proper nouns and accented speech, ~3x slower
#
# Any downloaded model shows up in the tray Settings window's model picker.
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
