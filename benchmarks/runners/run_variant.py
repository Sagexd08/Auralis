import json
import subprocess
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class VariantResult:
    model: str
    preprocess: str
    dataset: str
    references: list[str] = field(default_factory=list)
    hypotheses: list[str] = field(default_factory=list)
    total_elapsed_s: float = 0.0
    total_audio_s: float = 0.0

def audio_duration_s(wav_path: Path) -> float:
    with wave.open(str(wav_path), "rb") as wf:
        return wf.getnframes() / float(wf.getframerate())

def run_variant(
    bench_cli: Path,
    model_path: Path,
    dataset_dir: Path,
    preprocess: str,
    model_name: str,
    dataset_name: str,
) -> VariantResult:
    manifest = json.loads((dataset_dir / "manifest.json").read_text())
    result = VariantResult(model=model_name, preprocess=preprocess, dataset=dataset_name)

    for clip in manifest["clips"]:
        audio_path = dataset_dir / clip["audio"]
        result.references.append(clip["text"])
        result.total_audio_s += audio_duration_s(audio_path)

        start = time.monotonic()
        proc = subprocess.run(
            [
                str(bench_cli),
                "--model", str(model_path),
                "--input", str(audio_path),
                "--preprocess", preprocess,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        result.total_elapsed_s += time.monotonic() - start
        result.hypotheses.append(proc.stdout.strip())

    return result
