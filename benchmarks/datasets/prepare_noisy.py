import argparse
import json
import wave
from pathlib import Path

import numpy as np

def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sample_rate

def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

def mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_power = np.mean(signal**2) + 1e-12
    noise_power = np.mean(noise**2) + 1e-12
    target_noise_power = signal_power / (10.0 ** (snr_db / 10.0))
    scale = np.sqrt(target_noise_power / noise_power)
    return signal + noise * scale

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True, help="clean dataset dir (with manifest.json)")
    parser.add_argument("--dest", type=Path, required=True, help="output dir for the noisy dataset")
    parser.add_argument("--snr-db", type=float, default=5.0, help="target signal-to-noise ratio in dB")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest_path = args.src / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    args.dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for clip in manifest["clips"]:
        signal, sample_rate = read_wav(args.src / clip["audio"])
        noise = rng.standard_normal(len(signal)).astype(np.float32)
        noisy = mix_at_snr(signal, noise, args.snr_db)
        write_wav(args.dest / clip["audio"], noisy, sample_rate)

    (args.dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest['clips'])} noisy clip(s) to {args.dest} at {args.snr_db} dB SNR")

if __name__ == "__main__":
    main()
