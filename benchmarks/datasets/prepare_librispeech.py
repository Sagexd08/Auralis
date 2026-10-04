import argparse
import json
import wave
from pathlib import Path

import numpy as np

def write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, required=True)
    parser.add_argument("--count", type=int, default=20, help="number of clips to pull")
    args = parser.parse_args()

    from datasets import load_dataset

    args.dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)

    clips = []
    for i, row in enumerate(ds.take(args.count)):
        clip_id = f"ls{i:04d}"
        audio = row["audio"]
        write_wav(args.dest / f"{clip_id}.wav", np.asarray(audio["array"], dtype=np.float32), audio["sampling_rate"])
        clips.append({"id": clip_id, "audio": f"{clip_id}.wav", "text": row["text"].lower()})

    (args.dest / "manifest.json").write_text(json.dumps({"clips": clips}, indent=2))
    print(f"Wrote {len(clips)} clip(s) to {args.dest}")

if __name__ == "__main__":
    main()
