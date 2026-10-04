from __future__ import annotations
import argparse
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import torch
from .checkpoint import load_model
from .model import AuralisSTT


class ExportWrapper(torch.nn.Module):
    def __init__(self, model: AuralisSTT):
        super().__init__()
        self.model = model

    def forward(self, feats: torch.Tensor, lengths: torch.Tensor):
        log_probs, _ = self.model(feats, lengths)
        return log_probs


def export_onnx(model: AuralisSTT, out_path, opset: int = 18, sample_frames: int = 160) -> Path:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model = model.eval().cpu()
    feats = torch.randn(1, model.cfg.n_mels, sample_frames)
    lengths = torch.tensor([sample_frames], dtype=torch.long)
    program = torch.onnx.export(
        ExportWrapper(model),
        (feats, lengths),
        input_names=["features", "lengths"],
        output_names=["log_probs"],
        dynamic_shapes={
            "feats": {0: torch.export.Dim("batch", min=1, max=8), 2: torch.export.Dim("frames", min=16, max=6000)},
            "lengths": {0: torch.export.Dim("batch", min=1, max=8)},
        },
        dynamo=True,
        opset_version=opset,
    )
    program.save(str(out_path))
    simplify_shape_annotations(out_path)
    return out_path


def simplify_shape_annotations(onnx_path) -> None:
    import onnx

    model = onnx.load(str(onnx_path))
    graph = model.graph
    tensors = list(graph.value_info) + list(graph.input) + list(graph.output)
    for t in tensors:
        for d in t.type.tensor_type.shape.dim:
            if d.HasField("dim_param") and not d.dim_param.isidentifier():
                d.ClearField("dim_param")
    for o in graph.output:
        dims = o.type.tensor_type.shape.dim
        if len(dims) == 3:
            dims[1].dim_param = "steps"
    onnx.save(model, str(onnx_path))


def verify_onnx(model: AuralisSTT, onnx_path, frames=(97, 160, 233), tolerance: float = 1e-3) -> float:
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    worst = 0.0
    model = model.eval().cpu()
    for t in frames:
        feats = torch.randn(1, model.cfg.n_mels, t)
        lengths = torch.tensor([t], dtype=torch.long)
        with torch.no_grad():
            expected, _ = model(feats, lengths)
        got = session.run(None, {"features": feats.numpy(), "lengths": lengths.numpy()})[0]
        if got.shape != tuple(expected.shape):
            raise AssertionError(f"shape {got.shape} != {tuple(expected.shape)} at {t} frames")
        worst = max(worst, float(np.abs(got - expected.numpy()).max()))
    if worst > tolerance:
        raise AssertionError(f"onnx differs from torch by {worst:.3e} (> {tolerance:.0e})")
    return worst


def export_checkpoint(checkpoint_dir, out_dir, name: str = "auralis", languages=("en",)) -> Path:
    model, tokenizer = load_model(checkpoint_dir)
    out_dir = Path(out_dir)
    onnx_path = export_onnx(model, out_dir / f"{name}.onnx")
    worst = verify_onnx(model, onnx_path)
    shutil.copyfile(Path(checkpoint_dir) / "tokenizer.json", out_dir / f"{name}.tokenizer.json")
    meta = {
        "name": name,
        "languages": list(languages),
        "n_mels": model.cfg.n_mels,
        "parameters": model.num_parameters(),
        "vocab_size": model.cfg.vocab_size,
        "verified_max_abs_diff": worst,
        "source_checkpoint": Path(checkpoint_dir).name,
    }
    (out_dir / f"{name}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return onnx_path


def main(argv=None):
    ap = argparse.ArgumentParser(prog="auralis-export")
    ap.add_argument("checkpoint")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="auralis")
    ap.add_argument("--languages", default="en")
    a = ap.parse_args(argv)
    path = export_checkpoint(a.checkpoint, a.out_dir, a.name, a.languages.split(","))
    print(path)


if __name__ == "__main__":
    main()
