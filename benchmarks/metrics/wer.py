import re
from dataclasses import dataclass

import jiwer

def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

@dataclass
class VariantMetrics:
    wer: float
    cer: float
    rtf: float
    num_clips: int

def compute_metrics(references: list[str], hypotheses: list[str], total_elapsed_s: float, total_audio_s: float) -> VariantMetrics:
    if len(references) != len(hypotheses):
        raise ValueError(f"references/hypotheses length mismatch: {len(references)} vs {len(hypotheses)}")

    norm_refs = [normalize(r) for r in references]
    norm_hyps = [normalize(h) for h in hypotheses]

    wer = jiwer.wer(norm_refs, norm_hyps)
    cer = jiwer.cer(norm_refs, norm_hyps)
    rtf = total_elapsed_s / total_audio_s if total_audio_s > 0 else float("nan")

    return VariantMetrics(wer=wer, cer=cer, rtf=rtf, num_clips=len(references))
