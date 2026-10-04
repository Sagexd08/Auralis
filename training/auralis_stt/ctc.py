from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def greedy_decode(log_probs: torch.Tensor, lengths: torch.Tensor, blank: int = 0) -> list[list[int]]:
    best = log_probs.argmax(dim=-1).cpu().numpy()
    out = []
    for row, n in zip(best, lengths.cpu().tolist()):
        prev, ids = -1, []
        for t in row[:n]:
            if t != prev and t != blank:
                ids.append(int(t))
            prev = t
        out.append(ids)
    return out


def ctc_loss(log_probs: torch.Tensor, targets: torch.Tensor, input_lengths: torch.Tensor, target_lengths: torch.Tensor, blank: int = 0) -> torch.Tensor:
    return F.ctc_loss(log_probs.transpose(0, 1), targets, input_lengths, target_lengths, blank=blank, reduction="mean", zero_infinity=True)


def ctc_neg_log_likelihood(log_probs: np.ndarray, target: list[int], blank: int = 0) -> float:
    ext = [blank]
    for s in target:
        ext += [s, blank]
    T, S = log_probs.shape[0], len(ext)
    neg_inf = -np.inf
    alpha = np.full((T, S), neg_inf)
    alpha[0, 0] = log_probs[0, ext[0]]
    if S > 1:
        alpha[0, 1] = log_probs[0, ext[1]]
    for t in range(1, T):
        for s in range(S):
            terms = [alpha[t - 1, s]]
            if s >= 1:
                terms.append(alpha[t - 1, s - 1])
            if s >= 2 and ext[s] != blank and ext[s] != ext[s - 2]:
                terms.append(alpha[t - 1, s - 2])
            alpha[t, s] = np.logaddexp.reduce(terms) + log_probs[t, ext[s]]
    final = alpha[T - 1, S - 1] if S == 1 else np.logaddexp(alpha[T - 1, S - 1], alpha[T - 1, S - 2])
    return float(-final)
