from __future__ import annotations


def edit_distance(ref: list, hyp: list) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(r.split(), h.split()) for r, h in zip(refs, hyps))
    words = sum(len(r.split()) for r in refs)
    return errors / max(words, 1)


def cer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(list(r.replace(" ", "")), list(h.replace(" ", ""))) for r, h in zip(refs, hyps))
    chars = sum(len(r.replace(" ", "")) for r in refs)
    return errors / max(chars, 1)
