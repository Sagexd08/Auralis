from __future__ import annotations
import json
import unicodedata
from collections import Counter
from pathlib import Path

BLANK = "<blank>"
UNK = "<unk>"
LANG_TOKENS = ["<LANG_EN>", "<LANG_HI>", "<LANG_BN>", "<LANG_JA>"]
SPECIALS = [BLANK, UNK] + LANG_TOKENS
WORD_MARK = "▁"
VERSION = 1


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def pretokens(text: str) -> list[str]:
    return [WORD_MARK + w for w in normalize(text).split(" ") if w]


class Tokenizer:
    def __init__(self, vocab: list[str], merges: list[tuple[str, str]]):
        self.vocab = vocab
        self.merges = [tuple(m) for m in merges]
        self.rank = {m: i for i, m in enumerate(self.merges)}
        self.index = {t: i for i, t in enumerate(vocab)}

    @property
    def blank_id(self) -> int:
        return self.index[BLANK]

    @property
    def unk_id(self) -> int:
        return self.index[UNK]

    def __len__(self) -> int:
        return len(self.vocab)

    @classmethod
    def train(cls, texts, vocab_size: int) -> "Tokenizer":
        word_freq = Counter()
        for t in texts:
            word_freq.update(pretokens(t))
        chars = sorted({c for w in word_freq for c in w})
        vocab = list(SPECIALS) + chars
        words = {w: list(w) for w in word_freq}
        merges: list[tuple[str, str]] = []
        while len(vocab) < vocab_size:
            pairs = Counter()
            for w, syms in words.items():
                f = word_freq[w]
                for a, b in zip(syms, syms[1:]):
                    pairs[(a, b)] += f
            if not pairs:
                break
            best = min(pairs, key=lambda p: (-pairs[p], p))
            merged = best[0] + best[1]
            merges.append(best)
            vocab.append(merged)
            for w, syms in words.items():
                i, out = 0, []
                while i < len(syms):
                    if i < len(syms) - 1 and (syms[i], syms[i + 1]) == best:
                        out.append(merged)
                        i += 2
                    else:
                        out.append(syms[i])
                        i += 1
                words[w] = out
        return cls(vocab, merges)

    def _encode_word(self, word: str) -> list[str]:
        syms = list(word)
        while len(syms) > 1:
            best, best_rank = None, None
            for i in range(len(syms) - 1):
                r = self.rank.get((syms[i], syms[i + 1]))
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = i, r
            if best is None:
                break
            syms[best:best + 2] = [syms[best] + syms[best + 1]]
        return syms

    def encode(self, text: str) -> list[int]:
        ids = []
        for w in pretokens(text):
            for s in self._encode_word(w):
                ids.append(self.index.get(s, self.unk_id))
        return ids

    def decode(self, ids) -> str:
        pieces = [self.vocab[i] for i in ids if self.vocab[i] not in SPECIALS]
        return "".join(pieces).replace(WORD_MARK, " ").strip()

    def save(self, path) -> None:
        Path(path).write_text(
            json.dumps({"version": VERSION, "vocab": self.vocab, "merges": [list(m) for m in self.merges]}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path) -> "Tokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(d["vocab"], [tuple(m) for m in d["merges"]])
