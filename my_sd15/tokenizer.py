"""CLIP BPE Tokenizer for SD 1.5."""

import json
import os

import regex


def _bytes_to_unicode():
    """GPT-2 byte-to-unicode mapping."""
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


class CLIPTokenizer:
    def __init__(self, vocab, merges, bos_id, eos_id):
        self.vocab = vocab  # str -> int
        self.merges = merges  # list of (str, str) pairs
        self._bos_token_id = bos_id
        self._eos_token_id = eos_id
        self.byte_encoder = _bytes_to_unicode()
        # Build merge ranking
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}
        self.pat = regex.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
            regex.IGNORECASE,
        )

    @classmethod
    def from_pretrained(cls, tokenizer_dir):
        with open(os.path.join(tokenizer_dir, "vocab.json")) as f:
            vocab = json.load(f)
        with open(os.path.join(tokenizer_dir, "merges.txt")) as f:
            lines = f.read().strip().split("\n")
        # Skip header line starting with #
        merges = []
        for line in lines:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 2:
                merges.append(tuple(parts))

        bos_id = vocab["<|startoftext|>"]
        eos_id = vocab["<|endoftext|>"]
        return cls(vocab, merges, bos_id, eos_id)

    @property
    def bos_token_id(self):
        return self._bos_token_id

    @property
    def eos_token_id(self):
        return self._eos_token_id

    def _bpe(self, token):
        """Apply BPE merges to a token string. The last character gets </w> suffix."""
        if len(token) == 0:
            return []
        # Add </w> to last character
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            # Find the lowest-rank pair
            pairs = {}
            for i in range(len(word) - 1):
                pair = (word[i], word[i + 1])
                if pair not in pairs:
                    rank = self.bpe_ranks.get(pair, float("inf"))
                    pairs[pair] = rank
            best_pair = min(pairs, key=pairs.get)
            if pairs[best_pair] == float("inf"):
                break
            # Merge the best pair
            merged = best_pair[0] + best_pair[1]
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == best_pair[0] and word[i + 1] == best_pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def encode(self, text):
        """Encode text to list of 77 token IDs."""
        text = text.lower().strip()
        tokens = [self._bos_token_id]

        if text:
            for match in self.pat.findall(text):
                # Encode bytes to unicode
                encoded = "".join(self.byte_encoder[b] for b in match.encode("utf-8"))
                # Apply BPE
                bpe_tokens = self._bpe(encoded)
                for bt in bpe_tokens:
                    if bt in self.vocab:
                        tokens.append(self.vocab[bt])

        tokens.append(self._eos_token_id)

        # Truncate to 77 (keep BOS at start, replace last with EOS if needed)
        if len(tokens) > 77:
            tokens = tokens[:76] + [self._eos_token_id]

        # Pad with EOS to 77
        tokens = tokens + [self._eos_token_id] * (77 - len(tokens))

        return tokens
