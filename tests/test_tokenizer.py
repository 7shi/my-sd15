"""Verify CLIP tokenizer against test data and known properties."""

import os

import pytest
import torch

from my_sd15.loader import DEFAULT_WEIGHTS_DIR
from my_sd15.tokenizer import CLIPTokenizer

TOKENIZER_DIR = os.path.join(DEFAULT_WEIGHTS_DIR, "tokenizer")


def tokenizer_available():
    return os.path.exists(os.path.join(TOKENIZER_DIR, "vocab.json"))


@pytest.fixture
def tokenizer():
    return CLIPTokenizer.from_pretrained(TOKENIZER_DIR)


@pytest.mark.skipif(not tokenizer_available(), reason="tokenizer files not found")
class TestTokenizerProperties:
    def test_output_length(self, tokenizer):
        """Output is always exactly 77 tokens."""
        assert len(tokenizer.encode("hello")) == 77
        assert len(tokenizer.encode("")) == 77
        assert len(tokenizer.encode("a " * 1000)) == 77

    def test_starts_with_bos(self, tokenizer):
        """First token is always BOS (<|startoftext|>)."""
        ids = tokenizer.encode("anything")
        assert ids[0] == tokenizer.bos_token_id

    def test_has_eos(self, tokenizer):
        """EOS token appears in the output."""
        ids = tokenizer.encode("hello")
        assert tokenizer.eos_token_id in ids

    def test_padding_is_eos(self, tokenizer):
        """Padding tokens are EOS."""
        ids = tokenizer.encode("hi")
        # After EOS, all remaining should be EOS (padding)
        eos_pos = ids.index(tokenizer.eos_token_id)
        for i in range(eos_pos, 77):
            assert ids[i] == tokenizer.eos_token_id

    def test_lowercased(self, tokenizer):
        """Input is lowercased: 'Hello' and 'hello' produce same tokens."""
        assert tokenizer.encode("Hello World") == tokenizer.encode("hello world")

    def test_empty_string(self, tokenizer):
        """Empty string produces BOS + EOS + padding."""
        ids = tokenizer.encode("")
        assert ids[0] == tokenizer.bos_token_id
        assert ids[1] == tokenizer.eos_token_id
        assert all(t == tokenizer.eos_token_id for t in ids[2:])

    def test_truncation(self, tokenizer):
        """Long input is truncated to 77 tokens with EOS at the end."""
        ids = tokenizer.encode("word " * 200)
        assert len(ids) == 77
        assert ids[0] == tokenizer.bos_token_id
        assert ids[76] == tokenizer.eos_token_id


@pytest.mark.skipif(not tokenizer_available(), reason="tokenizer files not found")
class TestTokenizerTestData:
    def test_cond_ids(self, tokenizer, tokenizer_data, metadata):
        """Tokenization of the test prompt matches saved data."""
        ids = tokenizer.encode(metadata["prompt"])
        assert ids == tokenizer_data["cond_ids"].tolist()

    def test_uncond_ids(self, tokenizer, tokenizer_data):
        """Tokenization of empty string matches saved data."""
        ids = tokenizer.encode("")
        assert ids == tokenizer_data["uncond_ids"].tolist()
