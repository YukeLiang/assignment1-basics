import pickle
from typing import Iterable, Iterator

import regex as re

from src.bpe_train import pretokenize_chunk


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.encoder = {v: k for k, v in vocab.items()}
        self.merges = merges
        self.special_tokens = special_tokens or []

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        """
        Class method
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the
        same format that your BPE training code output) and (optionally) a list of special tokens.
        This method should accept the following additional parameters:
        vocab_filepath: str
        merges_filepath: str
        special_tokens: list[str] | None = None
        """
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def merge(self, pre_token_bytes_tuple: tuple[bytes, ...]) -> tuple[bytes]:
        """Given a tuple of pre-tokenized bytes, return a list of merged tokens according to the BPE merges."""
        i = 0
        merged_tokens = pre_token_bytes_tuple
        continue_merging = True
        while len(merged_tokens) > 1 and continue_merging:
            result = []
            # continue merging until no more merges can be applied
            while i < len(pre_token_bytes_tuple):
                if (
                    i < len(pre_token_bytes_tuple) - 1
                    and (pre_token_bytes_tuple[i], pre_token_bytes_tuple[i + 1]) in self.merges
                ):
                    result.append(pre_token_bytes_tuple[i] + pre_token_bytes_tuple[i + 1])
                    i += 1  # skip the next token since it has been merged
                else:
                    result.append(pre_token_bytes_tuple[i])
                i += 1
            if len(result) == len(pre_token_bytes_tuple):
                continue_merging = False  # no more merges can be applied
            merged_tokens = tuple(result)
        return merged_tokens

    def pretokenize(self, text: str, special_tokens: list[str]) -> tuple[bytes, ...]:
        """Pretokenize an input text into a tuple of bytes, handling special tokens."""
        # sort special tokens by length in descending order to avoid partial matches
        if special_tokens:
            special_tokens.sort(key=len, reverse=True)
        # regex pattern to split on special tokens from the input file to avoid counting them in the pre-tokenization step
        split_pattern = r"(" + "|".join(re.escape(token) for token in (special_tokens or [])) + r")"
        # split the chunk into pre-tokens using regex and count the occurrences of each pre-token
        text_processed = re.split(split_pattern, text)
        for sub_chunk in text_processed:
            # Run pre-tokenization on your sub-chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_tokens = re.findall(PAT, sub_chunk.decode("utf-8", errors="ignore"), flags=re.IGNORECASE | re.UNICODE)
            for pre_token in pre_tokens:
                # Store the counts for each pre-token in byte arrays in a dictionary or other data structure
                pre_token_raw_bytes = pre_token.encode("utf-8")
                pre_token_bytes_tuple = tuple(pre_token_raw_bytes[i : i + 1] for i in range(len(pre_token_raw_bytes)))
        return pre_token_bytes_tuple

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # Implement the BPE encoding algorithm here, using the vocabulary and merges to convert the input text into a sequence of token IDs.
        # pretokenize the text into bytes
        # use encoder to convert the merged tokens into token IDs
        merged_pre_token_bytes = self.merge(text, self.special_tokens)
        encoded_ids = [self.encoder[token] for token in merged_pre_token_bytes if token in self.encoder]
        return encoded_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory."""
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode a sequence of token IDs into text."""
        # Implement the BPE decoding algorithm here, using the vocabulary to convert the sequence of token IDs back into text.
        decoded_tokens = [self.vocab[id] for id in ids if id in self.vocab]
        return b"".join(decoded_tokens).decode("utf-8", errors="ignore")

    # To test your Tokenizer against our provided tests, you will first need to implement the test
    # adapter at [adapters.get_tokenizer] . Then, run uv run pytest tests/test_tokenizer.py. Your
    # implementation should be able to pass all tests.


if __name__ == "__main__":
    special_tokens = ["<|endoftext|>", "<|pad|>", "<|unk|>"]
    split_pattern = r"(" + "|".join(re.escape(token) for token in (special_tokens or [])) + r")"
    split_pattern2 = "|".join(re.escape(token) for token in (special_tokens or []))
    print(re.split(split_pattern, "a,<|endoftext|><|pad|><|unk|>,b,<|endoftext|><|pad|><|unk|>,c"))
    # print (re.split(r"(,.,)", "a,.,b,.,c"))
    print(re.split(split_pattern2, "a,<|endoftext|><|pad|><|unk|>,b,<|endoftext|><|pad|><|unk|>,c"))
# print (re.split(r",.,", "a,.,b,.,c"))
