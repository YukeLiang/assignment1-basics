import pickle
from typing import Iterable, Iterator
import heapq

import regex as re

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        self.vocab = vocab
        self.encoder = {v: k for k, v in vocab.items()}
        self.merges = {merge: i for i, merge in enumerate(merges)}
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

    def merge(self, pre_token_bytes_tuple: tuple[bytes, ...]) -> tuple[bytes, ...]:
        """Given a tuple of pre-tokenized bytes, return a list of merged tokens according to the BPE merges."""
        merged_tokens = pre_token_bytes_tuple
        while len(merged_tokens) >= 1:
            # pair with lowest index in self.merges is chosen first
            chosen_pair = None
            for i in range(len(merged_tokens) - 1):
                pair = (merged_tokens[i], merged_tokens[i + 1])
                if pair in self.merges:
                    if chosen_pair is None or self.merges[pair] < self.merges[chosen_pair]:
                        chosen_pair = pair

            if chosen_pair is None:
                break  # no more merges can be applied

            # merge the chosen pair
            j = 0
            result = []
            while j < len(merged_tokens):
                if (
                    j < len(merged_tokens) - 1
                    and (merged_tokens[j], merged_tokens[j + 1]) in self.merges
                ):
                    pair = (merged_tokens[j], merged_tokens[j + 1])
                    if pair == chosen_pair:
                        result.append(merged_tokens[j] + merged_tokens[j + 1])
                        j += 2  # skip the next token since it has been merged
                        continue
                result.append(merged_tokens[j])
                j += 1
            merged_tokens = tuple(result)
        return merged_tokens

    def pretokenize(self, text: str, special_tokens: list[str]) -> tuple[tuple[bytes, ...], ...]:
        """Pretokenize an input text into a tuple of bytes, handling special tokens."""
        if special_tokens:
            # sort special tokens by length in descending order to avoid partial matches
            special_tokens.sort(key=len, reverse=True)
            # regex pattern to split on special tokens from the input file to avoid counting them in the pre-tokenization step
            split_pattern = r"(" + "|".join(re.escape(token) for token in (special_tokens or [])) + r")"
            # split the chunk into pre-tokens using regex and count the occurrences of each pre-token
            text_processed = re.split(split_pattern, text)
        else: 
        # When special_tokens is empty, split_pattern collapses to "()" (an empty capturing group), 
        # which matches zero-width everywhere and makes re.split fragment the text character-by-character instead of leaving 
        # it whole — so every word gets pre-tokenized one letter at a time and can never merge.
            text_processed = [text]

        res = tuple()
        for t in text_processed:
            if t in special_tokens:
                special_token_tuple = (t.encode("utf-8"),)
                res += (special_token_tuple,)
                continue
            # Run pre-tokenization on your sub-chunk and store the counts for each pre-token
            PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
            pre_tokens = re.findall(PAT, t, flags=re.IGNORECASE | re.UNICODE)
            for pre_token in pre_tokens:
                # Store the counts for each pre-token in byte arrays in a dictionary or other data structure
                pre_token_raw_bytes = pre_token.encode("utf-8")
                pre_token_bytes_tuple = tuple(pre_token_raw_bytes[i : i + 1] for i in range(len(pre_token_raw_bytes)))
                res += (pre_token_bytes_tuple,)
        return res

    def encode(self, text: str) -> list[int]:
        """Encode an input text into a sequence of token IDs."""
        # Implement the BPE encoding algorithm here, using the vocabulary and merges to convert the input text into a sequence of token IDs.
        # pretokenize the text into bytes
        # use encoder to convert the merged tokens into token IDs
        pre_token_bytes = self.pretokenize(text, self.special_tokens)
        merged_pre_token_bytes = []
        for pre_token in pre_token_bytes:
            merged_pre_token_bytes += self.merge(pre_token)
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

if __name__ == "__main__":
    pass