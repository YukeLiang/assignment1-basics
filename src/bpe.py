import multiprocessing
import os
from typing import BinaryIO, Counter

import regex as re


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pretokenize_chunk(chunk: bytes, special_tokens: list[str] | None = None) -> dict[tuple[bytes, ...], int]:
    # sort special tokens by length in descending order to avoid partial matches
    if special_tokens:
        special_tokens.sort(key=len, reverse=True)
    # regex pattern to split on special tokens from the input file to avoid counting them in the pre-tokenization step
    split_pattern = "|".join(re.escape(token) for token in (special_tokens or [])).encode("utf-8")
    # split the chunk into pre-tokens using regex and count the occurrences of each pre-token
    chunk_processed = re.split(split_pattern, chunk)
    # Run pre-tokenization on each sub-chunk and store the counts for each pre-token
    pre_token_counts: dict[tuple[bytes, ...], int] = {}
    for sub_chunk in chunk_processed:
        # Run pre-tokenization on your sub-chunk and store the counts for each pre-token
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        pre_tokens = re.findall(PAT, sub_chunk.decode("utf-8", errors="ignore"), flags=re.IGNORECASE | re.UNICODE)
        for pre_token in pre_tokens:
            # Store the counts for each pre-token in byte arrays in a dictionary or other data structure
            pre_token_raw_bytes = pre_token.encode("utf-8")
            pre_token_bytes_tuple = tuple(pre_token_raw_bytes[i : i + 1] for i in range(len(pre_token_raw_bytes)))
            pre_token_counts[pre_token_bytes_tuple] = pre_token_counts.get(pre_token_bytes_tuple, 0) + 1
    return pre_token_counts


def pretokenize_file(input_path: str, special_tokens: list[str] | None = None) -> dict[tuple[bytes, ...], int]:
    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")  # hardcoded per document
        pre_token_counts: dict[tuple[bytes, ...], int] = {}

        # Create a list of chunks to process
        chunks = []
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start)
            chunks.append(chunk)

        # use multiprocessing to parallelize the pre-tokenization of chunks
        multiprocessing_enabled = True
        if multiprocessing_enabled:
            with multiprocessing.Pool(processes=num_processes) as pool:
                # Map the parallel_pretokenize_chunk function to the chunks
                res = pool.starmap(pretokenize_chunk, ((chunk, special_tokens) for chunk in chunks))
                # Combine the results from all chunks into a single dictionary by adding the counts for each pre-token
                # for r in res:
                #     for pre_token_bytes_tuple, count in r.items():
                #         pre_token_counts[pre_token_bytes_tuple] = pre_token_counts.get(pre_token_bytes_tuple, 0) + count

                # use collections.Counter to combine the results from all chunks into a single dictionary by adding the counts for each pre-token
                pre_token_counts = dict(sum((Counter(chunk_result) for chunk_result in res), Counter()))
        else:
            # The following is a serial implementation, but you can parallelize this
            # by sending each start/end pair to a set of processes.
            for chunk in chunks:
                result = pretokenize_chunk(chunk, special_tokens)
                for pre_token_bytes_tuple, count in result.items():
                    pre_token_counts[pre_token_bytes_tuple] = pre_token_counts.get(pre_token_bytes_tuple, 0) + count
    return pre_token_counts


def get_pairs(pre_token_counts: dict[tuple[bytes, ...], int]) -> dict[tuple[bytes, bytes], int]:
    """Get the counts of pairs of consecutive bytes in the pre-token counts."""
    pairs: dict[tuple[bytes, bytes], int] = {}
    for pre_token_bytes_tuple, count in pre_token_counts.items():
        # Count the occurrences of each pair of consecutive bytes in the pre-token
        for i in range(len(pre_token_bytes_tuple) - 1):
            pair = (pre_token_bytes_tuple[i], pre_token_bytes_tuple[i + 1])
            pairs[pair] = pairs.get(pair, 0) + count
    return pairs


def merge_pair(input_counts: dict[tuple[bytes, ...], int], pair: tuple[bytes, bytes]) -> dict[tuple[bytes, ...], int]:
    """Merge a pair of consecutive bytes in the pre-token counts."""
    output_counts: dict[tuple[bytes, ...], int] = {}
    for pre_token_bytes_tuple, count in input_counts.items():
        merged_tuple = []
        i = 0
        while i < len(pre_token_bytes_tuple):
            if i < len(pre_token_bytes_tuple) - 1 and (pre_token_bytes_tuple[i], pre_token_bytes_tuple[i + 1]) == pair:
                merged_tuple.append(pair[0] + pair[1])  # Merge the pair into a single byte
                i += 2  # Skip the next byte since it's part of the merged pair
            else:
                merged_tuple.append(pre_token_bytes_tuple[i])
                i += 1
        output_counts[tuple(merged_tuple)] = output_counts.get(tuple(merged_tuple), 0) + count
    return output_counts


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str] | None = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a BPE tokenizer from a text corpus."""
    # initialize vocabulary with special tokens if provided and 256 byte values
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    vocab.update({i + 256: token.encode("utf-8") for i, token in enumerate(special_tokens or [])})
    merges: list[tuple[bytes, bytes]] = []

    # pretokenize the file and get the pre-token counts
    pre_token_counts = pretokenize_file(input_path, special_tokens)

    merge_count = vocab_size - 256 - len(special_tokens or [])
    for _ in range(merge_count):
        pairs = get_pairs(pre_token_counts)
        if not pairs:
            break
        # Find the most frequent pair of consecutive bytes
        # if count is the same, ties broken by choosing the greater lexicographical order one

        most_frequent_pair = max(pairs, key=lambda pair: (pairs[pair], pair))
        merges.append(most_frequent_pair)
        # Add the merged pair to the vocabulary with a new index
        vocab.update({len(vocab): most_frequent_pair[0] + most_frequent_pair[1]})
        # Merge the most frequent pair in the pre-token counts
        pre_token_counts = merge_pair(pre_token_counts, most_frequent_pair)

    return vocab, merges


# z = {}
# a = {(b'a',): 3, (b'b',): 2}
# b = {(b'a',): 1, (b'c',): 5}
# c = [a, b]
# for r in c:
#     for pre_token_bytes_tuple, count in r.items():
#         z[pre_token_bytes_tuple] = z.get(pre_token_bytes_tuple, 0) + count
# print(z)  # Output: {(b'a',): 4, (b'b',): 2, (b'c',): 5}

if __name__ == "__main__":
    # print (merge_pair({(b't', b'h', b'e'): 1}, (b't', b'h')))
    print(train_bpe("data/TinyStoriesV2-GPT4-valid.txt", 1000, ["<|endoftext|>"]))  # Replace with your input file path
