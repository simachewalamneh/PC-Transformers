import csv
import random
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from data_preparation.config import (
    data_dir, train_path, valid_path, test_path, vocab_size, special_tokens,
)

csv.field_size_limit(10**9)

CHUNK_SIZE_TOKENS = 512
SPLIT_SEED = 42
SPLIT_RATIOS = (0.8, 0.1, 0.1)  # train, valid, test


def _read_text_column(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    return rows[1][0] if len(rows) > 1 else rows[0][0]


def _write_text_column(path, text):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text"])
        writer.writerow([text])


def reconstruct_full_corpus():
    """Re-join the existing contiguous splits back into the original corpus."""
    parts = [_read_text_column(p) for p in (train_path, valid_path, test_path)]
    return "".join(parts)


def build_chunk_boundaries(text, chunk_size_tokens=CHUNK_SIZE_TOKENS):
    """Character-based chunk boundaries calibrated to ~chunk_size_tokens BPE tokens."""
    tmp_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tmp_tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=special_tokens, vocab_size=vocab_size)

    tmp_path = data_dir / "_tmp_full_corpus_for_chunking.txt"
    tmp_path.write_text(text, encoding="utf-8")
    try:
        tmp_tokenizer.train(files=[str(tmp_path)], trainer=trainer)
    finally:
        tmp_path.unlink(missing_ok=True)

    total_tokens = len(tmp_tokenizer.encode(text).ids)
    chars_per_token = len(text) / total_tokens
    chunk_size_chars = max(1, round(chunk_size_tokens * chars_per_token))

    boundaries = []
    pos = 0
    while pos < len(text):
        end = min(pos + chunk_size_chars, len(text))
        if end < len(text):
            # snap forward to the next newline so we don't split mid-line
            newline_pos = text.find("\n", end)
            if newline_pos != -1 and newline_pos - end < chunk_size_chars:
                end = newline_pos + 1
        boundaries.append((pos, end))
        pos = end

    return boundaries, total_tokens, chunk_size_chars


def split_corpus(seed=SPLIT_SEED, ratios=SPLIT_RATIOS, chunk_size_tokens=CHUNK_SIZE_TOKENS, text=None):
    if text is None:
        text = reconstruct_full_corpus()
    boundaries, total_tokens, chunk_size_chars = build_chunk_boundaries(text, chunk_size_tokens)
    chunks = [text[start:end] for start, end in boundaries]

    order = list(range(len(chunks)))
    rng = random.Random(seed)
    rng.shuffle(order)

    n = len(order)
    n_train = round(n * ratios[0])
    n_valid = round(n * ratios[1])

    # Keep each split's chunks in their original corpus order internally --
    # only the *assignment* of chunks to splits is shuffled.
    train_idx = sorted(order[:n_train])
    valid_idx = sorted(order[n_train:n_train + n_valid])
    test_idx = sorted(order[n_train + n_valid:])

    assert not (set(train_idx) & set(valid_idx) & set(test_idx))
    assert len(train_idx) + len(valid_idx) + len(test_idx) == n

    return {
        "train_text": "".join(chunks[i] for i in train_idx),
        "valid_text": "".join(chunks[i] for i in valid_idx),
        "test_text": "".join(chunks[i] for i in test_idx),
        "n_chunks": n,
        "chunk_size_chars": chunk_size_chars,
        "total_tokens_est": total_tokens,
        "chunk_assignment": {"train": train_idx, "valid": valid_idx, "test": test_idx},
        "full_text": text,
    }


def main():
    result = split_corpus()
    _write_text_column(train_path, result["train_text"])
    _write_text_column(valid_path, result["valid_text"])
    _write_text_column(test_path, result["test_text"])

    total_chars = len(result["train_text"]) + len(result["valid_text"]) + len(result["test_text"])
    print(f"Chunks: {result['n_chunks']} (~{result['chunk_size_chars']} chars each, "
          f"target {CHUNK_SIZE_TOKENS} tokens/chunk, corpus ~{result['total_tokens_est']} tokens)")
    for name, t in (("train", result["train_text"]), ("valid", result["valid_text"]), ("test", result["test_text"])):
        print(f"  {name}: {len(t):>8} chars  ({100 * len(t) / total_chars:5.1f}%)")


if __name__ == "__main__":
    main()
