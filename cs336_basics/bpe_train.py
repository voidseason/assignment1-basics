


def train_bpe(
    input_file: str,
    vocab_size: int,
    special_tokens:list[str],
) -> tuple[dict[int,bytes],list[tuple[bytes,bytes]]]:
    """
    Train BPE model on input_file.
    """
    # 1.Intalize the vocabulary
    vocab: dict[int, bytes] = {}
    # 1.1 Add all possible byte pairs
    for byte_id in range(256):
        vocab[byte_id] = bytes([byte_id])
    # 1.2 Add special tokens
    for token_str in special_tokens:
        token_bytes = token_str.encode("utf-8")
        vocab[len(vocab)] = token_bytes
    # 1.3 make judge enable
    if vocab_size < len(vocab):
        raise ValueError(
            f"vocab_size({vocab_size}) 小于初始词表大小({len(vocab)})，无法训练"
        )
    num_merges = vocab_size - len(vocab)

    # 2. count pretoken

