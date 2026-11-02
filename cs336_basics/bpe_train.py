import os
import multiprocessing
import regex as re  # 必须用regex库，支持Unicode属性\p{L}
from .pretokenization_example import find_chunk_boundaries

# GPT-2 官方预分词正则
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_single_chunk(args):
    """
    子进程执行函数：处理单个语料块，返回该块内的 pretoken 频次字典
    args 元组包含：文件路径、起始字节偏移、结束字节偏移、特殊token列表
    """
    file_path,start,end,special_tokens = args

    with open(file_path,"rb") as f:
        f.seek(start)
        chunk_bytes = f.read(end - start)

    chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
    split_re = '|'.join(re.escape(tok) for tok in special_tokens)
    documents = re.split(split_re, chunk_text)
    pretoken_re = re.compile(PRETOKEN_PATTERN)
    local_counts = {}  # key: tuple[bytes, ...], value: 出现次数

    for predocument in documents:
        if not predocument:
            continue
        # 用finditer迭代处理，不用一次性存所有pretoken，省内存
        for match in pretoken_re.finditer(predocument):
            pretoken_str = match.group(0)
            # 转成单字节tuple（可哈希，能当字典key）
            pretoken_bytes = tuple(bytes([b]) for b in pretoken_str.encode("utf-8"))
            # 累加计数
            local_counts[pretoken_bytes] = local_counts.get(pretoken_bytes, 0) + 1
    
    return local_counts



def build_pretoken_counts(
    input_path: str,
    special_tokens: list[str],
    num_workers: int = None,
    chunks_per_worker: int = 4
)-> dict[tuple[bytes, ...], int]:
    if num_workers is None:
        num_workers = multiprocessing.cpu_count()
    # the sum of chunk needed
    desired_num_chunks = num_workers * chunks_per_worker
    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(
            file=f,
            desired_num_chunks=desired_num_chunks,
            split_special_token=special_tokens[0].encode("utf-8")
        )
    # creat all task
    tasks = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tasks.append((input_path, start, end, special_tokens))
    # allot the task for process_pool
    # it will return a such as {"b'h',b'e',b'l',b'l',b'0'",5}
    global_counts = {}
    with multiprocessing.Pool(processes=num_workers) as pool:
        all_local_counts = pool.map(process_single_chunk, tasks)
    # the main process gather the poll_process result
    for local_dict in all_local_counts:
        for token, cnt in local_dict.items():
            global_counts[token] = global_counts.get(token, 0) + cnt

    return global_counts

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
    # 2.1 call the find_chunk_boundarise and allot the task for process_poll
    pretoken_dict = build_pretoken_counts()

    # 3.merge pretokrn






