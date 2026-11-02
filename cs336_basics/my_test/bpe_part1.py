import os
from cs336_basics.bpe_train import build_pretoken_counts, process_single_chunk

"""
the file is test to:
    2. count pretoken
"""

PATH = "/home/Voidseason/stanford/sc336/assignment1-basics/cs336_basics/my_test/tinystories_train.txt"
SPECIALS = ["<|endoftext|>"]

if __name__ == "__main__":
    size = os.path.getsize(PATH)

    # 基准：不分块，整份文件一次处理
    single = process_single_chunk((PATH, 0, size, SPECIALS))

    # 被测：分块 + 多进程 + 汇总
    parallel = build_pretoken_counts(PATH, SPECIALS, num_workers=2)

    assert parallel == single, "分块并行结果和整文件结果不一致"

    for key, cnt in parallel.items():
        assert isinstance(key, tuple) and key, "key 必须是非空元组"
        assert all(isinstance(b, bytes) and len(b) == 1 for b in key), "key 元素应是单字节"
        assert isinstance(cnt, int) and cnt > 0, "计数必须是正整数"

    # 特殊 token 不应被拆碎后进入统计
    assert (b"e", b"n", b"d", b"o", b"f", b"t", b"e", b"x", b"t") not in parallel
    assert (b"<", b"|") not in parallel
    assert (b"|", b">") not in parallel

    # 文件里内容块重复了 8 次，块首的 "Once" 应正好出现 8 次
    assert parallel.get((b"O", b"n", b"c", b"e")) == 6

    print("OK:", len(parallel), "种 pretoken，总计", sum(parallel.values()), "次")