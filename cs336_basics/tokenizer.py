import json
import regex as re
from typing import Iterable, Iterator

# GPT-2 官方预分词正则
PRETOKEN_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class Tokenizer:
    def __init__(self,vocab,merges,special_tokens=None):
        """
        vocabulary, list of merges, and (optionally) a list of special tokens. 
            args:
                vocab: dict[int, bytes]  
                merges: list[tuple[bytes, bytes]]  
                special_tokens: list[str] | None = None 
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []
        self.bytes_to_id: dict[bytes, int] = {b: idx for idx, b in vocab.items()}
        
        self.special_tokens_bytes: list[bytes] = []
        next_id = max(vocab.keys())+1
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            self.special_tokens_bytes.append(token_bytes)
            if token_bytes not in self.bytes_to_id:
                self.vocab[next_id] = token_bytes
                self.bytes_to_id[token_bytes] = next_id
                next_id+=1

        self._pretoken_re = re.compile(PRETOKEN_PATTERN)
        self.merges_dict = {pair: i for i, pair in enumerate(merges)}


    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        """
        Class method 
        that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the 
        same format that your BPE training code output) and (optionally) a list of special tokens. 
            args:
                vocab_filepath: str  
                merges_filepath: str  
                special_tokens: list[str] | None = None
        """
        # read vocab
        with open(vocab_filepath, encoding="utf-8") as f:
            vocab_raw = json.loads(f.read())
        
        # convert the [str,str]->[int,bytes]
        vocab:dict[int,bytes]
        vocab = {(int(k),bytes.fromhex(v) for k,v in vocab_raw.items())}

        # read merges
        merges:list[tuple[bytes,bytes]] = []
        with open(merges_filepath,encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split(" ")
            self.merges.append(bytes.fromhex(parts[0]),bytes.fromhex(parts[1]))
        return cls(vocab, merges, special_tokens)


    def _apply_merges(self,tokens:list[bytes])->list[bytes]:
        """
        将一个单个序列的tokens去merges
        注意：这里合并我们肯定得建立一个哈希表，否则遍历一个完整的merges:list[bytes,bytes]这样
             从头到尾遍历太耗费时间了,而且我们合并的时候，我们必须按照优先级，因为我们进行
             bpe的时候我们最后的merges规则就是按照优先级合并的,如果不按照优先级合并，大概率会出问题
             example:
                b't' b'h'
                b'th' b'e'
                b'h' b'e'
            the tokens is : "thehe"
            如果不按照优先级，而是匹配哪个是哪个的话，我们会导致最后tokens变成[b't' b'he' b'he']
            而不是[b'the',b'he']
        """
        while(len(tokens)>=2):
            min_rank = float("inf")
            min_idx = -1
            for i in range(len(tokens)-1):
                pair = (tokens[i], tokens[i + 1])
                rank = self.merges_dict.get(pair)
                if rank is not None and rank < min_rank:
                    min_rank = rank
                    min_idx = i
            # tokens内部已经没有任何可合并的pair了
            if min_idx == -1:
                break
            # 合并优先级最高的那一对
            merged = tokens[min_idx] + tokens[min_idx + 1]
            tokens = tokens[:min_idx] + [merged] + tokens[min_idx + 2:]
        
        return tokens


    def encode(self, text: str) -> list[int]:
        """
        Encode an input text into a sequence of token IDs.
        """
        ids = []

        for match in self._pretoken_re.finditer(text):
            pretoken_str = match.group(0)
            tokens = [bytes([b]) for b in pretoken_bytes]
            # 调用合并函数，进行合并
            tokens = self._apply_merges(tokens)
            for token in tokens:
                id = self.bytes_to_id[token]
                ids.append(id)

        return ids


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Given an iterable of 
        strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is 
        required for memory-efficient tokenization of large files that we cannot directly load into 
        memory
        """
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        Decode a sequence of token IDs into text.
        """
        byte_list = [self.vocab[tid] for tid in ids]
        full_bytes = b"".join(byte_list)
        return full_bytes.decode("utf-8", errors="replace")

    