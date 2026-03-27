"""
BPE（Byte Pair Encoding）分词器从零实现
GPT 系列使用的分词算法
"""
from collections import defaultdict
from typing import Dict, List, Tuple


class BPETokenizer:
    """
    BPE 分词器

    核心思想：
    1. 初始化：每个字符作为一个 token
    2. 统计所有相邻 token 对的频率
    3. 合并频率最高的 token 对
    4. 重复 2-3，直到词表达到目标大小
    """

    def __init__(self, vocab_size: int = 300):
        self.vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []  # 合并规则列表
        self.vocab: Dict[str, int] = {}

    def _get_word_freqs(self, corpus: List[str]) -> Dict[tuple, int]:
        """统计词频，每个词用字符元组表示"""
        freq: Dict[tuple, int] = defaultdict(int)
        for sentence in corpus:
            for word in sentence.split():
                # 在末尾加 </w> 标记词边界
                chars = tuple(list(word) + ["</w>"])
                freq[chars] += 1
        return dict(freq)

    def _get_pair_stats(self, word_freqs: Dict[tuple, int]) -> Dict[Tuple, int]:
        """统计所有相邻 token 对的频率"""
        pairs: Dict[Tuple, int] = defaultdict(int)
        for word, freq in word_freqs.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return dict(pairs)

    def _merge_pair(
        self, pair: Tuple[str, str], word_freqs: Dict[tuple, int]
    ) -> Dict[tuple, int]:
        """将指定的 token 对合并为新 token"""
        new_freqs: Dict[tuple, int] = {}
        merged = pair[0] + pair[1]

        for word, freq in word_freqs.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(merged)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_freqs[tuple(new_word)] = freq

        return new_freqs

    def train(self, corpus: List[str]):
        """在语料库上训练 BPE"""
        word_freqs = self._get_word_freqs(corpus)

        # 初始词表：所有字符
        all_chars = set()
        for word in word_freqs:
            all_chars.update(word)
        self.vocab = {c: i for i, c in enumerate(sorted(all_chars))}

        print(f"初始词表大小: {len(self.vocab)}")

        # 迭代合并
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pair_stats(word_freqs)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)  # type: ignore
            word_freqs = self._merge_pair(best_pair, word_freqs)
            merged_token = best_pair[0] + best_pair[1]
            self.merges.append(best_pair)
            self.vocab[merged_token] = len(self.vocab)

        print(f"训练完成，词表大小: {len(self.vocab)}，合并次数: {len(self.merges)}")

    def tokenize(self, text: str) -> List[str]:
        """对文本进行 BPE 分词"""
        tokens = []
        for word in text.split():
            word_tokens = list(word) + ["</w>"]
            # 应用所有合并规则
            for pair in self.merges:
                i = 0
                while i < len(word_tokens) - 1:
                    if word_tokens[i] == pair[0] and word_tokens[i + 1] == pair[1]:
                        word_tokens = (
                            word_tokens[:i]
                            + [pair[0] + pair[1]]
                            + word_tokens[i + 2:]
                        )
                    else:
                        i += 1
            tokens.extend(word_tokens)
        return tokens


if __name__ == "__main__":
    corpus = [
        "low lower lowest",
        "new newer newest",
        "old older oldest",
        "the tokenizer is a key component of llm",
        "bpe is used by gpt series models",
    ]

    tokenizer = BPETokenizer(vocab_size=80)
    tokenizer.train(corpus)

    test = "newer lowest"
    print(f"\n分词结果：'{test}' → {tokenizer.tokenize(test)}")
    print(f"\n前 10 条合并规则：{tokenizer.merges[:10]}")
