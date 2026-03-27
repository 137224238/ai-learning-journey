"""
RAGAS 评估框架实践
评估 RAG 系统的四大核心指标
"""
from dataclasses import dataclass
from typing import List
import math


@dataclass
class RAGSample:
    """单条评估样本"""
    question: str
    answer: str            # RAG 系统生成的答案
    contexts: List[str]    # 检索到的上下文
    ground_truth: str      # 标准答案


def cosine_sim(a: List[float], b: List[float]) -> float:
    """余弦相似度（模拟 embedding）"""
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x**2 for x in a))
    nb = math.sqrt(sum(x**2 for x in b))
    return dot / (na * nb + 1e-10)


class RAGASEvaluator:
    """
    RAGAS 评估器（简化版，理解指标含义）

    生产使用：
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy
    """

    def faithfulness(self, sample: RAGSample) -> float:
        """
        忠实度：答案中每个声明是否都有上下文支撑
        公式：supported_claims / total_claims
        """
        # 真实版：用 NLI 模型判断每个声明是否被上下文支持
        # 简化：检查答案词汇在上下文中的覆盖率
        answer_words = set(sample.answer.lower().split())
        context_words = set(" ".join(sample.contexts).lower().split())
        if not answer_words:
            return 0.0
        overlap = len(answer_words & context_words) / len(answer_words)
        return min(overlap * 1.2, 1.0)  # 简化估算

    def answer_relevancy(self, sample: RAGSample) -> float:
        """
        答案相关性：答案是否回答了问题
        真实版：用 LLM 基于答案反向生成问题，计算与原问题的 embedding 相似度
        """
        # 简化：检查问题关键词在答案中的出现率
        q_words = set(sample.question.lower().replace("？", "").replace("?", "").split())
        a_words = set(sample.answer.lower().split())
        if not q_words:
            return 0.0
        return len(q_words & a_words) / len(q_words)

    def context_precision(self, sample: RAGSample) -> float:
        """
        上下文精确率：检索到的块中有多少是真正有用的
        """
        if not sample.contexts:
            return 0.0
        gt_words = set(sample.ground_truth.lower().split())
        useful = sum(
            1 for ctx in sample.contexts
            if len(gt_words & set(ctx.lower().split())) > 2
        )
        return useful / len(sample.contexts)

    def context_recall(self, sample: RAGSample) -> float:
        """
        上下文召回率：标准答案所需信息有多少被检索到
        """
        gt_words = set(sample.ground_truth.lower().split())
        context_words = set(" ".join(sample.contexts).lower().split())
        if not gt_words:
            return 0.0
        return len(gt_words & context_words) / len(gt_words)

    def evaluate(self, samples: List[RAGSample]) -> dict:
        """批量评估，返回平均分"""
        results = {"faithfulness": [], "answer_relevancy": [],
                   "context_precision": [], "context_recall": []}

        for s in samples:
            results["faithfulness"].append(self.faithfulness(s))
            results["answer_relevancy"].append(self.answer_relevancy(s))
            results["context_precision"].append(self.context_precision(s))
            results["context_recall"].append(self.context_recall(s))

        avg = {k: sum(v)/len(v) for k, v in results.items()}
        avg["ragas_score"] = sum(avg.values()) / 4
        return avg


if __name__ == "__main__":
    samples = [
        RAGSample(
            question="什么是 RAG？",
            answer="RAG 是检索增强生成技术，结合检索系统和生成模型，能减少幻觉。",
            contexts=[
                "检索增强生成（RAG）结合了检索系统和生成模型的优点。",
                "RAG 可以有效减少大语言模型的幻觉问题。"
            ],
            ground_truth="RAG 是一种结合检索与生成的 AI 技术。"
        ),
        RAGSample(
            question="Transformer 的核心是什么？",
            answer="Transformer 的核心是注意力机制，特别是自注意力机制。",
            contexts=[
                "注意力机制是 Transformer 架构的核心。",
                "自注意力机制使模型能够关注输入序列中的不同位置。"
            ],
            ground_truth="注意力机制是 Transformer 的核心。"
        ),
    ]

    evaluator = RAGASEvaluator()
    scores = evaluator.evaluate(samples)

    print("=== RAGAS 评估结果 ===")
    for metric, score in scores.items():
        bar = "█" * int(score * 20)
        print(f"{metric:25s}: {score:.3f} |{bar}")
