"""
损失函数：从数学到代码，理解 LLM 训练目标
"""

import math
from typing import List


# ============================================================
# 1. 交叉熵损失（Cross-Entropy Loss）
# LLM 预训练的核心损失：预测下一个 token
# ============================================================

def cross_entropy_loss(logits: List[float], target_idx: int) -> float:
    """
    计算单个样本的交叉熵损失
    
    Args:
        logits: 模型输出的原始分数（未归一化）
        target_idx: 正确 token 的索引
    
    Returns:
        loss: 标量损失值
    """
    # Step 1: Softmax 归一化 → 概率分布
    max_logit = max(logits)  # 数值稳定性技巧
    exp_logits = [math.exp(x - max_logit) for x in logits]
    sum_exp = sum(exp_logits)
    probs = [x / sum_exp for x in exp_logits]
    
    # Step 2: 取目标 token 的对数概率（负号因为要最小化）
    loss = -math.log(probs[target_idx] + 1e-10)
    return loss


def batch_cross_entropy(logits_batch: List[List[float]], targets: List[int]) -> float:
    """批次平均交叉熵损失"""
    losses = [cross_entropy_loss(logits, target) 
              for logits, target in zip(logits_batch, targets)]
    return sum(losses) / len(losses)


# ============================================================
# 2. KL 散度（KL Divergence）
# RLHF 训练中用于约束策略偏离参考模型的程度
# KL(P || Q) = Σ P(x) * log(P(x) / Q(x))
# ============================================================

def kl_divergence(p: List[float], q: List[float]) -> float:
    """
    计算 KL(P || Q)：P 相对于 Q 的 KL 散度
    
    在 RLHF 中：
    - P = 当前策略模型的输出分布
    - Q = 参考模型（SFT model）的输出分布
    - 目标：让 P 不要偏离 Q 太远
    """
    assert abs(sum(p) - 1.0) < 1e-6, "P 必须是概率分布"
    assert abs(sum(q) - 1.0) < 1e-6, "Q 必须是概率分布"
    
    kl = 0.0
    for pi, qi in zip(p, q):
        if pi > 1e-10:  # 避免 log(0)
            kl += pi * math.log(pi / (qi + 1e-10))
    return kl


# ============================================================
# 3. 对比损失（Contrastive Loss / InfoNCE）
# Embedding 模型训练：让相似文本更近，不相似文本更远
# ============================================================

def cosine_similarity(a: List[float], b: List[float]) -> float:
    """余弦相似度"""
    dot = sum(ai * bi for ai, bi in zip(a, b))
    norm_a = math.sqrt(sum(ai ** 2 for ai in a))
    norm_b = math.sqrt(sum(bi ** 2 for bi in b))
    return dot / (norm_a * norm_b + 1e-10)


def info_nce_loss(
    query: List[float],
    positive: List[float],
    negatives: List[List[float]],
    temperature: float = 0.07
) -> float:
    """
    InfoNCE 对比损失（CLIP、Sentence-BERT 等使用）
    
    目标：最大化 query 与 positive 的相似度
         最小化 query 与 negatives 的相似度
    
    Args:
        temperature: 越小 → 分布越尖锐 → 对细微差异越敏感
    """
    # 计算相似度
    pos_sim = cosine_similarity(query, positive) / temperature
    neg_sims = [cosine_similarity(query, neg) / temperature for neg in negatives]
    
    # InfoNCE = -log(exp(pos) / (exp(pos) + Σexp(neg)))
    all_sims = [pos_sim] + neg_sims
    max_sim = max(all_sims)  # 数值稳定
    exp_sims = [math.exp(s - max_sim) for s in all_sims]
    
    loss = -math.log(exp_sims[0] / sum(exp_sims) + 1e-10)
    return loss


# ============================================================
# 演示与验证
# ============================================================

if __name__ == "__main__":
    print("=== 交叉熵损失 ===")
    # 模型对 3 个 token 的 logits，正确答案是第 1 个
    logits = [3.0, 1.0, 0.5]  # 第 0 个最高，说明模型预测正确
    loss_correct = cross_entropy_loss(logits, target_idx=0)
    loss_wrong = cross_entropy_loss(logits, target_idx=2)
    print(f"预测正确时的损失: {loss_correct:.4f}")
    print(f"预测错误时的损失: {loss_wrong:.4f}")
    print(f"困惑度 (Perplexity) = exp(loss) = {math.exp(loss_correct):.2f}")

    print("\n=== KL 散度 ===")
    # 相同分布 → KL = 0
    p = [0.7, 0.2, 0.1]
    q_same = [0.7, 0.2, 0.1]
    q_diff = [0.1, 0.1, 0.8]
    print(f"KL(P || P) = {kl_divergence(p, q_same):.4f} (应接近 0)")
    print(f"KL(P || Q_diff) = {kl_divergence(p, q_diff):.4f} (应较大)")

    print("\n=== InfoNCE 对比损失 ===")
    query_vec = [1.0, 0.0, 0.0]
    pos_vec = [0.9, 0.1, 0.0]    # 与 query 相似
    neg_vec1 = [0.0, 1.0, 0.0]   # 不相似
    neg_vec2 = [0.0, 0.0, 1.0]   # 不相似
    loss = info_nce_loss(query_vec, pos_vec, [neg_vec1, neg_vec2])
    print(f"对比损失: {loss:.4f}")
