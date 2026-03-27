"""
Transformer 从零实现（附详细注释）
基于论文《Attention Is All You Need》(Vaswani et al., 2017)

架构总览：
Input → Embedding + Positional Encoding
      → N × Encoder Layers (Self-Attention + FFN)
      → N × Decoder Layers (Masked Self-Attention + Cross-Attention + FFN)
      → Linear + Softmax → Output Probabilities
"""

import math
from typing import Optional


# ============================================================
# 工具函数：纯 Python 矩阵运算（教学用）
# ============================================================

def matmul(A: list, B: list) -> list:
    """矩阵乘法 A @ B"""
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = len(B), len(B[0])
    assert cols_A == rows_B, f"维度不匹配: {cols_A} != {rows_B}"
    C = [[sum(A[i][k] * B[k][j] for k in range(cols_A))
          for j in range(cols_B)]
         for i in range(rows_A)]
    return C


def softmax(x: list) -> list:
    """数值稳定的 Softmax"""
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    sum_exp = sum(exp_x)
    return [ei / sum_exp for ei in exp_x]


def transpose(A: list) -> list:
    """矩阵转置"""
    return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]


# ============================================================
# 1. 缩放点积注意力（Scaled Dot-Product Attention）
# 这是 Transformer 的核心计算单元
#
# Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V
# ============================================================

def scaled_dot_product_attention(
    Q: list,  # [seq_len, d_k]
    K: list,  # [seq_len, d_k]
    V: list,  # [seq_len, d_v]
    mask: Optional[list] = None  # 因果掩码（Decoder 使用）
) -> tuple:
    """
    缩放点积注意力
    
    参数解释：
    - Q (Query)：当前 token "想查询什么"
    - K (Key)：每个 token "我能提供什么信息"
    - V (Value)：每个 token "我的实际内容"
    - mask：因果掩码，防止 token 看到未来信息
    
    直觉理解：
    1. QK^T：计算当前 token 与所有 token 的相关性分数
    2. / sqrt(d_k)：缩放防止梯度消失（d_k 越大，点积值越大）
    3. softmax：转为概率分布（注意力权重）
    4. * V：加权求和，聚合相关信息
    """
    d_k = len(Q[0])
    
    # Step 1: QK^T → 注意力分数矩阵
    # 形状：[seq_len, seq_len]，scores[i][j] = q_i 对 k_j 的关注程度
    K_T = transpose(K)
    scores = matmul(Q, K_T)
    
    # Step 2: 缩放
    scale = math.sqrt(d_k)
    scores = [[s / scale for s in row] for row in scores]
    
    # Step 3: 应用掩码（因果掩码：只能看到当前及之前的 token）
    if mask is not None:
        NEG_INF = -1e9
        scores = [
            [s if mask[i][j] else NEG_INF for j, s in enumerate(row)]
            for i, row in enumerate(scores)
        ]
    
    # Step 4: Softmax → 注意力权重
    attention_weights = [softmax(row) for row in scores]
    
    # Step 5: 加权求和 → 输出
    output = matmul(attention_weights, V)
    
    return output, attention_weights


# ============================================================
# 2. 位置编码（Positional Encoding）
# Transformer 本身没有位置感知能力，需要手动注入位置信息
#
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
# ============================================================

def positional_encoding(seq_len: int, d_model: int) -> list:
    """
    生成正弦位置编码
    
    为什么用 sin/cos？
    1. 有界：值域 [-1, 1]，不会影响 embedding 的量级
    2. 唯一性：每个位置都有独特的编码
    3. 相对位置感知：PE(pos+k) 可由 PE(pos) 线性表示
    """
    PE = []
    for pos in range(seq_len):
        pe_row = []
        for i in range(d_model):
            angle = pos / (10000 ** (2 * (i // 2) / d_model))
            if i % 2 == 0:
                pe_row.append(math.sin(angle))
            else:
                pe_row.append(math.cos(angle))
        PE.append(pe_row)
    return PE


# ============================================================
# 3. 前馈网络（Feed-Forward Network）
# 每个位置独立处理（Position-wise FFN）
#
# FFN(x) = max(0, xW_1 + b_1) * W_2 + b_2
# 内部维度通常是 d_model 的 4 倍（信息瓶颈 → 扩展 → 压缩）
# ============================================================

def relu(x: float) -> float:
    return max(0.0, x)


def feed_forward(x: list, W1: list, b1: list, W2: list, b2: list) -> list:
    """
    两层线性变换 + ReLU
    x: [d_model]  → hidden: [d_ff=4*d_model] → output: [d_model]
    """
    # 第一层：扩展
    hidden = [
        relu(sum(x[j] * W1[j][i] for j in range(len(x))) + b1[i])
        for i in range(len(b1))
    ]
    # 第二层：压缩回 d_model
    output = [
        sum(hidden[j] * W2[j][i] for j in range(len(hidden))) + b2[i]
        for i in range(len(b2))
    ]
    return output


# ============================================================
# 4. Layer Normalization
# 对每个 token 的特征维度做归一化（区别于 Batch Norm）
#
# LayerNorm(x) = gamma * (x - mean) / (std + eps) + beta
# ============================================================

def layer_norm(x: list, gamma: list, beta: list, eps: float = 1e-6) -> list:
    """
    Layer Normalization
    
    为什么 LLM 用 LayerNorm 而不是 BatchNorm？
    - BatchNorm 依赖 batch 统计，推理时行为不稳定
    - LayerNorm 在单个样本上计算，更适合变长序列
    """
    mean = sum(x) / len(x)
    variance = sum((xi - mean) ** 2 for xi in x) / len(x)
    std = math.sqrt(variance + eps)
    
    normalized = [(xi - mean) / std for xi in x]
    output = [g * n + b for g, n, b in zip(gamma, normalized, beta)]
    return output


# ============================================================
# 5. 因果掩码（Causal Mask）
# Decoder 在训练时不能看到未来的 token（否则是作弊）
# ============================================================

def create_causal_mask(seq_len: int) -> list:
    """
    创建下三角因果掩码
    
    mask[i][j] = True 表示位置 i 可以关注位置 j
    即：只能看到自己和之前的 token
    
    示例（seq_len=4）:
    [[T, F, F, F],
     [T, T, F, F],
     [T, T, T, F],
     [T, T, T, T]]
    """
    return [[j <= i for j in range(seq_len)] for i in range(seq_len)]


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    # 模拟一个微型 Transformer 的注意力计算
    seq_len = 4
    d_k = 8

    print("=== 位置编码 ===")
    PE = positional_encoding(seq_len, d_k)
    for i, pe in enumerate(PE):
        print(f"  pos={i}: [{', '.join(f'{v:.3f}' for v in pe[:4])}...]")

    print("\n=== 因果掩码 ===")
    mask = create_causal_mask(seq_len)
    for row in mask:
        print(" ", ["✓" if m else "✗" for m in row])

    print("\n=== Scaled Dot-Product Attention ===")
    # 随机初始化 Q、K、V（实际是 embedding + 线性变换）
    import random
    random.seed(42)
    Q = [[random.gauss(0, 0.1) for _ in range(d_k)] for _ in range(seq_len)]
    K = [[random.gauss(0, 0.1) for _ in range(d_k)] for _ in range(seq_len)]
    V = [[random.gauss(0, 0.1) for _ in range(d_k)] for _ in range(seq_len)]

    output, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
    print(f"  输入形状: Q/K/V = [{seq_len}, {d_k}]")
    print(f"  输出形状: [{len(output)}, {len(output[0])}]")
    print(f"  注意力权重（第 3 个 token）: {[f'{w:.3f}' for w in weights[2]]}")
    print("  (第 3 个 token 只能关注前 3 个位置，最后一个权重为 0)")
