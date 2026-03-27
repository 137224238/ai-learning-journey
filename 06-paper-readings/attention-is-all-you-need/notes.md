# 论文精读：Attention Is All You Need

> Vaswani et al., Google Brain, NeurIPS 2017  
> 引用次数：100,000+（AI 史上最具影响力论文之一）

---

## 📌 核心贡献

1. **提出纯注意力架构**：完全摒弃 RNN/CNN，仅用注意力机制
2. **多头注意力**：并行关注多个子空间
3. **位置编码**：用正弦函数注入位置信息
4. **可并行训练**：解决 RNN 无法并行的问题

---

## 🏗️ 架构解析

### Encoder

```
Input Embedding + Positional Encoding
        ↓
┌─────────────────────────────────┐
│  Multi-Head Self-Attention      │
│  Add & Layer Norm               │
│  Feed-Forward Network (d_ff=2048) │
│  Add & Layer Norm               │
└─────────────────────────────────┘
   × N（原文 N=6）
        ↓
   Encoder Output（shape: [batch, seq_len, 512]）
```

### Decoder

```
Target Embedding + Positional Encoding
        ↓
┌───────────────────────────────────────┐
│  Masked Multi-Head Self-Attention     │  ← 因果掩码
│  Add & Layer Norm                     │
│  Multi-Head Cross-Attention           │  ← Q from decoder, K/V from encoder
│  Add & Layer Norm                     │
│  Feed-Forward Network                 │
│  Add & Layer Norm                     │
└───────────────────────────────────────┘
   × N
        ↓
   Linear + Softmax → Output Probabilities
```

---

## 🔑 关键公式

### Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

**为什么除以 $\sqrt{d_k}$？**
当 $d_k$ 较大时，点积的方差增大，Softmax 梯度变小（趋向饱和），
除以 $\sqrt{d_k}$ 将方差标准化为 1，保持梯度流动。

### Multi-Head Attention

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1,...,\text{head}_h)W^O$$

$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**为什么多头？**  
不同的头可以学习到不同类型的依赖关系（语法/语义/指代等）

---

## 💡 关键洞察

### 1. 为什么 Self-Attention 比 RNN 更好？

| 特性 | RNN | Self-Attention |
|------|-----|----------------|
| 并行计算 | ❌（顺序依赖） | ✅ |
| 长距离依赖 | 难（梯度消失） | 容易（直接连接） |
| 计算复杂度 | O(n·d²) | O(n²·d) |
| 路径长度 | O(n) | O(1) |

> **结论**：序列长度 n 较小时，Self-Attention 完胜 RNN

### 2. 位置编码的设计

$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}})$$

**为什么这样设计？**
- 每个位置编码唯一
- 编码在 [-1, 1] 之间，不破坏 embedding 尺度
- 理论上可以外推到训练时未见的长度
- $PE_{pos+k}$ 可由 $PE_{pos}$ 线性表示

### 3. 残差连接的重要性

每个子层：$\text{LayerNorm}(x + \text{SubLayer}(x))$

- 允许梯度直接流动（解决深层网络训练问题）
- 模型可以轻松学习"不做任何变换"（恒等映射）

---

## 📊 实验结果

| 任务 | BLEU | 训练时间 |
|------|------|----------|
| WMT 2014 EN-DE | 28.4 | 3.5 天（8×P100） |
| WMT 2014 EN-FR | 41.0 | 5 天（8×P100） |

超越此前所有模型，且训练更快。

---

## 🤔 批判性思考

### 局限性

1. **二次复杂度**：Self-Attention 是 O(n²)，长序列代价高
   - 解决方案：Longformer、FlashAttention、Mamba
   
2. **无法处理超长序列**：原始 Transformer 位置编码无法外推
   - 解决方案：RoPE、ALiBi
   
3. **位置编码固定**：不能动态适应不同任务
   - 解决方案：可学习位置编码（BERT、GPT）

### 被后续工作改进的部分

| 原版 | 现代改进 |
|------|---------|
| Sinusoidal PE | RoPE（旋转位置编码）in LLaMA |
| Post-LayerNorm | Pre-LayerNorm（训练更稳定） |
| Encoder-Decoder | Decoder-Only（GPT 系列） |
| Learned attention | Flash Attention（IO 感知实现） |

---

## 🔗 后续阅读路径

1. **BERT**（2018）：双向 Encoder，预训练语言模型
2. **GPT-1**（2018）：单向 Decoder，自监督预训练
3. **GPT-2/3**（2019/2020）：规模定律
4. **Flash Attention**（2022）：I/O 感知的高效注意力
5. **LLaMA**（2023）：开源高效 LLM，RoPE + Pre-LN

---

## ✍️ 个人笔记

> 记录阅读过程中的问题和理解

- **问题**：为什么选 $d_{model}=512$，$h=8$，所以 $d_k=64$？
  - **理解**：超参数，通过实验选定。$h=8$ 头时 $d_k=64$ 效果最好。

- **问题**：FFN 的 $d_{ff}=2048$ 为什么是 4 倍 $d_{model}$？
  - **理解**：类似"瓶颈"设计，先扩展再压缩，增加非线性表达能力。

- **待验证**：自己实现一个小型 Transformer 在玩具任务上训练。
  → 见 `02-llm-core/transformer-from-scratch/`

---

*阅读日期：2025-03-27 | 精读耗时：4 小时*
