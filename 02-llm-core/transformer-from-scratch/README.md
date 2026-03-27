# 手写 Transformer

## 学习目标
从零实现 Transformer，真正理解每一行代码背后的含义。

## 核心组件

```
Transformer
├── Embedding Layer          # Token → 向量空间
├── Positional Encoding      # 注入位置信息（sin/cos）
├── Encoder Stack
│   └── EncoderLayer × N
│       ├── Multi-Head Self-Attention  # 全局依赖捕获
│       ├── Add & LayerNorm            # 残差连接
│       ├── Feed-Forward Network       # 非线性变换
│       └── Add & LayerNorm
└── Decoder Stack
    └── DecoderLayer × N
        ├── Masked Multi-Head Attention   # 因果注意力
        ├── Add & LayerNorm
        ├── Cross-Attention               # 关注 Encoder 输出
        ├── Add & LayerNorm
        ├── Feed-Forward Network
        └── Add & LayerNorm
```

## 关键设计决策

| 设计 | 原因 |
|------|------|
| Scaled Dot-Product | 防止 d_k 过大导致梯度消失 |
| Multi-Head | 在不同子空间捕获不同类型的关系 |
| Residual Connection | 解决深层网络梯度消失 |
| Layer Norm | 稳定训练，适合变长序列 |
| Sinusoidal PE | 可泛化到训练时未见的序列长度 |

## 文件说明

```
transformer-from-scratch/
├── transformer.py      # 核心组件（本文件）
├── multi_head.py       # 多头注意力
├── full_model.py       # 完整模型组装
├── train_demo.py       # 玩具任务训练演示
└── README.md
```

## 学习建议
1. 先读 `transformer.py` 理解基本组件
2. 在纸上画出数据流动的形状变化
3. 用 `print(tensor.shape)` 在每一步验证维度
4. 与 Karpathy 的 nanoGPT 对比阅读

## 参考资源
- 📄 原论文：[Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- 🎥 [3Blue1Brown 可视化](https://www.youtube.com/watch?v=wjZofJX0v4M)
- 💻 [Karpathy nanoGPT](https://github.com/karpathy/nanoGPT)
