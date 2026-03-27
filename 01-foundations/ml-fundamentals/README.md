# ML 核心概念

## 学习目标
深刻理解 ML 的数学基础，为理解 LLM 的训练过程打下根基。

## 核心模块

### 1. 损失函数（Loss Functions）
- 交叉熵损失：为什么 LLM 用它
- KL 散度：RLHF 中的 KL 惩罚项
- 对比损失：Embedding 模型训练

### 2. 优化器（Optimizers）
- SGD → Momentum → Adam → AdamW
- 学习率调度：Warmup + Cosine Decay
- 梯度裁剪：防止梯度爆炸

### 3. 正则化（Regularization）
- Dropout：训练 vs 推理的区别
- Layer Normalization vs Batch Normalization
- Weight Decay

### 4. 反向传播（Backpropagation）
- 计算图与自动微分
- 手写一个简单的自动微分引擎

## 文件说明

```
ml-fundamentals/
├── loss_functions.py     # 各种损失函数实现与可视化
├── optimizers.py         # 优化器从零实现
├── backprop_engine.py    # 手写微型自动微分引擎
├── regularization.py     # 正则化技术对比
└── README.md
```

## 关键公式

**交叉熵损失**
$$L = -\sum_{i} y_i \log(\hat{y}_i)$$

**Adam 更新规则**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$
$$\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$$

## 学习资源
- [Andrej Karpathy - micrograd](https://github.com/karpathy/micrograd) ⭐
- [CS231n 课程笔记](https://cs231n.github.io/)
- 《Deep Learning》花书 第 6-8 章
