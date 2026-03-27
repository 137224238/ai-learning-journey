# ML 面试题精选 + 解析

> 高频题目，附我的答题框架

---

## 第一部分：Transformer & LLM 基础

### Q1: Attention 机制中为什么要除以 sqrt(d_k)？

**标准答案（30秒版）**：
防止点积值过大导致 Softmax 梯度消失。

**深度解释**：
- 假设 Q、K 的每个维度服从均值 0、方差 1 的分布
- 点积 $QK^T$ 的方差为 $d_k$（$d_k$ 个随机变量的和）
- 除以 $\sqrt{d_k}$ 将方差归一化为 1
- 方差大 → Softmax 输出趋向 one-hot → 梯度趋向 0 → 学习停止

```python
# 验证：d_k 对 Softmax 梯度的影响
import math

def softmax(x):
    e = [math.exp(xi) for xi in x]
    s = sum(e)
    return [ei/s for ei in e]

# d_k=64 时，点积可能很大
scores_large = [8.0, 2.0, 1.0, 0.5]   # 相当于 d_k=64 时的点积
scores_small = [1.0, 0.25, 0.125, 0.0625]  # 除以 sqrt(64)=8 后

print("未缩放:", softmax(scores_large))   # [0.997, 0.002, 0.001, ...]
print("缩放后:", softmax(scores_small))   # [0.37, 0.24, 0.21, ...] 更平滑
```

---

### Q2: 为什么 LayerNorm 比 BatchNorm 更适合 LLM？

| 维度 | BatchNorm | LayerNorm |
|------|-----------|-----------|
| 归一化方向 | 沿 Batch 维度 | 沿 Feature 维度 |
| 推理行为 | 需要 running stats | 完全一致 |
| 变长序列 | ❌ 处理困难 | ✅ 天然支持 |
| 小 batch size | ❌ 统计不稳定 | ✅ 不依赖 batch |
| 适用场景 | CV（图像） | NLP（序列） |

**一句话**：BatchNorm 依赖 batch 统计，变长序列和自回归推理时行为不一致；LayerNorm 在单个样本内归一化，更稳定。

---

### Q3: GPT 和 BERT 的区别？什么时候用哪个？

```
BERT（双向 Encoder）:
- 目标：理解（分类、NER、问答）
- 预训练：MLM（Masked LM）+ NSP
- 看全文 → 适合理解任务
- 代表：bert-base, RoBERTa, DeBERTa

GPT（单向 Decoder）:
- 目标：生成（对话、写作、代码）
- 预训练：CLM（Causal LM，预测下一个 token）
- 只看左边 → 适合生成任务
- 代表：GPT-4, LLaMA, Qwen

选型规则：
- 文本分类/NER → BERT
- 语义相似度 → Sentence-BERT
- 对话/生成 → GPT
- RAG Embedding → 专门的 Embedding 模型（BGE, text-embedding-3）
```

---

### Q4: 什么是 Hallucination？如何缓解？

**定义**：模型生成了听起来合理但实际上不准确或虚假的内容。

**根本原因**：
- LLM 是概率模型，生成的是"看起来合理"的 token，而非"事实正确"的内容
- 训练数据中存在错误信息
- 模型对于不知道的问题倾向于"创造"答案

**缓解方案**：

```
1. RAG（检索增强）
   - 将外部知识注入上下文，限制模型只基于检索内容回答
   - 效果：幻觉率降低 30-60%

2. 指令微调（Instruction Tuning）
   - 训练模型在不确定时说"我不知道"
   - 效果：提升可靠性，但需要高质量数据

3. Chain-of-Thought
   - 让模型一步步推理，错误更容易被发现

4. 自我一致性（Self-Consistency）
   - 多次生成，投票选择一致的答案

5. 事后验证
   - 用 NLI 模型检查答案是否被上下文支持
   - 幻觉检测模型（FactScore, RAGAS faithfulness）
```

---

## 第二部分：RAG 系统

### Q5: RAG 的评估指标有哪些？

**RAGAS 框架（最主流）**：

```python
# RAGAS 四大指标

# 1. Faithfulness（忠实度）
# 答案中的每个声明是否都可以从检索到的上下文中推导出
# 公式：支持的声明数 / 总声明数
faithfulness = supported_claims / total_claims  # 越高越好，满分=1

# 2. Answer Relevancy（答案相关性）
# 答案与问题的相关程度
# 方法：用答案反向生成问题，与原问题的 embedding 相似度
answer_relevancy = cosine_sim(generated_questions_embedding, original_question_embedding)

# 3. Context Precision（上下文精确率）
# 检索到的文档块中，真正有用的比例
context_precision = useful_chunks / total_retrieved_chunks

# 4. Context Recall（上下文召回率）
# 真正有用的信息有多少比例被检索到
context_recall = retrieved_relevant_info / total_relevant_info
```

**面试回答模板**：
"我们用 RAGAS 评估 RAG 系统，关注四个指标：Faithfulness（答案是否忠实于检索内容，防止幻觉）、Answer Relevancy（答案是否回答了问题）、Context Precision（检索是否精准）、Context Recall（是否检索全面）。同时加入端到端的人工评估作为 Ground Truth 校验。"

---

### Q6: Naive RAG vs Advanced RAG 的区别？

```
Naive RAG（基础）：
Query → Embedding → ANN 检索 → Top-K → LLM 生成

问题：
- 检索召回率不高（单路检索）
- 没有质量过滤
- 上下文窗口利用率低

Advanced RAG（高级）：

Pre-retrieval:
├── 查询重写（Query Rewriting）
├── 查询扩展（Multi-Query）
└── HyDE（生成假设文档）

Retrieval:
├── 多路检索（向量 + BM25 混合）
└── 父文档检索（Parent-Child Chunks）

Post-retrieval:
├── 重排序（Cross-Encoder Reranker）
├── 相关性过滤
└── 上下文压缩（减少噪音）

Generation:
├── Self-RAG（自我评估是否需要检索）
└── RAPTOR（递归摘要 + 检索）
```

---

## 第三部分：Agent 系统

### Q7: ReAct 和 Plan-and-Execute 的区别？

```
ReAct（交错推理）：
Thought → Action → Observation → Thought → Action → ...
- 边思考边行动
- 适合简单、线性任务
- 可以根据 Observation 动态调整

Plan-and-Execute（先规划后执行）：
Plan: [Step1, Step2, Step3]
Execute: Step1 → Step2 → Step3
Replanning（如果需要）

- 先整体规划，再逐步执行
- 适合复杂、多步骤任务
- 更容易并行化（无依赖的步骤可同时执行）
- LangGraph 实现更自然
```

---

## 第四部分：Python & 工程

### Q8: Python GIL 对 LLM 推理有什么影响？

**GIL（全局解释器锁）**：同一时刻只有一个线程可以执行 Python 字节码。

**对 LLM 的影响**：
- CPU 密集型（PyTorch 计算）：GIL 释放（C 扩展层面执行），影响小
- I/O 密集型（API 调用）：使用 `asyncio` 协程，比多线程更高效

**实战策略**：
```python
# LLM API 调用 → 使用 asyncio（I/O 密集）
async def batch_process(prompts):
    async with asyncio.Semaphore(10):  # 控制并发
        tasks = [call_llm(p) for p in prompts]
        return await asyncio.gather(*tasks)

# 文档解析（CPU 密集）→ 使用 multiprocessing
from concurrent.futures import ProcessPoolExecutor
with ProcessPoolExecutor() as executor:
    results = list(executor.map(parse_document, file_paths))
```

---

## 自测清单

- [ ] 能不看资料解释 Attention 的完整计算过程
- [ ] 能说清楚 BERT 和 GPT 的预训练目标差异
- [ ] 能设计一个完整的 RAG 评估流程
- [ ] 能解释 vLLM 为什么比 naive 推理快 3-5x
- [ ] 能写出 ReAct Agent 的伪代码
- [ ] 能设计 LLM 应用的监控方案
