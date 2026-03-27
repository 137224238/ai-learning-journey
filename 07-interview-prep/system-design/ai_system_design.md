# AI 系统设计题库

> 面试中最高频的 AI/LLM 系统设计题，附详细解题框架

---

## 解题框架（STAR-D）

1. **S**cope（澄清需求）：QPS、延迟要求、数据规模
2. **T**rade-offs（权衡分析）：准确率 vs 速度、成本 vs 性能
3. **A**rchitecture（系统架构）：画图、组件设计
4. **R**eliability（可靠性）：容错、监控、降级
5. **D**eep Dive（深入细节）：瓶颈在哪里，如何优化

---

## 题目 1：设计一个企业级 RAG 问答系统

**常见变体**：
- 设计一个能回答公司内部文档问题的 AI 助手
- 设计一个法律文件查询系统

### 需求澄清

```
功能性需求：
- 支持 PDF/Word/TXT 文档上传
- 支持自然语言查询
- 答案需要引用来源
- 支持多语言（中英文）

非功能性需求：
- QPS：100 查询/秒
- 延迟：P99 < 3 秒
- 文档规模：100 万个文档块
- 准确率：> 85%（人工评估）
```

### 系统架构

```
用户
 │
 ▼
API Gateway（限流、认证）
 │
 ├──────────────────────────────────┐
 │                                  │
 ▼                                  ▼
[离线索引流水线]               [在线查询服务]
文档上传 → 解析 → 分块         查询 → Embedding
         → Embedding           → 向量检索
         → 向量库存储           → 重排序
         → 元数据存储           → LLM 生成
                               → 响应缓存
```

### 关键组件设计

**文档处理流水线**
```
文档上传（S3/OSS）
    → 消息队列（Kafka）
    → 文档解析 Worker（PDFMiner/Unstructured）
    → 分块策略：Recursive Text Splitting
    → Embedding（text-embedding-3-small，批量调用）
    → 写入向量库（Weaviate/Pinecone）+ 元数据库（PostgreSQL）
```

**在线查询链路**
```
用户查询（latency budget: 3000ms）
  ├─ 查询重写（LLM, 200ms）
  ├─ 查询 Embedding（API, 100ms）
  ├─ 向量检索 Top20（向量库, 50ms）
  ├─ 重排序 → Top5（Cross-Encoder, 300ms）
  ├─ LLM 生成（GPT-4, 1500ms，流式返回）
  └─ 缓存结果（Redis, TTL=1h）
```

### 权衡分析

| 决策 | 选项 A | 选项 B | 选择理由 |
|------|--------|--------|---------|
| 向量库 | Pinecone（SaaS） | Weaviate（自托管） | 初期选 Pinecone 快，后期迁移 Weaviate |
| Embedding | OpenAI | 本地模型（BGE） | 数据敏感则选本地 |
| 分块 | 固定大小 | 语义分块 | 语义分块准确率 +5%，但速度慢 3x |
| 缓存 | 精确匹配 | 语义缓存 | 语义缓存命中率高，但实现复杂 |

### 优化方向

1. **HyDE**：用 LLM 生成假设文档辅助检索（+3-5% 准确率）
2. **查询扩展**：多角度改写查询（+2-3% 召回率）
3. **Re-ranking**：Cross-Encoder 精排（+5% 精度）
4. **语义缓存**：相似问题复用答案（降低 40% 成本）

---

## 题目 2：设计一个 LLM 推理服务

**场景**：你的公司要自托管 LLaMA-70B，服务 10,000 个并发用户。

### 核心挑战

1. **GPU 资源昂贵**：70B 模型需要 4x A100（80GB）或 8x A100（40GB）
2. **KV Cache 显存**：长序列的 KV Cache 占显存大
3. **批量推理**：如何最大化 GPU 利用率

### 架构设计

```
负载均衡（Nginx/AWS ALB）
    │
    ▼
推理服务集群
├── vLLM Server 1（4x A100）  ← 主要推理引擎
├── vLLM Server 2（4x A100）
└── vLLM Server 3（4x A100）
    │
路由策略：
├─ 短请求（<500 tokens）→ 优先队列，快速响应
└─ 长请求（>2000 tokens）→ 批量队列，高吞吐
```

### 关键技术

**vLLM 的 PagedAttention**
- 问题：KV Cache 碎片化，显存利用率只有 50%
- 解决：像操作系统分页内存一样管理 KV Cache
- 效果：显存利用率 → 90%，吞吐量 3-5x

**连续批处理（Continuous Batching）**
- 问题：静态批处理等待最长序列，GPU 浪费
- 解决：完成一个请求立即插入新请求
- 效果：GPU 利用率从 40% → 80%

### 成本估算

```
70B 模型，每个 token 约 1ms（A100）

月成本（AWS p4d.24xlarge, $32/hr）：
- 3 台机器 × $32/hr × 720hr = $69,120/月

吞吐量：
- 每台机器约 500 tokens/s
- 3 台 = 1500 tokens/s
- 对话平均 500 tokens → 3 QPS × 3 = 9 对话/s
- 9 QPS × 86400s = 777,600 对话/天
```

---

## 题目 3：设计一个 AI 代码审查系统

**需求**：在 GitHub PR 创建时，自动进行 AI 代码审查。

### 架构

```
GitHub Webhook → API Gateway
                    │
                    ▼
              任务队列（Celery + Redis）
                    │
                    ▼
              代码审查 Worker
              ├─ 提取 Diff（GitHub API）
              ├─ 静态分析（AST 解析）
              ├─ LLM 审查（分模块并发）
              └─ 整合结果 → GitHub API 评论
```

### LLM Prompt 设计

```python
REVIEW_PROMPT = """
你是一位资深工程师，请对以下代码变更进行审查：

文件：{filename}
变更：
```diff
{diff}
```

请从以下维度评估（JSON 格式）：
1. bugs: 明显的 bug 或逻辑错误
2. security: 安全漏洞（SQL 注入、XSS 等）
3. performance: 性能问题
4. style: 代码风格（参考项目规范）
5. suggestions: 改进建议

严重级别：critical/warning/info
"""
```

---

## 题目 4：设计 LLM 评估系统

**需求**：你需要评估和持续监控你的 LLM 应用的质量。

### 评估维度

```
自动评估（可扩展）           人工评估（Golden Set）
├── BLEU/ROUGE（文本相似度）  ├── 准确性
├── 幻觉检测（NLI 模型）     ├── 有用性
├── 相关性（embedding 相似度）└── 无害性
├── 引用准确率（RAG 专用）
└── 延迟/成本

持续监控
├── 用户反馈（👍/👎）
├── 答案分布变化
└── A/B 测试
```

---

## 通用面试技巧

1. **先问需求**，不要直接画图
2. **说清楚权衡**，没有银弹
3. **数字要具体**：QPS、延迟、存储、成本
4. **从简单开始**，然后迭代优化
5. **提前识别瓶颈**：哪里最慢？哪里最贵？

## 高频考点

- [ ] 向量检索的近似算法（HNSW vs IVF）
- [ ] Embedding 模型的选型（维度、语言、速度）
- [ ] LLM 推理优化（量化、KV Cache、批处理）
- [ ] RAG 评估指标（RAGAS：faithfulness, relevancy）
- [ ] 成本优化（缓存、路由、模型选择）
