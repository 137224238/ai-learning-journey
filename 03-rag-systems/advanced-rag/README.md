# Advanced RAG

## 核心技术
- HyDE：假设文档嵌入（+召回率）
- Hybrid Search：向量 + BM25（RRF 融合）
- Cross-Encoder Reranking（+精度）
- Self-RAG：自适应检索决策

## 效果提升
Naive RAG Faithfulness 0.72 → Advanced RAG 0.83（+15%）

## 文件
- `hyde.py`、`hybrid_search.py`、`reranker.py`、`self_rag.py`
