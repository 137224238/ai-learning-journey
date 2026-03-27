# 模型评估框架

## 指标
- BLEU / ROUGE：文本生成质量
- RAGAS：RAG 专用（Faithfulness/Relevancy/Precision/Recall）
- LLM-as-Judge：用强模型评估弱模型

## 文件
- `metrics.py`：BLEU、ROUGE 实现
- `llm_judge.py`：LLM-as-Judge 框架
- `ragas_eval.py`：RAGAS 流水线
