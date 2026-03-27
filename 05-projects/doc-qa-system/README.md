# 企业文档问答系统

## 项目简介
基于 RAG 的企业内部文档问答系统，支持 PDF/Word/TXT 文档上传和自然语言查询。

## 技术栈
- **后端**：FastAPI + Python 3.10+
- **向量库**：ChromaDB（本地）/ Pinecone（云端）
- **Embedding**：OpenAI text-embedding-3-small
- **生成模型**：GPT-4o-mini
- **文档解析**：PyPDF2 + python-docx

## 快速开始
```bash
pip install -r requirements.txt
cp .env.example .env  # 填入 OPENAI_API_KEY
python doc_qa_system.py
# 访问 http://localhost:8000/docs
```

## 系统架构
```
用户 → API → 文档处理 → 向量索引
              查询处理 → 检索 → 重排序 → LLM 生成 → 带引用的答案
```

## 性能指标（目标）
- RAGAS Faithfulness > 0.80
- Answer Relevancy > 0.75
- P99 延迟 < 3s

## 项目亮点（面试用）
1. 实现了 HyDE + Hybrid Search 提升检索质量
2. 加入了 Cross-Encoder Reranker
3. 有完整的 RAGAS 评估报告
