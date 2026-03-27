# 向量数据库对比实验

## 三大向量库

| 特性 | Chroma | Weaviate | Pinecone |
|------|--------|----------|----------|
| 部署 | 本地 | 自托管/云 | 云端 SaaS |
| 规模 | < 100 万 | 亿级 | 亿级 |
| 索引算法 | HNSW | HNSW | HNSW |
| 混合搜索 | ❌ | ✅ | ✅ |
| 费用 | 免费 | 开源免费 | 按用量 |

## 实验方案
相同数据集、相同 embedding，测试三库的：检索速度、准确率、内存占用

## 文件
- `chroma_demo.py`、`weaviate_demo.py`、`pinecone_demo.py`
- `benchmark.py`：对比测试脚本
