"""
Naive RAG：从原理到实现
基础 RAG 流水线：Indexing → Retrieval → Generation
"""

import math
import re
from typing import List, Tuple, Dict, Any
from dataclasses import dataclass, field


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Document:
    """文档块"""
    id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """检索结果"""
    document: Document
    score: float
    rank: int


# ============================================================
# 1. 文档处理：分块（Chunking）
# ============================================================

class TextSplitter:
    """
    文本分块器
    
    核心问题：如何分块？
    - 太短：缺乏上下文，语义不完整
    - 太长：引入噪音，超出 embedding 模型上下文
    - 最佳实践：chunk_size=512, overlap=50
    """
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_by_chars(self, text: str, doc_id: str = "doc") -> List[Document]:
        """按字符数分块（带重叠）"""
        chunks = []
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunk_text = text[start:end]
            
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_idx}",
                content=chunk_text,
                metadata={"start": start, "end": end, "chunk_idx": chunk_idx}
            ))
            
            chunk_idx += 1
            start += self.chunk_size - self.chunk_overlap  # 步长 = 块大小 - 重叠
        
        return chunks
    
    def split_by_sentences(self, text: str, doc_id: str = "doc") -> List[Document]:
        """按句子分块（更自然的语义边界）"""
        # 简单句子分割（生产环境用 spaCy 或 NLTK）
        sentences = re.split(r'(?<=[。！？.!?])\s*', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence
            else:
                if current_chunk:
                    chunks.append(Document(
                        id=f"{doc_id}_chunk_{chunk_idx}",
                        content=current_chunk,
                        metadata={"chunk_idx": chunk_idx}
                    ))
                    chunk_idx += 1
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(Document(
                id=f"{doc_id}_chunk_{chunk_idx}",
                content=current_chunk,
                metadata={"chunk_idx": chunk_idx}
            ))
        
        return chunks


# ============================================================
# 2. 向量化：TF-IDF（教学用，生产用 Sentence Transformers）
# ============================================================

class TFIDFVectorizer:
    """
    TF-IDF 向量化器（理解向量检索原理）
    
    生产环境替换为：
    - sentence-transformers (BERT-based)
    - OpenAI text-embedding-3-small
    - BGE / M3E（中文效果好）
    """
    
    def __init__(self):
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.is_fitted = False
    
    def tokenize(self, text: str) -> List[str]:
        """简单分词（生产用 jieba 或 spaCy）"""
        text = text.lower()
        # 中文：字级别分词；英文：词级别
        tokens = list(text) if any('\u4e00' <= c <= '\u9fff' for c in text) \
                 else re.findall(r'\b\w+\b', text)
        return tokens
    
    def fit(self, documents: List[str]):
        """在语料库上拟合，计算 IDF"""
        # 建立词表
        all_tokens = set()
        doc_token_sets = []
        for doc in documents:
            tokens = set(self.tokenize(doc))
            all_tokens |= tokens
            doc_token_sets.append(tokens)
        
        self.vocab = {token: idx for idx, token in enumerate(sorted(all_tokens))}
        
        # 计算 IDF：log((N + 1) / (df + 1)) + 1（平滑）
        N = len(documents)
        for token, idx in self.vocab.items():
            df = sum(1 for token_set in doc_token_sets if token in token_set)
            self.idf[token] = math.log((N + 1) / (df + 1)) + 1
        
        self.is_fitted = True
    
    def transform(self, text: str) -> List[float]:
        """将文本转为 TF-IDF 向量"""
        assert self.is_fitted, "请先调用 fit()"
        
        tokens = self.tokenize(text)
        token_counts: Dict[str, int] = {}
        for t in tokens:
            token_counts[t] = token_counts.get(t, 0) + 1
        
        vec = [0.0] * len(self.vocab)
        for token, count in token_counts.items():
            if token in self.vocab:
                tf = count / max(len(tokens), 1)  # 归一化词频
                idf = self.idf.get(token, 0)
                vec[self.vocab[token]] = tf * idf
        
        # L2 归一化（便于余弦相似度计算）
        norm = math.sqrt(sum(v ** 2 for v in vec)) + 1e-10
        return [v / norm for v in vec]


# ============================================================
# 3. 向量存储（内存版）
# ============================================================

class SimpleVectorStore:
    """
    内存向量存储
    
    生产替换为：
    - Chroma（轻量本地）
    - Weaviate（功能丰富）
    - Pinecone（云端托管）
    - pgvector（PostgreSQL 扩展）
    """
    
    def __init__(self):
        self.documents: List[Document] = []
        self.vectors: List[List[float]] = []
    
    def add(self, document: Document, vector: List[float]):
        self.documents.append(document)
        self.vectors.append(vector)
    
    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 5
    ) -> List[RetrievalResult]:
        """余弦相似度搜索"""
        if not self.vectors:
            return []
        
        # 计算所有文档的余弦相似度
        scores = []
        for i, vec in enumerate(self.vectors):
            dot = sum(a * b for a, b in zip(query_vector, vec))
            scores.append((i, dot))  # 已归一化，点积 = 余弦相似度
        
        # 排序取 top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for rank, (idx, score) in enumerate(scores[:top_k]):
            results.append(RetrievalResult(
                document=self.documents[idx],
                score=score,
                rank=rank + 1
            ))
        
        return results


# ============================================================
# 4. 完整 RAG 流水线
# ============================================================

class NaiveRAG:
    """
    基础 RAG 系统
    
    Pipeline:
    1. [离线] 文档分块 → 向量化 → 存储
    2. [在线] 查询向量化 → 检索 → 生成
    """
    
    def __init__(self, chunk_size: int = 200, top_k: int = 3):
        self.splitter = TextSplitter(chunk_size=chunk_size, chunk_overlap=30)
        self.vectorizer = TFIDFVectorizer()
        self.store = SimpleVectorStore()
        self.top_k = top_k
        self._raw_documents: List[str] = []
    
    # --- 离线索引阶段 ---
    
    def add_document(self, text: str, doc_id: str = "doc"):
        """添加文档到知识库"""
        self._raw_documents.append(text)
        # 返回分块结果，等待批量 fit 后 transform
        return self.splitter.split_by_sentences(text, doc_id)
    
    def build_index(self, documents: Dict[str, str]):
        """
        批量构建索引
        
        Args:
            documents: {doc_id: text} 字典
        """
        all_chunks = []
        for doc_id, text in documents.items():
            chunks = self.splitter.split_by_sentences(text, doc_id)
            all_chunks.extend(chunks)
        
        # 用所有块的文本拟合 TF-IDF
        self.vectorizer.fit([chunk.content for chunk in all_chunks])
        
        # 向量化并存储
        for chunk in all_chunks:
            vec = self.vectorizer.transform(chunk.content)
            self.store.add(chunk, vec)
        
        print(f"✅ 索引构建完成：{len(all_chunks)} 个文档块")
    
    # --- 在线查询阶段 ---
    
    def retrieve(self, query: str) -> List[RetrievalResult]:
        """检索相关文档块"""
        query_vec = self.vectorizer.transform(query)
        return self.store.similarity_search(query_vec, self.top_k)
    
    def generate(self, query: str, context: str) -> str:
        """
        生成回答（真实场景替换为 LLM API 调用）
        """
        # 模拟生成（真实：调用 OpenAI/Claude API）
        return (
            f"[模拟 LLM 生成]\n"
            f"基于检索到的上下文，回答问题：'{query}'\n\n"
            f"上下文摘要：{context[:100]}...\n\n"
            f"注：生产环境请替换为真实 LLM API 调用"
        )
    
    def query(self, question: str) -> Dict[str, Any]:
        """
        完整查询流程
        
        Returns:
            {
                "answer": str,
                "sources": List[RetrievalResult],
                "context": str
            }
        """
        # Step 1: 检索
        results = self.retrieve(question)
        
        if not results:
            return {"answer": "知识库中没有相关信息。", "sources": [], "context": ""}
        
        # Step 2: 构建上下文
        context_parts = []
        for result in results:
            context_parts.append(
                f"[来源 {result.rank}（相关度: {result.score:.3f}）]\n"
                f"{result.document.content}"
            )
        context = "\n\n".join(context_parts)
        
        # Step 3: 生成答案
        answer = self.generate(question, context)
        
        return {
            "answer": answer,
            "sources": results,
            "context": context
        }


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    # 构建知识库
    rag = NaiveRAG(chunk_size=150, top_k=3)
    
    documents = {
        "transformer_paper": """
        注意力机制是 Transformer 架构的核心。
        Transformer 由编码器和解码器两部分组成。
        自注意力机制使模型能够关注输入序列中的不同位置。
        多头注意力通过并行计算多个注意力头来捕获不同类型的依赖关系。
        位置编码用于向模型提供序列中每个 token 的位置信息。
        """,
        "rag_intro": """
        检索增强生成（RAG）结合了检索系统和生成模型的优点。
        RAG 的基本流程包括文档索引、查询检索和答案生成三个阶段。
        向量数据库用于存储文档的向量表示，支持语义相似度搜索。
        RAG 可以有效减少大语言模型的幻觉问题。
        通过检索外部知识，RAG 使模型能够回答训练数据之外的问题。
        """
    }
    
    rag.build_index(documents)
    
    # 查询
    questions = [
        "什么是注意力机制？",
        "RAG 如何工作？",
        "什么是幻觉问题？"
    ]
    
    for q in questions:
        print(f"\n{'='*50}")
        print(f"问题：{q}")
        result = rag.query(q)
        print(f"\n检索结果（Top {len(result['sources'])}）：")
        for src in result['sources']:
            print(f"  [{src.rank}] 相关度 {src.score:.3f}: {src.document.content[:60]}...")
        print(f"\n回答：{result['answer']}")
