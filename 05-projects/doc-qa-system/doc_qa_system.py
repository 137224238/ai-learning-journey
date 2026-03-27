"""
企业文档问答系统
项目结构：FastAPI + RAG + ChromaDB + OpenAI

技术栈：
- 后端：FastAPI
- 向量库：ChromaDB（本地）
- Embedding：OpenAI text-embedding-3-small
- 生成：GPT-4o-mini
- 文档解析：PyPDF2 + python-docx
"""

from __future__ import annotations

import os
import hashlib
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================
# 配置
# ============================================================

@dataclass
class Config:
    """系统配置"""
    # LLM
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    
    # RAG
    chunk_size: int = 512
    chunk_overlap: int = 64
    top_k: int = 5
    
    # 向量库
    chroma_persist_dir: str = "./chroma_db"
    collection_name: str = "enterprise_docs"
    
    # 服务
    host: str = "0.0.0.0"
    port: int = 8000


# ============================================================
# 数据模型（Pydantic，FastAPI 用）
# ============================================================

# 注：真实项目安装 pydantic 后使用
# from pydantic import BaseModel

@dataclass
class UploadRequest:
    """文档上传请求"""
    filename: str
    content: bytes
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class QueryRequest:
    """查询请求"""
    question: str
    top_k: Optional[int] = None
    filter_metadata: Optional[Dict[str, str]] = None


@dataclass
class Source:
    """引用来源"""
    document_id: str
    filename: str
    chunk_idx: int
    content: str
    score: float


@dataclass
class QueryResponse:
    """查询响应"""
    answer: str
    sources: List[Source]
    question: str
    tokens_used: int = 0


# ============================================================
# 文档处理
# ============================================================

class DocumentProcessor:
    """
    文档处理器
    支持：PDF, DOCX, TXT, MD
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def process(self, filepath: str) -> List[Dict]:
        """处理文档，返回分块列表"""
        path = Path(filepath)
        suffix = path.suffix.lower()
        
        if suffix == ".pdf":
            text = self._extract_pdf(filepath)
        elif suffix in (".docx", ".doc"):
            text = self._extract_docx(filepath)
        elif suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8")
        else:
            raise ValueError(f"不支持的文件格式：{suffix}")
        
        chunks = self._split_text(text, path.name)
        return chunks
    
    def _extract_pdf(self, filepath: str) -> str:
        """提取 PDF 文本（需安装 PyPDF2）"""
        try:
            import PyPDF2  # type: ignore
            with open(filepath, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            return f"[PDF 文件：{filepath}（需安装 PyPDF2）]"
    
    def _extract_docx(self, filepath: str) -> str:
        """提取 DOCX 文本（需安装 python-docx）"""
        try:
            import docx  # type: ignore
            doc = docx.Document(filepath)
            return "\n".join(para.text for para in doc.paragraphs if para.text)
        except ImportError:
            return f"[DOCX 文件：{filepath}（需安装 python-docx）]"
    
    def _split_text(self, text: str, filename: str) -> List[Dict]:
        """将文本分割为带元数据的块"""
        chunks = []
        words = text.split()
        
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = " ".join(chunk_words)
            
            doc_id = hashlib.md5(f"{filename}_{chunk_idx}".encode()).hexdigest()[:12]
            
            chunks.append({
                "id": doc_id,
                "content": chunk_text,
                "metadata": {
                    "filename": filename,
                    "chunk_idx": str(chunk_idx),
                    "word_count": str(len(chunk_words))
                }
            })
            
            chunk_idx += 1
            step = self.chunk_size - self.overlap
            i += max(step, 1)
        
        return chunks


# ============================================================
# 知识库（向量存储封装）
# ============================================================

class KnowledgeBase:
    """
    知识库管理
    封装 ChromaDB 操作（或其他向量库）
    """
    
    def __init__(self, config: Config):
        self.config = config
        self._collection = None
        self._setup()
    
    def _setup(self):
        """初始化向量库（模拟，真实环境安装 chromadb）"""
        # 真实代码：
        # import chromadb
        # client = chromadb.PersistentClient(path=self.config.chroma_persist_dir)
        # self._collection = client.get_or_create_collection(
        #     name=self.config.collection_name,
        #     metadata={"hnsw:space": "cosine"}
        # )
        self._mock_store: List[Dict] = []  # 模拟存储
        print("✅ 知识库初始化完成（模拟模式）")
    
    def add_chunks(self, chunks: List[Dict]):
        """添加文档块到向量库"""
        for chunk in chunks:
            # 真实代码：
            # embedding = self._get_embedding(chunk["content"])
            # self._collection.add(
            #     ids=[chunk["id"]],
            #     embeddings=[embedding],
            #     documents=[chunk["content"]],
            #     metadatas=[chunk["metadata"]]
            # )
            self._mock_store.append(chunk)
        print(f"  ✅ 已添加 {len(chunks)} 个文档块")
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """语义搜索"""
        # 真实代码：
        # embedding = self._get_embedding(query)
        # results = self._collection.query(
        #     query_embeddings=[embedding],
        #     n_results=top_k
        # )
        # return self._format_results(results)
        
        # 模拟：返回前 top_k 个
        results = []
        for chunk in self._mock_store[:top_k]:
            results.append({
                "id": chunk["id"],
                "content": chunk["content"],
                "metadata": chunk["metadata"],
                "score": 0.85  # 模拟相关度分数
            })
        return results
    
    def _get_embedding(self, text: str) -> List[float]:
        """获取文本 embedding（真实版）"""
        # import openai
        # client = openai.OpenAI(api_key=self.config.openai_api_key)
        # response = client.embeddings.create(
        #     model=self.config.embedding_model,
        #     input=text
        # )
        # return response.data[0].embedding
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """获取知识库统计信息"""
        return {
            "total_chunks": len(self._mock_store),
            "collection": self.config.collection_name
        }


# ============================================================
# 问答引擎
# ============================================================

class QAEngine:
    """
    RAG 问答引擎
    检索 → 构建 Prompt → 生成答案
    """
    
    SYSTEM_PROMPT = """你是一个企业文档助手，专门基于提供的文档回答问题。

回答原则：
1. 只使用提供的文档内容回答，不要使用外部知识
2. 如果文档中没有相关信息，明确说明"文档中没有该信息"
3. 回答要准确、简洁、有结构
4. 引用具体信息时，注明来源文档

文档内容：
{context}
"""
    
    def __init__(self, knowledge_base: KnowledgeBase, config: Config):
        self.kb = knowledge_base
        self.config = config
    
    def answer(self, request: QueryRequest) -> QueryResponse:
        """
        回答问题
        
        流程：
        1. 检索相关文档块
        2. 构建 RAG Prompt
        3. 调用 LLM 生成答案
        4. 返回带引用的响应
        """
        # Step 1: 检索
        top_k = request.top_k or self.config.top_k
        retrieved = self.kb.search(request.question, top_k=top_k)
        
        # Step 2: 构建上下文
        context_parts = []
        sources = []
        
        for i, chunk in enumerate(retrieved):
            context_parts.append(
                f"[文档 {i+1}] 来源：{chunk['metadata'].get('filename', 'unknown')}\n"
                f"{chunk['content']}"
            )
            sources.append(Source(
                document_id=chunk["id"],
                filename=chunk["metadata"].get("filename", "unknown"),
                chunk_idx=int(chunk["metadata"].get("chunk_idx", 0)),
                content=chunk["content"][:200] + "...",
                score=chunk["score"]
            ))
        
        context = "\n\n".join(context_parts)
        
        # Step 3: 生成答案
        answer = self._generate(request.question, context)
        
        return QueryResponse(
            answer=answer,
            sources=sources,
            question=request.question
        )
    
    def _generate(self, question: str, context: str) -> str:
        """调用 LLM 生成答案"""
        # 真实代码：
        # import openai
        # client = openai.OpenAI(api_key=self.config.openai_api_key)
        # response = client.chat.completions.create(
        #     model=self.config.chat_model,
        #     messages=[
        #         {"role": "system", "content": self.SYSTEM_PROMPT.format(context=context)},
        #         {"role": "user", "content": question}
        #     ]
        # )
        # return response.choices[0].message.content
        
        return f"[模拟答案] 根据文档内容，关于 '{question}' 的回答：{context[:200]}..."


# ============================================================
# FastAPI 应用（接口定义）
# ============================================================

API_ROUTES = """
# FastAPI 路由（生产使用）

from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="企业文档问答系统", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

config = Config()
processor = DocumentProcessor(config.chunk_size, config.chunk_overlap)
kb = KnowledgeBase(config)
engine = QAEngine(kb, config)


@app.post("/documents/upload")
async def upload_document(file: UploadFile):
    '''上传并索引文档'''
    content = await file.read()
    temp_path = f"/tmp/{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(content)
    
    chunks = processor.process(temp_path)
    kb.add_chunks(chunks)
    
    return {"message": f"成功索引 {len(chunks)} 个文档块", "filename": file.filename}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    '''问答查询'''
    return engine.answer(request)


@app.get("/stats")
async def get_stats():
    '''系统统计'''
    return kb.get_stats()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.host, port=config.port)
"""


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    config = Config()
    
    # 初始化组件
    processor = DocumentProcessor(chunk_size=50, overlap=10)
    kb = KnowledgeBase(config)
    engine = QAEngine(kb, config)
    
    # 模拟文档索引
    sample_text = """
    公司年假政策：每位员工每年享有15天带薪年假。
    入职第一年按比例计算年假天数。
    年假可跨年度携带，但不超过5天。
    
    报销政策：差旅费用需在出行后10个工作日内提交报销申请。
    单次报销超过5000元需总监审批。
    餐饮报销上限为每人每天200元。
    """
    
    chunks = processor._split_text(sample_text, "company_policy.txt")
    kb.add_chunks(chunks)
    
    # 查询
    request = QueryRequest(question="年假政策是什么？")
    response = engine.answer(request)
    
    print("=== 企业文档问答系统演示 ===")
    print(f"问题：{response.question}")
    print(f"\n答案：{response.answer}")
    print(f"\n引用来源：{len(response.sources)} 个文档块")
    for src in response.sources:
        print(f"  - {src.filename} (相关度: {src.score})")
    
    print("\n=== 知识库统计 ===")
    print(kb.get_stats())
