"""
异步编程：AI 工程中的并发 LLM 调用模式
"""

import asyncio
import time
from typing import List, Any


# ============================================================
# 1. 基础概念：事件循环与协程
# ============================================================

async def basic_coroutine():
    """协程：可以暂停和恢复的函数"""
    print("Step 1: 发送请求")
    await asyncio.sleep(0.1)   # 非阻塞等待（模拟网络 I/O）
    print("Step 2: 收到响应")
    return "result"


# ============================================================
# 2. 并发调用多个 LLM API（核心场景）
# ============================================================

async def mock_llm_call(prompt: str, model: str = "gpt-4") -> str:
    """模拟 LLM API 调用（真实场景替换为 openai.AsyncOpenAI）"""
    latency = len(prompt) * 0.01  # 模拟不同 prompt 有不同延迟
    await asyncio.sleep(latency)
    return f"[{model}] Response to: {prompt[:30]}..."


async def parallel_llm_calls(prompts: List[str]) -> List[str]:
    """
    并发调用 LLM，将多个串行请求变为并行
    
    串行：total_time = sum(each_request_time)
    并行：total_time ≈ max(each_request_time)
    """
    tasks = [mock_llm_call(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


# ============================================================
# 3. 带信号量的并发控制（防止过载）
# ============================================================

async def rate_limited_llm_calls(
    prompts: List[str],
    max_concurrent: int = 5
) -> List[str]:
    """
    用 Semaphore 控制最大并发数，防止触发 API Rate Limit
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def call_with_limit(prompt: str) -> str:
        async with semaphore:
            return await mock_llm_call(prompt)

    tasks = [call_with_limit(p) for p in prompts]
    return list(await asyncio.gather(*tasks))


# ============================================================
# 4. 异步流式输出（Streaming）
# ============================================================

async def stream_llm_response(prompt: str):
    """
    模拟流式输出（真实场景：openai stream=True）
    """
    tokens = f"Response to '{prompt}': The answer is ".split() + ["42", "!"]
    for token in tokens:
        await asyncio.sleep(0.05)  # 模拟 token 生成延迟
        yield token


async def display_stream(prompt: str):
    """消费流式输出"""
    print(f"\n🌊 Streaming response for: {prompt}")
    full_response = []
    async for token in stream_llm_response(prompt):
        print(token, end=" ", flush=True)
        full_response.append(token)
    print()  # 换行
    return " ".join(full_response)


# ============================================================
# 5. 异步 RAG 流水线（完整示例）
# ============================================================

async def async_retrieve(query: str) -> List[str]:
    """模拟异步向量检索"""
    await asyncio.sleep(0.05)
    return [
        f"[Doc 1] Relevant content for: {query}",
        f"[Doc 2] More context for: {query}",
    ]


async def async_rerank(query: str, docs: List[str]) -> List[str]:
    """模拟异步重排序"""
    await asyncio.sleep(0.03)
    return sorted(docs, reverse=True)  # 简单模拟


async def async_rag_pipeline(queries: List[str]) -> List[str]:
    """
    完整异步 RAG 流水线：
    每个 query 并发执行 retrieve → rerank → generate
    """
    async def process_single(query: str) -> str:
        # 步骤 1：检索
        docs = await async_retrieve(query)
        # 步骤 2：重排序
        ranked_docs = await async_rerank(query, docs)
        # 步骤 3：生成（将检索结果作为上下文）
        context = "\n".join(ranked_docs)
        prompt = f"Context:\n{context}\n\nQuestion: {query}"
        answer = await mock_llm_call(prompt)
        return answer

    # 所有 query 并发处理
    results = await asyncio.gather(*[process_single(q) for q in queries])
    return list(results)


# ============================================================
# 演示
# ============================================================

async def main():
    # --- 并行 vs 串行对比 ---
    prompts = [f"Question {i}" for i in range(10)]

    print("=== 串行调用（模拟）===")
    start = time.perf_counter()
    serial_results = [await mock_llm_call(p) for p in prompts[:3]]
    print(f"串行耗时: {time.perf_counter() - start:.2f}s")

    print("\n=== 并行调用 ===")
    start = time.perf_counter()
    parallel_results = await parallel_llm_calls(prompts[:3])
    print(f"并行耗时: {time.perf_counter() - start:.2f}s")
    print(f"结果数量: {len(parallel_results)}")

    # --- 流式输出 ---
    await display_stream("What is RAG?")

    # --- RAG 流水线 ---
    print("\n=== 异步 RAG 流水线 ===")
    queries = ["What is attention?", "How does RAG work?", "Explain transformers"]
    start = time.perf_counter()
    answers = await async_rag_pipeline(queries)
    print(f"处理 {len(queries)} 个查询耗时: {time.perf_counter() - start:.2f}s")
    for q, a in zip(queries, answers):
        print(f"  Q: {q}\n  A: {a}\n")


if __name__ == "__main__":
    asyncio.run(main())
