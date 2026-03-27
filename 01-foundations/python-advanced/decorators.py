"""
Python 装饰器：从原理到 AI 工程实战
"""

import time
import functools
import logging
from typing import Callable, TypeVar, Any
from collections import OrderedDict

F = TypeVar('F', bound=Callable[..., Any])

# ============================================================
# 1. 基础装饰器原理
# ============================================================

def simple_decorator(func: F) -> F:
    """最简单的装饰器：理解闭包本质"""
    @functools.wraps(func)  # 保留原函数元数据（__name__, __doc__ 等）
    def wrapper(*args, **kwargs):
        print(f"[Before] Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"[After] {func.__name__} returned {result}")
        return result
    return wrapper  # type: ignore


# ============================================================
# 2. 计时装饰器（AI 推理性能分析常用）
# ============================================================

def timer(func: F) -> F:
    """测量函数执行时间"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"⏱️  {func.__name__} 耗时 {elapsed:.3f}s")
        return result
    return wrapper  # type: ignore


# ============================================================
# 3. 重试装饰器（LLM API 调用必备）
# ============================================================

def retry(max_attempts: int = 3, delay: float = 1.0, exceptions=(Exception,)):
    """
    带参数的装饰器工厂：自动重试
    
    用法：
        @retry(max_attempts=3, delay=0.5, exceptions=(RateLimitError,))
        def call_llm(prompt: str) -> str: ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts:
                        wait = delay * (2 ** (attempt - 1))  # 指数退避
                        logging.warning(
                            f"[Retry] {func.__name__} attempt {attempt}/{max_attempts} "
                            f"failed: {e}. Retrying in {wait:.1f}s..."
                        )
                        time.sleep(wait)
                    else:
                        logging.error(f"[Retry] {func.__name__} all {max_attempts} attempts failed.")
            raise last_exception  # type: ignore
        return wrapper  # type: ignore
    return decorator


# ============================================================
# 4. LRU 缓存（减少重复 LLM 调用）
# ============================================================

def lru_cache(maxsize: int = 128):
    """简版 LRU 缓存装饰器（理解原理用，生产用 functools.lru_cache）"""
    def decorator(func: F) -> F:
        cache: OrderedDict = OrderedDict()

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 将参数转为可哈希的 key
            key = (args, tuple(sorted(kwargs.items())))
            if key in cache:
                cache.move_to_end(key)   # 移动到末尾（最近使用）
                return cache[key]
            result = func(*args, **kwargs)
            cache[key] = result
            if len(cache) > maxsize:
                cache.popitem(last=False) # 淘汰最久未使用的
            return result

        wrapper.cache_info = lambda: {"size": len(cache), "maxsize": maxsize}  # type: ignore
        wrapper.cache_clear = cache.clear  # type: ignore
        return wrapper  # type: ignore
    return decorator


# ============================================================
# 5. 速率限制装饰器（API 调用频率控制）
# ============================================================

def rate_limit(calls_per_second: float = 1.0):
    """控制函数调用频率，避免触发 API Rate Limit"""
    min_interval = 1.0 / calls_per_second
    last_called = [0.0]

    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            now = time.monotonic()
            elapsed = now - last_called[0]
            wait_time = min_interval - elapsed
            if wait_time > 0:
                time.sleep(wait_time)
            result = func(*args, **kwargs)
            last_called[0] = time.monotonic()
            return result
        return wrapper  # type: ignore
    return decorator


# ============================================================
# 6. 类装饰器：为任意类添加日志能力
# ============================================================

def add_logging(cls):
    """类装饰器：自动为所有公开方法添加日志"""
    for name, method in list(vars(cls).items()):
        if callable(method) and not name.startswith('_'):
            setattr(cls, name, timer(method))
    return cls


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    # --- 计时 ---
    @timer
    def slow_function():
        time.sleep(0.1)
        return "done"

    slow_function()

    # --- 重试 ---
    attempt_count = [0]

    @retry(max_attempts=3, delay=0.1, exceptions=(ValueError,))
    def flaky_api_call():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise ValueError("Simulated API error")
        return "Success on attempt 3"

    result = flaky_api_call()
    print(f"Result: {result}")

    # --- 缓存 ---
    @lru_cache(maxsize=3)
    def expensive_embedding(text: str) -> list:
        print(f"  Computing embedding for: {text[:20]}...")
        return [hash(text) % 100]  # 模拟

    expensive_embedding("hello world")
    expensive_embedding("hello world")  # 命中缓存
    print("Cache info:", expensive_embedding.cache_info())  # type: ignore
