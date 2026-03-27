# Python 进阶

## 学习目标
掌握 AI 工程中最常用的 Python 高级特性，写出更专业、更高效的代码。

## 核心模块

### 1. 装饰器（Decorators）
- 函数装饰器原理（闭包 + `__wrapped__`）
- 带参数的装饰器工厂
- 类装饰器
- 实战：`@retry`、`@rate_limit`、`@cache`、`@timer`

### 2. 异步编程（Asyncio）
- 事件循环机制
- `async/await` 语法
- `asyncio.gather` 并发调用多个 LLM API
- 实战：异步 RAG 流水线

### 3. 类型系统（Type System）
- `typing` 模块全览
- `Pydantic` 数据验证
- `Protocol` 与结构子类型
- 实战：LLM 响应的类型安全解析

### 4. 元编程（Metaprogramming）
- `__getattr__`、`__setattr__`、`__call__`
- 描述符协议
- `dataclasses` 与 `attrs`

## 文件说明

```
python-advanced/
├── decorators.py         # 装饰器实现与示例
├── async_patterns.py     # 异步编程模式
├── type_system.py        # 类型系统实践
├── metaprogramming.py    # 元编程技巧
├── exercises/            # 练习题
│   ├── ex01_decorator.py
│   ├── ex02_async.py
│   └── ex03_types.py
└── README.md
```

## 学习时长估计
- 装饰器：2 天
- 异步编程：3 天
- 类型系统：2 天
- 元编程：1 天
- **合计：约 8 天**
