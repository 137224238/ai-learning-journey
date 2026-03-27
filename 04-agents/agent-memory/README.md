# Agent 记忆系统

## 记忆类型
- **短期**：Context Window（当前对话）
- **长期**：向量数据库（跨会话持久化）
- **外部**：文件、数据库

## 实现
- 短期：滑动窗口 / 对话摘要压缩
- 长期：重要信息 → Embedding → 存储 → 检索召回

## 文件
- `memory_manager.py`、`conversation_summary.py`、`long_term_memory.py`
