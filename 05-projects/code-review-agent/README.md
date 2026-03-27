# AI 代码审查助手

## 功能
- 自动检测 Bug / 安全漏洞 / 性能问题
- 集成 GitHub PR Webhook
- 多 Agent 并行审查（提速 3x）

## 技术栈
FastAPI + GitHub API + OpenAI + AST 解析

## 亮点（面试用）
- AST 辅助 LLM，减少幻觉
- 严重程度分级：critical / warning / info
- 评分系统 + 改进建议
