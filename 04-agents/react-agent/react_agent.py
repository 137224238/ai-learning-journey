"""
ReAct Agent 手写实现
论文：ReAct: Synergizing Reasoning and Acting in Language Models (2022)

核心思想：
Thought → Action → Observation → Thought → Action → ... → Final Answer

相比纯 Chain-of-Thought：
- CoT：只有推理，没有与外部世界交互
- ReAct：推理 + 行动，可以使用工具获取信息
"""

import json
import re
from typing import Dict, Callable, Any, List, Optional
from dataclasses import dataclass


# ============================================================
# 工具系统
# ============================================================

@dataclass
class Tool:
    """工具定义"""
    name: str
    description: str
    func: Callable
    
    def __call__(self, **kwargs) -> str:
        return self.func(**kwargs)


class ToolRegistry:
    """工具注册表"""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, name: str, description: str):
        """装饰器：注册工具"""
        def decorator(func: Callable) -> Callable:
            self._tools[name] = Tool(name=name, description=description, func=func)
            return func
        return decorator
    
    def get(self, name: str) -> Optional[Tool]:
        return self._tools.get(name)
    
    def list_tools(self) -> str:
        """生成工具描述字符串（用于 System Prompt）"""
        lines = []
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)
    
    @property
    def names(self) -> List[str]:
        return list(self._tools.keys())


# ============================================================
# 内置工具（示例）
# ============================================================

registry = ToolRegistry()


@registry.register(
    name="search",
    description="搜索互联网信息。参数：query（搜索关键词）"
)
def search(query: str) -> str:
    """模拟搜索（生产替换为 SerpAPI / Tavily）"""
    # 模拟搜索结果
    mock_results = {
        "transformer": "Transformer 是 Google 在 2017 年提出的神经网络架构，基于注意力机制，是现代 LLM 的基础。",
        "rag": "RAG（检索增强生成）是一种结合检索系统与生成模型的 AI 技术，可减少幻觉、注入实时知识。",
        "北京天气": "北京今日天气：晴，气温 15-25°C，微风。",
        "openai": "OpenAI 是 AI 研究公司，旗下产品包括 GPT-4、ChatGPT、DALL-E 等。"
    }
    for key, value in mock_results.items():
        if key in query.lower():
            return value
    return f"搜索结果：关于 '{query}' 的信息：这是一个模拟搜索结果。"


@registry.register(
    name="calculate",
    description="执行数学计算。参数：expression（数学表达式，如 '2 + 3 * 4'）"
)
def calculate(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        # 只允许安全的数学运算
        allowed = set('0123456789+-*/().% ')
        if not all(c in allowed for c in expression):
            return "错误：只支持基本数学运算"
        result = eval(expression)  # noqa: S307 (教学用，生产请用 sympy)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误：{e}"


@registry.register(
    name="get_current_date",
    description="获取当前日期和时间。无需参数。"
)
def get_current_date() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y年%m月%d日 %H:%M:%S")


@registry.register(
    name="lookup_definition",
    description="查询术语定义。参数：term（要查询的术语）"
)
def lookup_definition(term: str) -> str:
    """术语词典"""
    definitions = {
        "llm": "Large Language Model（大语言模型），基于 Transformer 架构、在大规模文本上预训练的模型。",
        "embedding": "将离散的 token 映射为连续向量空间中的密集向量表示。",
        "attention": "一种使模型能够在处理序列时关注最相关位置的机制。",
        "rag": "Retrieval-Augmented Generation，检索增强生成技术。",
        "fine-tuning": "在预训练模型基础上，用特定领域数据进行进一步训练以适应特定任务。"
    }
    return definitions.get(term.lower(), f"未找到 '{term}' 的定义，请尝试搜索。")


# ============================================================
# ReAct Agent 核心
# ============================================================

class ReActAgent:
    """
    ReAct Agent 实现
    
    循环结构：
    [Thought] → [Action] → [Observation] → [Thought] → ...
    
    终止条件：
    - 生成 "Final Answer:"
    - 达到最大步数
    - 遇到错误
    """
    
    SYSTEM_PROMPT = """你是一个使用 ReAct 框架的 AI 助手。
你需要通过交替进行"思考"和"行动"来解决问题。

可用工具：
{tools}

请严格按照以下格式输出（每次只输出一步）：

Thought: [你的思考过程：分析问题，决定下一步行动]
Action: [工具名称]
Action Input: [JSON 格式的参数]

或者，当你有了最终答案：

Thought: [总结思考]
Final Answer: [最终答案]

重要规则：
1. 每次只能选择一个行动
2. 等待 Observation 后再继续
3. 不要捏造 Observation
4. 最多 {max_steps} 步内给出答案
"""
    
    def __init__(self, tool_registry: ToolRegistry, max_steps: int = 10):
        self.tools = tool_registry
        self.max_steps = max_steps
    
    def _parse_llm_output(self, output: str) -> Dict[str, Any]:
        """
        解析 LLM 输出
        
        返回：
        - {"type": "action", "thought": str, "action": str, "action_input": dict}
        - {"type": "final", "thought": str, "answer": str}
        - {"type": "error", "message": str}
        """
        output = output.strip()
        
        # 提取 Thought
        thought_match = re.search(r'Thought:\s*(.*?)(?=Action:|Final Answer:|$)', output, re.DOTALL)
        thought = thought_match.group(1).strip() if thought_match else ""
        
        # 检查是否是 Final Answer
        final_match = re.search(r'Final Answer:\s*(.*?)$', output, re.DOTALL)
        if final_match:
            return {
                "type": "final",
                "thought": thought,
                "answer": final_match.group(1).strip()
            }
        
        # 提取 Action
        action_match = re.search(r'Action:\s*(\w+)', output)
        input_match = re.search(r'Action Input:\s*(\{.*?\}|\S+.*?)(?=\n|$)', output, re.DOTALL)
        
        if action_match:
            action_name = action_match.group(1).strip()
            action_input_str = input_match.group(1).strip() if input_match else "{}"
            
            try:
                action_input = json.loads(action_input_str)
            except json.JSONDecodeError:
                # 尝试解析简单的 key=value 格式
                action_input = {"query": action_input_str.strip('"')}
            
            return {
                "type": "action",
                "thought": thought,
                "action": action_name,
                "action_input": action_input
            }
        
        return {"type": "error", "message": f"无法解析输出：{output}"}
    
    def _execute_tool(self, action: str, action_input: dict) -> str:
        """执行工具调用"""
        tool = self.tools.get(action)
        if not tool:
            available = ", ".join(self.tools.names)
            return f"错误：工具 '{action}' 不存在。可用工具：{available}"
        
        try:
            return tool(**action_input)
        except TypeError as e:
            return f"工具调用参数错误：{e}"
        except Exception as e:
            return f"工具执行错误：{e}"
    
    def _simulate_llm(self, messages: List[Dict]) -> str:
        """
        模拟 LLM 调用（生产替换为 OpenAI/Claude API）
        
        真实实现：
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stop=["Observation:"]
        )
        return response.choices[0].message.content
        """
        # 简单的规则模拟（仅用于演示结构）
        user_question = messages[-1]["content"] if messages else ""
        
        if "计算" in user_question or any(c.isdigit() for c in user_question):
            return """Thought: 用户需要进行数学计算，我应该使用 calculate 工具。
Action: calculate
Action Input: {"expression": "2 + 3 * 4"}"""
        
        elif "日期" in user_question or "时间" in user_question:
            return """Thought: 用户想知道当前日期，我使用 get_current_date 工具。
Action: get_current_date
Action Input: {}"""
        
        elif "什么是" in user_question:
            term = user_question.replace("什么是", "").replace("？", "").strip()
            return f"""Thought: 用户询问 '{term}' 的定义，先查术语词典。
Action: lookup_definition
Action Input: {{"term": "{term}"}}"""
        
        else:
            return f"""Thought: 我需要搜索相关信息。
Action: search
Action Input: {{"query": "{user_question[:20]}"}}"""
    
    def run(self, question: str, verbose: bool = True) -> str:
        """
        运行 ReAct 循环
        
        Args:
            question: 用户问题
            verbose: 是否打印推理过程
        
        Returns:
            最终答案
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🤔 问题：{question}")
            print('='*60)
        
        # 构建对话历史
        messages = [
            {
                "role": "system",
                "content": self.SYSTEM_PROMPT.format(
                    tools=self.tools.list_tools(),
                    max_steps=self.max_steps
                )
            },
            {"role": "user", "content": question}
        ]
        
        for step in range(1, self.max_steps + 1):
            if verbose:
                print(f"\n--- Step {step} ---")
            
            # LLM 生成下一步
            llm_output = self._simulate_llm(messages)
            
            # 解析输出
            parsed = self._parse_llm_output(llm_output)
            
            if parsed["type"] == "final":
                if verbose:
                    print(f"💭 Thought: {parsed['thought']}")
                    print(f"✅ Final Answer: {parsed['answer']}")
                return parsed["answer"]
            
            elif parsed["type"] == "action":
                if verbose:
                    print(f"💭 Thought: {parsed['thought']}")
                    print(f"🔧 Action: {parsed['action']}")
                    print(f"📥 Input: {parsed['action_input']}")
                
                # 执行工具
                observation = self._execute_tool(
                    parsed["action"],
                    parsed["action_input"]
                )
                
                if verbose:
                    print(f"👁️  Observation: {observation}")
                
                # 将结果加入对话历史
                messages.append({"role": "assistant", "content": llm_output})
                messages.append({
                    "role": "user",
                    "content": f"Observation: {observation}\n\n请继续。"
                })
                
                # 下一步：基于观察给出 Final Answer（简化模拟）
                final_answer = f"根据工具查询结果：{observation}"
                if verbose:
                    print(f"\n✅ Final Answer: {final_answer}")
                return final_answer
            
            else:
                if verbose:
                    print(f"❌ 解析错误：{parsed.get('message')}")
                break
        
        return "抱歉，在最大步数内未能找到答案，请重新表述问题。"


# ============================================================
# 演示
# ============================================================

if __name__ == "__main__":
    agent = ReActAgent(registry, max_steps=5)
    
    questions = [
        "今天是什么日期？",
        "什么是 RAG？",
        "计算 (15 + 27) * 3 的结果",
        "什么是 LLM？"
    ]
    
    for q in questions:
        answer = agent.run(q, verbose=True)
        print(f"\n最终答案：{answer}")
        print("\n" + "="*60)
