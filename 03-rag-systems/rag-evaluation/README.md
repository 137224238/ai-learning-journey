# RAGAS 评估框架实践

## 四大指标
1. **Faithfulness**：答案是否忠实于检索内容（防幻觉）
2. **Answer Relevancy**：答案是否回答了问题
3. **Context Precision**：检索到的内容有多少是有用的
4. **Context Recall**：有用信息被检索到的比例

## 评估流程
```
构建测试集（问题+标准答案+标准来源）
    → 运行 RAG 系统
    → RAGAS 自动评分
    → 分析薄弱项 → 优化
```

## 文件
- `ragas_pipeline.py`：评估完整流程
- `test_dataset.json`：示例测试集
- `results_analysis.py`：结果可视化分析
