"""
RAG 评估框架
- corpus/       评估语料（独立于生产知识库）
- dataset.json  QA 评估集（问题 + 应命中的原文片段 + 参考答案）
- metrics.py    检索指标（Recall@k / MRR / Precision@k）
- judge.py      LLM-as-judge 生成质量打分
- run_eval.py   CLI 运行器：隔离索引 -> 检索 -> 指标 -> 报告

用法:
    python -m eval.run_eval              # 仅检索指标
    python -m eval.run_eval --judge      # 检索 + 生成 + LLM 打分（需 DEEPSEEK_API_KEY）
"""
