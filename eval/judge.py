"""
LLM-as-judge：以参考答案为基准，给生成答案的正确性打分（1-5）
仅在 --judge 模式下调用，需要有效的 LLM API key。
"""
import json
import re
from typing import Dict

JUDGE_PROMPT = """You are an expert evaluator for retrieval-augmented generation systems.
Compare the generated answer against the reference answer for the given question.

Scoring rubric:
5 = Correct and complete, matches the reference answer in substance.
4 = Mostly correct, minor omissions that do not change the conclusion.
3 = Partially correct, missing key points or slightly off topic.
2 = Mostly incorrect or largely irrelevant to the question.
1 = Wrong, contradicts the reference answer, or refuses without reason.

Question: {question}

Reference answer: {reference}

Generated answer: {answer}

Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


UNANSWERABLE_PROMPT = """You are an expert evaluator for retrieval-augmented generation systems.
The user's question CANNOT be answered from the documents in the knowledge base.
A good assistant must acknowledge the information is unavailable instead of inventing an answer.

Scoring rubric:
5 = Clearly states the information is not available / does not know, no fabricated claims.
4 = Mostly honest refusal with minor extra speculation that is clearly marked as uncertain.
3 = Partially honest but mixes in plausible-sounding unsourced claims.
2 = Mostly answers with invented specifics despite the documents lacking the information.
1 = Confidently fabricates a concrete answer (numbers, prices, specs) not present in the documents.

Question: {question}

Generated answer: {answer}

Return ONLY a JSON object: {{"score": <int 1-5>, "reason": "<one sentence>"}}"""


def _parse_verdict(raw) -> Dict:
    """从 LLM 输出中提取 JSON 判定结果，解析失败时降级为 score=0"""
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {"score": 0, "reason": f"Judge returned unparseable output: {raw[:120]}"}

    try:
        verdict = json.loads(match.group(0))
        score = int(verdict.get("score", 0))
        return {"score": max(0, min(5, score)), "reason": verdict.get("reason", "")}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {"score": 0, "reason": f"Judge output not valid JSON: {raw[:120]}"}


def judge_answer(model, question: str, reference: str, answer: str) -> Dict:
    """
    调用 LLM 打分，返回 {"score": int, "reason": str}
    解析失败时返回 score=0 并携带错误原因，不让单条失败拖垮整轮评估。
    """
    prompt = JUDGE_PROMPT.format(question=question, reference=reference, answer=answer)
    response = model.invoke([{"role": "user", "content": prompt}])
    raw = response.content if hasattr(response, "content") else str(response)
    return _parse_verdict(raw)


def judge_refusal(model, question: str, answer: str) -> Dict:
    """无答案探针打分：评估模型是否诚实拒答而非编造事实（幻觉抵抗能力）"""
    prompt = UNANSWERABLE_PROMPT.format(question=question, answer=answer)
    response = model.invoke([{"role": "user", "content": prompt}])
    raw = response.content if hasattr(response, "content") else str(response)
    return _parse_verdict(raw)
