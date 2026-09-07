"""
LLM 成本追踪（阶段 3.3-C）：按自然日累计 token 消耗，超日预算时告警。

设计要点（仅告警、不拦截，与规划的"日预算告警"定位一致）：
- 内存态、按自然日（YYYY-MM-DD）累计；服务重启会清零，MVP 可接受——
  B 的每 IP 速率限制已兜住费用上限，C 专注可观测性与告警。
- 只告警不拦截：超预算记一条 WARNING（每天仅一次，避免刷屏），
  并在 /api/health 暴露当日用量。绝不因统计逻辑影响对话链路。
- 线程安全：chat-stream（事件循环+线程池）与查询翻译（asyncio.to_thread）
  在不同线程调用，用锁保护计数器。

token 来源：LLM 返回的 usage_metadata（流式需 model.py 开启 stream_usage=True，
最后一个 chunk 携带；invoke 直接返回）。抓不到时记 0，绝不用字符数伪造精确值。
"""
import os
import threading
from datetime import date
from typing import Optional, Tuple

from core.logger import logger


def extract_tokens(usage_metadata) -> Tuple[int, int]:
    """
    从 LLM 返回的 usage_metadata 提取 (input_tokens, output_tokens)。

    usage_metadata 可能是 dict（{"input_tokens":..,"output_tokens":..,"total_tokens":..}）
    或带同名属性的对象；缺失、为空或类型异常时统一返回 (0, 0)——
    统计失败绝不能冒泡影响对话。
    """
    if not usage_metadata:
        return 0, 0

    def _get(key: str) -> int:
        if isinstance(usage_metadata, dict):
            value = usage_metadata.get(key, 0)
        else:
            value = getattr(usage_metadata, key, 0)
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0

    return _get("input_tokens"), _get("output_tokens")


class UsageTracker:
    """按自然日累计 token 消耗的追踪器（仅告警，不拦截）"""

    def __init__(self, daily_budget: Optional[int] = None):
        # daily_budget=None 时从环境变量读取；<=0 表示不设预算（永不告警）
        if daily_budget is None:
            try:
                daily_budget = int(os.getenv("DAILY_TOKEN_BUDGET", "0") or 0)
            except ValueError:
                daily_budget = 0
        self.daily_budget = daily_budget
        self._lock = threading.Lock()
        self._day = date.today().isoformat()
        self._input = 0
        self._output = 0
        self._calls = 0
        self._alerted = False  # 当天是否已告警（避免每次调用刷屏）

    def _rollover_if_new_day(self) -> None:
        """跨自然日时重置计数器（调用方必须已持锁）"""
        today = date.today().isoformat()
        if today != self._day:
            self._day = today
            self._input = 0
            self._output = 0
            self._calls = 0
            self._alerted = False

    def record(self, usage_metadata) -> None:
        """记录一次 LLM 调用的 token 消耗；超日预算则告警（每天仅一次）"""
        input_tokens, output_tokens = extract_tokens(usage_metadata)
        with self._lock:
            self._rollover_if_new_day()
            self._input += input_tokens
            self._output += output_tokens
            self._calls += 1
            total = self._input + self._output
            should_alert = (
                self.daily_budget > 0
                and total >= self.daily_budget
                and not self._alerted
            )
            if should_alert:
                self._alerted = True
                # 在锁内快照要写进日志的值，锁外只做 IO
                day, calls, budget = self._day, self._calls, self.daily_budget

        if should_alert:
            logger.warning(
                f"LLM daily token budget exceeded: {total} tokens (budget {budget}) "
                f"on {day} after {calls} calls. Consider lowering CHAT_RATE_LIMIT "
                f"or investigating abuse."
            )

    def snapshot(self) -> dict:
        """返回当日用量快照（供 /api/health 暴露，只读）"""
        with self._lock:
            self._rollover_if_new_day()
            return {
                "date": self._day,
                "input_tokens": self._input,
                "output_tokens": self._output,
                "total_tokens": self._input + self._output,
                "calls": self._calls,
                "daily_budget": self.daily_budget,
            }


# 全局单例：main.py 与 core/rag.py 共用同一个追踪器，累计当日全部 LLM 消耗
usage_tracker = UsageTracker()
