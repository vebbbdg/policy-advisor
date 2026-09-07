"""
LLM 成本追踪单元测试（阶段 3.3-C）：纯内存、零 API 调用、不加载任何模型。
覆盖 token 提取（dict/对象/异常）、按日累计、预算告警（仅一次不刷屏）、
跨自然日重置、预算禁用（<=0 永不告警）。
"""
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.usage import UsageTracker, extract_tokens


class _ObjUsage:
    """模拟"带同名属性的对象"形态的 usage_metadata"""

    def __init__(self, input_tokens, output_tokens):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


def test_extract_tokens_from_dict():
    assert extract_tokens({"input_tokens": 10, "output_tokens": 5, "total_tokens": 15}) == (10, 5)


def test_extract_tokens_from_object():
    assert extract_tokens(_ObjUsage(7, 3)) == (7, 3)


def test_extract_tokens_graceful_on_missing_or_bad():
    # 统计失败绝不能抛异常：None/空/缺字段/非预期类型一律降级为 (0, 0)
    assert extract_tokens(None) == (0, 0)
    assert extract_tokens({}) == (0, 0)
    assert extract_tokens({"input_tokens": None}) == (0, 0)
    assert extract_tokens("garbage") == (0, 0)


def test_record_accumulates_and_snapshot():
    t = UsageTracker(daily_budget=0)
    t.record({"input_tokens": 10, "output_tokens": 5})
    t.record({"input_tokens": 20, "output_tokens": 5})
    snap = t.snapshot()
    assert snap["input_tokens"] == 30
    assert snap["output_tokens"] == 10
    assert snap["total_tokens"] == 40
    assert snap["calls"] == 2
    assert snap["daily_budget"] == 0


def test_no_alert_when_budget_disabled():
    # daily_budget<=0 表示不启用预算：无论消耗多少都不告警
    t = UsageTracker(daily_budget=0)
    t.record({"input_tokens": 100000, "output_tokens": 100000})
    assert t._alerted is False


def test_alert_fires_once_when_budget_exceeded(caplog):
    t = UsageTracker(daily_budget=100)
    with caplog.at_level(logging.WARNING, logger="chatbot"):
        t.record({"input_tokens": 60, "output_tokens": 10})   # total 70 < 100，未超
        assert t._alerted is False
        t.record({"input_tokens": 40, "output_tokens": 10})   # total 120 >= 100，触发告警
        assert t._alerted is True
        t.record({"input_tokens": 10, "output_tokens": 0})    # 仍超，但当天不再重复告警
    # 关键：整天只告警一次，避免每次调用都刷屏
    assert caplog.text.count("daily token budget exceeded") == 1


def test_rollover_resets_counters_on_new_day():
    t = UsageTracker(daily_budget=0)
    t.record({"input_tokens": 10, "output_tokens": 5})
    # 手动把内部日期改成过去某天，模拟"跨自然日"；下一次 record 应重置计数
    t._day = "2000-01-01"
    t._alerted = True
    t.record({"input_tokens": 1, "output_tokens": 1})
    snap = t.snapshot()
    assert snap["calls"] == 1          # 昨天的调用数已清零，只剩这一次
    assert snap["total_tokens"] == 2   # 昨天的 token 已清零
    assert snap["date"] != "2000-01-01"
