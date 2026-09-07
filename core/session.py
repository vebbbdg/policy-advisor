"""
多会话管理器
- 支持创建多个独立对话
- 每个会话有自己的消息历史
- 类似ChatGPT左侧会话列表功能
- 内存存储，重启服务会清空（生产环境可换Redis/SQLite）
"""
import threading
import uuid
from datetime import datetime
from typing import Dict, List, Optional


# 阶段 2.3：政策顾问人设 + 域外拒答 + 固定免责声明
# 提示词用英文（模型遵循度最高），但要求回答语言跟随用户提问语言
SYSTEM_PROMPT = (
    "You are Policy Advisor, an AI assistant specialized in US student visa work "
    "authorization: CPT, OPT, and STEM OPT for F-1/M-1 students. Your answers are "
    "grounded in official sources (USCIS and DHS Study in the States) retrieved for "
    "each question.\n\n"
    "Rules:\n"
    "1. Only answer questions about student visa work authorization and closely related "
    "policy topics (eligibility, application process, deadlines, maintaining status, "
    "employer requirements). For any other question - programming, math, general "
    "knowledge, casual chat - politely decline and explain that you only help with "
    "CPT/OPT/STEM OPT policy questions.\n"
    "2. Base every answer on the retrieved documents. If they do not contain the answer, "
    "say so honestly. Never invent policy details, deadlines, fees, or numbers.\n"
    "3. Reply in the language of the user's question (Chinese question -> Chinese answer, "
    "English question -> English answer).\n"
    "4. End every substantive policy answer with this exact disclaimer, in the language "
    "of your answer:\n"
    "   - Chinese: 以上信息仅供参考，不构成法律建议。重大决定请咨询学校 DSO 或持牌移民律师，并以 USCIS 官网最新政策为准。\n"
    "   - English: This information is for reference only and is not legal advice. For "
    "major decisions, consult your school's DSO or a licensed immigration attorney, and "
    "always refer to the latest policy on USCIS.gov.\n"
    "   Refusals (rule 1) do not need the disclaimer."
)


class SessionManager:
    """会话管理器：管理多个独立对话"""

    def __init__(self):
        # session_id -> {"messages": [...], "created_at": ..., "title": ..., "seq": ...}
        self._sessions: Dict[str, Dict] = {}
        # 单调递增序号：作为"最新在前"排序的依据，不受系统时钟精度影响
        # （快速连续创建的会话可能拿到相同的 datetime.now()，仅靠时间戳排序会退化成插入顺序）
        self._counter = 0
        # 保护会话字典的并发读写（FastAPI线程池 + async事件循环混合访问）
        self._lock = threading.Lock()

    def create_session(self, title: str = "New Chat") -> str:
        """创建新会话，返回session_id"""
        # 完整uuid4，避免截断后的碰撞与可枚举风险
        session_id = str(uuid.uuid4())
        with self._lock:
            self._counter += 1
            self._sessions[session_id] = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ],
                "created_at": datetime.now().isoformat(),
                "title": title,
                "seq": self._counter,
            }
        return session_id

    def get_messages(self, session_id: str) -> List[Dict]:
        """获取指定会话的消息历史"""
        with self._lock:
            if session_id not in self._sessions:
                return []
            return self._sessions[session_id]["messages"]

    def add_message(self, session_id: str, role: str, content: str):
        """向指定会话添加消息"""
        with self._lock:
            if session_id not in self._sessions:
                return
            self._sessions[session_id]["messages"].append({
                "role": role,
                "content": content
            })
            # 自动用第一条用户消息作为会话标题
            if role == "user" and self._sessions[session_id]["title"] == "New Chat":
                self._sessions[session_id]["title"] = content[:30] + ("..." if len(content) > 30 else "")

    def reset_session(self, session_id: str):
        """重置指定会话"""
        with self._lock:
            if session_id in self._sessions:
                self._sessions[session_id]["messages"] = [
                    {"role": "system", "content": SYSTEM_PROMPT}
                ]

    def delete_session(self, session_id: str):
        """删除指定会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]

    def list_sessions(self) -> List[Dict]:
        """列出所有会话（用于左侧边栏），最新创建的排在最前"""
        with self._lock:
            # 按单调递增序号倒序：seq 越大越新，稳定可靠，不受系统时钟精度影响
            ordered = sorted(
                self._sessions.items(), key=lambda kv: kv[1]["seq"], reverse=True
            )
            return [
                {
                    "id": sid,
                    "title": data["title"],
                    "created_at": data["created_at"],
                }
                for sid, data in ordered
            ]

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with self._lock:
            return session_id in self._sessions


# 全局单例
session_manager = SessionManager()
