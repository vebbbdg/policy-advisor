"""
多会话管理器（阶段 3.1：SQLite 持久化）
- 支持创建多个独立对话
- 每个会话有自己的消息历史
- 类似ChatGPT左侧会话列表功能
- 存储落盘到 SQLite（默认 data/app.db），重启服务不再丢失；表结构见 core/db.py
"""
import threading
import uuid
from typing import Dict, List

from sqlmodel import Session, col, select

from core.db import DEFAULT_DB_URL, MessageRecord, SessionRecord, make_engine


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
    """会话管理器：管理多个独立对话（持久化到 SQLite）"""

    def __init__(self, db_url: str = DEFAULT_DB_URL):
        # db_url：默认生产文件库 data/app.db；测试可传 sqlite:///:memory: 得到隔离库
        self.engine = make_engine(db_url)
        # 保护"读取当前最大 seq → 写入新会话"这段临界区，
        # 避免并发下两个会话抢到相同 seq（seq 是"最新在前"排序的唯一依据）
        self._lock = threading.Lock()

    def create_session(self, title: str = "New Chat", user_id: str | None = None) -> str:
        """创建新会话，返回 session_id；user_id 标记归属（阶段 3.2 会话隔离）"""
        # 完整 uuid4，避免截断后的碰撞与可枚举风险
        session_id = str(uuid.uuid4())
        with self._lock:
            with Session(self.engine) as db:
                db.add(SessionRecord(
                    id=session_id, title=title, seq=self._next_seq(db), user_id=user_id
                ))
                # 每个新会话的第一条消息永远是 system 人设提示词
                db.add(MessageRecord(
                    session_id=session_id, role="system", content=SYSTEM_PROMPT
                ))
                db.commit()
        return session_id

    def get_messages(self, session_id: str) -> List[Dict]:
        """获取指定会话的消息历史（按插入顺序，system 永远在最前）"""
        with Session(self.engine) as db:
            if db.get(SessionRecord, session_id) is None:
                return []
            rows = db.exec(
                select(MessageRecord)
                .where(MessageRecord.session_id == session_id)
                .order_by(MessageRecord.id)
            ).all()
            return [{"role": r.role, "content": r.content} for r in rows]

    def add_message(self, session_id: str, role: str, content: str):
        """向指定会话追加一条消息"""
        with self._lock:
            with Session(self.engine) as db:
                record = db.get(SessionRecord, session_id)
                if record is None:
                    return
                db.add(MessageRecord(
                    session_id=session_id, role=role, content=content
                ))
                # 自动用第一条用户消息作为会话标题
                if role == "user" and record.title == "New Chat":
                    record.title = content[:30] + ("..." if len(content) > 30 else "")
                    db.add(record)
                db.commit()

    def reset_session(self, session_id: str):
        """重置指定会话：清空历史消息，仅保留 system 人设提示词（标题不变）"""
        with self._lock:
            with Session(self.engine) as db:
                if db.get(SessionRecord, session_id) is None:
                    return
                self._delete_messages(db, session_id)
                db.add(MessageRecord(
                    session_id=session_id, role="system", content=SYSTEM_PROMPT
                ))
                db.commit()

    def delete_session(self, session_id: str):
        """删除指定会话及其全部消息"""
        with self._lock:
            with Session(self.engine) as db:
                record = db.get(SessionRecord, session_id)
                if record is None:
                    return
                self._delete_messages(db, session_id)
                db.delete(record)
                db.commit()

    def list_sessions(self, user_id: str | None = None) -> List[Dict]:
        """
        列出会话（用于左侧边栏），最新创建的排在最前。
        阶段 3.2：按 user_id 过滤，只返回归属该用户的会话（会话隔离）；
        user_id=None 匹配 user_id IS NULL 那一组（向后兼容未传用户的旧调用/测试）。
        """
        with Session(self.engine) as db:
            # 按单调递增 seq 倒序：seq 越大越新，稳定可靠，不受系统时钟精度影响
            rows = db.exec(
                select(SessionRecord)
                .where(SessionRecord.user_id == user_id)
                .order_by(SessionRecord.seq.desc())
            ).all()
            return [
                {"id": r.id, "title": r.title, "created_at": r.created_at}
                for r in rows
            ]

    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        with Session(self.engine) as db:
            return db.get(SessionRecord, session_id) is not None

    def owns_session(self, session_id: str, user_id: str | None) -> bool:
        """
        校验会话是否归属指定用户（阶段 3.2 越权防护）。
        会话不存在 → False；存在但归属不符 → False。main.py 用它拦截操作他人会话。
        """
        with Session(self.engine) as db:
            record = db.get(SessionRecord, session_id)
            return record is not None and record.user_id == user_id

    def count_user_messages(self, user_id: str) -> int:
        """
        统计某用户发出的消息条数（role='user'），跨其所有会话累计。
        供访客配额判断（阶段 3.2）：访客发满 GUEST_MESSAGE_LIMIT 条即拦截。
        """
        with Session(self.engine) as db:
            # 先取该用户的全部会话 id，再数这些会话里的 user 消息
            session_ids = db.exec(
                select(SessionRecord.id).where(SessionRecord.user_id == user_id)
            ).all()
            if not session_ids:
                return 0
            rows = db.exec(
                select(MessageRecord.id).where(
                    MessageRecord.role == "user",
                    col(MessageRecord.session_id).in_(session_ids),
                )
            ).all()
            return len(rows)

    def _next_seq(self, db: Session) -> int:
        """取当前最大 seq + 1（须在 self._lock 内、同一个 db 会话中调用）"""
        last = db.exec(
            select(SessionRecord).order_by(SessionRecord.seq.desc())
        ).first()
        return (last.seq + 1) if last else 1

    @staticmethod
    def _delete_messages(db: Session, session_id: str):
        """删除某会话的全部消息（内部工具，须在已开启的 db 会话内调用）"""
        rows = db.exec(
            select(MessageRecord).where(MessageRecord.session_id == session_id)
        ).all()
        for r in rows:
            db.delete(r)


# 全局单例：使用生产文件库 data/app.db（导入时自动建目录与表）
session_manager = SessionManager()
