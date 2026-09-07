"""
SQLite 持久化层（SQLModel）——阶段 3.1

职责边界：本文件只负责"数据结构 + 数据库连接"，
会话/消息的具体读写逻辑仍在 core/session.py 的 SessionManager 里。

为什么用 SQLModel：它把 Pydantic 模型和 SQLAlchemy 表合二为一，
一个类既是数据校验模型、又是数据库表，和 FastAPI 生态无缝衔接。

表设计：
- chat_session：一条 = 左侧边栏的一个对话
- chat_message：一条 = 对话里的一句话（system / user / assistant）
"""
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy.pool import StaticPool
from sqlmodel import SQLModel, Field, create_engine


# 生产环境默认的 SQLite 文件（与 data/vectordb、data/policy_corpus 同目录）。
# 沿用项目的相对路径约定：从项目根目录启动服务。
# 阶段 4 部署时，整个 data/ 目录会挂载到 Render 持久卷。
DEFAULT_DB_URL = "sqlite:///data/app.db"


class SessionRecord(SQLModel, table=True):
    """一条会话记录（对应左侧边栏的一个对话）"""

    __tablename__ = "chat_session"

    # uuid4 字符串主键：对外暴露、不可枚举（沿用现有设计）
    id: str = Field(primary_key=True)
    title: str = Field(default="New Chat")
    # 创建时间（ISO 字符串，保持与现有 list_sessions 输出结构一致）
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    # 单调递增序号：排序"最新在前"的依据，不受系统时钟精度影响
    seq: int = Field(default=0, index=True)
    # 预留字段：阶段 3.2 按用户隔离会话（当前全部为 None）
    user_id: Optional[str] = Field(default=None, index=True)


class MessageRecord(SQLModel, table=True):
    """一条消息记录（system / user / assistant）"""

    __tablename__ = "chat_message"

    # 自增整数主键：天然保证消息按插入顺序排列
    # （get_messages 按 id 升序取出即为时间序，system 提示词永远在最前）
    id: Optional[int] = Field(default=None, primary_key=True)
    # 外键指向所属会话；建索引加速"查某个会话的所有消息"
    session_id: str = Field(foreign_key="chat_session.id", index=True)
    role: str
    content: str


def _ensure_sqlite_dir(db_url: str) -> None:
    """文件型 SQLite 需要父目录已存在（SQLite 只会创建文件，不会创建目录）"""
    if db_url.startswith("sqlite:///") and ":memory:" not in db_url:
        db_path = Path(db_url.replace("sqlite:///", "", 1))
        db_path.parent.mkdir(parents=True, exist_ok=True)


def make_engine(db_url: str = DEFAULT_DB_URL):
    """
    创建数据库引擎并确保表结构存在（幂等：表已存在则无操作）。

    - 文件库（生产）：SQLite 需要 check_same_thread=False，因为 FastAPI 会用
      线程池在多个线程里复用同一个引擎。
    - 内存库（测试，sqlite:///:memory:）：额外用 StaticPool 复用同一条连接，
      否则每次新开连接都会得到一个全新的空内存库，数据根本存不住。
    """
    _ensure_sqlite_dir(db_url)

    connect_args = {}
    kwargs = {}
    if db_url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        if ":memory:" in db_url:
            kwargs["poolclass"] = StaticPool

    engine = create_engine(db_url, echo=False, connect_args=connect_args, **kwargs)
    SQLModel.metadata.create_all(engine)
    return engine
