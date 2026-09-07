"""
轻量认证（阶段 3.2）：JWT 签发/校验 + 访客模式 + 用户上下文。

MVP 范围（不依赖任何外部账号即可跑通）：
- 访客：前端首次访问调 /api/auth/guest 领一个访客 JWT（sub=guest:<uuid>），
  可发 GUEST_MESSAGE_LIMIT 条消息，用尽后引导登录——获客转化设计。
- 登录：/api/auth/login 目前是【占位实现】，接受邮箱即签发非访客 JWT，
  不做邮箱验证码 / Google OAuth 校验。真正接入邮件服务或 Google 时，
  只需在 issue_login_token 之前补一段凭据校验，JWT 签发、访客配额、
  会话隔离等基础设施全部复用，不返工。

安全说明：JWT_SECRET 必须用环境变量注入。生产环境（阶段 4 部署）务必设置
强随机值，否则任何人都能伪造 token 冒充他人。开发缺省值仅供本地，且启动即告警。
"""
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

import jwt
from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from core.logger import logger

# 开发缺省密钥：仅本地用；生产必须由 JWT_SECRET 环境变量覆盖
# （长度 >=32 字节，满足 HS256 推荐值，避免 PyJWT 的 InsecureKeyLengthWarning）
_DEV_SECRET = "dev-insecure-secret-change-me-please"
_SECRET = os.getenv("JWT_SECRET") or _DEV_SECRET
if not os.getenv("JWT_SECRET"):
    logger.warning(
        "JWT_SECRET not set - using an INSECURE dev default. "
        "Set a strong random JWT_SECRET in production!"
    )

_ALGORITHM = "HS256"
# token 有效期（天），可用环境变量覆盖
_EXPIRE_DAYS = int(os.getenv("JWT_EXPIRE_DAYS", "7") or 7)

# 访客 user_id 前缀：一眼可辨，也便于将来做数据清理
GUEST_PREFIX = "guest:"


@dataclass
class UserContext:
    """从 JWT 解析出的当前请求用户上下文"""

    user_id: str
    is_guest: bool


def _issue_token(user_id: str, is_guest: bool) -> str:
    """签发一个 HS256 JWT，携带 sub（用户标识）、is_guest、签发/过期时间"""
    now = datetime.now(timezone.utc)
    payload = {
        "sub": user_id,
        "is_guest": is_guest,
        "iat": now,
        "exp": now + timedelta(days=_EXPIRE_DAYS),
    }
    return jwt.encode(payload, _SECRET, algorithm=_ALGORITHM)


def issue_guest_token() -> Tuple[str, str]:
    """签发访客 token，返回 (token, user_id)；user_id 形如 guest:<uuid4>"""
    user_id = f"{GUEST_PREFIX}{uuid.uuid4()}"
    return _issue_token(user_id, is_guest=True), user_id


def issue_login_token(email: str) -> Tuple[str, str]:
    """
    【占位登录】接受邮箱即签发非访客 token，返回 (token, user_id)。

    TODO(阶段 3.2 完整实现)：接入邮箱验证码或 Google OAuth 时，
    在调用本函数之前先完成凭据校验（验证码比对 / OAuth 回调验签），
    校验通过再签发。此处 user_id 用规范化后的邮箱，签发逻辑无需改动。
    """
    user_id = email.strip().lower()
    return _issue_token(user_id, is_guest=False), user_id


def decode_token(token: str) -> UserContext:
    """
    校验并解析 token，返回 UserContext。
    无效 / 过期 / 缺 sub 一律抛 ValueError（由上层依赖转成 401）。
    """
    try:
        payload = jwt.decode(token, _SECRET, algorithms=[_ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise ValueError("token expired")
    except jwt.InvalidTokenError as e:
        raise ValueError(f"invalid token: {e}")

    sub = payload.get("sub")
    if not sub:
        raise ValueError("token missing subject")
    return UserContext(user_id=sub, is_guest=bool(payload.get("is_guest", False)))


# ===== FastAPI 认证依赖 =====
# auto_error=False：缺 Authorization 头时不自动抛 403，交由下面统一转成 401，
# 让前端能据 401 重新领取访客 token（而不是被 403 卡死）
_bearer = HTTPBearer(auto_error=False)


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
) -> UserContext:
    """
    受保护端点的认证依赖：从 Authorization: Bearer <token> 解析当前用户。
    token 缺失或无效 → 401（前端据此重新领访客 token）。
    """
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=401, detail="missing authentication token")
    try:
        return decode_token(credentials.credentials)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
