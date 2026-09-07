"""
认证单元测试（阶段 3.2）：纯 JWT 逻辑，无 DB、无 API、不加载任何模型。
覆盖访客/登录 token 签发-解析往返、邮箱规范化、过期/伪造/垃圾 token 拒绝、
缺 sub 拒绝、以及认证依赖 get_current_user 的 401 与正常解析。
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jwt
import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials

import core.auth as auth
from core.auth import (
    GUEST_PREFIX,
    UserContext,
    decode_token,
    get_current_user,
    issue_guest_token,
    issue_login_token,
)


def _future_exp():
    return datetime.now(timezone.utc) + timedelta(days=1)


def test_guest_token_roundtrip():
    token, user_id = issue_guest_token()
    ctx = decode_token(token)
    assert ctx.is_guest is True
    assert ctx.user_id == user_id
    assert user_id.startswith(GUEST_PREFIX)


def test_login_token_normalizes_email():
    token, user_id = issue_login_token("  User@Example.COM ")
    ctx = decode_token(token)
    assert ctx.is_guest is False
    assert ctx.user_id == "user@example.com"  # 去首尾空格 + 转小写规范化
    assert user_id == "user@example.com"


def test_decode_rejects_expired_token():
    past = datetime.now(timezone.utc) - timedelta(days=1)
    expired = jwt.encode(
        {"sub": "guest:x", "is_guest": True, "exp": past},
        auth._SECRET,
        algorithm=auth._ALGORITHM,
    )
    with pytest.raises(ValueError, match="expired"):
        decode_token(expired)


def test_decode_rejects_wrong_signature():
    # 用错误密钥伪造的 token：结构合法但签名对不上，必须拒绝
    forged = jwt.encode(
        {"sub": "guest:x", "is_guest": True, "exp": _future_exp()},
        "wrong-secret-0123456789-0123456789",
        algorithm="HS256",
    )
    with pytest.raises(ValueError):
        decode_token(forged)


def test_decode_rejects_garbage():
    with pytest.raises(ValueError):
        decode_token("not-a-real-jwt")


def test_decode_rejects_missing_subject():
    no_sub = jwt.encode(
        {"is_guest": True, "exp": _future_exp()},
        auth._SECRET,
        algorithm=auth._ALGORITHM,
    )
    with pytest.raises(ValueError, match="subject"):
        decode_token(no_sub)


def test_get_current_user_missing_token_raises_401():
    with pytest.raises(HTTPException) as exc:
        get_current_user(None)
    assert exc.value.status_code == 401


def test_get_current_user_invalid_token_raises_401():
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials="garbage")
    with pytest.raises(HTTPException) as exc:
        get_current_user(creds)
    assert exc.value.status_code == 401


def test_get_current_user_valid_token_returns_context():
    token, user_id = issue_guest_token()
    creds = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    ctx = get_current_user(creds)
    assert isinstance(ctx, UserContext)
    assert ctx.user_id == user_id
    assert ctx.is_guest is True
