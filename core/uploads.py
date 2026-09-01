"""
文件上传安全模块
- 扩展名白名单校验
- 文件大小限制
- 存储文件名重写（uuid前缀，防路径穿越与同名覆盖）
"""
import uuid
from pathlib import Path

# 与 core/rag.py 的 loader 支持范围保持一致
ALLOWED_EXTENSIONS = {".pdf", ".txt", ".docx", ".doc"}
# 单文件上限 10MB
MAX_UPLOAD_BYTES = 10 * 1024 * 1024


def validate_upload(filename: str | None, size: int) -> None:
    """
    校验上传文件，非法时抛出 ValueError
    - filename 为空或扩展名不在白名单 -> 拒绝
    - size 超过上限 -> 拒绝
    """
    if not filename:
        raise ValueError("No file provided")

    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise ValueError(
            f"Unsupported file type: '{ext}'. Supported: {sorted(ALLOWED_EXTENSIONS)}"
        )

    if size > MAX_UPLOAD_BYTES:
        raise ValueError(
            f"File too large: {size} bytes (max {MAX_UPLOAD_BYTES} bytes)"
        )


def safe_stored_name(filename: str) -> str:
    """
    生成安全的存储文件名：
    - Path(...).name 剥掉客户端文件名中可能携带的路径成分（防路径穿越）
    - uuid前缀避免同名文件互相覆盖
    """
    base_name = Path(filename).name
    return f"{uuid.uuid4().hex}_{base_name}"
