"""
工业级日志系统
- 同时输出到控制台和文件
- 区分INFO/WARNING/ERROR级别
- 生产环境必备，方便排查问题
"""
import logging
import sys
from pathlib import Path


def setup_logger(name: str = "chatbot") -> logging.Logger:
    """初始化并返回配置好的logger实例"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if logger.handlers:
        return logger

    # 日志格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

    # 2. 文件输出（自动创建logs目录）
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    logger.addHandler(file_handler)

    return logger


# 全局logger单例
logger = setup_logger()
