"""
大模型初始化模块
- LLM 对话模型（DeepSeek）
- Embedding 向量模型（用于RAG）
- Cross-Encoder 重排模型（用于检索结果精排）
- 解耦设计，切换模型只需改这里
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

from core.logger import logger

load_dotenv(override=True)


def _hf_model_cached(model_id: str) -> bool:
    """检查 HuggingFace 模型是否已在本地缓存（离线加载的前提）"""
    hf_home = Path(os.getenv("HF_HOME") or str(Path.home() / ".cache" / "huggingface"))
    cache_root = Path(os.getenv("HF_HUB_CACHE") or str(hf_home / "hub"))
    return (cache_root / f"models--{model_id.replace('/', '--')}" / "snapshots").exists()


def _enable_hf_offline():
    """
    强制 HuggingFace 全家桶只读本地缓存、不碰网络。
    必须在 import huggingface_hub / transformers / sentence_transformers 之前调用：
    这些库在导入时就把离线开关固化成常量，导入后再改环境变量无效。
    """
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"


EMBEDDING_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL_ID = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def init_llm_model():
    """初始化LLM对话模型（流式输出；开启 stream_usage 以返回 token 用量供成本追踪）"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_BASE_URL")

    model = init_chat_model(
        model="deepseek-chat",
        model_provider="deepseek",
        api_key=api_key,
        base_url=base_url,
        streaming=True,
        temperature=0.7,
        # 阶段 3.3-C：开启流式用量返回，最后一个 chunk 携带 usage_metadata 供成本追踪
        stream_usage=True,
    )
    return model


def init_embeddings():
    """
    初始化Embedding向量模型
    默认使用真实HuggingFace语义向量（首次启动会自动下载模型）
    测试/快速启动可设置 USE_FAKE_EMBEDDING=true 使用随机向量（无语义能力，仅供联调）
    """
    if os.getenv("USE_FAKE_EMBEDDING", "false").lower() == "true":
        try:
            from langchain_community.embeddings import FakeEmbeddings
            logger.warning("RAG: Using FakeEmbeddings (random vectors, NO semantic retrieval). "
                           "Unset USE_FAKE_EMBEDDING for real search.")
            return FakeEmbeddings(size=384)
        except Exception:
            return None

    # 真实HuggingFace embeddings：模型已缓存时强制离线加载（避免弱网下联网探测可选配置文件拖死启动）
    # 注意：离线开关必须在 import 前设置，否则被库固化后不生效（此前先 import 后设环境变量的写法无效）
    if _hf_model_cached(EMBEDDING_MODEL_ID):
        _enable_hf_offline()

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "30"
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_ID,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        logger.info("RAG: Using real HuggingFace embeddings (all-MiniLM-L6-v2)")
        return embeddings
    except Exception as e:
        # 降级方案：保证服务能启动，但明确告知检索无效
        logger.warning(f"HuggingFace embeddings failed ({e}), falling back to FakeEmbeddings")
        try:
            from langchain_community.embeddings import FakeEmbeddings
            return FakeEmbeddings(size=384)
        except Exception:
            return None


def init_reranker():
    """
    初始化Cross-Encoder重排模型（首次会自动下载，约80MB）
    ms-marco-MiniLM-L-6-v2：专为段落检索相关性训练，
    将(query, passage)拼在一起打分，能理解词形变化与改写。
    """
    # 同 embedding：模型已缓存时在 import 前设离线开关，避免联网探测可选配置文件拖慢重排。
    # embedding 与 reranker 通常由同一进程先后加载，而离线开关在导入时被固化，
    # 因此只在"重排模型也已缓存"时才启用离线，否则保持联网以支持首次下载。
    if _hf_model_cached(RERANKER_MODEL_ID):
        _enable_hf_offline()

    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(
            RERANKER_MODEL_ID,
            device="cpu",
        )
        logger.info("RAG: Using cross-encoder reranker (ms-marco-MiniLM-L-6-v2)")
        return reranker
    except Exception as e:
        logger.warning(f"Cross-encoder reranker unavailable ({e})")
        return None
