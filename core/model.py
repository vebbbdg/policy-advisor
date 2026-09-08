"""
大模型初始化模块
- LLM 对话模型（DeepSeek）
- Embedding 向量模型（用于RAG）
- Cross-Encoder 重排模型（用于检索结果精排）
- 解耦设计，切换模型只需改这里
"""
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings

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


class ONNXMiniLMEmbeddings(Embeddings):
    """
    langchain Embeddings 适配器，包装 chromadb 自带的 ONNX 版 all-MiniLM-L6-v2。
    与 torch 版是同一个模型（384 维、mean pooling、L2 归一化），但用 onnxruntime 推理：
    启动不 import torch/transformers，内存占用大幅下降，让免费档 512MB 实例也装得下。
    模型首次调用时自动下载到 ~/.cache/chroma/onnx_models（部署时在镜像内预烘焙）。
    注意：ONNX 与 torch 的向量数值略有差异，切换 backend 必须重建向量库索引。
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2
        self._ef = ONNXMiniLM_L6_V2()
        # 预热：触发模型下载（若未缓存）并把权重载入 onnxruntime 会话，让首条查询无需冷加载。
        # chromadb 用 httpx 默认 5s 超时下载，跨境到美国 S3 的慢网络会握手超时；这里临时把
        # 超时放宽到 120s，仅包裹预热调用、结束即还原，绝不影响其它 httpx 用途（如 LLM API）。
        import httpx
        original_stream = httpx.stream

        def _patient_stream(*args, **kwargs):
            kwargs.setdefault("timeout", 120.0)
            return original_stream(*args, **kwargs)

        httpx.stream = _patient_stream
        try:
            self._ef(["warmup"])
        finally:
            httpx.stream = original_stream

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [vec.tolist() for vec in self._ef(list(texts))]

    def embed_query(self, text: str) -> List[float]:
        return self._ef([text])[0].tolist()


def init_embeddings():
    """
    初始化Embedding向量模型，按 EMBEDDING_BACKEND 选择推理后端：
    - USE_FAKE_EMBEDDING=true：随机向量（无语义，仅测试联调），优先级最高
    - onnx（默认）：chromadb 自带 ONNX 版 all-MiniLM-L6-v2，onnxruntime 推理，
      启动不加载 torch，内存低（Render 免费档 512MB 可容纳）
    - torch：sentence-transformers 版 HuggingFaceEmbeddings，功能等价但启动即加载
      torch，内存高；仅本地对比/调试用
    切换 backend 会改变向量数值，必须重建向量库索引，否则检索错乱。
    """
    if os.getenv("USE_FAKE_EMBEDDING", "false").lower() == "true":
        try:
            from langchain_community.embeddings import FakeEmbeddings
            logger.warning("RAG: Using FakeEmbeddings (random vectors, NO semantic retrieval). "
                           "Unset USE_FAKE_EMBEDDING for real search.")
            return FakeEmbeddings(size=384)
        except Exception:
            return None

    backend = os.getenv("EMBEDDING_BACKEND", "onnx").strip().lower()
    if backend == "torch":
        return _init_torch_embeddings()
    if backend != "onnx":
        logger.warning(f"Unknown EMBEDDING_BACKEND={backend!r}, falling back to onnx")

    try:
        # 构造即预热（下载/载入模型），失败会抛异常并回退到 torch，绝不中断启动
        embeddings = ONNXMiniLMEmbeddings()
        logger.info("RAG: Using ONNX embeddings (all-MiniLM-L6-v2 via onnxruntime)")
        return embeddings
    except Exception as e:
        logger.warning(f"ONNX embeddings failed ({e}), falling back to torch HuggingFaceEmbeddings")
        return _init_torch_embeddings()


def _init_torch_embeddings():
    """torch 版 HuggingFaceEmbeddings（sentence-transformers）；启动即加载 torch，内存较高。"""
    # 模型已缓存时强制离线加载（避免弱网下联网探测可选配置文件拖死启动）
    # 注意：离线开关必须在 import 前设置，否则被库固化后不生效
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
