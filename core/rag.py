"""
RAG (Retrieval-Augmented Generation) 检索增强生成模块
- 支持上传 PDF / TXT / DOCX 文档
- 文档切片 + 向量化存储到 ChromaDB
- 提问时检索相关文档片段，注入LLM上下文
- 解决大模型幻觉、知识过时问题
- 硅谷GenAI项目核心加分项
"""
import os
import re
import sys
import types
from pathlib import Path
from typing import List, Optional

# ---- 阻断 langchain_text_splitters 的无用重依赖（阶段 4：省启动内存）----
# 该包 __init__ 会急切 import 其 sentence_transformers 子模块，后者在模块级
# `from sentence_transformers import SentenceTransformer`，连带把 torch + transformers
# （数百 MB）拖进内存。但我们只用纯字符切分的 RecursiveCharacterTextSplitter，压根不需要
# 那个语义切片器。故在导入前给该子模块塞一个占位 stub，让 __init__ 的
# `from .sentence_transformers import SentenceTransformersTokenTextSplitter` 拿到占位即可，
# boot 期就彻底不载入 torch。真正的 sentence_transformers 包不受影响（rerank 模式仍能懒加载）。
if "langchain_text_splitters.sentence_transformers" not in sys.modules:
    _st_stub = types.ModuleType("langchain_text_splitters.sentence_transformers")
    _st_stub.SentenceTransformersTokenTextSplitter = None  # 占位；本项目从不使用该类
    sys.modules["langchain_text_splitters.sentence_transformers"] = _st_stub

from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: E402
from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.model import init_embeddings, init_reranker
from core.retrieval import tokenize, rrf_fuse, top_by_scores
from core.logger import logger
from core.usage import usage_tracker


# 查询翻译专用 LLM 客户端（懒加载，未翻译过中文时零开销；无状态 HTTP 客户端，与对话模型互不影响）
_translator_llm = None


def _get_translator_llm():
    """懒加载翻译用 LLM 客户端"""
    global _translator_llm
    if _translator_llm is None:
        from core.model import init_llm_model
        _translator_llm = init_llm_model()
    return _translator_llm


def translate_query(text: str) -> str:
    """
    查询翻译（阶段 2.1 查询翻译方案）：中文提问翻译成英文，再检索英文语料。
    相比更换多语言嵌入模型：零索引迁移、术语（OPT/CPT/SEVIS/I-20）翻译零风险。
    - 无中文的提问原样返回（英文提问不受影响）
    - 翻译失败降级返回原查询：宁可检索效果打折，不让对话中断
    """
    if not re.search(r"[\u4e00-\u9fff]", text):
        return text
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        llm = _get_translator_llm()
        reply = llm.invoke([
            SystemMessage(
                "You are a translation engine. Translate the user's question into English "
                "for document retrieval. Keep policy terms (OPT, CPT, SEVIS, I-20, USCIS, STEM) "
                "unchanged. Return only the translation, without quotes or explanation."
            ),
            HumanMessage(text),
        ])
        # 阶段 3.3-C：翻译也是一次 DeepSeek 调用，计入当日 token 消耗（仅告警不拦截）
        usage_tracker.record(getattr(reply, "usage_metadata", None))
        translated = (reply.content or "").strip().strip('"').strip("'")
        if not translated:
            logger.warning("Query translation returned empty, using original query")
            return text
        logger.info(f"Query translated to English: {translated!r}")
        return translated
    except Exception as e:
        logger.warning(f"Query translation failed ({e}), using original query")
        return text


# 向量数据库持久化目录（阶段 4：可用 CHROMA_DIR / UPLOAD_DIR 指到 Render 持久卷）
VECTOR_DB_PATH = Path(os.getenv("CHROMA_DIR", "data/vectordb"))
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
VECTOR_DB_PATH.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class RAGEngine:
    """RAG检索引擎"""

    def __init__(self, persist_directory: str | None = None):
        """
        persist_directory: 向量库持久化目录，默认用生产目录；
        评估/测试场景传入临时目录以隔离索引，不污染生产数据。
        """
        self.embeddings = init_embeddings()
        self.vector_store = Chroma(
            persist_directory=persist_directory or str(VECTOR_DB_PATH),
            embedding_function=self.embeddings,
            collection_name="documents"
        )
        # 文档切片器：按字符数切分，有重叠
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,       # 每个片段1000字符
            chunk_overlap=200,     # 重叠200字符，保证上下文连贯
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        # cross-encoder重排模型懒加载：不用rerank模式时不下载模型（生产启动更快）
        self._reranker = None

    def _get_reranker(self):
        """懒初始化重排模型，首次调用时才下载"""
        if self._reranker is None:
            self._reranker = init_reranker()
        return self._reranker

    def _load_document(self, file_path: str) -> List[Document]:
        """根据文件类型加载文档"""
        # 懒加载文档加载器：langchain_community.document_loaders 在 import 时会连锁拉起
        # sentence_transformers/torch/transformers（数百 MB），挪到真正上传解析时才 import，
        # 避免部署启动期无谓占内存（阶段 4：让免费档 512MB 实例装得下 RAG 栈）。
        from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader

        ext = Path(file_path).suffix.lower()
        loaders = {
            ".pdf": PyPDFLoader,
            ".txt": TextLoader,
            ".docx": Docx2txtLoader,
            ".doc": Docx2txtLoader,
        }
        if ext not in loaders:
            raise ValueError(f"Unsupported file type: {ext}. Supported: {list(loaders.keys())}")

        loader_cls = loaders[ext]
        loader = loader_cls(file_path)
        return loader.load()

    def add_document(self, file_path: str, filename: str) -> int:
        """
        添加文档到向量库
        返回切分后的片段数量
        """
        logger.info(f"Loading document: {filename}")
        docs = self._load_document(file_path)

        # 切片
        chunks = self.text_splitter.split_documents(docs)
        logger.info(f"Split into {len(chunks)} chunks")

        # 添加元数据
        for chunk in chunks:
            chunk.metadata["source"] = filename

        # 存入向量库
        self.vector_store.add_documents(chunks)
        logger.info(f"Document {filename} indexed successfully")
        return len(chunks)

    def add_documents(self, docs: List[Document], batch_size: int = 32) -> int:
        """
        批量添加已加载的文档（携带自定义元数据），切片后入向量库。
        供语料管道使用：切片时每个 chunk 自动继承文档级元数据（如 source_url/crawl_date）。
        分批嵌入+写入：一次性 embed 整个语料会让 ONNX 推理内存瞬时飙升，撑爆 Render
        免费档 512MB；小批次把峰值压平（构建期烘焙索引与运行时上传文档都受益）。
        """
        chunks = self.text_splitter.split_documents(docs)
        for i in range(0, len(chunks), batch_size):
            self.vector_store.add_documents(chunks[i:i + batch_size])
        sources = {d.metadata.get("source", "?") for d in docs}
        logger.info(f"Indexed {len(chunks)} chunks from {len(docs)} documents: {sorted(sources)}")
        return len(chunks)

    def retrieve(self, query: str, top_k: int = 3) -> List[Document]:
        """检索与问题最相关的top_k个文档片段"""
        results = self.vector_store.similarity_search(query, k=top_k)
        return results

    def _all_documents(self) -> List[Document]:
        """从向量库取出全部文档（供BM25建立倒排索引）"""
        raw = self.vector_store.get(include=["documents", "metadatas"])
        docs = []
        for content, meta in zip(raw["documents"], raw["metadatas"]):
            docs.append(Document(page_content=content, metadata=meta or {}))
        return docs

    def retrieve_hybrid(self, query: str, top_k: int = 3, candidate_k: int = 20) -> List[Document]:
        """
        BM25 + 向量混合检索（RRF融合）
        - 向量路抓语义相似，BM25路抓关键词精确匹配，互补盲区
        - 两路各自取candidate_k候选，用RRF融合排名后返回top_k
        """
        from rank_bm25 import BM25Okapi

        all_docs = self._all_documents()
        if not all_docs:
            return []

        # 路1：向量检索候选（以内容作为唯一键参与融合）
        dense = self.retrieve(query, top_k=min(candidate_k, len(all_docs)))
        dense_ranking = [d.page_content for d in dense]

        # 路2：BM25全库打分后按分排序（小语料下每次重建索引开销可接受）
        bm25 = BM25Okapi([tokenize(d.page_content) for d in all_docs])
        scores = bm25.get_scores(tokenize(query))
        bm25_ranking = [
            all_docs[i].page_content
            for i in sorted(range(len(all_docs)), key=lambda i: scores[i], reverse=True)
            if scores[i] > 0
        ][:candidate_k]

        # RRF融合两路排名，按键找回原始Document
        fused_keys = rrf_fuse([dense_ranking, bm25_ranking])[:top_k]
        by_content = {d.page_content: d for d in all_docs}
        return [by_content[key] for key in fused_keys if key in by_content]

    def retrieve_reranked(self, query: str, top_k: int = 3, candidate_k: int = 20) -> List[Document]:
        """
        宽召回 + Cross-Encoder 精排
        - 先用混合检索取candidate_k个候选（保召回）
        - 再用cross-encoder对(query, chunk)逐个打分重排（保精度）
        - cross-encoder逐对推理较慢，只在小候选集上用，是经典两阶段检索配方
        """
        candidates = self.retrieve_hybrid(query, top_k=candidate_k, candidate_k=candidate_k)
        if not candidates:
            return []

        reranker = self._get_reranker()
        if reranker is None:
            logger.warning("Reranker unavailable, returning hybrid results without reranking")
            return candidates[:top_k]

        pairs = [(query, doc.page_content) for doc in candidates]
        scores = reranker.predict(pairs)
        return top_by_scores(candidates, list(scores), top_k)

    def retrieve_by_mode(self, query: str, top_k: int = 3) -> List[Document]:
        """
        按 RETRIEVER_MODE 环境变量分派检索模式（阶段 3.3-A）：
        - hybrid（默认）：BM25 + 向量 RRF 融合，评估 Recall@3=0.906 过验收线、无额外延迟
        - rerank：hybrid 宽召回 + cross-encoder 精排，排序精度最高但 +0.5~1s 延迟
        - dense：纯向量检索（最快，Recall@3=0.812）
        未知值降级到 hybrid（生产默认）并记一条告警，绝不因配置笔误中断对话。
        """
        mode = os.getenv("RETRIEVER_MODE", "hybrid").strip().lower()
        if mode == "dense":
            return self.retrieve(query, top_k)
        if mode == "rerank":
            return self.retrieve_reranked(query, top_k)
        if mode != "hybrid":
            logger.warning(f"Unknown RETRIEVER_MODE={mode!r}, falling back to hybrid")
        return self.retrieve_hybrid(query, top_k)

    def format_docs(self, docs: List[Document]) -> Optional[str]:
        """将检索结果格式化为上下文字符串"""
        if not docs:
            return None

        context_parts = []
        for i, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", "unknown")
            context_parts.append(f"[Document {i} from {source}]\n{doc.page_content}")

        return "\n\n".join(context_parts)

    def format_context(self, query: str, top_k: int = 3) -> Optional[str]:
        """检索相关文档并格式化为上下文字符串"""
        return self.format_docs(self.retrieve(query, top_k))

    def get_document_count(self) -> int:
        """获取已索引的文档片段数量"""
        return self.vector_store._collection.count()

    def clear_all(self):
        """清空向量库（按 id 删除，兼容新版 Chroma 不接受空 where 的行为）"""
        all_ids = self.vector_store.get(include=[])["ids"]
        if all_ids:
            self.vector_store._collection.delete(ids=all_ids)
        logger.info("Vector store cleared")


# 全局RAG引擎单例
rag_engine = RAGEngine()
