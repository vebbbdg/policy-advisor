"""
RAG (Retrieval-Augmented Generation) 检索增强生成模块
- 支持上传 PDF / TXT / DOCX 文档
- 文档切片 + 向量化存储到 ChromaDB
- 提问时检索相关文档片段，注入LLM上下文
- 解决大模型幻觉、知识过时问题
- 硅谷GenAI项目核心加分项
"""
import os
from pathlib import Path
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document

from core.model import init_embeddings, init_reranker
from core.retrieval import tokenize, rrf_fuse, top_by_scores
from core.logger import logger


# 向量数据库持久化目录
VECTOR_DB_PATH = Path("data/vectordb")
UPLOAD_DIR = Path("data/uploads")
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
