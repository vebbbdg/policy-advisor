"""政策语料入库（阶段 1.2）

读取 data/policy_corpus/ 的 Markdown 快照，解析 frontmatter 元数据，
切片后携带 source_url / crawl_date / policy_topic / content_hash 入生产向量库。

用法：python -m crawler.ingest [--rebuild]
  --rebuild 先清空生产向量库再入库（换嵌入模型/语料全量重灌时使用）
"""
import argparse
from pathlib import Path

from langchain_core.documents import Document

from crawler.crawl import CRAWL_DIR

# frontmatter 中需要随 chunk 进向量库的字段（阶段 2 引用与时效输出直接消费它们）
_META_KEYS = ("source_url", "crawl_date", "policy_topic", "source_org", "title", "content_hash")


def parse_snapshot(path: Path) -> Document:
    """解析单个语料快照：frontmatter 元数据 + 正文"""
    text = path.read_text(encoding="utf-8")
    meta: dict = {"source": path.name}

    body = text
    if text.startswith("---\n"):
        end = text.find("\n---\n", 4)
        if end != -1:
            for line in text[4:end].splitlines():
                key, _, value = line.partition(":")
                if key and value:
                    meta[key.strip()] = value.strip()
            body = text[end + 5:]

    # 只保留白名单字段，避免无关元数据进向量库
    meta = {k: v for k, v in meta.items() if k in _META_KEYS or k == "source"}
    return Document(page_content=body.strip(), metadata=meta)


def load_corpus(corpus_dir: Path = CRAWL_DIR) -> list[Document]:
    """加载全部语料快照为带元数据的 Document 列表"""
    docs = [parse_snapshot(p) for p in sorted(corpus_dir.glob("*.md"))]
    if not docs:
        raise FileNotFoundError(f"No corpus snapshots found in {corpus_dir}; run `python -m crawler.crawl` first")
    return docs


def ingest(rebuild: bool = False) -> int:
    """入库主流程，返回新增 chunk 数"""
    # 延迟导入：避免纯解析场景（如测试）触发向量库与模型初始化
    from core.rag import rag_engine

    if rebuild:
        rag_engine.clear_all()
    docs = load_corpus()
    return rag_engine.add_documents(docs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest policy corpus into the vector store")
    parser.add_argument("--rebuild", action="store_true", help="clear the vector store before ingesting")
    args = parser.parse_args()
    n = ingest(rebuild=args.rebuild)
    print(f"Ingested {n} chunks into the vector store")
