"""政策语料爬取（阶段 1.2）

低频抓取 SOURCES 里的官方页面，清洗为 Markdown 后带元数据快照落盘：
- 每份文档头部 YAML frontmatter：source_url / crawl_date / policy_topic /
  source_org / title / content_hash（后续政策变动监控对比 hash 用）
- 输出到 data/policy_corpus/（git 版本化），并写 manifest.json 汇总

用法：python -m crawler.crawl
"""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from crawler.extract import html_to_markdown, extract_title
from crawler.sources import SOURCES
from core.logger import logger

CRAWL_DIR = Path("data/policy_corpus")

# 礼貌抓取：标识身份 + 请求间隔，不并行
USER_AGENT = "policy-advisor-crawler/0.1 (portfolio research project; low-frequency crawl)"
DELAY_SECONDS = 2.0
TIMEOUT_SECONDS = 30


def fetch_page(url: str) -> str:
    """抓取单个页面，返回 HTML 文本；非 200 直接抛错由上层记录"""
    resp = requests.get(
        url,
        headers={"User-Agent": USER_AGENT},
        timeout=TIMEOUT_SECONDS,
    )
    resp.raise_for_status()
    return resp.text


def build_snapshot(source: dict, html: str, crawl_date: str) -> tuple[str, dict]:
    """HTML → Markdown 快照文本 + 元数据字典（含 content_hash）"""
    markdown = html_to_markdown(html)
    title = extract_title(html)
    meta = {
        "slug": source["slug"],
        "source_url": source["url"],
        "crawl_date": crawl_date,
        "policy_topic": source["policy_topic"],
        "source_org": source["source_org"],
        "title": title,
        "content_hash": hashlib.sha256(markdown.encode("utf-8")).hexdigest(),
        "char_count": len(markdown),
    }
    frontmatter = "\n".join(f"{k}: {v}" for k, v in meta.items() if k not in ("slug", "char_count"))
    snapshot = f"---\n{frontmatter}\n---\n\n# {title}\n\n{markdown}\n"
    return snapshot, meta


def crawl_all(out_dir: Path = CRAWL_DIR) -> list[dict]:
    """按清单顺序抓取全部语料源，返回每个源的元数据/失败记录"""
    out_dir.mkdir(parents=True, exist_ok=True)
    crawl_date = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    results = []

    for i, source in enumerate(SOURCES):
        if i:
            time.sleep(DELAY_SECONDS)
        slug = source["slug"]
        try:
            html = fetch_page(source["url"])
            snapshot, meta = build_snapshot(source, html, crawl_date)
            (out_dir / f"{slug}.md").write_text(snapshot, encoding="utf-8")
            meta["status"] = "ok"
            logger.info(f"Crawled {slug}: {meta['char_count']} chars")
        except Exception as e:  # noqa: BLE001 —— 单源失败不阻塞整批
            meta = {"slug": slug, "source_url": source["url"], "status": "error", "error": str(e)}
            logger.warning(f"Failed to crawl {slug}: {e}")
        results.append(meta)

    # manifest 汇总，供阶段 1.3 政策变动监控对比 content_hash
    manifest = {"crawl_date": crawl_date, "sources": results}
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    ok = sum(1 for r in results if r["status"] == "ok")
    logger.info(f"Crawl finished: {ok}/{len(results)} sources saved to {out_dir}")
    return results


if __name__ == "__main__":
    for r in crawl_all():
        print(f"  [{r['status']:<5}] {r['slug']}" + (f" — {r.get('error')}" if r["status"] == "error" else ""))
