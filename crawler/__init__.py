"""政策语料爬取与入库管道（阶段 1）

- crawl: 低频抓取官方政策页面 → 清洗为 Markdown + 元数据，快照存 data/policy_corpus/
- ingest: 读取语料快照 → 切片并携带来源/日期元数据入向量库
"""
