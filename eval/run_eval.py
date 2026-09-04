"""
RAG 评估运行器（阶段 2.4：面向政策语料）

流程：
1. 在临时目录中隔离索引生产政策语料 data/policy_corpus（不触碰生产 data/vectordb）
2. 对每条评估问题执行 top-k 检索，计算 Recall@k / MRR / Precision@k
3. --translate 模式下：用中文问法 question_zh 先翻译成英文再检索（阶段 2.1 查询翻译方案的数据验收）
4. --judge 模式下：注入上下文生成答案，再用 LLM-as-judge 打分（1-5）
5. 打印汇总表，并把明细保存到 eval/results/

用法:
    python -m eval.run_eval                                # 仅检索指标（英文问法基准）
    python -m eval.run_eval --translate                    # 中文问法→翻译→检索（对比英文基准）
    python -m eval.run_eval --top-k 5                      # 自定义 top-k
    python -m eval.run_eval --retriever hybrid             # BM25+向量混合检索（RRF）
    python -m eval.run_eval --retriever rerank             # 混合召回 + cross-encoder精排
    python -m eval.run_eval --judge                        # 附加生成质量评估（需 DEEPSEEK_API_KEY）
"""
import argparse
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

from eval.metrics import recall_at_k, mrr, precision_at_k
from eval.judge import judge_answer, judge_refusal

EVAL_DIR = Path(__file__).parent
DATASET_PATH = EVAL_DIR / "dataset.json"
RESULTS_DIR = EVAL_DIR / "results"

ANSWER_SYSTEM_PROMPT = (
    "Answer the user's question using ONLY the provided context. "
    "If the context does not contain the answer, say you don't know.\n\n"
    "Context:\n{context}"
)


def load_dataset(path: Path = DATASET_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(tmp_dir: str):
    """在临时目录新建独立的 RAG 引擎并索引生产政策语料（data/policy_corpus）"""
    from core.rag import RAGEngine
    from crawler.ingest import load_corpus

    engine = RAGEngine(persist_directory=tmp_dir)
    engine.add_documents(load_corpus())
    return engine


def release_engine(engine):
    """释放 Chroma 客户端持有的文件句柄（Windows 上不释放会导致临时目录无法删除）"""
    try:
        client = engine.vector_store._client
        if hasattr(client, "clear_system_cache"):
            client.clear_system_cache()
        else:
            client._system.stop()  # 旧版 chromadb 的底层停止接口
    except Exception:
        pass  # 释放失败不阻断评估，清理时会降级处理


def generate_answer(model, context: str, question: str) -> str:
    """基于检索上下文生成答案（评估用非流式调用）"""
    messages = [
        {"role": "system", "content": ANSWER_SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]
    response = model.invoke(messages)
    return response.content if hasattr(response, "content") else str(response)


def run_evaluation(top_k: int, use_judge: bool, retriever: str = "dense", translate: bool = False):
    dataset = load_dataset()
    model = None
    if use_judge:
        from core.model import init_llm_model
        model = init_llm_model()

    results = []
    tmp_dir = tempfile.mkdtemp(prefix="rag_eval_")
    try:
        engine = build_index(tmp_dir)
        print(f"Indexed {engine.get_document_count()} chunks from policy corpus\n")

        for item in dataset:
            answerable = item.get("answerable", True)
            # 翻译路径（阶段 2.1 验收）：用中文问法先翻译成英文再检索；无中文问法的条目回退英文原文
            if translate and item.get("question_zh"):
                from core.rag import translate_query
                query = translate_query(item["question_zh"])
            else:
                query = item["question"]

            if retriever == "hybrid":
                docs = engine.retrieve_hybrid(query, top_k=top_k)
            elif retriever == "rerank":
                docs = engine.retrieve_reranked(query, top_k=top_k)
            else:
                docs = engine.retrieve(query, top_k=top_k)
            chunks = [d.page_content for d in docs]

            row = {
                "id": item["id"],
                "question": item["question"],
                "query_used": query,
                "answerable": answerable,
                "retriever": retriever,
                "retrieved_sources": [d.metadata.get("source", "unknown") for d in docs],
            }

            # 检索指标仅对可回答的问题有意义；探针问题没有 ground-truth 片段
            if answerable:
                row.update({
                    "recall_at_k": recall_at_k(chunks, item["snippet"], top_k),
                    "mrr": mrr(chunks, item["snippet"]),
                    "precision_at_k": precision_at_k(chunks, item["snippet"], top_k),
                })

            if use_judge:
                # 生成上下文直接复用本轮检索结果（与指标口径一致）
                context = engine.format_docs(docs) or ""
                answer = generate_answer(model, context, item["question"])
                if answerable:
                    verdict = judge_answer(model, item["question"], item["reference_answer"], answer)
                else:
                    # 探针问题：考察是否诚实拒答而非编造（幻觉抵抗）
                    verdict = judge_refusal(model, item["question"], answer)
                row["generated_answer"] = answer
                row["judge_score"] = verdict["score"]
                row["judge_reason"] = verdict["reason"]

            results.append(row)
            line = f"  [{row['id']}]"
            if answerable:
                line += f" recall={row['recall_at_k']:.0f} mrr={row['mrr']:.2f}"
            else:
                line += " (unanswerable probe)"
            if use_judge:
                line += f" judge={row['judge_score']}/5"
            print(line)

        release_engine(engine)
    finally:
        # 宽容清理：句柄未完全释放时不抛异常，避免 Windows 上崩溃退出
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


def summarize(results, top_k: int, use_judge: bool, retriever: str = "dense", translate: bool = False):
    answerable = [r for r in results if r.get("answerable", True)]
    probes = [r for r in results if not r.get("answerable", True)]
    n = len(answerable)
    avg = lambda rows, key: sum(r[key] for r in rows) / len(rows) if rows else 0.0

    print("\n" + "=" * 46)
    print(f" Evaluation summary ({n} answerable + {len(probes)} probes, top_k={top_k}, retriever={retriever}, translate={translate})")
    print("=" * 46)
    print(f" Recall@{top_k}:     {avg(answerable, 'recall_at_k'):.3f}")
    print(f" MRR:          {avg(answerable, 'mrr'):.3f}")
    print(f" Precision@{top_k}: {avg(answerable, 'precision_at_k'):.3f}")
    if use_judge:
        judged = [r["judge_score"] for r in answerable if r.get("judge_score", 0) > 0]
        if judged:
            print(f" Judge avg:    {sum(judged) / len(judged):.2f} / 5")
        refused = [r["judge_score"] for r in probes if r.get("judge_score", 0) > 0]
        if refused:
            print(f" Refusal avg:  {sum(refused) / len(refused):.2f} / 5 (hallucination resistance)")
    print("=" * 46)


def save_results(results, top_k: int, use_judge: bool, retriever: str = "dense", translate: bool = False) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"eval_{stamp}_{retriever}_k{top_k}{'_translated' if translate else ''}{'' if not use_judge else '_judged'}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "retriever": retriever, "judged": use_judge, "translate": translate, "results": results}, f, indent=2, ensure_ascii=False)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="RAG evaluation runner")
    parser.add_argument("--top-k", type=int, default=3, help="Number of chunks to retrieve")
    parser.add_argument("--judge", action="store_true", help="Also evaluate answer quality via LLM-as-judge")
    parser.add_argument("--retriever", choices=["dense", "hybrid", "rerank"], default="dense",
                        help="dense = vector only; hybrid = BM25 + vector with RRF fusion; "
                             "rerank = hybrid candidates + cross-encoder reranking")
    parser.add_argument("--translate", action="store_true",
                        help="Evaluate the query-translation path: Chinese questions (question_zh) are "
                             "translated to English before retrieval (Phase 2.1 acceptance test)")
    args = parser.parse_args()

    results = run_evaluation(top_k=args.top_k, use_judge=args.judge, retriever=args.retriever, translate=args.translate)
    summarize(results, args.top_k, args.judge, args.retriever, args.translate)
    out_path = save_results(results, args.top_k, args.judge, args.retriever, args.translate)
    print(f"Details saved to {out_path}")


if __name__ == "__main__":
    main()
