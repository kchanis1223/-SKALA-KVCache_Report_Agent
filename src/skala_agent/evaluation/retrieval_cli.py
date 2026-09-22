"""김동찬 담당: 검색 성능 평가 CLI.

    uv run skala-eval-retrieval

평가셋의 한국어 질의로 색인을 검색해 Hit@1 / Hit@3 / MRR을 계산합니다.
지표 계산은 기존 `retrieval_metrics`를 그대로 사용합니다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from skala_agent.evaluation.dataset import DEFAULT_EVAL_SET, load_eval_set
from skala_agent.evaluation.retrieval import retrieval_metrics
from skala_agent.retrieval.embedding import BgeM3Embedder
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR, load_manifest
from skala_agent.retrieval.vector_store import NumpyVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="검색 성능(Hit@1/Hit@3/MRR)을 측정합니다.")
    parser.add_argument("--eval-set", default=str(DEFAULT_EVAL_SET))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model", default=None, help="기본값은 색인 manifest의 임베딩 모델")
    parser.add_argument("--top-k", type=int, default=10, help="MRR 계산에 쓰는 순위 깊이")
    parser.add_argument("--role", choices=["primary", "reference"], default=None)
    parser.add_argument("--exclude-references", action="store_true", help="References 청크 제외")
    parser.add_argument("--output", default=None, help="결과 JSON 저장 경로")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    eval_set = load_eval_set(args.eval_set)
    manifest = load_manifest(args.index_dir)
    store = NumpyVectorStore.load(args.index_dir)

    if eval_set.chunking_version != manifest.chunking.version:
        raise SystemExit(
            f"평가셋 청킹 버전({eval_set.chunking_version})과 색인"
            f"({manifest.chunking.version})이 다릅니다. 재라벨링이 필요합니다."
        )
    missing = eval_set.check_against_index({chunk.id for chunk in store.chunks})
    if missing:
        raise SystemExit(f"색인에 없는 정답 chunk ID: {missing}")

    embedder = BgeM3Embedder(args.model or manifest.embedding_model)
    rows: list[tuple[list[str], set[str]]] = []
    per_query = []
    for query in eval_set.queries:
        vector = embedder.encode([query.query])[0]
        results = store.search(vector, top_k=args.top_k, role=args.role)
        if args.exclude_references:
            results = [r for r in results if r.chunk.section != "References"]
        retrieved = [result.chunk.id for result in results]
        gold = set(query.gold_chunk_ids)
        rows.append((retrieved, gold))
        rank = next((i for i, cid in enumerate(retrieved, 1) if cid in gold), 0)
        per_query.append(
            {
                "id": query.id,
                "query": query.query,
                "rank": rank,
                "top1": retrieved[0] if retrieved else None,
            }
        )

    metrics = retrieval_metrics(rows)
    print(f"[eval] 질의 {len(rows)}개 · 색인 {manifest.chunk_count}청크")
    print(f"       모델 {embedder.model_name} · top_k={args.top_k} · role={args.role}")
    print(f"       exclude_references={args.exclude_references}")
    print("\n| 지표 | 값 |\n|---|---|")
    for key in ("hit@1", "hit@3", "mrr"):
        print(f"| {key} | {metrics[key]:.3f} |")
    misses = [item for item in per_query if item["rank"] == 0]
    if misses:
        print(f"\n미검출 {len(misses)}건:")
        for item in misses:
            print(f"  {item['id']} {item['query']}")
    if args.output:
        payload = {
            "metrics": metrics,
            "model": embedder.model_name,
            "top_k": args.top_k,
            "role": args.role,
            "exclude_references": args.exclude_references,
            "index": manifest.model_dump(mode="json"),
            "eval_set_version": eval_set.version,
            "per_query": per_query,
        }
        Path(args.output).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n결과 저장: {args.output}")
    return 0
