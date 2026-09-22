"""김동찬 담당: 저장된 색인에 질의하는 CLI.

    uv run skala-search "KV 캐시 양자화의 정확도 손실" --role primary

색인을 만든 모델과 같은 모델을 써야 하므로 manifest의 embedding_model을 기본값으로 씁니다.
"""

from __future__ import annotations

import argparse

from skala_agent.retrieval.embedding import BgeM3Embedder
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR, load_manifest
from skala_agent.retrieval.vector_store import NumpyVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="저장된 색인에서 관련 청크를 찾습니다.")
    parser.add_argument("query", nargs="+", help="검색 질의 (한국어 가능)")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--role", choices=["primary", "reference"], default=None)
    parser.add_argument("--paper-id", default=None)
    parser.add_argument("--section", default=None)
    parser.add_argument("--excerpt", type=int, default=140, help="본문 미리보기 길이")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = load_manifest(args.index_dir)
    store = NumpyVectorStore.load(args.index_dir)
    query = " ".join(args.query)
    vector = BgeM3Embedder(manifest.embedding_model).encode([query])[0]
    results = store.search(
        vector,
        top_k=args.top_k,
        role=args.role,
        paper_id=args.paper_id,
        section=args.section,
    )
    print(f"[search] {query!r} · 색인 {manifest.chunk_count}청크 · 모델 {manifest.embedding_model}")
    if not results:
        print("  결과 없음 (필터 조건을 확인하세요)")
        return 0
    for rank, result in enumerate(results, 1):
        chunk = result.chunk
        preview = " ".join(chunk.text.split())[: args.excerpt]
        print(f"\n{rank}. {result.score:.4f}  {chunk.id}  p{chunk.page}  [{chunk.section}]")
        print(f"   {preview}...")
    return 0
