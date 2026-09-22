"""김동찬 담당: 색인 생성 CLI.

    uv run skala-index --tokenizer path/to/tokenizer.json

모델과 tokenizer 파일은 커밋하지 않습니다. huggingface.co가 막힌 환경에서는
내려받은 `tokenizer.json` 경로를 넘기세요.
"""

from __future__ import annotations

import argparse

from skala_agent.retrieval.chunker import HFTokenizer
from skala_agent.retrieval.config import DEFAULT_CHUNKING
from skala_agent.retrieval.embedding import BgeM3Embedder
from skala_agent.retrieval.indexing import DEFAULT_INDEX_DIR, build_index


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="논문 PDF를 청킹·임베딩해 색인을 만듭니다.")
    parser.add_argument("--documents", default="configs/documents.json")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--model", default="BAAI/bge-m3")
    parser.add_argument(
        "--tokenizer",
        default=DEFAULT_CHUNKING.tokenizer,
        help="HuggingFace 모델 이름 또는 내려받은 tokenizer.json 경로",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = (
        HFTokenizer.from_file(args.tokenizer)
        if args.tokenizer.endswith(".json")
        else HFTokenizer.from_pretrained(args.tokenizer)
    )
    manifest = build_index(
        BgeM3Embedder(args.model),
        tokenizer,
        documents_path=args.documents,
        index_dir=args.index_dir,
        batch_size=args.batch_size,
    )
    print(
        f"[index] {manifest.chunk_count}개 청크, {manifest.dimension}차원 "
        f"-> {args.index_dir} (모델 {manifest.embedding_model})"
    )
    return 0
