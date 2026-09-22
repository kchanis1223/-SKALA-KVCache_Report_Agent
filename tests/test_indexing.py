import json

import pytest

from skala_agent.retrieval.index_cli import build_parser
from skala_agent.retrieval.indexing import build_index_from_chunks, load_manifest
from skala_agent.retrieval.vector_store import NumpyVectorStore
from skala_agent.schemas import Chunk

URL = "https://arxiv.org/abs/2606.12556"


def make_chunk(cid, text="본문", page=1):
    return Chunk(
        id=cid,
        text=text,
        paper_id="itme",
        camp="hw",
        role="primary",
        section="Method",
        page=page,
        source_url=URL,
    )


class FakeEmbedder:
    model_name = "fake-embedder"

    def encode(self, texts):
        return [[float(len(t)), 1.0, 0.0] for t in texts]


def test_build_index_writes_store_and_manifest(tmp_path):
    chunks = [make_chunk("itme-p1-1", "짧음", 1), make_chunk("itme-p2-1", "조금 더 김", 2)]
    manifest = build_index_from_chunks(
        chunks, FakeEmbedder(), tokenizer_name="test-word", index_dir=tmp_path / "idx"
    )
    assert manifest.chunk_count == 2
    assert manifest.dimension == 3
    assert manifest.embedding_model == "fake-embedder"
    assert manifest.chunking.chunk_size_tokens == 1500
    assert len(NumpyVectorStore.load(tmp_path / "idx")) == 2


def test_manifest_is_readable_json(tmp_path):
    build_index_from_chunks(
        [make_chunk("itme-p1-1")], FakeEmbedder(), tokenizer_name="t", index_dir=tmp_path / "idx"
    )
    payload = json.loads((tmp_path / "idx" / "manifest.json").read_text(encoding="utf-8"))
    assert payload["tokenizer"] == "t"
    assert load_manifest(tmp_path / "idx").chunk_count == 1


def test_build_index_rejects_empty_chunks(tmp_path):
    with pytest.raises(ValueError):
        build_index_from_chunks([], FakeEmbedder(), tokenizer_name="t", index_dir=tmp_path / "idx")


def test_batching_covers_every_chunk(tmp_path):
    chunks = [make_chunk(f"itme-p{i}-1", f"text {i}", i) for i in range(1, 8)]
    manifest = build_index_from_chunks(
        chunks, FakeEmbedder(), tokenizer_name="t", index_dir=tmp_path / "idx", batch_size=3
    )
    assert manifest.chunk_count == 7


def test_cli_defaults_point_at_bge_m3():
    args = build_parser().parse_args([])
    assert args.model == "BAAI/bge-m3"
    assert args.tokenizer == "BAAI/bge-m3"
    assert args.index_dir.endswith("bge-m3")


def test_bge_embedder_defaults_avoid_silent_truncation():
    """FlagEmbedding 기본 max_length(512)를 쓰면 1500 토큰 청크가 잘립니다."""
    from skala_agent.retrieval.config import DEFAULT_CHUNKING
    from skala_agent.retrieval.embedding import BgeM3Embedder

    embedder = BgeM3Embedder()
    assert embedder.max_length >= DEFAULT_CHUNKING.chunk_size_tokens
    assert embedder.model_name == "BAAI/bge-m3"


def test_bge_embedder_rejects_invalid_max_length():
    from skala_agent.retrieval.embedding import BgeM3Embedder

    with pytest.raises(ValueError):
        BgeM3Embedder(max_length=0)


def test_bge_embedder_returns_empty_without_loading_model():
    """빈 입력은 모델을 로딩하지 않고 바로 반환합니다."""
    from skala_agent.retrieval.embedding import BgeM3Embedder

    assert BgeM3Embedder().encode([]) == []


def test_search_cli_parses_filters():
    from skala_agent.retrieval.search_cli import build_parser

    args = build_parser().parse_args(["KV", "캐시", "양자화", "--role", "primary", "--top-k", "5"])
    assert " ".join(args.query) == "KV 캐시 양자화"
    assert args.role == "primary"
    assert args.top_k == 5


def test_search_cli_defaults_have_no_filter():
    from skala_agent.retrieval.search_cli import build_parser

    args = build_parser().parse_args(["query"])
    assert args.role is None and args.paper_id is None and args.section is None
