import pytest

from skala_agent.retrieval.factory import IndexNotBuiltError, load_retriever, try_load_retriever
from skala_agent.retrieval.indexing import build_index_from_chunks
from skala_agent.schemas import Chunk


class FakeEmbedder:
    model_name = "fake"

    def encode(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def make_chunk(cid, text):
    return Chunk(
        id=cid,
        text=text,
        paper_id="itme",
        camp="hw",
        role="primary",
        section="Method",
        page=1,
        source_url="https://arxiv.org/abs/2606.12556",
    )


def build(tmp_path):
    chunks = [make_chunk("itme-p1-1", "짧음"), make_chunk("itme-p2-1", "아주 긴 본문 텍스트")]
    build_index_from_chunks(chunks, FakeEmbedder(), tokenizer_name="t", index_dir=tmp_path / "idx")
    return tmp_path / "idx"


def test_load_retriever_restores_searchable_index(tmp_path):
    retriever = load_retriever(build(tmp_path), embedder=FakeEmbedder())
    results = retriever.retrieve("아주 긴 본문 텍스트", top_k=1)
    assert results[0].chunk.id == "itme-p2-1"
    assert results[0].chunk.source_url is not None


def test_missing_index_raises_with_build_command(tmp_path):
    with pytest.raises(IndexNotBuiltError) as exc:
        load_retriever(tmp_path / "없음", embedder=FakeEmbedder())
    assert "skala-index" in str(exc.value)


def test_try_load_returns_none_when_index_missing(tmp_path):
    assert try_load_retriever(tmp_path / "없음", embedder=FakeEmbedder()) is None


def test_try_load_returns_retriever_when_index_exists(tmp_path):
    assert try_load_retriever(build(tmp_path), embedder=FakeEmbedder()) is not None
