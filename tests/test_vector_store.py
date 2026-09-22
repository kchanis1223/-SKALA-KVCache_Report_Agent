import pytest

from skala_agent.retrieval.retriever import DenseRetriever
from skala_agent.retrieval.vector_store import NumpyVectorStore
from skala_agent.schemas import Chunk

URL = "https://arxiv.org/abs/2504.19874"


def make_chunk(cid, *, paper_id="tq", role="primary", section="Method", page=1, text="본문"):
    return Chunk(
        id=cid,
        text=text,
        paper_id=paper_id,
        camp="sw",
        role=role,
        section=section,
        page=page,
        source_url=URL,
    )


class FakeEmbedder:
    """텍스트를 결정적 2차원 벡터로 바꾸는 테스트용 임베더."""

    model_name = "fake"

    def encode(self, texts):
        return [[float(len(t)), 1.0] for t in texts]


def test_search_orders_by_cosine_similarity():
    store = NumpyVectorStore()
    store.upsert(
        [make_chunk("a"), make_chunk("b"), make_chunk("c")],
        [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1]],
    )
    results = store.search([1.0, 0.0], top_k=2)
    assert [r.chunk.id for r in results] == ["a", "c"]
    assert results[0].score > results[1].score


def test_search_fills_score_metadata():
    store = NumpyVectorStore()
    store.upsert([make_chunk("a")], [[1.0, 0.0]])
    result = store.search([1.0, 0.0])[0]
    assert result.score_type == "dense"
    assert result.dense_score == pytest.approx(result.score)
    assert result.sparse_score is None


def test_upsert_replaces_same_id_without_duplicating():
    store = NumpyVectorStore()
    store.upsert([make_chunk("a", text="예전")], [[1.0, 0.0]])
    store.upsert([make_chunk("a", text="최신")], [[0.0, 1.0]])
    assert len(store) == 1
    assert store.search([0.0, 1.0])[0].chunk.text == "최신"


def test_filters_narrow_candidates():
    store = NumpyVectorStore()
    store.upsert(
        [
            make_chunk("a", role="primary", paper_id="tq"),
            make_chunk("b", role="reference", paper_id="tq"),
            make_chunk("c", role="primary", paper_id="itme"),
        ],
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
    )
    assert [r.chunk.id for r in store.search([1.0, 0.0], top_k=5, role="reference")] == ["b"]
    assert [r.chunk.id for r in store.search([1.0, 0.0], top_k=5, paper_id="itme")] == ["c"]
    assert store.search([1.0, 0.0], top_k=5, paper_id="none") == []


def test_empty_store_returns_no_result():
    assert NumpyVectorStore().search([1.0, 0.0]) == []


def test_dimension_mismatch_is_rejected():
    store = NumpyVectorStore()
    store.upsert([make_chunk("a")], [[1.0, 0.0]])
    with pytest.raises(ValueError):
        store.upsert([make_chunk("b")], [[1.0, 0.0, 0.0]])


def test_save_and_load_roundtrip(tmp_path):
    store = NumpyVectorStore()
    store.upsert([make_chunk("a", text="본문 A"), make_chunk("b")], [[1.0, 0.0], [0.0, 1.0]])
    store.save(tmp_path / "index")
    loaded = NumpyVectorStore.load(tmp_path / "index")
    assert len(loaded) == 2
    assert loaded.search([1.0, 0.0])[0].chunk.text == "본문 A"


def test_retriever_indexes_and_searches():
    retriever = DenseRetriever(FakeEmbedder())
    chunks = [make_chunk("a", text="짧음"), make_chunk("b", text="아주 긴 텍스트")]
    assert retriever.index(chunks) == 2
    results = retriever.retrieve("아주 긴 텍스트", top_k=1)
    assert results[0].chunk.id == "b"


def test_retriever_rejects_blank_query():
    with pytest.raises(ValueError):
        DenseRetriever(FakeEmbedder()).retrieve("   ")
