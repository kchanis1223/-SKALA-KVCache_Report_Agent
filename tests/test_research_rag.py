"""#35: provider 생성부터 논문 조사·도메인·보고서까지 오프라인으로 검증."""

import json
import sys

import pytest
from test_evaluation_agents import ModelFixture, SearchFixture
from test_retrieval_factory import FakeEmbedder

from skala_agent import adapters, evaluate_cli
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.retrieval.factory import load_retriever
from skala_agent.retrieval.indexing import build_index_from_chunks
from skala_agent.runtime import load_provider
from skala_agent.schemas import Chunk
from skala_agent.workflow.graph import build_graph, initial_state


class ResearchModel(ModelFixture):
    def invoke(self, messages):
        payload = json.loads(messages[1]["content"])
        if "claim" in payload:
            return '{"supports_claim":true}'
        output = json.loads(super().invoke(messages))
        for finding in output["findings"]:
            if finding["question_id"] in {"overview", "scope", "limitations", "experiments"}:
                finding["rationale"] = payload["sources"][0]["content"]
        return json.dumps(output)


@pytest.fixture
def retriever(tmp_path):
    chunks = [
        Chunk(
            id=f"{paper}-p{page}-1",
            text=f"{paper} reduced memory 20% on GPU A batch 8.",
            paper_id=paper,
            camp=camp,
            role=role,
            page=page,
            section="References" if page == 2 else "Method",
            source_url=f"https://example.org/{paper}",
        )
        for paper, camp, role, page in (
            ("turboquant", "sw", "primary", 1),
            ("turboquant", "sw", "primary", 2),
            ("itme", "hw", "primary", 1),
            ("review", "sw", "reference", 1),
        )
    ]
    directory = tmp_path / "index"
    build_index_from_chunks(chunks, FakeEmbedder(), tokenizer_name="test", index_dir=directory)
    return load_retriever(directory, embedder=FakeEmbedder())


def test_research_reads_own_primary_papers_and_keeps_provenance(retriever):
    model = ResearchModel()
    provider = EvaluationProvider(model, SearchFixture(empty=True), retriever=retriever)
    analysis, evidence = provider.research(initial_state()["selected_technologies"])
    assert len(model.calls) == 2
    assert all(a.status == "assessed" for a in analysis.values())
    assert all(
        a.scope and a.limitations and a.experiments and a.evidence_ids for a in analysis.values()
    )
    assert all(e.chunk_id.startswith(e.technology_id + "-") and e.page for e in evidence)
    assert all(e.source_type == "paper" and not e.supports_claim for e in evidence)
    assert any(e.section_or_page == "References" for e in evidence)
    again, second = provider.research(initial_state()["selected_technologies"])
    assert analysis == again and evidence == second


def test_rag_e2e_publishes_verified_paper_quote_with_page_and_chunk(retriever):
    model = ResearchModel()
    provider = EvaluationProvider(model, SearchFixture(empty=True), retriever=retriever)
    state = build_graph(provider).invoke(initial_state())
    assert all(a.status == "assessed" for a in state["tech_analysis"].values())
    assert all("Retriever 미연결" not in a.rationale for a in state["analyses"]["domain"])
    assert "chunk_id: turboquant-p" in state["report"]
    assert "page: 1" in state["report"] or "page: 2" in state["report"]
    assert "reduced memory 20% on GPU A batch 8." in state["report"]
    assert "실험 및 조건:" in state["report"]


def test_no_index_or_no_hits_is_explicitly_pending_without_model_call(retriever):
    model = ResearchModel()
    technologies = initial_state()["selected_technologies"]
    provider = EvaluationProvider(model, SearchFixture(empty=True))
    analysis, evidence = provider.research(technologies)
    assert all(
        a.status == "pending" and "Retriever 미연결" in a.overview for a in analysis.values()
    )
    assert not evidence and not model.calls
    # Use an empty retriever rather than a malformed stored index.
    from skala_agent.retrieval.retriever import DenseRetriever

    provider.retriever = DenseRetriever(FakeEmbedder())
    analysis, evidence = provider.research(technologies)
    assert all(
        a.status == "pending" and "검색 결과가 없어" in a.overview for a in analysis.values()
    )
    assert not evidence and not model.calls


def test_invented_research_quote_is_rejected(retriever):
    model = ResearchModel(
        alter=lambda findings: findings[0]["citations"][0].update(quote="invented")
    )
    provider = EvaluationProvider(model, SearchFixture(empty=True), retriever=retriever)
    with pytest.raises(ModelOutputError, match="원문과 일치"):
        provider.research(initial_state()["selected_technologies"])


def test_real_factory_autoloads_index_and_respects_explicit_override(monkeypatch, retriever):
    monkeypatch.setattr(
        adapters,
        "read_environment",
        lambda _: {
            "TAVILY_API_KEY": "fixture",
            "OPENAI_API_KEY": "fixture",
            "RAG_INDEX_DIR": "chosen",
        },
    )
    calls = []

    def load(path):
        calls.append(path)
        return retriever

    monkeypatch.setattr(adapters, "try_load_retriever", load)
    provider = load_provider("real")
    assert provider.retriever is retriever
    assert provider.evaluators["domain"].retriever is retriever
    assert calls == ["chosen"]
    assert adapters.build_provider(retriever=None).retriever is None
    assert calls == ["chosen"]
    load_provider("demo")
    assert calls == ["chosen"]


def test_evaluate_cli_autoloads_retriever(monkeypatch, tmp_path, retriever):
    monkeypatch.setattr(adapters, "read_environment", lambda _: {"TAVILY_API_KEY": "fixture"})
    monkeypatch.setattr(adapters, "try_load_retriever", lambda _: retriever)
    monkeypatch.setattr(adapters, "TavilySearch", lambda *a, **k: SearchFixture(empty=True))

    class Router:
        def __init__(self, settings):
            self.settings = settings

        def for_agent(self, name):
            return ResearchModel()

    monkeypatch.setattr(adapters, "ModelRouter", Router)
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr(
        sys, "argv", ["skala-evaluate", "--perspective", "domain", "--output", str(output)]
    )
    evaluate_cli.main()
    data = json.loads(output.read_text())
    assert any(e["chunk_id"] and e["page"] for e in data["evidence"])
    assert all("Retriever 미연결" not in a["rationale"] for a in data["assessments"])


@pytest.mark.parametrize("change", [{"paper_id": "other"}, {"role": "reference"}])
def test_research_rejects_wrong_paper_or_reference_results(retriever, change):
    from skala_agent.schemas import RetrievalResult

    chunk = retriever.store.chunks[0].model_copy(update=change)

    class WrongRetriever:
        def retrieve(self, *args, **kwargs):
            return [RetrievalResult(chunk=chunk, score=1)]

    provider = EvaluationProvider(ResearchModel(), SearchFixture(), retriever=WrongRetriever())
    with pytest.raises(ValueError, match="primary"):
        provider.research(initial_state()["selected_technologies"])


def test_repair_schema_pairs_exact_quotes_with_their_sources():
    from skala_agent.agents.citations import quote_options
    from skala_agent.integrations.structured import StructuredExtractor

    documents = [
        {"id": "a", "content": "Cost fell 20% on GPU A. TTFT improved. TPOT unchanged."},
        {"id": "b", "content": "Accuracy preserved.\n" + "x" * 800},
    ]

    class Model:
        def invoke_structured(self, messages, schema):
            variants = schema["$defs"]["CitationDraft"]["anyOf"]
            for variant, document in zip(variants, documents, strict=True):
                assert variant["properties"]["source_id"]["const"] == document["id"]
                choices = variant["properties"]["quote"]["enum"]
                assert choices
                assert all(quote in document["content"] and len(quote) <= 300 for quote in choices)
                assert choices == quote_options(document["content"])
            # Native schema is supplied once, rather than duplicated in the system context.
            assert "anyOf" not in messages[0]["content"]
            return '{"findings":[]}'

    StructuredExtractor(Model()).extract("test", {"sources": documents, "exact_quotes": True})


def test_real_factory_does_not_hide_a_corrupt_index(monkeypatch):
    monkeypatch.setattr(adapters, "read_environment", lambda _: {"TAVILY_API_KEY": "fixture"})

    def broken_index(path):
        raise ValueError("corrupt index")

    monkeypatch.setattr(adapters, "try_load_retriever", broken_index)
    with pytest.raises(ValueError, match="corrupt index"):
        adapters.build_provider()
