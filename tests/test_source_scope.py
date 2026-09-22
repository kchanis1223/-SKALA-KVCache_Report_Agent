import pytest

from skala_agent.agents.source_scope import explicit_sla_negative, mentions_technology
from skala_agent.agents.web_evaluation import search_queries
from skala_agent.integrations.contracts import SearchDocument
from skala_agent.schemas import Technology


def test_itme_search_keeps_cxl_context_and_excludes_human_adoption():
    technology = Technology(id="itme", name="ITME", camp="hw")
    queries = search_queries("stakeholder", technology, "데이터센터")
    assert len(queries) == 5 and all('"CXL"' in q for q in queries)
    unrelated = SearchDocument(
        id="x",
        title="Family adoption",
        url="https://example.org",
        content="International adoption research.",
    )
    assert not mentions_technology(unrelated, technology)
    related = unrelated.model_copy(update={"title": "ITME: Inference Tiered Memory Expansion"})
    assert mentions_technology(related, technology)
    assert not mentions_technology(
        unrelated.model_copy(update={"content": "commitment"}), technology
    )


@pytest.mark.parametrize(
    "question,quote,expected",
    [
        ("latency_degraded", "no throughput penalty", False),
        ("latency_degraded", "TTFT improved. TPOT unchanged.", True),
        ("latency_degraded", "TTFT Speedup over Recompute", False),
        ("accuracy_loss", "negligible accuracy loss", False),
        ("accuracy_loss", "8% performance degradation", False),
        ("accuracy_loss", "no accuracy loss", True),
        ("accuracy_loss", "accuracy is preserved", True),
    ],
)
def test_measured_failure_cases_do_not_establish_sla(question, quote, expected):
    assert explicit_sla_negative(question, [quote]) is expected


def test_quote_repair_only_retries_invalid_question_once():
    import json

    from test_evaluation_agents import TECHS, ModelFixture, SearchFixture

    from skala_agent.evaluation_provider import EvaluationProvider

    class RepairModel(ModelFixture):
        def invoke(self, messages):
            output = json.loads(super().invoke(messages))
            if len(self.calls) == 1:
                output["findings"][0]["citations"][0]["quote"] = "paraphrased instead of copied"
            else:
                payload = json.loads(messages[1]["content"])
                assert [q["id"] for q in payload["questions"]] == ["trl_1"]
            return json.dumps(output)

    model = RepairModel()
    results, sources = EvaluationProvider(model, SearchFixture()).assess(
        "trl", TECHS[:1], "datacenter", {}, []
    )
    assert len(model.calls) == 2
    assert results[0].status == "assessed" and sources
    assert all("paraphrased" not in s.excerpt for s in sources)


def test_prompts_supply_current_evaluation_date():
    from datetime import date

    from skala_agent.agents.web_evaluation import load_prompt

    assert date.today().isoformat() in load_prompt("trl")


@pytest.mark.parametrize(
    "content,quote,expected",
    [
        ("LongBench,  ZeroSCROLLS", "LongBench, ZeroSCROLLS", "LongBench,  ZeroSCROLLS"),
        ("GPU A\nbatch 8", "GPU A batch 8", "GPU A\nbatch 8"),
        ("improved 20%", "improved 30%", None),
        ("latency did not improve", "latency improved", None),
        ("some text", " ", None),
        ("In this post, we validated ITME.", "We validated ITME.", "we validated ITME."),
        ("US memory", "us memory", None),
    ],
)
def test_only_whitespace_alignment_preserves_original_excerpt(content, quote, expected):
    from skala_agent.agents.citations import source_quote

    assert source_quote(content, quote) == expected


def test_native_schema_limits_repair_to_requested_question_ids():
    from skala_agent.integrations.structured import StructuredExtractor

    class Model:
        def invoke_structured(self, messages, schema):
            assert schema["properties"]["findings"]["minItems"] == 1
            assert schema["properties"]["findings"]["maxItems"] == 1
            assert schema["$defs"]["Finding"]["properties"]["question_id"]["enum"] == ["one"]
            return (
                '{"findings":[{"question_id":"one","answer":"unknown",'
                '"rationale":"no evidence","citations":[]}]}'
            )

    result = StructuredExtractor(Model()).extract("test", {"questions": [{"id": "one"}]})
    assert result.findings[0].question_id == "one"
