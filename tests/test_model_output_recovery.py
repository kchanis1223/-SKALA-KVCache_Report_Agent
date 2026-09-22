"""#34: malformed output recovery keeps source validation and retry limits intact."""

import json

import httpx
import pytest
from test_context_agents import MeasurementModel, RetrieverFixture
from test_evaluation_agents import TECHS, ModelFixture, SearchFixture

from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import ModelOutputError
from skala_agent.integrations.ollama import OllamaChat
from skala_agent.integrations.structured import StructuredExtractor


def test_truncated_native_output_retries_in_bounded_question_batches():
    calls = []

    def handler(request):
        body = json.loads(request.content)
        payload = json.loads(body["messages"][1]["content"])
        ids = [q["id"] for q in payload["questions"]]
        calls.append(ids)
        if len(ids) > 3:
            return httpx.Response(
                200,
                json={
                    "done": True,
                    "done_reason": "length",
                    "message": {"content": '{"findings":['},
                },
            )
        assert body["format"]["properties"]["findings"]["maxItems"] == len(ids)
        assert body["format"]["$defs"]["Finding"]["properties"]["question_id"]["enum"] == ids
        content = json.dumps(
            {
                "findings": [
                    {
                        "question_id": q,
                        "answer": "unknown",
                        "rationale": "No source",
                        "citations": [],
                    }
                    for q in ids
                ]
            }
        )
        return httpx.Response(200, json={"done": True, "message": {"content": content}})

    model = OllamaChat("qwen3:8b", transport=httpx.MockTransport(handler))
    questions = [{"id": f"q{i}"} for i in range(15)]
    draft = StructuredExtractor(model).extract("15 questions", {"questions": questions})
    assert [f.question_id for f in draft.findings] == [q["id"] for q in questions]
    assert len(calls) == 6
    assert all(len(batch) == 3 for batch in calls[1:])


def test_persistent_truncation_stops_and_preserves_model_output_error_contract():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx.Response(200, json={"done": True, "done_reason": "length"})

    model = OllamaChat("qwen3:8b", transport=httpx.MockTransport(handler))
    provider = EvaluationProvider(model, SearchFixture())
    results, sources = provider.assess("stakeholder", TECHS[:1], "datacenter", {}, [])
    assert len(calls) == 2
    assert results[0].status == "failed" and results[0].error.code == "ModelOutputError"
    assert not results[0].error.retryable and not sources


def test_native_schema_limits_source_ids():
    class Model:
        def invoke_structured(self, messages, schema):
            for name in ("CitationDraft", "MeasurementDraft"):
                assert schema["$defs"][name]["properties"]["source_id"]["enum"] == ["source-1"]
            return '{"findings":[]}'

    StructuredExtractor(Model()).extract("test", {"sources": [{"id": "source-1"}]})


def test_schema_failure_reports_error_type_without_response_content():
    class Model:
        def __init__(self):
            self.calls = 0

        def invoke(self, messages):
            self.calls += 1
            if self.calls == 2:
                assert "literal_error" in messages[-1]["content"]
                assert "PRIVATE_RESPONSE" not in messages[-1]["content"]
            return json.dumps(
                {
                    "findings": [
                        {
                            "question_id": "q",
                            "answer": "PRIVATE_RESPONSE",
                            "rationale": "test",
                            "citations": [],
                        }
                    ]
                }
            )

    model = Model()
    with pytest.raises(ModelOutputError, match="literal_error") as caught:
        StructuredExtractor(model).extract("test", {"questions": [{"id": "q"}]})
    assert "PRIVATE_RESPONSE" not in str(caught.value)
    assert model.calls == 2


def test_invalid_measurement_reextracts_only_affected_question():
    class Model(MeasurementModel):
        def invoke(self, messages):
            self.tamper = not self.calls
            if self.calls:
                payload = json.loads(messages[1]["content"])
                assert [q["id"] for q in payload["questions"]] == ["cost_measured"]
            return super().invoke(messages)

    model = Model()
    provider = EvaluationProvider(model, SearchFixture(), retriever=RetrieverFixture())
    results, sources = provider.assess("domain", TECHS[:1], "datacenter", {}, [])
    assert len(model.calls) == 2
    assert results[0].status == "assessed" and sources
    assert "99%" not in results[0].rationale
    assert "20%" in results[0].rationale


def test_unknown_measurements_are_discarded_without_losing_other_findings():
    class Model(ModelFixture):
        def invoke(self, messages):
            output = json.loads(super().invoke(messages))
            first = output["findings"][0]
            first.update(
                answer="unknown",
                citations=[],
                measurements=[
                    {
                        "source_id": "invented",
                        "metric": "cost",
                        "value": "99%",
                        "conditions": "unknown",
                    }
                ],
            )
            return json.dumps(output)

    model = Model()
    provider = EvaluationProvider(model, SearchFixture())
    results, sources = provider.assess("domain", TECHS[:1], "datacenter", {}, [])
    assert len(model.calls) == 1
    assert results[0].status != "failed" and sources
    assert "99%" not in results[0].rationale
    assert not results[0].signals[0].evidence_ids


def test_repaired_unknown_measurement_keeps_unaffected_questions():
    class Model(MeasurementModel):
        def invoke(self, messages):
            if not self.calls:
                self.tamper = True
                return super().invoke(messages)
            self.calls.append(messages)
            return json.dumps(
                {
                    "findings": [
                        {
                            "question_id": "cost_measured",
                            "answer": "unknown",
                            "rationale": "No exact measurement",
                            "citations": [],
                            "measurements": [],
                        }
                    ]
                }
            )

    model = Model()
    provider = EvaluationProvider(model, SearchFixture(), retriever=RetrieverFixture())
    results, sources = provider.assess("domain", TECHS[:1], "datacenter", {}, [])
    assert len(model.calls) == 2
    assert results[0].status == "assessed"
    assert results[0].details.cost == "개선 불명확"
    assert not results[0].signals[0].evidence_ids
    assert results[0].signals[1].evidence_ids and sources
    assert "99%" not in results[0].rationale


def test_schema_extra_field_name_is_not_exposed_in_diagnostics():
    class Model:
        def invoke(self, messages):
            return '{"findings": [], "PRIVATE_FIELD_NAME": "secret"}'

    with pytest.raises(ModelOutputError, match="extra_forbidden") as caught:
        StructuredExtractor(Model()).extract("test", {})
    assert "PRIVATE_FIELD_NAME" not in str(caught.value)


@pytest.mark.parametrize("perspective", ["trl", "market", "stakeholder", "domain"])
def test_initial_evaluation_constrains_quotes_not_only_the_repair(perspective):
    class Model(ModelFixture):
        def invoke_structured(self, messages, schema):
            payload = json.loads(messages[1]["content"])
            sources = {s["id"]: s["content"] for s in payload["sources"]}
            variants = schema["$defs"]["CitationDraft"]["anyOf"]
            assert len(variants) == len(sources)
            for variant in variants:
                props = variant["properties"]
                text = sources[props["source_id"]["const"]]
                assert all(quote in text for quote in props["quote"]["enum"])
            return self.invoke(messages)

    model = Model()
    provider = EvaluationProvider(model, SearchFixture())
    results, sources = provider.assess(perspective, TECHS[:1], "datacenter", {}, [])
    assert results[0].status != "failed" and sources
    assert len(model.calls) == 1
