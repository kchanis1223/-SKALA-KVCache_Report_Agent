import json

import pytest

from skala_agent.agents import evidence_validation
from skala_agent.agents.evidence_validation import EvidenceValidator
from skala_agent.evaluation_provider import EvaluationProvider
from skala_agent.integrations.contracts import IncompleteModelOutputError, ModelOutputError
from skala_agent.schemas import Evidence


def evidence(n, **updates):
    return Evidence(
        id=f"e{n}",
        technology_id="itme",
        claim=f"claim {n}",
        excerpt="supported",
        url=f"https://example.org/{n}",
        title="fixture",
        source_type="paper",
    ).model_copy(update=updates)


class Model:
    model = "fixture-model"
    base_url = "https://example.org"
    reasoning_effort = "low"

    def __init__(self):
        self.calls = []
        self.override = None

    def invoke_structured(self, messages, schema):
        items = json.loads(messages[1]["content"])["evidence"]
        self.calls.append(items)
        results = [
            {"id": e["id"], "supports_claim": e["excerpt"] == "supported"} for e in reversed(items)
        ]
        if self.override:
            return self.override(results)
        return json.dumps({"results": results})


def test_three_rounds_reduce_300_calls_to_21_and_unchanged_round_to_zero():
    model = Model()
    provider = EvaluationProvider(model, object())
    items = [evidence(n) for n in range(150)]
    for size, calls in [(50, 7), (100, 14), (150, 21), (150, 21)]:
        result = provider.validate_evidence(items[:size])
        assert len(model.calls) == calls
        assert [e.id for e in result] == [e.id for e in items[:size]]
        assert all(e.supports_claim for e in result)
    assert sum(map(len, model.calls)) == 150
    assert all(not e.supports_claim for e in items)


def test_false_results_are_cached_and_changed_excerpt_revokes_true():
    model, validator = Model(), EvidenceValidator()
    first = [evidence(1), evidence(2, excerpt="unsupported")]
    assert [e.supports_claim for e in validator.validate(first, model)] == [True, False]
    assert [e.supports_claim for e in validator.validate(first[::-1], model)] == [False, True]
    assert len(model.calls) == 1
    changed = evidence(1, excerpt="revoked", supports_claim=True)
    assert not validator.validate([changed], model)[0].supports_claim
    assert len(model.calls) == 2


@pytest.mark.parametrize(
    "field,value",
    [
        ("claim", "new claim"),
        ("url", "https://other.example/"),
        ("technology_id", "turboquant"),
        ("chunk_id", "chunk-new"),
    ],
)
def test_changed_context_invalidates_cache(field, value):
    model, validator = Model(), EvidenceValidator()
    item = evidence(1)
    validator.validate([item], model)
    validator.validate([Evidence.model_validate({**item.model_dump(), field: value})], model)
    assert len(model.calls) == 2


def test_equal_content_across_ids_is_checked_once_with_ids_preserved():
    model, validator = Model(), EvidenceValidator()
    first = evidence(1)
    items = [first, first.model_copy(update={"id": "another-owner"})]
    assert [e.id for e in validator.validate(items, model)] == [e.id for e in items]
    assert len(model.calls[0]) == 1


def test_model_settings_prompt_schema_and_instance_invalidate_cache(monkeypatch):
    model, validator = Model(), EvidenceValidator()
    items = [evidence(1)]
    validator.validate(items, model)
    for attr in ("model", "base_url", "reasoning_effort"):
        monkeypatch.setattr(model, attr, "changed")
        validator.validate(items, model)
    monkeypatch.setattr(evidence_validation, "PROMPT", "changed prompt")
    validator.validate(items, model)
    original = evidence_validation.BatchResult.model_json_schema()
    monkeypatch.setattr(
        evidence_validation.BatchResult,
        "model_json_schema",
        lambda: {**original, "title": "new schema"},
    )
    validator.validate(items, model)
    assert len(model.calls) == 6
    replacement = Model()
    validator.validate(items, replacement)
    assert len(replacement.calls) == 1


@pytest.mark.parametrize(
    "bad", ["missing", "duplicate", "unknown", "string_bool", "single", "json", "extra"]
)
def test_bad_batches_fail_closed_and_are_not_cached(bad):
    model, validator = Model(), EvidenceValidator()

    def invalid(results):
        if bad == "missing":
            results.pop()
        if bad == "duplicate":
            results.append(results[0])
        if bad == "unknown":
            results[0]["id"] = "other"
        if bad == "string_bool":
            results[0]["supports_claim"] = "true"
        if bad == "single":
            return '{"supports_claim":true}'
        if bad == "json":
            return '{"results":'
        if bad == "extra":
            results[0]["explanation"] = "unexpected"
        return json.dumps({"results": results})

    model.override = invalid
    items = [evidence(1), evidence(2)]
    with pytest.raises(ModelOutputError):
        validator.validate(items, model)
    assert len(model.calls) == 2
    assert not any(e.supports_claim for e in items)
    model.override = None
    assert all(e.supports_claim for e in validator.validate(items, model))
    assert len(model.calls) == 3


@pytest.mark.parametrize("error", [TimeoutError, ConnectionError, IncompleteModelOutputError])
def test_failed_requests_are_not_cached(error):
    model, validator = Model(), EvidenceValidator()

    def fail(_):
        raise error("fixture")

    model.override = fail
    expected = ModelOutputError if error is IncompleteModelOutputError else error
    with pytest.raises(expected):
        validator.validate([evidence(1)], model)
    count = len(model.calls)
    model.override = None
    assert validator.validate([evidence(1)], model)[0].supports_claim
    assert len(model.calls) == count + 1


def test_only_invalid_batch_is_retried():
    model, validator = Model(), EvidenceValidator(batch_size=2)

    def recover(results):
        if len(model.calls) == 2:
            return '{"results":[]}'
        return json.dumps({"results": results})

    model.override = recover
    assert len(validator.validate([evidence(n) for n in range(4)], model)) == 4
    assert [[e["id"] for e in c] for c in model.calls] == [["e0", "e1"], ["e2", "e3"], ["e2", "e3"]]


def test_utf8_payload_limit_splits_without_truncation():
    model = Model()
    items = [evidence(n, excerpt="가" * 100) for n in range(3)]
    validator = EvidenceValidator(max_payload_bytes=500)
    validator.validate(items, model)
    assert len(model.calls) == 3
    assert all(c[0]["excerpt"] == "가" * 100 for c in model.calls)
    with pytest.raises(ModelOutputError):
        validator.validate([evidence(4, excerpt="가" * 500)], model)
    assert len(model.calls) == 3


def test_bounded_cache_and_provider_isolation():
    model, validator = Model(), EvidenceValidator(cache_size=1)
    validator.validate([evidence(1), evidence(2)], model)
    validator.validate([evidence(1)], model)
    assert len(model.calls) == 2
    EvidenceValidator().validate([evidence(1)], model)
    assert len(model.calls) == 3


def test_empty_input_and_conflicting_ids_do_not_call_model():
    model, validator = Model(), EvidenceValidator(batch_size=1)
    assert validator.validate([], model) == []
    with pytest.raises(ValueError):
        validator.validate([evidence(1), evidence(1, excerpt="changed")], model)
    assert model.calls == []


def test_concurrent_identical_requests_share_verified_result():
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event

    entered, release = Event(), Event()
    model, validator = Model(), EvidenceValidator()

    def block(results):
        entered.set()
        assert release.wait(5)
        return json.dumps({"results": results})

    model.override = block
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(validator.validate, [evidence(1)], model)
        assert entered.wait(5)
        second = pool.submit(validator.validate, [evidence(1)], model)
        release.set()
        assert first.result() == second.result()
    assert len(model.calls) == 1


def test_plain_invoke_model_can_return_content_object():
    from types import SimpleNamespace

    class PlainModel:
        def invoke(self, messages):
            items = json.loads(messages[1]["content"])["evidence"]
            return SimpleNamespace(
                content=json.dumps(
                    {"results": [{"id": e["id"], "supports_claim": False} for e in items]}
                )
            )

    assert not EvidenceValidator().validate([evidence(1)], PlainModel())[0].supports_claim


@pytest.mark.parametrize("adapter", ["openai", "ollama"])
def test_batch_schema_and_id_mapping_through_http_adapters(adapter):
    import httpx

    from skala_agent.integrations.ollama import OllamaChat
    from skala_agent.integrations.openai import OpenAIResponses

    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        if adapter == "openai":
            messages = body["input"]
            schema = body["text"]["format"]["schema"]
            assert schema["required"] == ["results"]
            assert schema["$defs"]["ClaimResult"]["additionalProperties"] is False
        else:
            messages = body["messages"]
            assert "results" in body["format"]["properties"]
        items = json.loads(messages[1]["content"])["evidence"]
        output = json.dumps(
            {
                "results": [
                    {"id": e["id"], "supports_claim": e["id"] == "e1"} for e in reversed(items)
                ]
            }
        )
        response = (
            {
                "status": "completed",
                "output": [
                    {"type": "message", "content": [{"type": "output_text", "text": output}]}
                ],
            }
            if adapter == "openai"
            else {"done": True, "message": {"content": output}}
        )
        return httpx.Response(200, json=response)

    transport = httpx.MockTransport(handler)
    model = (
        OpenAIResponses("fixture", api_key="test", reasoning_effort="low", transport=transport)
        if adapter == "openai"
        else OllamaChat("qwen3:4b", transport=transport)
    )
    provider = EvaluationProvider(model, object())
    items = [evidence(1), evidence(2)]
    for _ in range(2):
        assert [e.supports_claim for e in provider.validate_evidence(items)] == [True, False]
    assert len(calls) == 1
