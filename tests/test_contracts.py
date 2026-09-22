import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from skala_agent.agents.validation import run, valid_sources
from skala_agent.evidence import evidence_id
from skala_agent.retrieval.config import DEFAULT_CHUNKING, ChunkingConfig, chunk_id
from skala_agent.schemas import (
    Assessment,
    Chunk,
    Evidence,
    MissingEvidence,
    RetrievalResult,
    TechAnalysis,
)
from skala_agent.workflow.state import merge_evidence

FIXTURES = Path(__file__).parent / "fixtures" / "contracts.json"


def test_shared_fixtures_validate_and_roundtrip():
    models = {
        "chunk": Chunk,
        "retrieval": RetrievalResult,
        "evidence": Evidence,
        "tech_analysis": TechAnalysis,
        "normal": Assessment,
        "pending": Assessment,
        "failed": Assessment,
        "missing": MissingEvidence,
    }
    examples = json.loads(FIXTURES.read_text())
    for key, model in models.items():
        result = model.model_validate(examples[key])
        assert model.model_validate_json(result.model_dump_json()) == result


def test_retracted_evidence_no_longer_validates_or_duplicates():
    examples = json.loads(FIXTURES.read_text())
    evidence = Evidence.model_validate(examples["evidence"])
    assessment = Assessment.model_validate(examples["normal"])
    updated = evidence.model_copy(update={"supports_claim": False})
    merged = merge_evidence([evidence], [updated])
    assert len(merged) == 1
    assert not valid_sources(assessment, merged)
    missing = run({"synthesis": [assessment], "evidence": merged})["missing_evidence"][0]
    assert missing.kind == "unsupported_claim"
    assert missing.claim == assessment.verdict
    assert missing.queries and missing.evidence_ids == [evidence.id]


def test_evidence_id_is_stable_and_namespaced():
    kwargs = dict(
        owner="trl",
        technology_id="turboquant",
        claim="fixture",
        url="https://example.org/paper",
        chunk_id="turboquant-p1-1",
    )
    assert evidence_id(**kwargs) == evidence_id(**kwargs)
    assert evidence_id(**kwargs) != evidence_id(**{**kwargs, "owner": "market"})
    assert evidence_id(**kwargs) != evidence_id(**{**kwargs, "claim": "changed"})


def test_same_id_cannot_replace_an_unrelated_claim():
    evidence = Evidence.model_validate(json.loads(FIXTURES.read_text())["evidence"])
    with pytest.raises(ValueError, match="ID 충돌"):
        merge_evidence([evidence], [evidence.model_copy(update={"claim": "unrelated"})])


def test_failure_requires_error_and_cannot_be_successful_verdict():
    base = dict(technology_id="itme", perspective="trl", verdict="TRL 9", rationale="test")
    with pytest.raises(ValidationError):
        Assessment(**base, status="failed")
    failed = Assessment(
        **base,
        status="failed",
        confidence="high",
        error={"code": "TimeoutError", "message": "요청 실패"},
    )
    assert failed.verdict == "판단 보류" and failed.confidence == "low"
    with pytest.raises(ValidationError):
        Assessment(**base, status="assessed", error=failed.error)
    with pytest.raises(ValidationError):
        Assessment(**base, details={"perspective": "market"})


def test_chunking_contract_and_invalid_score():
    assert DEFAULT_CHUNKING.chunk_size_tokens == 1500
    assert DEFAULT_CHUNKING.overlap_tokens == 200
    assert chunk_id("turboquant", 2, 1) == "turboquant-p2-1"
    with pytest.raises(ValueError):
        chunk_id("turboquant", 0, 1)
    with pytest.raises(ValidationError):
        ChunkingConfig(chunk_size_tokens=100, overlap_tokens=100)
    example = json.loads(FIXTURES.read_text())["retrieval"]
    with pytest.raises(ValidationError):
        RetrievalResult.model_validate({**example, "score": float("nan")})


def test_comment_field_names_are_accepted_but_canonical_names_are_serialized():
    examples = json.loads(FIXTURES.read_text())
    item = examples["normal"]
    item["reason"] = item.pop("rationale")
    assert "rationale" in Assessment.model_validate(item).model_dump()
    item = examples["evidence"]
    item["evidence_text"] = item.pop("excerpt")
    assert "excerpt" in Evidence.model_validate(item).model_dump()
