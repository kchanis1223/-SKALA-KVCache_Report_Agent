import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from skala_agent.evaluation_contracts import EvaluationOutput
from skala_agent.schemas import Assessment, Evidence

EXAMPLES = json.loads((Path(__file__).parent / "fixtures/evaluation_outputs.json").read_text())


@pytest.mark.parametrize("status", ["normal", "pending", "failed"])
@pytest.mark.parametrize("index", range(4))
def test_four_perspective_examples_roundtrip(status, index):
    output = EvaluationOutput.model_validate(EXAMPLES[status][index])
    assert EvaluationOutput.model_validate_json(output.model_dump_json()) == output
    assert (
        output.assessments[0].status
        == {"normal": "assessed", "pending": "pending", "failed": "failed"}[status]
    )


def invalid_payload(case):
    data = copy.deepcopy(EXAMPLES["normal"][0])
    assessment = data["assessments"][0]
    if case == "missing_technology":
        data["technology_ids"].append("missing")
    elif case == "duplicate_result":
        data["assessments"].append(copy.deepcopy(assessment))
    elif case == "wrong_perspective":
        data["perspective"] = "market"
    elif case == "missing_evidence":
        data["evidence"].pop()
    elif case == "cross_technology":
        data["evidence"][0]["technology_id"] = "another"
    elif case == "duplicate_evidence":
        data["evidence"].append(copy.deepcopy(data["evidence"][0]))
    elif case == "dangling_signal":
        assessment["signals"][0]["evidence_ids"] = ["invented"]
    elif case == "uncited_signal":
        assessment["signals"][0]["evidence_ids"] = []
    elif case == "unknown_field":
        assessment["confidance"] = "high"
    elif case == "blank_rationale":
        assessment["rationale"] = "  "
    elif case == "empty_excerpt":
        data["evidence"][0]["excerpt"] = "\n "
    elif case == "invalid_page":
        data["evidence"][0]["page"] = 0
    elif case == "invalid_confidence":
        assessment["confidence"] = 0.9
    elif case == "missing_details":
        assessment["details"] = None
    elif case == "incomplete_assessed":
        assessment["details"]["level"] = None
    elif case == "failed_without_error":
        assessment["status"] = "failed"
    elif case == "duplicate_question":
        assessment["signals"].append(copy.deepcopy(assessment["signals"][0]))
    return data


@pytest.mark.parametrize(
    "case",
    [
        "missing_technology",
        "duplicate_result",
        "wrong_perspective",
        "missing_evidence",
        "cross_technology",
        "duplicate_evidence",
        "dangling_signal",
        "uncited_signal",
        "unknown_field",
        "blank_rationale",
        "empty_excerpt",
        "invalid_page",
        "invalid_confidence",
        "missing_details",
        "incomplete_assessed",
        "failed_without_error",
        "duplicate_question",
    ],
)
def test_invalid_outputs_are_rejected(case):
    with pytest.raises(ValidationError):
        EvaluationOutput.model_validate(invalid_payload(case))


def test_single_url_multiple_chunks_is_low_without_mutating_input():
    data = copy.deepcopy(EXAMPLES["normal"][0])
    data["evidence"][1]["url"] = data["evidence"][0]["url"]
    original = Assessment.model_validate(data["assessments"][0])
    data["assessments"] = [original]
    output = EvaluationOutput.model_validate(data)
    assert output.assessments[0].confidence == "low"
    assert original.confidence == "medium"
    assert len(output.evidence) == 2


def test_unrelated_second_source_does_not_inflate_confidence():
    data = copy.deepcopy(EXAMPLES["normal"][0])
    a = data["assessments"][0]
    a["evidence_ids"] = a["evidence_ids"][:1]
    a["signals"][0]["evidence_ids"] = a["evidence_ids"]
    assert EvaluationOutput.model_validate(data).assessments[0].confidence == "low"


@pytest.mark.parametrize("case", ["missing", "duplicate", "bad_overall", "no_citation"])
def test_stakeholder_positions_are_consistent(case):
    data = copy.deepcopy(EXAMPLES["normal"][2])
    details = data["assessments"][0]["details"]
    if case == "missing":
        details["positions"].pop()
    elif case == "duplicate":
        details["positions"][-1] = copy.deepcopy(details["positions"][0])
    elif case == "bad_overall":
        details["overall"] = "부정적"
    else:
        next(p for p in details["positions"] if p["stance"] == "지지")["evidence_ids"] = []
    with pytest.raises(ValidationError):
        EvaluationOutput.model_validate(data)


def test_pending_is_low_and_model_copy_cannot_bypass_validation():
    data = copy.deepcopy(EXAMPLES["pending"][0])
    data["assessments"][0]["confidence"] = "high"
    assert EvaluationOutput.model_validate(data).assessments[0].confidence == "low"
    data = copy.deepcopy(EXAMPLES["normal"][0])
    data["evidence"][0] = Evidence.model_validate(data["evidence"][0]).model_copy(
        update={"page": 0}
    )
    with pytest.raises(ValidationError):
        EvaluationOutput.model_validate(data)


def test_schema_is_exportable_and_discriminates_details():
    schema = EvaluationOutput.model_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["$defs"]["Assessment"]["additionalProperties"] is False
    assert "discriminator" in json.dumps(schema["$defs"]["Assessment"]["properties"]["details"])
