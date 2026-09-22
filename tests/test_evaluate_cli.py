import json
import sys

import pytest

from skala_agent import evaluate_cli
from skala_agent.schemas import Assessment, Evidence


def test_cli_requires_search_key_before_loading_model(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["skala-evaluate"])
    with pytest.raises(SystemExit) as caught:
        evaluate_cli.main()
    assert caught.value.code == 2


def test_cli_exports_schema_valid_two_technology_results(monkeypatch, tmp_path):
    output = tmp_path / "evaluations.json"
    monkeypatch.setenv("TAVILY_API_KEY", "fixture")
    monkeypatch.setattr(
        sys, "argv", ["skala-evaluate", "--perspective", "trl", "--output", str(output)]
    )

    class Model:
        def __init__(self, **kwargs):
            pass

        def invoke(self, messages):
            raise AssertionError("empty search must not invoke model")

    class Search:
        def __init__(self, *args, **kwargs):
            pass

        def search(self, query):
            return []

    monkeypatch.setattr(evaluate_cli, "TransformersQwen", Model)
    monkeypatch.setattr(evaluate_cli, "TavilySearch", Search)
    evaluate_cli.main()
    result = json.loads(output.read_text())
    assessments = [Assessment.model_validate(a) for a in result["assessments"]]
    assert {a.technology_id for a in assessments} == {"turboquant", "itme"}
    assert all(a.status == "pending" for a in assessments)
    assert [Evidence.model_validate(e) for e in result["evidence"]] == []
