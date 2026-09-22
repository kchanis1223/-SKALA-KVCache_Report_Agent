import json
import sys

import pytest

from skala_agent import adapters, evaluate_cli
from skala_agent.schemas import Assessment


def test_cli_rejects_tavily_without_key(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.setattr(sys, "argv", ["skala-evaluate"])
    with pytest.raises(SystemExit) as caught:
        evaluate_cli.main()
    assert caught.value.code == 2


@pytest.mark.parametrize(
    "perspective, count", [("trl", 2), ("stakeholder", 2), ("domain", 2), ("all", 8)]
)
def test_cli_reads_env_and_exports_two_technology_results(
    monkeypatch, tmp_path, perspective, count
):
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("USE_SINGLE_MODEL=true\nTAVILY_API_KEY=fixture\n")
    for key in ("USE_SINGLE_MODEL", "TAVILY_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    class Search:
        def __init__(self, api_key, **kwargs):
            assert api_key == "fixture"

        def search(self, query):
            return []

    monkeypatch.setattr(adapters, "TavilySearch", Search)
    output = tmp_path / "evaluations.json"
    monkeypatch.setattr(
        sys, "argv", ["skala-evaluate", "--perspective", perspective, "--output", str(output)]
    )
    evaluate_cli.main()
    result = json.loads(output.read_text())
    assessments = [Assessment.model_validate(a) for a in result["assessments"]]
    assert {a.technology_id for a in assessments} == {"turboquant", "itme"}
    assert len(assessments) == count
    assert all(a.status == "pending" for a in assessments)
    assert result["evidence"] == []
    assert set(result["models"].values()) == {"qwen3:4b"}
