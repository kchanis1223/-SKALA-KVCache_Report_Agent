import json

import pytest
from pydantic import ValidationError

from skala_agent.evaluation.dataset import EvalSet, load_eval_set
from skala_agent.evaluation.retrieval_cli import build_parser

BASE = {
    "version": "v1",
    "chunking_version": "v1",
    "queries": [{"id": "q01", "query": "질문", "gold_chunk_ids": ["itme-p1-1"]}],
}


def test_real_eval_set_has_twenty_queries():
    eval_set = load_eval_set()
    assert len(eval_set.queries) == 20
    assert eval_set.chunking_version == "v1"


def test_real_eval_set_covers_all_three_papers():
    papers = {gold.rsplit("-p", 1)[0] for gold in load_eval_set().gold_ids()}
    assert papers == {"turboquant", "turboquant_analysis", "itme"}


def test_check_against_index_reports_missing_gold():
    eval_set = EvalSet.model_validate(BASE)
    assert eval_set.check_against_index({"itme-p1-1"}) == []
    assert eval_set.check_against_index({"other"}) == ["itme-p1-1"]


def test_duplicate_query_id_is_rejected():
    payload = json.loads(json.dumps(BASE))
    payload["queries"] = payload["queries"] * 2
    with pytest.raises(ValidationError):
        EvalSet.model_validate(payload)


def test_empty_gold_is_rejected():
    payload = json.loads(json.dumps(BASE))
    payload["queries"][0]["gold_chunk_ids"] = []
    with pytest.raises(ValidationError):
        EvalSet.model_validate(payload)


def test_duplicate_gold_is_rejected():
    payload = json.loads(json.dumps(BASE))
    payload["queries"][0]["gold_chunk_ids"] = ["itme-p1-1", "itme-p1-1"]
    with pytest.raises(ValidationError):
        EvalSet.model_validate(payload)


def test_cli_defaults_use_rank_depth_ten():
    args = build_parser().parse_args([])
    assert args.top_k == 10
    assert args.exclude_references is False
    assert args.model is None
