import pytest

from skala_agent.evaluation.retrieval import retrieval_metrics


def test_ranked_retrieval_with_misses_and_multiple_relevant_chunks():
    result = retrieval_metrics(
        [
            (["a", "b"], {"a", "b"}),
            (["x", "b"], {"b"}),
            (["x"], {"z"}),
        ]
    )
    assert result == pytest.approx({"hit@1": 1 / 3, "hit@3": 2 / 3, "mrr": 0.5})


def test_empty_ground_truth_is_rejected():
    with pytest.raises(ValueError):
        retrieval_metrics([(["a"], set())])
