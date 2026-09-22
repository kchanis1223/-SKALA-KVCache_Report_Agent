import json

import pytest
from pydantic import ValidationError

from skala_agent.retrieval.pipeline import DocumentSource, load_documents

PAYLOAD = {
    "documents": [
        {
            "paper_id": "itme",
            "camp": "hw",
            "role": "primary",
            "url": "https://arxiv.org/abs/2606.12556",
            "title": "ITME",
            "pages": 13,
            "local_path": "data/raw/itme.pdf",
        }
    ]
}


def write(tmp_path, payload):
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_load_documents_parses_known_fields(tmp_path):
    documents = load_documents(write(tmp_path, PAYLOAD))
    assert [d.paper_id for d in documents] == ["itme"]
    assert documents[0].camp == "hw" and documents[0].role == "primary"


def test_pdf_path_falls_back_to_paper_id(tmp_path):
    payload = {"documents": [dict(PAYLOAD["documents"][0])]}
    payload["documents"][0].pop("local_path")
    assert str(load_documents(write(tmp_path, payload))[0].pdf_path) == "data/raw/itme.pdf"


def test_unknown_camp_is_rejected():
    with pytest.raises(ValidationError):
        DocumentSource(paper_id="x", camp="fw", role="primary", url="https://example.com/x")


def test_real_config_is_loadable():
    documents = load_documents()
    assert {d.paper_id for d in documents} == {"turboquant", "turboquant_analysis", "itme"}
