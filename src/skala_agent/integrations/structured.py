"""Ollama JSON Schema 출력과 일반 invoke(messages) 모델 객체를 지원."""

import json

from pydantic import ValidationError

from skala_agent.agents.citations import quote_options
from skala_agent.integrations.contracts import EvaluationDraft, ModelOutputError


class StructuredExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, system: str, payload: dict) -> EvaluationDraft:
        response_schema = EvaluationDraft.model_json_schema()
        question_ids = [q["id"] for q in payload.get("questions", [])]
        if question_ids:
            response_schema["properties"]["findings"].update(
                minItems=len(question_ids), maxItems=len(question_ids)
            )
            response_schema["$defs"]["Finding"]["properties"]["question_id"]["enum"] = question_ids
        source_ids = [source["id"] for source in payload.get("sources", [])]
        if source_ids:
            for definition in ("CitationDraft", "MeasurementDraft"):
                response_schema["$defs"][definition]["properties"]["source_id"]["enum"] = source_ids
        if payload.get("exact_quotes") and source_ids:
            # 자유 생성으로 다시 의역하지 못하게 출처와 실제 부분문자열을 함께 제한합니다.
            response_schema["$defs"]["CitationDraft"] = {
                "anyOf": [
                    {
                        "type": "object",
                        "properties": {
                            "source_id": {"type": "string", "const": source["id"]},
                            "quote": {"type": "string", "enum": quote_options(source["content"])},
                        },
                        "required": ["source_id", "quote"],
                        "additionalProperties": False,
                    }
                    for source in payload["sources"]
                    if source["content"].strip()
                ]
            }
        # native format에 전달한 큰 스키마를 system에도 복제하면 입력 문맥을 소모합니다.
        schema_instruction = (
            ""
            if hasattr(self.model, "invoke_structured")
            else "\n출력 JSON Schema (답변에 복사하지 마세요):\n"
            + json.dumps(response_schema, ensure_ascii=False)
        )
        example = json.dumps(
            {
                "findings": [
                    {
                        "question_id": "example_id",
                        "answer": "unknown",
                        "rationale": "No evidence provided",
                        "citations": [],
                    }
                ]
            }
        )
        messages = [
            {
                "role": "system",
                "content": (
                    system
                    + schema_instruction
                    + "\nReturn an INSTANCE, not the schema. The only top-level key is findings."
                    + " Do not output $defs, properties, type, or required.\nExample answer:\n"
                    + example
                    + "\nReplace example_id with the supplied question IDs."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            if hasattr(self.model, "invoke_structured"):
                response = self.model.invoke_structured(messages, response_schema)
            else:
                response = self.model.invoke(messages)
            text = response if isinstance(response, str) else getattr(response, "content", None)
            try:
                draft = EvaluationDraft.model_validate_json(text)
                ids = [f.question_id for f in draft.findings]
                if question_ids and (
                    len(ids) != len(question_ids) or set(ids) != set(question_ids)
                ):
                    raise ValueError("질문 개수 또는 ID 불일치")
                return draft
            except (ValidationError, TypeError, ValueError):
                if attempt == 0:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "유효한 JSON을 반환하지 않았습니다. "
                                "지정 schema에 맞게 다시 출력하세요. "
                                "코드블록 없이 JSON만 반환하고 근거가 없으면 unknown으로 답하세요."
                            ),
                        }
                    )
        raise ModelOutputError("모델이 두 번 연속 유효한 구조화 출력을 반환하지 않았습니다.")
