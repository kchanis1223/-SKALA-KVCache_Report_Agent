"""invoke(messages) 인터페이스면 Transformers와 LangChain 모델을 동일하게 사용."""

import json

from pydantic import ValidationError

from skala_agent.integrations.contracts import EvaluationDraft, ModelOutputError


class StructuredExtractor:
    def __init__(self, model):
        self.model = model

    def extract(self, system: str, payload: dict) -> EvaluationDraft:
        schema = json.dumps(EvaluationDraft.model_json_schema(), ensure_ascii=False)
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
                    + "\n출력 JSON Schema (답변에 복사하지 마세요):\n"
                    + schema
                    + "\nReturn an INSTANCE, not the schema. The only top-level key is findings."
                    + " Do not output $defs, properties, type, or required.\nExample answer:\n"
                    + example
                    + "\nReplace example_id with the supplied question IDs."
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        for attempt in range(2):
            response = self.model.invoke(messages)
            text = response if isinstance(response, str) else getattr(response, "content", None)
            try:
                return EvaluationDraft.model_validate_json(text)
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
