"""4개 관점 평가 provider. 모델 객체만 주입하여 로컬/API 구현을 교체합니다."""

import json
import logging

from pydantic import BaseModel, ValidationError

from skala_agent.agents.paper_research import research_technologies
from skala_agent.agents.web_evaluation import SUPPORTED, WebEvaluator
from skala_agent.evaluation_contracts import validate_evaluation_output
from skala_agent.evidence import evidence_id
from skala_agent.integrations.contracts import ModelOutputError, ServiceConfigurationError
from skala_agent.integrations.structured import StructuredExtractor
from skala_agent.providers import DemoProvider
from skala_agent.schemas import AgentError, Assessment, Evidence

logger = logging.getLogger(__name__)


class ClaimSupport(BaseModel):
    supports_claim: bool


class EvaluationProvider(DemoProvider):
    def __init__(self, model=None, search=None, *, models=None, retriever=None):
        if search is None or (model is None) == (models is None):
            raise ValueError("search와 model 또는 models 중 하나를 지정하세요.")
        self.models = models
        self.validation_model = models.for_agent("validation") if models is not None else model
        self.evaluators = {
            perspective: WebEvaluator(
                StructuredExtractor(models.for_agent(perspective) if models is not None else model),
                search,
                retriever=retriever,
            )
            for perspective in SUPPORTED
        }
        self.search = search
        self.retriever = retriever
        self.research_model = StructuredExtractor(
            models.for_agent("research") if models is not None else model
        )

    def research(self, technologies):
        return research_technologies(technologies, self.retriever, self.research_model)

    def validate_evidence(self, evidence):
        """원문 발췌가 주장 자체를 직접·중립적으로 지지하는지 판정한다."""
        updates = []
        for item in evidence:
            messages = [
                {
                    "role": "system",
                    "content": (
                        "주장과 원문 발췌를 비교하세요. 발췌가 주장을 직접 지지하고 "
                        "홍보성·추측성 표현 없이 중립적으로 서술할 때만 supports_claim을 true로 "
                        "반환하세요. 그 외에는 false입니다. JSON만 반환하세요."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"claim": item.claim, "excerpt": item.excerpt}, ensure_ascii=False
                    ),
                },
            ]
            for attempt in range(2):
                response = (
                    self.validation_model.invoke_structured(
                        messages, ClaimSupport.model_json_schema()
                    )
                    if hasattr(self.validation_model, "invoke_structured")
                    else self.validation_model.invoke(messages)
                )
                text = response if isinstance(response, str) else getattr(response, "content", None)
                try:
                    verdict = ClaimSupport.model_validate_json(text)
                    updates.append(
                        item.model_copy(update={"supports_claim": verdict.supports_claim})
                    )
                    break
                except (ValidationError, TypeError, ValueError):
                    if attempt == 1:
                        raise ModelOutputError(
                            "근거 지지 판정 모델이 두 번 연속 유효한 출력을 반환하지 않았습니다."
                        ) from None
                    messages.append(
                        {
                            "role": "user",
                            "content": "supports_claim 불리언만 담은 유효한 JSON을 반환하세요.",
                        }
                    )
        return updates

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        if perspective not in SUPPORTED:
            return super().assess(perspective, technologies, domain, tech_analysis, evidence)
        assessments, collected = [], []
        for technology in technologies:
            try:
                assessment, sources = self.evaluators[perspective].evaluate(
                    perspective, technology, domain, tech_analysis.get(technology.id), evidence
                )
            except (
                TimeoutError,
                ConnectionError,
                ServiceConfigurationError,
                ModelOutputError,
            ) as exc:
                if isinstance(exc, ModelOutputError):
                    logger.warning(
                        "평가 출력 검증 실패 (%s/%s): %s",
                        perspective,
                        technology.id,
                        exc,
                    )
                assessment = Assessment(
                    technology_id=technology.id,
                    perspective=perspective,
                    verdict="판단 보류",
                    rationale="검색 또는 평가 모델 호출에 실패했습니다.",
                    status="failed",
                    error=AgentError(
                        code="ModelOutputError"
                        if isinstance(exc, ModelOutputError)
                        else type(exc).__name__,
                        message="검색·모델 설정 및 응답을 확인하세요.",
                        retryable=isinstance(exc, (TimeoutError, ConnectionError)),
                    ),
                )
                sources = []
            assessments.append(assessment)
            collected.extend(sources)
        return validate_evaluation_output(perspective, technologies, assessments, collected)

    def search_missing(self, missing):
        collected = {}
        for item in missing:
            if item.perspective not in SUPPORTED or not item.retryable:
                continue
            # 1회 재검색에서 기술·관점당 최대 2질의. 외부 backoff는 workflow 담당 범위.
            for query in list(dict.fromkeys(item.queries))[:2]:
                for document in self.search.search(query):
                    claim = f"{item.technology_id}: 추가 검색 자료 — {document.title}"
                    eid = evidence_id(
                        owner=item.perspective,
                        technology_id=item.technology_id,
                        claim=claim,
                        url=str(document.url),
                    )
                    collected[eid] = Evidence(
                        id=eid,
                        technology_id=item.technology_id,
                        claim=claim,
                        url=document.url,
                        title=document.title,
                        source_type=document.source_type,
                        excerpt=document.content[:800],
                    )
        return list(collected.values())
