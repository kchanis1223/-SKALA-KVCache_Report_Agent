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


class SingleClaimSupport(BaseModel):
    id: str
    supports_claim: bool


class BatchClaimSupport(BaseModel):
    results: list[SingleClaimSupport]


class EvaluationProvider(DemoProvider):
    def __init__(self, model=None, search=None, *, models=None, retriever=None):
        if search is None or (model is None) == (models is None):
            raise ValueError("search와 model 또는 models 중 하나를 지정하세요.")
        self.models = models
        self.validation_model = models.for_agent("validation") if models is not None else model
        self.synthesis_model = (
            models.for_agent("synthesis")
            if models is not None and not models.settings.use_single_model
            else None
        )
        self.report_model = (
            models.for_agent("report")
            if models is not None and not models.settings.use_single_model
            else None
        )
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
        self._validation_cache: dict[tuple[str, str, str, str], bool] = {}

    def _make_cache_key(self, item: Evidence, model_id: str) -> tuple[str, str, str, str]:
        return (item.id, item.claim, item.excerpt, model_id)

    def research(self, technologies):
        return research_technologies(technologies, self.retriever, self.research_model)

    def validate_evidence(self, evidence: list[Evidence], batch_size: int = 10) -> list[Evidence]:
        """원문 발췌가 주장 자체를 직접·중립적으로 지지하는지 판정한다 (배치 + 캐싱 적용)."""
        if not evidence:
            return []

        model_attr = getattr(self.validation_model, "model", None) or getattr(
            self.validation_model, "name", "default"
        )
        model_id = str(model_attr)

        cached_updates: list[Evidence] = []
        missing_items: list[Evidence] = []

        for item in evidence:
            key = self._make_cache_key(item, model_id)
            if key in self._validation_cache:
                supports = self._validation_cache[key]
                cached_updates.append(item.model_copy(update={"supports_claim": supports}))
            else:
                missing_items.append(item)

        if not missing_items:
            return cached_updates

        batched_updates: list[Evidence] = []
        for i in range(0, len(missing_items), batch_size):
            chunk = missing_items[i : i + batch_size]

            if len(chunk) == 1:
                item = chunk[0]
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "주장과 원문 발췌를 비교하세요. 발췌가 주장을 직접 지지하고 "
                            "홍보성·추측성 표현 없이 중립적으로 서술할 때만 supports_claim을 "
                            "true로 반환하세요. 그 외에는 false입니다. JSON만 반환하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {"claim": item.claim, "excerpt": item.excerpt}, ensure_ascii=False
                        ),
                    },
                ]
                supports = False
                for attempt in range(2):
                    response = (
                        self.validation_model.invoke_structured(
                            messages, ClaimSupport.model_json_schema()
                        )
                        if hasattr(self.validation_model, "invoke_structured")
                        else self.validation_model.invoke(messages)
                    )
                    text = (
                        response
                        if isinstance(response, str)
                        else getattr(response, "content", None)
                    )
                    try:
                        verdict = ClaimSupport.model_validate_json(text)
                        supports = verdict.supports_claim
                        break
                    except (ValidationError, TypeError, ValueError):
                        try:
                            parsed = BatchClaimSupport.model_validate_json(text)
                            verdict_map = {res.id: res.supports_claim for res in parsed.results}
                            if item.id in verdict_map:
                                supports = verdict_map[item.id]
                                break
                        except (ValidationError, TypeError, ValueError):
                            pass

                        if attempt == 1:
                            raise ModelOutputError(
                                "근거 지지 판정 모델이 두 번 연속 "
                                "유효한 출력을 반환하지 않았습니다."
                            ) from None
                        messages.append(
                            {
                                "role": "user",
                                "content": "supports_claim 불리언만 담은 유효한 JSON을 반환하세요.",
                            }
                        )
                key = self._make_cache_key(item, model_id)
                self._validation_cache[key] = supports
                batched_updates.append(item.model_copy(update={"supports_claim": supports}))
            else:
                payload = {
                    "claim": "batch_validation",
                    "evidence": [
                        {"id": item.id, "claim": item.claim, "excerpt": item.excerpt}
                        for item in chunk
                    ],
                }
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "주장과 원문 발췌 목록을 비교하세요. 발췌가 주장을 직접 지지하고 "
                            "홍보성·추측성 표현 없이 중립적으로 서술할 때만 해당 id의 "
                            "supports_claim을 true로 반환하세요. 그 외에는 false입니다. "
                            '결과는 {"results": [ {"id": "...", "supports_claim": true/false} ]} '
                            "유효한 JSON만 반환하세요."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(payload, ensure_ascii=False),
                    },
                ]

                verdict_map: dict[str, bool] = {}
                for attempt in range(2):
                    response = (
                        self.validation_model.invoke_structured(
                            messages, BatchClaimSupport.model_json_schema()
                        )
                        if hasattr(self.validation_model, "invoke_structured")
                        else self.validation_model.invoke(messages)
                    )
                    text = (
                        response
                        if isinstance(response, str)
                        else getattr(response, "content", None)
                    )
                    try:
                        parsed = BatchClaimSupport.model_validate_json(text)
                        verdict_map = {res.id: res.supports_claim for res in parsed.results}
                        break
                    except (ValidationError, TypeError, ValueError):
                        try:
                            single_verdict = ClaimSupport.model_validate_json(text)
                            verdict_map = {item.id: single_verdict.supports_claim for item in chunk}
                            break
                        except (ValidationError, TypeError, ValueError):
                            pass

                        if attempt == 1:
                            raise ModelOutputError(
                                "근거 지지 판정 모델이 두 번 연속 "
                                "유효한 출력을 반환하지 않았습니다."
                            ) from None
                        messages.append(
                            {
                                "role": "user",
                                "content": "results 리스트를 담은 유효한 JSON을 반환하세요.",
                            }
                        )

                for item in chunk:
                    supports = verdict_map.get(item.id, False)
                    key = self._make_cache_key(item, model_id)
                    self._validation_cache[key] = supports
                    batched_updates.append(item.model_copy(update={"supports_claim": supports}))

        all_updated_map = {item.id: item for item in (cached_updates + batched_updates)}
        return [all_updated_map[item.id] for item in evidence if item.id in all_updated_map]

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
