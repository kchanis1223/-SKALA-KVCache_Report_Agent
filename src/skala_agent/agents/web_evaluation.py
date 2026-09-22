"""모델·웹검색·논문 검색을 주입받는 4개 관점 평가. 공유 schema는 변경하지 않습니다."""

from datetime import date
from importlib.resources import files
from itertools import zip_longest

from skala_agent.agents.citations import align_citations, source_quote
from skala_agent.agents.context_rubrics import (
    DOMAIN_QUESTIONS,
    STAKEHOLDER_QUESTIONS,
    domain_details,
    stakeholder_details,
)
from skala_agent.agents.evaluation_rubrics import MARKET_QUESTIONS, TRL_QUESTIONS, market_details
from skala_agent.agents.source_scope import (
    ANCHORS,
    STAKEHOLDER_SEARCH_TERMS,
    explicit_sla_negative,
    mentions_technology,
)
from skala_agent.evidence import evidence_id
from skala_agent.integrations.contracts import (
    EvaluationDraft,
    ModelOutputError,
    SearchDocument,
    StructuredModel,
    WebSearch,
)
from skala_agent.integrations.tavily import document_id
from skala_agent.retrieval.interfaces import Retriever
from skala_agent.schemas import Assessment, Evidence, Signal, TRLDetails

SUPPORTED = ("trl", "market", "stakeholder", "domain")
QUESTIONS = {
    "trl": TRL_QUESTIONS,
    "market": MARKET_QUESTIONS,
    "stakeholder": STAKEHOLDER_QUESTIONS,
    "domain": DOMAIN_QUESTIONS,
}
PRIMARY = {"paper", "official", "market_report"}


def search_queries(perspective, technology, domain):
    anchor = ANCHORS.get(technology.id, f'"{technology.name}"')
    if perspective == "stakeholder":
        return [f"{anchor} {terms}" for terms in STAKEHOLDER_SEARCH_TERMS]
    suffixes = {
        "trl": (
            "prototype benchmark validation",
            "official repository serving integration",
            "product deployment production customer",
        ),
        "domain": (
            "datacenter cost GPU utilization power TCO benchmark conditions",
            "TTFT TPOT accuracy latency serving benchmark",
            "hardware integration multitenancy isolation failure independent review",
        ),
        "market": (
            "KV cache long context demand customers",
            "commercial adoption production",
            "framework ecosystem vendor support dependency",
        ),
    }
    return [f"{anchor} {suffix}" for suffix in suffixes[perspective]]


def load_prompt(perspective: str) -> str:
    if perspective not in SUPPORTED:
        raise ValueError("지원하지 않는 평가 관점입니다.")
    # 현재 시점을 사전학습 지식으로 추정해 2026년 자료를 임의로 배제하지 않게 합니다.
    prompt = files("skala_agent.prompts").joinpath(f"{perspective}.md").read_text(encoding="utf-8")
    return (
        f"평가 기준일: {date.today().isoformat()}. 현재 시점을 임의로 추정하지 마세요.\n" + prompt
    )


def bounded_documents(documents: list[SearchDocument]) -> list[SearchDocument]:
    # 원문 전체를 소형 모델에 넣지 않습니다. 웹 URL·논문 청크별 발췌를 최대 6건 사용합니다.
    unique = {}
    for document in documents:
        unique.setdefault((str(document.url), document.chunk_id), document)
    return [d.model_copy(update={"content": d.content[:800]}) for d in list(unique.values())[:6]]


class WebEvaluator:
    def __init__(
        self, model: StructuredModel, search: WebSearch, retriever: Retriever | None = None
    ):
        self.model = model
        self.search = search
        self.retriever = retriever

    def evaluate(self, perspective, technology, domain, analysis, existing):
        questions = QUESTIONS[perspective]
        batches = [
            self.search.search(query) for query in search_queries(perspective, technology, domain)
        ]
        documents = [
            d
            for row in zip_longest(*batches)
            for d in row
            if d is not None and mentions_technology(d, technology)
        ]
        rag_documents = []
        if perspective == "domain" and self.retriever is not None:
            for query in search_queries(perspective, technology, domain):
                for result in self.retriever.retrieve(query, top_k=3, role=None, paper_id=None):
                    chunk = result.chunk
                    rag_documents.append(
                        SearchDocument(
                            id="chunk:" + chunk.id,
                            title=chunk.paper_id,
                            url=chunk.source_url,
                            content=chunk.text,
                            source_type="paper",
                            chunk_id=chunk.id,
                            page=chunk.page,
                            section_or_page=chunk.section,
                            paper_role=chunk.role,
                        )
                    )
        # 이번 검색 결과에 없는 과거/추가 검색 근거도 후보에 포함합니다.
        documents.extend(
            SearchDocument(
                id="chunk:" + e.chunk_id if e.chunk_id else document_id(str(e.url)),
                title=e.title,
                url=e.url,
                content=e.excerpt,
                source_type=e.source_type,
                chunk_id=e.chunk_id,
                page=e.page,
                section_or_page=e.section_or_page,
            )
            for e in existing
            if e.technology_id == technology.id
        )
        # 재검색한 후보가 초기 검색 결과에 밀려 context 밖으로 사라지지 않게 합니다.
        recovered = [
            SearchDocument(
                id="chunk:" + e.chunk_id if e.chunk_id else document_id(str(e.url)),
                title=e.title,
                url=e.url,
                content=e.excerpt,
                source_type=e.source_type,
                chunk_id=e.chunk_id,
                page=e.page,
                section_or_page=e.section_or_page,
            )
            for e in reversed(existing)
            if e.technology_id == technology.id
            and e.id.startswith(perspective + "-")
            and e.claim.startswith(f"{technology.id}: 추가 검색 자료")
        ]
        # 논문과 웹이 서로를 전부 밀어내지 않도록 각 2건을 우선 예약합니다.
        reference = [d for d in rag_documents if d.paper_role == "reference"]
        primary = [d for d in rag_documents if d.paper_role == "primary"]
        rag = bounded_documents(reference[:1] + primary[:1] + rag_documents)
        web = bounded_documents(documents)
        documents = bounded_documents(recovered[:2] + rag[:2] + web[:2] + rag[2:] + web[2:])
        if documents:
            draft = self.model.extract(
                load_prompt(perspective),
                {
                    "technology": technology.model_dump(),
                    "domain": domain,
                    "tech_analysis": (
                        {
                            "overview": analysis.overview[:800],
                            "scope": [x[:200] for x in analysis.scope[:3]],
                            "limitations": [x[:200] for x in analysis.limitations[:3]],
                            "experiments": [x[:200] for x in analysis.experiments[:3]],
                        }
                        if analysis
                        else None
                    ),
                    "rag_available": self.retriever is not None
                    if perspective == "domain"
                    else None,
                    "questions": [{"id": q.id, "text": q.text} for q in questions],
                    "sources": [d.model_dump(mode="json") for d in documents],
                },
            )
            draft = align_citations(EvaluationDraft.model_validate(draft), documents)
            draft = repair_citations(
                self.model, perspective, technology, domain, questions, documents, draft
            )
        else:
            draft = EvaluationDraft(findings=[])
        result, sources = compile_assessment(
            perspective, technology, questions, documents, draft, existing
        )
        if perspective == "domain" and self.retriever is None:
            result.rationale += " 논문 Retriever 미연결: 웹 및 기존 근거만 사용했습니다."
        return result, sources


def repair_citations(model, perspective, technology, domain, questions, documents, draft):
    """원문과 다른 인용만 한 번 재추출합니다. 두 번째도 잘못되면 compile에서 실패합니다."""
    source_map = {d.id: d.content for d in documents}
    expected = {q.id for q in questions}
    ids = [f.question_id for f in draft.findings]
    if len(set(ids)) != len(ids) or set(ids) != expected:
        return draft  # 질문 누락/중복은 원래 계약 검증에서 거부합니다.
    bad = {
        f.question_id
        for f in draft.findings
        if f.answer != "unknown"
        and any(
            c.source_id not in source_map or c.quote not in source_map[c.source_id]
            for c in f.citations
        )
    }
    if not bad:
        return draft
    repaired = EvaluationDraft.model_validate(
        model.extract(
            load_prompt(perspective) + "\n이전 응답의 인용이 원문과 달랐습니다. "
            "이번 payload의 questions 개수만 답하세요. "
            "전체 관점의 질문 개수 안내는 이번 재추출에 적용하지 않습니다. "
            "원문 언어를 유지하고 번역하지 마세요. content에 있는 연속된 원문을 그대로 복사하세요. "
            "의역·문장 결합·생략 기호 추가를 금지합니다. "
            "정확한 인용을 못 찾으면 unknown과 빈 citations를 반환하세요.",
            {
                "technology": technology.model_dump(),
                "domain": domain,
                "questions": [{"id": q.id, "text": q.text} for q in questions if q.id in bad],
                "sources": [d.model_dump(mode="json") for d in documents],
            },
        )
    )
    repaired = align_citations(repaired, documents)
    replacements = {f.question_id: f for f in repaired.findings}
    if len(replacements) != len(repaired.findings) or set(replacements) != bad:
        raise ModelOutputError("인용 재추출 결과의 질문 ID가 다릅니다.")
    return EvaluationDraft(findings=[replacements.get(f.question_id, f) for f in draft.findings])


def compile_assessment(perspective, technology, questions, documents, draft, existing):
    """출처 ID·원문 발췌를 검사하고 모델 대신 등급을 집계합니다."""
    expected = {q.id for q in questions}
    answers = {f.question_id: f for f in draft.findings}
    if len(answers) != len(draft.findings) or not set(answers).issubset(expected):
        raise ModelOutputError("모델이 중복되거나 알 수 없는 질문 ID를 반환했습니다.")
    if documents and set(answers) != expected:
        raise ModelOutputError("모델 출력에 필수 조사 질문이 누락되었습니다.")
    source_map = {d.id: d for d in documents}
    old = {e.id: e for e in existing if e.technology_id == technology.id}
    evidence, signals, positive, response, reasons = {}, [], {}, {}, {}
    for question in questions:
        finding = answers.get(question.id)
        answer = finding.answer if finding else "unknown"
        measurements = finding.measurements if finding else []
        if (
            perspective == "domain"
            and question.id == "cost_measured"
            and answer == "yes"
            and not measurements
        ):
            answer = "unknown"
        citations = finding.citations if finding else []
        if answer == "unknown":
            citations = []
        references = []
        for citation in citations:
            document = source_map.get(citation.source_id)
            if document is None or citation.quote not in document.content:
                raise ModelOutputError(
                    "모델 인용이 제공한 출처 또는 원문 발췌와 일치하지 않습니다."
                )
            claim = f"{technology.name}: {question.text} 응답={answer}"
            eid = evidence_id(
                owner=perspective,
                technology_id=technology.id,
                claim=claim,
                url=str(document.url),
                chunk_id=document.chunk_id,
            )
            prior = old.get(eid)
            item = Evidence(
                id=eid,
                technology_id=technology.id,
                claim=claim,
                url=document.url,
                title=document.title,
                source_type=document.source_type,
                excerpt=citation.quote,
                chunk_id=document.chunk_id,
                page=document.page,
                section_or_page=document.section_or_page,
                # 추출과 의미적 검증은 별개입니다. 기존 검증과 발췌가 같을 때만 보존합니다.
                supports_claim=bool(
                    prior and prior.supports_claim and prior.excerpt == citation.quote
                ),
                confidence="low",
            )
            evidence[eid] = item
            references.append(item)
        for measurement in measurements:
            matching = [c.quote for c in citations if c.source_id == measurement.source_id]
            if not any(char.isdigit() for char in measurement.value):
                raise ModelOutputError("측정값에 수치가 없습니다.")
            if not any(
                all(
                    source_quote(quote, part) is not None
                    for part in (measurement.metric, measurement.value, measurement.conditions)
                )
                for quote in matching
            ):
                raise ModelOutputError("수치 또는 실험 조건이 연결된 원문 인용에 없습니다.")
        if (
            perspective == "domain"
            and answer == "no"
            and not explicit_sla_negative(question.id, [c.quote for c in citations])
        ):
            answer = "unknown"
            for item in references:
                evidence.pop(item.id, None)
            references = []
        # 인용 없는 yes/no를 확정 답변으로 인정하지 않습니다.
        if not references:
            answer = "unknown"
        grade = (
            "상"
            if any(e.source_type in PRIMARY for e in references)
            else "중"
            if references
            else "하"
        )
        response[question.id] = answer
        positive[question.id] = grade if answer == "yes" else "하"
        signals.append(
            Signal(
                question=f"[{question.id}] {question.text} (응답: {answer})",
                grade=grade,
                evidence_ids=list(dict.fromkeys(e.id for e in references)),
            )
        )
        if finding:
            reasons[question.id] = f"{question.id}={answer}: {finding.rationale}"
            if measurements:
                reasons[question.id] += " | " + " | ".join(
                    f"수치: {m.metric}={m.value}; 조건: {m.conditions}; 출처: {m.source_id}"
                    for m in measurements
                )

    if perspective == "trl":
        # 상용 통합/제품/운영 단계(7~9)는 1차 근거만 인정합니다.
        levels = [
            i
            for i in range(1, 10)
            if positive.get(f"trl_{i}") != "하" and (i < 7 or positive[f"trl_{i}"] == "상")
        ]
        level = max(levels, default=None)
        details = TRLDetails(level=level)
        verdict = f"TRL {level}" if level else "판단 보류"
        complete = level is not None
    elif perspective == "stakeholder":
        details = stakeholder_details(response, signals, reasons)
        complete = details.overall is not None
        verdict = details.overall or "판단 보류"
    elif perspective == "domain":
        details = domain_details(response)
        complete = (
            details.cost is not None
            and details.sla_risk != "판단 불가"
            and details.operations is not None
        )
        verdict = (
            f"원가: {details.cost or '판단 보류'} / SLA: {details.sla_risk} / "
            f"운영: {details.operations or '판단 보류'}"
        )
    else:
        details = market_details(positive, response.get("dependency", "unknown"), response)
        complete = all((details.demand, details.adoption, details.ecosystem))
        verdict = (
            f"수요: {details.demand or '판단 보류'} / "
            f"채택: {details.adoption or '판단 보류'} / "
            f"생태계: {details.ecosystem or '판단 보류'}"
        )
    sources = {str(e.url) for e in evidence.values()}
    result = Assessment(
        technology_id=technology.id,
        perspective=perspective,
        verdict=verdict,
        rationale=(
            "질문별 근거로 산출한 잠정 평가이며 최종 의미 검증이 필요합니다. "
            + " | ".join(reasons.values())
        )
        if reasons
        else "검색에서 평가 가능한 근거를 찾지 못했습니다.",
        confidence="medium"
        if complete
        and len(sources) >= 2
        and not (perspective == "domain" and details.cost == "개선 불명확")
        else "low",
        signals=signals,
        evidence_ids=list(evidence),
        details=details,
        status="assessed" if complete else "pending",
    )
    return result, list(evidence.values())
