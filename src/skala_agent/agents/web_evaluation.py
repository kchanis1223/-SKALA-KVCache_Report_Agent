"""모델/검색 서비스를 주입받는 TRL·시장성 평가. 공유 schema는 변경하지 않습니다."""

from importlib.resources import files

from skala_agent.agents.evaluation_rubrics import MARKET_QUESTIONS, TRL_QUESTIONS, market_details
from skala_agent.evidence import evidence_id
from skala_agent.integrations.contracts import (
    EvaluationDraft,
    ModelOutputError,
    SearchDocument,
    StructuredModel,
    WebSearch,
)
from skala_agent.integrations.tavily import document_id
from skala_agent.schemas import Assessment, Evidence, Signal, TRLDetails

SUPPORTED = ("trl", "market")
PRIMARY = {"paper", "official", "market_report"}


def search_queries(perspective, technology, domain):
    suffixes = {
        "trl": (
            "prototype benchmark validation",
            "official repository serving integration",
            "product deployment production customer",
        ),
        "market": (
            "KV cache long context demand customers",
            "commercial adoption production",
            "framework ecosystem vendor support dependency",
        ),
    }
    return [f'"{technology.name}" {domain} {suffix}' for suffix in suffixes[perspective]]


def load_prompt(perspective: str) -> str:
    if perspective not in SUPPORTED:
        raise ValueError("TRL 및 시장성 관점만 지원합니다.")
    return files("skala_agent.prompts").joinpath(f"{perspective}.md").read_text(encoding="utf-8")


def bounded_documents(documents: list[SearchDocument]) -> list[SearchDocument]:
    # 원문 전체를 소형 모델에 넣지 않습니다. URL별 첫 검색 발췌를 최대 6건 사용합니다.
    unique = {}
    for document in documents:
        unique.setdefault(str(document.url), document)
    return [d.model_copy(update={"content": d.content[:800]}) for d in list(unique.values())[:6]]


class WebEvaluator:
    def __init__(self, model: StructuredModel, search: WebSearch):
        self.model = model
        self.search = search

    def evaluate(self, perspective, technology, domain, analysis, existing):
        questions = TRL_QUESTIONS if perspective == "trl" else MARKET_QUESTIONS
        documents = []
        for query in search_queries(perspective, technology, domain):
            documents.extend(self.search.search(query))
        # 이번 검색 결과에 없는 과거/추가 검색 근거도 후보에 포함합니다.
        documents.extend(
            SearchDocument(
                id=document_id(str(e.url)),
                title=e.title,
                url=e.url,
                content=e.excerpt,
                source_type=e.source_type,
            )
            for e in existing
            if e.technology_id == technology.id
        )
        # 재검색한 후보가 초기 검색 결과에 밀려 context 밖으로 사라지지 않게 합니다.
        recovered = [
            SearchDocument(
                id=document_id(str(e.url)),
                title=e.title,
                url=e.url,
                content=e.excerpt,
                source_type=e.source_type,
            )
            for e in reversed(existing)
            if e.technology_id == technology.id
            and e.id.startswith(perspective + "-")
            and e.claim.startswith(f"{technology.id}: 추가 검색 자료")
        ]
        documents = bounded_documents(recovered[:2] + documents)
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
                    "questions": [{"id": q.id, "text": q.text} for q in questions],
                    "sources": [d.model_dump(mode="json") for d in documents],
                },
            )
            draft = EvaluationDraft.model_validate(draft)
        else:
            draft = EvaluationDraft(findings=[])
        return compile_assessment(perspective, technology, questions, documents, draft, existing)


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
    evidence, signals, positive, response, reasons = {}, [], {}, {}, []
    for question in questions:
        finding = answers.get(question.id)
        answer = finding.answer if finding else "unknown"
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
                owner=perspective, technology_id=technology.id, claim=claim, url=str(document.url)
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
                # 추출과 의미적 검증은 별개입니다. 기존 검증과 발췌가 같을 때만 보존합니다.
                supports_claim=bool(
                    prior and prior.supports_claim and prior.excerpt == citation.quote
                ),
                confidence="low",
            )
            evidence[eid] = item
            references.append(item)
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
            reasons.append(f"{question.id}={answer}: {finding.rationale}")

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
            "질문별 근거로 산출한 잠정 평가이며 최종 의미 검증이 필요합니다. " + " | ".join(reasons)
        )
        if reasons
        else "검색에서 평가 가능한 근거를 찾지 못했습니다.",
        confidence="medium" if complete and len(sources) >= 2 else "low",
        signals=signals,
        evidence_ids=list(evidence),
        details=details,
        status="assessed" if complete else "pending",
    )
    return result, list(evidence.values())
