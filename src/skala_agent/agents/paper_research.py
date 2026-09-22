"""Primary 논문 청크에서 기술 개요·범위·한계·실험 조건을 추출합니다."""

from importlib.resources import files

from skala_agent.agents.citations import align_citations
from skala_agent.evidence import evidence_id
from skala_agent.integrations.contracts import EvaluationDraft, ModelOutputError, SearchDocument
from skala_agent.schemas import Evidence, TechAnalysis

RESEARCH_QUESTIONS = (
    ("overview", "논문이 제시하는 기술의 원리와 접근 방식은 무엇인가?"),
    ("scope", "어떤 적용 대상·workload·사용 범위를 논문에서 다루는가?"),
    ("limitations", "명시한 제약·가정·한계는 무엇인가?"),
    ("experiments", "보고된 실험 결과와 장비·모델·측정 조건은 무엇인가?"),
)


def research_technologies(technologies, retriever, model):
    analyses, evidence = {}, []
    for technology in technologies:
        if retriever is None:
            analyses[technology.id] = TechAnalysis(
                technology_id=technology.id,
                overview="논문 Retriever 미연결: 색인을 생성한 뒤 기술 조사를 실행하세요.",
            )
            continue
        documents = {}
        for _, question in RESEARCH_QUESTIONS:
            results = retriever.retrieve(
                f"{technology.name} {question}",
                top_k=2,
                role="primary",
                paper_id=technology.id,
            )
            for result in results:
                chunk = result.chunk
                # 잘못 구현된 Retriever가 반환한 타 기술·reference 자료도 경계에서 차단합니다.
                if chunk.role != "primary" or chunk.paper_id != technology.id:
                    raise ValueError("기술 조사는 해당 기술의 primary 논문만 사용할 수 있습니다.")
                documents.setdefault(
                    chunk.id,
                    SearchDocument(
                        id="chunk:" + chunk.id,
                        title=chunk.paper_id,
                        url=chunk.source_url,
                        content=chunk.text[:1600],
                        source_type="paper",
                        chunk_id=chunk.id,
                        page=chunk.page,
                        section_or_page=chunk.section,
                        paper_role=chunk.role,
                    ),
                )
        if not documents:
            analyses[technology.id] = TechAnalysis(
                technology_id=technology.id,
                overview="해당 기술의 primary 논문 검색 결과가 없어 기술 조사를 보류했습니다.",
            )
            continue
        documents = list(documents.values())[:6]
        prompt = files("skala_agent.prompts").joinpath("research.md").read_text(encoding="utf-8")
        draft = EvaluationDraft.model_validate(
            model.extract(
                prompt,
                {
                    "technology": technology.model_dump(),
                    "questions": [{"id": key, "text": text} for key, text in RESEARCH_QUESTIONS],
                    "sources": [d.model_dump(mode="json") for d in documents],
                    "exact_quotes": True,
                },
            )
        )
        draft = align_citations(draft, documents)
        ids = [f.question_id for f in draft.findings]
        if len(ids) != len(RESEARCH_QUESTIONS) or set(ids) != {q[0] for q in RESEARCH_QUESTIONS}:
            raise ModelOutputError("기술 조사 질문 ID가 누락되거나 중복되었습니다.")
        source_map = {d.id: d for d in documents}
        fields, collected = {}, {}
        for finding in draft.findings:
            if finding.answer != "yes" or not finding.citations:
                continue
            for citation in finding.citations:
                source = source_map.get(citation.source_id)
                if source is None or citation.quote not in source.content:
                    raise ModelOutputError("기술 조사 인용이 제공된 논문 원문과 일치하지 않습니다.")
                eid = evidence_id(
                    owner="research",
                    technology_id=technology.id,
                    claim=finding.rationale,
                    url=str(source.url),
                    chunk_id=source.chunk_id,
                )
                collected[eid] = Evidence(
                    id=eid,
                    technology_id=technology.id,
                    claim=finding.rationale,
                    url=source.url,
                    title=source.title,
                    source_type="paper",
                    excerpt=citation.quote,
                    chunk_id=source.chunk_id,
                    page=source.page,
                    section_or_page=source.section_or_page,
                )
            fields[finding.question_id] = finding.rationale
        analyses[technology.id] = TechAnalysis(
            technology_id=technology.id,
            overview=fields.get(
                "overview", "논문에서 기술 개요를 확인하지 못해 판단을 보류했습니다."
            ),
            scope=[fields["scope"]] if "scope" in fields else [],
            limitations=[fields["limitations"]] if "limitations" in fields else [],
            experiments=[fields["experiments"]] if "experiments" in fields else [],
            evidence_ids=list(collected),
            status="assessed" if "overview" in fields else "pending",
        )
        evidence.extend(collected.values())
    return analyses, evidence
