"""설계서 4-5/4-6의 근거 질문과 결정론적 판정."""

from skala_agent.agents.evaluation_rubrics import Question
from skala_agent.schemas import DomainDetails, StakeholderDetails, StakeholderPosition

STAKEHOLDERS = {
    "gpu_vendor": "GPU·가속기 벤더",
    "memory_vendor": "메모리 벤더",
    "cloud_operator": "클라우드 사업자·운영자",
    "open_source": "서빙 프레임워크·오픈소스 커뮤니티",
    "investor": "투자 업계·애널리스트",
}
STAKEHOLDER_QUESTIONS = tuple(
    Question(f"{key}_{suffix}", f"{name}: {question}")
    for key, name in STAKEHOLDERS.items()
    for suffix, question in (
        ("support", "해당 기술의 채택·투자·지원 입장이 확인되는가?"),
        ("skeptical", "해당 기술의 한계·우려 또는 명시적 경쟁 입장이 확인되는가?"),
        ("aware", "해당 기술을 인지하고 언급한 자료가 있는가?"),
    )
)
DOMAIN_QUESTIONS = tuple(
    Question(key, text)
    for key, text in (
        (
            "cost_measured",
            "토큰당 원가·GPU 활용률·배치·전력·랙 밀도·TCO 개선 수치와 실험 조건이 있는가?",
        ),
        ("cost_conditional", "개선이 특정 workload·장비·배치·컨텍스트 조건에 한정되는가?"),
        ("cost_qualitative", "원가·효율 개선을 정성적으로 언급하는가?"),
        (
            "latency_degraded",
            "TTFT/TPOT 지연 악화가 보고되었는가? "
            "no는 두 지연의 악화 없음이 명시된 경우만 사용한다.",
        ),
        ("accuracy_loss", "정확도 손실이 보고되었는가? no는 정확도 보존이 명시된 경우만 사용한다."),
        ("sw_only", "기존 스택에서 SW 변경만으로 적용 가능하다고 명시하는가?"),
        ("hw_required", "HW·메모리·인터커넥트 추가 또는 변경이 필요한가?"),
        ("isolation_risk", "멀티테넌시 자원 격리·공유에 위험이 보고되었는가?"),
        ("blast_radius", "장애 반경이 노드를 넘어 확대되거나 운영 복잡도가 증가하는가?"),
    )
)


def stakeholder_details(answers, signals, reasons):
    positions = []
    for key in STAKEHOLDERS:
        ids = [f"{key}_{suffix}" for suffix in ("support", "skeptical", "aware")]
        support, skeptical, aware = [answers.get(q, "unknown") for q in ids]
        observed = support != "unknown" or skeptical != "unknown"
        stance = (
            "유보"
            if support == skeptical == "yes"
            else "지지"
            if support == "yes"
            else "회의적"
            if skeptical == "yes"
            else "유보"
            if aware == "yes" or observed
            else "자료 없음"
        )
        positions.append(
            StakeholderPosition(
                stakeholder=key,
                stance=stance,
                rationale=" | ".join(reasons[q] for q in ids if q in reasons)
                or "확인 가능한 입장 자료 없음",
                evidence_ids=list(
                    dict.fromkeys(
                        eid
                        for s in signals
                        if any(s.question.startswith(f"[{q}]") for q in ids)
                        for eid in s.evidence_ids
                    )
                ),
            )
        )
    observed = [p for p in positions if p.stance != "자료 없음"]
    overall = None
    if observed:
        overall = (
            "우호적"
            if sum(p.stance == "지지" for p in observed) > len(observed) / 2
            else "부정적"
            if sum(p.stance == "회의적" for p in observed) > len(observed) / 2
            else "혼재"
        )
    return StakeholderDetails(positions=positions, overall=overall)


def domain_details(answers):
    def yes(key):
        return answers.get(key) == "yes"

    cost = None
    if yes("cost_measured"):
        cost = "원가 개선 명확" if answers.get("cost_conditional") == "no" else "조건부 개선"
    elif any(
        answers.get(q, "unknown") != "unknown"
        for q in ("cost_measured", "cost_qualitative", "cost_conditional")
    ):
        cost = "개선 불명확"
    latency, accuracy = (answers.get(q, "unknown") for q in ("latency_degraded", "accuracy_loss"))
    sla = "판단 불가"
    if "unknown" not in (latency, accuracy):
        sla = ("낮음", "중간", "높음")[(latency == "yes") + (accuracy == "yes")]
    operations = (
        "인프라 전환 전제" if yes("hw_required") else "즉시 도입 가능" if yes("sw_only") else None
    )
    risks = []
    for key, label in (
        ("isolation_risk", "멀티테넌시 자원 격리 위험"),
        ("blast_radius", "장애 반경·운영 복잡도 증가"),
    ):
        if yes(key):
            risks.append(label)
    if yes("hw_required") and yes("sw_only"):
        risks.append("SW 단독 적용과 HW 변경 근거가 상충하므로 적용 조건 재검토 필요")
    return DomainDetails(cost=cost, sla_risk=sla, operations=operations, operational_risks=risks)
