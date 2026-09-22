"""설계서 4-3/4-4의 질문과 결정론적 집계 규칙. 최종 근거 검증은 #11 담당."""

from dataclasses import dataclass

from skala_agent.schemas import MarketDetails


@dataclass(frozen=True)
class Question:
    id: str
    text: str


TRL_QUESTIONS = tuple(
    Question(f"trl_{i}", text)
    for i, text in enumerate(
        (
            "KV cache 최적화 기초 원리·아이디어가 제시되었는가?",
            "구체적인 기술 개념과 적용 방식이 정의되었는가?",
            "개념 검증 또는 소규모 실험실 PoC 결과가 있는가?",
            "부품·구성요소 수준의 측정·검증 결과가 있는가?",
            "데이터센터와 유사한 실제 HW 환경에서 통합 성능이 검증되었는가?",
            "실제 LLM serving workload와 유사한 조건에서 시스템·시제품이 시연되었는가?",
            "실제 serving stack·운용 환경에 prototype이 통합되어 검증되었는가?",
            "제품·서비스가 완성되어 실제 데이터센터 배포 준비·안정성 검증을 마쳤는가?",
            "클라우드·데이터센터의 지속적인 상용 운영·고객 사용 사례가 확인되는가?",
        ),
        1,
    )
)

MARKET_QUESTIONS = (
    Question("demand_1", "KV cache 메모리 병목이 데이터센터 비용·성능 문제로 언급되는가?"),
    Question("demand_2", "long-context 및 대규모 LLM serving 수요가 증가 추세인가?"),
    Question("demand_3", "해당 SW 압축/HW 확장 접근에 비용을 지불할 고객군이 식별되는가?"),
    Question("adoption_1", "해당 기술이 실제 제품·서비스·클라우드에 적용된 사례가 있는가?"),
    Question("adoption_2", "해당 기술이 논문·프로토타입을 넘어 제품 단계로 진행되었는가?"),
    Question("adoption_3", "해당 기술을 고객·운영 환경에서 지속 사용하는 근거가 있는가?"),
    Question("ecosystem_1", "serving framework·모델·런타임의 호환·지원이 존재하는가?"),
    Question("ecosystem_2", "오픈소스 구현·표준화 단체·벤더의 지원이 존재하는가?"),
    Question("ecosystem_3", "구현에 필요한 HW 부품 또는 SW 공급망·의존성이 확보되었는가?"),
    Question("ecosystem_4", "다양한 모델·workload·데이터센터 환경에서 재현 가능한가?"),
    Question("dependency", "특정 HW·framework·벤더 없이는 구현·확산이 어려운 종속성이 있는가?"),
)


def market_details(positive_grades: dict[str, str], dependency: str, answers=None) -> MarketDetails:
    """등급은 affirmative 근거만 집계. 자료가 전혀 없는 축은 null로 보류."""

    answers = answers or {}

    def grades(axis, count):
        return [positive_grades.get(f"{axis}_{i}", "하") for i in range(1, count + 1)]

    def axis_grade(axis, labels):
        values = grades(axis, 3)
        if all(v == "하" for v in values) and not any(
            answers.get(f"{axis}_{i}") == "no" for i in range(1, 4)
        ):
            return None
        if values.count("상") >= 2 and "하" not in values:
            return labels[2]
        if values.count("하") >= 2:
            return labels[0]
        return labels[1]

    eco = grades("ecosystem", 4)
    if dependency == "yes":
        ecosystem = "생태계 부재"
    elif all(v == "하" for v in eco) and not any(
        answers.get(f"ecosystem_{i}") == "no" for i in range(1, 5)
    ):
        ecosystem = None
    elif eco.count("하") >= 2:
        ecosystem = "생태계 부재"
    elif eco.count("상") >= 3 and dependency == "no":
        ecosystem = "확보"
    else:
        ecosystem = "형성 중"
    return MarketDetails(
        demand=axis_grade("demand", ("수요 불명확", "수요 존재", "수요 확실")),
        adoption=axis_grade("adoption", ("연구 단계", "시범 적용", "상용 채택")),
        ecosystem=ecosystem,
        dependency_risks=(
            ["특정 공급자·스택에 대한 종속성 근거 확인"]
            if dependency == "yes"
            else ["종속성 미확인"]
            if dependency == "unknown"
            else []
        ),
    )
