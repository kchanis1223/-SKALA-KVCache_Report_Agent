"""윤소영 담당: 실행 모드별 provider 선택.

기본 demo 모드는 API 키와 네트워크 없이 동작합니다. real 모드는 김강휘 담당
adapter를 불러오며, adapter가 없으면 조용히 demo로 되돌아가지 않고 실패합니다.
"""

import threading
from collections import Counter
from typing import Literal, get_args

from skala_agent.providers import DemoProvider, Provider
from skala_agent.schemas import PERSPECTIVES

Mode = Literal["demo", "real"]
MODES: tuple[Mode, ...] = get_args(Mode)

ADAPTER_MODULE = "skala_agent.adapters"
ADAPTER_FACTORY = "build_provider"

# 관점 1건의 계산 시간 실측값(qwen3:4b). 상한 계산의 기준값입니다.
MEASURED_ASSESS_SECONDS = 300.0

# 외부 서비스 호출 1건의 상한. 0 이하이면 상한을 걸지 않습니다.
#
# ModelRouter가 모든 모델에 같은 lock을 공유하므로 추론이 직렬화됩니다. 그래서
# 병렬 fan-out으로 관점을 동시에 띄워도 실제 추론은 한 줄로 서고, 뒤에 선 관점은
# 자기 계산 시간뿐 아니라 앞선 관점들의 계산 시간까지 상한 안에서 감당합니다.
#
# 실측: 먼저 lock을 잡은 관점은 285.9초에 완주했지만 뒤에 선 세 관점은 계산을
# 시작하지도 못한 채 300초 상한에서 전부 끊겼습니다. 상한이 대기 시간을 덮지
# 못하면 "느린 관점"이 아니라 "줄 뒤에 선 관점"을 끊게 됩니다.
#
# 따라서 상한은 관점 1건이 아니라 직렬화된 관점 전체의 계산 시간을 덮어야 합니다.
# lock이 모델별로 분리되면 이 곱셈은 불필요해집니다(모델 계층 과제).
DEFAULT_TIMEOUT_SECONDS = MEASURED_ASSESS_SECONDS * len(PERSPECTIVES)

# 배치·캐시 도입 전 검증 실측: 검증 단계가 12~49건에
# 137~317초를 썼습니다(건당 4.6~12.9초). 상한을 건수와 무관하게 고정하면 근거가
# 쌓일수록 검증이 상한을 넘겼습니다. 현재 배치·캐시로 호출 수를 줄이지만
# 이 단계는 assess와 달리 관점 단위로 격리되지
# 않아 한 번의 초과가 실행 전체를 중단시킵니다. 보수적인 기존 상한을 유지합니다.
MEASURED_VALIDATE_SECONDS_PER_ITEM = 15.0

# 한 관점에서 상한을 **연속으로** 넘긴 횟수가 이만큼이면 남은 재평가를 건너뜁니다.
# 성공하면 0으로 되돌려, 느렸다가 회복한 관점을 영구 배제하지 않습니다.
MAX_ABANDONED_CALLS = 2


class ProviderUnavailableError(RuntimeError):
    """real 모드 provider를 불러오지 못한 경우."""


def load_provider(mode: Mode, *, timeout: float | None = None) -> Provider:
    if mode == "demo":
        provider: Provider = DemoProvider()
    elif mode == "real":
        provider = _load_real_provider()
    else:
        raise ValueError(f"알 수 없는 실행 모드: {mode}")
    if timeout is not None and timeout > 0:
        provider = TimeoutProvider(provider, timeout)
    return provider


def call_with_timeout(func, seconds: float, /, *args, **kwargs):
    """`seconds` 안에 끝나지 않으면 TimeoutError. 그 외 예외는 그대로 전달합니다."""
    box: dict[str, object] = {}

    def target() -> None:
        try:
            box["value"] = func(*args, **kwargs)
        except BaseException as exc:  # noqa: BLE001 - 호출자에게 그대로 전달
            box["error"] = exc

    # 파이썬은 실행 중인 스레드를 강제 종료할 수 없습니다. 상한을 넘긴 호출은
    # daemon 스레드로 남아 인터프리터 종료 시 함께 정리됩니다.
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    thread.join(seconds)
    if thread.is_alive():
        name = getattr(func, "__name__", repr(func))
        raise TimeoutError(f"provider.{name} 호출이 {seconds}초를 초과했습니다.")
    if "error" in box:
        raise box["error"]
    return box["value"]


class TimeoutProvider:
    """provider 호출마다 상한을 적용합니다.

    graph의 evaluate가 TimeoutError를 해당 관점의 실패로 처리하므로, 느린 외부
    서비스 하나가 전체 실행을 멈추지 않습니다.

    파이썬은 실행 중인 스레드를 강제 종료할 수 없으므로, 상한을 넘긴 호출은
    백그라운드에서 계속 자원을 씁니다. 응답하지 못하는 관점을 재시도 상한까지
    계속 호출하면 그런 호출이 쌓이면서 상한 시간만 반복해서 버립니다.

    그래서 **관점별로** 방치된 호출을 세고, `max_abandoned`건에 이르면 그
    관점의 재평가를 기다리지 않고 즉시 실패시킵니다. 관점별로 세기 때문에
    한 관점이 막혀도 나머지 관점은 정상적으로 진행합니다.

    호출 자체를 실제로 취소하려면 adapter가 HTTP 요청 수준의 타임아웃을 함께
    걸어야 합니다. 이 클래스는 그래프가 멈추지 않게 하는 상위 안전장치입니다.

    `research`와 `search_missing`은 한 번에 하나씩만 실행돼 쌓이지 않고 실패 시
    중단이 이미 방침이라, 상한만 적용하고 차단하지 않습니다.

    격리 범위가 단계마다 다릅니다. `assess`의 TimeoutError는 graph의 evaluate가
    해당 관점의 실패로 흡수하지만, `validate_evidence`의 TimeoutError는 validate
    노드를 그대로 통과해 실행 전체를 중단시킵니다. 검증은 배치·캐시를 사용하지만
    콜드 캐시 및 응답 복구를 위해 상한은 전달된 근거 건수에 비례해 유지합니다.
    """

    def __init__(
        self, inner: Provider, seconds: float, max_abandoned: int = MAX_ABANDONED_CALLS
    ) -> None:
        self.inner = inner
        self.seconds = seconds
        self.max_abandoned = max_abandoned
        self.abandoned: Counter[str] = Counter()

    def research(self, technologies):
        return call_with_timeout(self.inner.research, self.seconds, technologies)

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        if self.abandoned[perspective] >= self.max_abandoned:
            raise TimeoutError(
                f"[{perspective}] 상한을 연속 {self.abandoned[perspective]}회 넘겨 "
                "남은 재평가를 건너뜁니다."
            )
        try:
            result = call_with_timeout(
                self.inner.assess,
                self.seconds,
                perspective,
                technologies,
                domain,
                tech_analysis,
                evidence,
            )
        except TimeoutError:
            self.abandoned[perspective] += 1
            raise
        # 한 번이라도 상한 안에 끝났으면 방치 누적을 초기화합니다.
        self.abandoned[perspective] = 0
        return result

    def search_missing(self, missing):
        return call_with_timeout(self.inner.search_missing, self.seconds, missing)

    def validate_evidence(self, evidence):
        # 근거 건수에 비례한 상한. 기본 상한보다 짧아지지는 않습니다.
        seconds = max(self.seconds, MEASURED_VALIDATE_SECONDS_PER_ITEM * len(evidence))
        return call_with_timeout(self.inner.validate_evidence, seconds, evidence)


def _load_real_provider() -> Provider:
    from importlib import import_module

    try:
        module = import_module(ADAPTER_MODULE)
    except ModuleNotFoundError as exc:
        raise ProviderUnavailableError(
            f"real 모드 provider가 없습니다. {ADAPTER_MODULE}.{ADAPTER_FACTORY}()를 "
            "추가해 주세요 (담당: 김강휘). 지금은 --mode demo로 실행하세요."
        ) from exc

    factory = getattr(module, ADAPTER_FACTORY, None)
    if factory is None:
        raise ProviderUnavailableError(
            f"{ADAPTER_MODULE}에 {ADAPTER_FACTORY}()가 없습니다. "
            "Provider를 반환하는 factory를 추가해 주세요 (담당: 김강휘)."
        )
    return factory()
