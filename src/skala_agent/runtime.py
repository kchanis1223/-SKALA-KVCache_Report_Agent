"""윤소영 담당: 실행 모드별 provider 선택.

기본 demo 모드는 API 키와 네트워크 없이 동작합니다. real 모드는 김강휘 담당
adapter를 불러오며, adapter가 없으면 조용히 demo로 되돌아가지 않고 실패합니다.
"""

import threading
from collections import Counter
from typing import Literal, get_args

from skala_agent.providers import DemoProvider, Provider

Mode = Literal["demo", "real"]
MODES: tuple[Mode, ...] = get_args(Mode)

ADAPTER_MODULE = "skala_agent.adapters"
ADAPTER_FACTORY = "build_provider"

# 외부 서비스 호출 1건의 상한. 0 이하이면 상한을 걸지 않습니다.
DEFAULT_TIMEOUT_SECONDS = 120.0

# 한 관점에서 상한을 넘긴 호출이 이만큼 쌓이면 그 관점의 재평가를 건너뜁니다.
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
                f"[{perspective}] 상한을 넘긴 호출이 {self.abandoned[perspective]}건이라 "
                "재평가를 건너뜁니다."
            )
        try:
            return call_with_timeout(
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

    def search_missing(self, missing):
        return call_with_timeout(self.inner.search_missing, self.seconds, missing)


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
