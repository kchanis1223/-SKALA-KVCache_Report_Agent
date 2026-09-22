"""윤소영 담당: 실행 모드별 provider 선택.

기본 demo 모드는 API 키와 네트워크 없이 동작합니다. real 모드는 김강휘 담당
adapter를 불러오며, adapter가 없으면 조용히 demo로 되돌아가지 않고 실패합니다.
"""

import threading
from typing import Literal, get_args

from skala_agent.providers import DemoProvider, Provider

Mode = Literal["demo", "real"]
MODES: tuple[Mode, ...] = get_args(Mode)

ADAPTER_MODULE = "skala_agent.adapters"
ADAPTER_FACTORY = "build_provider"

# 외부 서비스 호출 1건의 상한. 0 이하이면 상한을 걸지 않습니다.
DEFAULT_TIMEOUT_SECONDS = 120.0


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
    """

    def __init__(self, inner: Provider, seconds: float) -> None:
        self.inner = inner
        self.seconds = seconds

    def research(self, technologies):
        return call_with_timeout(self.inner.research, self.seconds, technologies)

    def assess(self, perspective, technologies, domain, tech_analysis, evidence):
        return call_with_timeout(
            self.inner.assess,
            self.seconds,
            perspective,
            technologies,
            domain,
            tech_analysis,
            evidence,
        )

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
