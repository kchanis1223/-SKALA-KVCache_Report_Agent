"""윤소영 담당: 실행 모드별 provider 선택.

기본 demo 모드는 API 키와 네트워크 없이 동작합니다. real 모드는 김강휘 담당
adapter를 불러오며, adapter가 없으면 조용히 demo로 되돌아가지 않고 실패합니다.
"""

from typing import Literal, get_args

from skala_agent.providers import DemoProvider, Provider

Mode = Literal["demo", "real"]
MODES: tuple[Mode, ...] = get_args(Mode)

ADAPTER_MODULE = "skala_agent.adapters"
ADAPTER_FACTORY = "build_provider"


class ProviderUnavailableError(RuntimeError):
    """real 모드 provider를 불러오지 못한 경우."""


def load_provider(mode: Mode) -> Provider:
    if mode == "demo":
        return DemoProvider()
    if mode == "real":
        return _load_real_provider()
    raise ValueError(f"알 수 없는 실행 모드: {mode}")


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
