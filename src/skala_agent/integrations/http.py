import httpx

from skala_agent.integrations.contracts import ServiceConfigurationError


def post_json(url: str, payload: dict, *, timeout: float, headers=None, transport=None):
    try:
        with httpx.Client(timeout=timeout, transport=transport) as client:
            response = client.post(url, json=payload, headers=headers)
    except httpx.TimeoutException:
        raise TimeoutError("외부 서비스 응답 시간 초과") from None
    except httpx.TransportError:
        raise ConnectionError("외부 서비스 연결 실패") from None
    if response.status_code in (408, 429) or response.status_code >= 500:
        raise ConnectionError(f"외부 서비스 일시 오류 (HTTP {response.status_code})")
    if response.is_error:
        raise ServiceConfigurationError(
            f"서비스 설정·인증·모델 설치를 확인하세요 (HTTP {response.status_code})."
        )
    try:
        return response.json()
    except ValueError:
        raise ConnectionError("외부 서비스가 JSON을 반환하지 않았습니다.") from None
