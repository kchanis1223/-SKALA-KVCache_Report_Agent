"""김동찬 담당: 임베딩 어댑터.

기본 모델은 BAAI/bge-m3입니다. 모델 파일을 받을 수 없는 환경이 있어
`Embedder` 프로토콜로 주입받고, 어댑터는 지연 로딩합니다.
"""

from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    model_name: str

    def encode(self, texts: list[str]) -> list[list[float]]: ...


class BgeM3Embedder:
    """FlagEmbedding 기반 어댑터. 모델은 첫 호출 때 지연 로딩합니다.

    본문 색인과 질의는 같은 모델·같은 정규화를 써야 하므로 한 인스턴스를 공유하세요.
    """

    def __init__(self, model_name: str = "BAAI/bge-m3", *, use_fp16: bool = False) -> None:
        self.model_name = model_name
        self._use_fp16 = use_fp16
        self._model = None

    def _load(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except ImportError as exc:  # pragma: no cover - 선택 의존성
                message = (
                    "bge-m3 임베딩에는 FlagEmbedding이 필요합니다. "
                    "`uv sync --extra embedding`으로 설치하세요."
                )
                raise RuntimeError(message) from exc
            self._model = BGEM3FlagModel(self.model_name, use_fp16=self._use_fp16)
        return self._model

    def encode(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        output = self._load().encode(texts, return_dense=True, return_sparse=False)
        return [vector.tolist() for vector in output["dense_vecs"]]
