"""로컬 Python 추론. torch/transformers는 최초 invoke 시에만 로드합니다."""

from threading import Lock

from skala_agent.integrations.contracts import ModelOutputError, ServiceConfigurationError


class TransformersQwen:
    def __init__(
        self,
        model_id="Qwen/Qwen3-4B",
        *,
        device="auto",
        revision="main",
        max_new_tokens=2048,
        max_context_tokens=8192,
    ):
        if device not in {"auto", "cpu", "mps", "cuda"}:
            raise ValueError("device는 auto/cpu/mps/cuda 중 하나여야 합니다.")
        if max_new_tokens < 1 or max_context_tokens <= max_new_tokens:
            raise ValueError("context는 양수인 max_new_tokens보다 커야 합니다.")
        self.model_id = model_id
        self.device = device
        self.revision = revision
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._lock = Lock()

    def _load(self):
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError:
            raise ServiceConfigurationError(
                "로컬 모델 의존성을 uv sync --extra local로 설치하세요."
            ) from None
        device = self.device
        if device == "auto":
            device = (
                "cuda"
                if torch.cuda.is_available()
                else ("mps" if torch.backends.mps.is_available() else "cpu")
            )
        if device == "mps" and not torch.backends.mps.is_available():
            raise ServiceConfigurationError("현재 환경은 MPS를 지원하지 않습니다.")
        if device == "cuda" and not torch.cuda.is_available():
            raise ServiceConfigurationError("현재 환경은 CUDA를 지원하지 않습니다.")
        dtype = torch.float32 if device == "cpu" else torch.float16
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            model = (
                AutoModelForCausalLM.from_pretrained(
                    self.model_id,
                    revision=self.revision,
                    torch_dtype=dtype,
                    use_safetensors=True,
                )
                .to(device)
                .eval()
            )
        except OSError:
            raise ServiceConfigurationError(
                "모델 ID·revision·캐시·다운로드 연결을 확인하세요."
            ) from None
        self._torch, self._tokenizer, self._model = torch, tokenizer, model

    def invoke(self, messages: list[dict[str, str]]) -> str:
        # 병렬 Agent가 모델을 중복 로딩하거나 동시에 generate하지 않도록 보호합니다.
        with self._lock:
            self._load()
            inputs = self._tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            prompt_length = inputs["input_ids"].shape[-1]
            if prompt_length + self.max_new_tokens > self.max_context_tokens:
                raise ModelOutputError("프롬프트가 설정된 모델 context 예산을 초과했습니다.")
            with self._torch.inference_mode():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.8,
                    top_k=20,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = outputs[0][prompt_length:]
            if len(generated) >= self.max_new_tokens:
                raise ModelOutputError(
                    "출력 토큰 한도에 도달했습니다. 결과를 판정에 사용하지 않습니다."
                )
            return self._tokenizer.decode(generated, skip_special_tokens=True).strip()
