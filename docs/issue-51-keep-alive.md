# Ollama 모델 유지 시간 측정 (#51)

2026-09-22, macOS arm64, 시스템 메모리 16 GiB, Ollama 0.34.2에서 측정했습니다.
설치된 qwen3:4b와 qwen3:8b(Q4_K_M)를 사용했으며 모델 다운로드는 없었습니다.
시작 시 `/api/ps`는 비어 있었고 측정 후 두 모델을 해제했습니다.

## 조건

- `/api/chat`, 같은 메시지 `Reply with only OK.`, `think=false`, `stream=false`.
- `temperature=0`, `num_ctx=8192`, `num_predict=32`, HTTP timeout 180초.
- `keep_alive=0`과 `5m` 각각에서 4B→4B, 4B→8B→4B를 순차 실행.
- 각 시퀀스 전에 두 모델을 해제. OS 파일 캐시는 제거하지 않음.
- 각 조건 1회 탐색 측정으로, 통계적 벤치마크나 전체 보고서 실행 성능을 뜻하지 않음.
- 로딩 시간은 API `load_duration`(ns), 전체 시간은 클라이언트 monotonic clock.
- 메모리는 요청 직후 `/api/ps`의 모델별 `size_vram` 합계. 이 환경에서는 `size`와 같았음.
  시스템 전체 메모리나 순간 최대 사용량이 아니며, 통합 메모리에서 별도 RAM에 더할 값도 아님.

## 결과

| keep_alive | 시퀀스 | 단계 | 모델 | 로딩(초) | 전체(초) | 요청 후 상주량(GiB) |
| --- | --- | --- | --- | ---: | ---: | ---: |
| 0 | repeat | 1 | qwen3:4b | 1.021 | 1.812 | 0.000 |
| 0 | repeat | 2 | qwen3:4b | 0.517 | 1.306 | 0.000 |
| 0 | switch | 1 | qwen3:4b | 0.529 | 1.317 | 0.000 |
| 0 | switch | 2 | qwen3:8b | 2.032 | 2.490 | 0.000 |
| 0 | switch | 3 | qwen3:4b | 0.770 | 1.543 | 0.000 |
| 5m | repeat | 1 | qwen3:4b | 0.523 | 1.324 | 3.080 |
| 5m | repeat | 2 | qwen3:4b | 0.001 | 0.709 | 3.080 |
| 5m | switch | 1 | qwen3:4b | 0.522 | 1.324 | 3.080 |
| 5m | switch | 2 | qwen3:8b | 1.294 | 1.533 | 5.560 |
| 5m | switch | 3 | qwen3:4b | 1.049 | 1.864 | 8.559 |

4B 연속 두 번째 호출에서는 유지 설정으로 로딩 비용이 줄었습니다.
모델 전환에서는 `5m`이어도 8B 호출 시 4B가 해제되어, 다음 4B 호출이 다시 로드되었습니다.
마지막에는 4B와 8B가 함께 상주했습니다. 따라서 5분 유지는 모델 전환의 무로딩이나
메모리 사용 상한을 보장하지 않습니다. 이 결과는 기본 유지 시간의 효과와 메모리
trade-off를 확인하는 제한된 측정이며, 실행 순서·캐시·서버 스케줄링의 영향을 받습니다.

## 재현

모델을 사용하는 다른 작업이 없는 로컬 서버에서 아래 스크립트를 실행합니다.
각 시퀀스 전후 모델 해제를 요청하므로 다른 요청과 동시에 실행하지 않습니다.
Ollama 버전, 시스템 메모리, 모델 양자화와 서버 설정도 함께 기록합니다.

```python
import json, time, urllib.request
BASE = 'http://localhost:11434'
def call(path, payload=None):
    request = urllib.request.Request(BASE + path, data=None if payload is None else json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=180) as response:
        return json.load(response)
rows = []
models = ['qwen3:4b', 'qwen3:8b']
try:
    print(json.dumps({'version': call('/api/version')}), flush=True)
    for keep in [0, '5m']:
        for workload, sequence in [('repeat', [models[0], models[0]]), ('switch', [models[0], models[1], models[0]])]:
            for model in models:
                call('/api/generate', {'model': model, 'keep_alive': 0})
            for index, model in enumerate(sequence):
                start = time.monotonic()
                response = call('/api/chat', {'model': model, 'messages': [{'role': 'user', 'content': 'Reply with only OK.'}], 'think': False, 'stream': False, 'keep_alive': keep, 'options': {'temperature': 0, 'num_ctx': 8192, 'num_predict': 32}})
                row = {'keep_alive': keep, 'workload': workload, 'step': index+1, 'model': model, 'wall_s': round(time.monotonic()-start, 3), 'load_s': round(response.get('load_duration', 0)/1e9, 3), 'done': response.get('done'), 'resident': [{'model': m['name'], 'size_gib': round(m['size']/2**30,3), 'vram_gib': round(m.get('size_vram',0)/2**30,3)} for m in call('/api/ps')['models']]}
                rows.append(row)
                print(json.dumps(row), flush=True)
finally:
    for model in models:
        call('/api/generate', {'model': model, 'keep_alive': 0})
```
