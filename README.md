# SKALA Agent — KV Cache 기술 비교

## Subject

장문 LLM 추론에서 KV cache는 context length 증가에 따라 GPU HBM을 빠르게 소모합니다.

본 프로젝트는 이 메모리 병목을 해결하는 두 가지 상반된 접근을 선정하고, **데이터센터·클라우드 서빙 환경에서 기술 성숙도·시장성·이해관계자·도메인 적용성을 근거 기반으로 비교 평가하는 Agentic RAG 시스템**을 구현합니다.

* **SW:** TurboQuant — KV cache 자체를 압축
* **HW:** ITME — 메모리 계층을 확장

목표는 특정 기술의 우열을 결정하는 것이 아니라, **각 접근의 적용 조건과 trade-off를 명확하게 드러내는 것**입니다.

---

## Overview

* **Objective:** KV cache 최적화 기술을 복수 관점에서 중립적으로 비교 평가
* **Domain:** 데이터센터 · 클라우드 LLM Serving
* **Method:** Multi-Agent + Agentic RAG
* **Framework:** LangGraph
* **Local LLM:** Qwen3-4B / Ollama
* **Web Search:** Tavily
* **Embedding:** BAAI/bge-m3

### 핵심 Workflow

```text
기술 원문 조사
      ↓
┌─────┬─────┬─────┬─────┐
TRL   시장성  이해관계자  도메인
└─────┴─────┴─────┴─────┘
      ↓
   관점 종합
      ↓
   근거 검증
   ↙       ↘
부족        충분
 ↓           ↓
재검색      보고서 생성
 ↓
해당 Agent 재평가
```

---

## Selected Technologies

| 구분 | 기술             | 접근                            |
| -- | -------------- | ----------------------------- |
| SW | **TurboQuant** | KV cache를 저비트 양자화해 저장량 감소     |
| HW | **ITME**       | CXL 기반 계층형 메모리 확장으로 KV 수용량 증가 |

두 기술은 동일한 KV cache 병목을 해결하지만 접근이 정반대입니다.

```text
TurboQuant
→ 데이터를 작게 만든다
→ 정확도 / 압축률 trade-off

ITME
→ 저장 공간을 넓힌다
→ 성능 / 인프라 비용 trade-off
```

비교 대상은 Agent가 아닌 **Human 기반으로 선정**했습니다.
접근의 대립성, 적용 전제, 검증 주기의 차이를 기준으로 선정하여 이후 다관점 평가에 분석의 초점을 맞췄습니다.

---

## Key Features

* 논문 PDF 기반 **RAG 기술 조사**
* TRL·시장성·이해관계자·도메인 **4개 관점 병렬 평가**
* 각 주장과 출처를 연결하는 **Evidence 관리**
* 근거 부족 시 해당 항목만 **선택적 재검색**
* 재검색 후 관련 평가 Agent를 다시 실행하는 **Retry Loop**
* 관점 간 일치점뿐 아니라 **상충점과 Trade-off 탐지**
* 근거가 부족한 경우 결론을 생성하지 않고 **판단 보류**
* 한국어 질의 → 영어 논문 검색을 위한 **Cross-lingual Retrieval**

### 차별점

> **Agent의 첫 판단을 그대로 사용하지 않고, 근거를 검증한 뒤 부족한 관점만 다시 조사·평가한다.**

자세한 내용은 아래 [Differentiators](#differentiators)에 정리했습니다.

---

## Agents

| Agent                 | 역할                             |
| --------------------- | ------------------------------ |
| **Research**          | 논문에서 기술 원리·성능·한계·실험 조건 추출      |
| **TRL**               | 공개 근거를 TRL 1~9 기준에 매핑          |
| **Market**            | 시장 수요·채택·생태계 평가                |
| **Stakeholder**       | GPU·메모리·클라우드·개발자·투자자 관점 분석     |
| **Domain**            | 데이터센터의 비용·SLA·운영 적합성 평가        |
| **Synthesis**         | 4개 관점의 공통점·상충점·trade-off 종합    |
| **Validation**        | 주장과 Evidence의 대응 및 근거 부족 여부 검사 |
| **Additional Search** | 부족한 근거만 추가 검색 후 재평가 대상으로 전달    |
| **Report**            | 검증된 State를 기반으로 최종 평가 보고서 생성   |

---

## Agent Model Strategy

아홉 Agent 모두 같은 모델(GPT-5.4 mini)을 사용하고, 역할별로 **추론 강도(reasoning effort)** 만 다르게 배치했습니다.

| Agent             | Model        | Effort | 이유                             |
| ----------------- | ------------ | ------ | ------------------------------ |
| Research          | GPT-5.4 mini | low    | RAG 결과 구조화·정보 추출               |
| Additional Search | GPT-5.4 mini | low    | 부족 근거 재검색 결과 전달 (LLM 질의 재작성 미적용) |
| TRL               | GPT-5.4 mini | medium | 명시된 TRL 기준에 근거 매핑              |
| Market            | GPT-5.4 mini | medium | 검색 결과를 평가축에 분류                 |
| Stakeholder       | GPT-5.4 mini | medium | 주체별 입장 구조화                     |
| Domain            | GPT-5.4 mini | medium | 정해진 비용·SLA·운영 기준 평가            |
| **Synthesis**     | GPT-5.4 mini | **high**   | 관점 간 상충·trade-off를 종합하는 고난도 추론 |
| Validation        | GPT-5.4 mini | low    | 주장과 Evidence 대응 관계 검증          |
| Report            | GPT-5.4 mini | medium | 검증된 결과의 장문 보고서 구조화             |

> 모델을 하나로 통일해 배포·비용 구조를 단순화했습니다. 원격 API 호출은 로컬 GPU 메모리를 공유하지 않으므로 관점 fan-out이 실제로 병렬 실행됩니다. 로컬 Ollama 경로는 오프라인 재현·비교 실행용으로 남겨두었고 `LLM_PROVIDER=ollama`로 전환합니다. 모델 ID는 `OPENAI_MODEL` 환경변수로 덮어쓸 수 있습니다.

---

## RAG

### Documents

* TurboQuant 원 논문
* TurboQuant 독립 분석 자료
* ITME 원 논문

각 chunk에는 다음 metadata를 저장합니다.

```text
paper_id
camp
role
section
```

### Embedding Selection

후보:

* BAAI/bge-m3
* Qwen3-Embedding-0.6B
* multilingual-e5-large
* gte-multilingual-base

50개 기술 질의로 직접 평가했습니다.

| Model                 |   Hit@1 |   Hit@3 |        MRR |
| --------------------- | ------: | ------: | ---------: |
| **BGE-M3**            | **46%** | **74%** | **0.6127** |
| Qwen3-Embedding-0.6B  |     38% |     68% |     0.5513 |
| multilingual-e5-large |     32% |     58% |     0.5057 |
| gte-multilingual-base |     32% |     56% |     0.4894 |

따라서 **BGE-M3**를 최종 Embedding 모델로 선정했습니다.

---

## Architecture

```mermaid
flowchart TD
    S([시작]) --> A[기술 및 도메인 입력]
    A --> B[기술 조사 Agent · RAG]

    B --> C[TRL 평가 Agent]
    B --> D[시장성 평가 Agent]
    B --> E[이해관계자 평가 Agent]
    B --> F[도메인 평가 Agent · RAG]

    C --> G[종합 Agent]
    D --> G
    E --> G
    F --> G

    G --> H{근거 검증 Agent}

    H -->|근거 충분| J[보고서 생성 Agent]
    J --> Z([종료])

    H -->|근거 부족 · retry < 2| I[추가 검색 Agent]
    I --> K{재평가 대상}

    K --> C
    K --> D
    K --> E
    K --> F
```

핵심은:

> **병렬 평가 → 종합 → 근거 검증 → 부족한 관점만 재검색·재평가 → 보고서 생성**

입니다.

---

## State Design

Agent는 직접 결과를 주고받는 대신 공통 State를 통해 구조화된 데이터를 전달합니다.

```text
tech_analysis
trl_analysis
market_analysis
stakeholder_analysis
domain_analysis
evidence
missing_evidence
synthesis
report
```

관점별 State key를 분리해 병렬 실행 시 충돌을 방지하고, `evidence`는 Reducer를 통해 누적합니다.

---

## Differentiators

주제는 모든 조가 같습니다. 저희가 다르게 한 것은 설계에 적어둔 판단을 그대로 믿지 않고
숫자로 확인한 점입니다. 확인하는 과정에서 처음 세웠던 가정이 두 번 틀렸습니다.

### 계약을 먼저 고정하고 시작했다

다섯 명이 동시에 개발해야 했습니다. 보통은 각자 만든 뒤 마지막에 합치는데, 그러면
서로 기대하는 데이터 모양이 달라서 합치는 단계에서 시간을 다 씁니다.

그래서 코드를 쓰기 전에 State key와 Agent의 입출력 형태를 먼저 확정했습니다.
어떤 키를 누가 만들고 누가 읽는지, 병렬로 실행될 때 관점별로 나눌지 합칠지,
재실행하면 기존 결과를 덮을지 남길지를 정해두고 출발했습니다.
그리고 합성 데이터를 fixture로 만들어, 다른 사람 코드가 없어도 각자 자기 모듈을
끝까지 테스트할 수 있게 했습니다. 확정한 계약은 [`docs/interfaces.md`](docs/interfaces.md)에 있습니다.

### 검색 성능을 두 번 측정했다

임베딩으로 bge-m3를 고른 것 자체는 특별하지 않습니다. 다만 고르고 끝내지 않았습니다.

설계 단계에서는 후보 4종을 기술 질의 50문항으로 비교해 모델을 선정했습니다(위 Embedding Selection).
구현을 마친 뒤에는 실제로 만들어진 검색 파이프라인이 의도대로 동작하는지를 다시 확인했습니다.
이번에는 정답을 논문의 어느 청크에 있는지까지 직접 찾아 라벨링한 한국어 질의 20문항을 사용했습니다.

| 조건 | Hit@1 | Hit@3 | MRR |
| --- | ---: | ---: | ---: |
| 전체 청크 | 0.750 | 0.900 | 0.838 |
| References 청크 제외 | 0.750 | 0.850 | 0.812 |

20문항 중 17문항이 1위로 정답을 찾았습니다. 질의는 한국어, 논문은 영어입니다.

| 한국어 질의 | 1위로 나온 청크 | 점수 |
| --- | --- | ---: |
| CXL 기반 계층적 메모리 확장 | `itme-p4-1` (3.1 Architecture and Hardware Design) | 0.648 |
| KV 캐시 양자화가 정확도에 미치는 영향 | `turboquant-p20-1` (정확도 벤치마크 표) | 0.571 |
| 실험 환경과 측정 조건 | `turboquant_analysis-p11-1` (Setup) | 0.522 |

두 측정은 목적이 다릅니다. 앞의 50문항은 어떤 모델을 쓸지 고르기 위한 비교이고,
뒤의 20문항은 고른 모델로 만든 파이프라인을 검증한 것입니다. 수치를 섞어서 읽으면 안 됩니다.
평가셋과 재현 명령은 [`docs/issue-4-retrieval-evaluation.md`](docs/issue-4-retrieval-evaluation.md)에 있습니다.

### 측정 결과가 설계를 두 번 뒤집었다

**참고문헌은 빼는 게 낫다고 봤습니다.** 참고문헌 목록은 키워드는 많지만 근거로 쓸 수 없으니
검색에서 제외하면 정확도가 오를 것이라 판단하고, 참고문헌 청크를 자동으로 찾아내는 기능을
만들었습니다. 빼고 다시 측정하니 Hit@3가 0.900에서 0.850으로 떨어졌습니다.

원인은 한 페이지에 두 가지가 섞여 있었기 때문입니다. `turboquant-p21-1`은 위쪽에 Recall 그래프가
있고 아래쪽에서 참고문헌이 시작되는 페이지입니다. 청킹이 페이지 단위라 그 페이지 전체가
참고문헌으로 분류되었고, 정답이던 그래프가 같이 사라졌습니다.
결국 가정을 버렸습니다. 참고문헌 표시는 메타데이터로만 남기고 검색에서 거르지 않습니다.

**dense 임베딩만으로 충분하다고 봤습니다.** "KV, KQV, QKQV 세 가지 양자화 스킴의 정의"를
묻는 질의가 상위 10개 안에도 들지 못했습니다. 세 약어가 한두 글자만 달라서 구분이 되지 않습니다.
bge-m3를 고른 이유 중 하나가 약어와 고유명사가 많은 문서에 대응하기 위해서였는데,
그 보완이 아직 반영되지 않은 지점이 실측으로 드러났습니다. 다음에 무엇을 붙여야 하는지가
추측이 아니라 숫자로 확인됐습니다.

### 같은 결과가 다시 나오도록 만들었다

측정은 다시 돌렸을 때 같은 값이 나와야 의미가 있습니다. 그래서 네 가지를 고정했습니다.

논문은 `configs/documents.json`에 sha256을 적어 같은 판본인지 확인할 수 있게 했습니다.
청크 ID는 `{paper_id}-p{page}-{순번}` 형식이라 다시 색인해도 번호가 유지됩니다.
색인을 만들 때 쓴 임베딩 모델, 청킹 설정, 청크 수, 차원은 manifest 파일로 남깁니다.
평가셋과 색인의 청킹 버전이 다르면 측정을 시작하지 않고 멈춥니다.
청킹 설정이 바뀌면 정답 청크 번호가 전부 달라지는데, 그걸 모른 채 측정하면 틀린 값이 나오기 때문입니다.

논문 PDF는 저작권 때문에 저장소에 올리지 않습니다. 대신 주소와 sha256을 올려두어
같은 파일을 받아 같은 결과를 재현할 수 있습니다.

---

## Report Highlights

보고서는 어느 기술이 더 낫다는 결론을 내지 않습니다. 같은 병목을 두고 관점에 따라
평가가 어떻게 갈리는지를 보여주는 것이 목적입니다.

| 관점 | TurboQuant (SW) | ITME (HW) |
| --- | --- | --- |
| 운영 | 소프트웨어만 바꾸면 되므로 바로 적용 가능 | CXL 인프라가 먼저 깔려 있어야 함 |
| 서비스 품질 | 압축 과정에서 정확도 손실 위험 | 원본을 그대로 두므로 손실 없음 |
| 도입 장벽 | 낮음 | 높음 |

같은 기술이라도 어느 관점에서 보느냐에 따라 장점이 단점이 됩니다. TurboQuant은 당장
적용할 수 있지만 SLA 위반 위험을 떠안고, ITME는 위험이 없는 대신 인프라 전환이 전제입니다.
그래서 결론은 "이것을 쓰세요"가 아니라 "어느 위험을 감당할 수 있는지에 따라 답이 갈린다"입니다.

이해관계자 관점에서 대립이 가장 분명하게 드러납니다. GPU·가속기 벤더 입장에서는 메모리를
아껴주는 기술이 자사 칩 수요를 줄일 수 있어 반가운 일이 아닙니다. 반대로 메모리 벤더에게는
메모리 확장이 곧 매출입니다. 두 주체의 이해가 정면으로 부딪칩니다.

### 한계를 함께 적었습니다

이 평가는 공개된 정보를 바탕으로 한 추정입니다. 특히 기술 성숙도는 논문이 나온 시점과
실제로 쓰이기 시작하는 시점 사이에 간격이 있어서, 공개 자료만으로는 확인되지 않는 구간이 큽니다.

그래서 모든 판정에 근거 URL과 원문 발췌를 함께 남기고, 검증된 출처가 2건이 안 되면
confidence를 low로 표시합니다. 지지하는 근거를 찾지 못하면 결론을 만들지 않고 판단 보류로
기록합니다. URL이 존재한다는 사실만으로 주장을 확정하지 않고, 그 원문이 실제로 주장을
지지하는지 따로 판정합니다.

---

## Lessons Learned

### 실행해보지 않은 코드는 동작한다고 말할 수 없다

임베딩 연결 코드를 작성하고 테스트 178개를 통과시켰습니다. 실제 모델로 돌리기 전에
라이브러리 소스를 확인했는데, `encode()`에 `max_length`를 넘기지 않으면 512 토큰에서
문장이 잘리는 기본값이 적용되고 있었습니다. 저희가 만든 청크는 최대 1,500 토큰입니다.

그대로 실행했다면 오류 없이 내용의 3분의 2가 버려진 채 결과가 나왔을 것입니다.
함수 이름도 인자도 반환 형식도 전부 맞았고, 틀린 것은 기본값 하나였습니다.
테스트가 모두 통과하더라도 가짜 데이터로만 돌린 것이라면 아무것도 보장하지 못합니다.

### 계산해본 값과 실제로 재본 값은 달랐다

청크가 61개 나올 것이라고 계산했지만 실제 토크나이저로는 69개였습니다.
참고문헌을 제외하면 성능이 오를 것이라 예상했지만 오히려 떨어졌습니다.
두 번 다 예상이 틀렸고, 재봤기 때문에 알 수 있었습니다.

### 계약을 정해두면 병렬 개발은 되지만 통합은 따로 챙겨야 한다

각자 맡은 모듈은 문제없이 굴러갔습니다. 그런데 모듈과 모듈을 연결하는 일은 누구의 담당도
아니어서 계속 뒤로 밀렸습니다. 검색 파이프라인을 완성하고 성능까지 측정한 다음에도
워크플로에는 연결되지 않은 상태로 며칠이 지났습니다. 이슈로 등록하고 담당을 정하고 나서야
해결됐습니다. 다음에는 연결 작업 자체를 하나의 할 일로 두고 담당자와 기한을 붙이려고 합니다.

---

## Directory Structure

```text
├── data/                  # PDF·청크·벡터 문서 풀
├── src/skala_agent/
│   ├── agents/            # Agent
│   ├── prompts/           # Prompt
│   ├── retrieval/         # PDF / Embedding / Retrieval
│   ├── workflow/          # LangGraph
│   ├── providers.py       # LLM / Search Provider
│   └── cli.py             # 실행 Entry Point
├── outputs/               # 최종 보고서
├── tests/                 # 테스트
└── README.md
```

---

## Usage

```bash
make setup
make run
make test
make lint
```

필요한 외부 API Key는 `.env`로 관리하며 Repository에는 포함하지 않습니다.

로컬 Agent는 Ollama를 통해 실행합니다.

---

## Contributors

* **김동찬** — PDF Parsing, Chunking, Embedding, VectorStore, Retrieval
* **김강휘** — TRL·Market·Stakeholder·Domain Agent, Prompt, Structured Output
* **윤소영** — State, LangGraph Workflow, Fan-out/Fan-in, Retry Routing
* **이준형** — Synthesis, Evidence Validation, Report/Citation, E2E Test, README

---
