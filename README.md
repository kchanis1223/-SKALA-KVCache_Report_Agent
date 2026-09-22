# KV cache 최적화 기술 다관점 평가

LangGraph 기반 Multi-Agent + Agentic RAG로 KV cache 최적화 기술 2건을
기술성숙도·시장성·이해관계자·도메인 4가지 관점에서 비교 평가하고 보고서를 생성합니다.

| 항목 | 내용 |
|---|---|
| 대상 기술 (SW) | TurboQuant — 3비트 온라인 벡터 양자화 |
| 대상 기술 (HW) | ITME — CXL-Hybrid 계층적 메모리 확장 |
| 주 평가 도메인 | 데이터센터 · 클라우드 서빙 |
| 임베딩 | BAAI/bge-m3 (오픈소스) |
| 캠퍼스 / 반 | 판교 캠퍼스 6반 |

## 구조

```
.
├── data/sw, data/hw   논문 PDF (저작권상 미커밋)
├── src/
│   ├── agents/        에이전트별 모듈
│   └── rag/           인덱싱·검색
├── docs/              설계 문서
├── output/            생성된 보고서
└── notebooks/         실험·검증 노트북
```

## 문서

| 파일 | 내용 |
|---|---|
| `docs/design.md` | 설계 문서 — 기술 선정, RAG 적용 대상, 임베딩 선정, 4관점 평가 기준, State·그래프 설계 |
| `docs/design-v1.1.md` | 설계서 v1.1 (참고 자료) |
| `docs/TODO.md` | 설계·개발·발표 단계 체크리스트 |
| `docs/RAG-Design_*.docx` | 제출용 설계서 |

## 시작하기

```bash
cp .env.example .env    # 필요한 API KEY 입력
```

## Contributors

| 이름 | 담당 |
|---|---|
| 김동찬 | PDF 파싱·문서 전처리, Chunking/Metadata 설계, BGE-M3 임베딩, VectorDB·Retrieval 파이프라인, Retrieval 성능 평가(Hit@1·Hit@3·MRR) |
| 김강휘 | TRL·시장성·이해관계자·도메인 평가 Agent 구현, Agent별 Prompt 및 Structured Output 설계 |
| 윤소영 | LangGraph Workflow 구현, State Schema 설계, Fan-out/Fan-in 구조, Conditional Branch·Retry Loop, Agent 간 데이터 흐름 통합 |
| 이준형 | Synthesis Agent, Evidence Validation 로직, Report Generation Agent, Citation/Reference 정리, E2E 테스트 및 README 작성 |
| 권유나 | (역할 미정) |
