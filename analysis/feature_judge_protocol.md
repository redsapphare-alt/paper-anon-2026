# Task Feature Quantification Protocol

**작성**: 2026-05-02
**목적**: 5개 도메인 (Medical, Legal, Code, Logic, Long Context)을 task의 본질적 특성에 따라 정량적으로 특성화. 이 점수가 곡선 형태(decreasing/increasing/inverted-U)와 최적 agent 수의 예측 변수가 됨.

---

## Feature 정의

본 논문에서 task는 다음 4가지 축으로 특성화한다.

### 1. Decomposability (D ∈ [0,1])

**정의**: 한 문제를 **여러 독립적 sub-task로 쪼갠 후 병렬 처리한 결과를 합쳐서** 정답에 도달할 수 있는 정도.

| 범위 | 설명 | 예 |
|---|---|---|
| 0.0–0.2 | 분해 거의 불가. 통합적 단일 추론 필요 | 의료 진단(증상 패턴 통합), 단일 법조항 해석 |
| 0.3–0.5 | 부분 분해 가능. 일부 단계만 독립적 | MCQ 풀이(선택지별 검토), 형식 논리(전제별 검토) |
| 0.6–0.8 | 명확한 분해 가능. 병렬 처리에 적합 | 함수 구현(서브 함수로 분해), passage QA(섹션별 분담) |
| 0.9–1.0 | 완전 분해 가능. 합성으로 정답 | 다중 모듈 시스템, 독립 도구 호출 |

**판정 기준 질문**:
- (a) 답을 도출할 때 N개의 sub-task로 쪼갤 수 있는가?
- (b) 그 sub-task들이 서로 거의 독립적인가?
- (c) sub-task의 답을 합쳐서 (concat, vote, sum 등) 최종 답을 만들 수 있는가?
- 셋 다 yes → 0.7+, 둘 → 0.4-0.7, 하나 → 0.2-0.4, 모두 no → 0-0.2

### 2. Verifiability (V ∈ [0,1])

**정의**: 정답을 **결정적·자동적으로 검증**할 수 있는 정도.

| 범위 | 설명 | 예 |
|---|---|---|
| 0.0–0.2 | 검증 거의 불가. 주관적 판단 | 글의 품질 평가, 창작 |
| 0.3–0.5 | 부분 검증. 일부 측면만 객관적 | 번역(BLEU 보조), 요약(ROUGE 보조) |
| 0.6–0.8 | 카테고리 매칭으로 검증 가능 | MCQ 단일 정답, 분류 라벨 |
| 0.9–1.0 | 완전 결정적 검증 (실행 검증 등) | 단위 테스트 통과, 수학 답 일치 |

**판정 기준**:
- (a) 정답이 단일/유한 카테고리에 속하는가?
- (b) 자동 매칭(string match, 실행 결과)이 가능한가?
- (c) Verification 자체에 모호함이 없는가?

### 3. Knowledge concentration (K ∈ [0,1])

**정의**: 정답이 **좁은 도메인 전문 지식**에 의존하는 정도. (높을수록 일반 추론으로는 풀기 어려움)

| 범위 | 설명 | 예 |
|---|---|---|
| 0.0–0.2 | 일반 상식/주어진 텍스트로 충분 | reading comprehension |
| 0.3–0.5 | 어느 정도 전문성 필요하나 폭넓음 | 일반 STEM 문제, 형식 논리 |
| 0.6–0.8 | 좁은 도메인 지식 필수 | 의료 진단, 약리학 |
| 0.9–1.0 | 매우 좁고 깊은 전문 지식 (precedent, 특정 statute 등) | 특정 법 분야 case law |

**판정 기준**:
- (a) 정답이 특정 분야 교육·자격증 없이도 도달 가능한가? (yes → low K)
- (b) 일반 LLM이 사전학습으로 그 지식을 충분히 갖췄을 가능성이 높은가? (yes → mid K)
- (c) 주어진 question/context에 정답을 위한 정보가 포함되어 있는가? (yes → low K)

### 4. Solution diversity (S ∈ [0,1])

**정의**: 한 문제에 대해 **여러 valid solution path**가 존재하는 정도. (multi-agent voting에 유리한 정도)

| 범위 | 설명 | 예 |
|---|---|---|
| 0.0–0.2 | 정답으로의 길이 사실상 하나 | 단일 답 MCQ, 정확한 사실 회상 |
| 0.3–0.5 | 약간의 다양성 (다른 접근법) | 다른 추론 경로지만 같은 답 |
| 0.6–0.8 | 여러 valid 솔루션 | 코드 구현(여러 알고리즘), 의료 진단 differential |
| 0.9–1.0 | 매우 다양한 정답 | 창작, open-ended generation |

---

## 예측 가설 (Phase 3 입력)

**가설 H1**: 곡선 형태 = f(D, V, K, S)
- **Decreasing** (1-agent 최고): 높은 K + 낮은 D
  - 직관: 좁은 전문 지식이 핵심 → 단일 강한 agent로 충분, 추가 agent는 noise만 추가
- **Increasing** (4-agent 최고): 높은 D + 높은 V + 높은 S
  - 직관: 분업 가능하고 검증 가능하며 다양한 valid 솔루션 → multi-agent voting 효과
- **Inverted-U** (2-agent 최고): 중간 D + 높은 V + 중간 K
  - 직관: 약간의 cross-validation은 유익하나 합의 비용이 곧 supersedes

**경계 조건**:
- 모든 K=0.9 + D=0.2 도메인은 1-agent 최고일 것 (예측)
- 모든 V<0.4 도메인은 multi-agent voting 효과 미미 (검증 불가하므로)

---

## 도메인-수준 점수 (예비, hand-curated)

다음은 도메인 정의와 일반적 task 특성에 기반한 1차 추정. **LLM judge로 refine 예정**.

| Domain | D | V | K | S | 예측 곡선 형태 | 데이터 일치? |
|---|---|---|---|---|---|---|
| Legal (legalbench: privacy_policy_qa 등 binary) | 0.20 | 0.80 | 0.90 | 0.20 | **Decreasing** | ✅ Legal GPT/Claude 모두 단조 감소 (검증됨) |
| Medical (medqa: clinical vignette → MCQ) | 0.30 | 0.90 | 0.80 | 0.30 | Decreasing/flat | ⚠️ GPT 평탄, Claude 증가 (단 1-agent 부족) |
| Logic (mmlu logic: argument structure → MCQ) | 0.40 | 0.90 | 0.40 | 0.40 | Inverted-U/flat | ⚠️ Claude 약한 inverted-U |
| Long Context (quality: passage QA → MCQ) | 0.60 | 0.90 | 0.20 | 0.40 | **Inverted-U** | ✅ Claude 강한 2-peak |
| Code (humaneval: function impl with tests) | 0.70 | 1.00 | 0.50 | 0.70 | **Increasing** | ❌ 데이터 오염, 재실험 필요 |

**Justification per domain**:

#### Legal (D=0.2, V=0.8, K=0.9, S=0.2)
- LegalBench의 `privacy_policy_qa` 등은 단일 텍스트 → 단일 라벨(Relevant/Yes/...). 분해 거의 불가.
- 라벨이 categorical이라 매칭은 가능하지만 라벨 정의 자체가 모호 (Relevant vs Irrelevant).
- 특정 법 분야 (privacy law, contract law) 지식이 강하게 필요.
- 정답으로의 경로가 사실상 하나 (텍스트 → 카테고리).

#### Medical (D=0.3, V=0.9, K=0.8, S=0.3)
- 임상 vignette는 통합적 추론 (증상 + 검사 + 환자 정보를 종합한 진단). 분해 어려움.
- 4-option MCQ라 매칭은 깔끔.
- 의학 지식 필수, 단 logic도 일부 (감별진단).
- Differential diagnosis가 약간의 다양성을 줌.

#### Logic (D=0.4, V=0.9, K=0.4, S=0.4)
- formal_logic, philosophy 등. 일부는 분해 가능 (각 전제 검토), 일부는 통합적.
- 4-option MCQ.
- 도메인 지식보다 일반 추론이 큰 비중. K가 낮음.
- 여러 추론 경로 가능.

#### Long Context (D=0.6, V=0.9, K=0.2, S=0.4)
- passage가 길어서 (avg ~5,000 토큰) 분담 검토 효과 있음.
- 4-option MCQ.
- 정답이 passage 안에 있음. 외부 지식 거의 불필요. K 매우 낮음.
- 여러 단서를 통합하는 길은 다양.

#### Code (D=0.7, V=1.0, K=0.5, S=0.7)
- 함수를 helper로 쪼갤 수 있고, 여러 agent가 edge case 검토를 분담 가능.
- Unit test로 결정적 검증 (V=1.0).
- 프로그래밍 지식은 광범위하지만 깊지는 않음.
- 한 문제에 여러 valid 알고리즘.

---

## LLM Judge 검증 절차

판정 일관성 평가를 위해 다음을 수행:

1. **Domain-level (LLM judge)**: 위 4축에 대한 도메인 정의를 GPT-4o, Claude Sonnet 4.6에게 각각 점수 매기게 함 (3회 반복). 평균 + 표준편차 산출. 위 hand-curated 점수와 비교.

2. **Question-level (LLM judge)**: 도메인당 200개 question random sample. 각 question에 대해 4축 점수를 LLM judge가 매김. 도메인 평균을 (1)과 비교. 분산이 어느 도메인에서 크게 나오는지 보고.

3. **Inter-judge agreement**: 두 LLM judge간 ICC 또는 Pearson correlation. 0.7+ 이면 신뢰.

산출:
- `analysis/domain_features.json` — 최종 점수 (judge 평균)
- `analysis/feature_judge/` — 각 judge call의 raw 응답
- `analysis/judge_agreement.csv` — judge간 일치도
