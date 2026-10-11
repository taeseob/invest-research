# invest-research

개인 투자자를 위한 **규칙 기반 투자 전략 연구** 저장소입니다. 일봉·주봉·월봉 기반의 단순한 규칙을 만들고, 과거 데이터로 엄격하게 검증합니다.

AI 코딩 에이전트(Claude Code)와 함께 연구 환경을 만들고 실험하는 과정 자체도 이 저장소의 목적입니다.

> ⚠️ 이 저장소는 개인 연구 기록이며 투자 권유가 아닙니다. 여기 있는 전략과 결과는 과거 데이터에 대한 시험일 뿐이고, 미래 수익을 보장하지 않습니다.

---

## 목표

- **시장을 크게 이기는 것이 아닙니다.** "비슷한 수익률에 더 작은 하락, 그리고 실제로 지킬 수 있는 규칙"을 찾는 것이 목표입니다.
- 공개된 전략 대부분은 거래비용과 세금을 반영하면 단순 보유를 이기지 못합니다. 그래서 **버린 전략도 기록으로 남깁니다.**

## 원칙

1. **밤에 대응하지 않습니다.** 일봉·주봉·월봉 신호만 쓰고, 확인은 주 1회 이하로 합니다.
2. **단계마다 통과 기준이 있습니다.** 기준을 넘지 못하면 다음 단계로 가지 않습니다.
3. **미래 정보를 쓰지 않습니다.** 신호는 그 시점까지의 데이터로만 계산하고, 다음 거래일에 실행합니다. 이를 자동 테스트로 확인합니다.
4. **비용을 빼고 평가합니다.** 수수료, 슬리피지(체결 가격 차이), 세금, 배당을 반영합니다.
5. **모든 실험을 기록합니다.** 가설, 규칙, 결과, 판단을 [journal/](journal/)에 남깁니다.

## 단계

| 단계 | 내용 | 통과 기준 |
|---|---|---|
| 1 | 연구 환경 구축, 알려진 전략 재현 | 공개된 결과와 비슷하게 재현 |
| 2 | 자체 전략 연구 | 아래 "전략 통과 기준" 충족 |
| 3 | 모의 운용 6개월 | 신호를 규칙대로 100% 처리, 백테스트와의 차이를 설명 가능 |
| 4 | 실전 소액 운용 | 규칙 준수와 성과를 1년 이상 확인 |

**재현할 전략 (1단계)**: 규칙과 재현 계획은 [docs/phase1-strategies.md](docs/phase1-strategies.md)
- 10개월 이동평균 전략: Meb Faber, "A Quantitative Approach to Tactical Asset Allocation"
- 듀얼 모멘텀: Gary Antonacci, "Dual Momentum Investing"

**전략 통과 기준 (2단계)**
- 2008년, 2020년, 2022년 하락장을 포함한 20년 이상의 기간에서 시험
- 설계에 쓰지 않은 기간(표본 외 기간)에서도 성과 유지
- 비용·세금 반영 후, 단순 보유 대비 위험 대비 수익(샤프 지수)이 더 높거나, 수익률은 비슷한데 최대 낙폭이 뚜렷하게 작을 것
- 파라미터를 조금 바꿔도 결과가 무너지지 않을 것
- 규칙이 단순하고, 확인에 주 30분 이내

## 폴더 구조

```
invest-research/
├── src/          연구 코드 (데이터 수집, 백테스트, 전략, 지표)
├── tests/        테스트 (미래 정보 누수 검사 포함)
├── experiments/  실험별 실행 스크립트
├── journal/      연구 일지 (실험마다 하나)
├── reports/      결과 요약 (작은 표와 차트만)
├── docs/         설계 문서, 참고 자료 정리
└── data/         내려받은 시세 데이터 (저장소에 올리지 않음)
```

## 실행

Python 3.12 기준입니다.

```
python -m venv .venv
.venv\Scripts\activate          # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

python -m src.data              # 데이터 수집 (data/에 저장)
pytest                          # 테스트
```

## 참고 자료

- Meb Faber, "A Quantitative Approach to Tactical Asset Allocation" (SSRN)
- Gary Antonacci, "Dual Momentum Investing"
- Andreas Clenow, "Following the Trend"
- Ernest Chan, "Quantitative Trading"
