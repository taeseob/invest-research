# data

내려받은 시세 데이터를 두는 폴더입니다. 이 README를 빼고는 저장소에 올라가지 않습니다.

데이터는 언제든 `src/`의 수집 코드로 다시 받을 수 있어야 합니다. 모두 `python -m src.data`로 만들고, 이미 받은 원본으로 표만 다시 만들 때는 `python -m src.data --offline`을 씁니다.

| 파일 | 내용 |
|---|---|
| `raw/shiller_ie_data.xls` | Shiller S&P 500 월간 자료 원본 (shillerdata.com) |
| `raw/yahoo_gspc.json` | Yahoo `^GSPC` 일별 시세 원본 (1927-12-30부터) |
| `raw/french_factors.zip` | Ken French 3요인 자료 원본 (RF: 1개월 국채 월 수익률, 1926-07부터) |
| `raw/manifest.json` | 위 원본의 출처, 내려받은 시각(UTC), 크기, SHA256 |
| `sp500_daily.csv` | S&P 500 일별 종가 (미국 동부 날짜) |
| `sp500_month_end.csv` | **주 시험용.** 월말 종가, 배당(Shiller), 가격·총수익률, RF |
| `sp500_monthly_avg.csv` | 보조 시험용. Shiller 월평균 가격, 배당, 총수익률, RF |

주의:

- Shiller 가격은 월말 종가가 아니라 그 달 일별 종가의 평균입니다. 추세 추종 전략의 성과를 부풀리므로 결론의 근거로 쓰지 않습니다.
- Yahoo `^GSPC`는 1957년 3월 이전이 S&P 90 지수이고, 토요일 거래일(1952년까지)이 없습니다.
- 끝나지 않은 달(미국 동부 시간 기준)과 Shiller 각주가 일부 날짜뿐이라고 밝힌 달은 뺍니다.
