# data

내려받은 시세 데이터를 두는 폴더입니다. 이 README를 빼고는 저장소에 올라가지 않습니다.

데이터는 언제든 `src/`의 수집 코드로 다시 받을 수 있어야 합니다. 모두 `python -m src.data`로 만들고, 이미 받은 원본으로 표만 다시 만들 때는 `python -m src.data --offline`을 씁니다. KOSPI 표는 `python -m src.data --kospi`로 따로 만듭니다 (`--kospi --offline` 가능).

| 파일 | 내용 |
|---|---|
| `raw/shiller_ie_data.xls` | Shiller S&P 500 월간 자료 원본 (shillerdata.com) |
| `raw/yahoo_gspc.json` | Yahoo `^GSPC` 일별 시세 원본 (1927-12-30부터) |
| `raw/yahoo_spy.json` | Yahoo SPY 일별 시세와 배당 원본 (1993-01-29부터) |
| `raw/yahoo_sp500tr.json` | Yahoo `^SP500TR` 일별 시세 원본 (S&P 500 총수익 지수, 1988년부터) |
| `raw/french_factors.zip` | Ken French 3요인 자료 원본 (RF: 1개월 국채 월 수익률, 1926-07부터) |
| `raw/yahoo_ks11.json` | Yahoo `^KS11` KOSPI 일별 시세 원본 (1996-12-11부터) |
| `raw/yahoo_kodex200.json` | Yahoo `069500.KS` KODEX 200 일별 시세와 분배금 원본 |
| `raw/oecd_kor_ir3tib.csv` | OECD 한국 단기 금리 원본 (CD 91일물 월평균, 1996-01부터) |
| `raw/manifest.json` | 위 원본의 출처, 내려받은 시각(UTC), 크기, SHA256 |
| `sp500_daily.csv` | S&P 500 일별 종가 (미국 동부 날짜) |
| `sp500_month_end.csv` | **Faber 재현 주 시험용.** 월말 종가, 배당(Shiller), 가격·총수익률, RF |
| `spy_daily.csv` | SPY 일별 종가, 배당(세전, 배당락일), 가격·배당 포함 수익률, 일별 RF (월간 RF를 그 달 거래일 수로 기하 분할) |
| `sp500tr_daily.csv` | S&P 500 총수익 지수 일별 종가 (SPY 교차 확인용) |
| `sp500_monthly_avg.csv` | Faber 재현 보조 시험용. Shiller 월평균 가격, 배당, 총수익률, RF |
| `kospi_daily.csv` | KOSPI 일별 시가, 고가, 저가, 종가, 거래량, 일별 현금 수익률 (CD 금리 / 12를 그 달 거래일 수로 기하 분할). 배당 없는 가격 지수 |
| `kodex200_daily.csv` | KODEX 200 종가, 분배금, 수익률 (2009-06 ~ 2012-12만. KOSPI 합성 수익률 교차 확인용) |

주의:

- Shiller 가격은 월말 종가가 아니라 그 달 일별 종가의 평균입니다. 추세 추종 전략의 성과를 부풀리므로 결론의 근거로 쓰지 않습니다.
- Yahoo `^GSPC`는 1957년 3월 이전이 S&P 90 지수이고, 토요일 거래일(1952년까지)이 없습니다.
- 끝나지 않은 달(미국 동부 시간 기준)과 Shiller 각주가 일부 날짜뿐이라고 밝힌 달은 뺍니다.
- Yahoo `^KS11`에는 배당이 없고, 1998년 12월까지 있던 토요일 거래도 없습니다. 거래량이 0이고 고가와 저가가 같은 행은 뺍니다.
- Yahoo `069500.KS`는 2009-04 전 시세가 대부분 비어 있어 2008년을 포함한 시험에 쓸 수 없습니다.
- KRX 정보데이터시스템은 로그인이 필요해 자동으로 받지 않습니다. KOSPI 총수익 자료가 없어 배당은 실험에서 가정합니다.
