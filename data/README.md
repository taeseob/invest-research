# data

내려받은 시세 데이터를 두는 폴더입니다. 이 README를 빼고는 저장소에 올라가지 않습니다.

데이터는 언제든 `src/`의 수집 코드로 다시 받을 수 있어야 합니다. 모두 `python -m src.data`로 만들고, 이미 받은 원본으로 표만 다시 만들 때는 `python -m src.data --offline`을 씁니다. KOSPI 표는 `python -m src.data --kospi`로 따로 만듭니다 (`--kospi --offline` 가능). 인버스 ETF 표는 `python -m src.data --inverse`로 따로 만듭니다 (`--inverse --offline` 가능, 현금 수익률은 이미 받은 Ken French·OECD 원본 사용). GTAA 근사 재현용 5자산 표는 `python -m src.data --gtaa`로 따로 만듭니다 (`--gtaa --offline` 가능, S&P 500과 현금은 이미 받은 ^GSPC·Shiller·Ken French 원본 사용).

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
| `raw/yahoo_sh.json` | Yahoo SH(ProShares Short S&P500) 일별 시세와 분배금 원본 |
| `raw/yahoo_kodex_inverse.json` | Yahoo `114800.KS` KODEX 인버스 일별 시세 원본 |
| `sh_daily.csv` | SH 종가, 분배금, 수익률, 일별 RF (2006-06-21 ~ 2012-12-31만. 합성 인버스 교차 확인용) |
| `kodex_inverse_daily.csv` | KODEX 인버스 종가, 수익률, 일별 CD 금리 (2009-09-16 ~ 2012-12-28만. 합성 인버스 교차 확인용) |
| `raw/fed_h15_cmt.csv` | 연준 H.15 만기 고정 국채 금리 원본 (영업일, 1962년부터. FRED DGS10과 같은 자료) |
| `raw/nareit_monthly.xls` | FTSE Nareit 월간 지수 원본 (reit.com, 1971-12부터) |
| `raw/french_international_indices.zip`, `raw/french_international_indices_wout_div.zip` | Ken French 해외 지수 원본 (배당 포함·제외, 1975-01부터) |
| `raw/aqr_commodities_long_run.xlsx` | AQR "Commodities for the Long Run" 원본 (동일 비중 원자재 선물, 1877년부터) |
| `raw/yahoo_efa.json` | Yahoo EFA(iShares MSCI EAFE) 일별 시세 원본 (EAFE 대체 자료 교차 확인용) |
| `gtaa_monthly.csv` | **GTAA 근사 재현용.** 자산마다 총수익률(`_tr`)과 신호용 가격 수익률(`_pr`), S&P 500 신호용 총수익률(`us_stock_sig_tr`, 12개월 전 배당), 10년 국채 월말 금리, 리츠 세 지수, RF |
| `efa_monthly.csv` | EFA와 EAFE 대체 자료의 월 수익률 (2001-08 ~ 2012-12만. 교차 확인용) |

주의:

- Shiller 가격은 월말 종가가 아니라 그 달 일별 종가의 평균입니다. 추세 추종 전략의 성과를 부풀리므로 결론의 근거로 쓰지 않습니다.
- Yahoo `^GSPC`는 1957년 3월 이전이 S&P 90 지수이고, 토요일 거래일(1952년까지)이 없습니다.
- 끝나지 않은 달(미국 동부 시간 기준)과 Shiller 각주가 일부 날짜뿐이라고 밝힌 달은 뺍니다.
- Yahoo `^KS11`에는 배당이 없고, 1998년 12월까지 있던 토요일 거래도 없습니다. 거래량이 0이고 고가와 저가가 같은 행은 뺍니다.
- Yahoo `069500.KS`는 2009-04 전 시세가 대부분 비어 있어 2008년을 포함한 시험에 쓸 수 없습니다.
- KRX 정보데이터시스템은 로그인이 필요해 자동으로 받지 않습니다. KOSPI 총수익 자료가 없어 배당은 실험에서 가정합니다.
- Yahoo SH는 2016년(1:2)과 2024년(1:4)에 병합(역분할)이 있습니다. Yahoo는 종가와 분배금을 같은 비율로 조정하므로, 병합 전 구간의 배당 포함 수익률은 그대로입니다.
- Yahoo KODEX 인버스에는 2009-11-12와 2009-12-02가 없고, `^KS11`에는 2011-10-04가 없습니다.
- GTAA 표의 EAFE와 원자재는 대체 자료입니다. Ken French 해외 지수는 EAFE에 캐나다를 더한 비중으로 국가를 가중하고, AQR 원자재는 GSCI가 아니라 동일 비중입니다. 10년 국채는 금리로 합성한 값입니다.
- reit.com은 브라우저처럼 보이는 요청 머리글을 보내면 파일 대신 브라우저 확인 페이지(HTML)를 줍니다. 수집 코드는 기본 머리글로 받습니다.
- FRED(fred.stlouisfed.org)는 이 환경에서 연결이 끊겨, 같은 자료를 연준 H.15 원출처에서 받습니다.
