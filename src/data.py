"""시세 데이터 수집과 정리.

- Yahoo: S&P 500 일별 종가 (^GSPC, 1927-12-30부터). 월말 종가는 여기서 만듭니다.
- Yahoo: SPY 일별 종가와 배당 (1993-01-29부터), S&P 500 총수익 지수 (^SP500TR, 1988년부터, 교차 확인용)
- Shiller: S&P 500 월간 배당과 월평균 가격 (shillerdata.com, 1871년부터)
- Ken French: 1개월 국채 월 수익률 RF (1926-07부터)
- Yahoo: KOSPI 일별 시가, 고가, 저가, 종가 (^KS11, 1996-12-11부터), KODEX 200 종가와 분배금 (069500.KS, 교차 확인용)
- OECD: 원화 CD 91일물 월평균 금리 (IR3TIB, 1996-01부터)
- Yahoo: 인버스 ETF SH(2006-06-21부터)와 KODEX 인버스(114800.KS, 2009-09-16부터). 합성 인버스 교차 확인용

실행: python -m src.data (원본으로 표만 다시 만들 때: python -m src.data --offline)
KOSPI 표는 따로 만듭니다: python -m src.data --kospi (원본으로 다시 만들 때: --kospi --offline)
인버스 표도 따로 만듭니다: python -m src.data --inverse (--inverse --offline). 현금 수익률은 이미 받은 원본을 씁니다.
원본은 data/raw/에 두고, 내려받은 시각과 SHA256을 data/raw/manifest.json에 남깁니다.
"""

import csv
import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

from src.bonds import par_bond_monthly_returns

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
MANIFEST_PATH = RAW_DIR / "manifest.json"

SHILLER_PAGE = "https://shillerdata.com/"
# 예전 주소(econ.yale.edu)는 2023년 10월 이후 갱신되지 않으므로 쓰지 않습니다.
YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
YAHOO_START = -1325583000  # 1927-12-30, Yahoo가 밝힌 ^GSPC 첫 거래일. 이보다 늦게 시작하는 종목은 첫 거래일부터 옵니다.
# Yahoo 종목 기호와 원본 파일 이름
YAHOO_SYMBOLS = {"^GSPC": "yahoo_gspc.json", "SPY": "yahoo_spy.json", "^SP500TR": "yahoo_sp500tr.json"}
FRENCH_FACTORS = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"
# 미국 표와 따로 받으므로 YAHOO_SYMBOLS에 넣지 않습니다.
YAHOO_KR_SYMBOLS = {"^KS11": "yahoo_ks11.json", "069500.KS": "yahoo_kodex200.json"}
OECD_KR_RATE = (
    "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/KOR.M.IR3TIB.PA....."
    "?startPeriod=1996-01&format=csvfilewithlabels"
)
# Yahoo의 KODEX 200은 2009-04 전이 대부분 비어 있습니다. 교차 확인은 설계용 기간 안에서만 합니다.
KODEX200_RANGE = ("2009-06-01", "2012-12-31")
# 인버스 ETF: 합성 인버스(-1배)의 교차 확인용. 미국·한국 표와 따로 받습니다.
# 표본 외 기간을 보지 않도록 표에는 설계용 기간 안의 값만 남깁니다(상장일 ~ 2012년 끝).
YAHOO_INVERSE_SYMBOLS = {"SH": "yahoo_sh.json", "114800.KS": "yahoo_kodex_inverse.json"}
INVERSE_RANGE = {"SH": ("2006-06-21", "2012-12-31"), "114800.KS": ("2009-09-16", "2012-12-28")}
# GTAA 근사 재현용 (journal/2026-11-21-faber-gtaa-reproduction.md). 미국·한국·인버스 표와 따로 받습니다.
# 연준 H.15 만기 고정 국채 금리(영업일, 1962년부터). FRED DGS10과 같은 자료이고, FRED는 이 환경에서 접속이 끊겨 원출처를 씁니다.
FED_H15_CMT = (
    "https://www.federalreserve.gov/datadownload/Output.aspx?rel=H15&series=bf17364827e38702b42a58cf8eaa3f78"
    "&lastobs=&from=&to=&filetype=csv&label=include&layout=seriescolumn"
)
H15_10Y = "RIFLGFCY10_N.B"
NAREIT_MONTHLY = "https://www.reit.com/sites/default/files/returns/MonthlyHistoricalReturns.xls"
FRENCH_INT_INDICES = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Indices.zip"
FRENCH_INT_INDICES_WOUT_DIV = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_International_Indices_Wout_Div.zip"
AQR_COMMODITIES = (
    "https://www.aqr.com/-/media/AQR/Documents/Insights/Data-Sets/Commodities-for-the-Long-Run-Index-Level-Data-Monthly.xlsx"
)
AQR_EW_EXCESS = "Excess return of equal-weight commodities portfolio"
YAHOO_GTAA_SYMBOLS = {"EFA": "yahoo_efa.json"}  # EAFE 대체 자료의 교차 확인용
# 교차 확인 표에는 설계용 기간 안의 값만 남깁니다 (표본 외 기간을 보지 않도록).
EFA_CHECK_END = pd.Period("2012-12", freq="M")
# 듀얼 모멘텀 근사 재현용 (journal/2026-11-21-antonacci-dual-momentum-reproduction.md). GTAA 때 받은 원본을 다시 쓰고,
# 교차 확인용 ETF만 새로 받습니다. 교차 확인 표에는 설계용 기간(2013-12까지)만 남깁니다.
H15_5Y = "RIFLGFCY05_N.B"
YAHOO_DM_SYMBOLS = {"ACWX": "yahoo_acwx.json", "AGG": "yahoo_agg.json"}
DM_CHECK_END = pd.Period("2013-12", freq="M")
# 가격의 호가 단위 (backtest.snap_to_tick에 씁니다).
# SPY는 2001-01-26까지 1/64달러 단위, 그 뒤 센트 단위라 둘의 공배수인 1/1600달러를 씁니다. KOSPI 지수는 소수 둘째 자리입니다.
PRICE_TICK = {"SPY": 1 / 1600, "^KS11": 0.01, "^GSPC": 0.01}

US_MARKET_TZ = ZoneInfo("America/New_York")
KR_MARKET_TZ = ZoneInfo("Asia/Seoul")
TIMEOUT = 60
# 지역 설정에 따라 strftime("%b")가 "10월"처럼 나올 수 있어 직접 둡니다.
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# ---------------------------------------------------------------- 내려받기


def record_manifest(path: Path, url: str, manifest_path: Path = MANIFEST_PATH) -> dict:
    """내려받은 파일의 출처, 시각(UTC), 크기, SHA256을 manifest에 남깁니다."""
    content = path.read_bytes()
    entry = {
        "url": url,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    manifest[path.name] = entry
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return entry


def download(url: str, path: Path, **kwargs) -> Path:
    response = requests.get(url, timeout=TIMEOUT, **kwargs)
    response.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    record_manifest(path, response.url)
    return path


def find_shiller_url(html: str) -> str:
    """shillerdata.com 첫 화면에서 ie_data.xls 링크를 찾습니다. 링크는 갱신마다 바뀝니다."""
    match = re.search(r'//img1\.wsimg\.com[^"\\\s]+/ie_data\.xls[^"\\\s]*', html)
    if match is None:
        raise ValueError("shillerdata.com에서 ie_data.xls 링크를 찾지 못했습니다. 사이트 구조를 확인하세요.")
    return "https:" + match.group(0)


def download_shiller(path: Path = RAW_DIR / "shiller_ie_data.xls") -> Path:
    page = requests.get(SHILLER_PAGE, timeout=TIMEOUT)
    page.raise_for_status()
    return download(find_shiller_url(page.text), path)


def download_yahoo(symbol: str) -> Path:
    """일별 시세와 배당·분할 기록을 받습니다.

    Yahoo는 공식 API가 없습니다. 형식이 바뀌거나 막히면 parse_yahoo에서 오류가 납니다.
    """
    params = {
        "period1": YAHOO_START,
        "period2": int(datetime.now(timezone.utc).timestamp()),
        "interval": "1d",
        "events": "div|split",
        "includeAdjustedClose": "true",
    }
    url = YAHOO_CHART.format(symbol=requests.utils.quote(symbol, safe=""))
    name = {**YAHOO_SYMBOLS, **YAHOO_KR_SYMBOLS, **YAHOO_INVERSE_SYMBOLS, **YAHOO_GTAA_SYMBOLS, **YAHOO_DM_SYMBOLS}[symbol]
    return download(url, RAW_DIR / name, params=params, headers={"User-Agent": "Mozilla/5.0"})


def download_french(path: Path = RAW_DIR / "french_factors.zip") -> Path:
    return download(FRENCH_FACTORS, path)


def download_oecd_kr_rate(path: Path = RAW_DIR / "oecd_kor_ir3tib.csv") -> Path:
    return download(OECD_KR_RATE, path, headers={"User-Agent": "Mozilla/5.0"})


# ---------------------------------------------------------------- 정리


def current_us_month(now: datetime | None = None) -> pd.Period:
    """미국 동부 시간 기준 이번 달. 이 달과 그 뒤는 아직 끝나지 않은 달로 봅니다."""
    now = now or datetime.now(US_MARKET_TZ)
    return pd.Period(now.astimezone(US_MARKET_TZ).date(), freq="M")


def check_monthly_continuity(index: pd.PeriodIndex, name: str) -> None:
    """빠진 달이 있으면 shift(1)이 두 달 수익률을 한 달로 계산하므로 막습니다."""
    if len(index) == 0:
        raise ValueError(f"{name}: 데이터가 없습니다.")
    expected = pd.period_range(index[0], index[-1], freq="M")
    if not index.equals(expected):
        missing = expected.difference(index)
        raise ValueError(f"{name}: 빠진 달이 있습니다: {list(missing.astype(str))[:5]}")


def parse_shiller_date(value: float) -> pd.Period:
    """Shiller 날짜(1871.01, 2026.1 = 2026년 10월)를 월 단위 Period로 바꿉니다."""
    year, month = f"{value:.2f}".split(".")
    return pd.Period(year=int(year), month=int(month), freq="M")


def find_partial_month_notes(raw: pd.DataFrame) -> set[str]:
    """각주에서 "Oct price is Oct 2nd close"처럼 일부 날짜뿐인 달의 약자를 찾습니다."""
    notes = set()
    for value in raw.to_numpy().ravel():
        if isinstance(value, str):
            match = re.search(r"\b([A-Z][a-z]{2})[a-z]*\.? price is\b", value)
            if match:
                notes.add(match.group(1))
    return notes


def parse_shiller(raw: pd.DataFrame) -> pd.DataFrame:
    """ie_data.xls의 Data 시트(header=None으로 읽은 것)를 월간 표로 정리합니다.

    반환 열:
    - price: S&P 500 지수. 그 달 일별 종가의 평균입니다 (월말 종가가 아님).
    - dividend: 연간 배당 (12개월 합계, 연율)
    - cpi: 소비자물가지수
    - gs10: 10년 국채 금리 (연 %). 1953년 이전은 연간 값을 직선으로 이은 값입니다.

    각주가 "일부 날짜뿐"이라고 밝힌 마지막 달은 뺍니다.
    """
    header_rows = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "Date"]
    if len(header_rows) != 1:
        raise ValueError("Data 시트에서 'Date' 머리글 행을 찾지 못했습니다.")
    header = raw.loc[header_rows[0]].astype(str).str.strip()
    expected = {1: "P", 2: "D", 4: "CPI", 6: "Rate GS10"}
    for col, name in expected.items():
        if header.iloc[col] != name:
            raise ValueError(f"{col}번째 열 머리글이 '{name}'이 아니라 '{header.iloc[col]}'입니다. 파일 형식이 바뀌었는지 확인하세요.")

    body = raw.loc[header_rows[0] + 1 :, [0, 1, 2, 4, 6]]
    body.columns = ["date", "price", "dividend", "cpi", "gs10"]
    # 맨 아래 각주 행처럼 날짜가 숫자가 아닌 행은 버립니다.
    body = body[pd.to_numeric(body["date"], errors="coerce").notna()]
    body = body.apply(pd.to_numeric, errors="coerce")
    body = body[body["price"].notna()]

    body.index = pd.PeriodIndex([parse_shiller_date(v) for v in body["date"]], freq="M", name="month")
    if not body.index.is_unique or not body.index.is_monotonic_increasing:
        raise ValueError("Shiller 날짜가 중복되거나 순서가 맞지 않습니다.")
    body = body.drop(columns="date")

    if MONTH_ABBR[body.index[-1].month - 1] in find_partial_month_notes(raw.loc[header_rows[0] + 1 :]):
        body = body.iloc[:-1]
    check_monthly_continuity(body.index, "Shiller")
    return body


def yahoo_dates(timestamps, tz: ZoneInfo = US_MARKET_TZ) -> pd.DatetimeIndex:
    """Yahoo 타임스탬프를 그 시장의 날짜로 바꿉니다."""
    local = pd.to_datetime(timestamps, unit="s", utc=True).tz_convert(tz)
    # 시각이 장중이 아니거나 주말이면 날짜가 하루 밀렸을 수 있으므로 멈춥니다.
    if ((local.hour < 9) | (local.hour > 16) | (local.dayofweek >= 5)).any():
        raise ValueError("Yahoo 시각이 그 시장의 장중 평일이 아닌 행이 있습니다. 날짜 변환을 확인하세요.")
    return local.tz_localize(None).normalize()


def parse_yahoo(data: dict, tz: ZoneInfo = US_MARKET_TZ) -> pd.Series:
    """Yahoo 차트 응답을 그 시장의 날짜 기준 일별 종가 Series로 바꿉니다.

    Yahoo의 close는 분할만 반영하고 배당은 반영하지 않은 가격입니다.
    """
    try:
        result = data["chart"]["result"][0]
        timestamps = result["timestamp"]
        closes = result["indicators"]["quote"][0]["close"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Yahoo 응답 형식이 예상과 다릅니다.") from error

    close = pd.Series(closes, index=yahoo_dates(timestamps, tz), name="close", dtype="float64").dropna()
    close = close[~close.index.duplicated(keep="last")].sort_index()
    close.index.name = "date"
    if (close <= 0).any():
        raise ValueError("Yahoo 종가에 0 이하 값이 있습니다.")
    return close


def parse_yahoo_ohlc(data: dict, tz: ZoneInfo = KR_MARKET_TZ) -> pd.DataFrame:
    """Yahoo 차트 응답을 일별 시가, 고가, 저가, 종가, 거래량 표로 바꿉니다.

    종가가 없는 행과, 거래량이 0이면서 고가와 저가가 같은 행(거래가 없는 잘못된 행)은 뺍니다.
    """
    try:
        result = data["chart"]["result"][0]
        quote = result["indicators"]["quote"][0]
        columns = {c: quote[c] for c in ["open", "high", "low", "close", "volume"]}
        ohlc = pd.DataFrame(columns, index=yahoo_dates(result["timestamp"], tz), dtype="float64")
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Yahoo 응답 형식이 예상과 다릅니다.") from error

    ohlc = ohlc.dropna(subset=["close"])
    ohlc = ohlc[~ohlc.index.duplicated(keep="last")].sort_index()
    ohlc.index.name = "date"
    ohlc = ohlc[~((ohlc["volume"].fillna(0) == 0) & (ohlc["high"] == ohlc["low"]))]

    prices = ohlc[["open", "high", "low", "close"]]
    if prices.isna().any(axis=None) or (prices <= 0).any(axis=None):
        raise ValueError("Yahoo 시세에 비었거나 0 이하인 값이 있습니다.")
    tolerance = 1e-6  # 소수점 저장 오차
    upper, lower = prices[["open", "close"]].max(axis=1), prices[["open", "close"]].min(axis=1)
    if ((ohlc["high"] < upper - tolerance) | (ohlc["low"] > lower + tolerance)).any():
        raise ValueError("고가가 시가·종가보다 낮거나 저가가 시가·종가보다 높은 행이 있습니다.")
    return ohlc


def parse_yahoo_dividends(
    data: dict, tz: ZoneInfo = US_MARKET_TZ, between: tuple[str, str] | None = None, splits_after: str | None = None
) -> pd.Series:
    """Yahoo 배당 기록(배당락일 기준, 주당 금액)을 꺼냅니다.

    between: 이 날짜 범위의 기록만 씁니다. 시세가 비어 있는 시기의 기록은 날짜가 어긋날 수 있어 범위로 거릅니다.
    분할 기록이 있으면 멈춥니다. 분할이 있는 종목은 배당과 가격의 기준을 맞추는 처리가 따로 필요합니다.
    splits_after: 이 날짜 뒤의 분할은 허용합니다. Yahoo는 종가와 배당을 같은 비율로 조정하므로, 분할 전 구간의
        (종가 + 배당) / 전날 종가는 그대로입니다. SH(2016년 1:2, 2024년 1:4 병합)에서 배당률로 확인했습니다.
    """
    events = data["chart"]["result"][0].get("events", {})
    splits = list(events.get("splits", {}).values())
    if splits_after is not None:
        limit = pd.Timestamp(splits_after, tz=tz).timestamp() + 86400
        splits = [s for s in splits if s["date"] < limit]
    if splits:
        raise ValueError("분할 기록이 있습니다. 분할 처리를 넣기 전에는 이 종목을 쓰지 않습니다.")
    records = sorted(events.get("dividends", {}).values(), key=lambda r: r["date"])
    if between is not None:
        first, last = (pd.Timestamp(day, tz=tz).timestamp() for day in between)
        records = [r for r in records if first <= r["date"] < last + 86400]
    dividends = pd.Series(
        [r["amount"] for r in records], index=yahoo_dates([r["date"] for r in records], tz), name="dividend", dtype="float64"
    )
    dividends.index.name = "date"
    if not dividends.index.is_unique:
        raise ValueError("같은 날 배당 기록이 둘 이상입니다.")
    return dividends


def daily_rf(index: pd.DatetimeIndex, rf_monthly: pd.Series, current_month: pd.Period) -> pd.Series:
    """월간 RF를 그 달 거래일 수로 기하 분할한 일별 수익률.

    그 달 거래일 수는 월말에야 알 수 있지만, 매매 결정에는 쓰이지 않습니다.
    끝나지 않은 달은 거래일 수를 모르므로 NaN입니다.
    """
    months = index.to_period("M")
    days_in_month = pd.Series(1, index=index).groupby(months).transform("size").to_numpy()
    rf = (1 + rf_monthly.reindex(months).to_numpy()) ** (1 / days_in_month) - 1
    rf[months >= current_month] = float("nan")
    return pd.Series(rf, index=index, name="rf")


def daily_dividend(index: pd.DatetimeIndex, annual_dividend: pd.Series) -> pd.Series:
    """월간 연간 배당 D(Shiller, 지수 포인트, 인덱스는 월)를 그 달 거래일에 똑같이 나눈 일별 배당.

    t월의 배당 D[t]/12를 그 달 거래일 수로 나눠 매일 받습니다. 거래일 수는 월말에야 알지만 매매 결정에는 쓰이지 않습니다.
    D가 없는 달은 NaN입니다.
    """
    months = index.to_period("M")
    days_in_month = pd.Series(1, index=index).groupby(months).transform("size").to_numpy()
    dividend = annual_dividend.reindex(months).to_numpy() / 12 / days_in_month
    return pd.Series(dividend, index=index, name="dividend")


def build_etf_daily(close: pd.Series, dividends: pd.Series, rf_monthly: pd.Series, current_month: pd.Period) -> pd.DataFrame:
    """ETF 일별 표: 종가, 배당(세전, 배당락일), 가격 수익률, 배당 포함 수익률, 일별 RF."""
    if not dividends.index.isin(close.index).all():
        missing = dividends.index[~dividends.index.isin(close.index)]
        raise ValueError(f"거래일이 아닌 배당락일이 있습니다: {list(missing.strftime('%Y-%m-%d'))[:5]}")
    df = pd.DataFrame({"close": close})
    df["dividend"] = dividends.reindex(df.index).fillna(0.0)
    df["price_return"] = df["close"] / df["close"].shift(1) - 1
    df["total_return"] = (df["close"] + df["dividend"]) / df["close"].shift(1) - 1
    df["rf"] = daily_rf(df.index, rf_monthly, current_month)
    return df


def build_kospi_daily(ohlc: pd.DataFrame, cd_rate: pd.Series, current_month: pd.Period) -> pd.DataFrame:
    """KOSPI 일별 표: 시가, 고가, 저가, 종가, 거래량, 일별 현금 수익률.

    CD 금리(연율)를 12로 나눠 월 수익률로 보고, 그 달 거래일 수로 기하 분할합니다.
    배당은 없습니다. ^KS11은 가격 지수입니다.
    """
    df = ohlc.copy()
    df["rf"] = daily_rf(df.index, cd_rate / 12, current_month)
    return df


def parse_oecd_rate(text: str) -> pd.Series:
    """OECD 단기 금리 CSV에서 한국 월평균 금리를 소수(0.1266 = 연 12.66%)로 꺼냅니다."""
    raw = pd.read_csv(io.StringIO(text))
    required = {"REF_AREA", "FREQ", "MEASURE", "UNIT_MEASURE", "TIME_PERIOD", "OBS_VALUE"}
    if not required.issubset(raw.columns):
        raise ValueError(f"OECD 파일에 필요한 열이 없습니다: {sorted(required - set(raw.columns))}")
    expected = {"REF_AREA": "KOR", "FREQ": "M", "MEASURE": "IR3TIB", "UNIT_MEASURE": "PA"}
    for column, value in expected.items():
        if set(raw[column]) != {value}:
            raise ValueError(f"OECD {column} 값이 '{value}'만 있어야 합니다: {sorted(set(raw[column]))}")

    months = pd.PeriodIndex(raw["TIME_PERIOD"], freq="M", name="month")
    rate = pd.Series(raw["OBS_VALUE"].to_numpy(dtype="float64") / 100, index=months, name="cd_rate").sort_index()
    if rate.isna().any():
        raise ValueError("OECD 금리에 빈 값이 있습니다.")
    check_monthly_continuity(rate.index, "OECD 금리")
    return rate


def parse_french_rf(text: str) -> pd.Series:
    """Ken French 3요인 CSV에서 월간 RF를 소수(0.0022 = 0.22%)로 꺼냅니다.

    월간 행은 날짜가 YYYYMM 6자리이고, 아래쪽 연간 행은 YYYY 4자리라 섞이지 않습니다.
    """
    lines = text.splitlines()
    header_idx = next((i for i, line in enumerate(lines) if line.replace(" ", "") == ",Mkt-RF,SMB,HML,RF"), None)
    if header_idx is None:
        raise ValueError("Ken French 파일에서 머리글(,Mkt-RF,SMB,HML,RF)을 찾지 못했습니다.")

    months, values = [], []
    for line in lines[header_idx + 1 :]:
        match = re.match(r"^\s*(\d{6})\s*,(.*)$", line)
        if match is None:
            break  # 월간 표가 끝났습니다.
        fields = [f.strip() for f in match.group(2).split(",")]
        months.append(pd.Period(f"{match.group(1)[:4]}-{match.group(1)[4:]}", freq="M"))
        values.append(float(fields[3]) / 100)

    rf = pd.Series(values, index=pd.PeriodIndex(months, freq="M", name="month"), name="rf")
    check_monthly_continuity(rf.index, "Ken French RF")
    return rf


def read_french_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        names = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"Ken French zip 안의 CSV가 하나가 아닙니다: {names}")
        return archive.read(names[0]).decode("latin-1")


def total_return(price: pd.Series, dividend: pd.Series) -> pd.Series:
    """월간 총수익률. Shiller와 같은 방식입니다.

    total_return[t] = (price[t] + dividend[t] / 12) / price[t-1] - 1

    배당이 비어 있는 달(가장 최근 몇 달)은 NaN으로 둡니다. 가격만으로 채우지 않습니다.
    """
    return (price + dividend / 12) / price.shift(1) - 1


def month_end_close(daily: pd.Series) -> pd.DataFrame:
    """각 달 마지막 거래일의 종가와 그 날짜."""
    grouped = daily.groupby(daily.index.to_period("M"))
    out = pd.DataFrame({"last_trading_day": grouped.apply(lambda s: s.index[-1]), "close": grouped.last()})
    out.index.name = "month"
    return out


def build_month_end(daily: pd.Series, shiller: pd.DataFrame, rf: pd.Series, current_month: pd.Period) -> pd.DataFrame:
    """주 시험용 월간 표: 월말 종가, 배당(Shiller), 총수익률, RF.

    current_month와 그 뒤는 아직 끝나지 않은 달이라 뺍니다.
    """
    df = month_end_close(daily[daily.index.to_period("M") < current_month])
    check_monthly_continuity(df.index, "Yahoo 월말 종가")
    df["dividend"] = shiller["dividend"].reindex(df.index)
    df["price_return"] = df["close"] / df["close"].shift(1) - 1
    df["total_return"] = total_return(df["close"], df["dividend"])
    df["rf"] = rf.reindex(df.index)
    return df


def build_monthly_avg(shiller: pd.DataFrame, rf: pd.Series, current_month: pd.Period) -> pd.DataFrame:
    """보조 시험용 월간 표: Shiller 월평균 가격 기준. 월평균 문제 때문에 결론의 근거로 쓰지 않습니다."""
    df = shiller[shiller.index < current_month].copy()
    df["price_return"] = df["price"] / df["price"].shift(1) - 1
    df["total_return"] = total_return(df["price"], df["dividend"])
    df["rf"] = rf.reindex(df.index)  # 1926-07 이전은 NaN
    return df


# ---------------------------------------------------------------- GTAA 근사 재현용


def parse_h15_yield(text: str, identifier: str = H15_10Y) -> pd.Series:
    """연준 H.15 CSV에서 한 만기의 영업일 금리를 소수(0.0406 = 연 4.06%)로 꺼냅니다. 휴일 표기("ND")와 빈 값은 뺍니다."""
    rows = list(csv.reader(io.StringIO(text)))
    ids = next((r for r in rows if r and r[0].startswith("Unique Identifier")), None)
    header = next((i for i, r in enumerate(rows) if r and r[0] == "Time Period"), None)
    if ids is None or header is None:
        raise ValueError("H.15 파일에서 'Unique Identifier'나 'Time Period' 행을 찾지 못했습니다.")
    if f"H15/H15/{identifier}" not in ids:
        raise ValueError(f"H.15 파일에 {identifier} 열이 없습니다.")
    column = rows[header].index(identifier)
    dates, values = [], []
    for r in rows[header + 1 :]:
        value = r[column].strip() if len(r) > column else ""
        if value in ("", "ND"):
            continue
        dates.append(pd.Timestamp(r[0]))
        values.append(float(value) / 100)
    series = pd.Series(values, index=pd.DatetimeIndex(dates, name="date"), name=identifier)
    if not series.index.is_monotonic_increasing or not series.index.is_unique:
        raise ValueError("H.15 날짜가 중복되거나 순서가 맞지 않습니다.")
    return series


def month_end_value(daily: pd.Series) -> pd.Series:
    """각 달 마지막 값 (인덱스는 월)."""
    out = daily.groupby(daily.index.to_period("M")).last()
    out.index.name = "month"
    return out


def parse_nareit(raw: pd.DataFrame, name: str = "All REITs") -> pd.DataFrame:
    """reit.com 월간 파일(Index Data 시트, header=None)에서 한 지수의 총수익 지수와 가격 지수를 꺼냅니다.

    머리글은 세 줄입니다: 지수 이름 / Total·Price·Income / Return·Index. 위치는 글자로 찾고 확인합니다.
    """
    first = raw.iloc[:, 0].astype(str).str.strip()
    header_rows = raw.index[first == "Date"]
    if len(header_rows) != 1:
        raise ValueError("NAREIT 파일에서 'Date' 머리글 행을 찾지 못했습니다.")
    h = raw.index.get_loc(header_rows[0])
    names = raw.iloc[h - 2].astype(str).str.strip()
    matches = [i for i, v in enumerate(names) if v == name]
    if len(matches) != 1:
        raise ValueError(f"NAREIT 파일에서 '{name}' 지수를 찾지 못했습니다.")
    g = matches[0]
    expected = {(h - 1, g): "Total", (h, g + 1): "Index", (h - 1, g + 2): "Price", (h, g + 3): "Index"}
    for (row, col), text in expected.items():
        if str(raw.iat[row, col]).strip() != text:
            raise ValueError(f"NAREIT '{name}' 머리글이 예상과 다릅니다: ({row}, {col})는 '{text}'여야 합니다.")
    body = raw.iloc[h + 1 :]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    body = body[dates.notna()]
    out = pd.DataFrame(
        {"total_index": body.iloc[:, g + 1].astype(float).to_numpy(), "price_index": body.iloc[:, g + 3].astype(float).to_numpy()},
        index=pd.PeriodIndex(dates[dates.notna()].dt.to_period("M"), name="month"),
    )
    check_monthly_continuity(out.index, f"NAREIT {name}")
    if out.isna().any(axis=None) or (out <= 0).any(axis=None):
        raise ValueError(f"NAREIT {name}에 비었거나 0 이하인 값이 있습니다.")
    return out


def parse_french_international(text: str) -> pd.Series:
    """Ken French "International Index" 파일(.Dat)의 첫 구획(달러 기준 가치 가중 월 수익률)에서 Mkt 열을 소수로 꺼냅니다."""
    lines = text.splitlines()
    start = next((i for i, l in enumerate(lines) if "Value-Weight Dollar Returns" in l), None)
    if start is None or lines[start + 2].split()[:1] != ["Mkt"]:
        raise ValueError("Ken French 해외 지수 파일의 첫 구획(Value-Weight Dollar Returns, Mkt)을 찾지 못했습니다.")
    months, values = [], []
    for line in lines[start + 3 :]:
        parts = line.split()
        if not parts or not re.fullmatch(r"\d{6}", parts[0]):
            break  # 첫 구획이 끝났습니다.
        months.append(pd.Period(f"{parts[0][:4]}-{parts[0][4:]}", freq="M"))
        values.append(float(parts[1]) / 100)
    series = pd.Series(values, index=pd.PeriodIndex(months, freq="M", name="month"), name="mkt")
    check_monthly_continuity(series.index, "Ken French 해외 지수")
    if (series <= -0.99).any():
        raise ValueError("Ken French 해외 지수에 -99% 이하(결측 표기)가 있습니다.")
    return series


def read_french_dat(path: Path, name: str) -> str:
    with zipfile.ZipFile(path) as archive:
        return archive.read(name).decode("latin-1")


def parse_aqr_commodities(raw: pd.DataFrame, column: str = AQR_EW_EXCESS) -> pd.Series:
    """AQR "Commodities for the Long Run" 첫 시트(header=None)에서 한 열의 월 수익률을 꺼냅니다."""
    cells = raw.astype(str).apply(lambda col: col.str.strip())
    found = [(r, c) for r, c in zip(*np.nonzero((cells == column).to_numpy()))]
    if len(found) != 1:
        raise ValueError(f"AQR 파일에서 '{column}' 열을 찾지 못했습니다.")
    row, col = found[0]
    body = raw.iloc[row + 1 :]
    dates = pd.to_datetime(body.iloc[:, 0], errors="coerce")
    body = body[dates.notna()]
    series = pd.Series(
        body.iloc[:, col].astype(float).to_numpy(),
        index=pd.PeriodIndex(dates[dates.notna()].dt.to_period("M"), name="month"),
        name="commodity_er",
    )
    check_monthly_continuity(series.index, "AQR 원자재")
    if series.isna().any():
        raise ValueError("AQR 원자재 수익률에 빈 값이 있습니다.")
    return series


def parse_yahoo_adjclose(data: dict, tz: ZoneInfo = US_MARKET_TZ) -> pd.Series:
    """Yahoo의 수정 종가(분할과 배당 반영). 교차 확인에만 씁니다."""
    try:
        result = data["chart"]["result"][0]
        adjclose = result["indicators"]["adjclose"][0]["adjclose"]
    except (KeyError, IndexError, TypeError) as error:
        raise ValueError("Yahoo 응답에 수정 종가가 없습니다.") from error
    series = pd.Series(adjclose, index=yahoo_dates(result["timestamp"], tz), name="adjclose", dtype="float64").dropna()
    return series[~series.index.duplicated(keep="last")].sort_index()


def sp500_signal_returns(month_end: pd.DataFrame) -> pd.Series:
    """신호용 S&P 500 월 총수익률: (close[t] + D[t-12] / 12) / close[t-1] − 1.

    Shiller D는 직선 보간이라 최근 값에 앞으로의 배당이 섞여 있어, 신호에는 12개월 전 D를 씁니다.
    수익률 계산에는 쓰지 않습니다. 12개월 전 D가 있는 첫 달의 다음 달부터 값이 있습니다.
    """
    from src.monthly import total_return_signal_index

    first_lagged = month_end["dividend"].first_valid_index() + 12
    level = total_return_signal_index(month_end["close"].loc[first_lagged:].dropna(), month_end["dividend"], 12)
    return (level / level.shift(1) - 1).rename("us_stock_sig_tr")


def build_gtaa_monthly(
    month_end: pd.DataFrame, eafe: pd.Series, eafe_price: pd.Series, bond_yield: pd.Series,
    commodity_er: pd.Series, reits: dict[str, pd.DataFrame], current_month: pd.Period,
) -> pd.DataFrame:
    """GTAA 근사 재현용 월간 표. 열은 자산마다 총수익률(_tr)과 신호용 가격 수익률(_pr), 그리고 현금(rf)입니다.

    - us_stock: S&P 500 월말 종가 + Shiller 배당 (build_month_end의 표).
      us_stock_sig_tr은 신호용 총수익률로, 배당에 12개월 전 D를 씁니다. Shiller D는 직선 보간이라 최근 값에
      앞으로의 배당이 섞여 있기 때문입니다 (S&P 500 재현과 같은 규칙). 수익률 계산에는 쓰지 않습니다.
    - eafe: Ken French 해외 지수 전체(EAFE + 캐나다 비중), 가격 수익률은 배당 제외 파일
    - bond: 10년 국채 합성 (src/bonds.py). bond_yield는 월말 금리
    - commodity: AQR 동일 비중 원자재 선물. 총수익률 = 초과수익률 + 현금, 가격 수익률 = 초과수익률
    - reit_<이름>: NAREIT 지수들. 첫째가 기본입니다.
    current_month와 그 뒤는 끝나지 않은 달이라 뺍니다.
    """
    bond = par_bond_monthly_returns(bond_yield)
    months = pd.period_range("1971-12", current_month - 1, freq="M", name="month")
    rf = month_end["rf"].reindex(months)
    table = pd.DataFrame(index=months)
    table["us_stock_tr"] = month_end["total_return"].reindex(months)
    table["us_stock_pr"] = month_end["price_return"].reindex(months)
    table["us_stock_sig_tr"] = sp500_signal_returns(month_end).reindex(months)
    table["eafe_tr"] = eafe.reindex(months)
    table["eafe_pr"] = eafe_price.reindex(months)
    table["bond_tr"] = bond["total_return"].reindex(months)
    table["bond_pr"] = bond["price_return"].reindex(months)
    table["bond_yield"] = bond_yield.reindex(months)
    table["commodity_tr"] = commodity_er.reindex(months) + rf
    table["commodity_pr"] = commodity_er.reindex(months)
    for key, index in reits.items():
        table[f"{key}_tr"] = (index["total_index"] / index["total_index"].shift(1) - 1).reindex(months)
        table[f"{key}_pr"] = (index["price_index"] / index["price_index"].shift(1) - 1).reindex(months)
    table["rf"] = rf
    return table


def main_gtaa(offline: bool = False) -> None:
    """GTAA 근사 재현용 표를 만듭니다. S&P 500과 현금은 이미 받은 원본(^GSPC, Shiller, Ken French)을 씁니다(다시 받지 않음)."""
    paths = {
        "h15": RAW_DIR / "fed_h15_cmt.csv",
        "nareit": RAW_DIR / "nareit_monthly.xls",
        "french_int": RAW_DIR / "french_international_indices.zip",
        "french_int_wout_div": RAW_DIR / "french_international_indices_wout_div.zip",
        "aqr": RAW_DIR / "aqr_commodities_long_run.xlsx",
    }
    if not offline:
        download(FED_H15_CMT, paths["h15"], headers={"User-Agent": "Mozilla/5.0"})
        # 브라우저처럼 보이는 머리글을 보내면 reit.com은 파일 대신 브라우저 확인 페이지(HTML)를 줍니다. 기본 머리글로 받습니다.
        download(NAREIT_MONTHLY, paths["nareit"])
        download(FRENCH_INT_INDICES, paths["french_int"])
        download(FRENCH_INT_INDICES_WOUT_DIV, paths["french_int_wout_div"])
        download(AQR_COMMODITIES, paths["aqr"], headers={"User-Agent": "Mozilla/5.0"})
        for symbol in YAHOO_GTAA_SYMBOLS:
            download_yahoo(symbol)
    # 기준 시점은 H.15 원본을 내려받은 시각입니다. 그달과 그 뒤는 끝나지 않은 달이라 뺍니다.
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    now = datetime.fromisoformat(manifest[paths["h15"].name]["downloaded_at_utc"]).astimezone(US_MARKET_TZ)
    current_month = current_us_month(now)

    # S&P 500 월말 표: main()과 같은 원본으로 다시 만듭니다. 기준 시점은 main()처럼 ^GSPC 원본을 내려받은 시각입니다
    # (그 뒤의 달은 S&P 500 값이 비어 있습니다).
    gspc_now = datetime.fromisoformat(manifest[YAHOO_SYMBOLS["^GSPC"]]["downloaded_at_utc"]).astimezone(US_MARKET_TZ)
    shiller = parse_shiller(pd.read_excel(RAW_DIR / "shiller_ie_data.xls", sheet_name="Data", header=None))
    gspc = parse_yahoo(json.loads((RAW_DIR / YAHOO_SYMBOLS["^GSPC"]).read_text(encoding="utf-8")))
    rf = parse_french_rf(read_french_zip(RAW_DIR / "french_factors.zip"))
    month_end = build_month_end(gspc[gspc.index.date < gspc_now.date()], shiller, rf, current_us_month(gspc_now))

    bond_yield = month_end_value(parse_h15_yield(paths["h15"].read_text(encoding="utf-8")))
    bond_yield = bond_yield[bond_yield.index < current_month]
    check_monthly_continuity(bond_yield.index, "H.15 10년 금리")
    eafe = parse_french_international(read_french_dat(paths["french_int"], "Ind_all.Dat"))
    eafe_price = parse_french_international(read_french_dat(paths["french_int_wout_div"], "Ind_all.Dat"))
    commodity_er = parse_aqr_commodities(pd.read_excel(paths["aqr"], sheet_name=0, header=None))
    nareit_raw = pd.read_excel(paths["nareit"], sheet_name="Index Data", header=None)
    reits = {key: parse_nareit(nareit_raw, name) for key, name in
             [("reit", "All REITs"), ("reit_all_equity", "All Equity REITs"), ("reit_equity", "Equity REITs")]}

    table = build_gtaa_monthly(month_end, eafe, eafe_price, bond_yield, commodity_er, reits, current_month)
    efa = month_end_value(parse_yahoo_adjclose(json.loads((RAW_DIR / YAHOO_GTAA_SYMBOLS["EFA"]).read_text(encoding="utf-8"))))
    efa = efa[efa.index <= EFA_CHECK_END]
    efa_table = pd.DataFrame({"efa_tr": efa / efa.shift(1) - 1, "eafe_proxy_tr": table["eafe_tr"].reindex(efa.index)})

    for name, df in {"gtaa_monthly.csv": table, "efa_monthly.csv": efa_table}.items():
        df.to_csv(DATA_DIR / name)
        print(f"저장: data/{name} ({df.index[0]} ~ {df.index[-1]}, {len(df)}행)")
    print("열마다 값이 있는 첫 달과 마지막 달:")
    for column in table.columns:
        valid = table[column].dropna()
        print(f"  {column}: {valid.index[0]} ~ {valid.index[-1]}")


def build_dual_momentum_monthly(
    month_end: pd.DataFrame, ex_us: pd.Series, yield5: pd.Series, yield10: pd.Series, current_month: pd.Period,
) -> pd.DataFrame:
    """듀얼 모멘텀 근사 재현용 월간 표.

    - us_stock_tr: S&P 500 월말 종가 + Shiller 배당. us_stock_sig_tr: 신호용(12개월 전 배당, sp500_signal_returns)
    - ex_us_tr: 미국 외 주식 대체 자료 (Ken French 해외 지수 전체, EAFE + 캐나다 비중, 신흥국 없음)
    - bond_tr: 5년 만기 국채 합성 (Barclays US Aggregate 대체), bond10_tr: 10년 만기 국채 합성 (참고)
    - rf: Ken French RF
    current_month와 그 뒤는 끝나지 않은 달이라 뺍니다.
    """
    months = pd.period_range("1971-12", current_month - 1, freq="M", name="month")
    table = pd.DataFrame(index=months)
    table["us_stock_tr"] = month_end["total_return"].reindex(months)
    table["us_stock_sig_tr"] = sp500_signal_returns(month_end).reindex(months)
    table["ex_us_tr"] = ex_us.reindex(months)
    table["bond_tr"] = par_bond_monthly_returns(yield5, 5.0)["total_return"].reindex(months)
    table["bond10_tr"] = par_bond_monthly_returns(yield10, 10.0)["total_return"].reindex(months)
    table["rf"] = month_end["rf"].reindex(months)
    return table


def main_dual_momentum(offline: bool = False) -> None:
    """듀얼 모멘텀 근사 재현용 표를 만듭니다.

    S&P 500, 현금, H.15 금리, Ken French 해외 지수는 이미 받은 원본(main, main_gtaa)을 씁니다(다시 받지 않음).
    교차 확인용 ETF(ACWX, AGG)만 새로 받습니다.
    """
    if not offline:
        for symbol in YAHOO_DM_SYMBOLS:
            download_yahoo(symbol)
    h15_path = RAW_DIR / "fed_h15_cmt.csv"
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    # 기준 시점은 H.15 원본을 내려받은 시각입니다 (main_gtaa와 같음).
    now = datetime.fromisoformat(manifest[h15_path.name]["downloaded_at_utc"]).astimezone(US_MARKET_TZ)
    current_month = current_us_month(now)
    gspc_now = datetime.fromisoformat(manifest[YAHOO_SYMBOLS["^GSPC"]]["downloaded_at_utc"]).astimezone(US_MARKET_TZ)
    shiller = parse_shiller(pd.read_excel(RAW_DIR / "shiller_ie_data.xls", sheet_name="Data", header=None))
    gspc = parse_yahoo(json.loads((RAW_DIR / YAHOO_SYMBOLS["^GSPC"]).read_text(encoding="utf-8")))
    rf = parse_french_rf(read_french_zip(RAW_DIR / "french_factors.zip"))
    month_end = build_month_end(gspc[gspc.index.date < gspc_now.date()], shiller, rf, current_us_month(gspc_now))

    h15 = h15_path.read_text(encoding="utf-8")
    yields = {}
    for key, identifier in [("5y", H15_5Y), ("10y", H15_10Y)]:
        y = month_end_value(parse_h15_yield(h15, identifier))
        yields[key] = y[y.index < current_month]
        check_monthly_continuity(yields[key].index, f"H.15 {key} 금리")
    ex_us = parse_french_international(read_french_dat(RAW_DIR / "french_international_indices.zip", "Ind_all.Dat"))
    table = build_dual_momentum_monthly(month_end, ex_us, yields["5y"], yields["10y"], current_month)

    check = {}
    for symbol in YAHOO_DM_SYMBOLS:
        price = month_end_value(parse_yahoo_adjclose(json.loads((RAW_DIR / YAHOO_DM_SYMBOLS[symbol]).read_text(encoding="utf-8"))))
        price = price[price.index <= DM_CHECK_END]
        check[f"{symbol.lower()}_tr"] = price / price.shift(1) - 1
    check_table = pd.DataFrame(check)
    check_table["ex_us_proxy_tr"] = table["ex_us_tr"].reindex(check_table.index)
    check_table["bond_proxy_tr"] = table["bond_tr"].reindex(check_table.index)
    # 채권 대체 자료의 만기를 바꿔 AGG와 비교하는 원인 확인용 (판단에 쓰지 않음)
    for years, identifier in [(3, "RIFLGFCY03_N.B"), (7, "RIFLGFCY07_N.B"), (10, H15_10Y)]:
        y = month_end_value(parse_h15_yield(h15, identifier))
        y = y[y.index < current_month]
        bond = par_bond_monthly_returns(y.loc["1990-01":], float(years))["total_return"]
        check_table[f"bond{years}_proxy_tr"] = bond.reindex(check_table.index)

    for name, df in {"dual_momentum_monthly.csv": table, "dual_momentum_check.csv": check_table}.items():
        df.to_csv(DATA_DIR / name)
        print(f"저장: data/{name} ({df.index[0]} ~ {df.index[-1]}, {len(df)}행)")
    for column in table.columns:
        valid = table[column].dropna()
        print(f"  {column}: {valid.index[0]} ~ {valid.index[-1]}")


def main(offline: bool = False) -> None:
    if offline:
        # 이미 받은 원본으로 표만 다시 만듭니다.
        shiller_path = RAW_DIR / "shiller_ie_data.xls"
        yahoo_paths = {symbol: RAW_DIR / name for symbol, name in YAHOO_SYMBOLS.items()}
        french_path = RAW_DIR / "french_factors.zip"
    else:
        shiller_path = download_shiller()
        yahoo_paths = {symbol: download_yahoo(symbol) for symbol in YAHOO_SYMBOLS}
        french_path = download_french()

    shiller = parse_shiller(pd.read_excel(shiller_path, sheet_name="Data", header=None))
    yahoo = {symbol: json.loads(path.read_text(encoding="utf-8")) for symbol, path in yahoo_paths.items()}
    rf = parse_french_rf(read_french_zip(french_path))

    if offline:
        # 기준 시점은 ^GSPC 원본을 내려받은 시각입니다.
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        now = datetime.fromisoformat(manifest[YAHOO_SYMBOLS["^GSPC"]]["downloaded_at_utc"]).astimezone(US_MARKET_TZ)
    else:
        now = datetime.now(US_MARKET_TZ)
    current_month = current_us_month(now)

    def closed_days(close: pd.Series) -> pd.Series:
        return close[close.index.date < now.date()]  # 오늘은 장중일 수 있어 뺍니다.

    daily = closed_days(parse_yahoo(yahoo["^GSPC"]))
    month_end = build_month_end(daily, shiller, rf, current_month)
    monthly_avg = build_monthly_avg(shiller, rf, current_month)
    spy = build_etf_daily(
        closed_days(parse_yahoo(yahoo["SPY"])), parse_yahoo_dividends(yahoo["SPY"]), rf, current_month
    )
    sp500tr = closed_days(parse_yahoo(yahoo["^SP500TR"])).to_frame()

    outputs = {
        "sp500_daily.csv": daily.to_frame(),
        "sp500_month_end.csv": month_end,
        "sp500_monthly_avg.csv": monthly_avg,
        "spy_daily.csv": spy,
        "sp500tr_daily.csv": sp500tr,
    }
    for name, df in outputs.items():
        df.to_csv(DATA_DIR / name)
        print(f"저장: data/{name} ({df.index[0]} ~ {df.index[-1]}, {len(df)}행)")

    print(f"월말 표 총수익률 마지막 달: {month_end['total_return'].last_valid_index()}")
    print(f"월말 표 RF 마지막 달: {month_end['rf'].last_valid_index()}")
    print(f"월말 표 결측 개수:\n{month_end.isna().sum().to_string()}")


def main_kospi(offline: bool = False) -> None:
    """KOSPI 실험용 표를 만듭니다. 미국 데이터의 원본과 표는 건드리지 않습니다."""
    rate_path = RAW_DIR / "oecd_kor_ir3tib.csv"
    if offline:
        yahoo_paths = {symbol: RAW_DIR / name for symbol, name in YAHOO_KR_SYMBOLS.items()}
        # 기준 시점은 ^KS11 원본을 내려받은 시각입니다.
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        now = datetime.fromisoformat(manifest[YAHOO_KR_SYMBOLS["^KS11"]]["downloaded_at_utc"]).astimezone(KR_MARKET_TZ)
    else:
        yahoo_paths = {symbol: download_yahoo(symbol) for symbol in YAHOO_KR_SYMBOLS}
        download_oecd_kr_rate(rate_path)
        now = datetime.now(KR_MARKET_TZ)
    current_month = pd.Period(now.date(), freq="M")

    yahoo = {symbol: json.loads(path.read_text(encoding="utf-8")) for symbol, path in yahoo_paths.items()}
    cd_rate = parse_oecd_rate(rate_path.read_text(encoding="utf-8"))

    # 오늘은 장중일 수 있어 뺍니다.
    ohlc = parse_yahoo_ohlc(yahoo["^KS11"])
    kospi = build_kospi_daily(ohlc[ohlc.index.date < now.date()], cd_rate, current_month)

    first, last = KODEX200_RANGE
    kodex_close = parse_yahoo(yahoo["069500.KS"], KR_MARKET_TZ).loc[first:last]
    kodex_dividends = parse_yahoo_dividends(yahoo["069500.KS"], KR_MARKET_TZ, between=KODEX200_RANGE)
    kodex = build_etf_daily(kodex_close, kodex_dividends, cd_rate / 12, current_month)

    for name, df in {"kospi_daily.csv": kospi, "kodex200_daily.csv": kodex}.items():
        df.to_csv(DATA_DIR / name)
        print(f"저장: data/{name} ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}, {len(df)}행)")
    print(f"CD 금리 마지막 달: {cd_rate.index[-1]}")


def main_inverse(offline: bool = False) -> None:
    """인버스 ETF 교차 확인용 표를 만듭니다. 현금 수익률은 이미 받은 Ken French·OECD 원본을 씁니다(다시 받지 않음).

    표에는 INVERSE_RANGE(설계용 기간 안)의 값만 남깁니다.
    """
    if offline:
        yahoo_paths = {symbol: RAW_DIR / name for symbol, name in YAHOO_INVERSE_SYMBOLS.items()}
    else:
        yahoo_paths = {symbol: download_yahoo(symbol) for symbol in YAHOO_INVERSE_SYMBOLS}
    yahoo = {symbol: json.loads(path.read_text(encoding="utf-8")) for symbol, path in yahoo_paths.items()}
    us_rf = parse_french_rf(read_french_zip(RAW_DIR / "french_factors.zip"))
    cd_rate = parse_oecd_rate((RAW_DIR / "oecd_kor_ir3tib.csv").read_text(encoding="utf-8"))
    # 범위 끝이 2012년이라 끝나지 않은 달이 없습니다. daily_rf에는 범위 다음 달을 넘깁니다.
    after = pd.Period("2013-01", freq="M")

    first, last = INVERSE_RANGE["SH"]
    sh = build_etf_daily(
        parse_yahoo(yahoo["SH"]).loc[first:last],
        parse_yahoo_dividends(yahoo["SH"], between=INVERSE_RANGE["SH"], splits_after=last),
        us_rf, after,
    )
    first, last = INVERSE_RANGE["114800.KS"]
    kodex_inverse = build_etf_daily(
        parse_yahoo(yahoo["114800.KS"], KR_MARKET_TZ).loc[first:last],
        parse_yahoo_dividends(yahoo["114800.KS"], KR_MARKET_TZ, between=INVERSE_RANGE["114800.KS"]),
        cd_rate / 12, after,
    )
    for name, df in {"sh_daily.csv": sh, "kodex_inverse_daily.csv": kodex_inverse}.items():
        df.to_csv(DATA_DIR / name)
        print(f"저장: data/{name} ({df.index[0]:%Y-%m-%d} ~ {df.index[-1]:%Y-%m-%d}, {len(df)}행)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="시세 데이터 수집과 정리")
    parser.add_argument("--offline", action="store_true", help="내려받지 않고 data/raw/의 원본으로 표만 다시 만듭니다")
    parser.add_argument("--kospi", action="store_true", help="KOSPI 실험용 표만 만듭니다")
    parser.add_argument("--inverse", action="store_true", help="인버스 ETF(SH, KODEX 인버스) 교차 확인용 표만 만듭니다")
    parser.add_argument("--gtaa", action="store_true", help="GTAA 근사 재현용 5자산 월간 표만 만듭니다")
    parser.add_argument("--dual-momentum", action="store_true", help="듀얼 모멘텀 근사 재현용 월간 표만 만듭니다")
    args = parser.parse_args()
    if args.dual_momentum:
        main_dual_momentum(offline=args.offline)
    elif args.gtaa:
        main_gtaa(offline=args.offline)
    elif args.inverse:
        main_inverse(offline=args.offline)
    elif args.kospi:
        main_kospi(offline=args.offline)
    else:
        main(offline=args.offline)
