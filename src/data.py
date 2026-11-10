"""시세 데이터 수집과 정리.

- Yahoo: S&P 500 일별 종가 (^GSPC, 1927-12-30부터). 월말 종가는 여기서 만듭니다.
- Yahoo: SPY 일별 종가와 배당 (1993-01-29부터), S&P 500 총수익 지수 (^SP500TR, 1988년부터, 교차 확인용)
- Shiller: S&P 500 월간 배당과 월평균 가격 (shillerdata.com, 1871년부터)
- Ken French: 1개월 국채 월 수익률 RF (1926-07부터)
- Yahoo: KOSPI 일별 시가, 고가, 저가, 종가 (^KS11, 1996-12-11부터), KODEX 200 종가와 분배금 (069500.KS, 교차 확인용)
- OECD: 원화 CD 91일물 월평균 금리 (IR3TIB, 1996-01부터)

실행: python -m src.data (원본으로 표만 다시 만들 때: python -m src.data --offline)
KOSPI 표는 따로 만듭니다: python -m src.data --kospi (원본으로 다시 만들 때: --kospi --offline)
원본은 data/raw/에 두고, 내려받은 시각과 SHA256을 data/raw/manifest.json에 남깁니다.
"""

import hashlib
import io
import json
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests

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
    name = {**YAHOO_SYMBOLS, **YAHOO_KR_SYMBOLS}[symbol]
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


def parse_yahoo_dividends(data: dict, tz: ZoneInfo = US_MARKET_TZ, between: tuple[str, str] | None = None) -> pd.Series:
    """Yahoo 배당 기록(배당락일 기준, 주당 금액)을 꺼냅니다.

    between: 이 날짜 범위의 기록만 씁니다. 시세가 비어 있는 시기의 기록은 날짜가 어긋날 수 있어 범위로 거릅니다.
    분할 기록이 있으면 멈춥니다. 분할이 있는 종목은 배당과 가격의 기준을 맞추는 처리가 따로 필요합니다.
    """
    events = data["chart"]["result"][0].get("events", {})
    if events.get("splits"):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="시세 데이터 수집과 정리")
    parser.add_argument("--offline", action="store_true", help="내려받지 않고 data/raw/의 원본으로 표만 다시 만듭니다")
    parser.add_argument("--kospi", action="store_true", help="KOSPI 실험용 표만 만듭니다")
    args = parser.parse_args()
    if args.kospi:
        main_kospi(offline=args.offline)
    else:
        main(offline=args.offline)
