"""데이터 정리 코드 테스트. 네트워크 없이 작은 가짜 표로 확인합니다."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.data import (
    build_month_end,
    build_monthly_avg,
    check_monthly_continuity,
    current_us_month,
    find_partial_month_notes,
    find_shiller_url,
    parse_french_rf,
    parse_shiller,
    parse_shiller_date,
    parse_yahoo,
    record_manifest,
    total_return,
)


def months(start: str, periods: int) -> pd.PeriodIndex:
    return pd.period_range(start, periods=periods, freq="M", name="month")


def make_raw_shiller(rows: list[list], footnote: str = "Oct price is Oct 2nd close") -> pd.DataFrame:
    """ie_data.xls Data 시트와 같은 모양(머리글 위 제목 행, 머리글 행, 데이터, 각주)을 만듭니다."""
    width = 8
    header = ["Date", "P", "D", "E", "CPI", "Date Fraction", "Rate GS10", "Real Price"]
    title = ["Stock Market Data"] + [np.nan] * (width - 1)
    note = [np.nan, footnote] + [np.nan] * (width - 2)
    data = [[d, p, div, np.nan, cpi, np.nan, gs10, np.nan] for d, p, div, cpi, gs10 in rows]
    return pd.DataFrame([title, header, *data, note])


# ---------------------------------------------------------------- Shiller


def test_parse_shiller_date_handles_october():
    # 2026.1은 2026년 1월이 아니라 10월입니다.
    assert parse_shiller_date(2026.1) == pd.Period("2026-10", freq="M")
    assert parse_shiller_date(1871.01) == pd.Period("1871-01", freq="M")
    assert parse_shiller_date(1999.12) == pd.Period("1999-12", freq="M")


def test_parse_shiller_drops_partial_month_from_footnote():
    raw = make_raw_shiller([
        [2026.08, 100.0, 12.0, 300.0, 4.0],
        [2026.09, 110.0, np.nan, 301.0, 4.1],
        [2026.1, 111.0, np.nan, 302.0, 4.2],
    ])
    df = parse_shiller(raw)
    assert list(df.columns) == ["price", "dividend", "cpi", "gs10"]
    # 각주가 10월은 10월 2일 종가뿐이라고 밝혔으므로 10월은 빠집니다.
    assert list(df.index.astype(str)) == ["2026-08", "2026-09"]


def test_parse_shiller_keeps_last_month_without_partial_note():
    raw = make_raw_shiller(
        [[2026.08, 100.0, 12.0, 300.0, 4.0], [2026.09, 110.0, np.nan, 301.0, 4.1]],
        footnote="Sept/Oct CPI estimated",
    )
    assert list(parse_shiller(raw).index.astype(str)) == ["2026-08", "2026-09"]


def test_find_partial_month_notes():
    raw = pd.DataFrame([["Oct price is Oct 2nd close", "Sept/Oct CPI estimated", 1.0]])
    assert find_partial_month_notes(raw) == {"Oct"}
    assert find_partial_month_notes(pd.DataFrame([["Sept. price is Sept 3rd close"]])) == {"Sep"}


def test_parse_shiller_rejects_changed_layout():
    raw = make_raw_shiller([[2026.08, 100.0, 12.0, 300.0, 4.0]])
    raw.iloc[1, 2] = "E"  # 배당 열 자리가 바뀐 경우
    with pytest.raises(ValueError):
        parse_shiller(raw)


def test_parse_shiller_rejects_missing_month():
    raw = make_raw_shiller(
        [[2026.06, 100.0, 12.0, 300.0, 4.0], [2026.08, 101.0, 12.0, 300.0, 4.0]],
        footnote="",
    )
    with pytest.raises(ValueError, match="빠진 달"):
        parse_shiller(raw)


# ---------------------------------------------------------------- Yahoo, Ken French


def test_parse_yahoo_uses_new_york_dates():
    # 2026-10-09 16:00 뉴욕 = 20:00 UTC, 2026-10-08 16:00 뉴욕 = 20:00 UTC
    ts = [
        int(datetime(2026, 10, 8, 20, 0, tzinfo=timezone.utc).timestamp()),
        int(datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc).timestamp()),
        int(datetime(2026, 10, 9, 20, 0, tzinfo=timezone.utc).timestamp()),  # 중복
    ]
    data = {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": [100.0, None, 101.0]}]}}]}}
    close = parse_yahoo(data)
    assert list(close.index.strftime("%Y-%m-%d")) == ["2026-10-08", "2026-10-09"]
    assert list(close) == [100.0, 101.0]


def test_parse_yahoo_rejects_midnight_utc_timestamps():
    # 00:00 UTC는 뉴욕 전날 저녁 8시입니다. 날짜가 하루 밀리므로 받아들이지 않습니다.
    ts = [int(datetime(2026, 10, 9, 0, 0, tzinfo=timezone.utc).timestamp())]
    data = {"chart": {"result": [{"timestamp": ts, "indicators": {"quote": [{"close": [100.0]}]}}]}}
    with pytest.raises(ValueError, match="장중"):
        parse_yahoo(data)


def test_parse_yahoo_rejects_unexpected_format():
    with pytest.raises(ValueError):
        parse_yahoo({"chart": {"result": None, "error": {"code": "Not Found"}}})


FRENCH_TEXT = """This file was created using the 202608 CRSP database.

,Mkt-RF,SMB,HML,RF
192607,   2.89,  -2.42,  -2.75,   0.22
192608,   2.64,  -1.44,   4.13,   0.25

 Annual Factors: January-December
,Mkt-RF,SMB,HML,RF
  1927,  29.47,  -2.46,  -3.75,   3.12
"""


def test_parse_french_rf_reads_monthly_only():
    rf = parse_french_rf(FRENCH_TEXT)
    assert list(rf.index.astype(str)) == ["1926-07", "1926-08"]
    assert rf.iloc[0] == pytest.approx(0.0022)
    assert rf.iloc[1] == pytest.approx(0.0025)


# ---------------------------------------------------------------- 총수익률과 표 만들기


def test_total_return_uses_monthly_dividend():
    price = pd.Series([100.0, 102.0, 101.0], index=months("2000-01", 3))
    dividend = pd.Series([12.0, 12.0, np.nan], index=price.index)
    tr = total_return(price, dividend)
    assert np.isnan(tr.iloc[0])
    # (102 + 12/12) / 100 - 1 = 0.03
    assert tr.iloc[1] == pytest.approx(0.03)
    # 배당이 없는 달은 가격 수익률로 채우지 않습니다.
    assert np.isnan(tr.iloc[2])


def test_total_return_does_not_use_future_rows():
    price = pd.Series([100.0, 102.0, 101.0, 105.0], index=months("2000-01", 4))
    dividend = pd.Series([12.0] * 4, index=price.index)
    changed_price, changed_dividend = price.copy(), dividend.copy()
    changed_price.iloc[3], changed_dividend.iloc[3] = 999.0, 99.0
    pd.testing.assert_series_equal(
        total_return(price, dividend).iloc[:3], total_return(changed_price, changed_dividend).iloc[:3]
    )


def make_daily(prices: dict[str, float]) -> pd.Series:
    return pd.Series(list(prices.values()), index=pd.DatetimeIndex(list(prices), name="date"), name="close")


def test_build_month_end_uses_last_trading_day_and_drops_current_month():
    daily = make_daily({
        "2026-07-30": 90.0, "2026-07-31": 100.0,
        "2026-08-03": 101.0, "2026-08-31": 103.0,
        "2026-09-29": 104.0, "2026-09-30": 105.0,
        "2026-10-01": 106.0,  # 진행 중인 달
    })
    shiller = pd.DataFrame({"dividend": [12.0, 12.0, np.nan]}, index=months("2026-07", 3))
    rf = pd.Series([0.003, 0.003, 0.0031], index=months("2026-07", 3))
    df = build_month_end(daily, shiller, rf, current_month=pd.Period("2026-10", freq="M"))

    assert list(df.index.astype(str)) == ["2026-07", "2026-08", "2026-09"]
    assert list(df["close"]) == [100.0, 103.0, 105.0]
    assert df.loc[pd.Period("2026-08", freq="M"), "last_trading_day"] == pd.Timestamp("2026-08-31")
    # (103 + 12/12) / 100 - 1 = 0.04
    assert df["total_return"].iloc[1] == pytest.approx(0.04)
    assert np.isnan(df["total_return"].iloc[2])  # 9월 배당이 비어 있음
    assert df["price_return"].iloc[2] == pytest.approx(105.0 / 103.0 - 1)
    assert list(df["rf"]) == [0.003, 0.003, 0.0031]


def test_build_month_end_rejects_missing_month():
    daily = make_daily({"2026-06-30": 100.0, "2026-08-31": 101.0})
    shiller = pd.DataFrame({"dividend": [12.0] * 3}, index=months("2026-06", 3))
    rf = pd.Series([0.003] * 3, index=months("2026-06", 3))
    with pytest.raises(ValueError, match="빠진 달"):
        build_month_end(daily, shiller, rf, current_month=pd.Period("2026-10", freq="M"))


def test_build_monthly_avg_drops_current_month_and_aligns_rf():
    shiller = pd.DataFrame(
        {"price": [100.0, 101.0, 102.0], "dividend": [12.0] * 3, "cpi": [1.0] * 3, "gs10": [4.0] * 3},
        index=months("2026-08", 3),
    )
    rf = pd.Series([0.0030], index=months("2026-08", 1))
    df = build_monthly_avg(shiller, rf, current_month=pd.Period("2026-10", freq="M"))
    assert list(df.index.astype(str)) == ["2026-08", "2026-09"]
    assert df["rf"].iloc[0] == 0.0030
    assert np.isnan(df["rf"].iloc[1])


# ---------------------------------------------------------------- 기타


def test_current_us_month_uses_new_york_time():
    # 한국 시간 11월 1일 오전 = 뉴욕 10월 31일 저녁: 아직 10월입니다.
    now = datetime(2026, 10, 31, 23, 0, tzinfo=timezone.utc)
    assert current_us_month(now) == pd.Period("2026-10", freq="M")


def test_check_monthly_continuity():
    check_monthly_continuity(months("2026-01", 3), "ok")
    with pytest.raises(ValueError):
        check_monthly_continuity(pd.PeriodIndex(["2026-01", "2026-03"], freq="M"), "gap")


def test_record_manifest(tmp_path):
    path = tmp_path / "a.csv"
    path.write_bytes(b"abc")
    manifest_path = tmp_path / "manifest.json"
    record_manifest(path, "https://example.com/a.csv", manifest_path)
    entry = json.loads(manifest_path.read_text(encoding="utf-8"))["a.csv"]
    assert entry["url"] == "https://example.com/a.csv"
    assert entry["bytes"] == 3
    assert entry["sha256"] == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_find_shiller_url():
    html = '<a href="//img1.wsimg.com/blobby/go/abc/downloads/def/ie_data.xls?ver=123">ie_data</a>'
    assert find_shiller_url(html) == "https://img1.wsimg.com/blobby/go/abc/downloads/def/ie_data.xls?ver=123"
    with pytest.raises(ValueError):
        find_shiller_url("<html></html>")
