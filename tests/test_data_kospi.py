"""KOSPI 데이터 정리 코드 테스트. 네트워크 없이 작은 가짜 표로 확인합니다."""

import numpy as np
import pandas as pd
import pytest

from src.data import KR_MARKET_TZ, build_kospi_daily, parse_oecd_rate, parse_yahoo, parse_yahoo_dividends, parse_yahoo_ohlc

# 2020-01-06(월) ~ 01-09(목) 서울 09:00
TIMESTAMPS = [int(pd.Timestamp(f"2020-01-0{d} 09:00", tz="Asia/Seoul").timestamp()) for d in [6, 7, 8, 9]]


def yahoo_response(open_, high, low, close, volume, events=None, timestamps=TIMESTAMPS) -> dict:
    result = {
        "timestamp": timestamps,
        "indicators": {"quote": [{"open": open_, "high": high, "low": low, "close": close, "volume": volume}]},
    }
    if events:
        result["events"] = events
    return {"chart": {"result": [result], "error": None}}


def oecd_csv(rows: list[tuple[str, float | str]], ref_area="KOR", measure="IR3TIB") -> str:
    lines = ["REF_AREA,FREQ,MEASURE,UNIT_MEASURE,TIME_PERIOD,OBS_VALUE"]
    lines += [f"{ref_area},M,{measure},PA,{month},{value}" for month, value in rows]
    return "\n".join(lines)


def test_parse_ohlc_uses_seoul_dates_and_drops_empty_and_stale_rows():
    data = yahoo_response(
        [10.0, None, 11.0, 12.0],
        [11.0, None, 11.0, 13.0],
        [9.0, None, 11.0, 11.0],
        [10.5, None, 11.0, 12.5],
        [100.0, None, 0.0, 100.0],
    )
    ohlc = parse_yahoo_ohlc(data)
    # 01-07은 종가 없음, 01-08은 거래량 0이고 고가 = 저가
    assert list(ohlc.index.strftime("%Y-%m-%d")) == ["2020-01-06", "2020-01-09"]
    assert ohlc.loc["2020-01-09", "high"] == 13.0


def test_parse_ohlc_keeps_flat_row_with_volume():
    data = yahoo_response([10.0] * 4, [10.0] * 4, [10.0] * 4, [10.0] * 4, [5.0] * 4)
    assert len(parse_yahoo_ohlc(data)) == 4


def test_parse_ohlc_rejects_high_below_close():
    data = yahoo_response([10.0] * 4, [10.0, 10.0, 10.0, 9.0], [9.0] * 4, [10.0, 10.0, 10.0, 9.5], [5.0] * 4)
    with pytest.raises(ValueError, match="고가"):
        parse_yahoo_ohlc(data)


def test_parse_ohlc_rejects_non_positive_price():
    data = yahoo_response([10.0] * 4, [11.0] * 4, [0.0] * 4, [10.0] * 4, [5.0] * 4)
    with pytest.raises(ValueError, match="0 이하"):
        parse_yahoo_ohlc(data)


def test_seoul_date_is_not_shifted_to_previous_day():
    # 서울 09:00은 UTC 00:00입니다. 뉴욕 시간으로 바꾸면 전날이 되므로 시간대를 지정해야 합니다.
    data = yahoo_response([1.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4)
    close = parse_yahoo(data, KR_MARKET_TZ)
    assert close.index[0] == pd.Timestamp("2020-01-06")


def test_parse_dividends_filters_by_range_before_checking_dates():
    # 범위 밖 기록은 일요일(2007-04-29)이어도 오류를 내지 않습니다.
    sunday = int(pd.Timestamp("2007-04-29 09:00", tz="Asia/Seoul").timestamp())
    inside = int(pd.Timestamp("2010-04-29 09:00", tz="Asia/Seoul").timestamp())
    events = {"dividends": {"a": {"date": sunday, "amount": 100.0}, "b": {"date": inside, "amount": 200.0}}}
    data = yahoo_response([1.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4, [1.0] * 4, events=events)
    dividends = parse_yahoo_dividends(data, KR_MARKET_TZ, between=("2009-06-01", "2012-12-31"))
    assert dividends.to_dict() == {pd.Timestamp("2010-04-29"): 200.0}
    with pytest.raises(ValueError, match="장중 평일"):
        parse_yahoo_dividends(data, KR_MARKET_TZ)


def test_parse_oecd_rate_sorts_and_converts_to_decimal():
    rate = parse_oecd_rate(oecd_csv([("1997-03", 13.23), ("1997-01", 12.66), ("1997-02", 12.28)]))
    assert rate[pd.Period("1997-01", "M")] == pytest.approx(0.1266)
    assert list(rate.index.astype(str)) == ["1997-01", "1997-02", "1997-03"]


def test_parse_oecd_rate_rejects_missing_month_and_other_country():
    with pytest.raises(ValueError, match="빠진 달"):
        parse_oecd_rate(oecd_csv([("1997-01", 12.66), ("1997-03", 13.23)]))
    with pytest.raises(ValueError, match="REF_AREA"):
        parse_oecd_rate(oecd_csv([("1997-01", 12.66)], ref_area="JPN"))
    with pytest.raises(ValueError, match="빈 값"):
        parse_oecd_rate(oecd_csv([("1997-01", ""), ("1997-02", 12.28)]))


def test_build_kospi_daily_splits_annual_rate_by_trading_days():
    index = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-02-03", "2020-03-02"])
    ohlc = pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0}, index=index)
    rate = pd.Series([0.12, 0.24], index=pd.PeriodIndex(["2020-01", "2020-02"], freq="M"))
    daily = build_kospi_daily(ohlc, rate, pd.Period("2020-04", "M"))
    # 1월: 월 1%를 2거래일로 기하 분할, 2월: 월 2%를 1거래일로
    assert (1 + daily.loc["2020-01-06", "rf"]) * (1 + daily.loc["2020-01-07", "rf"]) == pytest.approx(1.01)
    assert daily.loc["2020-02-03", "rf"] == pytest.approx(0.02)
    # 금리가 없는 달은 NaN으로 남습니다.
    assert np.isnan(daily.loc["2020-03-02", "rf"])
