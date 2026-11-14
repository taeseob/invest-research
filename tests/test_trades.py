"""매매 목록(trade_list) 테스트와, 신호 낸 날 종가 체결(lag 0)의 미래 정보 검사."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import Costs, holding_from_signal, moving_average_signal, period_last_days, run_backtest, trade_list


def make_data(n: int, seed: int, drift=0.07, vol=0.16) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift / 252, vol / np.sqrt(252), n))
    return pd.DataFrame({"close": close, "dividend": 0.0, "rf": 0.0001}, index=pd.bdate_range("2000-01-03", periods=n))


def strategy_holding(data: pd.DataFrame, window: int = 20, lag: int = 0) -> pd.Series:
    weeks = period_last_days(data.index, "W-FRI")
    return holding_from_signal(moving_average_signal(data["close"], weeks, window), data.index, lag)


# ---------------------------------------------------------------- 매매 목록


def test_trade_list_finds_peak_and_drawdown_at_sale():
    index = pd.bdate_range("2020-01-01", periods=8)
    price = pd.Series([10, 11, 13, 12, 11.5, 11, 12, 14.0], index=index)
    after_close = pd.Series([0, 1, 1, 1, 0, 0, 1, 1], index=index, dtype=float)
    trades = trade_list(after_close, price)

    assert len(trades) == 2
    first = trades.iloc[0]
    assert (first["buy_date"], first["peak_date"], first["sell_date"]) == (index[1], index[2], index[4])
    assert (first["buy_price"], first["peak_price"], first["sell_price"]) == (11.0, 13.0, 11.5)
    assert first["from_peak"] == pytest.approx(11.5 / 13 - 1)
    assert first["from_buy"] == pytest.approx(11.5 / 11 - 1)
    assert first["weeks_peak_to_sell"] == pytest.approx((index[4] - index[2]).days / 7)
    assert not first["open"]

    # 기간 끝까지 보유: 매도가 아니므로 open이고, 마지막 날을 매도일 자리에 둡니다.
    last = trades.iloc[1]
    assert (last["buy_date"], last["sell_date"], last["open"]) == (index[6], index[7], True)
    assert last["from_peak"] == 0.0


def test_trade_list_position_at_start_and_sale_on_last_day():
    index = pd.bdate_range("2020-01-01", periods=6)
    price = pd.Series([10, 12, 11, 10, 9, 9.5], index=index)
    # 첫날(시작 전날)부터 보유 → 2일째 매도, 4일째 매수 → 마지막 날 매도
    after_close = pd.Series([1, 1, 0, 0, 1, 0], index=index, dtype=float)
    trades = trade_list(after_close, price)
    assert list(trades["buy_date"]) == [index[0], index[4]]
    assert list(trades["sell_date"]) == [index[2], index[5]]
    assert not trades["open"].any()  # 마지막 날에 판 것은 기간 끝까지 보유한 것과 다릅니다.
    assert trades["from_peak"].iloc[0] == pytest.approx(11 / 12 - 1)
    assert trades["from_peak"].iloc[1] == 0.0  # 9에 사서 9.5에 팔았으므로 고점이 매도일입니다.


def test_trade_list_equity_drawdown_includes_costs_and_dividends():
    index = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100, 110, 120, 105, 100, 100.0], "dividend": [0, 0, 2.0, 0, 0, 0], "rf": 0.0}, index=index)
    after_close = pd.Series([0, 1, 1, 1, 0, 0], index=index, dtype=float)
    result = run_backtest(data, after_close, str(index[1].date()), str(index[-1].date()), Costs(trade=0.01))
    window = after_close.reindex(result.daily.index)
    trades = trade_list(window, data["close"], result.daily["equity"])

    # 자산: 1일 종가 매수(비용 1%) → 2일 (120 + 2) / 110 → 3일 105 / 120 → 4일 100 / 105 후 매도(비용 1%)
    bought = 0.99
    peak = bought * 122 / 110
    sold = peak * 105 / 120 * 100 / 105 * 0.99
    assert trades["equity_from_peak"].iloc[0] == pytest.approx(sold / peak - 1)
    assert trades["from_peak"].iloc[0] == pytest.approx(100 / 120 - 1)  # 지수 기준은 배당과 비용이 없습니다.


def test_trade_list_matches_backtest_trade_count():
    data = make_data(3000, seed=7)
    holding = strategy_holding(data)
    start, end = data.index[300], data.index[-1]
    result = run_backtest(data, holding, start, end, Costs())
    trades = trade_list(holding.reindex(result.daily.index), data["close"])
    # 보유 구간마다 매수 한 번, 닫힌 구간마다 매도 한 번입니다. 기간 끝 청산은 매매 횟수에 들어가지 않습니다.
    assert len(trades) + (~trades["open"]).sum() == result.trades
    # 매도는 신호 낸 날(주 마지막 거래일)에만 일어납니다 (lag 0).
    weeks = period_last_days(data.index, "W-FRI")
    assert trades.loc[~trades["open"], "sell_date"].isin(weeks).all()
    assert (trades["from_peak"] <= 0).all()


def test_trade_list_rejects_missing_state():
    index = pd.bdate_range("2020-01-01", periods=3)
    with pytest.raises(ValueError):
        trade_list(pd.Series([0, np.nan, 1], index=index, dtype=float), pd.Series(1.0, index=index))


# ---------------------------------------------------------------- 신호 낸 날 종가 체결 (lag 0)


@pytest.mark.parametrize("cut", [400, 777, 1203])
def test_same_day_execution_does_not_use_data_after_t(cut):
    """신호 낸 날 종가 체결은 t의 종가로 t의 상태를 정합니다. t 뒤의 가격은 t까지의 상태와 수익률을 바꾸지 않아야 합니다."""
    data = make_data(1500, seed=1)
    changed = data.copy()
    rng = np.random.default_rng(2)
    changed.iloc[cut + 1 :, changed.columns.get_loc("close")] *= rng.uniform(0.5, 1.5, len(changed) - cut - 1)
    t = data.index[cut]

    a_hold, b_hold = strategy_holding(data), strategy_holding(changed)
    pd.testing.assert_series_equal(a_hold[:t], b_hold[:t])
    start = data.index[300]
    a_run = run_backtest(data, a_hold, start, data.index[-1], Costs()).daily
    b_run = run_backtest(changed, b_hold, start, data.index[-1], Costs()).daily
    pd.testing.assert_frame_equal(a_run[:t], b_run[:t])


def test_same_day_execution_changes_state_at_signal_close():
    data = make_data(1500, seed=3)
    weeks = period_last_days(data.index, "W-FRI")
    signal = moving_average_signal(data["close"], weeks, 20)
    holding = holding_from_signal(signal, data.index, lag=0)
    changes = signal[signal.notna() & signal.shift(1).notna() & (signal != signal.shift(1))]
    assert len(changes) > 0
    for day, value in changes.items():
        loc = data.index.get_loc(day)
        assert holding.iloc[loc] == value  # 신호 낸 날 종가에 체결
        assert holding.iloc[loc - 1] == 1.0 - value  # 그날 수익률은 체결 전 상태
