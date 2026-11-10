"""이동평균 + ADX 진입 필터 테스트: ADX 계산, 상태 전이, 미래 정보 검사, 무작위 보행 검사."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Costs,
    holding_from_signal,
    metrics,
    moving_average_adx_signal,
    moving_average_signal,
    period_bars,
    period_last_days,
    run_backtest,
    wilder_adx,
)

NO_COSTS = Costs(trade=0.0)


def make_ohlc(n: int, seed: int, drift=0.07, vol=0.16) -> pd.DataFrame:
    """추세 없는(일별 독립 수익률) 가짜 일봉. 고가와 저가는 종가 주변에 무작위로 둡니다."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift / 252, vol / np.sqrt(252), n))
    index = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame(
        {
            "high": close * (1 + np.abs(rng.normal(0, 0.006, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.006, n))),
            "close": close,
            "dividend": 0.0,
            "rf": 0.0,
        },
        index=index,
    )


def strategy_signal(ohlc: pd.DataFrame, window=10, adx_period=14, threshold=25.0) -> pd.Series:
    weeks = period_last_days(ohlc.index, "W-FRI")
    adx = wilder_adx(period_bars(ohlc, "W-FRI"), adx_period)
    return moving_average_adx_signal(ohlc["close"], weeks, window, adx, threshold)


def strategy_holding(ohlc: pd.DataFrame) -> pd.Series:
    return holding_from_signal(strategy_signal(ohlc), ohlc.index, lag=1)


# ---------------------------------------------------------------- 주봉


def test_period_bars_use_week_high_low_and_last_close():
    # 2020-01-06(월) ~ 01-10(금), 01-13(월)
    index = pd.DatetimeIndex(["2020-01-06", "2020-01-07", "2020-01-10", "2020-01-13"])
    ohlc = pd.DataFrame({"high": [10.0, 12.0, 11.0, 20.0], "low": [9.0, 8.0, 10.0, 19.0], "close": [9.5, 11.0, 10.5, 19.5]}, index=index)
    bars = period_bars(ohlc, "W-FRI")
    assert list(bars.index.strftime("%Y-%m-%d")) == ["2020-01-10", "2020-01-13"]
    assert bars.loc["2020-01-10"].to_dict() == {"high": 12.0, "low": 8.0, "close": 10.5}
    assert bars.loc["2020-01-13"].to_dict() == {"high": 20.0, "low": 19.0, "close": 19.5}


# ---------------------------------------------------------------- ADX 계산


def test_wilder_adx_matches_hand_calculation():
    # 기간 2. +DM: 2,1,0,2,1 / −DM: 0,0,1,0,0
    # 평활(+,−): 봉2 (3, 0) / 봉3 (1.5, 1) / 봉4 (2.75, 0.5) / 봉5 (2.375, 0.25)
    # DX: 봉2 100 / 봉3 20 / 봉4 2.25/3.25 / 봉5 2.125/2.625
    bars = pd.DataFrame({"high": [10.0, 12, 13, 12, 14, 15], "low": [9.0, 10, 11, 10, 12, 13]}, index=pd.bdate_range("2020-01-06", periods=6))
    adx = wilder_adx(bars, period=2).to_numpy()
    assert np.isnan(adx[:3]).all()
    expected_3 = (100 + 20) / 2
    expected_4 = (expected_3 + 100 * 2.25 / 3.25) / 2
    expected_5 = (expected_4 + 100 * 2.125 / 2.625) / 2
    assert adx[3:] == pytest.approx([expected_3, expected_4, expected_5])


def test_wilder_adx_matches_definition_with_true_range():
    # TR을 약분하지 않고 교과서 정의대로 계산한 값과 같아야 합니다.
    bars = make_ohlc(200, seed=5)
    period = 14
    high, low, close = bars["high"].to_numpy(), bars["low"].to_numpy(), bars["close"].to_numpy()
    tr = [max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1])) for i in range(1, 200)]
    up = [high[i] - high[i - 1] for i in range(1, 200)]
    down = [low[i - 1] - low[i] for i in range(1, 200)]
    plus = [u if u > d and u > 0 else 0.0 for u, d in zip(up, down)]
    minus = [d if d > u and d > 0 else 0.0 for u, d in zip(up, down)]

    s_tr, s_plus, s_minus = sum(tr[:period]), sum(plus[:period]), sum(minus[:period])
    dxs = []
    for k in range(period - 1, len(tr)):
        if k >= period:
            s_tr += tr[k] - s_tr / period
            s_plus += plus[k] - s_plus / period
            s_minus += minus[k] - s_minus / period
        plus_di, minus_di = 100 * s_plus / s_tr, 100 * s_minus / s_tr
        dxs.append(100 * abs(plus_di - minus_di) / (plus_di + minus_di))
    expected = sum(dxs[:period]) / period
    for dx in dxs[period:]:
        expected = (expected * (period - 1) + dx) / period

    assert wilder_adx(bars, period).iloc[-1] == pytest.approx(expected)


def test_wilder_adx_is_near_100_in_steady_trend_and_direction_free():
    n = 60
    rising = pd.DataFrame({"high": np.arange(n) + 101.0, "low": np.arange(n) + 99.0}, index=pd.bdate_range("2020-01-06", periods=n))
    falling = pd.DataFrame({"high": 301.0 - np.arange(n), "low": 299.0 - np.arange(n)}, index=rising.index)
    assert wilder_adx(rising, 14).iloc[-1] == pytest.approx(100.0)
    assert wilder_adx(falling, 14).iloc[-1] == pytest.approx(100.0)


def test_wilder_adx_is_zero_without_movement():
    n = 40
    flat = pd.DataFrame({"high": [101.0] * n, "low": [99.0] * n}, index=pd.bdate_range("2020-01-06", periods=n))
    assert wilder_adx(flat, 14).iloc[-1] == 0.0


def test_wilder_adx_first_value_needs_two_periods():
    bars = make_ohlc(60, seed=6)
    adx = wilder_adx(bars, 14)
    assert adx.first_valid_index() == bars.index[27]
    assert wilder_adx(bars.iloc[:27], 14).isna().all()


# ---------------------------------------------------------------- 상태 전이


def run_signal(prices, adx_values, window=2, threshold=25.0):
    days = pd.bdate_range("2020-01-06", periods=len(prices), freq="W-FRI")
    price = pd.Series(np.asarray(prices, dtype=float), index=days)
    adx = pd.Series(np.asarray(adx_values, dtype=float), index=days)
    return moving_average_adx_signal(price, days, window, adx, threshold).to_numpy()


def test_no_entry_when_adx_below_threshold():
    # 계속 이동평균 위지만 ADX가 낮습니다.
    signal = run_signal([10, 11, 12, 13], [np.nan, 24.9, 24.9, 24.9])
    assert np.isnan(signal[0])
    assert list(signal[1:]) == [0.0, 0.0, 0.0]


def test_entry_at_threshold_and_when_adx_rises_later():
    # 25 미만이면 진입하지 않으므로 25 정확히는 진입합니다. 이동평균 위에 있던 중 ADX가 올라와도 진입합니다.
    signal = run_signal([10, 11, 12, 13], [np.nan, 10.0, 25.0, 25.0])
    assert list(signal[1:]) == [0.0, 1.0, 1.0]


def test_holds_when_adx_falls_after_entry():
    signal = run_signal([10, 11, 12, 13, 14], [np.nan, 30.0, 5.0, 5.0, 5.0])
    assert list(signal[1:]) == [1.0, 1.0, 1.0, 1.0]


def test_exits_below_average_regardless_of_adx():
    # 15 < (20 + 15) / 2. ADX가 높아도 팝니다.
    signal = run_signal([10, 20, 15, 16], [np.nan, 30.0, 99.0, 10.0])
    assert list(signal[1:3]) == [1.0, 0.0]
    # 16 > 15.5로 이동평균 위로 돌아왔지만 ADX가 낮아 다시 들어가지 않습니다.
    assert signal[3] == 0.0


def test_exit_when_close_equals_average():
    # 15 == (15 + 15) / 2: 이동평균보다 높지 않으면 보유하지 않습니다.
    signal = run_signal([10, 15, 15], [np.nan, 30.0, 30.0])
    assert list(signal[1:]) == [1.0, 0.0]


def test_rejects_hole_in_adx_after_ready():
    with pytest.raises(ValueError, match="비어 있는"):
        run_signal([10, 11, 12, 13], [np.nan, 30.0, np.nan, 30.0])


def test_adx_filter_with_threshold_zero_equals_plain_moving_average():
    # ADX 조건이 항상 참이면 이동평균 단독과 같아야 합니다.
    ohlc = make_ohlc(800, seed=7)
    weeks = period_last_days(ohlc.index, "W-FRI")
    plain = moving_average_signal(ohlc["close"], weeks, 10)
    filtered = strategy_signal(ohlc, threshold=0.0)
    ready = filtered.notna()
    pd.testing.assert_series_equal(filtered[ready], plain[ready], check_names=False)


# ---------------------------------------------------------------- 미래 정보 검사


@pytest.mark.parametrize("cut", [333, 500, 777])
def test_adx_signal_and_returns_do_not_change_when_future_data_changes(cut):
    data = make_ohlc(1000, seed=1)
    changed = data.copy()
    rng = np.random.default_rng(2)
    scale = rng.uniform(0.5, 1.5, len(changed) - cut - 1)
    for column in ["high", "low", "close"]:
        changed.iloc[cut + 1 :, changed.columns.get_loc(column)] *= scale
    t = data.index[cut]

    # 주가 끝나지 않은 동안의 주봉은 쓰지 않습니다. t가 속한 주의 마지막 거래일이 t 뒤이면 그 주 값은 비교에서 뺍니다.
    weeks = period_last_days(data.index, "W-FRI")
    closed_weeks = weeks[weeks <= t]

    a_bars, b_bars = period_bars(data, "W-FRI"), period_bars(changed, "W-FRI")
    a_adx, b_adx = wilder_adx(a_bars, 14), wilder_adx(b_bars, 14)
    pd.testing.assert_series_equal(a_adx[closed_weeks], b_adx[closed_weeks])

    a_signal, b_signal = strategy_signal(data), strategy_signal(changed)
    pd.testing.assert_series_equal(a_signal[closed_weeks], b_signal[closed_weeks])

    a_hold, b_hold = strategy_holding(data), strategy_holding(changed)
    pd.testing.assert_series_equal(a_hold[:t], b_hold[:t])

    start = data.index[300]
    a_run = run_backtest(data, a_hold, start, data.index[-1], Costs()).daily
    b_run = run_backtest(changed, b_hold, start, data.index[-1], Costs()).daily
    pd.testing.assert_frame_equal(a_run[:t], b_run[:t])


def test_entry_executes_at_next_trading_day_close():
    data = make_ohlc(600, seed=3)
    signal = strategy_signal(data)
    entries = signal[(signal == 1.0) & (signal.shift(1) == 0.0)]
    assert len(entries) > 0
    holding = holding_from_signal(signal, data.index, lag=1)
    for day in entries.index:
        loc = data.index.get_loc(day)
        if loc + 1 < len(data):
            assert holding.iloc[loc] == 0.0
            assert holding.iloc[loc + 1] == 1.0


# ---------------------------------------------------------------- 무작위 보행 검사


def sharpe_gap(holding_fn, sims=40, years=20) -> np.ndarray:
    """추세 없는 가격에서 (전략 샤프 − 단순 보유 샤프)."""
    gaps = []
    for seed in range(sims):
        data = make_ohlc(252 * years, seed=200 + seed)
        start, end = data.index[300], data.index[-1]
        strategy = metrics(run_backtest(data, holding_fn(data), start, end, NO_COSTS))
        hold = metrics(run_backtest(data, pd.Series(1.0, index=data.index), start, end, NO_COSTS))
        gaps.append(strategy["sharpe"] - hold["sharpe"])
    return np.array(gaps)


def test_random_walk_gives_no_edge():
    assert sharpe_gap(strategy_holding).mean() < 0.05


def test_random_walk_check_catches_lookahead():
    # 검사 자체가 미래 정보를 잡아내는지 확인: 5거래일 앞당긴 보유 상태(미래 정보)는 걸려야 합니다.
    def cheating(data):
        return strategy_holding(data).shift(-5).ffill()

    assert sharpe_gap(cheating, sims=15).mean() > 0.2
