"""이동평균 + 기울기 필터 테스트: 상태 전이, 독립 구현 비교, 미래 정보 검사, 체결 시점, 무작위 보행 검사.
합성 ETF(synthetic_index_etf) 테스트도 여기에 둡니다."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Costs,
    holding_from_signal,
    metrics,
    moving_average_signal,
    moving_average_slope_signal,
    period_last_days,
    run_backtest,
    synthetic_index_etf,
)

NO_COSTS = Costs(trade=0.0)


def make_data(n: int, seed: int, drift=0.07, vol=0.16) -> pd.DataFrame:
    """추세 없는(일별 독립 수익률) 가짜 일봉."""
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift / 252, vol / np.sqrt(252), n))
    return pd.DataFrame({"close": close, "dividend": 0.0, "rf": 0.0}, index=pd.bdate_range("2000-01-03", periods=n))


def strategy_signal(data: pd.DataFrame, window=10, slope_lag=1) -> pd.Series:
    weeks = period_last_days(data.index, "W-FRI")
    return moving_average_slope_signal(data["close"], weeks, window, slope_lag)


def strategy_holding(data: pd.DataFrame) -> pd.Series:
    return holding_from_signal(strategy_signal(data), data.index, lag=1)


def run_signal(prices, window=2, slope_lag=1) -> np.ndarray:
    days = pd.bdate_range("2020-01-06", periods=len(prices), freq="W-FRI")
    price = pd.Series(np.asarray(prices, dtype=float), index=days)
    return moving_average_slope_signal(price, days, window, slope_lag).to_numpy()


# ---------------------------------------------------------------- 상태 전이
# 이동평균 2주: SMA[w] = (c[w-1] + c[w]) / 2, 기울기 = (c[w] - c[w-2]) / 2


def test_ready_week_starts_from_base_signal():
    # 기울기는 셋째 주에 처음 나옵니다. 그 주 종가 14 > 이동평균 13이라 보유로 시작합니다.
    signal = run_signal([10, 12, 14])
    assert np.isnan(signal[:2]).all()
    assert signal[2] == 1.0
    # 10 < 이동평균 11, 기울기 (10 - 10) / 2 = 0: 미보유로 시작합니다.
    assert run_signal([10, 12, 10])[2] == 0.0


def test_no_buy_while_average_is_falling():
    # 3주: 10 < 12.5 → 미보유로 시작
    # 4주: 12 > 11 이지만 기울기 (12 - 15) / 2 < 0 → 매수 금지
    # 5주: 16 > 14, 기울기 (16 - 10) / 2 > 0 → 매수
    signal = run_signal([20, 15, 10, 12, 16])
    assert list(signal[2:]) == [0.0, 0.0, 1.0]


def test_no_sell_while_average_is_rising():
    # 3주: 14 > 13 → 보유로 시작
    # 4주: 13 < 13.5 이지만 기울기 (13 - 12) / 2 > 0 → 매도 금지
    # 5주: 11 < 12, 기울기 (11 - 14) / 2 < 0 → 매도
    signal = run_signal([10, 12, 14, 13, 11])
    assert list(signal[2:]) == [1.0, 1.0, 0.0]


def test_flat_average_follows_base_signal():
    # 매도: 4주 12 < 13, 기울기 (12 - 12) / 2 = 0 → 막지 않음
    assert list(run_signal([10, 12, 14, 12])[2:]) == [1.0, 0.0]
    # 매수: 4주 10 > 9, 기울기 (10 - 10) / 2 = 0 → 막지 않음
    assert list(run_signal([12, 10, 8, 10])[2:]) == [0.0, 1.0]


def test_close_equal_to_average_is_not_above():
    # 이동평균 3주: SMA = [_, _, 9, 25/3, 10]
    # 4주: 8 < 8.33 → 미보유로 시작
    # 5주: 10 == (12 + 8 + 10) / 3이라 기본 신호가 0입니다. 기울기 > 0이어도 매수하지 않습니다.
    assert list(run_signal([10, 5, 12, 8, 10], window=3)[3:]) == [0.0, 0.0]


def test_slope_lag_compares_average_k_weeks_ago():
    # SMA(2) = [_, 20, 15, 7, 8]
    # 5주: 12 > 8, 지난주 대비 8 - 7 > 0 이지만 2주 전 대비 8 - 15 < 0
    prices = [20, 20, 10, 4, 12]
    lag1 = run_signal(prices, slope_lag=1)
    assert np.isnan(lag1[:2]).all()
    assert list(lag1[2:]) == [0.0, 0.0, 1.0]  # 간격 1주: 매수
    lag2 = run_signal(prices, slope_lag=2)
    assert np.isnan(lag2[:3]).all()  # 2주 전 이동평균이 있어야 하므로 한 주 늦게 시작합니다.
    assert list(lag2[3:]) == [0.0, 0.0]  # 간격 2주: 하락 중이라 매수 금지


def test_rejects_invalid_slope_lag():
    with pytest.raises(ValueError):
        run_signal([10, 11, 12], slope_lag=0)


def test_matches_independent_implementation_with_past_close():
    """이동평균 10주의 기울기 부호는 '종가 > 10주 전 종가'와 같습니다. 이 성질로 따로 구현해 비교합니다."""
    data = make_data(3000, seed=11)
    weeks = period_last_days(data.index, "W-FRI")
    close = data["close"].reindex(weeks).to_numpy()
    window = 10
    expected = np.full(len(close), np.nan)
    state = None
    for i in range(window, len(close)):
        average = close[i - window + 1 : i + 1].mean()
        above = close[i] > average
        rising, falling = close[i] > close[i - window], close[i] < close[i - window]
        if state is None:
            state = 1.0 if above else 0.0
        elif state == 0.0 and above and not falling:
            state = 1.0
        elif state == 1.0 and not above and not rising:
            state = 0.0
        expected[i] = state
    np.testing.assert_array_equal(strategy_signal(data).to_numpy(), expected)


def test_state_changes_only_when_base_signal_and_slope_agree():
    data = make_data(5000, seed=12)
    weeks = period_last_days(data.index, "W-FRI")
    signal = strategy_signal(data)
    base = moving_average_signal(data["close"], weeks, 10)
    average = data["close"].reindex(weeks).rolling(10).mean()
    slope = average - average.shift(1)

    previous = signal.shift(1)
    buys, sells = (previous == 0.0) & (signal == 1.0), (previous == 1.0) & (signal == 0.0)
    assert buys.sum() > 10 and sells.sum() > 10
    assert (base[buys] == 1.0).all() and (slope[buys] >= 0).all()
    assert (base[sells] == 0.0).all() and (slope[sells] <= 0).all()
    # 기본 신호와 다를 때는 반드시 기울기가 막은 경우입니다.
    differs = signal.notna() & (signal != base)
    assert differs.sum() > 0
    assert ((signal[differs] == 1.0) & (slope[differs] > 0) | (signal[differs] == 0.0) & (slope[differs] < 0)).all()


# ---------------------------------------------------------------- 미래 정보 검사


@pytest.mark.parametrize("cut", [333, 500, 777])
def test_signal_and_returns_do_not_change_when_future_data_changes(cut):
    data = make_data(1000, seed=1)
    changed = data.copy()
    rng = np.random.default_rng(2)
    changed.iloc[cut + 1 :, changed.columns.get_loc("close")] *= rng.uniform(0.5, 1.5, len(changed) - cut - 1)
    t = data.index[cut]

    a_signal, b_signal = strategy_signal(data), strategy_signal(changed)
    pd.testing.assert_series_equal(a_signal[a_signal.index <= t], b_signal[b_signal.index <= t])

    a_hold, b_hold = strategy_holding(data), strategy_holding(changed)
    pd.testing.assert_series_equal(a_hold[:t], b_hold[:t])

    start = data.index[300]
    a_run = run_backtest(data, a_hold, start, data.index[-1], Costs()).daily
    b_run = run_backtest(changed, b_hold, start, data.index[-1], Costs()).daily
    pd.testing.assert_frame_equal(a_run[:t], b_run[:t])


def test_signal_on_truncated_data_matches_full_data():
    # 데이터를 t에서 잘라 계산한 신호가 전체 데이터로 계산한 t까지의 신호와 같아야 합니다.
    data = make_data(1500, seed=4)
    full = strategy_signal(data)
    for cut in [400, 801, 1203]:
        part = strategy_signal(data.iloc[: cut + 1])
        closed = part.index[part.index < data.index[cut]]  # 잘린 마지막 주는 끝나지 않았을 수 있어 뺍니다.
        pd.testing.assert_series_equal(part[closed], full[closed])


def test_trades_execute_at_next_trading_day_close():
    data = make_data(1500, seed=3)
    signal = strategy_signal(data)
    changes = signal[signal.notna() & signal.shift(1).notna() & (signal != signal.shift(1))]
    assert len(changes) > 0
    holding = holding_from_signal(signal, data.index, lag=1)
    for day, value in changes.items():
        loc = data.index.get_loc(day)
        if loc + 1 < len(data):
            assert holding.iloc[loc] == 1.0 - value
            assert holding.iloc[loc + 1] == value


# ---------------------------------------------------------------- 무작위 보행 검사


def sharpe_gap(holding_fn, sims=40, years=20) -> np.ndarray:
    """추세 없는 가격에서 (전략 샤프 − 단순 보유 샤프)."""
    gaps = []
    for seed in range(sims):
        data = make_data(252 * years, seed=300 + seed)
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


# ---------------------------------------------------------------- 합성 ETF


def synthetic_input() -> pd.DataFrame:
    index = pd.DatetimeIndex(["2020-12-28", "2020-12-29", "2020-12-30", "2021-01-04", "2021-12-29", "2021-12-30"])
    return pd.DataFrame({"close": [100.0, 102.0, 101.0, 103.0, 110.0, 111.0], "rf": 0.0001}, index=index)


def test_synthetic_etf_deducts_expense_by_calendar_days():
    etf = synthetic_index_etf(synthetic_input(), 0.0, "december", 0.01, "cd")
    days = (etf.index - etf.index[0]).days.to_numpy()
    expected = synthetic_input()["close"].to_numpy() * 0.99 ** (days / 365.25)
    np.testing.assert_allclose(etf["close"].to_numpy(), expected)
    assert (etf["dividend"] == 0.0).all()


def test_synthetic_etf_pays_december_dividend_on_second_to_last_day():
    etf = synthetic_index_etf(synthetic_input(), 0.015, "december", 0.0, "cd")
    paid = etf["dividend"][etf["dividend"] > 0]
    # 2020년 12월: 28, 29, 30일 → 29일, 2021년 12월: 29, 30일 → 29일. 금액은 전날 종가 × 1.5%
    assert list(paid.index.strftime("%Y-%m-%d")) == ["2020-12-29", "2021-12-29"]
    assert paid.to_numpy() == pytest.approx([100.0 * 0.015, 103.0 * 0.015])


def test_synthetic_etf_even_dividend_and_zero_cash():
    etf = synthetic_index_etf(synthetic_input(), 0.0365, "even", 0.0, "zero")
    assert etf["dividend"].iloc[0] == 0.0
    # 2020-12-30 → 2021-01-04: 5일
    assert etf.loc["2021-01-04", "dividend"] == pytest.approx(0.0365 * 5 / 365.25 * 101.0)
    assert (etf["rf"] == 0.0).all()
    with pytest.raises(ValueError):
        synthetic_index_etf(synthetic_input(), 0.0, "december", 0.0, "unknown")
