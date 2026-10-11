"""백테스트 엔진 테스트: 미래 정보 검사, 체결 시점, 비용·세금 계산, 무작위 보행 검사."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Costs,
    holding_from_signal,
    metrics,
    moving_average_signal,
    period_last_days,
    run_backtest,
)

NO_COSTS = Costs(trade=0.0)


def make_data(prices, start="2020-01-01", dividend=None, rf=0.0) -> pd.DataFrame:
    index = pd.bdate_range(start, periods=len(prices))
    data = pd.DataFrame({"close": np.asarray(prices, dtype=float)}, index=index)
    data["dividend"] = 0.0 if dividend is None else dividend
    data["rf"] = rf
    return data


def random_walk(n: int, seed: int, drift=0.07, vol=0.16) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift / 252, vol / np.sqrt(252), n)
    return 100 * np.cumprod(1 + returns)


def strategy_holding(data: pd.DataFrame, window: int = 10, lag: int = 1) -> pd.Series:
    weeks = period_last_days(data.index, "W-FRI")
    return holding_from_signal(moving_average_signal(data["close"], weeks, window), data.index, lag)


# ---------------------------------------------------------------- 미래 정보 검사


@pytest.mark.parametrize("cut", [150, 333, 500])
def test_signal_and_returns_do_not_change_when_future_data_changes(cut):
    data = make_data(random_walk(700, seed=1))
    changed = data.copy()
    rng = np.random.default_rng(2)
    changed.iloc[cut + 1 :, 0] *= rng.uniform(0.5, 1.5, len(changed) - cut - 1)
    t = data.index[cut]

    weeks = period_last_days(data.index, "W-FRI")
    a = moving_average_signal(data["close"], weeks, 10)
    b = moving_average_signal(changed["close"], weeks, 10)
    pd.testing.assert_series_equal(a[a.index <= t], b[b.index <= t])

    ha, hb = strategy_holding(data), strategy_holding(changed)
    pd.testing.assert_series_equal(ha[:t], hb[:t])

    ra = run_backtest(data, ha, data.index[100], data.index[-1], Costs()).daily
    rb = run_backtest(changed, hb, data.index[100], data.index[-1], Costs()).daily
    # 마지막 행(청산·세금 정산)이 아닌 t까지는 같아야 합니다.
    pd.testing.assert_frame_equal(ra[:t], rb[:t])


def test_lag_cannot_be_negative():
    data = make_data([100.0] * 20)
    weeks = period_last_days(data.index, "W-FRI")
    with pytest.raises(ValueError):
        holding_from_signal(moving_average_signal(data["close"], weeks, 2), data.index, lag=-1)


# ---------------------------------------------------------------- 체결 시점


def test_signal_executes_at_next_trading_day_close():
    # 2020-01-03(금)에 신호가 나면 01-06(월) 종가에 체결합니다.
    index = pd.bdate_range("2019-12-30", "2020-01-10")
    signal = pd.Series([1.0], index=pd.DatetimeIndex(["2020-01-03"]))
    holding = holding_from_signal(signal, index, lag=1)
    assert np.isnan(holding["2020-01-03"])
    assert holding["2020-01-06"] == 1.0

    data = pd.DataFrame({"close": np.linspace(100, 109, len(index)), "dividend": 0.0, "rf": 0.0}, index=index)
    holding = holding.fillna(0.0)
    daily = run_backtest(data, holding, "2020-01-02", "2020-01-10", NO_COSTS).daily
    # 월요일(체결일) 수익률은 아직 현금, 화요일부터 주식 수익률
    assert daily.loc["2020-01-06", "holding"] == 0.0
    assert daily.loc["2020-01-06", "ret"] == 0.0
    assert daily.loc["2020-01-07", "holding"] == 1.0
    assert daily.loc["2020-01-07", "ret"] == pytest.approx(data.loc["2020-01-07", "close"] / data.loc["2020-01-06", "close"] - 1)


def test_period_last_days_handles_holiday_friday():
    # 2020-04-10은 성금요일 휴장: 그 주 마지막 거래일은 목요일
    index = pd.DatetimeIndex(["2020-04-06", "2020-04-07", "2020-04-08", "2020-04-09", "2020-04-13"])
    assert list(period_last_days(index, "W-FRI").strftime("%Y-%m-%d")) == ["2020-04-09", "2020-04-13"]


def test_moving_average_signal_includes_current_week():
    days = pd.bdate_range("2020-01-06", periods=3, freq="W-FRI")
    price = pd.Series([10.0, 20.0, 15.0], index=days)
    signal = moving_average_signal(price, days, 2)
    assert np.isnan(signal.iloc[0])
    assert signal.iloc[1] == 1.0  # 20 > 15
    assert signal.iloc[2] == 0.0  # 15 < 17.5


# ---------------------------------------------------------------- 비용, 배당, 세금


def test_buy_and_hold_costs_dividends_and_liquidation_tax():
    data = make_data([100.0, 110.0, 121.0], dividend=[0.0, 1.0, 0.0])
    holding = pd.Series(1.0, index=data.index)
    costs = Costs(trade=0.01, dividend_withholding=0.15, capital_gains_tax=0.22)
    result = run_backtest(data, holding, data.index[1], data.index[-1], costs)

    equity = 0.99  # 진입 비용
    equity *= (110 + 0.85) / 100
    equity *= 121 / 110
    assert result.daily["equity"].iloc[-1] == pytest.approx(equity)
    proceeds = equity * 0.99
    assert result.final_after_liquidation == pytest.approx(proceeds - 0.22 * (proceeds - 1.0))
    assert result.trades == 1


def test_realized_gains_are_taxed_at_year_end_with_same_year_losses_offset():
    index = pd.DatetimeIndex(["2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04"])
    data = pd.DataFrame({"close": [100.0, 120.0, 120.0, 90.0, 90.0], "dividend": 0.0, "rf": 0.0}, index=index)
    # 12-28 매수, 12-29 매도(+20%), 12-30 매수, 12-31 매도(-25%)
    holding = pd.Series([1.0, 0.0, 1.0, 0.0, 0.0], index=index)
    result = run_backtest(data, holding, "2020-12-29", "2021-01-04", Costs(trade=0.0, capital_gains_tax=0.22))
    gain1 = 1.2 - 1.0
    value = 1.2 * 0.75
    gain2 = value - 1.2
    expected_tax = 0.22 * max(0.0, gain1 + gain2)  # 같은 해 손실과 상계하면 순손실 → 세금 0
    assert result.daily.loc["2020-12-31", "tax"] == pytest.approx(expected_tax)
    assert result.daily["equity"].iloc[-1] == pytest.approx(value - expected_tax)
    assert result.trades == 4


def test_cash_earns_rf():
    data = make_data([100.0, 50.0, 25.0], rf=0.001)
    holding = pd.Series(0.0, index=data.index)
    result = run_backtest(data, holding, data.index[1], data.index[-1], NO_COSTS)
    assert result.daily["equity"].iloc[-1] == pytest.approx(1.001**2)
    assert result.trades == 0


def test_rejects_missing_holding_in_evaluation_window():
    data = make_data(random_walk(100, seed=3))
    holding = strategy_holding(data)  # 준비 기간은 NaN
    with pytest.raises(ValueError, match="보유 상태"):
        run_backtest(data, holding, data.index[1], data.index[-1], NO_COSTS)


def test_metrics_drawdown_and_recovery():
    data = make_data([100.0, 110.0, 55.0, 80.0, 110.0, 120.0])
    holding = pd.Series(1.0, index=data.index)
    m = metrics(run_backtest(data, holding, data.index[1], data.index[-1], NO_COSTS))
    assert m["mdd"] == pytest.approx(-0.5)
    assert m["mdd_peak"] == data.index[1]
    assert m["mdd_trough"] == data.index[2]
    assert m["mdd_recovery_months"] == pytest.approx((data.index[4] - data.index[1]).days / 30.4375)


# ---------------------------------------------------------------- 무작위 보행 검사


def sharpe_gap(holding_fn, sims=60, years=20) -> np.ndarray:
    """추세 없는 가격에서 (전략 샤프 − 단순 보유 샤프)."""
    gaps = []
    for seed in range(sims):
        data = make_data(random_walk(252 * years, seed=100 + seed))
        start, end = data.index[60], data.index[-1]
        strategy = metrics(run_backtest(data, holding_fn(data), start, end, NO_COSTS))
        hold = metrics(run_backtest(data, pd.Series(1.0, index=data.index), start, end, NO_COSTS))
        gaps.append(strategy["sharpe"] - hold["sharpe"])
    return np.array(gaps)


def test_random_walk_gives_no_edge():
    gaps = sharpe_gap(strategy_holding)
    # 추세가 없으면 이동평균 전략이 단순 보유보다 샤프가 뚜렷이 높을 이유가 없습니다.
    assert gaps.mean() < 0.05


def test_random_walk_check_catches_lookahead():
    # 검사 자체가 미래 정보를 잡아내는지 확인: 5거래일 앞당긴 보유 상태(미래 정보)는 걸려야 합니다.
    def cheating(data):
        return strategy_holding(data).shift(-5).ffill()

    assert sharpe_gap(cheating, sims=20).mean() > 0.3
