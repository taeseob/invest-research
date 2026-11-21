"""롱/숏(인버스) 테스트: 합성 인버스, 매매 비용·보수·세금 손계산, 미래 정보 검사."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import (
    Costs,
    holding_from_signal,
    long_short,
    metrics,
    moving_average_signal,
    period_last_days,
    run_backtest,
    synthetic_inverse,
)

DAY = 1 / 365.25


def make_data(n: int, seed: int, drift=0.07, vol=0.18) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 * np.cumprod(1 + rng.normal(drift / 252, vol / np.sqrt(252), n))
    dividend = np.where(np.arange(n) % 63 == 62, close * 0.005, 0.0)
    return pd.DataFrame({"close": close, "dividend": dividend, "rf": 0.0001}, index=pd.bdate_range("2000-01-03", periods=n))


# ---------------------------------------------------------------- 합성 인버스


def test_synthetic_inverse_by_hand():
    index = pd.DatetimeIndex(["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-06"])  # 마지막 날은 주말을 건너 3일
    underlying = pd.DataFrame({"close": [100, 102, 102, 99.0], "dividend": [0, 0, 1.0, 0]}, index=index)
    rf = pd.Series([0.0, 0.001, 0.001, 0.002], index=index)
    inverse = synthetic_inverse(underlying, rf, expense=0.01, borrow=0.005, interest_multiple=2.0)

    fee = 0.985  # 1 - 보수 - 차입 비용
    nav1 = (1 - 0.02 + 0.002) * fee**DAY
    nav2 = nav1 * (1 - (103 / 102 - 1) + 0.002) * fee**DAY  # 배당락일: 숏은 배당을 물어 줍니다
    nav3 = nav2 * (1 - (99 / 102 - 1) + 0.004) * fee ** (3 * DAY)
    assert inverse["close"].tolist() == pytest.approx([1.0, nav1, nav2, nav3], rel=1e-12)

    # 과표기준가: 담보 이자(1배) − 보수만 쌓입니다. 차입 비용과 지수 손익은 빠집니다.
    base1 = 1 + 1.0 * (0.001 - 0.01 * DAY)
    base2 = base1 + nav1 * (0.001 - 0.01 * DAY)
    base3 = base2 + nav2 * (0.002 - 0.01 * 3 * DAY)
    assert inverse["tax_base"].tolist() == pytest.approx([1.0, base1, base2, base3], rel=1e-12)


def test_synthetic_inverse_interest_multiple():
    index = pd.bdate_range("2020-01-06", periods=3)
    underlying = pd.DataFrame({"close": [100.0, 100.0, 100.0], "dividend": 0.0}, index=index)
    rf = pd.Series([0.0, 0.001, 0.001], index=index)
    for m in (0.0, 1.0, 2.0):
        inverse = synthetic_inverse(underlying, rf, expense=0.0, interest_multiple=m)
        assert inverse["close"].iloc[-1] == pytest.approx((1 + m * 0.001) ** 2)
        # 이자가 0배면 과표기준가도 늘지 않습니다.
        assert inverse["tax_base"].iloc[1] == pytest.approx(1 + min(m, 1.0) * 0.001)


def test_synthetic_inverse_has_no_lookahead():
    data = make_data(400, seed=3)
    base = synthetic_inverse(data[["close", "dividend"]], data["rf"], 0.009, 0.003)
    rng = np.random.default_rng(4)
    for t in rng.integers(10, 390, 20):
        changed = data.copy()
        changed.iloc[t + 1 :, changed.columns.get_loc("close")] *= rng.uniform(0.5, 1.5, len(data) - t - 1)
        changed.iloc[t + 1 :, changed.columns.get_loc("rf")] = 0.01
        other = synthetic_inverse(changed[["close", "dividend"]], changed["rf"], 0.009, 0.003)
        pd.testing.assert_frame_equal(base.iloc[: t + 1], other.iloc[: t + 1])


# ---------------------------------------------------------------- 매매 비용


def test_switch_long_to_short_pays_two_one_way_costs():
    index = pd.bdate_range("2020-01-06", periods=5)
    data = pd.DataFrame({"close": [100, 110, 99, 99, 108.9], "dividend": 0.0, "rf": 0.0}, index=index)
    inverse = pd.DataFrame({"close": [1.0, 0.9, 1.0, 1.0, 0.9]}, index=index)
    holding = pd.Series([1, 1, -1, -1, 1], index=index, dtype=float)
    result = run_backtest(data, holding, index[1], index[-1], Costs(trade=0.01), inverse)

    e0 = 0.99  # 시작 전날 주식 매수
    e1 = e0 * 1.1
    e2 = e1 * 99 / 110 * 0.99 * 0.99  # 주식 매도 + 인버스 매수
    e3 = e2  # 인버스 1.0 → 1.0
    e4 = e3 * 0.9 * 0.99 * 0.99  # 인버스 매도 + 주식 매수
    assert result.daily["equity"].tolist() == pytest.approx([e0, e1, e2, e3, e4], rel=1e-12)
    assert result.trades == 5
    assert result.daily["holding"].tolist() == [0.0, 1.0, 1.0, -1.0, -1.0]
    assert result.daily["trade"].tolist() == [True, False, True, False, True]
    assert result.final_after_liquidation == pytest.approx(e4 * 0.99)


def test_short_to_cash_and_cash_to_short_pay_one_cost_each():
    index = pd.bdate_range("2020-01-06", periods=4)
    data = pd.DataFrame({"close": 100.0, "dividend": 0.0, "rf": 0.001}, index=index)
    inverse = pd.DataFrame({"close": [1.0, 1.0, 1.0, 1.0]}, index=index)
    holding = pd.Series([-1, 0, -1, -1], index=index, dtype=float)
    result = run_backtest(data, holding, index[1], index[-1], Costs(trade=0.01), inverse)
    e0 = 0.99
    e1 = e0 * 0.99
    e2 = e1 * 1.001 * 0.99
    assert result.daily["equity"].tolist() == pytest.approx([e0, e1, e2, e2], rel=1e-12)
    assert result.trades == 3


# ---------------------------------------------------------------- 세금


def test_holding_period_tax_on_liquidation_uses_smaller_of_gain_and_tax_base():
    index = pd.bdate_range("2020-01-06", periods=3)
    data = pd.DataFrame({"close": 100.0, "dividend": 0.0, "rf": 0.0}, index=index)
    inverse = pd.DataFrame({"close": [1.0, 1.1, 1.2], "tax_base": [1.0, 1.01, 1.03]}, index=index)
    holding = pd.Series(-1.0, index=index)
    costs = Costs(trade=0.01, inverse_tax="holding_period", inverse_gain_tax=0.154)
    result = run_backtest(data, holding, index[1], index[-1], costs, inverse)
    # 투자 금액 0.99, 1주 가격 1.0. 매매차익 0.2와 과표 증가 0.03 중 작은 0.03에 과세
    assert result.daily["equity"].iloc[-1] == pytest.approx(0.99 * 1.2)
    assert result.final_after_liquidation == pytest.approx(0.99 * 1.2 * 0.99 - 0.154 * 0.03 * 0.99)


def test_holding_period_tax_ignores_losses_and_does_not_offset():
    index = pd.bdate_range("2020-01-06", periods=6)
    data = pd.DataFrame({"close": 100.0, "dividend": 0.0, "rf": 0.0}, index=index)
    inverse = pd.DataFrame(
        {"close": [1.0, 0.8, 0.8, 0.8, 1.0, 1.0], "tax_base": [1.0, 1.01, 1.01, 1.01, 1.02, 1.02]}, index=index
    )
    holding = pd.Series([-1, 1, 1, -1, -1, 0], index=index, dtype=float)
    costs = Costs(trade=0.0, inverse_tax="holding_period", inverse_gain_tax=0.154)
    result = run_backtest(data, holding, index[1], index[-1], costs, inverse)
    # 첫 인버스는 손실(-0.2)이라 세금 0. 두 번째는 0.8에 사서 1.0에 팔고, 과표 증가 0.01이 더 작습니다.
    # 첫 손실과 상계하지 않습니다.
    tax = 0.154 * 0.01 / 0.8 * 0.8
    assert result.daily["tax"].tolist() == pytest.approx([0, 0, 0, 0, 0, tax])
    assert result.final_after_liquidation == pytest.approx(1.0 - tax)


def test_pooled_tax_nets_stock_gain_and_inverse_loss_in_same_year():
    index = pd.DatetimeIndex(["2020-12-24", "2020-12-28", "2020-12-29", "2020-12-30", "2020-12-31", "2021-01-04"])
    data = pd.DataFrame({"close": [100, 120, 120, 120, 120, 120.0], "dividend": 0.0, "rf": 0.0}, index=index)
    inverse = pd.DataFrame({"close": [1, 1, 1, 0.9, 0.9, 0.9]}, index=index)
    holding = pd.Series([1, 1, -1, -1, 0, 0], index=index, dtype=float)
    result = run_backtest(data, holding, index[1], index[-1], Costs(trade=0.0, capital_gains_tax=0.22), inverse)
    # 주식 이익 0.2, 인버스 손실 1.2 × 0.1 = 0.12 → 순이익 0.08에 22%
    assert result.daily["tax"].iloc[4] == pytest.approx(0.22 * 0.08)
    assert result.final_after_liquidation == pytest.approx(1.08 - 0.22 * 0.08)


def test_inverse_with_same_prices_as_stock_gives_same_result_under_pooled_tax():
    """인버스 가격이 주식 가격(배당 없음)과 같으면, 인버스 보유와 주식 보유의 결과가 같아야 합니다.
    연말 세금을 내려고 일부 파는 처리와 기간 끝 정산까지 같은 경로를 타는지 봅니다."""
    data = make_data(800, seed=11)
    data["dividend"] = 0.0
    inverse = data[["close"]]
    costs = Costs(trade=0.002, capital_gains_tax=0.22)
    weeks = period_last_days(data.index, "W-FRI")
    signal = holding_from_signal(moving_average_signal(data["close"], weeks, 10), data.index, 1)
    start, end = data.index[60], data.index[-1]
    as_stock = run_backtest(data, signal, start, end, costs)
    as_inverse = run_backtest(data, -signal, start, end, costs, inverse)
    np.testing.assert_allclose(as_inverse.daily["equity"], as_stock.daily["equity"], rtol=1e-12)
    np.testing.assert_allclose(as_inverse.daily["tax"], as_stock.daily["tax"], rtol=1e-12, atol=1e-15)
    assert as_inverse.final_after_liquidation == pytest.approx(as_stock.final_after_liquidation, rel=1e-12)
    assert as_inverse.trades == as_stock.trades


def test_inputs_are_checked():
    index = pd.bdate_range("2020-01-06", periods=3)
    data = pd.DataFrame({"close": 100.0, "dividend": 0.0, "rf": 0.0}, index=index)
    holding = pd.Series(-1.0, index=index)
    with pytest.raises(ValueError, match="인버스 데이터가 없습니다"):
        run_backtest(data, holding, index[1], index[-1], Costs())
    with pytest.raises(ValueError, match="tax_base"):
        run_backtest(data, holding, index[1], index[-1], Costs(inverse_tax="holding_period"), data[["close"]])
    with pytest.raises(ValueError, match="함께 쓰는"):
        run_backtest(data, holding, index[1], index[-1], Costs(capital_gains_tax=0.22, inverse_tax="holding_period"))
    with pytest.raises(ValueError, match="1/0/-1"):
        run_backtest(data, holding * 0.5, index[1], index[-1], Costs(), data[["close"]])


# ---------------------------------------------------------------- 신호와 미래 정보


def test_long_short_maps_signal():
    signal = pd.Series([1.0, 0.0, np.nan])
    assert long_short(signal).tolist()[:2] == [1.0, -1.0]
    assert np.isnan(long_short(signal).iloc[2])


def run_strategy(data: pd.DataFrame, lag: int, window: int = 20):
    inverse = synthetic_inverse(data[["close", "dividend"]], data["rf"], 0.009, 0.003)
    weeks = period_last_days(data.index, "W-FRI")
    holding = holding_from_signal(long_short(moving_average_signal(data["close"], weeks, window)), data.index, lag)
    costs = Costs(trade=0.001, inverse_tax="holding_period", inverse_gain_tax=0.154)
    return holding, run_backtest(data, holding, data.index[150], data.index[-1], costs, inverse)


@pytest.mark.parametrize("lag", [0, 1])
def test_long_short_has_no_lookahead(lag):
    """t일 뒤의 가격과 금리를 바꿔도 t일까지의 보유 상태와 자산 가치가 그대로여야 합니다."""
    data = make_data(700, seed=21)
    holding, result = run_strategy(data, lag)
    assert (result.daily["holding"] == -1.0).any() and (result.daily["holding"] == 1.0).any()
    rng = np.random.default_rng(22)
    for t in rng.integers(160, 690, 15):
        changed = data.copy()
        changed.iloc[t + 1 :, changed.columns.get_loc("close")] *= rng.uniform(0.6, 1.4, len(data) - t - 1)
        changed.iloc[t + 1 :, changed.columns.get_loc("rf")] = 0.002
        other_holding, other = run_strategy(changed, lag)
        day = data.index[t]
        pd.testing.assert_series_equal(holding.loc[:day], other_holding.loc[:day])
        pd.testing.assert_series_equal(result.daily["equity"].loc[:day], other.daily["equity"].loc[:day])


def test_metrics_split_long_and_short_time():
    data = make_data(700, seed=31)
    _, result = run_strategy(data, 1)
    m = metrics(result)
    held = result.daily["holding"].iloc[1:]
    assert m["time_in_market"] == pytest.approx((held == 1).mean())
    assert m["time_short"] == pytest.approx((held == -1).mean())
    assert m["time_in_market"] + m["time_short"] == pytest.approx(1.0)
