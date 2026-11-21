"""월 단위 계산(Faber 논문 방식) 테스트: 미래 정보 검사, 체결 시점, 일별 엔진과의 일치, 지표, 무작위 보행 검사."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import Costs, holding_from_signal, period_last_days, run_backtest
from src.data import daily_dividend, daily_rf
from src.monthly import (
    month_end_returns,
    month_end_signal,
    monthly_metrics,
    monthly_series_signal,
    switch_months,
    timing_returns,
    total_return_signal_index,
)


def daily_prices(n: int, seed: int, start="2000-01-03", drift=0.07, vol=0.16) -> pd.Series:
    rng = np.random.default_rng(seed)
    returns = rng.normal(drift / 252, vol / np.sqrt(252), n)
    index = pd.bdate_range(start, periods=n)
    return pd.Series(100 * np.cumprod(1 + returns), index=index)


def month_end_table(price: pd.Series) -> pd.DataFrame:
    month_end = price.groupby(price.index.to_period("M")).last()
    return pd.DataFrame({"close": month_end, "total_return": month_end / month_end.shift(1) - 1})


# ---------------------------------------------------------------- 미래 정보 검사


@pytest.mark.parametrize("cut", [300, 777, 1500])
def test_signal_and_returns_do_not_change_when_future_data_changes(cut):
    price = daily_prices(2600, seed=3)
    changed = price.copy()
    changed.iloc[cut + 1 :] *= np.random.default_rng(4).uniform(0.5, 1.5, len(price) - cut - 1)
    t = price.index[cut].to_period("M")

    a, b = month_end_signal(price, 10), month_end_signal(changed, 10)
    # t월 말 종가가 아직 나오지 않았다면 t월 신호는 바뀔 수 있습니다. t월 전까지는 같아야 합니다.
    pd.testing.assert_series_equal(a[a.index < t], b[b.index < t])

    rf = pd.Series(0.003, index=a.index)
    ra = timing_returns(a, month_end_table(price)["total_return"], rf)
    rb = timing_returns(b, month_end_table(changed)["total_return"], rf)
    # t-1월까지의 수익률은 t-2월 신호와 t-1월 말 가격까지만 씁니다.
    pd.testing.assert_frame_equal(ra[ra.index < t], rb[rb.index < t])


def test_monthly_series_signal_does_not_look_ahead():
    months = pd.period_range("2000-01", periods=60, freq="M")
    series = pd.Series(np.random.default_rng(5).uniform(90, 110, 60), index=months)
    changed = series.copy()
    changed.iloc[31:] *= 3
    a, b = monthly_series_signal(series, 10), monthly_series_signal(changed, 10)
    pd.testing.assert_series_equal(a.iloc[:31], b.iloc[:31])


# ---------------------------------------------------------------- 신호와 체결 시점


def test_signal_uses_month_end_close_and_window_including_that_month():
    # 월말 종가 1, 2, 3, 0.5. 3개월 이동평균: 3월 2 → 3 > 2 보유, 4월 (2+3+0.5)/3 → 0.5 < 1.83 현금
    index = pd.DatetimeIndex(["2020-01-30", "2020-01-31", "2020-02-28", "2020-03-31", "2020-04-15", "2020-04-30"])
    price = pd.Series([9.0, 1.0, 2.0, 3.0, 9.0, 0.5], index=index)
    signal = month_end_signal(price, 3)
    assert signal.index.equals(pd.period_range("2020-01", "2020-04", freq="M"))
    assert np.isnan(signal["2020-01"]) and np.isnan(signal["2020-02"])
    assert signal["2020-03"] == 1.0 and signal["2020-04"] == 0.0


def test_equal_price_and_average_is_not_a_buy():
    months = pd.period_range("2000-01", periods=4, freq="M")
    signal = monthly_series_signal(pd.Series([5.0, 5.0, 5.0, 5.0], index=months), 3)
    assert signal.dropna().eq(0.0).all()


def test_return_of_month_uses_previous_month_signal():
    months = pd.period_range("2000-01", periods=4, freq="M")
    signal = pd.Series([1.0, 0.0, 1.0, 1.0], index=months)
    asset = pd.Series([0.10, 0.20, -0.30, 0.05], index=months)
    rf = pd.Series(0.01, index=months)
    out = timing_returns(signal, asset, rf)
    assert np.isnan(out["ret"].iloc[0])
    # 2월: 1월 신호 1 → 자산, 3월: 2월 신호 0 → 현금, 4월: 3월 신호 1 → 자산
    assert out["ret"].iloc[1:].tolist() == pytest.approx([0.20, 0.01, 0.05])


def test_missing_month_is_rejected():
    months = pd.PeriodIndex(["2000-01", "2000-02", "2000-04"], freq="M")
    s = pd.Series([1.0, 1.0, 1.0], index=months)
    with pytest.raises(ValueError):
        timing_returns(s, s, s)


def test_switch_months_lists_buy_and_sell():
    months = pd.period_range("2000-01", periods=5, freq="M")
    out = switch_months(pd.Series([np.nan, 1.0, 0.0, 0.0, 1.0], index=months))
    assert out["month"].astype(str).tolist() == ["2000-03", "2000-05"]
    assert out["action"].tolist() == ["매도", "매수"]


# ---------------------------------------------------------------- 일별 엔진과의 일치


def test_monthly_method_matches_daily_engine_at_month_end_close():
    """배당이 없으면 논문 방식(월 계산)과 일별 엔진(신호 낸 날 종가 체결, 비용 없음)의 월 수익률이 같아야 합니다.

    일별 현금 수익률은 월간 RF의 기하 분할이라, 한 달을 곱하면 월간 RF가 정확히 나옵니다.
    """
    price = daily_prices(2000, seed=7)
    months = price.index.to_period("M")
    rf_monthly = pd.Series(0.004, index=pd.period_range(months[0], months[-1], freq="M"))
    data = pd.DataFrame({"close": price, "dividend": 0.0, "rf": daily_rf(price.index, rf_monthly, months[-1] + 1)})

    signal = month_end_signal(price, 10)
    month_days = period_last_days(price.index, "M")
    daily_signal = pd.Series(signal.to_numpy(), index=month_days)
    holding = holding_from_signal(daily_signal, price.index, lag=0)
    start, end = "2001-01-01", str(price.index[-1].date())
    result = run_backtest(data, holding, start, end, Costs(trade=0.0))
    daily_based = month_end_returns(result.daily["equity"], result.start_date)

    table = month_end_table(price)
    monthly = timing_returns(signal, table["total_return"], rf_monthly)["ret"]
    monthly = monthly[(monthly.index >= pd.Period("2001-01", "M")) & (monthly.index <= months[-1])]
    # 마지막 달이 끝나지 않았으면 일별 쪽 마지막 값은 그 달 중간 값이라 뺍니다.
    pd.testing.assert_series_equal(daily_based.iloc[:-1], monthly.iloc[:-1], check_names=False, rtol=1e-10)


def test_monthly_method_matches_daily_engine_exactly_when_dividend_is_paid_at_month_end():
    """배당 D/12를 그달 마지막 거래일에 한 번에 받으면, 일별 엔진의 월 수익률이 (close[t] + D/12) / close[t-1] − 1과 정확히 같습니다.

    실제 실험은 배당을 그달 거래일에 나눠 받으므로 아주 조금 다릅니다. 그 차이가 배당 분할 방식 하나뿐이라는 확인입니다.
    """
    price = daily_prices(2000, seed=8)
    months = price.index.to_period("M")
    month_list = pd.period_range(months[0], months[-1], freq="M")
    annual = pd.Series(np.random.default_rng(9).uniform(2.0, 4.0, len(month_list)), index=month_list)
    rf_monthly = pd.Series(0.003, index=month_list)
    month_days = period_last_days(price.index, "M")
    dividend = pd.Series(0.0, index=price.index)
    dividend[month_days] = annual.reindex(month_days.to_period("M")).to_numpy() / 12
    data = pd.DataFrame({"close": price, "dividend": dividend, "rf": daily_rf(price.index, rf_monthly, months[-1] + 1)})

    signal = month_end_signal(price, 10)
    holding = holding_from_signal(pd.Series(signal.to_numpy(), index=month_days), price.index, lag=0)
    result = run_backtest(data, holding, "2001-01-01", str(month_days[-2].date()), Costs(trade=0.0))
    daily_based = month_end_returns(result.daily["equity"], result.start_date)

    month_end = price.reindex(month_days).to_numpy()
    total_return = pd.Series((month_end + annual.to_numpy() / 12) / np.roll(month_end, 1) - 1, index=month_list)
    monthly = timing_returns(signal, total_return, rf_monthly)["ret"].loc[daily_based.index]
    pd.testing.assert_series_equal(daily_based, monthly, check_names=False, rtol=1e-12)


def test_total_return_signal_index_by_hand():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.period_range("2000-01", periods=3, freq="M"))
    annual = pd.Series([12.0, 24.0, 36.0], index=pd.period_range("1999-01", periods=3, freq="M"))
    out = total_return_signal_index(close, annual, 12)
    # 2000-02: (110 + 24/12) / 100, 2000-03: (99 + 36/12) / 110. 12개월 전 D를 씁니다.
    assert out.tolist() == pytest.approx([1.0, 1.12, 1.12 * 102 / 110])


def test_total_return_signal_index_ignores_recent_dividends_and_future_data():
    """t까지의 지수는 t까지의 가격과 t-12까지의 D만 씁니다. 그 뒤 값을 바꿔도 그대로여야 합니다."""
    months = pd.period_range("2000-01", periods=80, freq="M")
    rng = np.random.default_rng(12)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.005, 0.04, 80)), index=months)
    annual = pd.Series(rng.uniform(2, 4, 92), index=pd.period_range("1999-01", periods=92, freq="M"))
    base = total_return_signal_index(close, annual, 12)
    for cut in [20, 45, 70]:
        t = months[cut]
        changed_close, changed_annual = close.copy(), annual.copy()
        changed_close[changed_close.index > t] *= 1.7
        # 보간된 D에는 앞으로의 배당이 섞여 있으므로, 최근 11개월의 D도 바꿔 봅니다.
        changed_annual[changed_annual.index > t - 12] *= 3.0
        out = total_return_signal_index(changed_close, changed_annual, 12)
        pd.testing.assert_series_equal(base[:t], out[:t])
        a, b = monthly_series_signal(base, 10), monthly_series_signal(out, 10)
        pd.testing.assert_series_equal(a[:t], b[:t])


def test_total_return_signal_index_rejects_missing_dividend():
    close = pd.Series([100.0, 110.0, 99.0], index=pd.period_range("2000-01", periods=3, freq="M"))
    annual = pd.Series([12.0, np.nan, 36.0], index=pd.period_range("1999-01", periods=3, freq="M"))
    with pytest.raises(ValueError, match="배당이 빈 달"):
        total_return_signal_index(close, annual, 12)


def test_total_return_signal_index_rejects_zero_lag_even_when_dividends_exist():
    """배당이 모든 달에 있어도 lag 0은 거부해야 합니다 (결측 검사가 대신 막는 것이 아니라 가드가 막는지)."""
    close = pd.Series([100.0, 110.0, 99.0], index=pd.period_range("2000-01", periods=3, freq="M"))
    annual = pd.Series(np.arange(1.0, 16.0), index=pd.period_range("1999-01", periods=15, freq="M"))
    with pytest.raises(ValueError, match="dividend_lag"):
        total_return_signal_index(close, annual, 0)


def test_daily_dividend_adds_up_to_monthly_dividend():
    index = pd.bdate_range("2020-01-01", "2020-03-31")
    annual = pd.Series([12.0, 24.0, 36.0], index=pd.period_range("2020-01", periods=3, freq="M"))
    dividend = daily_dividend(index, annual)
    total = dividend.groupby(index.to_period("M")).sum()
    assert total.tolist() == pytest.approx([1.0, 2.0, 3.0])


def test_month_end_returns_include_entry_cost_in_first_month():
    index = pd.bdate_range("2020-01-30", "2020-03-31")
    equity = pd.Series(0.999, index=index)  # 진입 비용 0.1%를 낸 뒤 값이 그대로
    out = month_end_returns(equity, index[0])
    assert out.index.astype(str).tolist() == ["2020-01", "2020-02", "2020-03"]
    assert out.iloc[0] == pytest.approx(-0.001)
    assert out.iloc[1:].tolist() == pytest.approx([0.0, 0.0])


# ---------------------------------------------------------------- 지표


def test_monthly_metrics_by_hand():
    months = pd.period_range("2000-01", periods=12, freq="M")
    ret = pd.Series([0.10, -0.20, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.10, 0.10], index=months)
    rf = pd.Series(0.0, index=months)
    held = pd.Series([1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], index=months, dtype=float)
    m = monthly_metrics(ret, rf, held)

    wealth = np.cumprod(1 + ret.to_numpy())
    assert m["cagr"] == pytest.approx(wealth[-1] - 1)  # 1년
    assert m["volatility"] == pytest.approx(np.std(ret, ddof=1) * np.sqrt(12))
    assert m["sharpe_paper"] == pytest.approx(m["cagr"] / m["volatility"])
    assert m["sharpe"] == pytest.approx(ret.mean() * 12 / (ret.std() * np.sqrt(12)))
    # 1월 말 1.1이 고점, 2월 말 0.88이 바닥 → -20%. 12월 말 0.88 × 1.05 × 1.21 = 1.118 ≥ 1.1 → 고점에서 11개월
    assert m["mdd"] == pytest.approx(-0.20)
    assert str(m["mdd_peak"]) == "2000-01" and str(m["mdd_trough"]) == "2000-02"
    assert m["mdd_recovery_months"] == 11
    assert m["positive_months"] == pytest.approx(4 / 12)
    assert m["time_in_market"] == pytest.approx(4 / 12)
    assert m["trades_per_year"] == pytest.approx(3)  # 처음 진입, 3월 매도, 11월 매수


def test_monthly_metrics_drawdown_from_start_and_no_recovery():
    months = pd.period_range("2000-01", periods=3, freq="M")
    ret = pd.Series([-0.10, -0.10, 0.05], index=months)
    m = monthly_metrics(ret, pd.Series(0.0, index=months))
    assert m["mdd"] == pytest.approx(0.81 - 1)
    assert str(m["mdd_peak"]) == "1999-12"
    assert np.isnan(m["mdd_recovery_months"])


def test_paper_sharpe_definition_matches_appendix_b():
    """원문 부록 B 1900년대 단순 보유: (9.93% − 4.76%) / 12.65% = 0.41. 정의만 확인합니다."""
    assert (0.0993 - 0.0476) / 0.1265 == pytest.approx(0.41, abs=0.005)


# ---------------------------------------------------------------- 무작위 보행 검사


def test_random_walk_strategy_is_not_clearly_better_than_buy_and_hold():
    """추세가 없는 가격에서 전략의 샤프가 단순 보유보다 뚜렷이 높으면 엔진이나 신호를 의심합니다."""
    rng = np.random.default_rng(11)
    gaps = []
    for _ in range(60):
        months = pd.period_range("1900-01", periods=600, freq="M")
        asset = pd.Series(rng.normal(0.007, 0.045, 600), index=months)
        level = (1 + asset).cumprod()
        rf = pd.Series(0.003, index=months)
        signal = monthly_series_signal(level, 10)
        out = timing_returns(signal, asset, rf).iloc[11:]
        strategy = monthly_metrics(out["ret"], out["rf"])
        hold = monthly_metrics(asset.iloc[11:], rf.iloc[11:])
        gaps.append(strategy["sharpe"] - hold["sharpe"])
    assert np.mean(gaps) < 0.05
