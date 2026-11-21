"""GTAA 근사 재현용 코드 테스트: 국채 합성, 자료 읽기, 1/N 포트폴리오, 미래 정보 검사, 무작위 보행 검사."""

import numpy as np
import pandas as pd
import pytest

from src.bonds import par_bond_monthly_returns
from src.data import (
    build_gtaa_monthly,
    month_end_value,
    parse_aqr_commodities,
    parse_french_international,
    parse_h15_yield,
    parse_nareit,
    parse_yahoo_adjclose,
)
from src.monthly import (
    equal_weight_portfolio,
    held_from_signals,
    level_from_returns,
    monthly_metrics,
    signals_from_returns,
)

MONTHS = pd.period_range("2000-01", periods=6, freq="M")


# ---------------------------------------------------------------- 국채 합성


def test_constant_yield_earns_the_yield_and_no_price_change():
    y = 0.06
    out = par_bond_monthly_returns(pd.Series(y, index=MONTHS))
    # 금리가 그대로면 가격은 반기 복리로 한 달 자란 값 (1 + y/2)^(1/6)입니다.
    assert out["total_return"].iloc[1:].tolist() == pytest.approx([(1 + y / 2) ** (1 / 6) - 1] * 5, rel=1e-12)
    assert out["total_return"].iloc[1] == pytest.approx(y / 12, abs=1e-4)
    assert out["price_return"].iloc[1:].abs().max() < 1e-4
    assert np.isnan(out["total_return"].iloc[0])


def test_rising_yield_loses_about_duration_times_change():
    out = par_bond_monthly_returns(pd.Series([0.05, 0.06], index=MONTHS[:2]))
    # 5% 쿠폰 10년 액면가 채권의 수정 듀레이션은 약 7.8입니다. 금리 1%p 상승 → 약 −7.8% + 쿠폰 한 달 + 볼록성
    assert -0.08 < out["total_return"].iloc[1] < -0.065
    assert out["price_return"].iloc[1] == pytest.approx(out["total_return"].iloc[1] - 0.05 / 12)


def test_bond_returns_do_not_use_future_yields():
    rng = np.random.default_rng(1)
    months = pd.period_range("1990-01", periods=40, freq="M")
    yields = pd.Series(rng.uniform(0.02, 0.08, 40), index=months)
    changed = yields.copy()
    changed.iloc[25:] += 0.03
    a, b = par_bond_monthly_returns(yields), par_bond_monthly_returns(changed)
    pd.testing.assert_frame_equal(a.iloc[:25], b.iloc[:25])


def test_bond_matches_cash_flow_formula_on_random_path():
    """임의의 금리 경로에서, 현금흐름을 하나씩 할인한 값과 같은지 (따로 짠 반복문)."""
    rng = np.random.default_rng(2)
    months = pd.period_range("1980-01", periods=30, freq="M")
    yields = pd.Series(rng.uniform(0.01, 0.15, 30), index=months)
    out = par_bond_monthly_returns(yields)
    for t in range(1, 30):
        coupon, y = yields.iloc[t - 1], yields.iloc[t]
        price = sum(coupon / 2 / (1 + y / 2) ** (k - 1 / 6) for k in range(1, 21)) + 1 / (1 + y / 2) ** (20 - 1 / 6)
        assert out["total_return"].iloc[t] == pytest.approx(price - 1, rel=1e-12, abs=1e-14)


def test_bond_rejects_missing_month():
    with pytest.raises(ValueError):
        par_bond_monthly_returns(pd.Series([0.05, 0.05], index=pd.PeriodIndex(["2000-01", "2000-03"], freq="M")))


# ---------------------------------------------------------------- 자료 읽기


def test_parse_h15_skips_holidays_and_converts_percent():
    text = "\n".join([
        '"Series Description","10-year"',
        '"Unique Identifier: ","H15/H15/RIFLGFCY10_N.B"',
        '"Time Period","RIFLGFCY10_N.B"',
        "1962-01-02,4.06",
        "1962-01-03,ND",
        "1962-01-04,4.03",
    ])
    out = parse_h15_yield(text)
    assert out.index.strftime("%Y-%m-%d").tolist() == ["1962-01-02", "1962-01-04"]
    assert out.tolist() == pytest.approx([0.0406, 0.0403])


def test_month_end_value_uses_last_available_day_when_month_end_is_a_holiday():
    text = "\n".join([
        '"Unique Identifier: ","H15/H15/RIFLGFCY10_N.B"',
        '"Time Period","RIFLGFCY10_N.B"',
        "1999-12-30,6.40",
        "1999-12-31,ND",
        "2000-01-31,6.68",
    ])
    out = month_end_value(parse_h15_yield(text))
    assert out.index.astype(str).tolist() == ["1999-12", "2000-01"]
    assert out.tolist() == pytest.approx([0.0640, 0.0668])


def test_parse_yahoo_adjclose_skips_empty_values():
    times = [pd.Timestamp(day + " 15:00", tz="UTC").timestamp() for day in ["2020-01-02", "2020-01-03", "2020-01-06"]]
    data = {"chart": {"result": [{"timestamp": times, "indicators": {"adjclose": [{"adjclose": [10.0, None, 11.0]}]}}]}}
    out = parse_yahoo_adjclose(data)
    assert out.index.strftime("%Y-%m-%d").tolist() == ["2020-01-02", "2020-01-06"]
    assert out.tolist() == [10.0, 11.0]


def test_build_gtaa_monthly_joins_sources_by_month():
    months = pd.period_range("1960-01", "1973-06", freq="M")
    rng = np.random.default_rng(4)
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.005, 0.03, len(months))), index=months)
    dividend = pd.Series(np.linspace(2.0, 4.0, len(months)), index=months)
    month_end = pd.DataFrame({
        "close": close, "dividend": dividend, "rf": 0.004,
        "total_return": (close + dividend / 12) / close.shift(1) - 1, "price_return": close / close.shift(1) - 1,
    })
    commodity_er = pd.Series(rng.normal(0, 0.03, len(months)), index=months)
    reit_index = pd.DataFrame({"total_index": np.linspace(100, 150, len(months)), "price_index": np.linspace(100, 120, len(months))},
                              index=months)
    table = build_gtaa_monthly(
        month_end, eafe=pd.Series(0.01, index=months), eafe_price=pd.Series(0.008, index=months),
        bond_yield=pd.Series(0.06, index=months), commodity_er=commodity_er, reits={"reit": reit_index},
        current_month=pd.Period("1973-06", freq="M"),
    )
    assert table.index[0] == pd.Period("1971-12", "M") and table.index[-1] == pd.Period("1973-05", "M")
    t = pd.Period("1972-05", "M")
    # 원자재 총수익률 = 그달 초과수익률 + 그달 현금
    assert table.loc[t, "commodity_tr"] == pytest.approx(commodity_er[t] + 0.004)
    assert table.loc[t, "reit_tr"] == pytest.approx(reit_index.loc[t, "total_index"] / reit_index.loc[t - 1, "total_index"] - 1)
    assert table.loc[t, "us_stock_tr"] == pytest.approx(month_end.loc[t, "total_return"])
    # 신호용 S&P 총수익률은 12개월 전 D를 씁니다.
    assert table.loc[t, "us_stock_sig_tr"] == pytest.approx((close[t] + dividend[t - 12] / 12) / close[t - 1] - 1)


def test_parse_h15_rejects_missing_series():
    with pytest.raises(ValueError):
        parse_h15_yield('"Unique Identifier: ","H15/H15/OTHER"\n"Time Period","OTHER"\n1962-01-02,4.0')


def nareit_sheet() -> pd.DataFrame:
    rows = [[None] * 6 for _ in range(10)]
    rows[0][0] = "FTSE Nareit U.S. Real Estate Index Series"
    rows[5][1] = "All REITs"
    rows[6][1], rows[6][3] = "Total", "Price"
    rows[7][:5] = ["Date", "Return", "Index", "Return", "Index"]
    rows[8] = [pd.Timestamp("1971-12-31"), None, 100.0, None, 100.0, None]
    rows[9] = [pd.Timestamp("1972-01-31"), 1.2, 101.2, 0.3, 100.3, None]
    return pd.DataFrame(rows)


def test_parse_nareit_reads_total_and_price_index():
    out = parse_nareit(nareit_sheet(), "All REITs")
    assert out.index.astype(str).tolist() == ["1971-12", "1972-01"]
    assert out["total_index"].tolist() == [100.0, 101.2]
    assert out["price_index"].tolist() == [100.0, 100.3]


def test_parse_nareit_checks_header_layout():
    sheet = nareit_sheet()
    sheet.iat[7, 2] = "Return"  # 총수익 지수 자리의 머리글이 다르면 멈춰야 합니다.
    with pytest.raises(ValueError):
        parse_nareit(sheet, "All REITs")


def test_parse_french_international_reads_only_first_block():
    text = "\n".join([
        "",
        "     Value-Weight Dollar Returns      All 4 Data Items Not Reqd",
        "                -- BE/ME --",
        "          Mkt   High    Low",
        "197501  17.79  16.73  17.76",
        "197502  13.41  15.49  12.55",
        "",
        "     Value-Weight Local  Returns      All 4 Data Items Not Reqd",
        "                -- BE/ME --",
        "          Mkt   High    Low",
        "197501   9.00   9.00   9.00",
    ])
    out = parse_french_international(text)
    assert out.index.astype(str).tolist() == ["1975-01", "1975-02"]
    assert out.tolist() == pytest.approx([0.1779, 0.1341])


def test_parse_aqr_commodities_finds_column_by_name():
    rows = [["title", None, None], [None, "Other", "Excess return of equal-weight commodities portfolio"],
            [pd.Timestamp("1877-02-28"), 0.5, -0.08], [pd.Timestamp("1877-03-30"), 0.5, 0.02]]
    out = parse_aqr_commodities(pd.DataFrame(rows))
    assert out.index.astype(str).tolist() == ["1877-02", "1877-03"]
    assert out.tolist() == pytest.approx([-0.08, 0.02])


# ---------------------------------------------------------------- 1/N 포트폴리오


def frames(returns, held, rf=0.01):
    cols = [f"a{i}" for i in range(len(returns[0]))]
    index = MONTHS[: len(returns)]
    return (pd.DataFrame(returns, index=index, columns=cols), pd.DataFrame(held, index=index, columns=cols, dtype=float),
            pd.Series(rf, index=index))


def test_portfolio_by_hand():
    r, h, rf = frames([[0.10, -0.20], [0.05, 0.00]], [[1, 0], [1, 1]])
    out = equal_weight_portfolio(r, h, rf)
    # 1월: (0.10 + 0.01) / 2, 2월: (0.05 + 0.00) / 2
    assert out["gross"].tolist() == pytest.approx([0.055, 0.025])
    assert out["invested"].tolist() == pytest.approx([0.5, 1.0])


def test_all_held_is_buy_and_hold_and_none_held_is_cash():
    r, h, rf = frames([[0.10, -0.20], [0.05, 0.03]], [[1, 1], [1, 1]])
    assert equal_weight_portfolio(r, h, rf)["gross"].tolist() == pytest.approx([-0.05, 0.04])
    r, h, rf = frames([[0.10, -0.20], [0.05, 0.03]], [[0, 0], [0, 0]])
    assert equal_weight_portfolio(r, h, rf)["gross"].tolist() == pytest.approx([0.01, 0.01])


def test_rebalancing_cost_by_hand():
    r, h, rf = frames([[0.10, -0.10], [0.00, 0.00]], [[1, 1], [1, 0]], rf=0.0)
    out = equal_weight_portfolio(r, h, rf, trade_cost=0.001)
    # 1월: 현금에서 두 자산을 0.5씩 삼 → 거래 1.0
    assert out["turnover"].iloc[0] == pytest.approx(1.0)
    # 2월 초: 비중이 0.55/1.0, 0.45/1.0으로 움직임. 목표 0.5, 0 → |0.5−0.55| + |0−0.45| = 0.5
    assert out["turnover"].iloc[1] == pytest.approx(0.5)
    assert out["ret"].iloc[0] == pytest.approx((1 - 0.001) * (1 + 0.0) - 1)
    no_cost = equal_weight_portfolio(r, h, rf, trade_cost=0.0)
    pd.testing.assert_series_equal(no_cost["ret"], no_cost["gross"], check_names=False)


def test_portfolio_rejects_missing_values():
    r, h, rf = frames([[0.10, np.nan]], [[1, 1]])
    with pytest.raises(ValueError):
        equal_weight_portfolio(r, h, rf)


def test_level_from_returns_starts_at_one_and_rejects_gaps():
    months = pd.period_range("2000-01", periods=4, freq="M")
    level = level_from_returns(pd.Series([np.nan, 0.10, -0.50, 0.20], index=months))
    assert level.index.astype(str).tolist() == ["2000-01", "2000-02", "2000-03", "2000-04"]
    assert level.tolist() == pytest.approx([1.0, 1.1, 0.55, 0.66])
    with pytest.raises(ValueError):
        level_from_returns(pd.Series([0.1, np.nan, 0.2], index=months[:3]))


def gtaa(returns: pd.DataFrame, rf: pd.Series, window: int = 10, cost: float = 0.0) -> pd.DataFrame:
    held = held_from_signals(signals_from_returns(returns, window))
    start = held.dropna().index[0]
    return equal_weight_portfolio(returns.loc[start:], held.loc[start:], rf.loc[start:], cost)


def test_held_in_a_month_does_not_depend_on_that_months_return():
    """t+1월 수익률만 바꿔도 t+1월 보유 상태는 그대로여야 합니다 (신호를 한 달 미루지 않으면 실패).

    t+1월 이후만 바꾸고 t월까지 비교하는 검사는 '같은 달 신호로 같은 달 수익을 먹는' 오류를 잡지 못합니다.
    """
    rng = np.random.default_rng(6)
    months = pd.period_range("1990-01", periods=120, freq="M")
    returns = pd.DataFrame(rng.normal(0.006, 0.05, (120, 3)), index=months, columns=list("abc"))
    base = held_from_signals(signals_from_returns(returns, 10))
    for month in [30, 61, 99]:
        changed = returns.copy()
        changed.iloc[month] = [-0.5, 0.5, -0.5]
        held = held_from_signals(signals_from_returns(changed, 10))
        pd.testing.assert_series_equal(base.iloc[month], held.iloc[month])
    # 모든 달에서: 그달 보유 상태 = 전달까지의 수익률만으로 다시 계산한 전달 말 신호 (두 달 미뤄도 실패)
    for month in range(11, 120):
        pd.testing.assert_series_equal(
            base.iloc[month], signals_from_returns(returns.iloc[:month], 10).iloc[-1], check_names=False
        )


@pytest.mark.parametrize("cut", [40, 77, 150])
def test_gtaa_does_not_change_when_future_returns_change(cut):
    rng = np.random.default_rng(3)
    months = pd.period_range("1990-01", periods=200, freq="M")
    returns = pd.DataFrame(rng.normal(0.006, 0.04, (200, 5)), index=months, columns=list("abcde"))
    rf = pd.Series(0.003, index=months)
    changed = returns.copy()
    changed.iloc[cut + 1 :] = rng.normal(-0.02, 0.08, (200 - cut - 1, 5))
    a, b = gtaa(returns, rf, cost=0.001), gtaa(changed, rf, cost=0.001)
    t = months[cut]
    pd.testing.assert_frame_equal(a.loc[:t], b.loc[:t])


def test_random_walk_gtaa_is_not_clearly_better_than_buy_and_hold():
    """추세가 없는 가짜 자산 다섯 개에서 GTAA 샤프가 단순 보유보다 뚜렷이 높으면 신호나 계산을 의심합니다."""
    rng = np.random.default_rng(13)
    gaps = []
    for _ in range(40):
        months = pd.period_range("1900-01", periods=480, freq="M")
        returns = pd.DataFrame(rng.normal(0.007, 0.045, (480, 5)), index=months, columns=list("abcde"))
        rf = pd.Series(0.003, index=months)
        timed = gtaa(returns, rf)
        hold = equal_weight_portfolio(returns.loc[timed.index], pd.DataFrame(1.0, index=timed.index, columns=returns.columns),
                                      rf.loc[timed.index])
        gaps.append(monthly_metrics(timed["ret"], rf.loc[timed.index])["sharpe"]
                    - monthly_metrics(hold["ret"], rf.loc[timed.index])["sharpe"])
    assert np.mean(gaps) < 0.05
