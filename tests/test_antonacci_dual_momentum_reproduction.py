"""듀얼 모멘텀 근사 재현 실험 스크립트의 계산 경로 테스트: 기간 자르기, 그달 보유와 그달 수익률, 신호 열 선택."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "antonacci_dual_momentum_reproduction", ROOT / "experiments" / "2026-11-21-antonacci-dual-momentum-reproduction.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)
KINDS = list(experiment.STRATEGIES)


def synthetic_table(seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    months = pd.period_range("1974-01", "1990-12", freq="M", name="month")
    n = len(months)
    table = pd.DataFrame({
        "us_stock_tr": rng.normal(0.009, 0.045, n), "ex_us_tr": rng.normal(0.008, 0.05, n),
        "bond_tr": rng.normal(0.006, 0.015, n), "bond10_tr": rng.normal(0.006, 0.025, n),
        # 현금도 달마다 다르게 둬야 현금이 한 달 밀리는 오류를 잡습니다.
        "rf": rng.uniform(0.002, 0.008, n),
    }, index=months)
    table["us_stock_sig_tr"] = table["us_stock_tr"] + rng.normal(0, 0.001, n)
    return table


def test_cut_drops_months_after_period_end():
    assert experiment.cut(synthetic_table(), "1985-12").index[-1] == pd.Period("1985-12", "M")


def test_run_ignores_data_after_period_end():
    table = synthetic_table()
    changed = table.copy()
    changed.loc[changed.index > pd.Period("1985-12", "M")] = 0.5
    for kind in KINDS:
        a = experiment.run(experiment.cut(table, "1985-12"), "1976-01", "1985-12", kind)[0]
        b = experiment.run(changed, "1976-01", "1985-12", kind)[0]
        pd.testing.assert_frame_equal(a, b)


def test_run_holds_last_months_choice_and_earns_this_months_return():
    """그달 보유 = 전달 말 선택, 그달 수익률 = 그 자산의 그달 수익률. 한 달이 밀리면 실패합니다."""
    table = synthetic_table()
    period = experiment.months("1976-01", "1985-12")
    returns = experiment.asset_returns(table).loc[period]
    for kind in KINDS:
        out, _ = experiment.run(table, "1976-01", "1985-12", kind)
        expected = experiment.choices(table, kind).shift(1).loc[period]
        assert out["held"].tolist() == expected.tolist()
        picked = np.array([returns.loc[month, asset] for month, asset in expected.items()])
        np.testing.assert_allclose(out["gross"].to_numpy(), picked)


def test_held_does_not_depend_on_this_months_returns():
    """그달 수익률만 바꿔도 그달 보유 자산은 그대로여야 합니다 (같은 달 신호로 같은 달 수익을 내면 실패)."""
    table = synthetic_table()
    for month in ["1979-06", "1983-02"]:
        changed = table.copy()
        changed.loc[pd.Period(month, "M"), ["us_stock_tr", "us_stock_sig_tr", "ex_us_tr", "bond_tr"]] = [-0.4, -0.4, 0.4, 0.0]
        for kind in KINDS:
            a = experiment.run(table, "1976-01", "1985-12", kind)[0]["held"]
            b = experiment.run(changed, "1976-01", "1985-12", kind)[0]["held"]
            assert a.loc[:month].tolist() == b.loc[:month].tolist()


def test_held_equals_choice_from_data_through_last_month():
    """그달 보유 = 전달 말까지 자른 자료로 다시 계산한 선택 (choices와 독립적인 검사)."""
    table = synthetic_table()
    for kind in KINDS:
        out, _ = experiment.run(table, "1976-01", "1985-12", kind)
        for month in out.index[::7]:
            recomputed = experiment.choices(experiment.cut(table, str(month - 1)), kind).iloc[-1]
            assert out.loc[month, "held"] == recomputed


def test_asset_returns_use_return_columns_and_half_half_acwi():
    table = synthetic_table()
    out = experiment.asset_returns(table)
    pd.testing.assert_series_equal(out["us_stock"], table["us_stock_tr"], check_names=False)
    pd.testing.assert_series_equal(out["ex_us"], table["ex_us_tr"], check_names=False)
    pd.testing.assert_series_equal(out["bond"], table["bond_tr"], check_names=False)
    pd.testing.assert_series_equal(out["acwi"], 0.5 * (table["us_stock_tr"] + table["ex_us_tr"]), check_names=False)


def test_momentum_uses_signal_columns_and_cash():
    table = synthetic_table()
    m = experiment.momentum(table, 12)

    def trailing(r):
        return (1 + r).rolling(12).apply(np.prod, raw=True) - 1

    np.testing.assert_allclose(m["us_stock"], trailing(table["us_stock_sig_tr"]), equal_nan=True)
    np.testing.assert_allclose(m["acwi"], trailing(0.5 * (table["us_stock_sig_tr"] + table["ex_us_tr"])), equal_nan=True)
    np.testing.assert_allclose(m["cash"], trailing(table["rf"]), equal_nan=True)


def test_absolute_only_rule_by_hand():
    table = synthetic_table()
    m = experiment.momentum(table, 12)
    picked = experiment.choices(table, "absolute_acwi")
    ready = m[["acwi", "cash"]].notna().all(axis=1)
    expected = np.where(m["acwi"] > m["cash"], "acwi", "bond")
    assert picked[ready].tolist() == expected[ready.to_numpy()].tolist()
    assert picked[~ready].isna().all()


def test_absolute_momentum_compares_with_cash():
    """현금 12개월 수익률이 아주 높으면 GEM과 절대 모멘텀만은 늘 채권, 아주 낮으면 채권을 들지 않아야 합니다."""
    table = synthetic_table()
    for rf, bond_expected in [(0.2, True), (-0.2, False)]:
        changed = table.copy()
        changed["rf"] = rf
        for kind in ["gem", "gem_selected", "absolute_acwi"]:
            picked = experiment.choices(changed, kind).dropna()
            assert (picked == "bond").all() if bond_expected else (picked != "bond").all()


def test_hold_and_metrics_use_this_months_returns_and_table_cash():
    table = synthetic_table()
    period = experiment.months("1976-01", "1985-12")
    m = experiment.hold(table, "1976-01", "1985-12", "us_stock")
    r = table["us_stock_tr"].loc[period]
    assert m["cagr"] == pytest.approx(np.prod(1 + r) ** (12 / len(r)) - 1)
    excess = r - table["rf"].loc[period]
    assert m["sharpe"] == pytest.approx(excess.mean() * 12 / (excess.std() * np.sqrt(12)))
    out, gem = experiment.run(table, "1976-01", "1985-12", "gem")
    excess = out["ret"] - table["rf"].loc[period]
    assert gem["sharpe"] == pytest.approx(excess.mean() * 12 / (excess.std() * np.sqrt(12)))


def test_main_rule_is_sp500_based_absolute_momentum():
    """미국 외가 미국보다 강하고 미국은 현금보다 약한 달: 주 규칙(S&P 500 기준)은 채권, 참고 규칙(고른 자산 기준)은
    미국 외, 상대 모멘텀만은 미국 외. 두 해석이 맞바뀌거나 상대 모멘텀만이 채권을 들면 실패합니다."""
    months = pd.period_range("1990-01", periods=24, freq="M", name="month")
    table = pd.DataFrame({"us_stock_tr": 0.0, "us_stock_sig_tr": -0.01, "ex_us_tr": 0.05, "bond_tr": 0.004,
                          "bond10_tr": 0.004, "rf": 0.003}, index=months)
    assert set(experiment.choices(table, "gem").dropna()) == {"bond"}
    assert set(experiment.choices(table, "gem_selected").dropna()) == {"ex_us"}
    assert set(experiment.choices(table, "relative").dropna()) == {"ex_us"}
    assert "bond" not in set(experiment.choices(synthetic_table(), "relative").dropna())


def test_run_charges_costs_and_counts_switches():
    table = synthetic_table()
    out, m = experiment.run(table, "1976-01", "1985-12", "gem", cost=0.001)
    assert out["cost"].iloc[0] == pytest.approx(0.001)
    assert out["switched"].sum() > 0
    np.testing.assert_allclose(out.loc[out["switched"], "cost"].to_numpy(), 0.002)
    np.testing.assert_allclose(out.loc[~out["switched"], "cost"].iloc[1:].to_numpy(), 0.0)
    assert m["switches_per_year"] == pytest.approx(out["switched"].sum() / (len(out) / 12))


def fake_prices(start: str, end: str, seed: int) -> pd.Series:
    rng = np.random.default_rng(seed)
    index = pd.period_range(start, end, freq="M")
    return pd.Series(100 * np.cumprod(1 + rng.normal(0.005, 0.03, len(index))), index=index)


def test_etf_table_replaces_returns_from_second_etf_month_and_ignores_later_prices():
    table = synthetic_table()  # 1974-01 ~ 1990-12
    prices = {"ACWX": fake_prices("1982-06", "1995-12", 1), "AGG": fake_prices("1980-01", "1995-12", 2)}
    out = experiment.etf_table(table, pd.Period("1984-01", "M"), prices)
    acwx = prices["ACWX"]
    # 첫 가격 달은 수익률이 없어 대체 자료 그대로, 그다음 달부터 ETF 그달 수익률
    assert out.loc[pd.Period("1982-06", "M"), "ex_us_tr"] == table.loc[pd.Period("1982-06", "M"), "ex_us_tr"]
    t = pd.Period("1986-03", "M")
    assert out.loc[t, "ex_us_tr"] == pytest.approx(acwx[t] / acwx[t - 1] - 1)
    assert out.loc[pd.Period("1980-01", "M"), "ex_us_tr"] == table.loc[pd.Period("1980-01", "M"), "ex_us_tr"]
    for column in ["us_stock_tr", "us_stock_sig_tr", "bond10_tr", "rf"]:
        pd.testing.assert_series_equal(out[column], table[column])
    # 표 끝 달 뒤의 가격을 바꿔도 그대로
    later = {k: v.copy() for k, v in prices.items()}
    for v in later.values():
        v[v.index > table.index[-1]] *= 3.0
    pd.testing.assert_frame_equal(out, experiment.etf_table(table, pd.Period("1984-01", "M"), later))


def test_etf_table_rejects_gaps_and_short_coverage():
    table = synthetic_table()
    good = {"ACWX": fake_prices("1982-06", "1995-12", 1), "AGG": fake_prices("1980-01", "1995-12", 2)}
    gap = {k: v.copy() for k, v in good.items()}
    gap["ACWX"] = gap["ACWX"].drop(pd.Period("1987-04", "M"))
    with pytest.raises(ValueError):
        experiment.etf_table(table, pd.Period("1984-01", "M"), gap)
    with pytest.raises(ValueError):  # ACWX 수익률은 1982-07부터라 1982-01부터 덮지 못함
        experiment.etf_table(table, pd.Period("1982-01", "M"), good)
    short = {k: v[v.index <= pd.Period("1989-06", "M")] for k, v in good.items()}
    with pytest.raises(ValueError):  # 표 끝 달(1990-12)까지 덮지 못함
        experiment.etf_table(table, pd.Period("1984-01", "M"), short)


def test_signal_uses_lagged_dividend_column_for_us_stock():
    table = synthetic_table()
    changed = table.copy()
    changed["us_stock_tr"] += 0.2  # 수익률용 열만 크게 바꿈
    for kind in KINDS:
        assert experiment.choices(table, kind).tolist() == experiment.choices(changed, kind).tolist()
