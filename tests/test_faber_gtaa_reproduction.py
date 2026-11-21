"""GTAA 근사 재현 실험 스크립트의 계산 경로 테스트: 기간 자르기, 신호를 한 달 미루는지, 신호 열 선택."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "faber_gtaa_reproduction", ROOT / "experiments" / "2026-11-21-faber-gtaa-reproduction.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def synthetic_table(seed: int = 8) -> pd.DataFrame:
    """실험 스크립트가 읽는 열을 모두 가진 가짜 월간 표 (1971-12 ~ 1985-12)."""
    rng = np.random.default_rng(seed)
    months = pd.period_range("1971-12", "1985-12", freq="M", name="month")
    table = pd.DataFrame(index=months)
    for asset in experiment.ASSETS:
        table[f"{asset}_tr"] = rng.normal(0.008, 0.04, len(months))
        table[f"{asset}_pr"] = table[f"{asset}_tr"] - 0.003
    table["us_stock_sig_tr"] = table["us_stock_tr"] + rng.normal(0, 0.001, len(months))
    table["rf"] = 0.004
    return table


def test_cut_drops_months_after_period_end():
    out = experiment.cut(synthetic_table(), "1980-12")
    assert out.index[-1] == pd.Period("1980-12", "M")


def test_run_ignores_data_after_period_end():
    """자른 표와, 자르지 않고 기간 끝 뒤 값만 바꾼 표의 결과가 같아야 합니다 (run 자체가 미래를 보지 않는지)."""
    table = synthetic_table()
    changed = table.copy()
    changed.loc[changed.index > pd.Period("1980-12", "M")] = 0.5
    for source in (None, "total_return", "price"):
        a = experiment.run(experiment.cut(table, "1980-12"), "1974-01", "1980-12", list(experiment.ASSETS), source)
        b = experiment.run(changed, "1974-01", "1980-12", list(experiment.ASSETS), source)
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_frame_equal(a[2], b[2])


def test_run_combines_last_months_signal_with_this_months_return():
    """그달 보유 상태 = 전달 말 신호, 그달 수익률 = 그 보유 상태로 그달 자산 수익률과 현금을 섞은 값.

    수익률이나 보유 상태가 한 달 밀리면(같은 달 신호로 같은 달 수익을 내면) 실패합니다.
    """
    table = synthetic_table()
    assets = list(experiment.ASSETS)
    period = experiment.months("1974-01", "1980-12")
    for source in ("total_return", "price"):
        out, _, held = experiment.run(table, "1974-01", "1980-12", assets, source)
        expected_held = experiment.signals(table, assets, source, 10).shift(1).loc[period]
        pd.testing.assert_frame_equal(held, expected_held, check_names=False)
        returns = table[[f"{a}_tr" for a in assets]].loc[period].to_numpy()
        rf = table["rf"].loc[period].to_numpy()
        h = held.to_numpy()
        np.testing.assert_allclose(out["gross"].to_numpy(), (h * returns + (1 - h) * rf[:, None]).mean(axis=1))


def test_run_holds_last_months_signal():
    """한 달의 수익률만 바꿔도 그 달 보유 상태와 그 달까지의 포트폴리오 비중은 그대로여야 합니다."""
    table = synthetic_table()
    for month in ["1976-05", "1979-11"]:
        changed = table.copy()
        t = pd.Period(month, "M")
        signal_columns = [c for c in table.columns if c.endswith(("_tr", "_pr"))]
        changed.loc[t, signal_columns] = -0.4
        for source in ("total_return", "price"):
            _, _, held_a = experiment.run(table, "1974-01", "1980-12", list(experiment.ASSETS), source)
            _, _, held_b = experiment.run(changed, "1974-01", "1980-12", list(experiment.ASSETS), source)
            pd.testing.assert_series_equal(held_a.loc[t], held_b.loc[t])
            pd.testing.assert_frame_equal(held_a.loc[:t], held_b.loc[:t])


def test_total_return_signal_uses_lagged_dividend_column_for_us_stock():
    """S&P 500의 총수익 지수 신호는 us_stock_sig_tr(12개월 전 D)로 계산해야 합니다."""
    table = synthetic_table()
    changed = table.copy()
    changed["us_stock_tr"] = changed["us_stock_tr"] + 0.2  # 수익률용 열만 크게 바꿈
    a = experiment.signals(table, ["us_stock"], "total_return", 10)
    b = experiment.signals(changed, ["us_stock"], "total_return", 10)
    pd.testing.assert_frame_equal(a, b)
