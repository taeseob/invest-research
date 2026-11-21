"""Faber 재현 실험 스크립트의 입력 준비 테스트: 기간 끝 뒤의 데이터가 신호와 수익률에 들어가지 않는지."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SPEC = importlib.util.spec_from_file_location(
    "faber_reproduction", ROOT / "experiments" / "2026-10-11-faber-sp500-reproduction.py"
)
experiment = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(experiment)


def synthetic_tables(scale_after: pd.Period | None = None):
    """가짜 일별 종가, 월말 표, Shiller D. scale_after를 주면 그 달 뒤의 가격과 배당을 바꿉니다."""
    rng = np.random.default_rng(21)
    index = pd.bdate_range("1927-12-01", "1935-06-28")
    close = pd.Series(100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(index))), index=index)
    months = pd.period_range("1926-01", "1935-06", freq="M")
    shiller = pd.Series(rng.uniform(2, 4, len(months)), index=months)
    if scale_after is not None:
        close[close.index.to_period("M") > scale_after] *= 1.5
        shiller[shiller.index > scale_after] *= 2.0
    close = close.round(2)  # 신호는 호가 단위(0.01) 격자에 있어야 합니다.

    month_end = close.groupby(close.index.to_period("M")).last().to_frame("close")
    month_end["dividend"] = shiller.reindex(month_end.index)
    month_end["total_return"] = (month_end["close"] + month_end["dividend"] / 12) / month_end["close"].shift(1) - 1
    month_end["rf"] = 0.002
    return close, month_end, shiller


def test_build_inputs_cuts_everything_at_period_end():
    end = pd.Period("1933-12", freq="M")
    inputs = experiment.build_inputs(*synthetic_tables(), end)
    assert inputs.daily_close.index.max() <= end.end_time
    assert inputs.daily.index.max() <= end.end_time
    assert inputs.signal_price.index.max() <= end.end_time
    assert inputs.month_end.index.max() == end
    assert inputs.tr_signal_index.index.max() == end


def test_build_inputs_signal_index_ignores_last_11_months_of_dividends():
    """Shiller D는 보간 때문에 최근 값에 앞으로의 배당이 섞여 있습니다. 신호용 지수는 12개월 전 D만 써야 하므로,
    기간 끝 전 11개월의 D를 바꿔도 그대로여야 합니다 (build_inputs의 배당 시차가 12보다 작아지면 실패)."""
    end = pd.Period("1933-12", freq="M")
    close, month_end, shiller = synthetic_tables()
    changed = shiller.copy()
    changed[changed.index > end - 12] *= 3.0
    a = experiment.build_inputs(close, month_end, shiller, end)
    b = experiment.build_inputs(close, month_end, changed, end)
    pd.testing.assert_series_equal(a.tr_signal_index, b.tr_signal_index)


def test_build_inputs_do_not_change_when_data_after_period_end_changes():
    end = pd.Period("1933-12", freq="M")
    a = experiment.build_inputs(*synthetic_tables(), end)
    b = experiment.build_inputs(*synthetic_tables(scale_after=end), end)
    pd.testing.assert_series_equal(a.daily_close, b.daily_close)
    pd.testing.assert_series_equal(a.signal_price, b.signal_price)
    pd.testing.assert_frame_equal(a.month_end, b.month_end)
    pd.testing.assert_series_equal(a.tr_signal_index, b.tr_signal_index)
    pd.testing.assert_frame_equal(a.daily, b.daily)
    for source in ("price", "total_return"):
        pd.testing.assert_series_equal(experiment.signal_for(a, 10, source), experiment.signal_for(b, 10, source))
