import numpy as np
import pandas as pd
import pytest

from src.backtest import Costs, run_backtest
from src.charts import Panel, holding_chart, holding_spans, start_at_one


def test_holding_spans_from_exposure():
    index = pd.bdate_range("2020-01-01", periods=8)
    # 첫 행은 기준일. 1~2일, 5~7일에 노출이 있습니다.
    exposure = pd.Series([0, 1, 1, 0, 0, 1, 1, 1], index=index, dtype=float)
    assert holding_spans(exposure) == [(index[0], index[2]), (index[4], index[7])]


def test_holding_spans_ignore_first_row_and_handle_no_holding():
    index = pd.bdate_range("2020-01-01", periods=4)
    assert holding_spans(pd.Series([1, 0, 0, 0], index=index, dtype=float)) == []
    assert holding_spans(pd.Series([0, 1, 1, 1], index=index, dtype=float)) == [(index[0], index[3])]


def test_holding_spans_match_execution_days_of_backtest():
    """구간은 holding_after_close가 1이 된 날(매수 체결일)에 시작해 0이 된 날(매도 체결일)에 끝나야 합니다."""
    index = pd.bdate_range("2020-01-01", periods=10)
    data = pd.DataFrame({"close": np.linspace(100, 110, 10), "dividend": 0.0, "rf": 0.0001}, index=index)
    after_close = pd.Series([0, 1, 1, 0, 1, 1, 1, 0, 0, 1], index=index, dtype=float)
    result = run_backtest(data, after_close, str(index[1].date()), str(index[-1].date()), Costs())

    # 마지막 날에 산 것은 기간 안에 수익률이 없어 구간이 아닙니다.
    assert holding_spans(result.daily["holding"]) == [(index[1], index[3]), (index[4], index[7])]


def test_start_at_one_matches_metrics_curve():
    """자산 가치 칸의 곡선은 metrics가 MDD를 잴 때 쓰는 곡선과 같아야 합니다 (첫날 1.0, 그 뒤는 일별 자산)."""
    index = pd.bdate_range("2020-01-01", periods=6)
    data = pd.DataFrame({"close": [100, 101, 99, 102, 104, 103.0], "dividend": 0.0, "rf": 0.0001}, index=index)
    result = run_backtest(data, pd.Series(1.0, index=index), str(index[1].date()), str(index[-1].date()), Costs())
    curve = start_at_one(result.daily["equity"])
    assert curve.iloc[0] == 1.0
    assert curve.iloc[1:].equals(result.daily["equity"].iloc[1:])
    assert result.daily["equity"].iloc[0] < 1.0  # 원본은 바꾸지 않음 (진입 비용을 낸 값 그대로)


def test_holding_chart_draws_only_backtest_window(tmp_path):
    index = pd.bdate_range("2020-01-01", periods=30)
    price = pd.Series(np.linspace(100, 130, 30), index=index)
    exposure = pd.Series(0.0, index=index[5:20])
    exposure.iloc[3:8] = 1.0
    equity = pd.Series(np.linspace(1.0, 1.2, 15), index=exposure.index)
    path = holding_chart(
        tmp_path / "chart.png", "제목", ["부제"], price, "가격", exposure,
        panels=[Panel("지표", price.pct_change(), 0.0, "기준", percent=True)],
        overlay=price.rolling(3).mean(), overlay_label="평균",
        equity=equity, hold_equity=equity * 1.01, strategy_label="전략",
    )
    assert path.exists() and path.stat().st_size > 0


def test_holding_chart_rejects_equity_from_other_period(tmp_path):
    index = pd.bdate_range("2020-01-01", periods=30)
    price = pd.Series(np.linspace(100, 130, 30), index=index)
    exposure = pd.Series(0.0, index=index[5:20])
    other = pd.Series(1.0, index=index[6:21])
    with pytest.raises(ValueError):
        holding_chart(tmp_path / "chart.png", "제목", [], price, "가격", exposure, [], equity=other, hold_equity=other)
    with pytest.raises(ValueError):
        holding_chart(tmp_path / "chart.png", "제목", [], price, "가격", exposure, [], equity=other)
