"""듀얼 모멘텀 계산 테스트: 12개월 수익률, 선택 규칙, 갈아타기 수익률과 비용, 미래 정보 검사, 무작위 보행 검사."""

import numpy as np
import pandas as pd
import pytest

from src.monthly import dual_momentum_choice, monthly_metrics, rotation_returns, trailing_return

MONTHS = pd.period_range("2000-01", periods=6, freq="M")


def test_trailing_return_includes_this_month():
    r = pd.Series([0.10, 0.20, -0.10], index=MONTHS[:3])
    out = trailing_return(r, 2)
    assert np.isnan(out.iloc[0])
    assert out.iloc[1:].tolist() == pytest.approx([1.1 * 1.2 - 1, 1.2 * 0.9 - 1])


def test_trailing_return_is_nan_when_window_has_a_gap():
    r = pd.Series([0.1, np.nan, 0.1, 0.1, 0.1], index=MONTHS[:5])
    out = trailing_return(r, 2)
    assert out.isna().tolist() == [True, True, True, False, False]


def choice(us, ex, cash, absolute="us"):
    index = MONTHS[: len(us)]
    return dual_momentum_choice(pd.Series(us, index=index, dtype=float), pd.Series(ex, index=index, dtype=float),
                                pd.Series(cash, index=index, dtype=float), absolute).tolist()


def test_choice_rules_by_hand():
    us, ex, cash = [0.20, 0.10, 0.01, -0.10, 0.10, 0.05], [0.10, 0.20, 0.20, -0.20, 0.10, 0.01], [0.05] * 6
    # 미국이 현금보다 높으면 둘 중 높은 쪽(같으면 미국), 아니면 채권. 미국 = 현금이면 채권(초과수익률이 양수여야 함)
    assert choice(us[:5] + [0.05], ex[:5] + [0.01], cash) == ["us_stock", "ex_us", "bond", "bond", "us_stock", "bond"]
    # 고른 자산 기준: 셋째 달은 미국 외(0.20 > 0.05)를 듭니다. 두 해석이 갈리는 경우입니다.
    assert choice(us, ex, cash, "selected")[2] == "ex_us"
    # 절대 모멘텀 없음: 늘 상대 모멘텀
    assert choice(us, ex, cash, None) == ["us_stock", "ex_us", "ex_us", "us_stock", "us_stock", "us_stock"]


def test_selected_rule_checks_the_chosen_asset_against_cash():
    # (미국, 미국 외, 현금): 미국 외가 이겼지만 현금보다 낮음 → 채권 / 미국이 이겼지만 현금보다 낮음 → 채권
    # 미국 외가 이기고 현금보다 높음 → 미국 외 (S&P 기준이면 미국이 현금보다 낮아 채권)
    us, ex, cash = [-0.05, 0.03, 0.01], [0.02, -0.10, 0.20], [0.04, 0.04, 0.04]
    assert choice(us, ex, cash, "selected") == ["bond", "bond", "ex_us"]
    assert choice(us, ex, cash, "us") == ["bond", "bond", "bond"]
    # 미국이 이기고 미국 > 현금, 미국 외 ≤ 현금 → 미국 (고른 자산인 미국만 현금과 비교)
    assert choice([0.10], [0.02], [0.04], "selected") == ["us_stock"]


def test_choice_is_missing_until_all_inputs_exist():
    out = choice([np.nan, 0.1], [0.1, 0.1], [0.0, 0.0])
    assert pd.isna(out[0]) and out[1] == "us_stock"


def test_rotation_returns_and_costs_by_hand():
    returns = pd.DataFrame({"us_stock": [0.10, 0.02, 0.03], "ex_us": [0.50, 0.04, -0.20], "bond": [0.01, 0.01, 0.01]},
                           index=MONTHS[:3])
    held = pd.Series(["us_stock", "us_stock", "bond"], index=MONTHS[:3])
    out = rotation_returns(returns, held, trade_cost=0.001)
    assert out["gross"].tolist() == pytest.approx([0.10, 0.02, 0.01])
    assert out["cost"].tolist() == pytest.approx([0.001, 0.0, 0.002])
    assert out["ret"].iloc[2] == pytest.approx(0.998 * 1.01 - 1)
    assert out["switched"].tolist() == [False, False, True]


def test_rotation_rejects_unknown_or_missing_asset():
    returns = pd.DataFrame({"us_stock": [0.1, 0.1]}, index=MONTHS[:2])
    with pytest.raises(ValueError):
        rotation_returns(returns, pd.Series(["us_stock", None], index=MONTHS[:2]))
    with pytest.raises(ValueError):
        rotation_returns(returns, pd.Series(["us_stock", "gold"], index=MONTHS[:2]))


def gem(returns: pd.DataFrame, rf: pd.Series, absolute="us", window=12) -> tuple[pd.Series, pd.Series]:
    m = trailing_return(pd.concat([returns, rf.rename("cash")], axis=1), window)
    picked = dual_momentum_choice(m["us_stock"], m["ex_us"], m["cash"], absolute)
    return picked, picked.shift(1)


def random_assets(n: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    months = pd.period_range("1900-01", periods=n, freq="M")
    returns = pd.DataFrame({"us_stock": rng.normal(0.007, 0.045, n), "ex_us": rng.normal(0.007, 0.05, n),
                            "bond": rng.normal(0.004, 0.015, n)}, index=months)
    return returns, pd.Series(0.003, index=months)


def test_choice_does_not_change_when_future_returns_change():
    returns, rf = random_assets(200, 1)
    base, base_held = gem(returns, rf)
    for cut in [40, 99, 150]:
        changed = returns.copy()
        changed.iloc[cut + 1 :] = 0.3
        picked, held = gem(changed, rf)
        pd.testing.assert_series_equal(base.iloc[: cut + 1], picked.iloc[: cut + 1])
        # 그달 수익률만 바꿔도 그달 보유 자산은 그대로
        one = returns.copy()
        one.iloc[cut] = [-0.5, 0.5, -0.5]
        assert gem(one, rf)[1].iloc[cut] == base_held.iloc[cut]


def test_held_equals_choice_recomputed_from_data_through_last_month():
    returns, rf = random_assets(150, 2)
    _, held = gem(returns, rf)
    for month in range(12, 150):
        recomputed, _ = gem(returns.iloc[:month], rf.iloc[:month])
        assert held.iloc[month] == recomputed.iloc[-1]


def test_random_walk_gem_is_not_clearly_better_than_fifty_fifty():
    """추세가 없는 가짜 자산에서 GEM 샤프가 50/50 단순 보유보다 뚜렷이 높으면 계산을 의심합니다."""
    gaps = []
    for seed in range(40):
        returns, rf = random_assets(480, 100 + seed)
        _, held = gem(returns, rf)
        start = held.first_valid_index()
        out = rotation_returns(returns.loc[start:], held.loc[start:])
        mix = returns.loc[start:, ["us_stock", "ex_us"]].mean(axis=1)
        gaps.append(monthly_metrics(out["ret"], rf.loc[start:])["sharpe"] - monthly_metrics(mix, rf.loc[start:])["sharpe"])
    assert np.mean(gaps) < 0.05
