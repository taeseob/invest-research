"""10주 이동평균 + ADX 진입 필터, KOSPI. 연구 일지: journal/2026-11-02-sma10w-adx25-kospi.md

실행:
    python experiments/2026-11-02-sma10w-adx25-kospi.py              # 설계용 기간
    python experiments/2026-11-02-sma10w-adx25-kospi.py --period oos # 표본 외 기간 (마지막에 한 번만)

결과 표는 reports/2026-11-02-sma10w-adx25-kospi/<기간>.md 에 저장합니다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import (  # noqa: E402
    Costs,
    holding_from_signal,
    metrics,
    moving_average_adx_signal,
    moving_average_signal,
    period_bars,
    period_last_days,
    run_backtest,
    wilder_adx,
)

PERIODS = {
    "design": ("1998-01-05", "2012-12-28"),
    "oos": ("2013-01-02", "2026-09-30"),
}
BASE = {
    "window": 10,
    "adx_period": 14,
    "threshold": 25.0,
    "adx_frame": "week",
    "lag": 1,
    "trade": 0.001,
    "dividend_yield": 0.015,
    "dividend_timing": "december",
    "expense": 0.0015,
    "cash": "cd",
}
PRE_TAX = {"dividend_withholding": 0.0, "interest_tax": 0.0}
AFTER_TAX = {"dividend_withholding": 0.154, "interest_tax": 0.154}


def load_kospi() -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / "kospi_daily.csv", index_col=0, parse_dates=True)


def investable(kospi: pd.DataFrame, dividend_yield: float, dividend_timing: str, expense: float, cash: str) -> pd.DataFrame:
    """지수를 따라가는 ETF를 합성합니다. 보수는 가격에서 빼고, 지수에 빠진 배당은 dividend 열로 더합니다.

    배당 금액은 전날 종가 × 수익률입니다. 지수에는 배당락으로 떨어진 가격이 이미 들어 있습니다.
    - december: 매년 12월의 마지막 거래일 전날(국내 12월 결산 기업의 배당락 시기)에 한 번에 받습니다.
    - even: 달력 일수에 비례해 매일 나눠 받습니다.
    """
    index = kospi.index
    years = (index - index[0]).days / 365.25
    close = kospi["close"] * (1 - expense) ** years
    previous = close.shift(1)

    if dividend_timing == "december":
        december = pd.Series(index, index=index)[index.month == 12]
        ex_days = december.groupby(december.index.year).nth(-2)
        dividend = pd.Series(0.0, index=index)
        dividend[pd.DatetimeIndex(ex_days)] = dividend_yield * previous[pd.DatetimeIndex(ex_days)]
    elif dividend_timing == "even":
        elapsed = pd.Series(index, index=index).diff().dt.days / 365.25
        dividend = (dividend_yield * elapsed * previous).fillna(0.0)
    else:
        raise ValueError(f"알 수 없는 배당 시점: {dividend_timing}")

    rf = kospi["rf"] if cash == "cd" else kospi["rf"] * 0.0
    return pd.DataFrame({"close": close, "dividend": dividend, "rf": rf}, index=index)


def signal_for(kospi: pd.DataFrame, window: int, adx_period: int, threshold: float, adx_frame: str, use_adx: bool) -> pd.Series:
    """신호는 배당과 보수가 빠진 원지수로 계산합니다. 그 주 마지막 거래일 종가까지의 값만 씁니다."""
    weeks = period_last_days(kospi.index, "W-FRI")
    if not use_adx:
        return moving_average_signal(kospi["close"], weeks, window)
    bars = period_bars(kospi, "W-FRI") if adx_frame == "week" else kospi
    return moving_average_adx_signal(kospi["close"], weeks, window, wilder_adx(bars, adx_period), threshold)


def run(kospi, start, end, tax, window, adx_period, threshold, adx_frame, lag, trade, dividend_yield, dividend_timing, expense, cash, kind="strategy"):
    """kind: strategy(이동평균 + ADX), sma_only(이동평균 단독), hold(단순 보유)"""
    data = investable(kospi, dividend_yield, dividend_timing, expense, cash)
    if kind == "hold":
        holding = pd.Series(1.0, index=data.index)
    else:
        signal = signal_for(kospi, window, adx_period, threshold, adx_frame, use_adx=kind == "strategy")
        holding = holding_from_signal(signal, data.index, lag)
    result = run_backtest(data, holding, start, end, Costs(trade=trade, capital_gains_tax=0.0, **tax))
    return result, metrics(result)


def row(name: str, m: dict) -> dict:
    recovery = "미회복" if pd.isna(m["mdd_recovery_months"]) else f"{m['mdd_recovery_months']:.0f}개월"
    return {
        "구분": name,
        "CAGR": f"{m['cagr']:.2%}",
        "CAGR (청산 후)": f"{m['cagr_after_liquidation']:.2%}",
        "변동성": f"{m['volatility']:.2%}",
        "샤프": f"{m['sharpe']:.2f}",
        "MDD": f"{m['mdd']:.1%}",
        "MDD 기간": f"{m['mdd_peak']:%Y-%m} ~ {m['mdd_trough']:%Y-%m}",
        "MDD 회복": recovery,
        "연간 매매": f"{m['trades_per_year']:.1f}",
        "보유 비율": f"{m['time_in_market']:.0%}",
    }


def table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def comparison_rows(kospi, start, end, tax, params) -> list[dict]:
    rows = []
    for name, kind in [("10주 + ADX", "strategy"), ("10주 단독", "sma_only"), ("단순 보유", "hold")]:
        _, m = run(kospi, start, end, tax, **params, kind=kind)
        rows.append(row(name, m))
    return rows


def filter_effect(kospi: pd.DataFrame, start: str, end: str) -> list[dict]:
    """ADX 필터가 진입을 얼마나 막았는지 주 단위로 셉니다."""
    weeks = period_last_days(kospi.index, "W-FRI")
    above = moving_average_signal(kospi["close"], weeks, BASE["window"])
    signal = signal_for(kospi, BASE["window"], BASE["adx_period"], BASE["threshold"], BASE["adx_frame"], use_adx=True)
    adx = wilder_adx(period_bars(kospi, "W-FRI"), BASE["adx_period"]).reindex(weeks)
    in_period = (weeks >= pd.Timestamp(start)) & (weeks <= pd.Timestamp(end))
    above, signal, adx = above[in_period], signal[in_period], adx[in_period]
    entries = int(((signal == 1.0) & (signal.shift(1) == 0.0)).sum())
    return [{
        "주 수": len(signal),
        "ADX ≥ 25인 주": f"{(adx >= BASE['threshold']).mean():.0%}",
        "이동평균 위인 주": f"{(above == 1.0).mean():.0%}",
        "보유한 주": f"{(signal == 1.0).mean():.0%}",
        "이동평균 위인데 필터로 미보유인 주": f"{((above == 1.0) & (signal == 0.0)).mean():.0%}",
        "진입 횟수": entries,
        "ADX 중앙값": f"{adx.median():.1f}",
    }]


def yearly_returns(kospi, start, end, params) -> list[dict]:
    runs = {name: run(kospi, start, end, PRE_TAX, **params, kind=kind)[0] for name, kind in [("10주 + ADX", "strategy"), ("10주 단독", "sma_only"), ("단순 보유", "hold")]}
    # 첫 행(시작 전날, 자산 1.0 기준일)은 빼고, 첫해 수익률은 1.0 대비로 계산합니다.
    year_end = pd.DataFrame({
        name: r.daily["equity"].iloc[1:].groupby(r.daily.index[1:].year).last() for name, r in runs.items()
    })
    yearly = year_end / year_end.shift(1).fillna(1.0) - 1
    yearly["단순 보유와 차이"] = yearly["10주 + ADX"] - yearly["단순 보유"]
    return [{"연도": y, **{c: f"{v:.1%}" for c, v in r.items()}} for y, r in yearly.iterrows()]


def kodex_check(kospi: pd.DataFrame, params: dict) -> list[dict]:
    """합성 ETF와 실제 KODEX 200의 연도별 총수익 비교. KODEX 200은 KOSPI 200을 따라가므로 지수 차이가 섞입니다."""
    kodex = pd.read_csv(ROOT / "data" / "kodex200_daily.csv", index_col=0, parse_dates=True)
    synth = investable(kospi, params["dividend_yield"], params["dividend_timing"], params["expense"], params["cash"])
    synth_return = (synth["close"] + synth["dividend"]) / synth["close"].shift(1) - 1
    rows = []
    for year in [2010, 2011, 2012]:
        actual = kodex.loc[str(year)]
        first_close = kodex["close"][kodex.index < f"{year}-01-01"].iloc[-1]
        rows.append({
            "연도": year,
            "KODEX 200 총수익": f"{(1 + actual['total_return']).prod() - 1:.1%}",
            "합성 KOSPI 총수익": f"{(1 + synth_return.loc[str(year)]).prod() - 1:.1%}",
            "KODEX 200 분배율 (분배금 / 연초 가격)": f"{actual['dividend'].sum() / first_close:.2%}",
            "합성 배당률": f"{params['dividend_yield']:.2%}",
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIODS, default="design")
    period = parser.parse_args().period
    start, end = PERIODS[period]
    kospi = load_kospi()

    sections = [f"# 10주 이동평균 + ADX 진입 필터, KOSPI: {period} ({start} ~ {end})", ""]
    sections += [
        "기본 가정: 주봉 ADX(14) ≥ 25일 때만 진입, 다음 거래일 종가 체결, 편도 매매 비용 0.1%, 현금은 CD 91일물, 원화 기준.",
        "합성 ETF: KOSPI 가격 지수 + 배당 연 1.5%(12월 마지막 거래일 전날 일괄) − 보수 연 0.15%.",
        "세후: 국내 주식형 ETF 일반 계좌 가정. 매매차익 비과세, 분배금과 현금 이자 15.4%.",
        "샤프의 무위험 수익률은 세전 CD 금리입니다. 'CAGR (청산 후)'는 기간 끝에 모두 판 뒤의 값입니다.",
        "",
    ]

    for label, tax in [("세전 (매매 비용과 보수만)", PRE_TAX), ("세후", AFTER_TAX)]:
        sections += [f"## 기본 결과: {label}", "", table(comparison_rows(kospi, start, end, tax, BASE)), ""]

    sections += ["## ADX 필터가 한 일", "", table(filter_effect(kospi, start, end)), ""]

    def sensitivity(title: str, variants: list[tuple[str, dict]], with_hold: bool = False) -> None:
        for label, tax in [("세전", PRE_TAX), ("세후", AFTER_TAX)]:
            rows = []
            for name, change in variants:
                params = {**BASE, **change}
                rows.append(row(name, run(kospi, start, end, tax, **params)[1]))
                if with_hold:
                    rows.append(row(f"단순 보유, {name}", run(kospi, start, end, tax, **params, kind="hold")[1]))
            sections.extend([f"## 민감도: {title} ({label})", "", table(rows), ""])

    sensitivity("ADX 기준값", [(f"ADX ≥ {t}", {"threshold": float(t)}) for t in [15, 20, 25, 30, 35]])
    sensitivity("ADX 기간", [(f"ADX({p})", {"adx_period": p}) for p in [10, 14, 20]])
    sensitivity("이동평균 기간", [(f"{w}주", {"window": w}) for w in [5, 8, 10, 12, 15, 20, 40]])
    sensitivity("ADX 시간 틀", [("주봉 ADX(14)", {"adx_frame": "week"}), ("일봉 ADX(14), 주 마지막 거래일에 읽기", {"adx_frame": "day"})])
    sensitivity("매매 비용", [(f"편도 {t:.2%}", {"trade": t}) for t in [0.0005, 0.001, 0.002]], with_hold=True)
    sensitivity(
        "배당",
        [(f"연 {y:.1%}, 12월 일괄", {"dividend_yield": y}) for y in [0.01, 0.015, 0.02]] + [("연 1.5%, 매일 균등", {"dividend_timing": "even"})],
        with_hold=True,
    )
    sensitivity("현금 수익률 (0%인 행은 샤프의 무위험 수익률도 0)", [("CD 금리", {"cash": "cd"}), ("0%", {"cash": "zero"})], with_hold=True)

    rows = [row("다음 거래일 종가 체결 (기본)", run(kospi, start, end, PRE_TAX, **BASE)[1])]
    rows.append(row("신호 낸 날 종가 체결 (참고값)", run(kospi, start, end, PRE_TAX, **{**BASE, "lag": 0})[1]))
    sections += ["## 민감도: 체결 시점 (세전)", "", table(rows), ""]

    sections += ["## 연도별 수익률 (세전)", "", table(yearly_returns(kospi, start, end, BASE)), ""]

    if period == "design":
        sections += ["## 교차 확인: 합성 ETF와 실제 KODEX 200", "", table(kodex_check(kospi, BASE)), ""]

    out = ROOT / "reports" / "2026-11-02-sma10w-adx25-kospi" / f"{period}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(sections))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
