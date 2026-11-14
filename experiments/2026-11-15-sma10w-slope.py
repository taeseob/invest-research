"""10주 이동평균 + 이동평균 기울기 필터, SPY와 KOSPI. 연구 일지: journal/2026-11-15-sma10w-slope.md

규칙: 10주 이동평균 신호를 따르되, 이동평균이 상승 중이면 매도하지 않고, 하락 중이면 매수하지 않습니다.
비교 대상: 단순 보유, 10주 이동평균 단독 (같은 기간, 같은 비용·세금).

실행:
    python experiments/2026-11-15-sma10w-slope.py              # 설계용 기간, 두 자산
    python experiments/2026-11-15-sma10w-slope.py --chart      # 보유 구간 차트만 (표는 다시 만들지 않음)
    python experiments/2026-11-15-sma10w-slope.py --period oos # 표본 외 기간 (마지막에 한 번만)

결과 표는 reports/2026-11-15-sma10w-slope/<기간>.md,
차트는 <기간>-holdings-<자산>-<전략>.png 에 저장합니다.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import (  # noqa: E402
    TRADING_DAYS,
    Costs,
    holding_from_signal,
    metrics,
    moving_average_signal,
    moving_average_slope_signal,
    period_last_days,
    run_backtest,
    snap_to_tick,
    synthetic_index_etf,
)
from src.charts import Panel, holding_chart, metrics_line  # noqa: E402
from src.data import PRICE_TICK  # noqa: E402

EXPERIMENT = "2026-11-15-sma10w-slope"
PERIOD_NAMES = {"design": "설계용 기간", "oos": "표본 외 기간"}
BASE = {"window": 10, "slope_lag": 1, "lag": 1, "week": "W-FRI", "trade": 0.001}
KINDS = [("기울기 필터", "slope"), ("10주 단독", "sma"), ("단순 보유", "hold")]
WEEKDAYS = [("월", "W-MON"), ("화", "W-TUE"), ("수", "W-WED"), ("목", "W-THU"), ("금", "W-FRI")]
NO_TAX = {"dividend_withholding": 0.0, "capital_gains_tax": 0.0, "interest_tax": 0.0}


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    price: pd.Series  # 차트에 그리는 가격 (배당 미포함)
    signal_price: pd.Series  # 신호를 계산하는 가격: price를 호가 단위 개수로 바꾼 것 (snap_to_tick)
    data: pd.DataFrame  # 수익률을 계산하는 표: close, dividend, rf
    price_label: str
    periods: dict
    subperiods: list  # (이름, 시작, 끝)
    taxes: list  # (이름, 세금 가정). 첫째가 세전, 둘째가 세후 기본
    notes: list  # 리포트 머리말
    chart_costs: str


def load_assets() -> list[Asset]:
    spy = pd.read_csv(ROOT / "data" / "spy_daily.csv", index_col=0, parse_dates=True)
    kospi = pd.read_csv(ROOT / "data" / "kospi_daily.csv", index_col=0, parse_dates=True)
    # KOSPI 실험과 같은 합성 ETF: 배당 연 1.5%(12월 일괄), 보수 연 0.15%, 현금 CD 91일물
    kospi_etf = synthetic_index_etf(kospi, 0.015, "december", 0.0015, "cd")
    return [
        Asset(
            key="spy",
            name="SPY",
            price=spy["close"],
            signal_price=snap_to_tick(spy["close"], PRICE_TICK["SPY"]),
            data=spy[["close", "dividend", "rf"]],
            price_label="SPY 일별 종가 (배당 미포함, 로그 눈금)",
            periods={"design": ("1994-01-03", "2012-12-31"), "oos": ("2013-01-02", "2026-08-31")},
            subperiods=[("1994~2000", "1994-01-03", "2000-12-29"), ("2001~2006", "2001-01-02", "2006-12-29"),
                        ("2007~2012", "2007-01-03", "2012-12-31")],
            taxes=[
                ("세전 (매매 비용만)", NO_TAX),
                ("세후", {**NO_TAX, "dividend_withholding": 0.15, "capital_gains_tax": 0.22}),
                ("세후 + 현금 이자소득세 15.4% (참고)", {"dividend_withholding": 0.15, "capital_gains_tax": 0.22, "interest_tax": 0.154}),
            ],
            notes=[
                "기본 가정: 다음 거래일 종가 체결, 편도 매매 비용 0.1%, 현금은 Ken French RF, 달러 기준.",
                "세후: 배당 원천징수 15%, 양도소득세 22%(연간 실현 순이익, 같은 해 손실만 상계, 기본공제 없음).",
                "세후 표의 단순 보유는 아직 내지 않은 양도세가 빠진 값입니다(CAGR (청산 후) 제외). 단순 보유에 유리한 쪽입니다.",
            ],
            chart_costs="비용: 편도 매매 비용 0.1%, 현금은 Ken French RF, 달러 기준.",
        ),
        Asset(
            key="kospi",
            name="KOSPI",
            price=kospi["close"],
            signal_price=snap_to_tick(kospi["close"], PRICE_TICK["^KS11"]),
            data=kospi_etf,
            price_label="KOSPI 지수 일별 종가 (배당 미포함, 로그 눈금)",
            periods={"design": ("1998-01-05", "2012-12-28"), "oos": ("2013-01-02", "2026-09-30")},
            subperiods=[("1998~2000", "1998-01-05", "2000-12-27"), ("2001~2006", "2001-01-02", "2006-12-28"),
                        ("2007~2012", "2007-01-02", "2012-12-28"), ("ETF 상장 후 (2002-10-14~)", "2002-10-14", "2012-12-28")],
            taxes=[
                ("세전 (매매 비용과 보수만)", NO_TAX),
                ("세후", {**NO_TAX, "dividend_withholding": 0.154, "interest_tax": 0.154}),
            ],
            notes=[
                "기본 가정: 다음 거래일 종가 체결, 편도 매매 비용 0.1%, 현금은 CD 91일물, 원화 기준.",
                "합성 ETF: KOSPI 가격 지수 + 배당 연 1.5%(12월 마지막 거래일 전날 일괄) − 보수 연 0.15%. 신호는 원지수로 계산합니다.",
                "세후: 국내 주식형 ETF 일반 계좌 가정. 매매차익 비과세, 분배금과 현금 이자 15.4%.",
            ],
            chart_costs="비용: 편도 매매 비용 0.1%, 합성 ETF(배당 연 1.5%, 보수 연 0.15%), 현금은 CD 91일물, 원화 기준.",
        ),
    ]


# ---------------------------------------------------------------- 실행


def holding_for(asset: Asset, kind: str, window: int, slope_lag: int, lag: int, week: str) -> pd.Series:
    """kind: slope(기울기 필터), sma(이동평균 단독), hold(단순 보유)"""
    if kind == "hold":
        return pd.Series(1.0, index=asset.data.index)
    weeks = period_last_days(asset.signal_price.index, week)
    if kind == "slope":
        signal = moving_average_slope_signal(asset.signal_price, weeks, window, slope_lag)
    elif kind == "sma":
        signal = moving_average_signal(asset.signal_price, weeks, window)
    else:
        raise ValueError(f"알 수 없는 전략: {kind}")
    return holding_from_signal(signal, asset.data.index, lag)


def run(asset: Asset, kind: str, start: str, end: str, tax: dict, window=10, slope_lag=1, lag=1, week="W-FRI", trade=0.001):
    holding = holding_for(asset, kind, window, slope_lag, lag, week)
    result = run_backtest(asset.data, holding, start, end, Costs(trade=trade, **tax))
    return result, metrics(result)


# ---------------------------------------------------------------- 표


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


def filter_effect(asset: Asset, start: str, end: str) -> tuple[list[dict], list[dict]]:
    """필터가 막은 신호(주 단위)와, 필터 때문에 10주 단독과 보유 상태가 달라진 날(일 단위)의 자산 초과수익률."""
    window, slope_lag = BASE["window"], BASE["slope_lag"]
    weeks = period_last_days(asset.signal_price.index, BASE["week"])
    weekly = asset.signal_price.reindex(weeks)
    average = weekly.rolling(window).mean()
    above = weekly > average
    slope = average - average.shift(slope_lag)
    signal = moving_average_slope_signal(asset.signal_price, weeks, window, slope_lag)
    previous = signal.shift(1)
    in_period = (weeks >= pd.Timestamp(start)) & (weeks <= pd.Timestamp(end))

    strategy, m = run(asset, "slope", start, end, NO_TAX, **BASE)
    plain, m_plain = run(asset, "sma", start, end, NO_TAX, **BASE)
    years = m["years"]
    blocked_buy = ((previous == 0.0) & above & (slope < 0))[in_period]
    blocked_sell = ((previous == 1.0) & ~above & (slope > 0))[in_period]
    weekly_rows = [{
        "주 수": int(in_period.sum()),
        "매수 금지로 무시한 주 (연간)": f"{blocked_buy.sum() / years:.1f}",
        "매도 금지로 무시한 주 (연간)": f"{blocked_sell.sum() / years:.1f}",
        "연간 매매: 기울기 필터": f"{m['trades_per_year']:.1f}",
        "연간 매매: 10주 단독": f"{m_plain['trades_per_year']:.1f}",
        "보유 비율: 기울기 필터": f"{m['time_in_market']:.0%}",
        "보유 비율: 10주 단독": f"{m_plain['time_in_market']:.0%}",
    }]

    data = asset.data
    excess = ((data["close"] + data["dividend"]) / data["close"].shift(1) - 1 - data["rf"]).reindex(strategy.daily.index)
    s, p = strategy.daily["holding"].iloc[1:] == 1.0, plain.daily["holding"].iloc[1:] == 1.0
    excess = excess.iloc[1:]
    # 기여: 필터가 주식이고 10주 단독이 현금인 날은 필터가 초과수익률만큼 더 벌고, 반대인 날은 그만큼 덜 법니다.
    daily_rows, total = [], 0.0
    for label, mask, sign in [
        ("필터가 붙잡아 둔 날 (10주 단독 현금, 필터 주식)", s & ~p, 1.0),
        ("필터가 막은 날 (10주 단독 주식, 필터 현금)", ~s & p, -1.0),
        ("둘 다 주식", s & p, 0.0),
        ("둘 다 현금", ~s & ~p, 0.0),
    ]:
        days = excess[mask]
        annual = days.mean() * TRADING_DAYS
        t_value = days.mean() / (days.std() / np.sqrt(len(days))) if len(days) > 1 else np.nan
        contribution = sign * mask.mean() * annual
        total += contribution
        daily_rows.append({
            "구분": label,
            "날 수": int(mask.sum()),
            "날 비율": f"{mask.mean():.1%}",
            "자산 초과수익률 (연환산)": f"{annual:.1%}" if len(days) else "-",
            "t값": f"{t_value:+.2f}" if len(days) > 1 else "-",
            "CAGR 차이 기여 (%p)": f"{contribution * 100:+.2f}" if sign else "0 (같은 상태)",
        })
    saving = (m_plain["trades_per_year"] - m["trades_per_year"]) * BASE["trade"]
    total += saving
    blank = {"날 수": "-", "날 비율": "-", "자산 초과수익률 (연환산)": "-", "t값": "-"}
    daily_rows += [
        {"구분": "매매 비용 절감 (연간 매매 차이 × 편도 비용)", **blank, "CAGR 차이 기여 (%p)": f"{saving * 100:+.2f}"},
        {"구분": "합계 (어림값)", **blank, "CAGR 차이 기여 (%p)": f"{total * 100:+.2f}"},
        {"구분": "실제 CAGR 차이 (기울기 필터 − 10주 단독)", **blank, "CAGR 차이 기여 (%p)": f"{(m['cagr'] - m_plain['cagr']) * 100:+.2f}"},
    ]
    return weekly_rows, daily_rows


def yearly_returns(asset: Asset, start: str, end: str) -> list[dict]:
    runs = {name: run(asset, kind, start, end, NO_TAX, **BASE)[0] for name, kind in KINDS}
    # 첫 행(시작 전날, 자산 1.0 기준일)은 빼고, 첫해 수익률은 1.0 대비로 계산합니다.
    year_end = pd.DataFrame({name: r.daily["equity"].iloc[1:].groupby(r.daily.index[1:].year).last() for name, r in runs.items()})
    yearly = year_end / year_end.shift(1).fillna(1.0) - 1
    yearly["기울기 필터 − 단순 보유"] = yearly["기울기 필터"] - yearly["단순 보유"]
    yearly["기울기 필터 − 10주 단독"] = yearly["기울기 필터"] - yearly["10주 단독"]
    return [{"연도": y, **{c: f"{v:.1%}" for c, v in r.items()}} for y, r in yearly.iterrows()]


def asset_report(asset: Asset, period: str) -> list[str]:
    start, end = asset.periods[period]
    pre_tax, after_tax = asset.taxes[0][1], asset.taxes[1][1]
    lines = [f"# {asset.name}: {PERIOD_NAMES[period]} ({start} ~ {end})", "", *asset.notes,
             "샤프의 무위험 수익률은 세전 현금 수익률입니다. 'CAGR (청산 후)'는 기간 끝에 모두 팔고 세금까지 정산한 뒤의 값입니다.", ""]

    for label, tax in asset.taxes:
        rows = [row(name, run(asset, kind, start, end, tax, **BASE)[1]) for name, kind in KINDS]
        lines += [f"## 기본 결과: {label}", "", table(rows), ""]

    weekly_rows, daily_rows = filter_effect(asset, start, end)
    lines += ["## 필터가 한 일 (세전)", "", table(weekly_rows), "", table(daily_rows), "",
              "필터가 맞게 골랐다면 '붙잡아 둔 날'의 초과수익률은 양수, '막은 날'은 음수여야 합니다.",
              "초과수익률은 자산 수익률(배당 포함) − 현금 수익률입니다. 기여는 날 비율 × 연환산 초과수익률이고, 필터가 현금인 날은 부호를 바꿉니다.",
              "산술 평균으로 어림한 값이라 합계가 실제 CAGR 차이와 조금 다릅니다. "
              "t값은 일별 초과수익률의 평균 / 표준오차이고, 수익률의 자기상관을 무시하므로 실제 유의성은 이보다 약합니다.", ""]

    def sensitivity(title: str, variants: list[tuple[str, str, dict]], taxes=None) -> None:
        for label, tax in taxes or [("세전", pre_tax), ("세후", after_tax)]:
            rows = [row(name, run(asset, kind, start, end, tax, **{**BASE, **change})[1]) for name, kind, change in variants]
            lines.extend([f"## 민감도: {title} ({label})", "", table(rows), ""])

    # 행 이름에 "10주 단독"을 쓰면 안 됩니다. 이 표의 단독 행은 그 행의 이동평균 기간입니다.
    sensitivity("이동평균 기간", [
        (f"{w}주, {name}", kind, {"window": w}) for w in [5, 8, 10, 12, 15, 20, 40]
        for name, kind in [("기울기 필터", "slope"), ("이동평균 단독", "sma")]
    ] + [("단순 보유", "hold", {})])
    sensitivity("기울기 비교 간격", [(f"{k}주 전 이동평균과 비교", "slope", {"slope_lag": k}) for k in [1, 2, 4]])
    sensitivity("매매 비용", [
        (f"편도 {t:.2%}, {name}", kind, {"trade": t}) for t in [0.0005, 0.001, 0.002] for name, kind in KINDS
    ])
    sensitivity("신호 요일 (주 마감 요일)", [
        (f"{day}요일 마감, {name}", kind, {"week": week}) for day, week in WEEKDAYS for name, kind in KINDS[:2]
    ])
    sensitivity("체결 시점", [
        (f"다음 거래일 종가 체결 (기본), {name}", kind, {}) for name, kind in KINDS[:2]
    ] + [(f"신호 낸 날 종가 체결 (참고값), {name}", kind, {"lag": 0}) for name, kind in KINDS[:2]], taxes=[("세전", pre_tax)])

    rows = []
    for label, sub_start, sub_end in asset.subperiods:
        rows += [row(f"{label}, {name}", run(asset, kind, sub_start, sub_end, after_tax, **BASE)[1]) for name, kind in KINDS]
    lines += ["## 하위 기간 (세후)", "", table(rows), ""]

    lines += ["## 연도별 수익률 (세전)", "", table(yearly_returns(asset, start, end)), ""]
    return lines


# ---------------------------------------------------------------- 차트


def chart(asset: Asset, period: str) -> list[Path]:
    """기본 설정(세전)의 보유 구간, 신호에 쓴 지표, 전략과 단순 보유의 자산 가치를 그립니다.
    기울기 필터와 10주 단독을 따로 그립니다. 지표는 신호와 같은 배당 미포함 가격으로 계산하되,
    눈금이 가격 단위로 나오도록 호가 단위로 바꾸기 전의 가격을 씁니다. 비율이라 부호는 신호와 같지만,
    정확한 동률(종가 = 이동평균, 기울기 = 0)인 주는 float32 잔여 오차 때문에 0이 아닌 아주 작은 값으로 그려질 수 있습니다.
    """
    start, end = asset.periods[period]
    window = BASE["window"]
    weeks = period_last_days(asset.price.index, BASE["week"])
    weekly = asset.price.reindex(weeks)
    average = weekly.rolling(window).mean()
    change = average / average.shift(BASE["slope_lag"]) - 1

    deviation = Panel(f"이격도: 주간 종가 / {window}주 이동평균 - 1", weekly / average - 1, 0.0, "이동평균", percent=True)
    slope = Panel(
        f"기울기: {window}주 이동평균의 주간 변화율 (0 위 = 상승 중이라 매도 금지, 0 아래 = 하락 중이라 매수 금지)",
        change, 0.0, "보합", percent=True,
    )
    note = "괄호 안은 같은 비용의 단순 보유. 음영은 매수 체결일 종가 ~ 매도 체결일 종가."
    specs = [
        (
            "slope", "sma10w-slope", f"{asset.name} {window}주 이동평균 + 기울기 필터", f"{window}주 이동평균 + 기울기 필터",
            [
                f"규칙: 현금일 때 주 마지막 거래일 종가 > {window}주 이동평균이고 이동평균이 하락 중이 아니면 매수, "
                "보유 중 종가 ≤ 이동평균이고 이동평균이 상승 중이 아니면 매도.",
                "상승·하락은 이번 주와 지난주 이동평균의 비교. 신호 낸 날의 다음 거래일 종가에 체결.",
            ],
            [deviation, slope],
        ),
        (
            "sma", "sma10w", f"{asset.name} {window}주 이동평균 단독 (비교용)", f"{window}주 이동평균 단독",
            [f"규칙: 주 마지막 거래일 종가가 {window}주 이동평균보다 높으면 보유, 아니면 현금. 신호 낸 날의 다음 거래일 종가에 체결."],
            [deviation],
        ),
    ]

    pre_tax = asset.taxes[0][1]
    hold_result, hold = run(asset, "hold", start, end, pre_tax, **BASE)
    paths = []
    for kind, slug, name, short_name, rule, panels in specs:
        result, m = run(asset, kind, start, end, pre_tax, **BASE)
        paths.append(holding_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-{asset.key}-{slug}.png",
            title=f"{name}: 보유 구간 ({PERIOD_NAMES[period]} {start} ~ {end})",
            subtitle=[f"세전  ·  {metrics_line(m, hold)}", *rule, f"{asset.chart_costs} {note}"],
            price=asset.price,
            price_label=asset.price_label,
            overlay=average,
            overlay_label=f"{window}주 이동평균 (주간 종가)",
            exposure=result.daily["holding"],
            panels=panels,
            equity=result.daily["equity"],
            hold_equity=hold_result.daily["equity"],
            strategy_label=short_name,
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIOD_NAMES, default="design")
    parser.add_argument("--chart", action="store_true", help="보유 구간 차트만 만듭니다")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    assets = load_assets()

    if args.chart:
        for asset in assets:
            for path in chart(asset, args.period):
                print(f"저장: {path.relative_to(ROOT)}")
        return

    lines = [f"# 10주 이동평균 + 기울기 필터: {PERIOD_NAMES[args.period]}", "",
             "규칙: 10주 이동평균 신호를 따르되, 이동평균이 지난주보다 높으면(상승 중) 매도하지 않고, 낮으면(하락 중) 매수하지 않습니다.",
             "비교 대상은 같은 기간, 같은 비용·세금의 10주 이동평균 단독과 단순 보유입니다.", ""]
    for asset in assets:
        lines += ["", *[line.replace("# ", "## ", 1) if line.startswith("#") else line for line in asset_report(asset, args.period)]]

    out = ROOT / "reports" / EXPERIMENT / f"{args.period}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
