"""20주·40주 이동평균 단독, SPY와 KOSPI (신호 낸 날 종가 체결). 연구 일지: journal/2026-11-15-sma20w-40w.md

규칙: 주 마지막 거래일 종가가 N주 이동평균보다 높으면 보유, 아니면 현금 (N = 20, 40).
체결: 사용자 지정에 따라 신호 낸 날 종가 (저장소 기본 규칙은 다음 거래일 종가).
비교 대상: 같은 기간, 같은 비용·세금의 단순 보유. 민감도 분석은 사용자 요청으로 하지 않습니다.

실행:
    python experiments/2026-11-15-sma20w-40w.py              # 설계용 기간, 두 자산
    python experiments/2026-11-15-sma20w-40w.py --chart      # 보유 구간 차트만 (표는 다시 만들지 않음)
    python experiments/2026-11-15-sma20w-40w.py --period oos # 표본 외 기간 (마지막에 한 번만)

결과 표는 reports/2026-11-15-sma20w-40w/<기간>.md,
차트는 <기간>-holdings-<자산>-sma<N>w.png 에 저장합니다.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import (  # noqa: E402
    Costs,
    holding_from_signal,
    metrics,
    moving_average_signal,
    period_last_days,
    run_backtest,
    snap_to_tick,
    synthetic_index_etf,
    trade_list,
)
from src.charts import Panel, holding_chart, metrics_line  # noqa: E402
from src.data import PRICE_TICK  # noqa: E402

EXPERIMENT = "2026-11-15-sma20w-40w"
PERIOD_NAMES = {"design": "설계용 기간", "oos": "표본 외 기간"}
WINDOWS = [20, 40]
LAG = 0  # 신호 낸 날 종가 체결 (사용자 지정)
TRADE = 0.001
WEEK = "W-FRI"
NO_TAX = {"dividend_withholding": 0.0, "capital_gains_tax": 0.0, "interest_tax": 0.0}
# 매도 시 고점 대비 하락 폭 구간 (양수로 적은 하락률의 경계)
BUCKETS = [0.0, 0.05, 0.10, 0.15, 0.20, float("inf")]


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    price: pd.Series  # 차트와 고점 대비 하락에 쓰는 가격 (배당 미포함 종가)
    signal_price: pd.Series  # 신호를 계산하는 가격: price를 호가 단위 개수로 바꾼 것 (snap_to_tick)
    data: pd.DataFrame  # 수익률을 계산하는 표: close, dividend, rf
    price_label: str
    periods: dict
    taxes: list  # (이름, 세금 가정). 첫째가 세전, 둘째가 세후
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
            taxes=[
                ("세전 (매매 비용만)", NO_TAX),
                ("세후", {**NO_TAX, "dividend_withholding": 0.15, "capital_gains_tax": 0.22}),
            ],
            notes=[
                "기본 가정: 신호 낸 날 종가 체결, 편도 매매 비용 0.1%, 현금은 Ken French RF, 달러 기준.",
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
            taxes=[
                ("세전 (매매 비용과 보수만)", NO_TAX),
                ("세후", {**NO_TAX, "dividend_withholding": 0.154, "interest_tax": 0.154}),
            ],
            notes=[
                "기본 가정: 신호 낸 날 종가 체결, 편도 매매 비용 0.1%, 현금은 CD 91일물, 원화 기준.",
                "합성 ETF: KOSPI 가격 지수 + 배당 연 1.5%(12월 마지막 거래일 전날 일괄) − 보수 연 0.15%. 신호는 원지수로 계산합니다.",
                "세후: 국내 주식형 ETF 일반 계좌 가정. 매매차익 비과세, 분배금과 현금 이자 15.4%.",
            ],
            chart_costs="비용: 편도 매매 비용 0.1%, 합성 ETF(배당 연 1.5%, 보수 연 0.15%), 현금은 CD 91일물, 원화 기준.",
        ),
    ]


# ---------------------------------------------------------------- 실행


def holding_for(asset: Asset, window: int | None) -> pd.Series:
    """window가 None이면 단순 보유입니다."""
    if window is None:
        return pd.Series(1.0, index=asset.data.index)
    weeks = period_last_days(asset.signal_price.index, WEEK)
    signal = moving_average_signal(asset.signal_price, weeks, window)
    return holding_from_signal(signal, asset.data.index, LAG)


def run(asset: Asset, window: int | None, start: str, end: str, tax: dict):
    holding = holding_for(asset, window)
    result = run_backtest(asset.data, holding, start, end, Costs(trade=TRADE, **tax))
    return result, metrics(result), holding


def trades_for(asset: Asset, window: int, start: str, end: str) -> pd.DataFrame:
    """세전 백테스트의 보유 구간별 고점 대비 하락. 자산 기준은 세전 자산 가치(배당 재투자, 매매 비용 포함)입니다."""
    result, _, holding = run(asset, window, start, end, asset.taxes[0][1])
    return trade_list(holding.reindex(result.daily.index), asset.price, result.daily["equity"])


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


def drawdown_summary(trades: pd.DataFrame) -> dict:
    """매도한 구간만 요약합니다. 하락률은 음수로 적습니다 (-12.3% = 고점에서 12.3% 내린 뒤 매도)."""
    sold = trades[~trades["open"]]
    drop = -sold["from_peak"]
    summary = {
        "매도 횟수": len(sold),
        "평균": f"{sold['from_peak'].mean():.1%}",
        "중앙값": f"{sold['from_peak'].median():.1%}",
        "가장 큰 하락": f"{sold['from_peak'].min():.1%}",
        "가장 작은 하락": f"{sold['from_peak'].max():.1%}",
    }
    for low, high in zip(BUCKETS[:-1], BUCKETS[1:]):
        label = f"{low:.0%} 이상" if high == float("inf") else f"{low:.0%}~{high:.0%}"
        summary[f"하락 {label}"] = int(((drop >= low) & (drop < high)).sum())
    summary["자산 기준 평균"] = f"{sold['equity_from_peak'].mean():.1%}"
    summary["자산 기준 가장 큰 하락"] = f"{sold['equity_from_peak'].min():.1%}"
    summary["고점→매도 평균 주"] = f"{sold['weeks_peak_to_sell'].mean():.1f}"
    summary["매수가보다 높게 판 비율 (지수 기준)"] = f"{(sold['from_buy'] > 0).mean():.0%}"
    return summary


def trade_rows(trades: pd.DataFrame) -> list[dict]:
    rows = []
    for number, t in enumerate(trades.itertuples(index=False), start=1):
        rows.append({
            "번호": number,
            "매수일": f"{t.buy_date:%Y-%m-%d}",
            "매수가": f"{t.buy_price:,.2f}",
            "보유 중 고점일": f"{t.peak_date:%Y-%m-%d}",
            "고점": f"{t.peak_price:,.2f}",
            "매도일": "(기간 끝까지 보유)" if t.open else f"{t.sell_date:%Y-%m-%d}",
            "매도가": f"{t.sell_price:,.2f}",
            "고점 대비 (지수)": f"{t.from_peak:.1%}",
            "고점 대비 (자산)": f"{t.equity_from_peak:.1%}",
            "매수가 대비": f"{t.from_buy:.1%}",
            "보유 주": f"{t.weeks_held:.0f}",
            "고점→매도 주": f"{t.weeks_peak_to_sell:.0f}",
        })
    return rows


def asset_report(asset: Asset, period: str) -> list[str]:
    start, end = asset.periods[period]
    lines = [f"# {asset.name}: {PERIOD_NAMES[period]} ({start} ~ {end})", "", *asset.notes,
             "샤프의 무위험 수익률은 세전 현금 수익률입니다. 'CAGR (청산 후)'는 기간 끝에 모두 팔고 세금까지 정산한 뒤의 값입니다.", ""]

    for label, tax in asset.taxes:
        rows = [row(f"{w}주 이동평균", run(asset, w, start, end, tax)[1]) for w in WINDOWS]
        rows.append(row("단순 보유", run(asset, None, start, end, tax)[1]))
        lines += [f"## 기본 결과: {label}", "", table(rows), ""]

    trades = {w: trades_for(asset, w, start, end) for w in WINDOWS}
    lines += [
        "## 매도 시 보유 중 고점 대비 하락 (요약)", "",
        table([{"전략": f"{w}주 이동평균", **drawdown_summary(trades[w])} for w in WINDOWS]), "",
        "- 고점은 그 매매의 매수 체결일부터 매도 체결일까지(양 끝 포함) 가장 높은 종가입니다. 하락률은 매도 체결일 종가 / 고점 − 1입니다.",
        "- 지수 기준은 배당 미포함 종가입니다. 보유 중 배당락이 있으면 그만큼 하락을 크게 잽니다.",
        "- 자산 기준은 세전 자산 가치(배당 재투자)의 보유 중 고점 대비입니다. 하락률에 남는 비용은 매도 비용뿐입니다"
        "(매수 비용은 고점과 매도 값에 함께 들어가 약분됩니다).",
        "- '매수가보다 높게 판 비율'은 지수 종가 기준이라 매매 비용과 배당이 빠져 있습니다. 하락 구간은 반올림 전 값으로 셉니다"
        "(예: -5.05%는 '-5.0%'로 찍히지만 5~10% 구간입니다).",
        "- 기간 끝까지 들고 있던 보유 구간은 매도가 아니므로 요약에서 뺍니다. 매수일이 시작 전날이면 평가 시작 때 이미 보유한 구간입니다.", "",
    ]
    for w in WINDOWS:
        lines += [f"## 매매 목록: {w}주 이동평균 (세전)", "", table(trade_rows(trades[w])), ""]
    return lines


# ---------------------------------------------------------------- 차트


def chart(asset: Asset, period: str) -> list[Path]:
    """기본 설정(세전)의 보유 구간, 이격도, 전략과 단순 보유의 자산 가치를 그립니다. 20주와 40주를 따로 그립니다.
    지표는 신호와 같은 배당 미포함 가격으로 계산하되, 눈금이 가격 단위로 나오도록 호가 단위로 바꾸기 전의 가격을 씁니다.
    """
    start, end = asset.periods[period]
    pre_tax = asset.taxes[0][1]
    weeks = period_last_days(asset.price.index, WEEK)
    weekly = asset.price.reindex(weeks)
    hold_result, hold, _ = run(asset, None, start, end, pre_tax)
    paths = []
    for window in WINDOWS:
        average = weekly.rolling(window).mean()
        result, m, _ = run(asset, window, start, end, pre_tax)
        sold = trades_for(asset, window, start, end)
        sold = sold[~sold["open"]]
        paths.append(holding_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-{asset.key}-sma{window}w.png",
            title=f"{asset.name} {window}주 이동평균: 보유 구간 ({PERIOD_NAMES[period]} {start} ~ {end})",
            subtitle=[
                f"세전  ·  {metrics_line(m, hold)}",
                f"규칙: 주 마지막 거래일 종가가 {window}주 이동평균보다 높으면 보유, 아니면 현금. "
                "신호 낸 날 종가에 체결 (사용자 지정. 저장소 기본 규칙은 다음 거래일 종가 체결).",
                f"매도 시 보유 중 고점 대비 하락 (지수 종가): 평균 {sold['from_peak'].mean():.1%}, 중앙값 {sold['from_peak'].median():.1%}, "
                f"가장 큰 하락 {sold['from_peak'].min():.1%} ({len(sold)}회 매도)",
                f"{asset.chart_costs} 괄호 안은 같은 비용의 단순 보유. 음영은 매수 체결일 종가 ~ 매도 체결일 종가.",
            ],
            price=asset.price,
            price_label=asset.price_label,
            overlay=average,
            overlay_label=f"{window}주 이동평균 (주간 종가)",
            exposure=result.daily["holding"],
            panels=[Panel(f"이격도: 주간 종가 / {window}주 이동평균 - 1", weekly / average - 1, 0.0, "이동평균", percent=True)],
            equity=result.daily["equity"],
            hold_equity=hold_result.daily["equity"],
            strategy_label=f"{window}주 이동평균",
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

    lines = [f"# 20주·40주 이동평균 단독: {PERIOD_NAMES[args.period]}", "",
             "규칙: 주 마지막 거래일 종가가 N주 이동평균보다 높으면 보유, 아니면 현금 (N = 20, 40). 100% 주식 또는 100% 현금.",
             "**체결: 신호 낸 날 종가 (사용자 지정).** 저장소 기본 규칙(다음 거래일 종가 체결)과 다릅니다. "
             "종가를 보고 그 종가에 사는 것이라, 장 마감 직전 가격으로 판단해 종가 주문을 내는 방식의 근사입니다.",
             "비교 대상은 같은 기간, 같은 비용·세금의 단순 보유입니다. 파라미터 민감도는 사용자 요청으로 하지 않았습니다.", ""]
    for asset in assets:
        lines += ["", *[line.replace("# ", "## ", 1) if line.startswith("#") else line for line in asset_report(asset, args.period)]]

    out = ROOT / "reports" / EXPERIMENT / f"{args.period}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
