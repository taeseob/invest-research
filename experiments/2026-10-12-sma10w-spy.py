"""10주 이동평균, SPY 단일 자산. 연구 일지: journal/2026-10-12-sma10w-spy.md

실행:
    python experiments/2026-10-12-sma10w-spy.py              # 설계용 기간
    python experiments/2026-10-12-sma10w-spy.py --period oos # 표본 외 기간 (마지막에 한 번만)

결과 표는 reports/2026-10-12-sma10w-spy/<기간>.md 에 저장합니다.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import Costs, holding_from_signal, metrics, moving_average_signal, period_last_days, run_backtest  # noqa: E402

PERIODS = {
    "design": ("1994-01-03", "2012-12-31"),
    "oos": ("2013-01-02", "2026-08-31"),
}
BASE = {"window": 10, "lag": 1, "trade": 0.001, "signal_price": "close"}
PRE_TAX = {"dividend_withholding": 0.0, "capital_gains_tax": 0.0}
AFTER_TAX = {"dividend_withholding": 0.15, "capital_gains_tax": 0.22}
# 사전 계획에 없던 가정. 검증에서 현금 이자 과세가 빠졌다는 지적을 받고 별도 행으로 추가했습니다.
AFTER_TAX_INTEREST = {**AFTER_TAX, "interest_tax": 0.154}


def load_spy() -> pd.DataFrame:
    data = pd.read_csv(ROOT / "data" / "spy_daily.csv", index_col=0, parse_dates=True)
    # 신호용 총수익 기준 가격 (배당락일에 알려진 배당만 씀)
    data["tr_index"] = (1 + data["total_return"].fillna(0.0)).cumprod()
    return data


def run(data, start, end, window, lag, trade, signal_price, tax, buy_and_hold=False):
    costs = Costs(trade=trade, **tax)
    if buy_and_hold:
        holding = pd.Series(1.0, index=data.index)
    else:
        weeks = period_last_days(data.index, "W-FRI")
        signal = moving_average_signal(data[signal_price], weeks, window)
        holding = holding_from_signal(signal, data.index, lag)
    result = run_backtest(data, holding, start, end, costs)
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIODS, default="design")
    period = parser.parse_args().period
    start, end = PERIODS[period]
    data = load_spy()

    sections = [f"# 10주 이동평균, SPY: {period} ({start} ~ {end})", ""]
    sections += [
        "기본 가정: 다음 거래일 종가 체결, 편도 매매 비용 0.1%, 현금은 Ken French RF, 달러 기준.",
        "세후: 배당 원천징수 15%, 양도소득세 22%(연간 실현 순이익, 기본공제 없음). "
        "'CAGR (청산 후)'는 기간 끝에 모두 팔고 마지막 해 세금까지 정산한 뒤의 값입니다.",
        "나머지 지표(CAGR, 변동성, 샤프, MDD)는 이 정산 전 일별 자산으로 계산합니다. "
        "그래서 세후 표에서 단순 보유는 아직 내지 않은 세금(이연 세금)이 빠진 값이고, 전략은 매년 낸 세금이 들어간 값입니다.",
        "",
    ]

    basic = [("세전 (매매 비용만)", PRE_TAX), ("세후", AFTER_TAX), ("세후 + 현금 이자소득세 15.4% (사후 추가)", AFTER_TAX_INTEREST)]
    for label, tax in basic:
        rows = []
        _, m = run(data, start, end, **BASE, tax=tax)
        rows.append(row("10주 이동평균", m))
        _, m = run(data, start, end, **BASE, tax=tax, buy_and_hold=True)
        rows.append(row("단순 보유", m))
        sections += [f"## 기본 결과: {label}", "", table(rows), ""]

    for label, tax in [("세전", PRE_TAX), ("세후", AFTER_TAX)]:
        rows = []
        for window in [5, 8, 10, 12, 15, 20, 40]:
            _, m = run(data, start, end, **{**BASE, "window": window}, tax=tax)
            rows.append(row(f"{window}주", m))
        sections += [f"## 민감도: 이동평균 기간 ({label})", "", table(rows), ""]

    rows = []
    for trade in [0.0005, 0.001, 0.002]:
        _, m = run(data, start, end, **{**BASE, "trade": trade}, tax=PRE_TAX)
        rows.append(row(f"10주, 편도 {trade:.2%}", m))
        _, m = run(data, start, end, **{**BASE, "trade": trade}, tax=PRE_TAX, buy_and_hold=True)
        rows.append(row(f"단순 보유, 편도 {trade:.2%}", m))
    sections += ["## 민감도: 매매 비용 (세전)", "", table(rows), ""]

    rows = []
    _, m = run(data, start, end, **{**BASE, "signal_price": "tr_index"}, tax=PRE_TAX)
    rows.append(row("10주, 총수익 기준 신호", m))
    _, m = run(data, start, end, **{**BASE, "lag": 0}, tax=PRE_TAX)
    rows.append(row("10주, 신호 낸 날 종가 체결 (참고값)", m))
    sections += ["## 민감도: 신호 가격, 체결 시점 (세전)", "", table(rows), ""]

    strategy, _ = run(data, start, end, **BASE, tax=PRE_TAX)
    hold, _ = run(data, start, end, **BASE, tax=PRE_TAX, buy_and_hold=True)
    # 첫 행(시작 전날, 자산 1.0 기준일)은 빼고, 첫해 수익률은 1.0 대비로 계산합니다.
    year_end = pd.DataFrame({
        name: r.daily["equity"].iloc[1:].groupby(r.daily.index[1:].year).last()
        for name, r in [("10주 이동평균", strategy), ("단순 보유", hold)]
    })
    yearly = year_end / year_end.shift(1).fillna(1.0) - 1
    yearly["차이"] = yearly["10주 이동평균"] - yearly["단순 보유"]
    yearly_rows = [{"연도": y, **{c: f"{v:.1%}" for c, v in r.items()}} for y, r in yearly.iterrows()]
    sections += ["## 연도별 수익률 (세전)", "", table(yearly_rows), ""]

    out = ROOT / "reports" / "2026-10-12-sma10w-spy" / f"{period}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(sections), encoding="utf-8")
    print("\n".join(sections))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
