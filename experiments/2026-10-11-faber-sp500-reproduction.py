"""Faber 10개월 이동평균, S&P 500 단일 자산 재현. 연구 일지: journal/2026-10-11-faber-sp500-reproduction.md

규칙: 매월 마지막 거래일 종가가 10개월 이동평균(월말 종가)보다 높으면 보유, 아니면 현금. 100% 주식 또는 100% 현금.
- 논문 방식: 신호 낸 날(월말) 종가에 체결, 비용 없음, 월 수익률로 계산. 1단계 재현 판단은 이 결과로 합니다.
- 저장소 방식: 다음 거래일 종가에 체결, 편도 0.1%, 일별 엔진으로 계산.
수익률: S&P 500 월말 종가(Yahoo ^GSPC) + Shiller 배당(D/12), 현금은 Ken French RF(1개월 국채).

실행:
    python experiments/2026-10-11-faber-sp500-reproduction.py              # 재현 기간 (1929-01 ~ 2012-12)
    python experiments/2026-10-11-faber-sp500-reproduction.py --chart      # 보유 구간 차트만
    python experiments/2026-10-11-faber-sp500-reproduction.py --period oos # 표본 외 기간 (마지막에 한 번만)

결과 표는 reports/2026-10-11-faber-sp500-reproduction/<기간>.md,
차트는 <기간>-holdings-<체결 방식>.png 에 저장합니다.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import Costs, holding_from_signal, metrics, period_last_days, run_backtest, snap_to_tick  # noqa: E402
from src.charts import Panel, holding_chart, metrics_line  # noqa: E402
from src.data import PRICE_TICK, daily_dividend, daily_rf  # noqa: E402
from src.monthly import (  # noqa: E402
    month_end_returns,
    month_end_signal,
    monthly_metrics,
    monthly_series_signal,
    switch_months,
    timing_returns,
    total_return_signal_index,
)

EXPERIMENT = "2026-10-11-faber-sp500-reproduction"
PERIOD_NAMES = {"design": "재현 기간 (설계용)", "oos": "표본 외 기간"}
PERIODS = {"design": ("1929-01", "2012-12"), "oos": ("2013-01", "2026-06")}
SUBPERIODS = [("1929-01", "1933-12"), ("1934-01", "2012-12")]
WINDOW = 10
WINDOWS = [6, 8, 9, 10, 11, 12]
TRADE = 0.001

# 원문 부록 B (66~67쪽). (CAGR, 변동성, 샤프, MDD, 국채)
PAPER_DECADES = {
    1930: {"hold": (-0.0047, 0.3798, -0.03, -0.7984), "timing": (0.0315, 0.1697, 0.15, -0.3063), "tbill": 0.0064},
    1940: {"hold": (0.0899, 0.1611, 0.53, -0.2812), "timing": (0.0552, 0.1363, 0.37, -0.3474), "tbill": 0.0047},
    1950: {"hold": (0.1926, 0.1192, 1.44, -0.1505), "timing": (0.1740, 0.1142, 1.34, -0.1499), "tbill": 0.0205},
    1960: {"hold": (0.0776, 0.1211, 0.30, -0.2225), "timing": (0.0712, 0.0889, 0.34, -0.1279), "tbill": 0.0407},
    1970: {"hold": (0.0588, 0.1599, -0.04, -0.4264), "timing": (0.0840, 0.1085, 0.18, -0.1588), "tbill": 0.0650},
    1980: {"hold": (0.1755, 0.1639, 0.51, -0.2958), "timing": (0.1527, 0.1446, 0.42, -0.2326), "tbill": 0.0923},
    1990: {"hold": (0.1821, 0.1343, 0.98, -0.1537), "timing": (0.1309, 0.1204, 0.67, -0.1647), "tbill": 0.0498},
    2000: {"hold": (-0.0094, 0.1613, -0.23, -0.5095), "timing": (0.0773, 0.0827, 0.61, -0.0682), "tbill": 0.0272},
}
# 재현 판단 기준 (일지 "재현 판단 기준", 결과 계산 전에 정함)
HOLD_CAGR_TOLERANCE = 0.005
TIMING_CAGR_TOLERANCE = 0.010
TIMING_SHARPE_TOLERANCE = 0.10
# 원문 본문(25~26쪽)의 매도 시점과, 그 문장에 맞는 매도 신호 달. 월말 종가에 팔면 다음 달부터 현금입니다.
PAPER_SELLS = {
    "2000-09": "원문 \"2000년 10월에 매도\" (9월 말 종가에 팔면 10월부터 현금)",
    "2007-12": "원문 \"2007년 12월 31일 매도\"",
}


@dataclass(frozen=True)
class Inputs:
    daily_close: pd.Series  # ^GSPC 일별 종가 (배당 미포함)
    signal_price: pd.Series  # 신호용: 호가 단위(0.01) 개수로 바꾼 일별 종가
    month_end: pd.DataFrame  # 인덱스 월. close, dividend(Shiller D, 연간), total_return, rf
    tr_signal_index: pd.Series  # 총수익 지수 신호용 지수. 배당은 12개월 전 D만 씁니다.
    daily: pd.DataFrame  # 일별 엔진용: close, dividend(D/12를 그달 거래일 수로 나눔), rf


def load_inputs(period: str) -> Inputs:
    """data/의 표를 읽어 build_inputs에 넘깁니다."""
    daily_close = pd.read_csv(ROOT / "data" / "sp500_daily.csv", index_col=0, parse_dates=True)["close"]
    month_end = pd.read_csv(ROOT / "data" / "sp500_month_end.csv", index_col=0)
    month_end.index = pd.PeriodIndex(month_end.index, freq="M")
    shiller = pd.read_csv(ROOT / "data" / "sp500_monthly_avg.csv", index_col=0)
    shiller.index = pd.PeriodIndex(shiller.index, freq="M")
    return build_inputs(daily_close, month_end, shiller["dividend"], pd.Period(PERIODS[period][1], freq="M"))


def build_inputs(daily_close: pd.Series, month_end: pd.DataFrame, shiller_dividend: pd.Series, end: pd.Period) -> Inputs:
    """기간 끝(end월)까지만 남깁니다. 그 뒤 데이터는 신호, 수익률, 차트 어디에도 들어가지 않습니다.

    month_end: 인덱스 월, close, dividend(Shiller D), total_return, rf 열. shiller_dividend: Shiller D (총수익 지수 신호용,
    month_end보다 12개월 이상 앞에서 시작).
    """
    daily_close = daily_close[daily_close.index.to_period("M") <= end]
    month_end = month_end[month_end.index <= end]
    if month_end[["total_return", "rf"]].iloc[1:].isna().any(axis=None):
        raise ValueError("기간 안에 총수익률이나 현금 수익률이 빈 달이 있습니다.")
    tr_index = total_return_signal_index(month_end["close"], shiller_dividend[shiller_dividend.index <= end], 12)

    daily = pd.DataFrame({
        "close": daily_close,
        "dividend": daily_dividend(daily_close.index, month_end["dividend"]),
        "rf": daily_rf(daily_close.index, month_end["rf"], end + 1),
    })
    return Inputs(
        daily_close=daily_close,
        signal_price=snap_to_tick(daily_close, PRICE_TICK["^GSPC"]),
        month_end=month_end,
        tr_signal_index=tr_index,
        daily=daily,
    )


# ---------------------------------------------------------------- 실행


def signal_for(inputs: Inputs, window: int, source: str = "price") -> pd.Series:
    """월 신호 (인덱스 월). source: "price"(가격, 기본) 또는 "total_return"(총수익 지수, 민감도)."""
    if source == "price":
        return month_end_signal(inputs.signal_price, window)
    if source == "total_return":
        return monthly_series_signal(inputs.tr_signal_index, window)
    raise ValueError(source)


def months(start: str, end: str) -> slice:
    return slice(pd.Period(start, freq="M"), pd.Period(end, freq="M"))


def paper_run(inputs: Inputs, start: str, end: str, window: int | None = WINDOW, source: str = "price") -> dict:
    """논문 방식: 월말 종가 체결, 비용 없음. window가 None이면 단순 보유."""
    table = inputs.month_end
    if window is None:
        ret, rf = table["total_return"].loc[months(start, end)], table["rf"].loc[months(start, end)]
        return monthly_metrics(ret, rf, pd.Series(1.0, index=ret.index))
    out = timing_returns(signal_for(inputs, window, source), table["total_return"], table["rf"]).loc[months(start, end)]
    return monthly_metrics(out["ret"], out["rf"], out["held"])


def daily_holding(inputs: Inputs, window: int | None, lag: int, source: str = "price") -> pd.Series:
    if window is None:
        return pd.Series(1.0, index=inputs.daily.index)
    signal = signal_for(inputs, window, source)
    month_days = period_last_days(inputs.daily.index, "M")
    if not month_days.to_period("M").equals(signal.index):
        raise ValueError("월 신호와 월말 거래일이 맞지 않습니다.")
    return holding_from_signal(pd.Series(signal.to_numpy(), index=month_days), inputs.daily.index, lag)


def daily_run(inputs: Inputs, start: str, end: str, window: int | None, lag: int, trade: float, source: str = "price"):
    """일별 엔진. 지표는 월말 자산으로 계산하고, 매매 횟수와 보유 비율은 일별 결과에서 가져옵니다."""
    first_day = pd.Period(start, freq="M").start_time
    last_day = pd.Period(end, freq="M").end_time.normalize()
    holding = daily_holding(inputs, window, lag, source)
    result = run_backtest(inputs.daily, holding, str(first_day.date()), str(last_day.date()), Costs(trade=trade))
    ret = month_end_returns(result.daily["equity"], result.start_date)
    rf = inputs.month_end["rf"].loc[months(start, end)]
    if not ret.index.equals(rf.index):
        raise ValueError("일별 엔진의 월과 현금 수익률의 월이 다릅니다.")
    m = monthly_metrics(ret, rf)
    m["cagr_after_liquidation"] = result.final_after_liquidation ** (1 / m["years"]) - 1
    m["trades_per_year"] = result.trades / m["years"]
    m["time_in_market"] = (result.daily["holding"].iloc[1:] == 1.0).mean()
    return m, result


# ---------------------------------------------------------------- 표


def pct(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}%}"


def row(name: str, m: dict) -> dict:
    recovery = "미회복" if pd.isna(m["mdd_recovery_months"]) else f"{m['mdd_recovery_months']:.0f}개월"
    return {
        "구분": name,
        "CAGR": pct(m["cagr"]),
        "CAGR (청산 후)": pct(m["cagr_after_liquidation"]) if "cagr_after_liquidation" in m else pct(m["cagr"]),
        "변동성": pct(m["volatility"]),
        "샤프 (원문 방식)": f"{m['sharpe_paper']:.2f}",
        "샤프": f"{m['sharpe']:.2f}",
        "MDD": pct(m["mdd"], 1),
        "MDD 기간": f"{m['mdd_peak']} ~ {m['mdd_trough']}",
        "MDD 회복": recovery,
        "연간 매매": f"{m['trades_per_year']:.2f}",
        "보유 비율": f"{m['time_in_market']:.0%}",
        "수익 달 비율": f"{m['positive_months']:.1%}",
    }


def table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


SOURCE_TITLES = {
    "price": "가격 신호 (사전 계획의 판단 대상)",
    "total_return": "총수익 지수 신호 (결과를 본 뒤 추가한 원인 확인)",
}


def decade_section(inputs: Inputs, source: str = "price") -> list[str]:
    """원문 부록 B와 같은 10년 단위 비교, 재현 판단 기준 계산."""
    hold_rows, timing_rows, checks = [], [], []
    for decade, paper in PAPER_DECADES.items():
        start, end = f"{decade}-01", f"{decade + 9}-12"
        hold, timing = paper_run(inputs, start, end, None), paper_run(inputs, start, end, WINDOW, source)
        name = f"{decade}년대"
        for rows, ours, theirs in ((hold_rows, hold, paper["hold"]), (timing_rows, timing, paper["timing"])):
            rows.append({
                "10년": name,
                "CAGR 원문": pct(theirs[0]), "CAGR 재현": pct(ours["cagr"]), "차이": f"{(ours['cagr'] - theirs[0]) * 100:+.2f}%p",
                "변동성 원문": pct(theirs[1]), "변동성 재현": pct(ours["volatility"]),
                "샤프 원문": f"{theirs[2]:.2f}", "샤프 재현": f"{ours['sharpe_paper']:.2f}",
                "샤프 차이": f"{ours['sharpe_paper'] - theirs[2]:+.2f}",
                "MDD 원문": pct(theirs[3], 2), "MDD 재현": pct(ours["mdd"], 2),
            })
        paper_gap = paper["timing"][0] - paper["hold"][0]
        our_gap = timing["cagr"] - hold["cagr"]
        checks.append({
            "10년": name,
            "단순 보유 CAGR 0.5%p 이내": abs(hold["cagr"] - paper["hold"][0]) <= HOLD_CAGR_TOLERANCE,
            "타이밍 CAGR 1.0%p 이내": abs(timing["cagr"] - paper["timing"][0]) <= TIMING_CAGR_TOLERANCE,
            "타이밍 샤프 0.10 이내": abs(timing["sharpe_paper"] - paper["timing"][2]) <= TIMING_SHARPE_TOLERANCE,
            "타이밍 − 단순 보유 원문": f"{paper_gap * 100:+.2f}%p",
            "타이밍 − 단순 보유 재현": f"{our_gap * 100:+.2f}%p",
            "부호 같음": np.sign(paper_gap) == np.sign(our_gap),
            "현금 원문 (90일)": pct(paper["tbill"]),
            "현금 재현 (1개월)": pct(timing["rf_cagr"]),
        })

    check = pd.DataFrame(checks)
    count = {column: int(check[column].sum()) for column in
             ["단순 보유 CAGR 0.5%p 이내", "타이밍 CAGR 1.0%p 이내", "타이밍 샤프 0.10 이내", "부호 같음"]}
    data_ok = count["단순 보유 CAGR 0.5%p 이내"] == len(check)
    passed = count["타이밍 CAGR 1.0%p 이내"] >= 6 and count["타이밍 샤프 0.10 이내"] >= 6 and count["부호 같음"] >= 7
    # 사전 계획의 판단 대상은 가격 신호뿐입니다. 총수익 지수 신호는 결과를 본 뒤의 원인 확인이라 판단 문구를 쓰지 않습니다.
    if source == "price":
        verdict = f"- **재현 판단 (사전 기준, 위 세 가지 모두): {'통과' if passed else '미통과'}**"
    else:
        verdict = f"- 사후 확인 (판단에 쓰지 않음): 위 세 가지를 {'모두 충족' if passed else '충족하지 못함'}"
    shown = check.copy()
    for column in count:
        shown[column] = shown[column].map({True: "예", False: "아니오"})
    timing_title = f"타이밍 (10개월 이동평균, 보유 비율 {paper_run(inputs, '1929-01', '2012-12', WINDOW, source)['time_in_market']:.0%})"
    return [
        f"## 원문 부록 B와 비교: 10년 단위, 논문 방식, {SOURCE_TITLES[source]}", "",
        "각 10년의 월 수익률만으로 계산합니다. 낙폭은 그 10년의 시작부터 잽니다. 샤프는 원문 방식((CAGR − 현금 CAGR) / 변동성)입니다.", "",
        "### 단순 보유", "", table(hold_rows), "",
        f"### {timing_title}", "", table(timing_rows), "",
        "### 재현 판단 기준", "", table(shown.to_dict("records")), "",
        f"- 데이터 확인 (단순 보유 CAGR 0.5%p 이내, 8개 모두): {count['단순 보유 CAGR 0.5%p 이내']}/8 → {'충족' if data_ok else '미충족'}",
        f"- 타이밍 CAGR 1.0%p 이내 (6개 이상): {count['타이밍 CAGR 1.0%p 이내']}/8",
        f"- 타이밍 샤프 0.10 이내 (6개 이상): {count['타이밍 샤프 0.10 이내']}/8",
        f"- 타이밍 − 단순 보유 부호 같음 (7개 이상): {count['부호 같음']}/8",
        verdict, "",
    ]


def switch_section(inputs: Inputs, start: str, end: str, source: str = "price") -> list[str]:
    signal = signal_for(inputs, WINDOW, source)
    switches = switch_months(signal)
    inside = switches[(switches["month"] >= pd.Period(start, "M")) & (switches["month"] <= pd.Period(end, "M"))]
    sells = set(inside.loc[inside["action"] == "매도", "month"].astype(str))
    lines = [
        f"## 신호가 바뀐 달: {SOURCE_TITLES[source]}", "",
        "월말 종가로 신호를 낸 달입니다. 논문 방식은 그날 종가에, 저장소 방식은 다음 거래일 종가에 체결합니다.", "",
    ]
    # 원문 매도 시점은 그 달이 이 기간 안에 있을 때만 비교합니다. 기간 밖이면 "없음"이 원문과 다르다는 뜻으로 읽히기 때문입니다.
    in_period = {month: text for month, text in PAPER_SELLS.items() if start <= month <= end}
    if in_period:
        lines += ["**원문 본문의 매도 시점과 비교 (참고):** " + "; ".join(
            f"{text} → {month} 매도 신호 {'있음' if month in sells else '없음'}" for month, text in in_period.items()), ""]
    if pd.Period(start, "M") < pd.Period("1990-01", "M"):
        recent = inside[inside["month"] >= pd.Period("1990-01", "M")]
        lines += [f"### 1990-01 ~ {end}", "",
                  ", ".join(f"{m} {a}" for m, a in zip(recent["month"].astype(str), recent["action"])), ""]
    lines += [
        f"### 전체 ({start} ~ {end}, {len(inside)}회)", "",
        ", ".join(f"{m} {a}" for m, a in zip(inside["month"].astype(str), inside["action"])), "",
    ]
    return lines


def report(inputs: Inputs, period: str) -> list[str]:
    start, end = PERIODS[period]
    paper, paper_hold = paper_run(inputs, start, end), paper_run(inputs, start, end, None)
    next_day, next_result = daily_run(inputs, start, end, WINDOW, lag=1, trade=TRADE)
    next_hold, hold_result = daily_run(inputs, start, end, None, lag=1, trade=TRADE)
    engine_check, _ = daily_run(inputs, start, end, WINDOW, lag=0, trade=0.0)
    engine_hold, _ = daily_run(inputs, start, end, None, lag=0, trade=0.0)

    lines = [
        f"# Faber 10개월 이동평균, S&P 500 재현: {PERIOD_NAMES[period]} ({start} ~ {end})", "",
        "규칙: 매월 마지막 거래일 종가가 10개월 이동평균(월말 종가)보다 높으면 보유, 아니면 현금. 신호는 가격(배당 미포함)으로 계산합니다.",
        "수익률: S&P 500 월말 종가(Yahoo ^GSPC) + Shiller 배당(D/12, 약 6개월 늦음), 현금은 Ken French RF(1개월 국채), 달러 기준, 세금 없음.",
        "- **논문 방식:** 신호 낸 날(월말) 종가 체결, 비용 없음, 월 수익률로 계산. 1단계 재현 판단은 이 결과로 합니다.",
        "- **다음 거래일 체결:** 저장소 기본 규칙. 일별 엔진, 편도 매매 비용 0.1%, 일별 배당은 D/12를 그달 거래일 수로 나눠 매일 더합니다. "
        "단순 보유도 같은 비용(시작할 때 매수 1회)입니다.",
        "- **일별 엔진 확인:** 논문 방식을 일별 엔진으로 계산한 값(신호 낸 날 종가, 비용 없음). 배당을 나눠 받는 방식만 달라 논문 방식과 거의 같아야 합니다.",
        "- 지표는 모두 월말 값으로 계산합니다. 변동성 = 월 수익률 표준편차 × √12. "
        "샤프 (원문 방식) = (CAGR − 현금 CAGR) / 변동성, 샤프 = 월 초과수익률 평균 × 12 / (월 초과수익률 표준편차 × √12).",
        "- 연간 매매는 매수와 매도를 각각 한 번으로 셉니다(왕복 1회 = 2회). 'CAGR (청산 후)'는 기간 끝에 팔 때의 매매 비용을 뺀 값입니다.", "",
        "## 기본 결과", "",
        table([
            row("타이밍: 논문 방식", paper),
            row("단순 보유: 논문 방식", paper_hold),
            row("타이밍: 다음 거래일 체결, 편도 0.1%", next_day),
            row("단순 보유: 같은 비용", next_hold),
            row("일별 엔진 확인: 타이밍 (신호 낸 날, 비용 없음)", engine_check),
            row("일별 엔진 확인: 단순 보유 (비용 없음)", engine_hold),
        ]), "",
    ]
    daily_strategy, daily_hold = metrics(next_result), metrics(hold_result)
    lines += [
        "### 참고: 다음 거래일 체결의 일별 기준 지표", "",
        "이전 실험(SPY 등)과 같은 일별 기준입니다. 샤프는 일별 초과수익률, MDD는 일별 종가 기준입니다.", "",
        table([
            {"구분": name, "CAGR": pct(m["cagr"]), "변동성": pct(m["volatility"]), "샤프": f"{m['sharpe']:.2f}",
             "MDD": pct(m["mdd"], 1), "MDD 기간": f"{m['mdd_peak']:%Y-%m-%d} ~ {m['mdd_trough']:%Y-%m-%d}"}
            for name, m in (("타이밍 (다음 거래일, 0.1%)", daily_strategy), ("단순 보유 (같은 비용)", daily_hold))
        ]), "",
    ]

    if period == "design":
        lines += decade_section(inputs, "price")
        lines += [
            "가격 신호가 기준을 넘지 못한 뒤, 원인을 찾으려고 아래를 추가했습니다(일지 \"결과를 본 뒤 바꾼 것\"). "
            "원문은 \"All data series are total return series\"라고만 적었고, 신호를 무엇으로 계산했는지 밝히지 않았습니다. "
            "총수익 지수의 배당은 12개월 전 D입니다.", "",
        ]
        lines += decade_section(inputs, "total_return")

    sub_rows = []
    for sub_start, sub_end in SUBPERIODS:
        if pd.Period(sub_start, "M") < pd.Period(start, "M") or pd.Period(sub_end, "M") > pd.Period(end, "M"):
            continue
        label = f"{sub_start} ~ {sub_end}"
        sub_rows += [
            row(f"{label} 타이밍: 논문 방식", paper_run(inputs, sub_start, sub_end)),
            row(f"{label} 단순 보유: 논문 방식", paper_run(inputs, sub_start, sub_end, None)),
            row(f"{label} 타이밍: 다음 거래일, 0.1%", daily_run(inputs, sub_start, sub_end, WINDOW, 1, TRADE)[0]),
            row(f"{label} 단순 보유: 같은 비용", daily_run(inputs, sub_start, sub_end, None, 1, TRADE)[0]),
        ]
    if sub_rows:
        lines += ["## 하위 기간", "", "1929~1933년은 월 수익률의 자기상관이 유난히 커서 따로 봅니다.", "", table(sub_rows), ""]

    sens = []
    for window in WINDOWS:
        sens.append(row(f"{window}개월: 논문 방식", paper_run(inputs, start, end, window)))
        sens.append(row(f"{window}개월: 다음 거래일, 0.1%", daily_run(inputs, start, end, window, 1, TRADE)[0]))
    sens.append(row("(참고) 10개월 총수익 지수 신호: 논문 방식", paper_run(inputs, start, end, WINDOW, "total_return")))
    sens.append(row("(참고) 10개월 총수익 지수 신호: 다음 거래일, 0.1%",
                    daily_run(inputs, start, end, WINDOW, 1, TRADE, "total_return")[0]))
    lines += [
        "## 파라미터를 바꿨을 때", "",
        "이동평균 기간(개월)과 신호 가격을 바꿉니다. 총수익 지수 신호는 배당으로 12개월 전 D를 씁니다"
        "(Shiller D는 직선 보간이라 최근 값에 앞으로의 값이 섞여 있습니다).", "",
        f"비교: 단순 보유 논문 방식 CAGR {pct(paper_hold['cagr'])}, 샤프 (원문 방식) {paper_hold['sharpe_paper']:.2f}, "
        f"MDD {pct(paper_hold['mdd'], 1)}. 같은 비용의 단순 보유 CAGR {pct(next_hold['cagr'])}.", "",
        table(sens), "",
    ]
    lines += switch_section(inputs, start, end, "price")
    lines += switch_section(inputs, start, end, "total_return")
    return lines


# ---------------------------------------------------------------- 차트


def chart(inputs: Inputs, period: str) -> list[Path]:
    """가격 신호의 논문 방식(신호 낸 날, 비용 없음)과 다음 거래일 체결(편도 0.1%), 총수익 지수 신호의 논문 방식을
    따로 그립니다. 모두 일별 엔진 결과입니다."""
    start, end = PERIODS[period]
    month_days = period_last_days(inputs.daily_close.index, "M")
    monthly_close = inputs.daily_close.reindex(month_days)
    average = monthly_close.rolling(WINDOW).mean()
    tr_index = pd.Series(inputs.tr_signal_index.reindex(month_days.to_period("M")).to_numpy(), index=month_days)
    tr_gap = tr_index / tr_index.rolling(WINDOW).mean() - 1
    price_rule = "매월 마지막 거래일 종가가 10개월 이동평균(월말 종가)보다 높으면 보유, 아니면 현금"
    tr_rule = "매월 말 총수익 지수(배당은 12개월 전 D)가 그 10개월 이동평균보다 높으면 보유, 아니면 현금"
    price_panel = Panel("이격도: 월말 종가 / 10개월 이동평균 - 1", monthly_close / average - 1, 0.0, "이동평균", percent=True)
    tr_panel = Panel("신호 지표: 월말 총수익 지수 / 그 10개월 이동평균 - 1", tr_gap, 0.0, "이동평균", percent=True)
    # (파일 이름, 이름, 신호, 체결 지연, 매매 비용, 체결 설명, 비용 설명, 규칙, 이동평균 겹쳐 그리기, 지표 칸)
    settings = [
        ("paper", "논문 방식", "price", 0, 0.0, "신호 낸 날(월말) 종가에 체결 (원문 방식)", "비용 없음", price_rule, True, price_panel),
        ("next-day", "다음 거래일 체결", "price", 1, TRADE, "다음 거래일 종가에 체결 (저장소 기본 규칙)", "편도 매매 비용 0.1%",
         price_rule, True, price_panel),
        ("paper-total-return", "총수익 지수 신호, 논문 방식", "total_return", 0, 0.0,
         "신호 낸 날(월말) 종가에 체결 (원문 방식). 결과를 본 뒤 추가한 원인 확인용", "비용 없음", tr_rule, False, tr_panel),
    ]
    paths = []
    for key, name, source, lag, trade, execution, cost, rule, show_average, panel in settings:
        m, result = daily_run(inputs, start, end, WINDOW, lag, trade, source)
        hold, hold_result = daily_run(inputs, start, end, None, lag, trade)
        paths.append(holding_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-{key}.png",
            title=f"S&P 500 10개월 이동평균 ({name}): 보유 구간, {PERIOD_NAMES[period]} {start} ~ {end}",
            subtitle=[
                f"세금 없음, 월말 값 기준  ·  {metrics_line(m, hold)}",
                f"규칙: {rule}. {execution}.",
                f"비용: {cost}. 배당은 Shiller D/12를 그달 거래일에 나눠 받음, 현금은 Ken French RF(1개월 국채), 달러 기준. "
                "괄호 안은 같은 비용의 단순 보유. 음영은 매수 체결일 종가 ~ 매도 체결일 종가.",
            ],
            price=inputs.daily_close,
            price_label="S&P 500 일별 종가 (^GSPC, 배당 미포함, 로그 눈금)",
            overlay=average if show_average else None,
            overlay_label="10개월 이동평균 (월말 종가)" if show_average else "",
            exposure=result.daily["holding"],
            panels=[panel],
            equity=result.daily["equity"],
            hold_equity=hold_result.daily["equity"],
            strategy_label=f"10개월 이동평균 ({name})",
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIOD_NAMES, default="design")
    parser.add_argument("--chart", action="store_true", help="보유 구간 차트만 만듭니다")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    inputs = load_inputs(args.period)

    if args.chart:
        for path in chart(inputs, args.period):
            print(f"저장: {path.relative_to(ROOT)}")
        return

    lines = report(inputs, args.period)
    out = ROOT / "reports" / EXPERIMENT / f"{args.period}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
