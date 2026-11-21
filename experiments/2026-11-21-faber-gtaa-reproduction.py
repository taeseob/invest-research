"""Faber GTAA(5자산 10개월 이동평균) 근사 재현. 연구 일지: journal/2026-11-21-faber-gtaa-reproduction.md

규칙: 자산마다 월말 지수가 10개월 이동평균보다 높으면 그 자산 20%, 아니면 그 20%를 현금. 매달 20%씩 다시 맞춥니다.
체결은 논문 방식(월말 종가, 같은 날)뿐입니다. 대체 자료 일부가 월간 값이라 다음 거래일 체결은 계산할 수 없습니다.
데이터: data/gtaa_monthly.csv (python -m src.data --gtaa). 원자재와 EAFE는 대체 자료입니다 (일지 "데이터").

실행:
    python experiments/2026-11-21-faber-gtaa-reproduction.py --data-check   # 데이터 점검 (자산별 단순 보유, 전략 없음)
    python experiments/2026-11-21-faber-gtaa-reproduction.py                # 재현 기간 (1976-01 ~ 2012-12)
    python experiments/2026-11-21-faber-gtaa-reproduction.py --chart        # 보유 구간 차트만
    python experiments/2026-11-21-faber-gtaa-reproduction.py --period oos   # 표본 외 기간 (마지막에 한 번만)

결과는 reports/2026-11-21-faber-gtaa-reproduction/ 에 저장합니다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.charts import Panel, holding_chart, metrics_line, portfolio_chart  # noqa: E402
from src.monthly import (  # noqa: E402
    equal_weight_portfolio,
    held_from_signals,
    level_from_returns,
    monthly_metrics,
    signals_from_returns,
)

EXPERIMENT = "2026-11-21-faber-gtaa-reproduction"
PERIOD_NAMES = {"design": "재현 기간 (설계용)", "oos": "표본 외 기간"}
PERIODS = {"design": ("1976-01", "2012-12"), "oos": ("2013-01", "2025-05")}
DATA_CHECK = ("1973-01", "2012-12")
ASSETS = {
    "us_stock": "미국 주식 (S&P 500)",
    "eafe": "선진국 주식 (EAFE 대체: Ken French)",
    "bond": "미국 10년 국채 (합성)",
    "commodity": "원자재 (AQR 동일 비중)",
    "reit": "리츠 (NAREIT All REITs)",
}
WITHOUT_COMMODITY = ["us_stock", "eafe", "bond", "reit"]
SOURCES = {"total_return": "총수익 지수 신호", "price": "가격 신호"}
# 신호를 계산하는 수익률 열. S&P 500의 총수익 지수 신호는 12개월 전 Shiller D를 씁니다(us_stock_sig_tr, 보간된 그달 D에는
# 앞으로의 배당이 섞여 있음). 다른 자산의 총수익률은 그달까지 실제로 알려진 값입니다.
SIGNAL_COLUMNS = {
    "total_return": {"us_stock": "us_stock_sig_tr", "eafe": "eafe_tr", "bond": "bond_tr", "commodity": "commodity_tr",
                     "reit": "reit_tr"},
    "price": {a: f"{a}_pr" for a in ASSETS},
}
SHARPE_LABEL = "샤프(저장소 방식)"
WINDOW = 10
WINDOWS = [3, 6, 9, 10, 12]
TRADE = 0.001

# 원문 Figure 4 (17쪽), 1973~2012년: (CAGR, 변동성, 샤프(5.41%), MDD)
PAPER_ASSETS = {
    "rf": (0.0541, 0.0095, 0.00, 0.0),
    "us_stock": (0.0970, 0.1569, 0.27, -0.5095),
    "eafe": (0.0917, 0.1761, 0.21, -0.5640),
    "bond": (0.0818, 0.0844, 0.33, -0.1579),
    "commodity": (0.0832, 0.2055, 0.14, -0.6765),
    "reit": (0.0965, 0.1813, 0.23, -0.6788),
}
# 원문 Figure 14 (32쪽): 연도 → (단순 보유, GTAA)
PAPER_YEARLY = {
    1973: (1.01, 7.34), 1974: (-11.80, 11.92), 1975: (20.18, 1.46), 1976: (14.93, 15.88), 1977: (8.25, 7.32),
    1978: (13.74, 11.98), 1979: (17.99, 14.79), 1980: (19.18, 11.91), 1981: (-3.12, 3.52), 1982: (20.46, 21.03),
    1983: (18.24, 16.33), 1984: (9.32, 6.52), 1985: (26.03, 25.65), 1986: (25.25, 21.28), 1987: (8.68, 11.65),
    1988: (18.49, 11.76), 1989: (19.10, 17.96), 1990: (-1.08, 4.92), 1991: (18.07, 6.20), 1992: (3.90, 4.74),
    1993: (11.89, 12.79), 1994: (1.76, 2.45), 1995: (22.75, 21.74), 1996: (19.31, 19.25), 1997: (9.96, 9.94),
    1998: (-0.49, 7.34), 1999: (14.16, 13.06), 2000: (12.73, 13.82), 2001: (-9.73, 3.15), 2002: (1.99, 3.28),
    2003: (25.88, 20.45), 2004: (17.45, 15.06), 2005: (11.73, 7.63), 2006: (12.04, 14.21), 2007: (7.96, 9.80),
    2008: (-30.01, -0.59), 2009: (18.74, 12.77), 2010: (14.04, 3.65), 2011: (2.81, 2.49), 2012: (11.57, 1.26),
}
# 원문 Figure 18 (40쪽), 1973~2012년 이동평균 위 비율
PAPER_ABOVE = {"us_stock": 0.7292, "eafe": 0.6917, "bond": 0.7583, "commodity": 0.6521, "reit": 0.7292}
# 원문 Figure 16 (34쪽), 2006~2012년: (CAGR, 변동성, 샤프, MDD)
PAPER_2006 = {"hold": (0.0394, 0.1496, 0.16, -0.4600), "gtaa": (0.0601, 0.0727, 0.61, -0.0942)}
# 재현 판단 기준 (일지, 결과 계산 전에 정함)
DATA_TOLERANCE = 0.005
CAGR_TOLERANCE = 0.010
YEARLY_CORRELATION = 0.90


# ---------------------------------------------------------------- 데이터


def load(period_end: str) -> pd.DataFrame:
    """data/gtaa_monthly.csv를 읽어 기간 끝까지만 남깁니다."""
    table = pd.read_csv(ROOT / "data" / "gtaa_monthly.csv", index_col=0)
    table.index = pd.PeriodIndex(table.index, freq="M", name="month")
    return cut(table, period_end)


def cut(table: pd.DataFrame, period_end: str) -> pd.DataFrame:
    """기간 끝까지만 남깁니다. 그 뒤 데이터는 신호, 수익률, 차트 어디에도 들어가지 않습니다."""
    return table[table.index <= pd.Period(period_end, freq="M")]


def months(start: str, end: str) -> slice:
    return slice(pd.Period(start, freq="M"), pd.Period(end, freq="M"))


def paper_cagr(first: int, last: int, column: int) -> float:
    """원문 Figure 14의 연도별 수익률을 곱해 기간 CAGR을 냅니다 (column 0: 단순 보유, 1: GTAA)."""
    r = np.array([PAPER_YEARLY[y][column] for y in range(first, last + 1)]) / 100
    return float(np.prod(1 + r) ** (1 / len(r)) - 1)


# ---------------------------------------------------------------- 실행


def signals(table: pd.DataFrame, assets: list[str], source: str, window: int) -> pd.DataFrame:
    """자산마다 t월 말 신호 (1 보유, 0 현금). 지수는 수익률을 곱해 만들고, 그 이동평균과 비교합니다."""
    columns = {a: SIGNAL_COLUMNS[source][a] for a in assets}
    frame = table[list(columns.values())].copy()
    frame.columns = assets
    return signals_from_returns(frame, window)


def run(table: pd.DataFrame, start: str, end: str, assets: list[str], source: str | None = "total_return",
        window: int = WINDOW, cost: float = 0.0):
    """source가 None이면 단순 보유(모두 1)입니다. 반환: 포트폴리오 표, 지표, 보유 상태."""
    returns = table[[f"{a}_tr" for a in assets]].loc[months(start, end)]
    returns.columns = assets
    rf = table["rf"].loc[months(start, end)]
    if source is None:
        held = pd.DataFrame(1.0, index=returns.index, columns=assets)
    else:
        held = held_from_signals(signals(table, assets, source, window)).loc[months(start, end)]
        if held.isna().any(axis=None):
            raise ValueError(f"평가 기간에 신호가 없는 달이 있습니다: {held[held.isna().any(axis=1)].index[0]}")
    out = equal_weight_portfolio(returns, held, rf, cost)
    m = monthly_metrics(out["ret"], rf)
    values = held.to_numpy()
    switches = int((values[0] == 1.0).sum() + (values[1:] != values[:-1]).sum())
    m["time_in_market"] = out["invested"].mean()
    m["trades_per_year"] = switches / m["years"]
    m["turnover_per_year"] = out["turnover"].sum() / m["years"]
    return out, m, held


def yearly(ret: pd.Series) -> pd.Series:
    return (1 + ret).groupby(ret.index.year).prod() - 1


# ---------------------------------------------------------------- 표


def pct(value: float, digits: int = 2) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}%}"


def table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def row(name: str, m: dict) -> dict:
    recovery = "미회복" if pd.isna(m["mdd_recovery_months"]) else f"{m['mdd_recovery_months']:.0f}개월"
    return {
        "구분": name,
        "CAGR": pct(m["cagr"]),
        "변동성": pct(m["volatility"]),
        "샤프 (원문 방식)": f"{m['sharpe_paper']:.2f}",
        "샤프": f"{m['sharpe']:.2f}",
        "MDD": pct(m["mdd"], 1),
        "MDD 기간": f"{m['mdd_peak']} ~ {m['mdd_trough']}",
        "MDD 회복": recovery,
        "투자 비율": f"{m.get('time_in_market', 1.0):.0%}",
        "연간 매매": f"{m.get('trades_per_year', 0):.1f}",
        "연간 회전율": f"{m.get('turnover_per_year', 0):.0%}",
    }


def asset_stats(r: pd.Series, rf: pd.Series) -> dict:
    m = monthly_metrics(r, rf)
    m["sharpe_541"] = (m["cagr"] - 0.0541) / m["volatility"]
    return m


def data_check() -> list[str]:
    """자산별 단순 보유를 원문 Figure 4와 비교합니다. 전략(신호)은 계산하지 않습니다."""
    start, end = DATA_CHECK
    data = load(end)
    rf = data["rf"].loc[months(start, end)]
    candidates = [
        ("rf", "현금 (Ken French RF, 1개월 국채)", "rf"),
        ("us_stock", "S&P 500", "us_stock_tr"),
        ("bond", "미국 10년 국채 (H.15 금리로 합성)", "bond_tr"),
        ("reit", "NAREIT All REITs (기본)", "reit_tr"),
        ("reit", "NAREIT All Equity REITs (비교)", "reit_all_equity_tr"),
        ("reit", "NAREIT Equity REITs (비교)", "reit_equity_tr"),
        ("commodity", "원자재: AQR 동일 비중 + 현금 (구성이 달라 기준 아님)", "commodity_tr"),
    ]
    rows, checks = [], []
    for key, name, column in candidates:
        m = asset_stats(data[column].loc[months(start, end)], rf)
        paper = PAPER_ASSETS[key]
        rows.append({
            "자산": name, "CAGR 재현": pct(m["cagr"]), "CAGR 원문": pct(paper[0]),
            "차이": f"{(m['cagr'] - paper[0]) * 100:+.2f}%p",
            "변동성 재현": pct(m["volatility"]), "변동성 원문": pct(paper[1]),
            "샤프(5.41%) 재현": f"{m['sharpe_541']:.2f}", "샤프 원문": f"{paper[2]:.2f}",
            "MDD 재현": pct(m["mdd"]), "MDD 원문": pct(paper[3]),
        })
        if column in ("us_stock_tr", "bond_tr", "reit_tr"):
            checks.append(f"- {name}: CAGR 차이 {(m['cagr'] - paper[0]) * 100:+.2f}%p → "
                          f"{'기준 안 (0.5%p 이내)' if abs(m['cagr'] - paper[0]) <= DATA_TOLERANCE else '기준 밖'}")

    eafe = data["eafe_tr"].loc[months("1975-01", end)]
    m_eafe = asset_stats(eafe, data["rf"].loc[eafe.index])
    efa = pd.read_csv(ROOT / "data" / "efa_monthly.csv", index_col=0)
    efa.index = pd.PeriodIndex(efa.index, freq="M")
    efa = efa.loc[months("2001-09", end)].dropna()
    tracking = efa["eafe_proxy_tr"] - efa["efa_tr"]
    efa_years = (1 + efa).groupby(efa.index.year).prod() - 1
    return [
        "# GTAA 근사 재현: 데이터 점검 (자산별 단순 보유, 전략 없음)", "",
        f"기간: {start} ~ {end} (원문 Figure 4와 같은 기간). 월 수익률 기준, 비용 없음. 샤프(5.41%)는 원문처럼 (CAGR − 5.41%) / 변동성입니다.", "",
        table(rows), "",
        "### 데이터 확인 기준 (S&P 500, 10년 국채, 리츠 기본 지수가 0.5%p 이내)", "", *checks, "",
        "### EAFE 대체 자료 (Ken French 해외 지수 전체, EAFE + 캐나다 비중)", "",
        f"- 1975-01 ~ {end}: CAGR {pct(m_eafe['cagr'])}, 변동성 {pct(m_eafe['volatility'])}, MDD {pct(m_eafe['mdd'])}. "
        "원문 Figure 4(1973~2012년 9.17%, 17.61%, -56.40%)와 기간이 달라 기준으로 쓰지 않습니다.",
        f"- 실제 EAFE ETF(EFA, Yahoo 수정 종가)와 비교, {efa.index[0]} ~ {efa.index[-1]}: 월 수익률 상관 "
        f"{efa['eafe_proxy_tr'].corr(efa['efa_tr']):.3f}, 월 차이의 표준편차 {tracking.std():.2%}, "
        f"연평균 차이 {tracking.mean() * 12:+.2%}p (대체 자료 − EFA). EFA 수익률은 운용보수(연 0.3~0.4%)를 뺀 값이고, "
        "대체 자료에는 보수가 없습니다.",
        "", "| 연도 | 대체 자료 | EFA | 차이 |", "|---|---|---|---|",
        *[f"| {y} | {pct(r['eafe_proxy_tr'], 1)} | {pct(r['efa_tr'], 1)} | {(r['eafe_proxy_tr'] - r['efa_tr']) * 100:+.1f}%p |"
          for y, r in efa_years.iterrows()],
        "",
    ]


def comparison_section(table_: pd.DataFrame, start: str, end: str) -> list[str]:
    """원문 Figure 14와의 비교와 재현 판단 기준."""
    first, last = int(start[:4]), int(end[:4])
    paper_hold, paper_gtaa = paper_cagr(first, last, 0), paper_cagr(first, last, 1)
    hold = run(table_, start, end, list(ASSETS), None)
    lines = ["## 원문 Figure 14와 비교", "",
             f"원문 {first}~{last}년 CAGR은 연도별 수익률을 곱해 냈습니다: 단순 보유 {pct(paper_hold)}, GTAA {pct(paper_gtaa)}.", ""]
    years = list(range(first, last + 1))
    paper_y = pd.DataFrame({"hold": [PAPER_YEARLY[y][0] / 100 for y in years], "gtaa": [PAPER_YEARLY[y][1] / 100 for y in years]},
                           index=years)
    ours_hold = yearly(hold[0]["ret"])
    results = {}
    for source in SOURCES:
        out, m, _ = run(table_, start, end, list(ASSETS), source)
        ours = yearly(out["ret"])
        corr = ours.corr(paper_y["gtaa"])
        checks = {
            "(가) GTAA CAGR 1.0%p 이내": abs(m["cagr"] - paper_gtaa) <= CAGR_TOLERANCE,
            "(나) 연도별 상관 0.90 이상": corr >= YEARLY_CORRELATION,
            "(다) 샤프가 단순 보유보다 높음": m["sharpe_paper"] > hold[1]["sharpe_paper"],
            "(다) MDD가 단순 보유의 절반 이하": abs(m["mdd"]) <= abs(hold[1]["mdd"]) / 2,
        }
        results[source] = (out, m, ours, corr, checks)
    for source, (out, m, ours, corr, checks) in results.items():
        label = "주 판단" if source == "total_return" else "같은 기준으로 적음, 주 판단 아님"
        passed = all(checks.values())
        lines += [
            f"### {SOURCES[source]} ({label})", "",
            f"- GTAA CAGR {pct(m['cagr'])} (원문 {pct(paper_gtaa)}, 차이 {(m['cagr'] - paper_gtaa) * 100:+.2f}%p), "
            f"연도별 상관 {corr:.3f}, 연도별 차이 절댓값 중앙값 {(ours - paper_y['gtaa']).abs().median() * 100:.1f}%p",
            f"- 같은 데이터의 단순 보유: 샤프(원문 방식) {hold[1]['sharpe_paper']:.2f}, MDD {pct(hold[1]['mdd'], 1)} / "
            f"GTAA 샤프 {m['sharpe_paper']:.2f}, MDD {pct(m['mdd'], 1)}",
            *[f"- {k}: {'충족' if v else '미충족'}" for k, v in checks.items()],
            f"- **{'근사 재현 판단 (사전 기준)' if source == 'total_return' else '같은 기준 (참고)'}: {'통과' if passed else '미통과'}**", "",
        ]
    lines += [
        "### 단순 보유 (참고)", "",
        f"- CAGR {pct(hold[1]['cagr'])} (원문 {pct(paper_hold)}, 차이 {(hold[1]['cagr'] - paper_hold) * 100:+.2f}%p), "
        f"연도별 상관 {ours_hold.corr(paper_y['hold']):.3f}", "",
        "### 연도별 수익률", "",
        "| 연도 | 단순 보유 원문 | 단순 보유 재현 | GTAA 원문 | GTAA 재현 (총수익 지수 신호) | 차이 | GTAA 재현 (가격 신호) |",
        "|---|---|---|---|---|---|---|",
    ]
    tr_years, pr_years = results["total_return"][2], results["price"][2]
    for y in years:
        lines.append(f"| {y} | {pct(paper_y.loc[y, 'hold'], 1)} | {pct(ours_hold[y], 1)} | {pct(paper_y.loc[y, 'gtaa'], 1)} | "
                     f"{pct(tr_years[y], 1)} | {(tr_years[y] - paper_y.loc[y, 'gtaa']) * 100:+.1f}%p | {pct(pr_years[y], 1)} |")
    return lines + [""]


def above_section(table_: pd.DataFrame) -> list[str]:
    """원문 Figure 18: 1973~2012년 자산별 이동평균 위 비율. 대체 자료가 아닌 세 자산만 원문과 비교합니다."""
    rows = []
    for a, name in ASSETS.items():
        r = {"자산": name, "원문": pct(PAPER_ABOVE[a], 1)}
        for source, label in SOURCES.items():
            s = signals(table_, [a], source, WINDOW)[a].loc[months(*DATA_CHECK)].dropna()
            r[label] = f"{s.mean():.1%} ({s.index[0]}~)"
        rows.append(r)
    return ["## 원문 Figure 18과 비교: 이동평균 위에 있던 비율 (1973~2012년, 참고)", "",
            "EAFE 대체 자료는 1975-09부터, 원자재는 구성이 달라 비교 기준이 아닙니다.", "", table(rows), ""]


def report(period: str) -> list[str]:
    start, end = PERIODS[period]
    table_ = load(end)
    all_assets = list(ASSETS)
    gtaa_tr, gtaa_pr = run(table_, start, end, all_assets, "total_return"), run(table_, start, end, all_assets, "price")
    hold = run(table_, start, end, all_assets, None)
    lines = [
        f"# Faber GTAA 근사 재현: {PERIOD_NAMES[period]} ({start} ~ {end})", "",
        "규칙: 자산마다 월말 지수가 10개월 이동평균보다 높으면 그 자산 20%, 아니면 그 20%를 현금(Ken French RF). 매달 20%씩 다시 맞춥니다.",
        "체결: 월말 종가, 같은 날(논문 방식). 대체 자료 일부가 월간 값이라 다음 거래일 체결은 계산할 수 없습니다. 기본은 비용과 세금이 없습니다.",
        "자산: " + ", ".join(ASSETS.values()) + ". 원자재와 EAFE는 대체 자료입니다(일지 \"데이터\").",
        "지표는 월말 값입니다. 샤프 (원문 방식) = (CAGR − 현금 CAGR) / 변동성. 연간 매매는 자산별 신호 전환과 첫 진입을 더한 수, "
        "연간 회전율은 매달 다시 맞추며 사고판 위험자산 비중의 합입니다.", "",
        "## 기본 결과", "",
        table([
            row("GTAA: 총수익 지수 신호 (주 판단)", gtaa_tr[1]),
            row("GTAA: 가격 신호", gtaa_pr[1]),
            row("단순 보유 (20%씩, 매달 맞춤)", hold[1]),
            row("(참고) GTAA 총수익 지수 신호, 편도 0.1%", run(table_, start, end, all_assets, "total_return", cost=TRADE)[1]),
            row("(참고) 단순 보유, 편도 0.1%", run(table_, start, end, all_assets, None, cost=TRADE)[1]),
            row("(참고) 4자산 GTAA (원자재 제외, 25%씩), 총수익 지수 신호", run(table_, start, end, WITHOUT_COMMODITY, "total_return")[1]),
            row("(참고) 4자산 단순 보유", run(table_, start, end, WITHOUT_COMMODITY, None)[1]),
        ]), "",
    ]
    if period == "design":
        lines += comparison_section(table_, start, end)
        lines += above_section(table_)
        sub = [row(f"2006~2012 {name}", run(table_, "2006-01", "2012-12", all_assets, source)[1])
               for source, name in [("total_return", "GTAA 총수익 지수 신호"), ("price", "GTAA 가격 신호"), (None, "단순 보유")]]
        lines += ["## 2006~2012년 (원문 Figure 16, 참고)", "",
                  f"원문: 단순 보유 {pct(PAPER_2006['hold'][0])} / {pct(PAPER_2006['hold'][1])} / {PAPER_2006['hold'][2]:.2f} / "
                  f"{pct(PAPER_2006['hold'][3], 1)}, GTAA {pct(PAPER_2006['gtaa'][0])} / {pct(PAPER_2006['gtaa'][1])} / "
                  f"{PAPER_2006['gtaa'][2]:.2f} / {pct(PAPER_2006['gtaa'][3], 1)} (CAGR / 변동성 / 샤프 / MDD)", "", table(sub), ""]

    asset_rows = []
    rf = table_["rf"].loc[months(start, end)]
    for a, name in ASSETS.items():
        asset_rows.append(row(f"{name}: 단순 보유", monthly_metrics(table_[f"{a}_tr"].loc[months(start, end)], rf)))
        out, m, _ = run(table_, start, end, [a], "total_return")
        asset_rows.append(row(f"{name}: 10개월 이동평균 (총수익 지수 신호)", m))
    lines += ["## 자산별 (각 자산 100%로 따로 계산, 참고)", "", table(asset_rows), ""]

    sens = []
    for window in WINDOWS:
        for source, label in SOURCES.items():
            sens.append(row(f"{window}개월, {label}", run(table_, start, end, all_assets, source, window)[1]))
    lines += ["## 이동평균 기간을 바꿨을 때", "", f"비교: 단순 보유 CAGR {pct(hold[1]['cagr'])}, 샤프 (원문 방식) "
              f"{hold[1]['sharpe_paper']:.2f}, MDD {pct(hold[1]['mdd'], 1)}.", "", table(sens), ""]
    return lines


# ---------------------------------------------------------------- 차트


def month_end_index(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp(how="end").normalize()


def with_start(series: pd.Series, value: float) -> pd.Series:
    """시작 전달 말 값을 앞에 붙입니다 (holding_chart는 첫 행을 기준일로 봅니다)."""
    base = pd.Series([value], index=pd.PeriodIndex([series.index[0] - 1], freq="M"))
    out = pd.concat([base, series])
    out.index = month_end_index(out.index)
    return out


def chart(period: str) -> list[Path]:
    start, end = PERIODS[period]
    table_ = load(end)
    all_assets = list(ASSETS)
    rf = table_["rf"].loc[months(start, end)]
    hold_out, hold_m, _ = run(table_, start, end, all_assets, None)
    hold_wealth = with_start((1 + hold_out["ret"]).cumprod(), 1.0)
    paths = []
    short_names = {"us_stock": "미국 주식", "eafe": "선진국 주식 (대체)", "bond": "10년 국채", "commodity": "원자재 (대체)",
                   "reit": "리츠"}
    for source, label in SOURCES.items():
        out, m, held = run(table_, start, end, all_assets, source)
        levels = {}
        for a in all_assets:
            level = level_from_returns(table_[SIGNAL_COLUMNS[source][a]]).loc[pd.Period(start, "M") - 1 : pd.Period(end, "M")]
            level.index = month_end_index(level.index)
            levels[short_names[a]] = level
        exposures = {short_names[a]: with_start(held[a], held[a].iloc[0]) for a in all_assets}
        paths.append(portfolio_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-gtaa-{source.replace('_', '-')}.png",
            title=f"GTAA 5자산 10개월 이동평균 ({label}): {PERIOD_NAMES[period]} {start} ~ {end}",
            subtitle=[
                f"비용·세금 없음, 월말 값 기준  ·  CAGR {m['cagr']:.2%} (단순 보유 {hold_m['cagr']:.2%})  ·  "
                f"MDD {m['mdd']:.1%} ({hold_m['mdd']:.1%})  ·  샤프(원문 방식) {m['sharpe_paper']:.2f} ({hold_m['sharpe_paper']:.2f})  ·  "
                f"평균 투자 비율 {m['time_in_market']:.0%}  ·  연간 매매 {m['trades_per_year']:.1f}회",
                "규칙: 자산마다 월말 지수가 10개월 이동평균보다 높으면 그 자산 20%, 아니면 그 20%를 현금. 매달 20%씩 다시 맞춤. "
                "월말 종가에 체결 (원문 방식).",
                "자산: S&P 500, EAFE 대체(Ken French), 10년 국채(합성), 원자재 대체(AQR 동일 비중), NAREIT All REITs. "
                "현금은 Ken French RF. 괄호 안은 같은 조건의 단순 보유(20%씩).",
                f"맨 위 칸은 신호에 쓴 지수({label})입니다. 샤프(원문 방식) = (CAGR - 현금 CAGR) / 변동성.",
            ],
            levels=levels,
            exposures=exposures,
            invested=with_start(out["invested"], out["invested"].iloc[0]),
            equity=with_start((1 + out["ret"]).cumprod(), 1.0),
            hold_equity=hold_wealth,
            strategy_label=f"GTAA ({label})",
        ))
    for a, name in ASSETS.items():
        out, m, held = run(table_, start, end, [a], "total_return")
        asset_hold = monthly_metrics(table_[f"{a}_tr"].loc[months(start, end)], rf)
        # 맨 위 칸과 이동평균은 신호에 쓴 지수입니다 (S&P 500은 12개월 전 배당을 쓴 지수).
        level = level_from_returns(table_[SIGNAL_COLUMNS["total_return"][a]])
        average = level.rolling(WINDOW).mean()
        level_d, average_d = level.copy(), average.copy()
        level_d.index, average_d.index = month_end_index(level.index), month_end_index(average.index)
        exposure = with_start(held[a], held[a].iloc[0])
        asset_wealth = with_start((1 + table_[f"{a}_tr"].loc[months(start, end)]).cumprod(), 1.0)
        paths.append(holding_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-{a.replace('_', '-')}.png",
            title=f"GTAA 구성 자산: {name}, 10개월 이동평균 (총수익 지수 신호): {PERIOD_NAMES[period]} {start} ~ {end}",
            subtitle=[
                f"이 자산 100%로 따로 계산, 비용·세금 없음, 월말 값  ·  {metrics_line(m, asset_hold, SHARPE_LABEL)}",
                "규칙: 월말 총수익 지수가 10개월 이동평균보다 높으면 보유, 아니면 현금(Ken French RF). 월말 종가에 체결 (원문 방식).",
            ],
            price=level_d,
            price_label=f"{name} 신호용 총수익 지수 (로그 눈금)",
            overlay=average_d,
            overlay_label="10개월 이동평균",
            exposure=exposure,
            panels=[Panel("이격도: 월말 지수 / 10개월 이동평균 - 1", level_d / average_d - 1, 0.0, "이동평균", percent=True)],
            equity=with_start((1 + out["ret"]).cumprod(), 1.0),
            hold_equity=asset_wealth,
            strategy_label="10개월 이동평균",
            holding_label="자산 보유 구간",
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIOD_NAMES, default="design")
    parser.add_argument("--chart", action="store_true", help="보유 구간 차트만 만듭니다")
    parser.add_argument("--data-check", action="store_true", help="자산별 단순 보유만 원문과 비교합니다 (전략 없음)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = ROOT / "reports" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.data_check:
        lines, out = data_check(), out_dir / "data-check.md"
    elif args.chart:
        for path in chart(args.period):
            print(f"저장: {path.relative_to(ROOT)}")
        return
    else:
        lines, out = report(args.period), out_dir / f"{args.period}.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
