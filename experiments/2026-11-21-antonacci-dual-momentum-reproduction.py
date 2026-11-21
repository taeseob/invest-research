"""Antonacci 듀얼 모멘텀(GEM) 근사 재현. 연구 일지: journal/2026-11-21-antonacci-dual-momentum-reproduction.md

규칙 (주): 매달 말 S&P 500의 12개월 수익률이 현금(T-bill)보다 높으면, S&P 500과 미국 외 주식 중 12개월 수익률이 높은 쪽을
100% 들고, 아니면 채권을 듭니다. 월말 종가에 같은 날 체결합니다(원문 방식). 비용과 세금은 없습니다.
데이터: data/dual_momentum_monthly.csv (python -m src.data --dual-momentum). 미국 외 주식과 채권은 대체 자료입니다.

실행:
    python experiments/2026-11-21-antonacci-dual-momentum-reproduction.py --data-check   # 대체 자료 교차 확인 (전략 없음)
    python experiments/2026-11-21-antonacci-dual-momentum-reproduction.py                # 재현 기간 (1976-01 ~ 2013-12)
    python experiments/2026-11-21-antonacci-dual-momentum-reproduction.py --chart        # 보유 구간 차트만
    python experiments/2026-11-21-antonacci-dual-momentum-reproduction.py --period oos   # 표본 외 기간 (마지막에 한 번만)

결과는 reports/2026-11-21-antonacci-dual-momentum-reproduction/ 에 저장합니다.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json  # noqa: E402

from src.charts import CATEGORICAL, INK, portfolio_chart  # noqa: E402
from src.data import (  # noqa: E402
    RAW_DIR,
    YAHOO_DM_SYMBOLS,
    check_monthly_continuity,
    month_end_value,
    parse_yahoo_adjclose,
)
from src.monthly import (  # noqa: E402
    dual_momentum_choice,
    level_from_returns,
    monthly_metrics,
    rotation_returns,
    trailing_return,
)

EXPERIMENT = "2026-11-21-antonacci-dual-momentum-reproduction"
PERIOD_NAMES = {"design": "재현 기간 (설계용)", "oos": "표본 외 기간"}
PERIODS = {"design": ("1976-01", "2013-12"), "oos": ("2014-01", "2025-12")}
WINDOW = 12
WINDOWS = [6, 9, 12]
TRADE = 0.001
NAMES = {"us_stock": "미국 주식", "ex_us": "미국 외 주식 (대체)", "bond": "채권 (5년 국채 합성)", "acwi": "ACWI 대체 (50/50)"}
# 차트 색은 자산마다 고정합니다 (차트가 달라도 같은 자산은 같은 색).
COLORS = {NAMES[a]: CATEGORICAL[i] for i, a in enumerate(["us_stock", "ex_us", "bond", "acwi"])}
# 전략: (설명, 절대 모멘텀 기준). "absolute_acwi"는 ACWI 대체 기준 절대 모멘텀만입니다.
STRATEGIES = {
    "gem": "GEM (주 규칙: S&P 500 기준 절대 모멘텀)",
    "gem_selected": "GEM (참고 규칙: 고른 자산 기준 절대 모멘텀)",
    "relative": "상대 모멘텀만",
    "absolute_acwi": "절대 모멘텀만 (ACWI 대체 기준)",
}
ABSOLUTE = {"gem": "us", "gem_selected": "selected", "relative": None}
# 책(2014)의 1974~2013년 수치, 2차 자료 인용 (일지 "원문"): (CAGR, 변동성, 샤프, MDD)
BOOK = {
    "gem": (0.1743, 0.1264, 0.87, -0.2272),
    "relative": (0.1441, np.nan, 0.52, -0.5306),
    "absolute_acwi": (0.1266, np.nan, 0.57, -0.2376),
    "acwi": (0.0885, np.nan, 0.22, -0.6021),
}
# 판단 기준 (일지, 결과 계산 전에 정함)
CAGR_TOLERANCE = 0.015
MDD_TOLERANCE = 0.05
EX_US_CORRELATION = 0.95
BOND_CORRELATION = 0.80
# 책 105쪽 인용(2차 자료): 절대 모멘텀이 도운 해, 상대 모멘텀이 도운 해
BOOK_ABSOLUTE_YEARS = [1982, 2001, 2009]
BOOK_RELATIVE_YEARS = [1986, 1987, 1988, 2004, 2005, 2006, 2007]


# ---------------------------------------------------------------- 데이터


def load(period_end: str) -> pd.DataFrame:
    table = pd.read_csv(ROOT / "data" / "dual_momentum_monthly.csv", index_col=0)
    table.index = pd.PeriodIndex(table.index, freq="M", name="month")
    return cut(table, period_end)


def cut(table: pd.DataFrame, period_end: str) -> pd.DataFrame:
    """기간 끝까지만 남깁니다. 그 뒤 데이터는 신호, 수익률, 차트 어디에도 들어가지 않습니다."""
    return table[table.index <= pd.Period(period_end, freq="M")]


def months(start: str, end: str) -> slice:
    return slice(pd.Period(start, freq="M"), pd.Period(end, freq="M"))


ETF_COLUMNS = {"ACWX": "ex_us_tr", "AGG": "bond_tr"}


def etf_month_end_prices() -> dict[str, pd.Series]:
    """ACWX, AGG의 월말 수정 종가 (Yahoo 원본에서). 인덱스는 월."""
    out = {}
    for symbol in ETF_COLUMNS:
        data = json.loads((RAW_DIR / YAHOO_DM_SYMBOLS[symbol]).read_text(encoding="utf-8"))
        out[symbol] = month_end_value(parse_yahoo_adjclose(data))
    return out


def etf_table(table: pd.DataFrame, required_start: pd.Period, prices: dict[str, pd.Series] | None = None) -> pd.DataFrame:
    """표본 외 참고용: 미국 외 주식과 채권을 실제 ETF(ACWX, AGG)의 월 수익률로 바꾼 표.

    - 그달 수익률 = 그달 말 가격 / 전달 말 가격 − 1. ETF 첫 달은 수익률이 없어 대체 자료 그대로입니다.
    - table 끝 달 뒤의 가격은 쓰지 않습니다.
    - required_start(보통 평가 시작 − 12개월, 신호 준비 포함) ~ table 끝 달을 ETF 수익률이 빈 달 없이 덮어야 합니다.
      그 앞의 달은 대체 자료 그대로이고, 평가와 신호에는 쓰이지 않습니다.
    prices: 기호 → 월말 가격. 넘기지 않으면 Yahoo 원본에서 읽습니다 (테스트에서는 가짜 가격을 넘깁니다).
    """
    prices = etf_month_end_prices() if prices is None else prices
    out = table.copy()
    end = table.index[-1]
    for symbol, column in ETF_COLUMNS.items():
        price = prices[symbol]
        price = price[price.index <= end]
        check_monthly_continuity(price.index, f"{symbol} 월말 가격")
        returns = (price / price.shift(1) - 1).dropna()
        if returns.empty or returns.index[0] > required_start or returns.index[-1] < end:
            raise ValueError(f"{symbol} 수익률이 {required_start} ~ {end}를 덮지 않습니다.")
        out.loc[returns.index.intersection(out.index), column] = returns
    return out


def asset_returns(table: pd.DataFrame, bond_column: str = "bond_tr") -> pd.DataFrame:
    """수익률용 열. ACWI 대체는 미국 주식과 미국 외 주식을 반씩, 매달 다시 맞춘 것입니다."""
    out = pd.DataFrame({"us_stock": table["us_stock_tr"], "ex_us": table["ex_us_tr"], "bond": table[bond_column]})
    out["acwi"] = 0.5 * (out["us_stock"] + out["ex_us"])
    return out


def momentum(table: pd.DataFrame, window: int) -> pd.DataFrame:
    """t월 말 12개월(window개월) 수익률. 미국 주식은 신호용 열(12개월 전 배당)을 씁니다."""
    signal_returns = pd.DataFrame({
        "us_stock": table["us_stock_sig_tr"], "ex_us": table["ex_us_tr"], "cash": table["rf"],
    })
    signal_returns["acwi"] = 0.5 * (signal_returns["us_stock"] + signal_returns["ex_us"])
    return trailing_return(signal_returns, window)


# ---------------------------------------------------------------- 실행


def choices(table: pd.DataFrame, kind: str, window: int = WINDOW) -> pd.Series:
    """t월 말 선택 (자산 열 이름). 준비 기간은 결측값입니다."""
    m = momentum(table, window)
    if kind == "absolute_acwi":
        ready = m[["acwi", "cash"]].notna().all(axis=1)
        picked = pd.Series(np.where(m["acwi"] > m["cash"], "acwi", "bond"), index=m.index, dtype=object)
        return picked.where(ready)
    return dual_momentum_choice(m["us_stock"], m["ex_us"], m["cash"], ABSOLUTE[kind])


def run(table: pd.DataFrame, start: str, end: str, kind: str, window: int = WINDOW, cost: float = 0.0,
        bond_column: str = "bond_tr"):
    """전략을 돌립니다. 반환: 월 수익률 표, 지표. 그달 보유 자산 = 전달 말 선택."""
    held = choices(table, kind, window).shift(1).loc[months(start, end)]
    if held.isna().any():
        raise ValueError(f"평가 기간에 선택이 없는 달이 있습니다: {held[held.isna()].index[0]}")
    returns = asset_returns(table, bond_column).loc[months(start, end)]
    out = rotation_returns(returns, held, cost)
    m = monthly_metrics(out["ret"], table["rf"].loc[months(start, end)])
    m["switches_per_year"] = out["switched"].sum() / m["years"]
    m["share"] = out["held"].value_counts(normalize=True).to_dict()
    return out, m


def hold(table: pd.DataFrame, start: str, end: str, asset: str) -> dict:
    returns = asset_returns(table).loc[months(start, end), asset]
    m = monthly_metrics(returns, table["rf"].loc[months(start, end)])
    m["switches_per_year"], m["share"] = 0.0, {asset: 1.0}
    return m


def yearly(ret: pd.Series) -> pd.Series:
    return (1 + ret).groupby(ret.index.year).prod() - 1


# ---------------------------------------------------------------- 표


def pct(value: float, digits: int = 2) -> str:
    return "" if pd.isna(value) else f"{value:.{digits}%}"


def table_md(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


def share_text(share: dict) -> str:
    return ", ".join(f"{NAMES[a].split(' (')[0]} {v:.0%}" for a, v in sorted(share.items(), key=lambda x: -x[1]))


def row(name: str, m: dict) -> dict:
    recovery = "미회복" if pd.isna(m["mdd_recovery_months"]) else f"{m['mdd_recovery_months']:.0f}개월"
    return {
        "구분": name,
        "CAGR": pct(m["cagr"]),
        "변동성": pct(m["volatility"]),
        "샤프": f"{m['sharpe']:.2f}",
        "샤프 (원문 방식)": f"{m['sharpe_paper']:.2f}",
        "MDD": pct(m["mdd"], 1),
        "MDD 기간": f"{m['mdd_peak']} ~ {m['mdd_trough']}",
        "MDD 회복": recovery,
        "연간 교체": f"{m['switches_per_year']:.1f}",
        "보유 비율": share_text(m["share"]),
    }


def data_check() -> list[str]:
    """대체 자료와 실제 ETF의 월 수익률 비교 (2013-12까지). 전략은 계산하지 않습니다."""
    check = pd.read_csv(ROOT / "data" / "dual_momentum_check.csv", index_col=0)
    check.index = pd.PeriodIndex(check.index, freq="M")
    lines = ["# 듀얼 모멘텀 근사 재현: 데이터 점검 (대체 자료 교차 확인, 전략 없음)", "",
             "모두 월 수익률입니다. ETF는 Yahoo 수정 종가(분배금 재투자)이고, 운용보수를 뺀 값입니다. 대체 자료에는 보수가 없습니다.", ""]
    results = {}
    for proxy, etf, label, threshold, first in [
        ("ex_us_proxy_tr", "acwx_tr", "미국 외 주식 대체(Ken French, 신흥국 없음) vs ACWX(MSCI ACWI ex US)", EX_US_CORRELATION, "2008-04"),
        ("bond_proxy_tr", "agg_tr", "채권 대체(5년 국채 합성) vs AGG(US Aggregate)", BOND_CORRELATION, "2003-10"),
    ]:
        pair = check.loc[pd.Period(first, "M"):, [proxy, etf]].dropna()
        corr = pair[proxy].corr(pair[etf])
        years = (1 + pair).groupby(pair.index.year).prod() - 1
        n = len(pair) / 12
        cagr = (1 + pair).prod() ** (1 / n) - 1
        results[label] = corr >= threshold
        lines += [
            f"## {label}", "",
            f"- 기간 {pair.index[0]} ~ {pair.index[-1]}: 월 수익률 상관 **{corr:.3f}** (기준 {threshold:.2f} 이상 → "
            f"{'충족' if corr >= threshold else '미충족'}), 월 차이의 표준편차 {(pair[proxy] - pair[etf]).std():.2%}",
            f"- CAGR: 대체 자료 {pct(cagr[proxy])}, ETF {pct(cagr[etf])}", "",
            "| 연도 | 대체 자료 | ETF | 차이 |", "|---|---|---|---|",
            *[f"| {y} | {pct(r[proxy], 1)} | {pct(r[etf], 1)} | {(r[proxy] - r[etf]) * 100:+.1f}%p |" for y, r in years.iterrows()],
            "",
        ]
    lines += [f"**데이터 확인 (두 기준 모두): {'충족' if all(results.values()) else '미충족'}**", ""]

    # 원인 확인 (판단에 쓰지 않음): 채권 대체 자료의 시차와 만기, 금융위기 1년을 뺀 상관
    pair = check.loc[pd.Period("2003-10", "M"):].dropna(subset=["agg_tr"])
    crisis = (pair.index >= pd.Period("2008-07", "M")) & (pair.index <= pd.Period("2009-06", "M"))
    rows = []
    for years, column in [(3, "bond3_proxy_tr"), (5, "bond_proxy_tr"), (7, "bond7_proxy_tr"), (10, "bond10_proxy_tr")]:
        rows.append({
            "만기": f"{years}년", "상관 (같은 달)": f"{pair['agg_tr'].corr(pair[column]):.3f}",
            "대체 자료 한 달 앞": f"{pair['agg_tr'].corr(pair[column].shift(-1)):.3f}",
            "대체 자료 한 달 뒤": f"{pair['agg_tr'].corr(pair[column].shift(1)):.3f}",
            "2008-07~2009-06 제외": f"{pair.loc[~crisis, 'agg_tr'].corr(pair.loc[~crisis, column]):.3f}",
        })
    before = pair.index < pd.Period("2008-07", "M")
    after = pair.index > pd.Period("2009-06", "M")
    lines += ["## 원인 확인: 채권 대체 자료와 AGG (판단에 쓰지 않음)", "",
              "데이터 확인이 미충족이라 원인을 찾으려고 더했습니다. 주 채권 자료는 사전 계획대로 5년 국채입니다. "
              "2008-07~2009-06 제외 구간은 결과를 보고 고른 사후 탐색이라, 위기 전·후로 나눈 상관도 함께 냅니다.", "",
              table_md(rows), "",
              f"- 5년 국채, 위기 전(2003-10 ~ 2008-06) 상관 {pair.loc[before, 'agg_tr'].corr(pair.loc[before, 'bond_proxy_tr']):.3f}, "
              f"위기 후(2009-07 ~ 2013-12) 상관 {pair.loc[after, 'agg_tr'].corr(pair.loc[after, 'bond_proxy_tr']):.3f}",
              f"- AGG 월 수익률 중 절댓값 2.5%를 넘는 달: "
              + ", ".join(f"{m} {v:+.2%}" for m, v in pair["agg_tr"][pair["agg_tr"].abs() > 0.025].items()), ""]
    return lines


def comparison_section(table: pd.DataFrame, start: str, end: str) -> list[str]:
    gem = run(table, start, end, "gem")[1]
    relative = run(table, start, end, "relative")[1]
    absolute = run(table, start, end, "absolute_acwi")[1]
    acwi = hold(table, start, end, "acwi")
    ours = {"gem": gem, "relative": relative, "absolute_acwi": absolute, "acwi": acwi}
    labels = {"gem": "GEM", "relative": "상대 모멘텀만", "absolute_acwi": "절대 모멘텀만", "acwi": "ACWI 단순 보유 (재현은 50/50 대체)"}
    rows = [{
        "구분": labels[k],
        "CAGR 책": pct(BOOK[k][0]), "CAGR 재현": pct(ours[k]["cagr"]), "차이": f"{(ours[k]['cagr'] - BOOK[k][0]) * 100:+.2f}%p",
        "MDD 책": pct(BOOK[k][3], 1), "MDD 재현": pct(ours[k]["mdd"], 1),
        "샤프 책": f"{BOOK[k][2]:.2f}", "샤프 재현": f"{ours[k]['sharpe']:.2f}",
    } for k in BOOK]
    checks = {
        "(가) GEM CAGR이 책과 1.5%p 이내": abs(gem["cagr"] - BOOK["gem"][0]) <= CAGR_TOLERANCE,
        "(나) GEM MDD가 책과 5%p 이내": abs(gem["mdd"] - BOOK["gem"][3]) <= MDD_TOLERANCE,
        "(다) CAGR 순서 GEM > 상대 모멘텀만 > ACWI 대체": gem["cagr"] > relative["cagr"] > acwi["cagr"],
        "(다) GEM MDD가 상대 모멘텀만과 ACWI 대체보다 작음": abs(gem["mdd"]) < min(abs(relative["mdd"]), abs(acwi["mdd"])),
    }
    passed = all(checks.values())
    return [
        "## 책(1974~2013년)과 비교", "",
        f"재현 기간은 {start} ~ {end}입니다(책보다 1974~1975년이 빠짐). 책 수치는 2차 자료 인용이고, 샤프의 정의를 알 수 없어 기준으로 쓰지 않습니다.", "",
        table_md(rows), "",
        *[f"- {k}: {'충족' if v else '미충족'}" for k, v in checks.items()],
        f"- **근사 재현 판단 (사전 기준, 위 네 가지 모두): {'통과' if passed else '미통과'}**", "",
    ]


def book_years_section(table: pd.DataFrame, start: str, end: str) -> list[str]:
    """책 105쪽 인용: 절대 모멘텀이 도운 해(채권 보유)와 상대 모멘텀이 도운 해(미국 외 보유). 참고."""
    out, _ = run(table, start, end, "gem")
    held = out["held"]
    rows = []
    for year in BOOK_ABSOLUTE_YEARS + BOOK_RELATIVE_YEARS:
        months_held = held[held.index.year == year].value_counts().to_dict()
        rows.append({"연도": year, "책이 말한 효과": "절대 모멘텀" if year in BOOK_ABSOLUTE_YEARS else "상대 모멘텀",
                     "채권 보유 달": months_held.get("bond", 0), "미국 외 보유 달": months_held.get("ex_us", 0),
                     "미국 보유 달": months_held.get("us_stock", 0)})
    return ["## 책 본문의 연도와 비교 (참고)", "",
            "책 105쪽 인용(2차 자료): GEM은 1982·2001·2009년에 절대 모멘텀(채권), 1986~1988·2004~2007년에 상대 모멘텀(미국 외)의 덕을 봤다.", "",
            table_md(rows), ""]


def report(period: str) -> list[str]:
    start, end = PERIODS[period]
    table = load(end)
    main_rows = [row(STRATEGIES[k], run(table, start, end, k)[1]) for k in STRATEGIES]
    main_rows += [row(f"{NAMES[a]} 단순 보유", hold(table, start, end, a)) for a in ["acwi", "us_stock", "ex_us", "bond"]]
    main_rows += [
        row("(참고) GEM, 편도 0.1%", run(table, start, end, "gem", cost=TRADE)[1]),
        row("(참고) GEM, 채권을 10년 국채로", run(table, start, end, "gem", bond_column="bond10_tr")[1]),
    ]
    lines = [
        f"# Antonacci 듀얼 모멘텀(GEM) 근사 재현: {PERIOD_NAMES[period]} ({start} ~ {end})", "",
        "규칙 (주): 매달 말 S&P 500의 12개월 수익률이 현금(1개월 국채)보다 높으면 S&P 500과 미국 외 주식 중 12개월 수익률이 높은 쪽, "
        "아니면 채권을 100% 듭니다. 월말 종가에 같은 날 체결(원문 방식), 비용·세금 없음.",
        "자산: 미국 주식(S&P 500 + Shiller 배당, 신호는 12개월 전 배당), 미국 외 주식(Ken French 해외 지수 전체: EAFE + 캐나다, 신흥국 없음), "
        "채권(연준 H.15 5년 국채 금리로 합성), 현금(Ken French RF). ACWI 대체는 미국 50% + 미국 외 50%(매달 맞춤)입니다.",
        "샤프 = 월 초과수익률 평균 × 12 / (표준편차 × √12). 샤프 (원문 방식) = (CAGR − 현금 CAGR) / 변동성. 연간 교체는 자산을 바꾼 횟수입니다.", "",
        "## 기본 결과", "", table_md(main_rows), "",
    ]
    if period == "design":
        lines += comparison_section(table, start, end)
        lines += book_years_section(table, start, end)
    else:
        gem, acwi, sp = run(table, start, end, "gem")[1], hold(table, start, end, "acwi"), hold(table, start, end, "us_stock")
        sharpe_ok, mdd_ok = gem["sharpe"] > acwi["sharpe"], abs(gem["mdd"]) <= abs(acwi["mdd"]) / 2
        etf = etf_table(table, pd.Period(start, freq="M") - WINDOW)
        etf_rows = [row("(참고) GEM, 미국 외·채권을 실제 ETF(ACWX, AGG)로", run(etf, start, end, "gem")[1]),
                    row("(참고) ACWI 대체 단순 보유, 미국 외를 ACWX로", hold(etf, start, end, "acwi"))]
        oos_months = months(start, end)
        corr_ex = table["ex_us_tr"].loc[oos_months].corr(etf["ex_us_tr"].loc[oos_months])
        corr_bond = table["bond_tr"].loc[oos_months].corr(etf["bond_tr"].loc[oos_months])
        lines += [
            "## 표본 외 기간 판단 (기준은 보기 전에 정함)", "",
            f"- GEM 샤프 {gem['sharpe']:.2f} vs ACWI 대체 {acwi['sharpe']:.2f} → {'높음' if sharpe_ok else '높지 않음'}",
            f"- GEM MDD {pct(gem['mdd'], 1)} vs ACWI 대체 {pct(acwi['mdd'], 1)}의 절반 → {'이하' if mdd_ok else '넘음'}",
            f"- **원문 결론 유지: {'예' if sharpe_ok and mdd_ok else '아니오'}**",
            f"- 참고: CAGR GEM {pct(gem['cagr'])}, ACWI 대체 {pct(acwi['cagr'])}, S&P 500 {pct(sp['cagr'])} "
            f"(S&P 500 샤프 {sp['sharpe']:.2f}, MDD {pct(sp['mdd'], 1)})", "",
            "## 참고: 실제 ETF로 바꾼 결과 (판단에 쓰지 않음, 표본 외 기간을 보기 전에 넣기로 정함)", "",
            "미국 외 주식은 ACWX(MSCI ACWI ex US, 신흥국 포함), 채권은 AGG(US Aggregate)의 Yahoo 수정 종가입니다(운용보수를 뺀 값). "
            "신호와 수익률을 모두 ETF로 계산합니다. S&P 500은 그대로입니다.", "",
            f"- 이 기간 대체 자료와 ETF의 월 수익률 상관: 미국 외 {corr_ex:.3f}, 채권 {corr_bond:.3f}", "",
            table_md(etf_rows), "",
        ]

    out, _ = run(table, start, end, "gem")
    returns = asset_returns(table).loc[months(start, end)]
    years = pd.DataFrame({"GEM": yearly(out["ret"]), **{NAMES[a]: yearly(returns[a]) for a in ["us_stock", "ex_us", "bond", "acwi"]}})
    held_year = out["held"].groupby(out.index.year).agg(lambda s: share_text(s.value_counts(normalize=True).to_dict()))
    lines += ["## 연도별 수익률", "", "| 연도 | " + " | ".join(years.columns) + " | GEM 보유 |", "|" + "---|" * (len(years.columns) + 2)]
    for y, r in years.iterrows():
        lines.append(f"| {y} | " + " | ".join(pct(v, 1) for v in r) + f" | {held_year[y]} |")
    lines.append("")

    sens = []
    for window in WINDOWS:
        for kind in ["gem", "gem_selected", "relative"]:
            sens.append(row(f"{window}개월, {STRATEGIES[kind]}", run(table, start, end, kind, window)[1]))
    lines += ["## 판단 기간을 바꿨을 때", "", table_md(sens), ""]
    return lines


# ---------------------------------------------------------------- 차트


def month_end_index(index: pd.PeriodIndex) -> pd.DatetimeIndex:
    return index.to_timestamp(how="end").normalize()


def with_start(series: pd.Series, value) -> pd.Series:
    base = pd.Series([value], index=pd.PeriodIndex([series.index[0] - 1], freq="M"))
    out = pd.concat([base, series])
    out.index = month_end_index(out.index)
    return out


def chart(period: str) -> list[Path]:
    """GEM(주 규칙)과 비교 전략(상대 모멘텀만, 절대 모멘텀만)을 따로 그립니다. 단순 보유는 ACWI 대체(50/50)입니다."""
    start, end = PERIODS[period]
    table = load(end)
    returns = asset_returns(table)
    acwi = hold(table, start, end, "acwi")
    acwi_wealth = with_start((1 + returns["acwi"].loc[months(start, end)]).cumprod(), 1.0)
    m_all = momentum(table, WINDOW)
    paths = []
    settings = [
        ("gem", ["us_stock", "ex_us", "bond"], ["us_stock", "ex_us"],
         "S&P 500의 12개월 수익률이 현금보다 높으면 S&P 500과 미국 외 주식 중 12개월 수익률이 높은 쪽, 아니면 채권"),
        ("relative", ["us_stock", "ex_us"], ["us_stock", "ex_us"],
         "S&P 500과 미국 외 주식 중 12개월 수익률이 높은 쪽 (채권으로 피하지 않음)"),
        ("absolute_acwi", ["acwi", "bond"], ["acwi"], "ACWI 대체(50/50)의 12개월 수익률이 현금보다 높으면 ACWI 대체, 아니면 채권"),
    ]
    for kind, assets, momentum_assets, rule in settings:
        out, m = run(table, start, end, kind)
        levels = {}
        for a in assets:
            level = level_from_returns(returns[a]).loc[pd.Period(start, "M") - 1 : pd.Period(end, "M")]
            level.index = month_end_index(level.index)
            levels[NAMES[a]] = level
        held = out["held"]
        exposures = {NAMES[a]: with_start((held == a).astype(float), float(held.iloc[0] == a)) for a in assets}
        indicators = {}
        for a in momentum_assets:
            series = m_all[a].loc[months(start, end)]
            series.index = month_end_index(series.index)
            indicators[NAMES[a]] = series
        # 현금 기준선은 절대 모멘텀이 있는 규칙에만 그립니다 (상대 모멘텀만은 현금과 비교하지 않음).
        if kind != "relative":
            cash = m_all["cash"].loc[months(start, end)]
            cash.index = month_end_index(cash.index)
            indicators["현금 (1개월 국채) 12개월"] = cash
        paths.append(portfolio_chart(
            ROOT / "reports" / EXPERIMENT / f"{period}-holdings-{kind.replace('_', '-')}.png",
            title=f"{STRATEGIES[kind]}: {PERIOD_NAMES[period]} {start} ~ {end}",
            subtitle=[
                f"비용·세금 없음, 월말 값 기준  ·  CAGR {m['cagr']:.2%} (ACWI 대체 {acwi['cagr']:.2%})  ·  "
                f"MDD {m['mdd']:.1%} ({acwi['mdd']:.1%})  ·  샤프 {m['sharpe']:.2f} ({acwi['sharpe']:.2f})  ·  "
                f"연간 교체 {m['switches_per_year']:.1f}회  ·  보유 비율 {share_text(m['share'])}",
                f"규칙: 매달 말 {rule}. 월말 종가에 체결 (원문 방식).",
                "자산: S&P 500, 미국 외 주식 대체(Ken French, 신흥국 없음), 채권 대체(5년 국채 합성). 현금은 Ken French RF. "
                "괄호 안은 ACWI 대체(미국 50% + 미국 외 50%) 단순 보유.",
                "샤프 = 월 초과수익률 평균 × 12 / (표준편차 × √12). 신호 칸의 미국 주식은 12개월 전 배당을 쓴 신호용 수익률입니다.",
            ],
            levels=levels,
            exposures=exposures,
            equity=with_start((1 + out["ret"]).cumprod(), 1.0),
            hold_equity=acwi_wealth,
            strategy_label=STRATEGIES[kind].split(" (")[0],
            indicators=indicators,
            indicator_title="신호: 그달 말 기준 12개월 수익률" + (" (점선은 현금 기준선)" if kind != "relative" else ""),
            bands_title="보유 구간: 한 번에 한 자산만 100% 들고, 칠한 곳이 그 자산을 들고 있던 구간",
            colors=COLORS,
            # 자산 가치 칸: 전략은 무채색, 단순 보유는 ACWI 대체의 고정 색 (위 칸의 자산 색과 겹치지 않게)
            equity_colors=(INK, COLORS[NAMES["acwi"]]),
            hold_label="단순 보유: ACWI 대체 (50/50)",
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--period", choices=PERIOD_NAMES, default="design")
    parser.add_argument("--chart", action="store_true", help="보유 구간 차트만 만듭니다")
    parser.add_argument("--data-check", action="store_true", help="대체 자료와 ETF만 비교합니다 (전략 없음)")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = ROOT / "reports" / EXPERIMENT
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.chart:
        for path in chart(args.period):
            print(f"저장: {path.relative_to(ROOT)}")
        return
    lines, out = (data_check(), out_dir / "data-check.md") if args.data_check else (report(args.period), out_dir / f"{args.period}.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
