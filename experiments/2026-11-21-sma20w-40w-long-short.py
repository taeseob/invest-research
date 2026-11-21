"""20주·40주 이동평균 롱/숏(인버스), SPY와 KOSPI. 연구 일지: journal/2026-11-21-sma20w-40w-long-short.md

규칙: 주 마지막 거래일 종가가 N주 이동평균보다 높으면 주식 100%, 아니면 -1배 인버스 ETF 100% (N = 20, 40).
비교 대상: 같은 기간, 같은 비용·세금의 롱/현금(기본 전략)과 단순 보유.
체결: 신호 낸 날 종가가 주 기준(직전 실험과 같음), 다음 거래일 종가를 참고로 함께 냅니다.
인버스는 지수로 합성합니다(src/backtest.py의 synthetic_inverse). 실제 상품(SH, KODEX 인버스)과의 비교는 --tracking.

실행:
    python experiments/2026-11-21-sma20w-40w-long-short.py --tracking   # 합성 인버스 vs 실제 상품 (설계용 기간 안)
    python experiments/2026-11-21-sma20w-40w-long-short.py              # 설계용 기간 결과
    python experiments/2026-11-21-sma20w-40w-long-short.py --chart      # 보유 구간 차트만

결과는 reports/2026-11-21-sma20w-40w-long-short/ 에 저장합니다.
"""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import (  # noqa: E402
    Costs,
    holding_from_signal,
    long_short,
    metrics,
    moving_average_signal,
    period_last_days,
    run_backtest,
    snap_to_tick,
    synthetic_index_etf,
    synthetic_inverse,
    trade_list,
)
from src.charts import Panel, holding_chart, metrics_line  # noqa: E402
from src.data import PRICE_TICK  # noqa: E402

EXPERIMENT = "2026-11-21-sma20w-40w-long-short"
REPORT_DIR = ROOT / "reports" / EXPERIMENT
PERIOD = "design"  # 표본 외 기간은 이번에 보지 않습니다. 코드에도 넣지 않습니다.
WINDOWS = [20, 40]
MAIN_LAG = 0  # 신호 낸 날 종가 체결 (사용자 지정, 주 기준)
LAGS = [0, 1]  # 1: 다음 거래일 종가 체결 (저장소 기본, 참고)
TRADE = 0.001
WEEK = "W-FRI"
NO_TAX = {"dividend_withholding": 0.0, "capital_gains_tax": 0.0, "interest_tax": 0.0}
TRADE_GRID = [0.0, 0.0005, 0.001, 0.002, 0.003, 0.005]  # 손익분기 표의 편도 매매 비용
BREAK_EVEN_MAX = 0.02  # 손익분기 비용을 찾는 범위의 끝
INTEREST_GRID = [2.0, 1.0, 0.0]

# 인버스 보수 (현재 공시). SH: ProShares 펀드 페이지 총보수 0.88%. KODEX 인버스: 삼성자산운용 2025-05-30 자료 총보수 0.64%.
SH_EXPENSE = 0.0088
KODEX_INVERSE_EXPENSE = 0.0064
# 교차 확인에서 기초 지수 대신 쓰는 KODEX 200(배당 포함)은 자기 보수만큼 낮으므로, 그만큼 합성 인버스에서 더 뺍니다.
KODEX200_EXPENSE = 0.0015
# 사전 계획에서 정한 이자 배수(실제 상품과 가장 잘 맞는 값)와 차입 비용(맞춘 값이 음수라 0)
SPY_INTEREST, KOSPI_INTEREST = 2.0, 1.0
BORROW = 0.0
# 직전 실험 검증에서 찾은 Yahoo KOSPI 종가 오류 (네이버 일봉 종가)
KOSPI_FIX = ("2000-02-18", 879.14)


def read(name: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / "data" / name, index_col=0, parse_dates=True)


# ---------------------------------------------------------------- 합성 인버스 교차 확인


def total_return_index(df: pd.DataFrame) -> pd.Series:
    """종가와 배당으로 배당 재투자 지수를 만듭니다. 첫날 1."""
    growth = (df["close"] + df["dividend"]) / df["close"].shift(1)
    return growth.fillna(1.0).cumprod()


def implied_borrow(actual: pd.Series, synthetic: pd.Series, expense: float) -> float:
    """합성 인버스(차입 비용 0)의 끝값이 실제와 같아지는 연 차입 비용.

    차입 비용 b는 매일 (1 − 보수 − b)^(경과 일수/365.25)로 빠지므로, 끝값의 비율은 ((1 − 보수 − b)/(1 − 보수))^연수입니다.
    음수면 실제가 합성보다 좋았다는 뜻입니다.
    """
    years = (actual.index[-1] - actual.index[0]).days / 365.25
    ratio = (actual.iloc[-1] / actual.iloc[0]) / (synthetic.iloc[-1] / synthetic.iloc[0])
    return (1 - expense) * (1 - ratio ** (1 / years))


def tracking_rows(actual: pd.Series, synthetic: dict[str, pd.Series]) -> tuple[list[dict], dict]:
    """연도별 수익률(실제, 합성들)과 전체 요약. actual과 synthetic은 같은 날짜, 첫날 기준으로 맞춥니다."""
    rows = []
    years = sorted(set(actual.index.year))
    for year in years:
        in_year = actual.index.year == year
        # 그해 수익률: 전해 마지막 날(없으면 첫날) → 그해 마지막 날
        before = actual.index[actual.index.year < year]
        first = before[-1] if len(before) else actual.index[0]
        last = actual.index[in_year][-1]
        row = {"연도": f"{year}" + (" (부분)" if first == actual.index[0] or year == years[-1] and last.month < 12 else ""),
               "실제": actual[last] / actual[first] - 1}
        for name, series in synthetic.items():
            row[name] = series[last] / series[first] - 1
        rows.append(row)

    span = (actual.index[-1] - actual.index[0]).days / 365.25
    summary = {"기간": f"{actual.index[0]:%Y-%m-%d} ~ {actual.index[-1]:%Y-%m-%d} ({span:.1f}년)",
               "실제 CAGR": (actual.iloc[-1] / actual.iloc[0]) ** (1 / span) - 1}
    actual_daily = actual.pct_change().dropna()
    for name, series in synthetic.items():
        daily = series.pct_change().dropna()
        diff = (actual_daily - daily).dropna()
        summary[f"{name} CAGR"] = (series.iloc[-1] / series.iloc[0]) ** (1 / span) - 1
        summary[f"{name} 일별 상관"] = actual_daily.corr(daily)
        summary[f"{name} 추적 오차(연율)"] = diff.std() * np.sqrt(252)
    return rows, summary


def tracking() -> list[str]:
    lines = ["# 합성 인버스와 실제 인버스 ETF 비교 (설계용 기간 안)", "",
             "합성: 매일 -1배 × 기초 지수 배당 포함 수익률 + m × 현금 수익률 − 보수 (src/backtest.py의 synthetic_inverse).",
             "실제: Yahoo 종가와 분배금으로 만든 배당 재투자 지수. 2013년 이후 자료는 쓰지 않습니다.", ""]
    implied = {}

    # SH: S&P 500 총수익 지수(^SP500TR)를 기초로, 현금은 Ken French RF
    spy, sp500tr, sh = read("spy_daily.csv"), read("sp500tr_daily.csv"), read("sh_daily.csv")
    under = pd.DataFrame({"close": sp500tr["close"].reindex(sh.index), "dividend": 0.0})
    rf = spy["rf"].reindex(sh.index)
    actual = total_return_index(sh)
    synthetic = {f"합성 m={m}": synthetic_inverse(under, rf, SH_EXPENSE, 0.0, m)["close"] for m in (2, 1)}
    implied["SH"] = {name: implied_borrow(actual, s, SH_EXPENSE) for name, s in synthetic.items()}
    lines += section("SH (ProShares Short S&P500)", actual, synthetic, implied["SH"],
                     f"기초: ^SP500TR, 현금: Ken French RF, 보수 연 {SH_EXPENSE:.2%}. 날짜가 SPY와 같습니다.")

    # KODEX 인버스: KODEX 200(분배금 포함)을 기초로, 현금은 CD 91일물. KODEX 200 보수만큼 더 뺍니다.
    kodex200, kodex_inverse = read("kodex200_daily.csv"), read("kodex_inverse_daily.csv")
    base = kodex200.loc[kodex_inverse.index[0]:]
    common = kodex_inverse.index.intersection(base.index)
    synthetic = {
        f"합성 m={m}": synthetic_inverse(base[["close", "dividend"]], base["rf"], KODEX_INVERSE_EXPENSE, KODEX200_EXPENSE, m)["close"].reindex(common)
        for m in (2, 1)
    }
    actual = total_return_index(kodex_inverse).reindex(common)
    implied["KODEX 인버스"] = {name: implied_borrow(actual, s, KODEX_INVERSE_EXPENSE + KODEX200_EXPENSE)
                              for name, s in synthetic.items()}
    lines += section("KODEX 인버스 (114800)", actual, synthetic, implied["KODEX 인버스"],
                     f"기초: KODEX 200 배당 재투자(KOSPI200 − 보수 연 {KODEX200_EXPENSE:.2%}), 현금: CD 91일물, "
                     f"보수 연 {KODEX_INVERSE_EXPENSE:.2%} + KODEX 200 보수 보정 {KODEX200_EXPENSE:.2%}. "
                     f"Yahoo에 KODEX 인버스가 빠진 날({len(base.index.difference(kodex_inverse.index))}일)은 공통 날짜로만 비교합니다.")
    return lines


def section(title: str, actual: pd.Series, synthetic: dict, implied: dict, note: str) -> list[str]:
    rows, summary = tracking_rows(actual, synthetic)
    table_rows = [{k: (v if isinstance(v, str) else f"{v:.2%}") for k, v in r.items()} for r in rows]
    for r, raw in zip(table_rows, rows):
        for name in synthetic:
            r[f"차이 (실제 − {name})"] = f"{raw['실제'] - raw[name]:+.2%}"
    lines = [f"## {title}", "", note, "", table(table_rows), ""]
    for key, value in summary.items():
        if isinstance(value, str):
            lines.append(f"- {key}: {value}")
        elif "상관" in key:
            lines.append(f"- {key}: {value:.4f}")
        else:
            lines.append(f"- {key}: {value:.2%}")
    for name, b in implied.items():
        lines.append(f"- {name}에서 끝값을 실제와 맞추는 연 차입 비용: {b:+.2%}")
    return lines + [""]


def table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


# ---------------------------------------------------------------- 자산


@dataclass
class Asset:
    key: str
    name: str
    price: pd.Series  # 차트와 매매 목록에 쓰는 가격 (배당 미포함 종가)
    signal_price: pd.Series  # 신호를 계산하는 가격 (호가 단위 개수)
    data: pd.DataFrame  # 주식 쪽 수익률 표: close, dividend, rf
    underlying: pd.DataFrame  # 인버스의 기초: close, dividend (배당 포함 수익률을 만듦)
    inverse_expense: float
    interest: float  # 사전 계획의 이자 배수 m
    period: tuple[str, str]
    pre_tax: dict
    after_tax: dict
    price_label: str
    notes: list
    chart_costs: str
    _inverse: dict = field(default_factory=dict)

    def inverse(self, m: float | None = None, expense: float | None = None) -> pd.DataFrame:
        """합성 인버스. 현금 수익률이 있는 구간(끝나지 않은 달 전)까지만 만듭니다."""
        m = self.interest if m is None else m
        expense = self.inverse_expense if expense is None else expense
        if (m, expense) not in self._inverse:
            rf = self.data["rf"]
            last = rf.last_valid_index()
            if rf.loc[:last].iloc[1:].isna().any():
                raise ValueError("현금 수익률이 중간에 비어 있습니다.")
            self._inverse[(m, expense)] = synthetic_inverse(
                self.underlying.loc[:last], rf.loc[:last].fillna(0.0), expense, BORROW, m
            )
        return self._inverse[(m, expense)]


def load_assets(fix_kospi: bool = False) -> list[Asset]:
    spy = read("spy_daily.csv")
    sp500tr = read("sp500tr_daily.csv")
    kospi = read("kospi_daily.csv")
    if fix_kospi:
        day, close = KOSPI_FIX
        kospi.loc[day, "close"] = close
    if not spy.index.isin(sp500tr.index).all():
        raise ValueError("SPY 거래일 중 ^SP500TR에 없는 날이 있습니다.")
    # 직전 실험과 같은 합성 ETF(주식 쪽)와, 보수 없는 같은 지수(인버스의 기초)
    kospi_etf = synthetic_index_etf(kospi, 0.015, "december", 0.0015, "cd")
    kospi_index = synthetic_index_etf(kospi, 0.015, "december", 0.0, "cd")
    common_notes = ["매매 비용: 편도 0.1% (매도와 매수마다). 주식↔인버스 전환은 매도 + 매수라 두 번 냅니다. 연간 매매는 매수와 매도를 각각 한 번으로 셉니다."]
    return [
        Asset(
            key="spy", name="SPY",
            price=spy["close"],
            signal_price=snap_to_tick(spy["close"], PRICE_TICK["SPY"]),
            data=spy[["close", "dividend", "rf"]],
            underlying=pd.DataFrame({"close": sp500tr["close"].reindex(spy.index), "dividend": 0.0}),
            inverse_expense=SH_EXPENSE, interest=SPY_INTEREST,
            period=("1994-01-03", "2012-12-31"),
            pre_tax=NO_TAX,
            after_tax={**NO_TAX, "dividend_withholding": 0.15, "capital_gains_tax": 0.22, "inverse_tax": "pooled"},
            price_label="SPY 일별 종가 (배당 미포함, 로그 눈금)",
            notes=[
                *common_notes,
                f"인버스(SH 기준) 합성: -1배 × S&P 500 총수익 지수 일별 수익률 + {SPY_INTEREST:g} × 현금 수익률 − 보수 연 {SH_EXPENSE:.2%}. "
                "현금은 Ken French RF, 달러 기준.",
                "세후: 배당 원천징수 15%, 양도소득세 22%. SPY와 SH의 매매차익을 합친 연간 실현 순이익(같은 해 손실만 상계, 기본공제 없음).",
                "세후 표의 단순 보유는 아직 내지 않은 양도세가 빠진 값입니다(CAGR (청산 후) 제외). 단순 보유에 유리한 쪽입니다.",
            ],
            chart_costs=f"비용: 편도 매매 비용 0.1%, 인버스 보수 연 {SH_EXPENSE:.2%}, 현금은 Ken French RF, 달러 기준.",
        ),
        Asset(
            key="kospi", name="KOSPI",
            price=kospi["close"],
            signal_price=snap_to_tick(kospi["close"], PRICE_TICK["^KS11"]),
            data=kospi_etf,
            underlying=kospi_index[["close", "dividend"]],
            inverse_expense=KODEX_INVERSE_EXPENSE, interest=KOSPI_INTEREST,
            period=("1998-01-05", "2012-12-28"),
            pre_tax=NO_TAX,
            after_tax={**NO_TAX, "dividend_withholding": 0.154, "interest_tax": 0.154,
                       "inverse_tax": "holding_period", "inverse_gain_tax": 0.154},
            price_label="KOSPI 지수 일별 종가 (배당 미포함, 로그 눈금)",
            notes=[
                *common_notes,
                "주식 쪽 합성 ETF: KOSPI 가격 지수 + 배당 연 1.5%(12월 마지막 거래일 전날 일괄) − 보수 연 0.15% (직전 실험과 같음).",
                f"인버스(KODEX 인버스 기준) 합성: -1배 × (KOSPI + 배당 연 1.5%) 일별 수익률 + {KOSPI_INTEREST:g} × 현금 수익률 − 보수 연 "
                f"{KODEX_INVERSE_EXPENSE:.2%}. 현금은 CD 91일물, 원화 기준.",
                "세후: 주식 쪽 매매차익 비과세, 분배금과 현금 이자 15.4%. 인버스는 팔 때마다 min(매매차익, 과표기준가 증가분) × 15.4% "
                "(과표에는 담보 이자 − 보수만 쌓임, 손실 상계 없음).",
            ],
            chart_costs=f"비용: 편도 매매 비용 0.1%, 인버스 보수 연 {KODEX_INVERSE_EXPENSE:.2%}, 주식 쪽 합성 ETF(배당 연 1.5%, 보수 연 0.15%), "
                        "현금은 CD 91일물, 원화 기준.",
        ),
    ]


# ---------------------------------------------------------------- 실행

KINDS = {"long_short": "롱/숏", "long_cash": "롱/현금", "hold": "단순 보유"}


def holding_for(asset: Asset, kind: str, window: int | None, lag: int) -> pd.Series:
    if kind == "hold":
        return pd.Series(1.0, index=asset.data.index)
    weeks = period_last_days(asset.signal_price.index, WEEK)
    signal = moving_average_signal(asset.signal_price, weeks, window)
    if kind == "long_short":
        signal = long_short(signal)
    return holding_from_signal(signal, asset.data.index, lag)


def run(asset: Asset, kind: str, window: int | None, lag: int, tax: dict, trade: float = TRADE,
        m: float | None = None, expense: float | None = None):
    holding = holding_for(asset, kind, window, lag)
    start, end = asset.period
    inverse = asset.inverse(m, expense) if kind == "long_short" else None
    result = run_backtest(asset.data, holding, start, end, Costs(trade=trade, **tax), inverse)
    return result, metrics(result), holding


def strategy_name(kind: str, window: int | None) -> str:
    return KINDS[kind] if window is None else f"{window}주 {KINDS[kind]}"


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
        "주식 / 인버스 보유": f"{m['time_in_market']:.0%} / {m['time_short']:.0%}",
    }


def standard_tables(asset: Asset, lag: int) -> list[str]:
    lines = []
    for label, tax in [("세전 (매매 비용과 보수만)", asset.pre_tax), ("세후", asset.after_tax)]:
        rows = []
        for window in WINDOWS:
            for kind in ("long_short", "long_cash"):
                rows.append(row(strategy_name(kind, window), run(asset, kind, window, lag, tax)[1]))
        rows.append(row("단순 보유", run(asset, "hold", None, lag, tax)[1]))
        lines += [f"#### {label}", "", table(rows), ""]
    return lines


def criteria_rows(asset: Asset) -> list[dict]:
    """사전 판단 기준 1(단순 보유 대비)과 2(롱/현금 대비)를 세후 값으로 확인합니다."""
    hold = run(asset, "hold", None, MAIN_LAG, asset.after_tax)[1]
    rows = []
    for lag in LAGS:
        for window in WINDOWS:
            ls = run(asset, "long_short", window, lag, asset.after_tax)[1]
            lc = run(asset, "long_cash", window, lag, asset.after_tax)[1]
            a = ls["sharpe"] > hold["sharpe"]
            cagr_gap = ls["cagr_after_liquidation"] - hold["cagr_after_liquidation"]
            mdd_gap = ls["mdd"] - hold["mdd"]
            b = cagr_gap >= -0.01 and mdd_gap >= 0.15
            c = ls["sharpe"] > lc["sharpe"]
            rows.append({
                "체결": "신호 낸 날 종가" if lag == 0 else "다음 거래일 종가 (참고)",
                "전략": f"{window}주 롱/숏",
                "샤프 (롱/숏 / 단순 보유 / 롱/현금)": f"{ls['sharpe']:.2f} / {hold['sharpe']:.2f} / {lc['sharpe']:.2f}",
                "(가) 샤프 > 단순 보유": "충족" if a else "못함",
                "청산 후 CAGR 차이 (단순 보유 대비)": f"{cagr_gap:+.2%}",
                "MDD 차이 (단순 보유 대비)": f"{mdd_gap:+.1%}",
                "(나)": "충족" if b else "못함",
                "기준 1": "충족" if a or b else "못함",
                "기준 2: 샤프 > 롱/현금": "충족" if c else "못함",
            })
    return rows


def cost_decomposition(asset: Asset, window: int) -> list[str]:
    """비용을 하나씩 켜며 청산 후 CAGR과 샤프가 얼마나 줄어드는지 봅니다 (신호 낸 날 체결)."""
    steps = [
        ("1. 비용 없음", 0.0, 0.0, asset.pre_tax),
        ("2. + 매매 비용 (편도 0.1%)", TRADE, 0.0, asset.pre_tax),
        (f"3. + 인버스 보수 (연 {asset.inverse_expense:.2%}) = 세전 기본", TRADE, asset.inverse_expense, asset.pre_tax),
        ("4. + 세금 = 세후 기본", TRADE, asset.inverse_expense, asset.after_tax),
    ]
    rows, previous = [], {}
    for label, trade, expense, tax in steps:
        r = {"단계": label}
        for kind, w in [("long_short", window), ("long_cash", window), ("hold", None)]:
            result, m, _ = run(asset, kind, w, MAIN_LAG, tax, trade=trade, expense=expense)
            name = KINDS[kind]
            cagr = m["cagr_after_liquidation"]
            change = "" if kind not in previous else f" ({cagr - previous[kind]:+.2%}p)"
            r[f"{name} 청산 후 CAGR"] = f"{cagr:.2%}{change}"
            r[f"{name} 샤프"] = f"{m['sharpe']:.2f}"
            previous[kind] = cagr
        rows.append(r)

    # 매매 비용 검산: 연간 체결 횟수 × 편도 비용 ≈ 2단계의 CAGR 감소
    checks = []
    for kind, w in [("long_short", window), ("long_cash", window)]:
        _, m, _ = run(asset, kind, w, MAIN_LAG, asset.pre_tax)
        checks.append(f"{KINDS[kind]} 연 {m['trades_per_year']:.1f}회 × 0.1% = 연 {m['trades_per_year'] * TRADE:.2%}")
    drag = rebalancing_effect(asset, window)
    return [
        f"#### {window}주", "", table(rows), "",
        "- 괄호는 앞 단계 대비 변화입니다. 3단계는 롱/현금과 단순 보유에는 해당이 없어 2단계와 같습니다.",
        "- 매매 비용 검산(체결 횟수 × 편도 비용, 2단계 감소와 비슷해야 함): " + ", ".join(checks) + ".",
        f"- 매일 비율을 다시 맞추는 효과(숏 구간 {drag['spans']}회): 일별 -1배 복리와 비율을 다시 맞추지 않는 공매도(구간 수익률의 반대)의 "
        f"로그 차이 합이 {drag['total']:+.3f}, 연율 {drag['annual']:+.2%}p입니다. 양수면 매일 다시 맞추는 쪽이 유리했다는 뜻입니다. "
        "보수와 이자는 빼고 비교했고, 이 효과는 위 표의 모든 단계에 이미 들어 있습니다.",
        "",
    ]


def short_spans(asset: Asset, window: int, lag: int = MAIN_LAG) -> tuple[pd.DataFrame, object]:
    """세전 롱/숏의 인버스 보유 구간. trade_list의 가격 자리에 합성 인버스 가격을 넣습니다."""
    result, _, holding = run(asset, "long_short", window, lag, asset.pre_tax)
    days = result.daily.index
    short = (holding.reindex(days) == -1.0).astype(float)
    spans = trade_list(short, asset.inverse()["close"].reindex(days))
    return spans, result


def rebalancing_effect(asset: Asset, window: int) -> dict:
    spans, result = short_spans(asset, window)
    under = asset.underlying.reindex(result.daily.index)
    growth = ((under["close"] + under["dividend"]) / under["close"].shift(1)).to_numpy()
    total = 0.0
    for t in spans.itertuples(index=False):
        locs = slice(result.daily.index.get_loc(t.buy_date) + 1, result.daily.index.get_loc(t.sell_date) + 1)
        daily = np.prod(2 - growth[locs])  # 매일 -1배: (1 − TR) = 2 − (1 + TR)
        static = 2 - np.prod(growth[locs])  # 구간 수익률의 반대
        total += np.log(daily / static)
    years = (result.daily.index[-1] - result.start_date).days / 365.25
    return {"spans": len(spans), "total": total, "annual": total / years}


def short_trade_rows(asset: Asset, window: int) -> tuple[list[dict], dict]:
    spans, result = short_spans(asset, window)
    rf = result.daily["rf"]
    rows, log_inverse, log_cash = [], [], []
    for number, t in enumerate(spans.itertuples(index=False), start=1):
        cash = float(np.prod(1 + rf.loc[t.buy_date:t.sell_date].iloc[1:])) - 1
        log_inverse.append(np.log(1 + t.from_buy))
        log_cash.append(np.log(1 + cash))
        rows.append({
            "번호": number,
            "인버스 매수일": f"{t.buy_date:%Y-%m-%d}",
            "인버스 매도일": "(기간 끝까지 보유)" if t.open else f"{t.sell_date:%Y-%m-%d}",
            "보유 주": f"{t.weeks_held:.0f}",
            "지수 변화": f"{asset.price[t.sell_date] / asset.price[t.buy_date] - 1:+.1%}",
            "인버스 수익률": f"{t.from_buy:+.1%}",
            "같은 기간 현금": f"{cash:+.1%}",
            "현금 대비": f"{(1 + t.from_buy) / (1 + cash) - 1:+.1%}",
        })
    r = spans["from_buy"]
    excess = np.array(log_inverse) - np.array(log_cash)
    summary = {
        "전략": f"{window}주 롱/숏",
        "인버스 구간": len(spans),
        "이긴 비율 (수익률 > 0)": f"{(r > 0).mean():.0%}",
        "현금보다 나은 비율": f"{(excess > 0).mean():.0%}",
        "평균": f"{r.mean():+.1%}",
        "중앙값": f"{r.median():+.1%}",
        "가장 큰 이익": f"{r.max():+.1%}",
        "가장 큰 손실": f"{r.min():+.1%}",
        "로그 수익 합 (인버스)": f"{sum(log_inverse):+.3f}",
        "로그 수익 합 (같은 기간 현금)": f"{sum(log_cash):+.3f}",
        "현금 대비 합": f"{excess.sum():+.3f}",
    }
    return rows, summary


def break_even(asset: Asset, window: int, other: str) -> str:
    """롱/숏의 세후 샤프가 other(롱/현금 또는 단순 보유)와 같아지는 편도 매매 비용."""
    def gap(trade: float) -> float:
        ls = run(asset, "long_short", window, MAIN_LAG, asset.after_tax, trade=trade)[1]["sharpe"]
        o = run(asset, other, None if other == "hold" else window, MAIN_LAG, asset.after_tax, trade=trade)[1]["sharpe"]
        return ls - o

    low, high = 0.0, BREAK_EVEN_MAX
    g_low, g_high = gap(low), gap(high)
    if g_low <= 0:
        return "0%에서도 낮음"
    if g_high > 0:
        return f"{BREAK_EVEN_MAX:.0%}에서도 높음"
    for _ in range(30):
        middle = (low + high) / 2
        if gap(middle) > 0:
            low = middle
        else:
            high = middle
    return f"{(low + high) / 2:.3%}"


def break_even_tables(asset: Asset) -> list[str]:
    lines = []
    for window in WINDOWS:
        rows = []
        for trade in TRADE_GRID:
            r = {"편도 매매 비용": f"{trade:.2%}"}
            for kind, w in [("long_short", window), ("long_cash", window), ("hold", None)]:
                m = run(asset, kind, w, MAIN_LAG, asset.after_tax, trade=trade)[1]
                r[f"{KINDS[kind]} 청산 후 CAGR / 샤프"] = f"{m['cagr_after_liquidation']:.2%} / {m['sharpe']:.2f}"
            rows.append(r)
        lines += [
            f"#### {window}주", "", table(rows), "",
            f"- 롱/숏 세후 샤프가 롱/현금과 같아지는 편도 비용: {break_even(asset, window, 'long_cash')}",
            f"- 롱/숏 세후 샤프가 단순 보유와 같아지는 편도 비용: {break_even(asset, window, 'hold')}",
            "",
        ]
    return lines


def interest_table(asset: Asset) -> list[str]:
    rows = []
    for window in WINDOWS:
        for m_value in INTEREST_GRID:
            label = f"{window}주 롱/숏, m = {m_value:g}" + (" (기본)" if m_value == asset.interest else "")
            pre = run(asset, "long_short", window, MAIN_LAG, asset.pre_tax, m=m_value)[1]
            post = run(asset, "long_short", window, MAIN_LAG, asset.after_tax, m=m_value)[1]
            rows.append({
                "구분": label,
                "세전 CAGR": f"{pre['cagr']:.2%}",
                "세전 샤프": f"{pre['sharpe']:.2f}",
                "세후 청산 후 CAGR": f"{post['cagr_after_liquidation']:.2%}",
                "세후 샤프": f"{post['sharpe']:.2f}",
                "MDD (세후)": f"{post['mdd']:.1%}",
            })
    for window in WINDOWS:
        post = run(asset, "long_cash", window, MAIN_LAG, asset.after_tax)[1]
        rows.append({"구분": f"참고: {window}주 롱/현금", "세전 CAGR": "", "세전 샤프": "",
                     "세후 청산 후 CAGR": f"{post['cagr_after_liquidation']:.2%}", "세후 샤프": f"{post['sharpe']:.2f}",
                     "MDD (세후)": f"{post['mdd']:.1%}"})
    return [table(rows), ""]


def yearly_table(asset: Asset) -> list[str]:
    """각 해의 마지막 자산 / 전해 마지막 자산 − 1. 첫해의 출발은 시작 전날(진입 비용 전 1.0)입니다."""
    curves = {"단순 보유": run(asset, "hold", None, MAIN_LAG, asset.pre_tax)[0].daily["equity"]}
    for window in WINDOWS:
        for kind in ("long_cash", "long_short"):
            curves[strategy_name(kind, window)] = run(asset, kind, window, MAIN_LAG, asset.pre_tax)[0].daily["equity"]
    rows = []
    first_index = curves["단순 보유"].index
    for year in sorted(set(first_index[1:].year)):
        r = {"연도": str(year)}
        for name, equity in curves.items():
            curve = equity.copy()
            curve.iloc[0] = 1.0
            begin = curve[curve.index.year < year].iloc[-1]
            end = curve[curve.index.year == year].iloc[-1]
            r[name] = f"{end / begin - 1:+.1%}"
        rows.append(r)
    return [table(rows), ""]


def fix_reference(asset_fixed: Asset) -> list[str]:
    rows = []
    for window in WINDOWS:
        for kind in ("long_short", "long_cash"):
            m = run(asset_fixed, kind, window, MAIN_LAG, asset_fixed.after_tax)[1]
            rows.append({"구분": strategy_name(kind, window), "세후 청산 후 CAGR": f"{m['cagr_after_liquidation']:.2%}",
                         "세후 샤프": f"{m['sharpe']:.2f}", "세후 MDD": f"{m['mdd']:.1%}"})
    m = run(asset_fixed, "hold", None, MAIN_LAG, asset_fixed.after_tax)[1]
    rows.append({"구분": "단순 보유", "세후 청산 후 CAGR": f"{m['cagr_after_liquidation']:.2%}",
                 "세후 샤프": f"{m['sharpe']:.2f}", "세후 MDD": f"{m['mdd']:.1%}"})
    return [table(rows), ""]


def asset_report(asset: Asset, asset_fixed: Asset | None) -> list[str]:
    start, end = asset.period
    lines = [f"## {asset.name}: 설계용 기간 ({start} ~ {end})", "", *[f"- {n}" for n in asset.notes],
             "- 샤프의 무위험 수익률은 세전 현금 수익률입니다. 'CAGR (청산 후)'는 기간 끝에 모두 팔고 세금까지 정산한 뒤의 값입니다.", ""]
    lines += ["### 기본 결과: 신호 낸 날 종가 체결", "", *standard_tables(asset, 0)]
    lines += ["### 참고: 다음 거래일 종가 체결", "", *standard_tables(asset, 1)]
    lines += ["### 사전 판단 기준 확인 (세후)", "", table(criteria_rows(asset)), "",
              "- 기준 1: (가) 또는 (나). (나)는 청산 후 CAGR이 단순 보유보다 1%p 이내로 낮고 MDD가 15%p 이상 작은 것입니다.",
              "- 기준 2: 같은 체결 방식의 롱/현금보다 샤프가 높은 것입니다. 단순 보유 샤프는 체결 방식과 상관없이 같습니다.", ""]
    lines += ["### 거래 비용 검증 1: 비용 분해 (신호 낸 날 체결)", ""]
    for window in WINDOWS:
        lines += cost_decomposition(asset, window)
    lines += ["### 거래 비용 검증 2: 편도 매매 비용을 바꿨을 때 (세후, 신호 낸 날 체결)", "", *break_even_tables(asset)]
    lines += ["### 거래 비용 검증 3: 인버스 이자 가정 (신호 낸 날 체결)", "",
              "m은 현금 수익률의 배수입니다. 2는 담보 이자 + 선물·스왑 가격의 이자, 1은 담보 이자만, 0은 이자 없음입니다.", "",
              *interest_table(asset)]
    summaries, trade_tables = [], []
    for window in WINDOWS:
        rows, summary = short_trade_rows(asset, window)
        summaries.append(summary)
        trade_tables += [f"#### {window}주 롱/숏", "", table(rows), ""]
    lines += ["### 인버스 구간 요약 (세전, 신호 낸 날 체결)", "", table(summaries), "",
              "- 인버스 수익률은 합성 인버스 가격의 매수일 종가 → 매도일 종가 변화입니다. 보수와 이자는 들어 있고, 매매 비용은 빠져 있습니다.",
              "- '같은 기간 현금'은 롱/현금 전략이 그 구간에 받았을 세전 현금 수익률입니다. '현금 대비'가 숏을 더한 몫입니다.", ""]
    lines += ["### 연도별 수익률 (세전, 신호 낸 날 체결, 배당 재투자, 청산 전)", "", *yearly_table(asset)]
    if asset_fixed is not None:
        day, close = KOSPI_FIX
        lines += [f"### 참고: {day} 종가를 {close}로 고친 결과 (세후, 신호 낸 날 체결)", "", *fix_reference(asset_fixed)]
    lines += ["### 인버스 구간 목록 (세전, 신호 낸 날 체결)", "", *trade_tables]
    return lines


def table(rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(str(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([header, sep, *body])


# ---------------------------------------------------------------- 차트


def chart(asset: Asset) -> list[Path]:
    """롱/숏(세전, 신호 낸 날 체결)의 보유 구간, 이격도, 자산 가치. 롱/현금 차트는 직전 실험의 것을 링크합니다."""
    start, end = asset.period
    weeks = period_last_days(asset.price.index, WEEK)
    weekly = asset.price.reindex(weeks)
    hold_result, hold, _ = run(asset, "hold", None, MAIN_LAG, asset.pre_tax)
    paths = []
    for window in WINDOWS:
        average = weekly.rolling(window).mean()
        result, m, _ = run(asset, "long_short", window, MAIN_LAG, asset.pre_tax)
        _, summary = short_trade_rows(asset, window)
        paths.append(holding_chart(
            REPORT_DIR / f"{PERIOD}-holdings-{asset.key}-sma{window}w-long-short.png",
            title=f"{asset.name} {window}주 이동평균 롱/숏: 보유 구간 (설계용 기간 {start} ~ {end})",
            subtitle=[
                f"세전  ·  {metrics_line(m, hold)}",
                f"규칙: 주 마지막 거래일 종가가 {window}주 이동평균보다 높으면 주식 100%, 아니면 -1배 인버스 100%. "
                "신호 낸 날 종가에 체결 (사용자 지정. 저장소 기본 규칙은 다음 거래일 종가 체결).",
                f"인버스 구간 {summary['인버스 구간']}회: 이긴 비율 {summary['이긴 비율 (수익률 > 0)']}, 평균 {summary['평균']}, "
                f"가장 큰 이익 {summary['가장 큰 이익']}, 가장 큰 손실 {summary['가장 큰 손실']} (보수 포함, 매매 비용 전)",
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
            strategy_label=f"{window}주 롱/숏",
        ))
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracking", action="store_true", help="합성 인버스와 실제 상품을 비교합니다")
    parser.add_argument("--chart", action="store_true", help="보유 구간 차트만 만듭니다")
    args = parser.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    if args.tracking:
        lines = tracking()
        out = REPORT_DIR / "inverse-tracking.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines), encoding="utf-8")
        print("\n".join(lines))
        print(f"저장: {out.relative_to(ROOT)}")
        return

    assets = load_assets()
    if args.chart:
        for asset in assets:
            for path in chart(asset):
                print(f"저장: {path.relative_to(ROOT)}")
        return

    fixed = {a.key: a for a in load_assets(fix_kospi=True)}
    lines = ["# 20주·40주 이동평균 롱/숏(인버스): 설계용 기간", "",
             "규칙: 주 마지막 거래일 종가가 N주 이동평균보다 높으면 주식 100%, 아니면 -1배 인버스 ETF 100% (N = 20, 40).",
             "비교: 같은 신호의 롱/현금(이동평균 아래에서 현금, 직전 실험의 기본 전략)과 단순 보유. 같은 기간, 같은 비용·세금입니다.",
             "**체결: 신호 낸 날 종가가 주 기준 (사용자 지정).** 저장소 기본 규칙(다음 거래일 종가 체결)의 결과는 참고로 함께 냅니다.",
             "인버스는 지수로 합성했습니다. 실제 상품과의 비교는 [inverse-tracking.md](inverse-tracking.md)에 있습니다. "
             "파라미터(이동평균 기간) 민감도는 사용자 요청으로 하지 않았습니다.", ""]
    for asset in assets:
        lines += ["", *asset_report(asset, fixed[asset.key] if asset.key == "kospi" else None)]

    out = REPORT_DIR / f"{PERIOD}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"저장: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
