"""보유 구간 차트.

지수(가격) 선그래프 위에 전략이 주식을 보유한 구간을 음영으로 칠하고, 전략에 쓴 보조지표를 그 아래 칸에 따로 그립니다.
그리는 기간은 백테스트 결과(run_backtest의 daily)의 기간과 같습니다. 그 밖의 데이터는 잘라 냅니다.
"""

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

# 색. 계열은 가격, 기준선(이동평균, 지표 기준값), 보유 구간 세 가지입니다.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
PRICE = "#2a78d6"
REFERENCE = "#eb6834"
HOLDING = "#1baf7a"
HOLDING_ALPHA = 0.16


@dataclass(frozen=True)
class Panel:
    """지수 차트 아래에 붙는 보조지표 칸 하나."""

    title: str
    values: pd.Series
    reference: float  # 기준선 값 (이격도 0, ADX 25 등)
    reference_label: str
    percent: bool = False  # True면 0.05를 5%로 표시합니다


def holding_spans(exposure: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """일별 노출을 보유 구간 (매수 체결일, 매도 체결일)의 목록으로 바꿉니다.

    exposure: run_backtest의 daily["holding"]. d일 값이 1이면 d-1일 종가부터 d일 종가까지 주식을 들고 있었습니다.
    그래서 노출이 1인 연속 구간 [i, j]는 index[i-1] 종가에 사서 index[j] 종가에 판 구간입니다.
    첫 행은 시작 전날(기준일)이라 노출 값을 쓰지 않습니다. 기간 끝까지 보유했다면 구간 끝은 마지막 날입니다.
    """
    held = exposure.to_numpy()[1:] == 1.0
    padded = np.concatenate([[False], held, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    # held[k]는 index[k + 1]의 노출입니다. held[s:e]가 1이면 index[s] ~ index[e] 구간입니다.
    return [(exposure.index[s], exposure.index[e]) for s, e in zip(edges[::2], edges[1::2])]


def metrics_line(strategy: dict, hold: dict) -> str:
    """부제에 넣을 한 줄 요약. 괄호 안은 같은 비용 가정의 단순 보유입니다."""
    return (
        f"CAGR {strategy['cagr']:.2%} (단순 보유 {hold['cagr']:.2%})  ·  "
        f"MDD {strategy['mdd']:.1%} ({hold['mdd']:.1%})  ·  "
        f"샤프 {strategy['sharpe']:.2f} ({hold['sharpe']:.2f})  ·  "
        f"보유 비율 {strategy['time_in_market']:.0%}  ·  연간 매매 {strategy['trades_per_year']:.1f}회"
    )


def _log_ticks(low: float, high: float) -> list[float]:
    """로그 눈금에 쓸 깔끔한 값. 너무 많으면 1, 2, 5 배수만 남깁니다."""
    ticks = [
        m * 10.0**k
        for k in range(int(np.floor(np.log10(low))), int(np.ceil(np.log10(high))) + 1)
        for m in (1, 1.5, 2, 3, 5, 7)
    ]
    inside = [t for t in ticks if low <= t <= high]
    if len(inside) > 7:
        inside = [t for t in inside if round(t / 10.0 ** np.floor(np.log10(t)), 6) in (1, 2, 5)]
    return inside


def _korean_font() -> str:
    """설치된 한글 글꼴 하나. 없으면 기본 글꼴을 씁니다 (한글이 깨질 수 있음)."""
    installed = {font.name for font in font_manager.fontManager.ttflist}
    return next((name for name in ("Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK KR") if name in installed), "sans-serif")


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=AXIS, labelcolor=INK_SECONDARY, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def start_at_one(equity: pd.Series) -> pd.Series:
    """run_backtest의 일별 자산에서 첫날(시작 전날) 값을 1.0으로 바꿉니다.

    첫날 값은 진입 비용을 낸 뒤의 값이라, metrics의 MDD 계산처럼 비용을 내기 전 1.0에서 시작하게 합니다.
    """
    curve = equity.copy()
    curve.iloc[0] = 1.0
    return curve


def _draw_equity(ax: plt.Axes, strategy: pd.Series, hold: pd.Series, strategy_label: str, height: float) -> None:
    """자산 가치 칸. 두 곡선 끝에 최종값을 적고, 값이 가까우면 글자가 겹치지 않게 벌립니다."""
    curves = [("단순 보유", hold, PRICE, "단순 보유 (같은 비용)"), ("전략", strategy, HOLDING, f"전략: {strategy_label}")]
    for _, curve, color, label in curves:
        ax.plot(curve.index, curve.to_numpy(), color=color, linewidth=1.4, solid_joinstyle="round", label=label)
        ax.plot(curve.index[-1], curve.iloc[-1], "o", markersize=5.5, color=color, markeredgecolor=SURFACE,
                markeredgewidth=1.5, clip_on=False, zorder=5)
    ax.axhline(1.0, color=AXIS, linewidth=0.8)

    low = min(curve.min() for _, curve, _, _ in curves) * 0.92
    high = max(curve.max() for _, curve, _, _ in curves) * 1.08
    ax.set_yscale("log")
    ax.set_ylim(low, high)
    ax.yaxis.set_major_locator(FixedLocator(_log_ticks(low, high)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())

    # 끝값 글자의 세로 위치 (칸 높이 대비 비율). 0.17인치보다 가까우면 가운데를 기준으로 벌립니다.
    position = [(np.log(curve.iloc[-1]) - np.log(low)) / (np.log(high) - np.log(low)) for _, curve, _, _ in curves]
    gap = 0.17 / height
    if abs(position[0] - position[1]) < gap:
        middle, sign = sum(position) / 2, 1 if position[0] >= position[1] else -1
        position = [middle + sign * gap / 2, middle - sign * gap / 2]
    for (name, curve, _, _), y in zip(curves, position):
        ax.text(1.008, y, f"{name} {curve.iloc[-1]:.2f}", transform=ax.transAxes, va="center", ha="left",
                fontsize=9, color=INK_SECONDARY)

    ax.set_title("자산 가치: 시작일 = 1, 배당 재투자, 세전, 기간 끝 청산 전 (로그 눈금)", loc="left", fontsize=10, color=INK, pad=5)
    ax.legend(loc="lower right", bbox_to_anchor=(1, 1.0), borderaxespad=0.3, borderpad=0, handletextpad=0.5,
              columnspacing=1.6, frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, ncols=2)


def holding_chart(
    path: Path,
    title: str,
    subtitle: list[str],
    price: pd.Series,
    price_label: str,
    exposure: pd.Series,
    panels: list[Panel],
    overlay: pd.Series | None = None,
    overlay_label: str = "",
    equity: pd.Series | None = None,
    hold_equity: pd.Series | None = None,
    strategy_label: str = "",
) -> Path:
    """가격 선, 보유 구간 음영, 보조지표 칸, 자산 가치 칸을 그려 PNG로 저장합니다.

    exposure의 첫날(시작 전날)부터 마지막 날까지만 그립니다. 가격과 지표는 그 밖의 값을 잘라 냅니다.
    equity와 hold_equity(run_backtest의 daily["equity"])를 넘기면 맨 아래에 전략과 단순 보유의 자산 가치를 그립니다.
    """
    first, last = exposure.index[0], exposure.index[-1]
    if (equity is None) != (hold_equity is None):
        raise ValueError("equity와 hold_equity는 함께 넘겨야 합니다.")
    for curve in (equity, hold_equity):
        if curve is not None and not curve.index.equals(exposure.index):
            raise ValueError("자산 가치의 날짜가 exposure와 다릅니다. 같은 기간의 백테스트 결과를 넘기세요.")

    def clip(series: pd.Series) -> pd.Series:
        return series[(series.index >= first) & (series.index <= last)]

    price = clip(price)
    spans = holding_spans(exposure)

    plt.rcParams.update({"font.family": _korean_font(), "axes.unicode_minus": False})
    # 칸 높이와 머리(제목, 부제, 범례) 높이. 단위는 인치이고, 머리는 부제 줄 수에 따라 늘어납니다.
    heights = [4.6] + [1.7] * len(panels) + ([2.4] if equity is not None else [])
    header = 1.15 + 0.25 * len(subtitle)
    height = header + sum(heights) + 0.55
    fig, axes = plt.subplots(
        len(heights), 1, sharex=True, figsize=(14, height), facecolor=SURFACE,
        gridspec_kw={"height_ratios": heights},
    )
    axes = np.atleast_1d(axes)
    fig.subplots_adjust(left=0.06, right=0.92, top=1 - header / height, bottom=0.5 / height, hspace=0.2)

    for ax in axes:
        _style(ax)
        for start, end in spans:
            ax.axvspan(start, end, color=HOLDING, alpha=HOLDING_ALPHA, linewidth=0)

    # 지수 칸
    main = axes[0]
    main.plot(price.index, price.to_numpy(), color=PRICE, linewidth=1.1, solid_joinstyle="round")
    handles = [Line2D([], [], color=PRICE, linewidth=2, label=price_label)]
    if overlay is not None:
        overlay = clip(overlay)
        main.plot(overlay.index, overlay.to_numpy(), color=REFERENCE, linewidth=0.9)
        handles.append(Line2D([], [], color=REFERENCE, linewidth=2, label=overlay_label))
    handles.append(Patch(facecolor=HOLDING, alpha=HOLDING_ALPHA * 2.5, linewidth=0, label="주식 보유 구간"))
    main.set_yscale("log")
    low, high = price.min(), price.max()
    if overlay is not None:
        low, high = min(low, overlay.min()), max(high, overlay.max())
    main.set_ylim(low * 0.95, high * 1.05)
    main.yaxis.set_major_locator(FixedLocator(_log_ticks(low * 0.95, high * 1.05)))
    main.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    main.yaxis.set_minor_locator(NullLocator())
    main.yaxis.set_minor_formatter(NullFormatter())
    # 범례는 가격선을 가리지 않도록 칸 바깥 위에 둡니다.
    main.legend(handles=handles, loc="lower left", bbox_to_anchor=(0, 1.0), borderaxespad=0.3, borderpad=0,
                handletextpad=0.5, columnspacing=1.6, frameon=False, fontsize=9.5, labelcolor=INK_SECONDARY, ncols=3)

    # 보조지표 칸
    for ax, panel in zip(axes[1 : 1 + len(panels)], panels):
        values = clip(panel.values)
        ax.plot(values.index, values.to_numpy(), color=INK_SECONDARY, linewidth=1.0)
        ax.axhline(panel.reference, color=REFERENCE, linewidth=1.0)
        label_at = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(1.006, panel.reference, panel.reference_label, transform=label_at, va="center", ha="left",
                fontsize=9, color=INK_SECONDARY)
        ax.set_title(panel.title, loc="left", fontsize=10, color=INK, pad=5)
        if panel.percent:
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))

    # 자산 가치 칸
    if equity is not None:
        _draw_equity(axes[-1], start_at_one(equity), start_at_one(hold_equity), strategy_label, heights[-1])

    years = (last - first).days / 365.25
    axes[-1].xaxis.set_major_locator(mdates.YearLocator(base=1 if years <= 10 else 2))
    axes[-1].xaxis.set_minor_locator(mdates.YearLocator(base=1))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axes[-1].set_xlim(first, last)

    fig.text(0.06, 1 - 0.32 / height, title, fontsize=15, weight="bold", color=INK, va="top")
    for i, line in enumerate(subtitle):
        fig.text(0.06, 1 - (0.68 + 0.24 * i) / height, line, fontsize=9.5, color=INK_SECONDARY, va="top")

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path
