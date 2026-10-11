"""백테스트 엔진.

일별 데이터(종가, 배당, 현금 수익률)에서 주간·월간 신호를 받아, 정한 시점의 종가에 체결합니다.
보유 상태는 100% 주식 또는 100% 현금입니다.

시점 약속:
- holding_after_close[d]: d일 종가에 체결을 마친 뒤의 보유 상태. d일 종가까지의 정보로 정해집니다.
- d일 하루의 수익률(d-1일 종가 → d일 종가)은 holding_after_close[d-1] 상태로 계산합니다.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class Costs:
    """비용과 세금 가정.

    - trade: 편도 매매 비용 (수수료와 슬리피지). 사거나 팔 때마다 거래 금액에 대해 냅니다.
    - dividend_withholding: 배당 원천징수율
    - capital_gains_tax: 연간 실현 순이익에 매기는 양도소득세율. 같은 해 손실만 상계하고, 그해 마지막 거래일에 뺍니다.
    """

    trade: float = 0.001
    dividend_withholding: float = 0.0
    capital_gains_tax: float = 0.0


# ---------------------------------------------------------------- 신호


def period_last_days(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """각 기간의 마지막 거래일. freq는 "W-FRI"(금요일로 끝나는 주) 또는 "M"(월).

    휴장일은 미리 공시되므로, 실제 데이터로 마지막 거래일을 정해도 미래 정보가 아닙니다.
    단, 데이터 맨 끝의 기간은 끝나지 않았을 수 있으므로 평가 기간 끝은 데이터 끝보다 앞에 둡니다.
    """
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(days.groupby(index.to_period(freq)).last().to_numpy(), name=index.name)


def moving_average_signal(price: pd.Series, signal_days: pd.DatetimeIndex, window: int) -> pd.Series:
    """signal_days의 종가가 최근 window개 signal_days 종가(그날 포함)의 평균보다 높으면 1, 아니면 0.

    평균을 낼 데이터가 모자란 준비 기간은 NaN입니다.
    """
    sampled = price.reindex(signal_days)
    if sampled.isna().any():
        raise ValueError("신호 날짜에 가격이 없습니다.")
    average = sampled.rolling(window).mean()
    signal = (sampled > average).astype(float)
    signal[average.isna()] = np.nan
    return signal


def holding_from_signal(signal: pd.Series, index: pd.DatetimeIndex, lag: int = 1) -> pd.Series:
    """신호를 각 거래일 종가 직후의 보유 상태로 바꿉니다.

    lag=1: 신호를 낸 날의 다음 거래일 종가에 체결합니다 (저장소 기본 규칙).
    lag=0: 신호를 낸 날 종가에 체결합니다 (현실에서는 불가능한 참고값).
    """
    if lag < 0:
        raise ValueError("lag는 0 이상이어야 합니다. 음수는 미래 정보입니다.")
    locs = index.get_indexer(signal.index)
    if (locs < 0).any():
        raise ValueError("신호 날짜가 거래일 목록에 없습니다.")
    exec_locs = locs + lag
    valid = exec_locs < len(index)
    holding = pd.Series(np.nan, index=index, name="holding_after_close")
    holding.iloc[exec_locs[valid]] = signal.to_numpy()[valid]
    return holding.ffill()


# ---------------------------------------------------------------- 백테스트


@dataclass
class Result:
    daily: pd.DataFrame  # equity(청산 전), ret, rf, holding(그날 수익률에 적용한 보유 상태), trade, tax
    final_after_liquidation: float  # 기간 끝에 모두 팔고 세금까지 낸 뒤의 자산
    start_date: pd.Timestamp  # 평가 시작 전날 (자산 1.0 기준일)
    trades: int


def run_backtest(data: pd.DataFrame, holding_after_close: pd.Series, start: str, end: str, costs: Costs) -> Result:
    """평가 시작 전날 종가에 자산 1.0으로 시작해 일별로 계산합니다.

    data: close, dividend(주당, 배당락일), rf(일별 현금 수익률) 열. 모두 같은 일별 인덱스.
    holding_after_close: 각 거래일 종가에 체결을 마친 뒤의 보유 상태 (0 또는 1).
    """
    index = data.index
    days = index[(index >= pd.Timestamp(start)) & (index <= pd.Timestamp(end))]
    if len(days) == 0:
        raise ValueError("평가 기간에 거래일이 없습니다.")
    prev_loc = index.get_loc(days[0]) - 1
    if prev_loc < 0:
        raise ValueError("평가 시작 전날 데이터가 없습니다.")
    start_date = index[prev_loc]

    window = data.loc[start_date:days[-1]]
    holding = holding_after_close.reindex(window.index)
    if holding.isna().any() or not holding.isin([0.0, 1.0]).all():
        raise ValueError("평가 기간(시작 전날 포함)에 보유 상태가 0/1이 아닌 날이 있습니다. 신호 준비 기간을 확인하세요.")
    if window["rf"].iloc[1:].isna().any():
        raise ValueError("평가 기간에 현금 수익률(rf)이 없는 날이 있습니다.")

    close = window["close"].to_numpy()
    dividend = window["dividend"].to_numpy() * (1 - costs.dividend_withholding)
    rf = window["rf"].to_numpy()
    target = holding.to_numpy()
    year_end = _last_trading_day_of_year(index).reindex(window.index).to_numpy()

    n = len(window)
    equity = np.empty(n)
    ret = np.zeros(n)
    held = np.zeros(n)
    traded = np.zeros(n, dtype=bool)
    taxes = np.zeros(n)

    # 시작 전날 종가: 보유 상태로 시작하면 진입 비용을 냅니다.
    value, in_stock, basis, realized, trades = 1.0, False, 0.0, 0.0, 0
    value, in_stock, basis, realized, did_trade = _trade(value, in_stock, target[0] == 1.0, basis, realized, costs)
    trades += did_trade
    equity[0], traded[0] = value, did_trade

    for i in range(1, n):
        start_value = equity[i - 1]
        held[i] = 1.0 if in_stock else 0.0
        if in_stock:
            value = start_value * (close[i] + dividend[i]) / close[i - 1]
        else:
            value = start_value * (1 + rf[i])
        value, in_stock, basis, realized, did_trade = _trade(value, in_stock, target[i] == 1.0, basis, realized, costs)
        trades += did_trade
        traded[i] = did_trade
        if year_end[i] or i == n - 1:
            tax = costs.capital_gains_tax * max(0.0, realized)
            value -= tax
            taxes[i], realized = tax, 0.0
        equity[i] = value
        ret[i] = value / start_value - 1

    # 기간 끝 청산: 보유 중이면 팔고 그 이익에 세금을 냅니다.
    final = equity[-1]
    if in_stock:
        proceeds = final * (1 - costs.trade)
        final = proceeds - costs.capital_gains_tax * max(0.0, proceeds - basis)

    daily = pd.DataFrame(
        {"equity": equity, "ret": ret, "rf": np.nan_to_num(rf), "holding": held, "trade": traded, "tax": taxes},
        index=window.index,
    )
    return Result(daily=daily, final_after_liquidation=final, start_date=start_date, trades=trades)


def _trade(value: float, in_stock: bool, want_stock: bool, basis: float, realized: float, costs: Costs):
    """종가에 상태를 바꿉니다. 매수 금액(비용 포함)을 취득가로 두고, 매도 시 순매도액 − 취득가를 실현 이익으로 셉니다."""
    if want_stock == in_stock:
        return value, in_stock, basis, realized, False
    if want_stock:
        return value * (1 - costs.trade), True, value, realized, True
    proceeds = value * (1 - costs.trade)
    return proceeds, False, 0.0, realized + proceeds - basis, True


def _last_trading_day_of_year(index: pd.DatetimeIndex) -> pd.Series:
    last = pd.Series(index, index=index).groupby(index.year).transform("max")
    return pd.Series(index == last.to_numpy(), index=index)


# ---------------------------------------------------------------- 지표


def metrics(result: Result) -> dict:
    """표준 리포트 지표. 일별 수익률 기준입니다. 첫 행(시작 전날)은 수익률 계산에서 뺍니다."""
    daily = result.daily
    ret = daily["ret"].iloc[1:]
    excess = ret - daily["rf"].iloc[1:]
    equity = daily["equity"]
    years = (daily.index[-1] - result.start_date).days / 365.25

    curve = pd.concat([pd.Series([1.0], index=[result.start_date]), equity.iloc[1:]])
    drawdown = curve / curve.cummax() - 1
    trough = drawdown.idxmin()
    peak = curve.loc[:trough].idxmax()
    recovered = curve.loc[trough:][curve.loc[trough:] >= curve.loc[peak]]
    recovery_months = (recovered.index[0] - peak).days / 30.4375 if len(recovered) else np.nan

    return {
        "cagr": equity.iloc[-1] ** (1 / years) - 1,
        "cagr_after_liquidation": result.final_after_liquidation ** (1 / years) - 1,
        "volatility": ret.std() * np.sqrt(TRADING_DAYS),
        "sharpe": excess.mean() * TRADING_DAYS / (excess.std() * np.sqrt(TRADING_DAYS)),
        "mdd": drawdown.min(),
        "mdd_peak": peak,
        "mdd_trough": trough,
        "mdd_recovery_months": recovery_months,  # NaN이면 기간 안에 회복하지 못함
        "trades_per_year": result.trades / years,
        "time_in_market": daily["holding"].iloc[1:].mean(),
        "years": years,
    }
