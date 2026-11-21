"""백테스트 엔진.

일별 데이터(종가, 배당, 현금 수익률)에서 주간·월간 신호를 받아, 정한 시점의 종가에 체결합니다.
보유 상태는 100% 주식(1), 100% 현금(0), 100% 인버스 ETF(-1, 하락 베팅) 중 하나입니다.

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
    - interest_tax: 현금 이자에 매기는 세율 (원천징수로 보고 매일 뺍니다)
    - inverse_tax: 인버스 ETF 매매차익의 과세 방식
        - "pooled": 주식 매매차익과 합쳐 연간 실현 순이익에 capital_gains_tax를 매깁니다 (해외 상장 ETF 양도소득세).
        - "holding_period": 인버스를 팔 때마다 min(매매차익, 과표기준가 증가분) × inverse_gain_tax를 원천징수합니다.
          손실과 상계하지 않습니다 (국내 상장 인버스 ETF의 보유기간 과세). capital_gains_tax와 함께 쓰지 않습니다.
    - inverse_gain_tax: "holding_period"의 세율
    """

    trade: float = 0.001
    dividend_withholding: float = 0.0
    capital_gains_tax: float = 0.0
    interest_tax: float = 0.0
    inverse_tax: str = "pooled"
    inverse_gain_tax: float = 0.0


# ---------------------------------------------------------------- 투자 수익률 합성


def synthetic_index_etf(
    index_daily: pd.DataFrame, dividend_yield: float, dividend_timing: str, expense: float, cash: str
) -> pd.DataFrame:
    """배당이 빠진 가격 지수로 지수 추종 ETF를 합성합니다. 보수는 가격에서 빼고, 지수에 빠진 배당은 dividend 열로 더합니다.

    index_daily: close(가격 지수), rf(일별 현금 수익률) 열.
    배당 금액은 전날 종가 × 수익률입니다. 지수에는 배당락으로 떨어진 가격이 이미 들어 있습니다.
    - december: 매년 12월의 마지막 거래일 전날(국내 12월 결산 기업의 배당락 시기)에 한 번에 받습니다.
    - even: 달력 일수에 비례해 매일 나눠 받습니다.
    cash: "cd"면 rf를 그대로, "zero"면 현금 수익률 0.
    """
    index = index_daily.index
    years = (index - index[0]).days / 365.25
    close = index_daily["close"] * (1 - expense) ** years
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

    if cash == "cd":
        rf = index_daily["rf"]
    elif cash == "zero":
        rf = index_daily["rf"] * 0.0
    else:
        raise ValueError(f"알 수 없는 현금 가정: {cash}")
    return pd.DataFrame({"close": close, "dividend": dividend, "rf": rf}, index=index)


def synthetic_inverse(
    underlying: pd.DataFrame, rf: pd.Series, expense: float, borrow: float = 0.0, interest_multiple: float = 2.0
) -> pd.DataFrame:
    """기초 지수의 일별 배당 포함 수익률로 매일 -1배를 맞추는 인버스 ETF를 합성합니다.

    underlying: close, dividend(배당락일의 주당 배당) 열. rf: 일별 현금 수익률 (같은 인덱스).
    NAV[t] = NAV[t-1] × (1 − TR[t] + m × rf[t]) × (1 − expense − borrow) ** (경과 일수 / 365.25)
    - TR[t] = (close[t] + dividend[t]) / close[t-1] − 1. 숏은 배당을 물어 주므로 배당 포함 수익률의 반대입니다.
    - m × rf[t]: 담보 현금의 이자(1배)와, 선물·스왑 가격에 들어 있는 이자(1배)입니다. 기본 m = 2.
    - expense: 운용 보수(연율). borrow: 차입·스왑 비용과 그 밖의 추적 차이(연율). 둘 다 달력 일수로 나눠 뺍니다.
    tax_base: 국내 인버스 ETF의 1주당 과표기준가. 담보 이자 − 보수만 쌓이고, 선물 손익(상장주식 기초 장내파생상품,
        과세 제외)과 차입 비용은 빠집니다. 담보 이자는 min(m, 1) × rf로 봅니다.
        tax_base[t] = tax_base[t-1] + NAV[t-1] × (min(m, 1) × rf[t] − expense × 경과 일수 / 365.25)
    첫날 NAV와 tax_base는 1입니다. 각 날의 값은 그날까지의 데이터로만 계산합니다.
    """
    if not underlying.index.equals(rf.index):
        raise ValueError("기초 지수와 현금 수익률의 날짜가 다릅니다.")
    close = underlying["close"].to_numpy(dtype=float)
    dividend = underlying["dividend"].to_numpy(dtype=float)
    rate = rf.to_numpy(dtype=float)
    elapsed = np.concatenate([[0.0], np.diff(underlying.index.to_numpy()).astype("timedelta64[D]").astype(float)]) / 365.25

    nav, base = np.empty(len(close)), np.empty(len(close))
    nav[0] = base[0] = 1.0
    for t in range(1, len(close)):
        total_return = (close[t] + dividend[t]) / close[t - 1] - 1
        nav[t] = nav[t - 1] * (1 - total_return + interest_multiple * rate[t]) * (1 - expense - borrow) ** elapsed[t]
        base[t] = base[t - 1] + nav[t - 1] * (min(interest_multiple, 1.0) * rate[t] - expense * elapsed[t])
    if np.isnan(nav).any():
        raise ValueError("합성 인버스에 빈 값이 있습니다. 기초 지수나 현금 수익률이 빈 날을 확인하세요.")
    return pd.DataFrame({"close": nav, "tax_base": base}, index=underlying.index)


# ---------------------------------------------------------------- 신호


def period_last_days(index: pd.DatetimeIndex, freq: str) -> pd.DatetimeIndex:
    """각 기간의 마지막 거래일. freq는 "W-FRI"(금요일로 끝나는 주) 또는 "M"(월).

    휴장일은 미리 공시되므로, 실제 데이터로 마지막 거래일을 정해도 미래 정보가 아닙니다.
    단, 데이터 맨 끝의 기간은 끝나지 않았을 수 있으므로 평가 기간 끝은 데이터 끝보다 앞에 둡니다.
    """
    days = pd.Series(index, index=index)
    return pd.DatetimeIndex(days.groupby(index.to_period(freq)).last().to_numpy(), name=index.name)


def snap_to_tick(price: pd.Series, tick: float) -> pd.Series:
    """가격을 호가 단위의 정수 배(단위: 호가 단위 개수)로 바꿉니다. 신호 계산에만 씁니다.

    Yahoo 가격은 float32로 저장되어 126.36이 126.36000061035156처럼 남습니다. 그대로 이동평균과 비교하면
    실제 가격으로는 같은 값(종가 = 이동평균)이 '높다'로 판정될 수 있습니다. 정수로 바꾸면 합이 정확해져
    종가와 이동평균의 비교, 이동평균 기울기의 부호가 정확해집니다. 비교는 비율이라 단위를 바꿔도 뜻은 같습니다.
    격자에서 호가 단위의 1/4 넘게 벗어난 가격이 있으면 호가 단위 가정이 틀린 것이므로 멈춥니다.
    """
    units = price / tick
    rounded = units.round()
    off = (units - rounded).abs()
    if (off > 0.25).any():
        raise ValueError(f"호가 단위 {tick}의 격자에서 벗어난 가격이 있습니다: {off.idxmax()}")
    return rounded


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


def period_bars(ohlc: pd.DataFrame, freq: str) -> pd.DataFrame:
    """일봉을 기간 봉(고가, 저가, 종가)으로 묶습니다. 인덱스는 각 기간의 마지막 거래일입니다."""
    grouped = ohlc.groupby(ohlc.index.to_period(freq))
    bars = pd.DataFrame({"high": grouped["high"].max(), "low": grouped["low"].min(), "close": grouped["close"].last()})
    bars.index = period_last_days(ohlc.index, freq)
    return bars


def wilder_adx(bars: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ADX. 방향이 아니라 추세의 세기를 나타냅니다.

    bars: high, low 열을 가진 봉 표. +DI와 −DI의 분모가 같은 TR 평활이라 DX에서 약분되므로, TR과 종가는 필요 없습니다.
    각 값은 그 봉까지의 데이터로만 계산합니다. 첫 값은 2 * period번째 봉에 나오고, 그 전은 NaN입니다.
    """
    high, low = bars["high"].to_numpy(), bars["low"].to_numpy()
    n = len(bars)
    adx = np.full(n, np.nan)
    if n >= 2 * period:
        up, down = high[1:] - high[:-1], low[:-1] - low[1:]
        plus_dm = np.where((up > down) & (up > 0), up, 0.0)
        minus_dm = np.where((down > up) & (down > 0), down, 0.0)

        # i번째 봉의 움직임은 plus_dm[i - 1]입니다. 첫 평활값은 앞 period개의 합입니다.
        smooth_plus, smooth_minus = plus_dm[:period].sum(), minus_dm[:period].sum()
        dx = np.zeros(n)
        for i in range(period, n):
            if i > period:
                smooth_plus += plus_dm[i - 1] - smooth_plus / period
                smooth_minus += minus_dm[i - 1] - smooth_minus / period
            total = smooth_plus + smooth_minus
            dx[i] = 100 * abs(smooth_plus - smooth_minus) / total if total > 0 else 0.0

        adx[2 * period - 1] = dx[period : 2 * period].mean()
        for i in range(2 * period, n):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return pd.Series(adx, index=bars.index, name="adx")


def moving_average_adx_signal(
    price: pd.Series, signal_days: pd.DatetimeIndex, window: int, adx: pd.Series, threshold: float
) -> pd.Series:
    """이동평균 신호에 ADX 진입 필터를 더합니다. 진입에만 쓰고 청산에는 쓰지 않습니다.

    - 미보유: 종가 > 이동평균이고 ADX >= threshold이면 보유 (그 주에 조건을 보고 진입합니다)
    - 보유: 종가 <= 이동평균이면 미보유
    두 지표가 모두 준비되기 전은 NaN이고, 준비된 첫 주는 미보유에서 시작합니다.
    """
    above = moving_average_signal(price, signal_days, window)
    adx_on_days = adx.reindex(signal_days)
    ready = above.notna() & adx_on_days.notna()
    if ready.any() and not ready[ready.idxmax() :].all():
        raise ValueError("지표가 준비된 뒤에 비어 있는 신호 날짜가 있습니다.")

    signal = pd.Series(np.nan, index=signal_days, name="signal")
    state = 0.0
    for day in signal_days[ready]:
        if state == 1.0:
            state = above[day]
        elif above[day] == 1.0 and adx_on_days[day] >= threshold:
            state = 1.0
        signal[day] = state
    return signal


def moving_average_slope_signal(price: pd.Series, signal_days: pd.DatetimeIndex, window: int, slope_lag: int = 1) -> pd.Series:
    """이동평균 신호에 이동평균 기울기 필터를 더합니다.

    m = 종가 > 이동평균, 기울기 = 이번 주 이동평균 − slope_lag주 전 이동평균.
    - 미보유: m이고 기울기 >= 0이면 보유. 기울기 < 0(하락 중)이면 매수하지 않습니다.
    - 보유: m이 아니고 기울기 <= 0이면 미보유. 기울기 > 0(상승 중)이면 매도하지 않습니다.
    기울기가 준비되기 전은 NaN이고, 준비된 첫 주는 m으로 시작합니다.
    """
    if slope_lag < 1:
        raise ValueError("slope_lag는 1 이상이어야 합니다.")
    sampled = price.reindex(signal_days)
    if sampled.isna().any():
        raise ValueError("신호 날짜에 가격이 없습니다.")
    average = sampled.rolling(window).mean()
    above = (sampled > average).to_numpy()
    slope = (average - average.shift(slope_lag)).to_numpy()

    signal = np.full(len(signal_days), np.nan)
    state = np.nan
    for i in np.flatnonzero(~np.isnan(slope)):
        if np.isnan(state):
            state = 1.0 if above[i] else 0.0
        elif state == 0.0 and above[i] and slope[i] >= 0:
            state = 1.0
        elif state == 1.0 and not above[i] and slope[i] <= 0:
            state = 0.0
        signal[i] = state
    return pd.Series(signal, index=signal_days, name="signal")


def long_short(signal: pd.Series) -> pd.Series:
    """0/1 신호를 롱/숏 신호로 바꿉니다. 1은 주식(1), 0은 인버스(-1). 준비 기간의 NaN은 그대로 둡니다."""
    return signal * 2 - 1


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
    daily: pd.DataFrame  # equity(청산 전), ret, rf, holding(그날 수익률에 적용한 보유 상태 1/0/-1), trade, tax
    final_after_liquidation: float  # 기간 끝에 모두 팔고 마지막 해 세금까지 정산한 뒤의 자산
    start_date: pd.Timestamp  # 평가 시작 전날 (자산 1.0 기준일)
    trades: int  # 매수와 매도를 각각 한 번으로 센 체결 횟수. 주식에서 인버스로 바꾸면 2회입니다.


def run_backtest(
    data: pd.DataFrame, holding_after_close: pd.Series, start: str, end: str, costs: Costs,
    inverse: pd.DataFrame | None = None,
) -> Result:
    """평가 시작 전날 종가에 자산 1.0으로 시작해 일별로 계산합니다.

    data: close, dividend(주당, 배당락일), rf(일별 현금 수익률) 열. 모두 같은 일별 인덱스.
    holding_after_close: 각 거래일 종가에 체결을 마친 뒤의 보유 상태. 1 주식, 0 현금, -1 인버스.
    inverse: 보유 상태 -1일 때 들고 있는 인버스 ETF. close(1주 가격) 열, inverse_tax가 "holding_period"면
        tax_base(1주당 과표기준가) 열도 필요합니다 (synthetic_inverse의 결과). 배당은 없는 것으로 봅니다.

    상태를 바꿀 때는 가진 것을 먼저 팔고 새것을 삽니다. 매도와 매수마다 편도 매매 비용을 냅니다.
    매수 금액(비용 포함)을 취득가로 두고, 매도 시 순매도액 − 취득가를 실현 이익으로 셉니다.
    """
    if costs.inverse_tax not in ("pooled", "holding_period"):
        raise ValueError(f"알 수 없는 인버스 과세 방식: {costs.inverse_tax}")
    if costs.inverse_tax == "holding_period" and costs.capital_gains_tax > 0:
        raise ValueError("인버스 보유기간 과세와 연간 양도소득세를 함께 쓰는 경우는 지원하지 않습니다.")
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
    if holding.isna().any() or not holding.isin([-1.0, 0.0, 1.0]).all():
        raise ValueError("평가 기간(시작 전날 포함)에 보유 상태가 1/0/-1이 아닌 날이 있습니다. 신호 준비 기간을 확인하세요.")
    if window["rf"].iloc[1:].isna().any():
        raise ValueError("평가 기간에 현금 수익률(rf)이 없는 날이 있습니다.")
    holding_period_tax = costs.inverse_tax == "holding_period"
    if (holding == -1.0).any():
        needed = ["close", "tax_base"] if holding_period_tax else ["close"]
        if inverse is None:
            raise ValueError("보유 상태 -1(인버스)이 있는데 인버스 데이터가 없습니다.")
        if not set(needed) <= set(inverse.columns):
            raise ValueError(f"인버스 데이터에 {needed} 열이 필요합니다.")
        frame = inverse.reindex(window.index)
        if frame[needed].isna().any(axis=None):
            raise ValueError("평가 기간에 인버스 가격(또는 과표기준가)이 없는 날이 있습니다.")
        inv_close = frame["close"].to_numpy()
        inv_base = frame["tax_base"].to_numpy() if holding_period_tax else None
    else:
        inv_close = inv_base = None

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

    # pos: 지금 보유 상태. basis: 취득가(주식·인버스 공통). realized: 그해 연간 과세 대상 실현 손익.
    # entry: 인버스 매수 때의 (1주 가격, 1주당 과표기준가, 비용을 뺀 투자 금액). 보유기간 과세에 씁니다.
    pos, basis, realized, trades, entry = 0.0, 0.0, 0.0, 0, None

    def inverse_sale_tax(i: int) -> float:
        """인버스를 i일 종가에 팔 때의 보유기간 과세. min(매매차익, 과표기준가 증가분) × 세율, 음수면 0."""
        price, base, invested = entry
        gain = min(inv_close[i] - price, inv_base[i] - base) / price * invested
        return costs.inverse_gain_tax * max(0.0, gain)

    def switch(i: int, value: float, want: float) -> float:
        """i일 종가에 보유 상태를 want로 바꿉니다. 바꾼 뒤의 자산을 돌려줍니다."""
        nonlocal pos, basis, realized, trades, entry
        if want == pos:
            return value
        if pos != 0.0:
            proceeds = value * (1 - costs.trade)
            if pos == -1.0 and holding_period_tax:
                tax = inverse_sale_tax(i)
                proceeds -= tax
                taxes[i] += tax
            else:
                realized += proceeds - basis
            value, basis, entry = proceeds, 0.0, None
            trades += 1
        if want != 0.0:
            basis = value
            value *= 1 - costs.trade
            if want == -1.0 and holding_period_tax:
                entry = (inv_close[i], inv_base[i], value)
            trades += 1
        traded[i] = True
        pos = want
        return value

    # 시작 전날 종가: 보유 상태로 시작하면 진입 비용을 냅니다.
    equity[0] = switch(0, 1.0, target[0])

    for i in range(1, n):
        start_value = equity[i - 1]
        held[i] = pos
        if pos == 1.0:
            value = start_value * (close[i] + dividend[i]) / close[i - 1]
        elif pos == -1.0:
            value = start_value * inv_close[i] / inv_close[i - 1]
        else:
            value = start_value * (1 + rf[i] * (1 - costs.interest_tax))
        value = switch(i, value, target[i])
        # 마지막 날의 세금은 청산 손익과 합쳐 아래에서 한 번에 정산합니다.
        if year_end[i] and i < n - 1:
            tax = costs.capital_gains_tax * max(0.0, realized)
            realized = 0.0
            if pos != 0.0 and tax > 0:
                # 들고 있는 것을 일부 팔아 세금을 낸 것으로 봅니다. 판 비율만큼 취득가를 줄이고,
                # 그 부분의 이익은 다음 해 실현 이익으로 넘깁니다. (이 매도의 매매 비용은 무시)
                sold = tax / value
                realized = sold * (value - basis)
                basis *= 1 - sold
            value -= tax
            taxes[i] += tax
        equity[i] = value
        ret[i] = value / start_value - 1

    # 기간 끝 정산: 보유 중이면 팔고, 그해 실현 손익과 청산 손익을 합쳐 한 번만 과세합니다.
    # 인버스 보유기간 과세는 매도할 때 따로 냅니다. 일별 자산(equity)은 이 정산 전 값입니다.
    final = equity[-1]
    if pos != 0.0:
        final *= 1 - costs.trade
        if pos == -1.0 and holding_period_tax:
            final -= inverse_sale_tax(n - 1)
        else:
            realized += final - basis
    final_tax = costs.capital_gains_tax * max(0.0, realized)
    final -= final_tax

    daily = pd.DataFrame(
        {"equity": equity, "ret": ret, "rf": np.nan_to_num(rf), "holding": held, "trade": traded, "tax": taxes},
        index=window.index,
    )
    return Result(daily=daily, final_after_liquidation=final, start_date=start_date, trades=trades)


def _last_trading_day_of_year(index: pd.DatetimeIndex) -> pd.Series:
    last = pd.Series(index, index=index).groupby(index.year).transform("max")
    return pd.Series(index == last.to_numpy(), index=index)


# ---------------------------------------------------------------- 지표


def trade_list(holding_after_close: pd.Series, price: pd.Series, equity: pd.Series | None = None) -> pd.DataFrame:
    """보유 구간(매수 체결일 ~ 매도 체결일)마다 보유 중 고점과 매도 시 고점 대비 하락률을 냅니다.

    holding_after_close: 평가 기간(시작 전날 포함)의 체결 뒤 보유 상태 (run_backtest에 넘긴 것을 그 기간으로 자른 것).
        첫날이 1이면 시작 전날 종가에 산 것으로 봅니다 (run_backtest도 그날 진입 비용을 냅니다).
    price: 일별 종가. 고점은 매수 체결일부터 매도 체결일까지(양 끝 포함) 종가의 최댓값입니다.
    equity: run_backtest의 daily["equity"]를 넘기면, 같은 방식의 자산 기준 하락률도 냅니다. 배당과 매도 비용이
        반영되고, 매수 비용은 고점과 매도 값에 함께 들어가 비율에서 약분됩니다.
    기간 끝까지 들고 있던 구간은 open=True이고, 매도일 자리에 마지막 날을 둡니다 (매도가 아닙니다).
    """
    held = holding_after_close.to_numpy()
    if np.isnan(held).any() or not np.isin(held, [0.0, 1.0]).all():
        raise ValueError("보유 상태가 0/1이 아닌 날이 있습니다.")
    days = holding_after_close.index
    spans, buy = [], days[0] if held[0] == 1.0 else None
    for i in range(1, len(days)):
        if held[i] == 1.0 and held[i - 1] == 0.0:
            buy = days[i]
        elif held[i] == 0.0 and held[i - 1] == 1.0:
            spans.append((buy, days[i], False))
            buy = None
    if buy is not None:
        spans.append((buy, days[-1], True))

    rows = []
    for buy, sell, still_open in spans:
        segment = price.loc[buy:sell]
        peak_day = segment.idxmax()
        row = {
            "buy_date": buy,
            "buy_price": price[buy],
            "peak_date": peak_day,
            "peak_price": segment.max(),
            "sell_date": sell,
            "sell_price": price[sell],
            "from_peak": price[sell] / segment.max() - 1,
            "from_buy": price[sell] / price[buy] - 1,
            "weeks_held": (sell - buy).days / 7,
            "weeks_peak_to_sell": (sell - peak_day).days / 7,
            "open": still_open,
        }
        if equity is not None:
            value = equity.loc[buy:sell]
            row["equity_from_peak"] = value.iloc[-1] / value.max() - 1
        rows.append(row)
    columns = ["buy_date", "buy_price", "peak_date", "peak_price", "sell_date", "sell_price", "from_peak", "from_buy",
               "weeks_held", "weeks_peak_to_sell", "open"] + (["equity_from_peak"] if equity is not None else [])
    return pd.DataFrame(rows, columns=columns)


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
        "time_in_market": (daily["holding"].iloc[1:] == 1.0).mean(),  # 주식 보유 비율
        "time_short": (daily["holding"].iloc[1:] == -1.0).mean(),  # 인버스 보유 비율
        "years": years,
    }
