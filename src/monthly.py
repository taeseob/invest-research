"""월 단위 계산: Faber 논문 방식의 타이밍 수익률과 월 단위 지표.

논문 방식은 매월 마지막 거래일 종가로 신호를 보고, 같은 날 종가에 체결합니다. 비용은 없습니다.
월 수익률 표의 인덱스는 월(PeriodIndex)입니다. t월의 값은 t월 마지막 거래일 종가까지의 정보입니다.

시점 약속:
- signal[t]: t월 마지막 거래일 종가까지로 정한 신호 (1 보유, 0 현금)
- t월의 수익률(t-1월 말 → t월 말)은 signal[t-1]로 정합니다.
"""

import numpy as np
import pandas as pd

from src.backtest import moving_average_signal, period_last_days

MONTHS_PER_YEAR = 12


def month_end_signal(daily_price: pd.Series, window: int) -> pd.Series:
    """각 달 마지막 거래일 종가가 최근 window개 월말 종가(그달 포함)의 평균보다 높으면 1, 아니면 0.

    daily_price: 일별 가격 (신호용. 호가 단위로 바꾼 가격을 넘기면 동률 판정이 정확합니다).
    반환값의 인덱스는 월이고, 평균을 낼 월이 모자란 준비 기간은 NaN입니다.
    데이터 맨 끝의 달은 끝나지 않았을 수 있으므로, 부르는 쪽에서 끝난 달까지만 넘깁니다.
    """
    month_ends = period_last_days(daily_price.index, "M")
    signal = moving_average_signal(daily_price, month_ends, window)
    signal.index = month_ends.to_period("M")
    signal.index.name = "month"
    return signal


def monthly_series_signal(series: pd.Series, window: int) -> pd.Series:
    """월간 값(인덱스가 월)에서 바로 이동평균 신호를 냅니다. 총수익 지수 신호처럼 일별 값이 없을 때 씁니다."""
    average = series.rolling(window).mean()
    signal = (series > average).astype(float)
    signal[average.isna()] = np.nan
    return signal


def total_return_signal_index(close: pd.Series, annual_dividend: pd.Series, dividend_lag: int = 12) -> pd.Series:
    """신호용 총수익 지수 (인덱스는 월). 첫 달은 1입니다.

    index[t] = index[t-1] × (close[t] + D[t - dividend_lag] / 12) / close[t-1]
    close: 월말 종가. annual_dividend: Shiller 연간 배당 D(월별 값).
    Shiller D는 분기(1934년까지는 연간) 값을 직선으로 이은 것이라, 최근 값에 앞으로의 배당이 섞여 있습니다.
    그래서 dividend_lag개월 전 값만 씁니다. 첫 달 뒤에 쓸 D가 비어 있으면 멈춥니다(조용히 채우지 않음).
    """
    if dividend_lag < 1:
        raise ValueError("dividend_lag는 1 이상이어야 합니다. 0이면 보간에 섞인 앞으로의 배당을 씁니다.")
    expected = pd.period_range(close.index[0], close.index[-1], freq="M")
    if not close.index.equals(expected):
        raise ValueError("월말 종가에 빠진 달이 있습니다.")
    lagged = pd.Series(annual_dividend.reindex(close.index - dividend_lag).to_numpy(), index=close.index)
    growth = (close + lagged / 12) / close.shift(1)
    if growth.iloc[1:].isna().any():
        raise ValueError(f"총수익 지수에 쓸 배당이 빈 달이 있습니다: {growth.iloc[1:][growth.iloc[1:].isna()].index[0]}")
    growth.iloc[0] = 1.0
    return growth.cumprod().rename("tr_index")


def timing_returns(signal: pd.Series, asset_return: pd.Series, rf: pd.Series) -> pd.DataFrame:
    """R[t] = held[t] × asset_return[t] + (1 − held[t]) × rf[t], held[t] = signal[t-1].

    세 Series는 같은 월 인덱스여야 합니다. 빠진 달이 있으면 shift가 두 달을 건너뛰므로 막습니다.
    반환: ret(전략 월 수익률), held(그달 수익률에 적용한 보유 상태), rf
    """
    for name, series in {"signal": signal, "asset_return": asset_return, "rf": rf}.items():
        if not isinstance(series.index, pd.PeriodIndex) or series.index.freqstr != "M":
            raise ValueError(f"{name}의 인덱스가 월(PeriodIndex)이 아닙니다.")
    if not (signal.index.equals(asset_return.index) and signal.index.equals(rf.index)):
        raise ValueError("신호, 자산 수익률, 현금 수익률의 월이 다릅니다.")
    expected = pd.period_range(signal.index[0], signal.index[-1], freq="M")
    if not signal.index.equals(expected):
        raise ValueError("빠진 달이 있습니다.")
    held = signal.shift(1)
    ret = held * asset_return + (1 - held) * rf
    return pd.DataFrame({"ret": ret, "held": held, "rf": rf})


def month_end_returns(daily_equity: pd.Series, start_date: pd.Timestamp) -> pd.Series:
    """일별 자산(run_backtest의 daily["equity"])을 월 수익률로 바꿉니다.

    시작 전날의 자산은 진입 비용을 내기 전 1.0으로 봅니다 (metrics의 MDD 계산과 같음).
    그래서 첫 달 수익률에 진입 비용이 들어갑니다. 첫 행(시작 전날)은 그 앞 달에 속하므로 월 수익률에서 뺍니다.
    """
    equity = daily_equity[daily_equity.index > start_date]
    month_end = equity.groupby(equity.index.to_period("M")).last()
    previous = month_end.shift(1)
    previous.iloc[0] = 1.0
    returns = month_end / previous - 1
    returns.index.name = "month"
    return returns


def monthly_metrics(ret: pd.Series, rf: pd.Series, held: pd.Series | None = None) -> dict:
    """월 수익률로 낸 지표. 원문(Faber)과 같은 단위입니다.

    - cagr: 기간 전체의 연복리 수익률. 연수는 달 수 / 12.
    - volatility: 월 수익률 표준편차(표본) × √12
    - sharpe_paper: (CAGR − 현금 CAGR) / 변동성. 원문 부록 B 수치로 역산한 정의입니다.
    - sharpe: 월 초과수익률 평균 × 12 / (월 초과수익률 표준편차 × √12). 저장소 방식.
    - mdd: 월말 값 기준 최대 낙폭. 기간 시작(1.0)부터 잽니다. mdd_recovery_months는 고점부터 고점을 되찾은 달까지
      달 수이고, 기간 안에 되찾지 못하면 NaN입니다.
    - held를 넘기면 보유 비율과 매매 횟수를 냅니다. 매수와 매도를 각각 한 번으로 세고,
      첫 달에 보유 상태로 시작하면 진입 1회로 셉니다 (run_backtest와 같음).
    """
    if ret.isna().any() or rf.isna().any():
        raise ValueError("월 수익률이나 현금 수익률에 빈 값이 있습니다.")
    if not ret.index.equals(rf.index):
        raise ValueError("수익률과 현금 수익률의 월이 다릅니다.")
    months = len(ret)
    years = months / MONTHS_PER_YEAR
    wealth = (1 + ret).cumprod()
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    rf_cagr = (1 + rf).prod() ** (1 / years) - 1
    volatility = ret.std() * np.sqrt(MONTHS_PER_YEAR)
    excess = ret - rf

    # 기간 시작 직전(1.0)을 앞에 붙여 낙폭을 잽니다.
    curve = pd.Series(np.concatenate([[1.0], wealth.to_numpy()]))
    drawdown = curve / curve.cummax() - 1
    trough = int(drawdown.idxmin())
    peak = int(curve[: trough + 1].idxmax())
    after = curve[trough:]
    recovered = after[after >= curve[peak]]
    labels = [ret.index[0] - 1, *ret.index]  # curve의 i번째 값은 labels[i] 월말 값

    out = {
        "cagr": cagr,
        "volatility": volatility,
        "sharpe_paper": (cagr - rf_cagr) / volatility,
        "sharpe": excess.mean() * MONTHS_PER_YEAR / (excess.std() * np.sqrt(MONTHS_PER_YEAR)),
        "mdd": drawdown.min(),
        "mdd_peak": labels[peak],
        "mdd_trough": labels[trough],
        "mdd_recovery_months": int(recovered.index[0]) - peak if len(recovered) else np.nan,
        "positive_months": (ret > 0).mean(),
        "rf_cagr": rf_cagr,
        "years": years,
    }
    if held is not None:
        if not held.index.equals(ret.index) or held.isna().any() or not held.isin([0.0, 1.0]).all():
            raise ValueError("보유 상태는 수익률과 같은 월에 0/1이어야 합니다.")
        values = held.to_numpy()
        trades = int(values[0] == 1.0) + int((values[1:] != values[:-1]).sum())
        out["time_in_market"] = values.mean()
        out["trades_per_year"] = trades / years
    return out


def switch_months(signal: pd.Series) -> pd.DataFrame:
    """신호가 바뀐 달의 목록. 매도 신호를 낸 달이 원문 본문의 매도 시점과 같은지 볼 때 씁니다."""
    values = signal.dropna()
    changed = values[values.diff().fillna(0) != 0]
    return pd.DataFrame({"month": changed.index, "action": np.where(changed.to_numpy() == 1.0, "매수", "매도")})
