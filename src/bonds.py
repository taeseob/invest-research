"""만기 고정 국채의 월 수익률 합성.

매월 말 그 만기의 국채 금리로 액면가 채권(쿠폰 = 그 금리)을 사고, 한 달 뒤 그달 말 금리로 평가해 파는 방식입니다.
금리 자료(연준 H.15의 만기 고정 국채 금리, 반기 복리 채권 수익률)만으로 총수익률을 만들 때 씁니다.
"""

import numpy as np
import pandas as pd


def par_bond_monthly_returns(month_end_yield: pd.Series, maturity_years: float = 10.0) -> pd.DataFrame:
    """월말 금리(소수, 인덱스는 월)로 만기 고정 국채의 월 총수익률과 가격 수익률을 만듭니다.

    t-1월 말에 금리 y[t-1]로 쿠폰 y[t-1]의 액면가 채권(가격 1, 반기 쿠폰)을 삽니다.
    t월 말에는 남은 만기가 한 달 줄었으므로, 각 현금흐름까지 남은 시간은 k − 1/6 반기입니다(k = 1 … 2 × 만기).
    그 현금흐름을 y[t]로 할인한 값이 t월 말 가격(경과 이자 포함)입니다.
    - total_return[t] = 가격 − 1
    - price_return[t] = total_return[t] − y[t-1] / 12 (한 달 치 쿠폰을 뺀 가격 변화. 신호용)
    t월의 값은 t-1월 말과 t월 말 금리만 씁니다. 금리가 빈 달 다음 달은 NaN입니다.
    """
    periods = int(round(maturity_years * 2))
    if periods < 1 or abs(periods - maturity_years * 2) > 1e-9:
        raise ValueError("만기는 반년 단위여야 합니다.")
    expected = pd.period_range(month_end_yield.index[0], month_end_yield.index[-1], freq="M")
    if not isinstance(month_end_yield.index, pd.PeriodIndex) or not month_end_yield.index.equals(expected):
        raise ValueError("금리의 인덱스는 빠진 달이 없는 월이어야 합니다.")

    times = np.arange(1, periods + 1) - 1 / 6  # 반기 단위
    previous = month_end_yield.shift(1).to_numpy()
    current = month_end_yield.to_numpy()
    discount = (1 + current[:, None] / 2) ** -times[None, :]
    price = previous / 2 * discount.sum(axis=1) + discount[:, -1]
    total = pd.Series(price - 1, index=month_end_yield.index)
    return pd.DataFrame({"total_return": total, "price_return": total - previous / 12})
