import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import ExponentialSmoothing

DISCONTINUED_ZERO_MONTHS = 3
MAX_WINDOW_MONTHS = 24


def _trim_leading_zeros(series: pd.Series) -> pd.Series:
    nonzero_idx = series.to_numpy().nonzero()[0]
    if len(nonzero_idx) == 0:
        return series.iloc[0:0]
    return series.iloc[nonzero_idx[0]:]


def _is_discontinued(series: pd.Series) -> bool:
    if len(series) < DISCONTINUED_ZERO_MONTHS:
        return False
    return (series.tail(DISCONTINUED_ZERO_MONTHS) == 0).all()


def forecast_series(series: pd.Series, periods: int = 1):
    """
    series: valores ordenados cronológicamente (histórico completo, incluyendo ceros).
    Devuelve (forecast_values: list[float], method: str)
    """
    series = series.astype(float)

    if _is_discontinued(series):
        return [0.0] * periods, "discontinued"

    trimmed = _trim_leading_zeros(series)
    n = len(trimmed)

    if n == 0:
        return [0.0] * periods, "no_data"

    if n < 6:
        value = trimmed.tail(min(3, n)).mean()
        return [float(value)] * periods, "naive_mean_short_history"

    if n < 24:
        try:
            model = ExponentialSmoothing(
                trimmed.to_numpy(), trend="add", seasonal=None, damped_trend=True
            ).fit()
            forecast = model.forecast(periods)
            return [max(0.0, float(v)) for v in forecast], "holt_trend_no_seasonal"
        except Exception:
            value = trimmed.tail(3).mean()
            return [float(value)] * periods, "naive_mean_fallback"

    try:
        model = ExponentialSmoothing(
            trimmed.to_numpy(),
            trend="add",
            seasonal="add",
            seasonal_periods=12,
            damped_trend=True,
        ).fit()
        forecast = model.forecast(periods)
        return [max(0.0, float(v)) for v in forecast], "holt_winters_seasonal"
    except Exception:
        try:
            model = ExponentialSmoothing(
                trimmed.to_numpy(), trend="add", seasonal=None, damped_trend=True
            ).fit()
            forecast = model.forecast(periods)
            return [max(0.0, float(v)) for v in forecast], "holt_trend_fallback"
        except Exception:
            value = trimmed.tail(3).mean()
            return [float(value)] * periods, "naive_mean_fallback"
