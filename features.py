import numpy as np
import pandas as pd

from explore_data import load_data

LAGS = [1, 2, 3, 12]
ROLLING_WINDOWS = [3, 6]


def build_feature_table(df: pd.DataFrame = None) -> pd.DataFrame:
    if df is None:
        df = load_data()
    df = df.sort_values(["region", "product", "period"]).reset_index(drop=True)

    group_cols = ["region", "product"]

    for lag in LAGS:
        df[f"lag_{lag}"] = df.groupby(group_cols)["target"].shift(lag)

    shifted = df.groupby(group_cols)["target"].shift(1)
    for window in ROLLING_WINDOWS:
        df[f"rolling_mean_{window}"] = shifted.groupby(
            [df["region"], df["product"]]
        ).transform(lambda s: s.rolling(window).mean())

    df["trend_3m"] = df["lag_1"] - df["lag_3"]
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["months_since_start"] = df.groupby(group_cols).cumcount()

    return df


FEATURE_COLUMNS_NUMERIC = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_12",
    "rolling_mean_3",
    "rolling_mean_6",
    "trend_3m",
    "month_sin",
    "month_cos",
    "months_since_start",
]
FEATURE_COLUMNS_CATEGORICAL = ["region", "product", "product_type"]


def get_model_ready_data(df: pd.DataFrame):
    """Filas con suficiente historial (requieren lag_12) y sin nulos en features."""
    required_cols = FEATURE_COLUMNS_NUMERIC + FEATURE_COLUMNS_CATEGORICAL + ["target"]
    clean = df.dropna(subset=required_cols).copy()
    return clean


if __name__ == "__main__":
    df = build_feature_table()
    clean = get_model_ready_data(df)
    print(f"Filas totales tras feature engineering: {len(df)}")
    print(f"Filas utilizables para entrenar (con historial suficiente): {len(clean)}")
    print("\nEjemplo de features:")
    print(clean[["period", "region", "product"] + FEATURE_COLUMNS_NUMERIC + ["target"]].head(8))
