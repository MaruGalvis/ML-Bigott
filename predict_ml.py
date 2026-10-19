import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from explore_data import load_data
from features import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC, build_feature_table, get_model_ready_data
from train_model import build_preprocessor

DISCONTINUED_ZERO_MONTHS = 3

# Mejores hiperparámetros encontrados por RandomizedSearchCV en train_model.py
BEST_RF_PARAMS = dict(
    n_estimators=100,
    min_samples_leaf=1,
    max_features=1.0,
    max_depth=8,
    random_state=42,
)


def add_future_rows(df: pd.DataFrame) -> pd.DataFrame:
    last_period = df["period"].max()
    next_period = last_period + pd.DateOffset(months=1)

    last_rows = (
        df.sort_values("period")
        .groupby(["region", "product"], as_index=False)
        .tail(1)
        .copy()
    )
    last_rows["period"] = next_period
    last_rows["month"] = next_period.month
    last_rows["year"] = next_period.year
    last_rows["target"] = float("nan")

    return pd.concat([df, last_rows], ignore_index=True), next_period


def train_final_model(train_df: pd.DataFrame):
    feature_cols = FEATURE_COLUMNS_CATEGORICAL + FEATURE_COLUMNS_NUMERIC
    X = train_df[feature_cols]
    y = train_df["target"]

    pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(**BEST_RF_PARAMS)),
        ]
    )
    pipeline.fit(X, y)
    return pipeline, feature_cols


def apply_discontinued_rule(predictions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Si las últimas N observaciones reales de una región-producto fueron 0,
    se respeta el 0 (venta detenida) sin importar lo que prediga el modelo."""
    last_n = (
        history.sort_values("period")
        .groupby(["region", "product"])
        .tail(DISCONTINUED_ZERO_MONTHS)
    )
    is_discontinued = (
        last_n.groupby(["region", "product"])["target"]
        .apply(lambda s: len(s) == DISCONTINUED_ZERO_MONTHS and (s == 0).all())
    )
    discontinued_keys = set(is_discontinued[is_discontinued].index)

    predictions = predictions.copy()
    mask = predictions.apply(
        lambda row: (row["region"], row["product"]) in discontinued_keys, axis=1
    )
    predictions.loc[mask, "predicted_target"] = 0.0
    if mask.any():
        print("Regla de negocio aplicada (forzado a 0 por venta detenida):")
        print(predictions.loc[mask, ["region", "product"]].to_string(index=False))
    return predictions


def build_pivot(predictions: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot(index="product", columns="region", values="predicted_target")
    pivot["SOP"] = pivot.sum(axis=1)
    total_row = pivot.sum(axis=0)
    total_row.name = "Total"
    pivot = pd.concat([pivot, total_row.to_frame().T])
    return pivot


def generate_predictions(df: pd.DataFrame = None):
    """Entrena el modelo con el histórico disponible y predice el siguiente mes.

    Devuelve (result, next_period) donde result tiene columnas
    region, product, predicted_target, period.
    """
    if df is None:
        df = load_data()
    extended_raw, next_period = add_future_rows(df)

    features_df = build_feature_table(extended_raw)
    train_data = get_model_ready_data(features_df[features_df["period"] <= df["period"].max()])

    model, feature_cols = train_final_model(train_data)

    future_rows = features_df[features_df["period"] == next_period].copy()
    missing = future_rows[feature_cols].isnull().any(axis=1)
    skipped = future_rows.loc[missing, ["region", "product"]] if missing.any() else None
    if missing.any():
        future_rows = future_rows.loc[~missing]

    preds = model.predict(future_rows[feature_cols]).clip(min=0)
    future_rows["predicted_target"] = preds.round(-2)

    result = apply_discontinued_rule(
        future_rows[["region", "product", "predicted_target"]], df
    )
    result["period"] = next_period

    return result, next_period, skipped


def main():
    df = load_data()
    result, next_period, skipped = generate_predictions(df)

    if skipped is not None:
        print("Advertencia: filas sin suficiente historial para predecir (se omiten):")
        print(skipped)

    print(f"=== Predicción (Random Forest + regla de negocio) para {next_period.strftime('%B %Y')} ===\n")
    pivot = build_pivot(result.drop(columns="period"))
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(pivot)

    result.to_csv("predictions_ml_detail.csv", index=False)
    pivot.to_csv("predictions_ml_pivot.csv")
    print("\nGuardado: predictions_ml_detail.csv y predictions_ml_pivot.csv")


if __name__ == "__main__":
    main()
