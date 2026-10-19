import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from features import FEATURE_COLUMNS_CATEGORICAL, FEATURE_COLUMNS_NUMERIC, build_feature_table, get_model_ready_data

TEST_MONTHS = 3


def time_based_split(df: pd.DataFrame):
    periods = sorted(df["period"].unique())
    test_periods = periods[-TEST_MONTHS:]
    train = df[~df["period"].isin(test_periods)]
    test = df[df["period"].isin(test_periods)]
    return train, test


def build_preprocessor():
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURE_COLUMNS_CATEGORICAL),
        ],
        remainder="passthrough",
    )


def wape(actual, predicted):
    actual = np.asarray(actual)
    predicted = np.asarray(predicted)
    total = actual.sum()
    if total == 0:
        return 0.0
    return np.abs(predicted - actual).sum() / total * 100


def evaluate(name, pipeline, X_test, y_test):
    preds = np.clip(pipeline.predict(X_test), 0, None)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    w = wape(y_test, preds)
    print(f"{name}: MAE={mae:,.0f}  RMSE={rmse:,.0f}  WAPE={w:.1f}%")
    return {"name": name, "pipeline": pipeline, "mae": mae, "rmse": rmse, "wape": w, "preds": preds}


def main():
    df = build_feature_table()
    clean = get_model_ready_data(df)
    train, test = time_based_split(clean)

    feature_cols = FEATURE_COLUMNS_CATEGORICAL + FEATURE_COLUMNS_NUMERIC
    X_train, y_train = train[feature_cols], train["target"]
    X_test, y_test = test[feature_cols], test["target"]

    print(f"Train: {len(train)} filas | Test: {len(test)} filas ({TEST_MONTHS} meses)\n")

    tscv = TimeSeriesSplit(n_splits=4)

    results = []

    rf_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", RandomForestRegressor(random_state=42)),
        ]
    )
    rf_param_grid = {
        "model__n_estimators": [100, 200, 300, 400],
        "model__max_depth": [None, 5, 8, 12, 16],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", 0.5, 1.0],
    }
    rf_search = RandomizedSearchCV(
        rf_pipeline,
        rf_param_grid,
        n_iter=25,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)
    print("Mejores hiperparámetros Random Forest:", rf_search.best_params_)
    results.append(evaluate("Random Forest", rf_search.best_estimator_, X_test, y_test))

    gb_pipeline = Pipeline(
        [
            ("preprocess", build_preprocessor()),
            ("model", GradientBoostingRegressor(random_state=42)),
        ]
    )
    gb_param_grid = {
        "model__n_estimators": [100, 200, 300],
        "model__max_depth": [2, 3, 4, 5],
        "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
        "model__subsample": [0.7, 0.85, 1.0],
    }
    gb_search = RandomizedSearchCV(
        gb_pipeline,
        gb_param_grid,
        n_iter=25,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        random_state=42,
        n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)
    print("Mejores hiperparámetros Gradient Boosting:", gb_search.best_params_)
    results.append(evaluate("Gradient Boosting", gb_search.best_estimator_, X_test, y_test))

    print("\n=== Comparación final (conjunto de prueba, 3 meses ocultos) ===")
    for r in sorted(results, key=lambda x: x["wape"]):
        print(f"{r['name']:20s} WAPE={r['wape']:.1f}%  MAE={r['mae']:,.0f}")

    best = min(results, key=lambda x: x["wape"])
    print(f"\nModelo ganador: {best['name']} (WAPE {best['wape']:.1f}%)")

    return best, clean, feature_cols


if __name__ == "__main__":
    main()
