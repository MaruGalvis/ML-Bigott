import numpy as np
import pandas as pd

from explore_data import load_data
from forecast import forecast_series

HOLDOUT_MONTHS = 3


def run_backtest():
    df = load_data()
    results = []

    for (region, product), group in df.groupby(["region", "product"]):
        group = group.sort_values("period")
        series = group.set_index("period")["target"]

        if len(series) <= HOLDOUT_MONTHS + 3:
            continue

        train = series.iloc[:-HOLDOUT_MONTHS]
        actual = series.iloc[-HOLDOUT_MONTHS:]

        forecast_values, method = forecast_series(train, periods=HOLDOUT_MONTHS)

        for i, (period, actual_value) in enumerate(actual.items()):
            predicted = forecast_values[i]
            error = predicted - actual_value
            abs_pct_error = (
                abs(error) / actual_value * 100 if actual_value != 0 else (0 if predicted == 0 else np.nan)
            )
            results.append(
                {
                    "region": region,
                    "product": product,
                    "period": period,
                    "actual": actual_value,
                    "predicted": round(predicted, 0),
                    "error": round(error, 0),
                    "abs_pct_error": abs_pct_error,
                    "method": method,
                }
            )

    return pd.DataFrame(results)


def wape(group: pd.DataFrame) -> float:
    total_actual = group["actual"].sum()
    if total_actual == 0:
        return 0.0
    return group["error"].abs().sum() / total_actual * 100


def main():
    results = run_backtest()
    results.to_csv("backtest_results.csv", index=False)

    print("=== MAPE (simple, sensible a valores chicos) por producto ===")
    mape_by_product = (
        results.groupby("product")["abs_pct_error"].mean().sort_values(ascending=False)
    )
    print(mape_by_product.round(1))

    print("\n=== WAPE (ponderado por volumen real, más representativo del negocio) por producto ===")
    wape_by_product = results.groupby("product").apply(wape).sort_values(ascending=False)
    print(wape_by_product.round(1))

    print(f"\n=== WAPE global === {wape(results):.1f}%")
    print(f"=== MAPE global (referencia) === {results['abs_pct_error'].mean():.1f}%")

    print("\n=== Peores 10 predicciones individuales (mayor error absoluto) ===")
    worst = results.reindex(results["error"].abs().sort_values(ascending=False).index).head(10)
    print(worst.to_string(index=False))

    print("\nDetalle completo exportado a backtest_results.csv")


if __name__ == "__main__":
    main()
