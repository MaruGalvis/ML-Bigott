import pandas as pd

from explore_data import load_data
from forecast import forecast_series


def predict_all(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    last_period = df["period"].max()
    next_period = last_period + pd.DateOffset(months=1)

    for (region, product), group in df.groupby(["region", "product"]):
        group = group.sort_values("period")
        series = group.set_index("period")["target"]
        forecast_values, method = forecast_series(series, periods=1)
        rows.append(
            {
                "region": region,
                "product": product,
                "predicted_target": round(forecast_values[0], -2),
                "method": method,
            }
        )

    result = pd.DataFrame(rows)
    result.attrs["next_period"] = next_period
    return result


def build_pivot(predictions: pd.DataFrame) -> pd.DataFrame:
    pivot = predictions.pivot(index="product", columns="region", values="predicted_target")
    pivot["SOP"] = pivot.sum(axis=1)
    total_row = pivot.sum(axis=0)
    total_row.name = "Total"
    pivot = pd.concat([pivot, total_row.to_frame().T])
    return pivot


def main():
    df = load_data()
    predictions = predict_all(df)
    next_period = predictions.attrs.get("next_period")

    print(f"=== Predicción para {next_period.strftime('%B %Y')} ===\n")

    pivot = build_pivot(predictions)
    pd.set_option("display.float_format", lambda x: f"{x:,.0f}")
    print(pivot)

    predictions.to_csv("predictions_detail.csv", index=False)
    pivot.to_csv("predictions_pivot.csv")
    print("\nGuardado: predictions_detail.csv y predictions_pivot.csv")

    print("\n=== Método usado por producto/región ===")
    print(predictions.groupby("method").size())


if __name__ == "__main__":
    main()
