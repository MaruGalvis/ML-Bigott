import pandas as pd

from db import get_engine

QUERY = """
SELECT
    st."year",
    st."month",
    r."name" AS region,
    p."name" AS product,
    p."type" AS product_type,
    st."targetSalesVolume" AS target
FROM sales_target st
JOIN region r ON r."regionId" = st."regionId"
JOIN product p ON p."productId" = st."productId"
ORDER BY st."year", st."month", region, product;
"""


def load_data() -> pd.DataFrame:
    engine = get_engine()
    df = pd.read_sql(QUERY, engine)
    df["period"] = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1))
    return df


def main():
    df = load_data()

    print("=== Resumen general ===")
    print(f"Filas totales: {len(df)}")
    print(f"Rango de fechas: {df['period'].min().date()} -> {df['period'].max().date()}")
    print(f"Regiones ({df['region'].nunique()}): {sorted(df['region'].unique())}")
    print(f"Productos ({df['product'].nunique()}): {sorted(df['product'].unique())}")

    print("\n=== Meses cubiertos ===")
    months = df[["year", "month"]].drop_duplicates().sort_values(["year", "month"])
    print(months.to_string(index=False))
    print(f"Total meses distintos: {len(months)}")

    print("\n=== Combinaciones región-producto esperadas vs presentes por mes ===")
    expected_combos = df["region"].nunique() * df["product"].nunique()
    per_month = df.groupby(["year", "month"]).size().reset_index(name="filas")
    per_month["combos_esperadas"] = expected_combos
    per_month["faltantes"] = per_month["combos_esperadas"] - per_month["filas"]
    print(per_month.to_string(index=False))

    print("\n=== Valores en cero por producto (posibles descontinuados o sin datos) ===")
    zero_share = (
        df.assign(is_zero=df["target"] == 0)
        .groupby("product")["is_zero"]
        .mean()
        .sort_values(ascending=False)
    )
    print((zero_share * 100).round(1).astype(str) + "%")

    print("\n=== Estadísticos por producto (para detectar outliers) ===")
    stats = df.groupby("product")["target"].describe()
    print(stats.round(1))

    print("\n=== Valores nulos ===")
    print(df.isnull().sum())

    df.to_csv("sales_target_history.csv", index=False)
    print("\nHistórico completo exportado a sales_target_history.csv")


if __name__ == "__main__":
    main()
