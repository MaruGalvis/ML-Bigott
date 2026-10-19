import math

import matplotlib.pyplot as plt
import pandas as pd

from explore_data import load_data


def plot_by_product(df: pd.DataFrame, out_path: str):
    products = sorted(df["product"].unique())
    n = len(products)
    cols = 2
    rows = math.ceil(n / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows), sharex=True)
    axes = axes.flatten()

    for ax, product in zip(axes, products):
        sub = df[df["product"] == product]
        for region, group in sub.groupby("region"):
            group = group.sort_values("period")
            ax.plot(group["period"], group["target"], marker="o", markersize=2, label=region)
        ax.set_title(product)
        ax.tick_params(axis="x", rotation=45)

    for ax in axes[n:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4)
    fig.suptitle("Target de ventas por producto y región (2024-2026)", y=1.02, fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    print(f"Guardado: {out_path}")


def plot_national_total(df: pd.DataFrame, out_path: str):
    total = df.groupby("period")["target"].sum().reset_index().sort_values("period")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(total["period"], total["target"], marker="o")
    ax.set_title("Total nacional (todas las regiones y productos)")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", dpi=130)
    print(f"Guardado: {out_path}")


def main():
    df = load_data()
    plot_by_product(df, "plot_by_product.png")
    plot_national_total(df, "plot_national_total.png")


if __name__ == "__main__":
    main()
