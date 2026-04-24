"""
plots.py
This script takes all the CSV data we generated in benchmark.py
and turns it into nice-looking graphs for the final report.

To run:
    python experiments/plots.py
"""
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import numpy as np

RESULTS_DIR = Path(__file__).parent / "results"
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ── Style Setup ───────────────────────────────────────────────────────────────
# Making the graphs look clean and modern for the report
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
COLORS = {"tfidf": "#4C72B0", "hybrid": "#8172B3", "minhash": "#DD8452", "simhash": "#55A868"}

def _save(name: str) -> None:
    path = FIGURES_DIR / name
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved -> {path}")

# ── Plot 1: Latency comparison (Exp 1) ───────────────────────────────────────

def plot_latency_comparison() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp1_comparison.csv")
    means = {
        "TF-IDF":      df["tfidf_latency_ms"].mean(),
        "Hybrid LSH":  df["hybrid_lsh_latency_ms"].mean() if "hybrid_lsh_latency_ms" in df else None,
        "MinHash+LSH": df["minhash_latency_ms"].mean(),
        "SimHash":     df["simhash_latency_ms"].mean(),
    }
    means = {k: v for k, v in means.items() if v is not None}
    colors = {
        "TF-IDF": COLORS["tfidf"],
        "Hybrid LSH": COLORS["hybrid"],
        "MinHash+LSH": COLORS["minhash"],
        "SimHash": COLORS["simhash"],
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.keys(), means.values(),
                  color=[colors[label] for label in means],
                  edgecolor="white", linewidth=0.8)
    
    # Add the exact numbers on top of the bars
    ax.bar_label(bars, fmt="%.2f ms", padding=3, fontsize=10)
    ax.set_ylabel("Average Query Latency (ms)")
    ax.set_title("Exact vs Approximate: Query Latency")
    _save("exp1_latency.png")

def plot_precision_comparison() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp1_comparison.csv")
    means = {
        "TF-IDF (baseline)": 1.0,
        "Hybrid LSH":         df["hybrid_lsh_precision5"].mean() if "hybrid_lsh_precision5" in df else None,
        "MinHash+LSH":       df["minhash_precision5"].mean(),
        "SimHash":           df["simhash_precision5"].mean(),
    }
    means = {k: v for k, v in means.items() if v is not None}
    colors = {
        "TF-IDF (baseline)": COLORS["tfidf"],
        "Hybrid LSH": COLORS["hybrid"],
        "MinHash+LSH": COLORS["minhash"],
        "SimHash": COLORS["simhash"],
    }
    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(means.keys(), means.values(),
                  color=[colors[label] for label in means],
                  edgecolor="white", linewidth=0.8)
    ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Precision@5")
    ax.set_title("Exact vs Approximate: Precision@5\n(TF-IDF used as ground truth)")
    _save("exp1_precision.png")

# ── Plot 2a: MinHash sensitivity ──────────────────────────────────────────────

def plot_minhash_sensitivity() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp2a_minhash_sensitivity.csv")

    # We'll put latency and precision side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Latency vs num_perm grouped by num_bands
    for bands, grp in df.groupby("num_bands"):
        axes[0].plot(grp["num_perm"], grp["avg_latency_ms"],
                     marker="o", label=f"bands={bands}")
    axes[0].set_xlabel("Number of Permutations (num_perm)")
    axes[0].set_ylabel("Avg Latency (ms)")
    axes[0].set_title("MinHash: num_perm vs Query Latency")
    axes[0].legend()

    # Precision vs num_perm
    for bands, grp in df.groupby("num_bands"):
        axes[1].plot(grp["num_perm"], grp["avg_precision5"],
                     marker="s", label=f"bands={bands}")
    axes[1].set_xlabel("Number of Permutations (num_perm)")
    axes[1].set_ylabel("Avg Precision@5")
    axes[1].set_title("MinHash: num_perm vs Precision@5")
    axes[1].legend()

    _save("exp2a_minhash_sensitivity.png")

# ── Plot 2b: SimHash sensitivity ──────────────────────────────────────────────

def plot_simhash_sensitivity() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp2b_simhash_sensitivity.csv")
    # Replace "None" string with a number so matplotlib doesn't freak out
    df["hamming_threshold"] = df["hamming_threshold"].replace("None", 64).astype(float)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df["hamming_threshold"], df["avg_precision5"],
            marker="o", color=COLORS["simhash"], label="Precision@5")
    ax.plot(df["hamming_threshold"], df["avg_recall5"],
            marker="s", color=COLORS["tfidf"], linestyle="--", label="Recall@5")
    ax.set_xlabel("Hamming Threshold (max differing bits)")
    ax.set_ylabel("Score")
    ax.set_title("SimHash: Hamming Threshold vs Precision & Recall")
    ax.legend()
    _save("exp2b_simhash_sensitivity.png")

# ── Plot 3: Scalability ───────────────────────────────────────────────────────

def plot_scalability() -> None:
    df = pd.read_csv(RESULTS_DIR / "exp3_scalability.csv")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(df["num_chunks"], df["tfidf_latency_ms"],
            marker="o", color=COLORS["tfidf"], label="TF-IDF (exact)")
    if "hybrid_lsh_latency_ms" in df:
        ax.plot(df["num_chunks"], df["hybrid_lsh_latency_ms"],
                marker="D", color=COLORS["hybrid"], label="Hybrid LSH")
    ax.plot(df["num_chunks"], df["minhash_latency_ms"],
            marker="s", color=COLORS["minhash"], label="MinHash+LSH")
    ax.plot(df["num_chunks"], df["simhash_latency_ms"],
            marker="^", color=COLORS["simhash"], label="SimHash")
    
    # Log scale is crucial here, otherwise the graph just looks like a hockey stick
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Chunks (log scale)")
    ax.set_ylabel("Query Latency in ms (log scale)")
    ax.set_title("Scalability: Query Latency vs Corpus Size")
    ax.legend()
    _save("exp3_scalability.png")

if __name__ == "__main__":
    print("Generating experiment figures...")
    plot_latency_comparison()
    plot_precision_comparison()
    plot_minhash_sensitivity()
    plot_simhash_sensitivity()
    plot_scalability()
    print(f"\nAll figures saved to {FIGURES_DIR.resolve()}")
