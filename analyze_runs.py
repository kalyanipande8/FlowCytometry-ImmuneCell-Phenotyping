"""
analyze_runs.py

Applies the automated GMM gating pipeline (gating.py) to every simulated
run in data/raw/, validates it against the known ground-truth population
labels, and writes:

  - output/gating_results_per_run.csv     overall accuracy per run
  - output/population_summary.csv         per-population recall, pooled across all runs
  - output/figures/example_gating.png     example 2-marker scatter (true vs predicted)
  - output/figures/accuracy_by_population.png
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

import simulate_flow_data as sim
from gating import automated_gate, per_population_accuracy

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "output")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")


def run_gating_on_all_runs(raw_dir=sim.RAW_DIR):
    registry = pd.read_csv(os.path.join(raw_dir, "run_registry.csv"))

    per_run_rows = []
    all_per_pop = []
    example_gated = None

    for _, row in registry.iterrows():
        run_id = row["run_id"]
        events = pd.read_csv(os.path.join(raw_dir, f"{run_id}_events.csv"))

        run_seed = int(run_id.split("_")[1])
        gated, _ = automated_gate(events, n_populations=row["n_populations"], random_state=run_seed)
        per_pop, overall = per_population_accuracy(gated)
        per_pop["run_id"] = run_id

        per_run_rows.append({
            "run_id": run_id,
            "n_cells": row["n_cells"],
            "n_populations_detected": gated["predicted_population"].nunique(),
            "overall_accuracy": overall,
            "is_sanity_run": row["is_sanity_run"],
        })
        all_per_pop.append(per_pop)

        if run_id == "run_001":
            example_gated = gated

    per_run_df = pd.DataFrame(per_run_rows)
    per_pop_df = pd.concat(all_per_pop, ignore_index=True)
    return per_run_df, per_pop_df, example_gated


def pooled_population_summary(per_pop_df):
    summary = (
        per_pop_df.groupby("population")
        .agg(n_true_cells=("n_true_cells", "sum"), n_correct=("n_correct", "sum"))
        .reset_index()
    )
    summary["recall"] = summary["n_correct"] / summary["n_true_cells"]
    return summary.sort_values("recall", ascending=False).reset_index(drop=True)


def make_figures(example_gated, pop_summary):
    os.makedirs(FIG_DIR, exist_ok=True)

    # Example gating scatter: CD4 vs CD8 with true labels vs predicted labels
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    populations = sorted(example_gated["true_population"].unique())
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(populations)}

    for pop in populations:
        sub = example_gated[example_gated["true_population"] == pop]
        axes[0].scatter(sub["CD4"], sub["CD8"], s=4, alpha=0.5, color=color_map[pop], label=pop)
    axes[0].set_title("Ground truth")
    axes[0].set_xlabel("CD4 (log fluorescence)")
    axes[0].set_ylabel("CD8 (log fluorescence)")

    for pop in populations:
        sub = example_gated[example_gated["predicted_population"] == pop]
        axes[1].scatter(sub["CD4"], sub["CD8"], s=4, alpha=0.5, color=color_map[pop], label=pop)
    axes[1].set_title("Automated GMM gating")
    axes[1].set_xlabel("CD4 (log fluorescence)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle("Example run: automated gating vs. known ground truth (CD4 vs CD8)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "example_gating.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Per-population recall bar chart
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.barh(pop_summary["population"], pop_summary["recall"] * 100, color="#4C72B0")
    ax2.set_xlabel("Recall (%) pooled across all runs")
    ax2.set_title("Automated gating accuracy by immune-cell population")
    ax2.set_xlim(0, 100)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIG_DIR, "accuracy_by_population.png"), dpi=150)
    plt.close(fig2)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    per_run_df, per_pop_df, example_gated = run_gating_on_all_runs()
    pop_summary = pooled_population_summary(per_pop_df)

    per_run_df.to_csv(os.path.join(OUTPUT_DIR, "gating_results_per_run.csv"), index=False)
    pop_summary.to_csv(os.path.join(OUTPUT_DIR, "population_summary.csv"), index=False)

    make_figures(example_gated, pop_summary)

    print(f"Runs analyzed: {len(per_run_df)}")
    print(f"Mean overall gating accuracy: {per_run_df['overall_accuracy'].mean() * 100:.1f}%")
    print(f"Populations detected per run (mean): {per_run_df['n_populations_detected'].mean():.1f}")
    print("\nPer-population recall (pooled):")
    print(pop_summary.to_string(index=False))

    return per_run_df, pop_summary


if __name__ == "__main__":
    main()
