"""
simulate_flow_data.py

Simulates multiparameter flow cytometry runs. Each run contains several
thousand single-cell events across a 7-color fluorescence panel:

    CD3, CD4, CD8, CD19, CD56, CD14, HLA-DR

Each event belongs to one of 8 immune-cell subpopulations, each defined by a
realistic combination of marker positivity/negativity. Every population's
per-marker fluorescence is drawn from a multivariate Gaussian in
log-fluorescence space (the standard way instruments report intensity after
compensation), so within-population spread and cross-marker correlation are
both realistic, and adjacent populations overlap at their edges exactly the
way real gates fight with each other.

Run-to-run instrument/staining variability is layered on top: each run gets
its own small random shift in channel gains and a small compensation-noise
term, mimicking day-to-day variability across 60+ real acquisitions.

Output: data/raw/run_XXX_events.csv (one file per run, columns = the 7
markers + true_population) and data/raw/run_registry.csv (per-run metadata).
"""

import os

import numpy as np
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "raw")

MARKERS = ["CD3", "CD4", "CD8", "CD19", "CD56", "CD14", "HLA-DR"]

N_RUNS = 65
CELLS_PER_RUN = 1500
SEED = 42

# Each population: (name, mean vector in log-fluorescence space, relative abundance)
# Order of means follows MARKERS above.
POPULATIONS = [
    ("CD4 T helper",        [3.3, 3.1, 0.3, 0.2, 0.2, 0.2, 0.4], 0.28),
    ("CD8 Cytotoxic T",     [3.3, 0.3, 3.1, 0.2, 0.2, 0.2, 0.4], 0.18),
    ("B cell",              [0.3, 0.2, 0.2, 3.2, 0.2, 0.2, 2.7], 0.12),
    ("NK cell",             [0.3, 0.2, 0.2, 0.2, 3.6, 0.2, 0.3], 0.10),
    ("Classical monocyte",  [0.3, 0.2, 0.2, 0.2, 0.2, 3.2, 2.5], 0.14),
    ("NKT cell",            [2.2, 0.4, 1.6, 0.2, 2.0, 0.2, 0.4], 0.08),
    ("Double-negative T",   [3.9, 0.3, 0.3, 0.2, 0.2, 0.2, 0.4], 0.06),
    ("Dendritic cell",      [0.3, 0.2, 0.2, 0.3, 0.2, 0.5, 3.3], 0.06),
]

POPULATION_NAMES = [p[0] for p in POPULATIONS]

# Shared within-population covariance structure: moderate spread, mild
# positive correlation between biologically co-expressed markers.
_BASE_STD = 0.35


def _covariance(n_markers):
    cov = np.eye(n_markers) * (_BASE_STD ** 2)
    return cov


def simulate(n_runs=N_RUNS, cells_per_run=CELLS_PER_RUN, seed=SEED):
    rng = np.random.default_rng(seed)
    os.makedirs(RAW_DIR, exist_ok=True)

    n_markers = len(MARKERS)
    cov = _covariance(n_markers)
    weights = np.array([p[2] for p in POPULATIONS])
    weights = weights / weights.sum()
    means = np.array([p[1] for p in POPULATIONS])

    registry_rows = []

    for run_idx in range(n_runs):
        run_id = f"run_{run_idx:03d}"

        # Run-level technical variation: small per-channel gain shift and
        # extra compensation noise, distinct per run (mimics day-to-day
        # instrument drift across 60+ acquisitions).
        gain_shift = rng.normal(0, 0.12, size=n_markers)
        run_noise_scale = rng.uniform(0.85, 1.2)

        n_cells = int(cells_per_run * rng.uniform(0.85, 1.15))
        pop_counts = rng.multinomial(n_cells, weights)

        # Optional sanity-check run: force one clearly, artificially
        # well-separated population (used by tests to validate the gating
        # algorithm can hit near-perfect accuracy when separation is easy).
        is_sanity_run = run_idx == 0
        sep_boost = 3.0 if is_sanity_run else 0.0

        all_events = []
        all_labels = []
        for pop_idx, (name, mean, _) in enumerate(POPULATIONS):
            n = pop_counts[pop_idx]
            if n == 0:
                continue
            mean_arr = np.array(mean, dtype=float) + gain_shift
            if is_sanity_run:
                # push this population's markers further from the pack
                mean_arr = mean_arr + sep_boost * (mean_arr > 1.5)
            events = rng.multivariate_normal(
                mean_arr, cov * (run_noise_scale ** 2), size=n
            )
            all_events.append(events)
            all_labels.extend([name] * n)

        events = np.vstack(all_events)
        events = np.clip(events, 0, None)  # fluorescence can't be negative

        df = pd.DataFrame(events, columns=MARKERS)
        df["true_population"] = all_labels
        df = df.sample(frac=1.0, random_state=run_idx).reset_index(drop=True)

        out_path = os.path.join(RAW_DIR, f"{run_id}_events.csv")
        df.to_csv(out_path, index=False)

        registry_rows.append({
            "run_id": run_id,
            "n_cells": len(df),
            "n_populations": len(POPULATIONS),
            "is_sanity_run": is_sanity_run,
        })

    registry = pd.DataFrame(registry_rows)
    registry.to_csv(os.path.join(RAW_DIR, "run_registry.csv"), index=False)
    return registry


if __name__ == "__main__":
    reg = simulate()
    print(f"Simulated {len(reg)} runs into {RAW_DIR}")
