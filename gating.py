"""
gating.py

Automated gating: a real unsupervised clustering algorithm (Gaussian Mixture
Model, sklearn) that identifies immune-cell subpopulations directly from
multi-channel fluorescence data, with no hardcoded marker thresholds.

For each run:
  1. Fit a GaussianMixture with n_components = number of expected
     populations on the (standardized) marker intensities.
  2. Assign every event to its most likely component.
  3. Map each cluster to the immune-cell population whose true label is
     the majority vote within that cluster (label matching only - the
     clustering itself never sees the true labels).
  4. Compare the mapped cluster labels against the known ground-truth
     simulated labels to compute a real per-run and per-population
     accuracy metric.
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from simulate_flow_data import MARKERS


def automated_gate(df, n_populations, random_state=0):
    """Run GMM-based automated gating on one run's event table.

    Returns the input dataframe with two added columns:
      - cluster: raw GMM component id
      - predicted_population: cluster id mapped to a population name via
        majority-vote against true_population (used only for evaluation/
        labeling, not for fitting the model).
    """
    X = df[MARKERS].values
    X_scaled = StandardScaler().fit_transform(X)

    gmm = GaussianMixture(
        n_components=n_populations,
        covariance_type="full",
        n_init=3,
        random_state=random_state,
    )
    clusters = gmm.fit_predict(X_scaled)

    out = df.copy()
    out["cluster"] = clusters

    # Map each cluster to the population name that appears most often in it.
    cluster_to_pop = {}
    for c in np.unique(clusters):
        mask = clusters == c
        majority = out.loc[mask, "true_population"].value_counts().idxmax()
        cluster_to_pop[c] = majority
    out["predicted_population"] = out["cluster"].map(cluster_to_pop)

    return out, gmm


def per_population_accuracy(gated_df):
    """Per-population recall (fraction of true members correctly assigned)
    and overall accuracy (fraction of all events correctly assigned)."""
    rows = []
    for pop, group in gated_df.groupby("true_population"):
        correct = (group["predicted_population"] == pop).sum()
        rows.append({
            "population": pop,
            "n_true_cells": len(group),
            "n_correct": correct,
            "recall": correct / len(group),
        })
    per_pop = pd.DataFrame(rows).sort_values("population").reset_index(drop=True)

    overall_accuracy = (gated_df["predicted_population"] == gated_df["true_population"]).mean()
    return per_pop, overall_accuracy
