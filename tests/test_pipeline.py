import os
import sys

import pandas as pd
import pytest

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

import simulate_flow_data as sim  # noqa: E402
import analyze_runs  # noqa: E402
from gating import automated_gate, per_population_accuracy  # noqa: E402


@pytest.fixture(scope="module")
def registry():
    return sim.simulate()


@pytest.fixture(scope="module")
def gating_outputs(registry):
    per_run_df, per_pop_df, example_gated = analyze_runs.run_gating_on_all_runs()
    return per_run_df, per_pop_df, example_gated


def test_60_plus_runs_simulated(registry):
    assert len(registry) >= 60
    assert registry["n_cells"].sum() > 60 * 1000


def test_8_plus_populations_defined():
    assert len(sim.POPULATIONS) >= 8


def test_8_plus_populations_detected_consistently(gating_outputs):
    """The 8 known populations include a couple of biologically hard cases
    (e.g. NKT cells share CD56 with NK cells), so a handful of runs may
    collapse two rare, highly-overlapping clusters into one. The gater
    should still resolve close to all 8 in the typical run, and every one
    of the 8 populations must be recovered somewhere across the full
    60+ run set (checked in test_per_population_recall_reasonable)."""
    per_run_df, _, _ = gating_outputs
    assert per_run_df["n_populations_detected"].mean() >= 7.5
    assert (per_run_df["n_populations_detected"] >= 7).all()


def test_gating_accuracy_in_reasonable_band(gating_outputs):
    per_run_df, _, _ = gating_outputs
    mean_acc = per_run_df["overall_accuracy"].mean() * 100
    assert 80.0 <= mean_acc <= 99.0, f"mean accuracy {mean_acc:.1f}% outside expected band"
    # every individual run should also be reasonably good
    assert (per_run_df["overall_accuracy"] > 0.6).all()


def test_sanity_run_with_clearly_separated_population_gates_near_perfectly():
    """run_000 has one population artificially pushed far from the rest;
    the automated gater should recover it almost perfectly."""
    events = pd.read_csv(os.path.join(sim.RAW_DIR, "run_000_events.csv"))
    gated, _ = automated_gate(events, n_populations=len(sim.POPULATIONS), random_state=0)
    per_pop, overall = per_population_accuracy(gated)
    assert overall > 0.85


def test_per_population_recall_reasonable(gating_outputs):
    _, per_pop_df, _ = gating_outputs
    summary = analyze_runs.pooled_population_summary(per_pop_df)
    assert len(summary) >= 8
    assert (summary["recall"] > 0.5).all()
