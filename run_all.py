"""
run_all.py

Runs the full pipeline end-to-end:
  1. Simulate 65 multiparameter flow cytometry runs (8 immune-cell
     subpopulations each) -> data/raw/
  2. Automated GMM-based gating on every run, validated against
     ground-truth labels -> output/

Run with:  python run_all.py
"""

import simulate_flow_data
import analyze_runs


def main():
    print("--- Simulating flow cytometry runs ---")
    registry = simulate_flow_data.simulate()
    print(f"Simulated {len(registry)} runs, {registry['n_cells'].sum()} total events")

    print("\n--- Automated gating + validation across all runs ---")
    analyze_runs.main()


if __name__ == "__main__":
    main()
