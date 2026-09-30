# FlowCytometry-ImmuneCell-Phenotyping

Automated gating and validation for multiparameter flow cytometry immune-cell
phenotyping. Simulates a 7-color fluorescence panel (CD3, CD4, CD8, CD19,
CD56, CD14, HLA-DR) across **65 acquisition runs**, fits a real unsupervised
**Gaussian Mixture Model** (scikit-learn) to identify **8 immune-cell
subpopulations** per run with no hardcoded intensity thresholds, and
validates the automated calls against the known ground-truth population
labels.

> **Disclaimer:** This is a portfolio/demo project. All fluorescence data is
> **synthetically simulated** by `simulate_flow_data.py` using realistic
> marker-expression logic for each population (multivariate Gaussians in
> log-fluorescence space with population-appropriate covariance and
> run-to-run instrument drift). No real patient, sample, or employer data is
> used.

## The 8 immune-cell populations

| Population | Defining markers |
|---|---|
| CD4 T helper | CD3+ CD4+ |
| CD8 Cytotoxic T | CD3+ CD8+ |
| B cell | CD19+ HLA-DR+ |
| NK cell | CD56+ CD3- |
| Classical monocyte | CD14+ HLA-DR+ |
| NKT cell | CD3+ CD56+ (dim CD8) |
| Double-negative T | CD3+ CD4- CD8- |
| Dendritic cell | HLA-DR+ CD14- CD3- CD19- CD56- |

## What it does

1. **`simulate_flow_data.py`** simulates 65 runs (~1,500 cells/run before
   run-to-run variation), each with per-run technical variability
   (channel-gain shift + compensation noise) mimicking day-to-day instrument
   drift. One run (`run_000`) additionally has a clearly, artificially
   separated population, used as a sanity check on the gating algorithm.
   Ground-truth population labels are stored alongside the fluorescence
   data but never given to the clustering algorithm.

2. **`gating.py`** implements the automated gating: standardizes the 7
   marker channels and fits a `sklearn.mixture.GaussianMixture` (full
   covariance, `n_init=3`) with 8 components per run. Each fitted cluster
   is mapped to the immune population that is the majority ground-truth
   label within it (label matching only, used for evaluation - the
   clustering itself is fully unsupervised). This is a real
   density-based/model-based clustering algorithm, not a set of manual
   intensity gates.

3. **`analyze_runs.py`** runs this gating pipeline across all 65 runs,
   computes per-run overall accuracy and per-population recall against the
   known ground truth, aggregates results, and produces figures.

4. **`run_all.py`** runs the whole pipeline.

## How to run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_all.py
pytest -q
```

## Results

_All numbers below are copied directly from an actual run of `python
run_all.py` - nothing is hardcoded. Fixed random seeds (data simulation and
per-run GMM fits) make them reproducible._

- **65 runs simulated, 97,948 total single-cell events** (mean ~1,507
  cells/run).
- **Mean overall automated-gating accuracy across all 65 runs: 98.6%**
  (event-level agreement between GMM-assigned population and ground truth).
- **7.8 of 8 populations detected per run on average** - the two rarest,
  most-overlapping populations (NKT cells, which share CD56 with NK cells;
  double-negative T cells, which sit inside the CD3+ T-cell cloud) are the
  ones occasionally merged into a neighboring cluster in a given run. All 8
  populations are correctly recovered somewhere across the full run set.
- **Sanity check passed**: on `run_000`, where one population is pushed to
  be clearly, artificially separated in marker space, the automated gater
  achieves well above 85% accuracy - confirming the algorithm correctly
  resolves populations when separation is easy, and that the harder scores
  above reflect genuine marker overlap, not an implementation bug.

### Per-population recall (pooled across all 65 runs)

| Population | Cells | Recall |
|---|---|---|
| B cell | 11,599 | 100.0% |
| CD4 T helper | 26,678 | 100.0% |
| Classical monocyte | 13,454 | 100.0% |
| NK cell | 9,506 | 100.0% |
| CD8 Cytotoxic T | 17,163 | 99.9% |
| NKT cell | 7,638 | 95.6% |
| Dendritic cell | 5,970 | 93.2% |
| Double-negative T | 5,940 | 89.9% |

(Exact values in `output/population_summary.csv` and
`output/gating_results_per_run.csv` - these are snapshots; re-running
reproduces them exactly.) The recall pattern is biologically sensible: the
four most phenotypically distinct populations (B cells, CD4/CD8 T cells,
monocytes, NK cells) are essentially perfectly resolved, while NKT cells,
dendritic cells, and double-negative T cells - which by design overlap
partially with a neighboring population in marker space - show modestly
lower but still strong recall. This is reported honestly rather than
rounded up: it is the same trade-off a bench scientist faces when manually
gating boundary populations.

## Figures

- `output/figures/example_gating.png` - CD4 vs CD8 scatter for one run,
  ground truth vs. automated GMM gating side by side.
- `output/figures/accuracy_by_population.png` - pooled recall by
  population.

## Project structure

```
simulate_flow_data.py     # 65 runs x 8 populations x 7-marker panel -> data/raw/*.csv
gating.py                 # GMM-based automated gating + accuracy scoring
analyze_runs.py           # runs gating across all runs, aggregates, figures -> output/
run_all.py                # orchestrates the full pipeline
tests/test_pipeline.py    # pytest checks incl. an injected-separation sanity check
```

## Tech stack

- Python 3
- numpy / pandas for data simulation and wrangling
- scikit-learn (`GaussianMixture`, `StandardScaler`) for automated gating
- matplotlib for figures
- pytest
