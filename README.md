# Uncertainty-Quantified Reservoir-Parameter Prediction from Well Logs with Blind-Well Validation

This release provides the code, configuration files, test suite, reproducible SPWLA and forward-modelled datasets, and representative figures for uncertainty-quantified reservoir-parameter prediction from well logs.

## Confidentiality notice

The field well-log data, well identifiers, raw interpretations, trained weights, and depth-wise field predictions are confidential. They are not included in this release and must not be uploaded to a public repository. Field-related code is retained to document the complete methodology, but it can be executed only by authorised users who configure access to the private data locally.

The `outputs/` directory is intentionally excluded because it may contain trained models or intermediate results that could disclose confidential information.

## Repository layout

```text
well_log_uncertainty_inversion_release/
├── configs/                 # Experiment, training, and transfer configurations
├── dataset/                 # Public SPWLA and forward-modelled input data
│   ├── train.csv            # SPWLA dataset
│   └── forward_dataset/     # Forward-modelled well-log CSV files
├── figures/                 # Representative interpretation-result figures
├── src/bnn_inversion/       # Package: data, models, training, uncertainty, results
├── tests/                   # Unit and data-contract tests
├── tools/                   # Experiment orchestration and plotting scripts
├── pyproject.toml           # Dependencies and package metadata
└── README.md                # This document
```

## Installation

Python 3.10 or later is required; Python 3.11 is recommended. Run the following commands from the repository root.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
```

Core dependencies include PyTorch, NumPy, Pandas, scikit-learn, SciPy, PyYAML, Matplotlib, and Joblib. The default configuration uses CUDA. For a CPU-only smoke test, append `--override runtime.device=cpu` to a command.

## Data organisation and processing

The public `dataset/` directory contains only the SPWLA and forward-modelled datasets. The private field dataset is deliberately absent.

| Dataset | Location | Input logs | Prediction targets | Processing convention |
|---|---|---|---|---|
| Field | Authorised private location | GR, CAL, SP, AC, CNL, DEN, RT | PHIF, SW, PERM | Confidential; PERM is modelled in `log10` space |
| SPWLA | `dataset/train.csv` | CALI, DEN, GR, NEU, RDEP, RMED | PHIF, SW, VSH | RDEP and RMED are log-transformed; well-wise split |
| Forward-modelled | `dataset/forward_dataset/*.csv` | Configured well-log curves | PHIF, SW, VSH | Used for idealised generalisation evaluation |

PHIF, SW, and VSH are represented as fractions in `[0, 1]`. Positive resistivity variables are log-transformed during preprocessing. The data audit checks schema, missingness, units, and physical ranges. The preprocessor is fitted on training data only and reused for validation, testing, and transfer evaluation.

The SPWLA, forward, and SPWLA-transfer configurations use the bundled public data through `data.root: dataset`. Field configurations retain their private external data path and therefore require local authorisation and configuration.

Run data audits:

```powershell
.\.venv\Scripts\python.exe -m bnn_inversion.cli audit --config configs/spwla.yaml
.\.venv\Scripts\python.exe -m bnn_inversion.cli audit --config configs/forward.yaml
```

## Training and experiment reproduction

### Single training run

```powershell
.\.venv\Scripts\python.exe -m bnn_inversion.cli train --config configs/spwla.yaml
.\.venv\Scripts\python.exe -m bnn_inversion.cli train --config configs/forward.yaml
```

CPU smoke test:

```powershell
.\.venv\Scripts\python.exe -m bnn_inversion.cli train `
  --config configs/fast.yaml `
  --override runtime.device=cpu `
  --override training.epochs=2
```

### Main experiment matrix

```powershell
.\.venv\Scripts\python.exe -m bnn_inversion.cli run-matrix --config configs/spwla.yaml
```

For a smaller debugging run, use options such as `--methods M1,M5,M9` and `--seeds 0,1`.

### Full workflow and supplementary experiments

```powershell
# Preview the workflow without starting training
.\.venv\Scripts\python.exe tools\run_all_experiments.py --fast --dry-run

# Run supplementary uncertainty ablation, risk identification, and calibration experiments
.\.venv\Scripts\python.exe tools\run_supplementary_experiments.py --dry-run
.\.venv\Scripts\python.exe tools\run_supplementary_experiments.py --sections summary
```

The supplementary experiments compare MC Dropout, Bayesian neural networks, simple fusion, and conservative fusion. They also evaluate precision, recall, and error enrichment for risk flags, together with validation-residual-based conformal calibration.

### Plotting

```powershell
.\.venv\Scripts\python.exe tools\plot_uncertainty_supplementary_results.py
.\.venv\Scripts\python.exe tools\plot_three_dataset_composite_logs.py --scale 1000
```

The field representative-well plotting command is retained for authorised users only:

```powershell
.\.venv\Scripts\python.exe tools\plot_field_suspect_well_composite.py --full-well --compress-missing --well 220
```

## Figures

The `figures/` directory contains representative final interpretation figures:

- `field_representative_well_220.pdf`: confidential-field representative-well result, provided only as a final rendered figure with no raw data.
- `field_fullwell_non220.pdf`: field full-well interpretation using a non-220 well.
- `spwla_fullwell.pdf`: SPWLA full-well interpretation.
- `forward_fullwell.pdf`: forward-modelled full-well interpretation.

All composite interpretation figures use the following legend:

1. `Reference interpretation`
2. `Point prediction`
3. `95% prediction interval`
4. `Model-disagreement interval`
5. `Interval miss`

For PHIF, the displayed interval follows the specified label--prediction combination rule and a fixed visual half-width. PERM, SW, and other targets retain their original model-interval drawing logic. Missing depth sections are compressed in full-well plots.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

The test suite covers data adapters, audits, preprocessing, well-based splits, sequence windows, model behaviour, uncertainty fusion, metrics, experiment orchestration, and plotting-data contracts.
