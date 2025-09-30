# OLS CLI

OLS CLI is a command-line companion for fast experimentation with ordinary least squares (OLS) and closely related linear models. Point the tool at a headered CSV, choose which columns represent the target and the predictors, and it will take care of validation, preprocessing, solving, diagnostics, and reporting in a single run.

## Capabilities

- Fit **linear**, **ridge**, or **lasso** regression without leaving the terminal.
- Optional feature normalization to zero mean and unit variance.
- Automatic handling of intercept terms for every model.
- Safe defaults for regularization hyperparameters with the option to override.
- Rich text report that includes coefficients, fit metrics, residual summary, and leverage/Cook's distance diagnostics for OLS.
- Persist reports to disk for later review.
- Guard rails for common data-quality pitfalls (missing columns, non-numeric cells, zero-variance predictors, constant targets).

## Requirements

- Rust 1.75 or newer.
- CSV dataset with:
  - A header row.
  - Numeric values in the target and feature columns.
  - At least one row of observations.
- All selected feature columns must exhibit variance; columns that stay constant across the dataset are rejected to protect downstream math.

## Installation

```bash
git clone https://github.com/<your-org>/ols-cli
cd ols-cli
cargo build
```

The crate builds as a binary; running through `cargo run` is recommended during development.

## Quick Start

1. Inspect the dataset to decide the target column (the value you want to predict) and the feature columns (independent variables).
2. Run the CLI with the `fit` subcommand:

   ```bash
   cargo run -- fit data/boston_housing/housing_sample.csv \
     --target price \
     --features bedrooms,bathrooms,sqft,age \
     --model linear
   ```

3. Review the configuration echo and the generated regression report. The report is printed to stdout and can optionally be saved to a file using `--output`.

4. Iterate by swapping model types, adjusting alphas, or toggling normalization until you are satisfied with the fit.

### Using the bundled fixtures

The `data/` directory ships with three curated CSVs that exercise different solver behaviors:

- `data/boston_housing/housing_sample.csv`: OLS baseline with multiple predictors.
- `data/energy_efficiency/process_yield.csv`: Stable example where normalization and ridge regularization help with collinearity.
- `data/marketing_mix/ad_performance.csv`: Illustrates sparse coefficients when running lasso.

Try a ridge fit with normalization:

```bash
cargo run -- fit data/energy_efficiency/process_yield.csv \
  --target yield \
  --features temperature,pressure,catalyst,time_on_stream \
  --model ridge \
  --alpha 2.0 \
  --normalize
```

## CLI Usage

Run `cargo run -- --help` to see global help and `cargo run -- fit --help` for subcommand details. The most important flags are documented below.

| Flag | Alias | Required | Default | Description |
| --- | --- | --- | --- | --- |
| `DATASET` | – | ✅ (positional) | – | Path to the input CSV file. Must exist unless `--dry-run` is used. |
| `--target <NAME>` | `-t` | ✅ | – | Column to predict. Must exist in the header. |
| `--features <A,B,C>` | `-f` | ✅ | – | Comma-separated list of feature columns. Duplicates are rejected. |
| `--model <linear|ridge|lasso>` | – | ❌ | `linear` | Regression algorithm. |
| `--alpha <VALUE>` | – | Conditional | Omitted | Regularization strength. Required for ridge and lasso unless defaults apply. |
| `--normalize` | – | ❌ | `false` | Normalize each feature to zero mean / unit variance prior to fitting. |
| `--output <PATH>` | `-o` | ❌ | – | Persist the textual report to disk alongside console output. |
| `--dry-run` | – | ❌ | `false` | Validate arguments and dataset schema without running the solver. |

### How default alphas work

- Ridge regression defaults to `alpha = 1.0` when `--alpha` is omitted.
- Lasso regression defaults to `alpha = 0.5` when `--alpha` is omitted.
- Linear regression ignores alpha entirely.

Specify an explicit value when you want to explore alternative regularization strengths.

### Normalization details

When `--normalize` is provided, each feature is centered and scaled using statistics calculated across the dataset. The transformation is applied before solving, and the mean/std-dev per feature are captured in the report under the “Normalization” section and echoed as a note.

### Dry runs

Use `--dry-run` to confirm flag values and dataset accessibility without executing the math pipeline. This is especially useful in scripts or CI jobs where the dataset might not yet be staged.

```bash
cargo run -- fit data/boston_housing/housing_sample.csv \
  --target price \
  --features bedrooms,bathrooms \
  --dry-run
```

## Example Output

```
--> Configuration
Model: ordinary least squares
Dataset: data/boston_housing/housing_sample.csv
Target: price
Features: bedrooms, bathrooms, sqft, age
Normalization: disabled
alpha = n/a

--> Report
Model: ordinary least squares
Rows: 8
Target: price
Features: bedrooms, bathrooms, sqft, age
Generated at: 2025-09-30T19:24:16Z
Alpha: n/a

Metrics:
  R^2: 0.992554
  Adjusted R^2: 0.982625
  RMSE: 7591.695482
  MAE: 6212.132656

Residuals:
  Mean: -0.000000
  Std Dev: 7591.695482
  Min: -14855.808844
  Max: 8494.369679
  Max |residual|: 14855.808844

Coefficients:
  intercept       115415.545180
  bedrooms        55639.659434
  bathrooms       -96607.250755
  sqft              236.503708
  age             -2851.139797

Influence diagnostics (OLS):
  Leverage threshold: 0.990000
  Row   5 leverage: 1.000000
  Row   4 leverage: 0.808775
  Row   6 leverage: 0.728921
  Cook's distance threshold: 0.500000
  Row   5 Cook's D: 2920.030927
  Row   4 Cook's D: 2.076752
  Row   1 Cook's D: 0.234031
```

### Interpreting the report

- **Metrics**: Core goodness-of-fit metrics (R², adjusted R² when degrees of freedom allow, RMSE, MAE).
- **Residuals**: Summary statistics for prediction errors to spot skewness or outliers.
- **Coefficients**: Intercept followed by each feature in the order provided. Values correspond to the normalized space when `--normalize` is active.
- **Influence diagnostics**: Available for OLS fits when leverage and Cook's distance are well-defined. Top entries surface potential outliers or high-influence points.
- **Notes**: Additional context such as normalization or model-specific solver details. Expect a note when diagnostics cannot be computed (e.g., too few rows or a singular design matrix).

## Validation & Error Handling

The CLI fails fast with actionable messages when:

- The dataset path does not exist (unless `--dry-run` is set).
- Required columns are missing from the header.
- Numeric parsing fails for any selected column.
- A feature column has zero variance.
- The target column is constant, making regression undefined.

These checks are also covered by automated tests built around synthetic CSVs to keep confidence high.

## Development Workflow

- `cargo fmt` – ensure formatting matches repository style.
- `cargo check` – quick type-check before committing.
- `cargo test` – runs unit tests across preprocessing, solver logic, and end-to-end fit scenarios using synthetic data.

The `tests` suite simulates edge cases such as zero-variance predictors, default alpha selection, and lasso shrinkage to catch regressions early.

## Project Layout

- `src/main.rs` – binary entry point.
- `src/cli.rs` – Clap-powered CLI definitions.
- `src/config.rs` – runtime configuration, defaults, and validation.
- `src/regression/preprocess.rs` – CSV parsing, normalization statistics, and design matrix construction.
- `src/regression/solve.rs` – linear, ridge, and lasso solvers plus metrics.
- `src/regression/diagnostics.rs` – residual summaries and influence calculations for OLS.
- `src/regression/report.rs` – report struct and renderer.
- `src/regression/mod.rs` – orchestration glue that wires preprocessing, solving, diagnostics, and reporting together.

## Roadmap Ideas

- Config file support (TOML/JSON/YAML) for repeatable experiments.
- JSON/CSV report emitters alongside the human-readable format.
- Prediction-only mode backed by persisted coefficients.
- Cross-validation helpers for hyperparameter search.
- Additional diagnostics (variance inflation, residual plots, ASCII sparklines).

Feedback and contributions are welcome—open an issue with ideas or questions as you explore the CLI.
