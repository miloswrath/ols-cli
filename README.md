# OLS CLI

OLS CLI is a work-in-progress command line tool for running ordinary least squares and closely related regression models directly from your terminal. The goal is to streamline exploratory modeling on CSV datasets without needing to open a notebook or spreadsheet.

> ⚠️ The solver is not implemented yet. The current scaffold validates input, inspects the dataset, and produces a structured report placeholder so you can iterate on the UX while the math pipeline is built out.

## Getting Started

### Prerequisites

- Rust 1.75+ (edition 2021)
- CSV dataset with headers that include your target column and one or more feature columns

### Running the CLI

```bash
cargo run -- fit data/example.csv --target price --features bedrooms,bathrooms,area --model ridge --alpha 0.75 --normalize
```

Use `--dry-run` to preview configuration without touching the dataset:

```bash
cargo run -- fit data/example.csv --target price --features bedrooms,bathrooms --dry-run
```

> Tip: `--features` accepts a comma-separated list; quote the argument if your shell treats commas specially.

### Expected Output

The current implementation prints a configuration summary followed by a report stub that details:

- Model variant in use (linear, ridge, lasso)
- Number of rows detected in the dataset
- Feature and target column names
- Notes highlighting missing functionality or inferred defaults

If you pass `--output path/to/report.txt`, the same report is written to disk.

## Command Reference

`fit`
- `DATASET` (positional): Path to a CSV file with headers
- `--target, -t <COLUMN>`: Name of the target column
- `--features, -f <COLUMNS>`: Comma-separated list of feature columns
- `--model <linear|ridge|lasso>`: Regression variant (default `linear`)
- `--alpha <ALPHA>`: Regularization strength for ridge/lasso (optional; defaults applied when omitted)
- `--normalize`: Request feature normalization (no-op for now)
- `--output <PATH>`: Persist the report to disk
- `--dry-run`: Validate inputs and print configuration only

Run `cargo run -- --help` to view global help and `cargo run -- fit --help` for command-specific examples.

## Project Layout

- `src/main.rs` ties the binary entry point to the library crate.
- `src/cli.rs` defines the Clap-powered interface.
- `src/config.rs` converts CLI arguments into a validated runtime configuration and fills default hyperparameters.
- `src/regression.rs` contains the dataset inspection pass and report scaffolding.

The code is organized so that the regression engine can live in `regression.rs` (or a sub-module) while CLI/UX logic remains separate.

## Development Workflow

1. `cargo fmt` to keep formatting consistent.
2. `cargo check` for a fast sanity pass.
3. Add targeted unit or integration tests as solver pieces come online.

## Roadmap

1. **Model metrics and diagnostics**
   - Compute R², adjusted R², RMSE, MAE, and residual summaries.
   - Generate leverage and influence diagnostics for OLS.
2. **Config and experiment management**
   - Support loading hyperparameters from `toml`/`yaml` configs in addition to CLI flags.
   - Emit machine-readable reports (JSON) for downstream tooling.
3. **Prediction mode**
   - Allow loading a trained model artifact and applying it to new data.
   - Surface prediction intervals where applicable.
4. **Testing and quality gates**
   - Add property-based tests for the solver.
   - Introduce integration fixtures with synthetic datasets to guard against regressions.
5. **Developer ergonomics**
   - Add `cargo xtask` commands for sample data generation and benchmark runs.
   - Package the binary via `cargo install` and provide release artifacts.
6. **Cross-validation and model selection**
   - Add k-fold cross-validation to estimate generalization error.
   - Provide grid/random search helpers for tuning regularization strengths.
7. **CLI UX enhancements**
   - Offer interactive prompts for column selection when flags are omitted.
   - Support templated report output (markdown/HTML) for easier sharing.

Contributions and refinements to this roadmap are welcome as requirements solidify.
