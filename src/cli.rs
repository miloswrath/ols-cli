use std::path::PathBuf;

use clap::{Args, Parser, Subcommand};

use crate::ModelKind;

/// Command-line interface definition for the OLS CLI.
#[derive(Parser, Debug)]
#[command(
    name = "ols-cli",
    version,
    about = "Run least-squares regressions from the terminal"
)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Commands,
}

#[derive(Subcommand, Debug)]
pub enum Commands {
    /// Fit a regression model against a CSV dataset.
    Fit(FitArgs),
}

#[derive(Args, Debug)]
pub struct FitArgs {
    /// Path to the CSV dataset that contains your features and target columns.
    #[arg(value_name = "DATASET")]
    pub dataset: PathBuf,

    /// Target column to predict.
    #[arg(short, long, value_name = "COLUMN")]
    pub target: String,

    /// Comma-separated list of feature columns to include in the model.
    #[arg(short, long, value_name = "COLUMNS", value_delimiter = ',', num_args = 1..)]
    pub features: Vec<String>,

    /// Regression variant to run (linear, ridge, lasso).
    #[arg(long, value_enum, default_value = "linear")]
    pub model: ModelKind,

    /// Regularization strength for ridge/lasso.
    #[arg(long, value_name = "ALPHA")]
    pub alpha: Option<f64>,

    /// Whether to scale features before fitting.
    #[arg(long)]
    pub normalize: bool,

    /// Write reports to this location instead of stdout.
    #[arg(long, value_name = "PATH")]
    pub output: Option<PathBuf>,

    /// Preview configuration without executing the solver.
    #[arg(long)]
    pub dry_run: bool,
}
