mod cli;
pub mod config;
pub mod regression;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use serde::{Deserialize, Serialize};

use cli::{Cli, Commands, FitArgs};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, ValueEnum, Serialize, Deserialize)]
pub enum ModelKind {
    Linear,
    Ridge,
    Lasso,
}

impl ModelKind {
    pub fn label(self) -> &'static str {
        match self {
            ModelKind::Linear => "ordinary least squares",
            ModelKind::Ridge => "ridge regression",
            ModelKind::Lasso => "lasso regression",
        }
    }

    pub fn requires_alpha(self) -> bool {
        matches!(self, ModelKind::Ridge | ModelKind::Lasso)
    }
}

impl std::fmt::Display for ModelKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label())
    }
}

pub fn run() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Fit(args) => handle_fit(args),
    }
}

fn handle_fit(args: FitArgs) -> Result<()> {
    let config = config::RunConfig::from_fit_args(args).with_defaults();
    config.validate()?;

    println!("--> Configuration\n{}", config.summary());

    if config.dry_run {
        println!("\nDry run requested: skipping solver execution.");
        return Ok(());
    }

    let report = regression::fit_least_squares(&config)?;
    println!("\n--> Report\n{}", report.render());

    if let Some(path) = &config.output {
        report.persist(path)?;
        println!("\nReport written to {}", path.display());
    }

    Ok(())
}
