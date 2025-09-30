use std::path::PathBuf;

use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

use crate::{cli::FitArgs, ModelKind};

pub const DEFAULT_RIDGE_ALPHA: f64 = 1.0;
pub const DEFAULT_LASSO_ALPHA: f64 = 0.5;

/// Runtime configuration compiled from CLI input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunConfig {
    pub dataset: PathBuf,
    pub target: String,
    pub features: Vec<String>,
    pub model: ModelKind,
    pub alpha: Option<f64>,
    pub normalize: bool,
    pub output: Option<PathBuf>,
    pub dry_run: bool,
}

impl RunConfig {
    pub fn from_fit_args(args: FitArgs) -> Self {
        Self {
            dataset: args.dataset,
            target: args.target,
            features: args.features,
            model: args.model,
            alpha: args.alpha,
            normalize: args.normalize,
            output: args.output,
            dry_run: args.dry_run,
        }
    }

    /// Fill in defaults (e.g. alpha) when the user omitted them.
    pub fn with_defaults(mut self) -> Self {
        if self.alpha.is_none() && self.model.requires_alpha() {
            self.alpha = default_alpha(self.model);
        }
        self
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            !self.target.trim().is_empty(),
            "target column must be provided"
        );
        ensure!(
            !self.features.is_empty(),
            "at least one feature column is required"
        );

        if self.model.requires_alpha() && self.alpha.is_none() {
            bail!(
                "Model '{}' requires --alpha; a default could not be inferred",
                self.model
            );
        }

        if !self.dry_run && !self.dataset.exists() {
            bail!(
                "Dataset '{}' does not exist; use --dry-run to preview without the file",
                self.dataset.display()
            );
        }

        Ok(())
    }

    pub fn summary(&self) -> String {
        let alpha_text = match (self.model.requires_alpha(), self.alpha) {
            (true, Some(alpha)) => format!("alpha = {}", alpha),
            (true, None) => "alpha = <missing>".to_string(),
            _ => "alpha = n/a".to_string(),
        };

        format!(
            concat!(
                "Model: {}\n",
                "Dataset: {}\n",
                "Target: {}\n",
                "Features: {}\n",
                "Normalization: {}\n",
                "{}"
            ),
            self.model,
            self.dataset.display(),
            self.target,
            self.features.join(", "),
            if self.normalize {
                "enabled"
            } else {
                "disabled"
            },
            alpha_text
        )
    }
}

pub fn default_alpha(model: ModelKind) -> Option<f64> {
    match model {
        ModelKind::Linear => None,
        ModelKind::Ridge => Some(DEFAULT_RIDGE_ALPHA),
        ModelKind::Lasso => Some(DEFAULT_LASSO_ALPHA),
    }
}
