use std::fs;
use std::path::Path;

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use crate::ModelKind;

#[derive(Debug, Clone)]
pub struct FeatureScaling {
    pub feature: String,
    pub mean: f64,
    pub std_dev: f64,
}

#[derive(Debug, Clone)]
pub struct RegressionMetrics {
    pub r2: f64,
    pub adj_r2: Option<f64>,
    pub rmse: f64,
    pub mae: f64,
}

#[derive(Debug, Clone)]
pub struct RegressionReport {
    pub(crate) model: ModelKind,
    pub(crate) rows: usize,
    pub(crate) features: Vec<String>,
    pub(crate) target: String,
    pub(crate) timestamp: DateTime<Utc>,
    pub(crate) alpha: Option<f64>,
    pub(crate) coefficients: Vec<(String, f64)>,
    pub(crate) metrics: RegressionMetrics,
    pub(crate) scaling: Option<Vec<FeatureScaling>>,
    pub(crate) notes: Vec<String>,
}

impl RegressionReport {
    pub fn new(
        model: ModelKind,
        rows: usize,
        features: Vec<String>,
        target: String,
        alpha: Option<f64>,
        coefficients: Vec<(String, f64)>,
        metrics: RegressionMetrics,
        scaling: Option<Vec<FeatureScaling>>,
        notes: Vec<String>,
    ) -> Self {
        Self {
            model,
            rows,
            features,
            target,
            timestamp: Utc::now(),
            alpha,
            coefficients,
            metrics,
            scaling,
            notes,
        }
    }

    pub fn render(&self) -> String {
        let mut lines = Vec::new();
        lines.push(format!("Model: {}", self.model));
        lines.push(format!("Rows: {}", self.rows));
        lines.push(format!("Target: {}", self.target));
        lines.push(format!("Features: {}", self.features.join(", ")));
        lines.push(format!(
            "Generated at: {}",
            self.timestamp
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        ));
        lines.push(format!(
            "Alpha: {}",
            self.alpha
                .map(|a| format!("{:.6}", a))
                .unwrap_or_else(|| "n/a".to_string())
        ));

        lines.push(String::new());
        lines.push("Metrics:".to_string());
        lines.push(format!("  R^2: {:.6}", self.metrics.r2));
        if let Some(adj) = self.metrics.adj_r2 {
            lines.push(format!("  Adjusted R^2: {:.6}", adj));
        }
        lines.push(format!("  RMSE: {:.6}", self.metrics.rmse));
        lines.push(format!("  MAE: {:.6}", self.metrics.mae));

        lines.push(String::new());
        lines.push("Coefficients:".to_string());
        for (name, value) in &self.coefficients {
            lines.push(format!("  {:<15} {:>12.6}", name, value));
        }

        if let Some(scaling) = &self.scaling {
            lines.push(String::new());
            lines.push("Normalization:".to_string());
            for stat in scaling {
                lines.push(format!(
                    "  {:<15} mean={:>12.6} std={:>12.6}",
                    stat.feature, stat.mean, stat.std_dev
                ));
            }
        }

        if !self.notes.is_empty() {
            lines.push(String::new());
            lines.push("Notes:".to_string());
            for note in &self.notes {
                lines.push(format!("  - {}", note));
            }
        }

        lines.join("\n")
    }

    pub fn persist(&self, path: &Path) -> Result<()> {
        fs::write(path, self.render())
            .with_context(|| format!("failed to write report to {}", path.display()))
    }
}
