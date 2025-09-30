use std::collections::HashSet;
use std::fs;
use std::path::Path;

use anyhow::{bail, ensure, Context, Result};
use chrono::{DateTime, Utc};
use csv::ReaderBuilder;

use crate::config::{default_alpha, RunConfig};
use crate::ModelKind;

#[derive(Debug, Clone)]
pub struct RegressionReport {
    model: ModelKind,
    rows: usize,
    features: Vec<String>,
    target: String,
    timestamp: DateTime<Utc>,
    notes: Vec<String>,
}

impl RegressionReport {
    pub fn new(
        model: ModelKind,
        rows: usize,
        features: Vec<String>,
        target: String,
        notes: Vec<String>,
    ) -> Self {
        Self {
            model,
            rows,
            features,
            target,
            timestamp: Utc::now(),
            notes,
        }
    }

    pub fn render(&self) -> String {
        let heading = format!(
            concat!(
                "Model: {}\n",
                "Rows: {}\n",
                "Target: {}\n",
                "Features: {}\n",
                "Generated at: {}\n"
            ),
            self.model,
            self.rows,
            self.target,
            self.features.join(", "),
            self.timestamp
                .to_rfc3339_opts(chrono::SecondsFormat::Secs, true)
        );

        if self.notes.is_empty() {
            heading
        } else {
            format!("{}\n\nNotes:\n- {}", heading, self.notes.join("\n- "))
        }
    }

    pub fn persist(&self, path: &Path) -> Result<()> {
        fs::write(path, self.render())
            .with_context(|| format!("failed to write report to {}", path.display()))
    }
}

pub fn fit_least_squares(config: &RunConfig) -> Result<RegressionReport> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&config.dataset)
        .with_context(|| format!("failed to open dataset {}", config.dataset.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("{}: unable to read CSV header", config.dataset.display()))?
        .iter()
        .map(|h| h.trim().to_string())
        .collect::<HashSet<_>>();

    ensure!(
        headers.contains(&config.target),
        "target column '{}' not found in dataset header",
        config.target
    );

    for feature in &config.features {
        ensure!(
            headers.contains(feature),
            "feature column '{}' not found in dataset header",
            feature
        );
    }

    let mut row_count = 0usize;
    for record in reader.records() {
        record.with_context(|| format!("{}: failed to parse CSV row", config.dataset.display()))?;
        row_count += 1;
    }

    if row_count == 0 {
        bail!(
            "dataset '{}' does not contain any rows",
            config.dataset.display()
        );
    }

    let mut notes =
        vec!["Solver pipeline not implemented yet; see README roadmap for details.".to_string()];

    if config.normalize {
        notes.push("Requested feature normalization (stub).".to_string());
    }

    if config.model.requires_alpha() {
        let alpha = config
            .alpha
            .or_else(|| default_alpha(config.model))
            .expect("alpha must be resolved for models that require it");
        notes.push(format!("Using alpha = {:.4}", alpha));
    }

    Ok(RegressionReport::new(
        config.model,
        row_count,
        config.features.clone(),
        config.target.clone(),
        notes,
    ))
}
