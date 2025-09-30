mod preprocess;
mod report;
mod solve;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, ensure, Context, Result};
use csv::ReaderBuilder;
use nalgebra::DVector;

use crate::config::{default_alpha, RunConfig};
use crate::ModelKind;

pub use report::{FeatureScaling, RegressionMetrics, RegressionReport};

pub fn fit_least_squares(config: &RunConfig) -> Result<RegressionReport> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(&config.dataset)
        .with_context(|| format!("failed to open dataset {}", config.dataset.display()))?;

    let headers = reader
        .headers()
        .with_context(|| format!("{}: unable to read CSV header", config.dataset.display()))?;

    let mut header_map: HashMap<String, usize> = HashMap::new();
    for (idx, raw) in headers.iter().enumerate() {
        header_map.insert(raw.trim().to_string(), idx);
    }

    ensure!(
        header_map.contains_key(&config.target),
        "target column '{}' not found in dataset header",
        config.target
    );

    let mut seen_features = HashSet::new();
    for feature in &config.features {
        ensure!(
            seen_features.insert(feature),
            "feature column '{}' listed multiple times",
            feature
        );
        ensure!(
            header_map.contains_key(feature),
            "feature column '{}' not found in dataset header",
            feature
        );
    }

    let target_idx = header_map[&config.target];
    let feature_indices: Vec<usize> = config
        .features
        .iter()
        .map(|name| header_map[name])
        .collect();

    let mut feature_rows: Vec<Vec<f64>> = Vec::new();
    let mut target_values: Vec<f64> = Vec::new();

    for (row_idx, record) in reader.records().enumerate() {
        let record = record.with_context(|| {
            format!(
                "{}: failed to parse CSV row {}",
                config.dataset.display(),
                row_idx + 2
            )
        })?;

        let target = preprocess::parse_numeric_field(
            &record,
            target_idx,
            &config.dataset,
            &config.target,
            row_idx,
        )?;

        let mut row = Vec::with_capacity(feature_indices.len());
        for (idx, name) in feature_indices.iter().zip(&config.features) {
            let value =
                preprocess::parse_numeric_field(&record, *idx, &config.dataset, name, row_idx)?;
            row.push(value);
        }

        feature_rows.push(row);
        target_values.push(target);
    }

    if feature_rows.is_empty() {
        bail!(
            "dataset '{}' does not contain any records after header row",
            config.dataset.display()
        );
    }

    let mut working_rows = feature_rows;
    let scaling_stats = preprocess::compute_feature_scaling(&working_rows, &config.features)?;

    let scaling = if config.normalize {
        preprocess::apply_normalization(&mut working_rows, &scaling_stats);
        Some(scaling_stats.clone())
    } else {
        None
    };

    let design = preprocess::build_design_matrix(&working_rows);
    let target_vector = DVector::from_vec(target_values);

    let alpha_used = if config.model.requires_alpha() {
        Some(
            config
                .alpha
                .or_else(|| default_alpha(config.model))
                .ok_or_else(|| anyhow!("alpha must be supplied for model '{}'.", config.model))?,
        )
    } else {
        None
    };

    let coefficients = match config.model {
        ModelKind::Linear => solve::solve_linear(&design, &target_vector)?,
        ModelKind::Ridge => solve::solve_ridge(
            &design,
            &target_vector,
            alpha_used.unwrap_or_else(|| default_alpha(ModelKind::Ridge).unwrap()),
        )?,
        ModelKind::Lasso => solve::solve_lasso(
            &design,
            &target_vector,
            alpha_used.unwrap_or_else(|| default_alpha(ModelKind::Lasso).unwrap()),
        )?,
    };

    let predictions = &design * &coefficients;
    let metrics = solve::compute_metrics(&target_vector, &predictions, config.features.len())?;

    let mut coefficient_labels = Vec::with_capacity(config.features.len() + 1);
    coefficient_labels.push("intercept".to_string());
    coefficient_labels.extend(config.features.iter().cloned());

    let coefficient_pairs = coefficient_labels
        .into_iter()
        .zip(coefficients.iter().copied())
        .collect();

    let mut notes = Vec::new();
    if config.normalize {
        notes.push("Features normalized to zero mean and unit variance.".to_string());
    }
    if config.model == ModelKind::Lasso {
        notes.push(format!(
            "Lasso solved via coordinate descent (tol {}, max {} iters).",
            solve::LASSO_TOLERANCE,
            solve::LASSO_MAX_ITERS
        ));
    }

    Ok(RegressionReport::new(
        config.model,
        working_rows.len(),
        config.features.clone(),
        config.target.clone(),
        alpha_used,
        coefficient_pairs,
        metrics,
        scaling,
        notes,
    ))
}
