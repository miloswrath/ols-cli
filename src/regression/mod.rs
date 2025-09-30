mod diagnostics;
mod preprocess;
mod report;
mod solve;

use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, bail, ensure, Context, Result};
use csv::ReaderBuilder;
use nalgebra::DVector;

use crate::config::{default_alpha, RunConfig};
use crate::ModelKind;

pub use diagnostics::{InfluenceDiagnostics, ResidualSummary};
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
    let residual_vector = &target_vector - &predictions;
    let metrics = solve::compute_metrics(&target_vector, &predictions, config.features.len())?;
    let residual_summary = diagnostics::summarize_residuals(&residual_vector);

    let mut notes = Vec::new();

    let diagnostics = if config.model == ModelKind::Linear {
        match diagnostics::leverage_and_influence(&design, &residual_vector)? {
            Some(diag) => Some(diag),
            None => {
                notes.push(
                    "Leverage diagnostics unavailable (singular design or insufficient degrees of freedom)."
                        .to_string(),
                );
                None
            }
        }
    } else {
        notes.push("Influence diagnostics skipped for regularized models.".to_string());
        None
    };

    let mut coefficient_labels = Vec::with_capacity(config.features.len() + 1);
    coefficient_labels.push("intercept".to_string());
    coefficient_labels.extend(config.features.iter().cloned());

    let coefficient_pairs = coefficient_labels
        .into_iter()
        .zip(coefficients.iter().copied())
        .collect();

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
        residual_summary,
        diagnostics,
        notes,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{RunConfig, DEFAULT_LASSO_ALPHA, DEFAULT_RIDGE_ALPHA};
    use approx::assert_abs_diff_eq;
    use std::io::Write;
    use tempfile::NamedTempFile;

    fn write_dataset(contents: &str) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("temp file");
        write!(file, "{}", contents).expect("write temp dataset");
        file.flush().expect("flush temp dataset");
        file
    }

    fn base_config(dataset: &NamedTempFile) -> RunConfig {
        RunConfig {
            dataset: dataset.path().to_path_buf(),
            target: "target".to_string(),
            features: vec!["feature".to_string()],
            model: ModelKind::Linear,
            alpha: None,
            normalize: false,
            output: None,
            dry_run: false,
        }
    }

    #[test]
    fn linear_fit_recovers_expected_coefficients() {
        let dataset = write_dataset("target,feature\n1,0\n3,1\n5,2\n7,3\n");
        let config = base_config(&dataset);
        config.validate().expect("valid config");

        let report = fit_least_squares(&config).expect("fit should succeed");
        assert_eq!(report.rows, 4);
        assert_abs_diff_eq!(report.coefficients[0].1, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(report.coefficients[1].1, 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(report.metrics.r2, 1.0, epsilon = 1e-12);
        assert!(report.diagnostics.is_none());
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("Leverage diagnostics unavailable")));
    }

    #[test]
    fn ridge_fit_uses_default_alpha_and_shrinks_feature() {
        let dataset = write_dataset("target,feature\n1,0\n3,1\n5,2\n7,3\n");
        let mut config = base_config(&dataset);
        config.model = ModelKind::Ridge;
        config = config.with_defaults();
        config.validate().expect("valid ridge config");

        let report = fit_least_squares(&config).expect("ridge fit");
        assert_eq!(report.alpha, Some(DEFAULT_RIDGE_ALPHA));
        assert_abs_diff_eq!(report.coefficients[0].1, 1.5, epsilon = 1e-10);
        assert_abs_diff_eq!(report.coefficients[1].1, 5.0 / 3.0, epsilon = 1e-10);
        assert!(report.notes.iter().any(|note| note.contains("regularized")));
    }

    #[test]
    fn lasso_with_large_alpha_drives_feature_to_zero() {
        let dataset = write_dataset("target,feature\n1,0\n3,1\n5,2\n7,3\n");
        let mut config = base_config(&dataset);
        config.model = ModelKind::Lasso;
        config.alpha = Some(10.0);
        config = config.with_defaults();
        config.validate().expect("valid lasso config");

        let report = fit_least_squares(&config).expect("lasso fit");
        assert_eq!(report.alpha, Some(10.0));
        assert_abs_diff_eq!(report.coefficients[0].1, 4.0, epsilon = 1e-6);
        assert_abs_diff_eq!(report.coefficients[1].1, 0.0, epsilon = 1e-6);
        assert!(report
            .notes
            .iter()
            .any(|note| note.contains("Lasso solved via coordinate descent")));
    }

    #[test]
    fn lasso_uses_default_alpha_when_missing() {
        let dataset = write_dataset("target,feature\n1,0\n3,1\n5,2\n7,3\n");
        let mut config = base_config(&dataset);
        config.model = ModelKind::Lasso;
        config = config.with_defaults();
        config.validate().expect("valid config");

        let report = fit_least_squares(&config).expect("lasso fit");
        assert_eq!(report.alpha, Some(DEFAULT_LASSO_ALPHA));
    }

    #[test]
    fn fit_fails_when_feature_has_no_variance() {
        let dataset = write_dataset("target,feature\n1,2\n3,2\n5,2\n");
        let config = base_config(&dataset);
        config.validate().expect("valid config");

        let err = fit_least_squares(&config).unwrap_err();
        assert!(err
            .to_string()
            .contains("feature 'feature' has zero variance across dataset"));
    }
}
