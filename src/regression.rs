use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;

use anyhow::{anyhow, bail, ensure, Context, Result};
use chrono::{DateTime, Utc};
use csv::ReaderBuilder;
use nalgebra::{DMatrix, DVector};

use crate::config::{default_alpha, RunConfig};
use crate::ModelKind;

const LASSO_MAX_ITERS: usize = 1_000;
const LASSO_TOLERANCE: f64 = 1e-6;

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
    model: ModelKind,
    rows: usize,
    features: Vec<String>,
    target: String,
    timestamp: DateTime<Utc>,
    alpha: Option<f64>,
    coefficients: Vec<(String, f64)>,
    metrics: RegressionMetrics,
    scaling: Option<Vec<FeatureScaling>>,
    notes: Vec<String>,
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

        let target = parse_numeric_field(
            &record,
            target_idx,
            &config.dataset,
            &config.target,
            row_idx,
        )?;

        let mut row = Vec::with_capacity(feature_indices.len());
        for (idx, name) in feature_indices.iter().zip(&config.features) {
            let value = parse_numeric_field(&record, *idx, &config.dataset, name, row_idx)?;
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
    let scaling_stats = compute_feature_scaling(&working_rows, &config.features)?;

    let scaling = if config.normalize {
        apply_normalization(&mut working_rows, &scaling_stats);
        Some(scaling_stats.clone())
    } else {
        None
    };

    let design = build_design_matrix(&working_rows);
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
        ModelKind::Linear => solve_linear(&design, &target_vector)?,
        ModelKind::Ridge => solve_ridge(
            &design,
            &target_vector,
            alpha_used.unwrap_or_else(|| default_alpha(ModelKind::Ridge).unwrap()),
        )?,
        ModelKind::Lasso => solve_lasso(
            &design,
            &target_vector,
            alpha_used.unwrap_or_else(|| default_alpha(ModelKind::Lasso).unwrap()),
        )?,
    };

    let predictions = &design * &coefficients;
    let metrics = compute_metrics(&target_vector, &predictions, config.features.len())?;

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
            LASSO_TOLERANCE, LASSO_MAX_ITERS
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

fn parse_numeric_field(
    record: &csv::StringRecord,
    index: usize,
    dataset: &Path,
    column: &str,
    row_idx: usize,
) -> Result<f64> {
    let raw = record.get(index).ok_or_else(|| {
        anyhow!(
            "{}: column '{}' missing at row {}",
            dataset.display(),
            column,
            row_idx + 2
        )
    })?;

    let trimmed = raw.trim();
    ensure!(
        !trimmed.is_empty(),
        "{}: column '{}' empty at row {}",
        dataset.display(),
        column,
        row_idx + 2
    );

    trimmed.parse::<f64>().with_context(|| {
        format!(
            "{}: column '{}' must be numeric at row {} (found '{}')",
            dataset.display(),
            column,
            row_idx + 2,
            raw
        )
    })
}

fn compute_feature_scaling(
    data: &[Vec<f64>],
    feature_names: &[String],
) -> Result<Vec<FeatureScaling>> {
    let row_count = data.len();
    ensure!(row_count > 0, "no feature rows to analyze");

    let feature_count = feature_names.len();
    let mut stats = Vec::with_capacity(feature_count);

    for j in 0..feature_count {
        let mut sum = 0.0;
        for row in data {
            sum += row[j];
        }
        let mean = sum / row_count as f64;

        let mut variance_sum = 0.0;
        for row in data {
            let diff = row[j] - mean;
            variance_sum += diff * diff;
        }

        let variance = variance_sum / row_count as f64;
        let std_dev = variance.sqrt();
        ensure!(
            std_dev.is_finite() && std_dev > 0.0,
            "feature '{}' has zero variance across dataset",
            feature_names[j]
        );

        stats.push(FeatureScaling {
            feature: feature_names[j].clone(),
            mean,
            std_dev,
        });
    }

    Ok(stats)
}

fn apply_normalization(data: &mut [Vec<f64>], scaling: &[FeatureScaling]) {
    for row in data.iter_mut() {
        for (j, stat) in scaling.iter().enumerate() {
            row[j] = (row[j] - stat.mean) / stat.std_dev;
        }
    }
}

fn build_design_matrix(features: &[Vec<f64>]) -> DMatrix<f64> {
    let rows = features.len();
    let cols = if rows > 0 { features[0].len() } else { 0 };
    let mut buffer = Vec::with_capacity(rows * (cols + 1));

    for row in features {
        buffer.push(1.0); // intercept
        buffer.extend(row.iter().copied());
    }

    DMatrix::from_row_slice(rows, cols + 1, &buffer)
}

fn solve_linear(design: &DMatrix<f64>, target: &DVector<f64>) -> Result<DVector<f64>> {
    design
        .clone()
        .qr()
        .solve(target)
        .ok_or_else(|| anyhow!("failed to solve normal equations; design matrix may be singular"))
}

fn solve_ridge(design: &DMatrix<f64>, target: &DVector<f64>, alpha: f64) -> Result<DVector<f64>> {
    ensure!(alpha >= 0.0, "ridge alpha must be non-negative");

    let cols = design.ncols();
    let mut gram = design.transpose() * design;
    for i in 0..cols {
        gram[(i, i)] += alpha;
    }
    // Do not regularize the intercept term.
    gram[(0, 0)] -= alpha;

    let rhs = design.transpose() * target;

    let chol = gram
        .cholesky()
        .ok_or_else(|| anyhow!("ridge system is not positive-definite; try increasing alpha"))?;

    Ok(chol.solve(&rhs))
}

fn solve_lasso(design: &DMatrix<f64>, target: &DVector<f64>, alpha: f64) -> Result<DVector<f64>> {
    ensure!(alpha > 0.0, "lasso alpha must be positive");

    let rows = design.nrows();
    let cols = design.ncols();

    let columns: Vec<DVector<f64>> = (0..cols).map(|j| design.column(j).into_owned()).collect();

    let mut beta = DVector::zeros(cols);
    let mut y_pred = DVector::zeros(rows);

    for iter in 0..LASSO_MAX_ITERS {
        let mut max_delta = 0.0_f64;

        {
            let column0 = &columns[0];
            let residual = target - &y_pred;
            let denom = column0.dot(column0);
            ensure!(denom > 0.0, "intercept column has zero norm");
            let delta = residual.dot(column0) / denom;
            if delta.abs() > 0.0 {
                beta[0] += delta;
                y_pred += column0 * delta;
                max_delta = max_delta.max(delta.abs());
            }
        }

        for j in 1..cols {
            let column = &columns[j];
            let denom = column.dot(column);
            ensure!(denom > 0.0, "feature column {} has zero norm", j);

            let mut residual = target - &y_pred;
            residual += column * beta[j];
            let rho = column.dot(&residual);
            let new_beta = soft_threshold(rho, alpha) / denom;
            let delta: f64 = new_beta - beta[j];

            if delta.abs() > 0.0 {
                beta[j] = new_beta;
                y_pred += column * delta;
                max_delta = max_delta.max(delta.abs());
            }
        }

        if max_delta < LASSO_TOLERANCE {
            return Ok(beta);
        }

        if iter == LASSO_MAX_ITERS - 1 {
            return Err(anyhow!(
                "lasso solver failed to converge within {} iterations",
                LASSO_MAX_ITERS
            ));
        }
    }

    unreachable!()
}

fn compute_metrics(
    actual: &DVector<f64>,
    predicted: &DVector<f64>,
    feature_count: usize,
) -> Result<RegressionMetrics> {
    let n = actual.len();
    ensure!(n > 0, "cannot compute metrics without observations");

    let residuals = actual - predicted;
    let ss_res = residuals.iter().map(|r| r * r).sum::<f64>();

    let mean_actual = actual.iter().sum::<f64>() / n as f64;
    let ss_tot = actual
        .iter()
        .map(|value| {
            let diff = value - mean_actual;
            diff * diff
        })
        .sum::<f64>();

    ensure!(
        ss_tot.is_finite() && ss_tot > 0.0,
        "target column has zero variance; regression is undefined"
    );

    let r2 = 1.0 - (ss_res / ss_tot);
    let rmse = (ss_res / n as f64).sqrt();
    let mae = residuals.iter().map(|r| r.abs()).sum::<f64>() / n as f64;

    let predictors = feature_count + 1; // intercept included
    let adj_r2 = if n > predictors {
        let numerator = (1.0 - r2) * (n as f64 - 1.0);
        let denominator = n as f64 - predictors as f64;
        Some(1.0 - numerator / denominator)
    } else {
        None
    };

    Ok(RegressionMetrics {
        r2,
        adj_r2,
        rmse,
        mae,
    })
}

fn soft_threshold(value: f64, alpha: f64) -> f64 {
    if value > alpha {
        value - alpha
    } else if value < -alpha {
        value + alpha
    } else {
        0.0
    }
}
