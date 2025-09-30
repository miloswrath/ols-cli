use std::path::Path;

use anyhow::{anyhow, ensure, Context, Result};
use csv::StringRecord;
use nalgebra::DMatrix;

use super::report::FeatureScaling;

pub(crate) fn parse_numeric_field(
    record: &StringRecord,
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

pub(crate) fn compute_feature_scaling(
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

pub(crate) fn apply_normalization(data: &mut [Vec<f64>], scaling: &[FeatureScaling]) {
    for row in data.iter_mut() {
        for (j, stat) in scaling.iter().enumerate() {
            row[j] = (row[j] - stat.mean) / stat.std_dev;
        }
    }
}

pub(crate) fn build_design_matrix(features: &[Vec<f64>]) -> DMatrix<f64> {
    let rows = features.len();
    let cols = if rows > 0 { features[0].len() } else { 0 };
    let mut buffer = Vec::with_capacity(rows * (cols + 1));

    for row in features {
        buffer.push(1.0); // intercept
        buffer.extend(row.iter().copied());
    }

    DMatrix::from_row_slice(rows, cols + 1, &buffer)
}
