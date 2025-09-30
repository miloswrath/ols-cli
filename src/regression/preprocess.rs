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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use csv::StringRecord;
    use std::path::Path;

    #[test]
    fn parse_numeric_field_handles_whitespace() {
        let record = StringRecord::from(vec!["  42.5  "]);
        let value = parse_numeric_field(&record, 0, Path::new("dataset.csv"), "value", 0)
            .expect("should parse numeric field");
        assert_abs_diff_eq!(value, 42.5);
    }

    #[test]
    fn parse_numeric_field_rejects_empty() {
        let record = StringRecord::from(vec!["   "]);
        let err =
            parse_numeric_field(&record, 0, Path::new("dataset.csv"), "value", 0).unwrap_err();
        assert!(err
            .to_string()
            .contains("dataset.csv: column 'value' empty at row 2"));
    }

    #[test]
    fn parse_numeric_field_rejects_non_numeric() {
        let record = StringRecord::from(vec!["abc"]);
        let err =
            parse_numeric_field(&record, 0, Path::new("dataset.csv"), "value", 4).unwrap_err();
        assert!(err
            .to_string()
            .contains("dataset.csv: column 'value' must be numeric at row 6"));
    }

    #[test]
    fn compute_feature_scaling_returns_stats() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let names = vec!["feature_a".to_string(), "feature_b".to_string()];

        let stats = compute_feature_scaling(&data, &names).expect("scaling should succeed");
        assert_eq!(stats.len(), 2);

        assert_eq!(stats[0].feature, "feature_a");
        assert_abs_diff_eq!(stats[0].mean, 3.0);
        assert_abs_diff_eq!(stats[0].std_dev, (8.0f64 / 3.0).sqrt());

        assert_eq!(stats[1].feature, "feature_b");
        assert_abs_diff_eq!(stats[1].mean, 4.0);
        assert_abs_diff_eq!(stats[1].std_dev, (8.0f64 / 3.0).sqrt());
    }

    #[test]
    fn compute_feature_scaling_rejects_zero_variance() {
        let data = vec![vec![2.0], vec![2.0], vec![2.0]];
        let names = vec!["constant".to_string()];

        let err = compute_feature_scaling(&data, &names).unwrap_err();
        assert!(err
            .to_string()
            .contains("feature 'constant' has zero variance across dataset"));
    }

    #[test]
    fn compute_feature_scaling_requires_rows() {
        let data: Vec<Vec<f64>> = Vec::new();
        let names = vec!["f".to_string()];
        let err = compute_feature_scaling(&data, &names).unwrap_err();
        assert!(err.to_string().contains("no feature rows to analyze"));
    }

    #[test]
    fn apply_normalization_zeroes_means() {
        let mut data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
        let stats = compute_feature_scaling(&data, &["a".into(), "b".into()])
            .expect("scaling should succeed");

        apply_normalization(&mut data, &stats);

        for column in 0..2 {
            let mean = data.iter().map(|row| row[column]).sum::<f64>() / data.len() as f64;
            let variance = data
                .iter()
                .map(|row| {
                    let diff = row[column] - mean;
                    diff * diff
                })
                .sum::<f64>()
                / data.len() as f64;

            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-12);
            assert_abs_diff_eq!(variance, 1.0, epsilon = 1e-12);
        }
    }

    #[test]
    fn build_design_matrix_inserts_intercept() {
        let design = build_design_matrix(&[vec![2.0], vec![4.0]]);
        assert_eq!(design.nrows(), 2);
        assert_eq!(design.ncols(), 2);
        assert_abs_diff_eq!(design[(0, 0)], 1.0);
        assert_abs_diff_eq!(design[(1, 0)], 1.0);
        assert_abs_diff_eq!(design[(0, 1)], 2.0);
        assert_abs_diff_eq!(design[(1, 1)], 4.0);
    }
}
