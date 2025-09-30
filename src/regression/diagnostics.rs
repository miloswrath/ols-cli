use anyhow::Result;
use nalgebra::{DMatrix, DVector};

/// Summary statistics for residuals produced by a regression fit.
#[derive(Debug, Clone)]
pub struct ResidualSummary {
    pub mean: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub max_abs: f64,
}

/// Top-level leverage and influence diagnostics for ordinary least squares.
#[derive(Debug, Clone)]
pub struct InfluenceDiagnostics {
    pub leverage_threshold: f64,
    pub leverage: Vec<(usize, f64)>,
    pub cooks_threshold: f64,
    pub cooks_distance: Vec<(usize, f64)>,
}

pub(crate) fn summarize_residuals(residuals: &DVector<f64>) -> ResidualSummary {
    let n = residuals.len().max(1) as f64;
    let mean = residuals.iter().sum::<f64>() / n;

    let mut variance_sum = 0.0;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut max_abs = 0.0;

    for value in residuals.iter().copied() {
        variance_sum += {
            let diff = value - mean;
            diff * diff
        };
        if value < min {
            min = value;
        }
        if value > max {
            max = value;
        }
        let abs = value.abs();
        if abs > max_abs {
            max_abs = abs;
        }
    }

    let std_dev = (variance_sum / n).sqrt();

    ResidualSummary {
        mean,
        std_dev,
        min,
        max,
        max_abs,
    }
}

pub(crate) fn leverage_and_influence(
    design: &DMatrix<f64>,
    residuals: &DVector<f64>,
) -> Result<Option<InfluenceDiagnostics>> {
    let n = design.nrows();
    let p = design.ncols();

    if n == 0 || p == 0 || residuals.len() != n {
        return Ok(None);
    }

    if n <= p {
        // Not enough degrees of freedom to evaluate Cook's distance safely.
        return Ok(None);
    }

    let xtx = design.transpose() * design;
    let xtx_inverse = match xtx.cholesky() {
        Some(chol) => chol.inverse(),
        None => return Ok(None),
    };

    let residual_sum_squares = residuals.iter().map(|r| r * r).sum::<f64>();
    let mse = residual_sum_squares / (n as f64 - p as f64);
    if !mse.is_finite() || mse <= 0.0 {
        return Ok(None);
    }

    let mut leverage_values = Vec::with_capacity(n);
    let mut cooks_values = Vec::with_capacity(n);

    for (idx, residual) in residuals.iter().copied().enumerate() {
        let row = design.row(idx).transpose();
        let leverage_matrix = row.transpose() * &xtx_inverse * &row;
        let leverage = leverage_matrix[(0, 0)];
        if !leverage.is_finite() {
            return Ok(None);
        }

        leverage_values.push((idx + 1, leverage));

        if leverage >= 1.0 {
            // Cook's distance is undefined when leverage is 1.
            continue;
        }

        let cooks =
            (residual * residual) / (mse * p as f64) * (leverage / (1.0 - leverage).powi(2));
        if cooks.is_finite() {
            cooks_values.push((idx + 1, cooks));
        }
    }

    leverage_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    cooks_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    leverage_values.truncate(3);
    cooks_values.truncate(3);

    let leverage_threshold = (2.0 * p as f64 / n as f64).min(0.99);
    let cooks_threshold = 4.0 / n as f64;

    Ok(Some(InfluenceDiagnostics {
        leverage_threshold,
        leverage: leverage_values,
        cooks_threshold,
        cooks_distance: cooks_values,
    }))
}
