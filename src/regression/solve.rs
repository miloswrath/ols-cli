use anyhow::{anyhow, ensure, Result};
use nalgebra::{DMatrix, DVector};

use super::report::RegressionMetrics;

pub(crate) const LASSO_MAX_ITERS: usize = 1_000;
pub(crate) const LASSO_TOLERANCE: f64 = 1e-6;

pub(crate) fn solve_linear(design: &DMatrix<f64>, target: &DVector<f64>) -> Result<DVector<f64>> {
    let gram = design.transpose() * design;
    let rhs = design.transpose() * target;

    let chol = gram
        .cholesky()
        .ok_or_else(|| anyhow!("failed to factor design matrix; columns may be collinear"))?;

    Ok(chol.solve(&rhs))
}

pub(crate) fn solve_ridge(
    design: &DMatrix<f64>,
    target: &DVector<f64>,
    alpha: f64,
) -> Result<DVector<f64>> {
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

pub(crate) fn solve_lasso(
    design: &DMatrix<f64>,
    target: &DVector<f64>,
    alpha: f64,
) -> Result<DVector<f64>> {
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

pub(crate) fn compute_metrics(
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

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn sample_design_and_target() -> (DMatrix<f64>, DVector<f64>) {
        let design = DMatrix::from_row_slice(3, 2, &[1.0, 0.0, 1.0, 1.0, 1.0, 2.0]);
        let target = DVector::from_vec(vec![1.0, 3.0, 5.0]);
        (design, target)
    }

    #[test]
    fn solve_linear_recovers_exact_coefficients() {
        let (design, target) = sample_design_and_target();
        let beta = solve_linear(&design, &target).expect("ols solution");

        assert_abs_diff_eq!(beta[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(beta[1], 2.0, epsilon = 1e-10);
    }

    #[test]
    fn solve_ridge_shrinks_feature_weight() {
        let (design, target) = sample_design_and_target();
        let beta = solve_ridge(&design, &target, 1.0).expect("ridge solution");

        assert_abs_diff_eq!(beta[0], 5.0 / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(beta[1], 4.0 / 3.0, epsilon = 1e-10);
    }

    #[test]
    fn solve_lasso_zeroes_coefficients_when_alpha_large() {
        let (design, target) = sample_design_and_target();
        let beta = solve_lasso(&design, &target, 10.0).expect("lasso solution");

        assert_abs_diff_eq!(beta[0], 3.0, epsilon = 1e-6);
        assert_abs_diff_eq!(beta[1], 0.0, epsilon = 1e-6);
    }

    #[test]
    fn compute_metrics_handles_perfect_fit() {
        let actual = DVector::from_vec(vec![1.0, 3.0, 5.0]);
        let predicted = actual.clone();
        let metrics = compute_metrics(&actual, &predicted, 1).expect("metrics");

        assert_abs_diff_eq!(metrics.r2, 1.0, epsilon = 1e-12);
        assert_eq!(metrics.adj_r2, Some(1.0));
        assert_abs_diff_eq!(metrics.rmse, 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(metrics.mae, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn compute_metrics_rejects_zero_variance_targets() {
        let actual = DVector::from_vec(vec![2.0, 2.0, 2.0]);
        let predicted = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        let err = compute_metrics(&actual, &predicted, 1).unwrap_err();
        assert!(err
            .to_string()
            .contains("target column has zero variance; regression is undefined"));
    }

    #[test]
    fn soft_threshold_behaves_as_expected() {
        assert_abs_diff_eq!(super::soft_threshold(5.0, 2.0), 3.0);
        assert_abs_diff_eq!(super::soft_threshold(-5.0, 2.0), -3.0);
        assert_abs_diff_eq!(super::soft_threshold(1.5, 2.0), 0.0);
    }
}
