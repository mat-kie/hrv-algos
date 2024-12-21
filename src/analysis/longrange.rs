//! Nonlinear heartrate variability analysis algorithms.

use core::f64;

use anyhow::anyhow;
use anyhow::Result;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::DVectorView;

pub trait DetrendAlgorithm {
    fn detrend(&self, data: &[f64]) -> Result<Vec<f64>>;
}
pub enum DetrendStrategy {
    Linear,
    Custom(Box<dyn DetrendAlgorithm>),
}

impl DetrendAlgorithm for DetrendStrategy {
    fn detrend(&self, data: &[f64]) -> Result<Vec<f64>> {
        match self {
            DetrendStrategy::Linear => LinearDetrend.detrend(data),
            DetrendStrategy::Custom(detrender) => detrender.detrend(data),
        }
    }
}
pub struct LinearDetrend;

impl DetrendAlgorithm for LinearDetrend {
    fn detrend(&self, data: &[f64]) -> Result<Vec<f64>> {
        if data.len() < 2 {
            return Err(anyhow!(
                "Data must contain at least two elements for detrending."
            ));
        }
        let x: Vec<f64> = (0..data.len()).map(|i| i as f64).collect();
        let ((a, b), _) = linear_fit(&x, data)?;
        let detrended: Vec<f64> = data
            .iter()
            .zip(x.iter())
            .map(|(&y, &i)| y - (a * i + b))
            .collect();

        Ok(detrended)
    }
}

/// DFA calculation result
#[derive(Debug, Clone)]
pub struct DFAnalysis {
    pub alpha: f64,
    pub intercept: f64,
    pub r_squared: f64,
}

impl DFAnalysis {
    pub fn new(
        data: &[f64],
        min_window: usize,
        max_window: usize,
        window_count: usize,
        detrender: DetrendStrategy,
    ) -> Result<Self> {
        if data.len() < 64 {
            return Err(anyhow!("Data length must be at least 64 points"));
        }
        if min_window < 4 {
            return Err(anyhow!("Minimum window size must be at least 4"));
        }
        if max_window >= data.len() / 4 {
            return Err(anyhow!("Maximum window size too large"));
        }
        let data = DVectorView::from(data);
        let mean = data.mean();
        let integrated: Vec<f64> = data
            .iter()
            .scan(0.0, |state, &x| {
                *state += x - mean;
                Some(*state)
            })
            .collect();

        // Generate window sizes (log space)
        let windows = logspace_windows(min_window, max_window, window_count);

        // Calculate fluctuations for each window size
        let mut log_n: Vec<f64> = Vec::with_capacity(windows.len());
        let mut log_f: Vec<f64> = Vec::with_capacity(windows.len());

        for window in windows {
            let (fluctuation, n_segments) = integrated
                .chunks(window)
                .map(|slice| {
                    let detrended = DVector::from(detrender.detrend(slice).unwrap());
                    let sum_sq = detrended.dot(&detrended);
                    sum_sq / window as f64
                })
                .fold((0f64, 0usize), |(fluctuation, n_segments), f| {
                    (fluctuation + f, n_segments + 1)
                });
            if n_segments > 0 {
                let f_n = (fluctuation / n_segments as f64).sqrt();
                log_n.push((window as f64).ln());
                log_f.push(f_n.ln());
            }
        }

        // Linear regression on log-log data
        let ((alpha, intercept), r_squared) = linear_fit(&log_n, &log_f)?;

        Ok(DFAnalysis {
            alpha,
            intercept,
            r_squared,
        })
    }
}

fn logspace_windows(min_window: usize, max_window: usize, window_count: usize) -> Vec<usize> {
    (0..window_count)
        .map(|i| {
            let log_min = (min_window as f64).ln();
            let log_max = (max_window as f64).ln();
            let log_step = (log_max - log_min) / (window_count - 1) as f64;
            (((log_min + i as f64 * log_step).exp()).round() as usize)
                .max(min_window)
                .min(max_window)
        })
        .collect()
}

fn linear_fit(x: &[f64], y: &[f64]) -> Result<((f64, f64), f64)> {
    if x.len() < 2 {
        return Err(anyhow!(
            "Data must contain at least two elements for linear fit."
        ));
    }
    if x.len() != y.len() {
        return Err(anyhow!("X and Y data must have the same length."));
    }
    let prob_matrix = DMatrix::from_columns(&[
        DVector::from_column_slice(x),
        DVector::from_element(x.len(), 1.0),
    ]);
    let y = DVectorView::from(y);
    let result = lstsq::lstsq(&prob_matrix, &y.into(), f64::EPSILON).map_err(|e| anyhow!(e))?;

    let y_mean = y.mean();
    let tss: f64 = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum();
    let r_squared = 1.0 - (result.residuals / tss);

    Ok(((result.solution[0], result.solution[1]), r_squared))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_fit() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let ((slope, intercept), r_sqr) = linear_fit(&x, &y).unwrap();
        assert!(
            (slope - 2.0).abs() < 1e-6,
            "Slope should be approximately 2.0"
        );
        assert!(
            (intercept - 0.0).abs() < 1e-6,
            "Intercept should be approximately 0.0"
        );
        assert!(
            r_sqr > 0.999,
            "R-squared should be close to 1.0 for perfect fit."
        );
    }

    #[test]
    fn test_linear_fit_with_noise() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.1, 3.9, 6.2, 7.8, 10.1];
        let ((slope, intercept), _) = linear_fit(&x, &y).unwrap();
        assert!(
            (slope - 2.0).abs() < 0.2,
            "Slope should be approximately 2.0"
        );
        assert!(
            (intercept - 0.0).abs() < 0.2,
            "Intercept should be approximately 0.0"
        );
    }

    #[test]
    fn test_linear_fit_error() {
        let x = [1.0];
        let y = [2.0];
        let result = linear_fit(&x, &y);
        assert!(
            result.is_err(),
            "Linear fit should fail with less than 2 elements."
        );
    }

    #[test]
    fn test_linear_fit_error_mismatch() {
        let x = [1.0, 2.0];
        let y = [2.0, 3.0, 4.0];
        let result = linear_fit(&x, &y);
        assert!(
            result.is_err(),
            "Linear fit should fail with mismatch between x and y data."
        );
    }
}
