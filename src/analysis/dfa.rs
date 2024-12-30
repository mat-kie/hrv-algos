//! This module provides functionality for Detrended Fluctuation Analysis (DFA) and Unbiased Detrended Fluctuation Analysis (UDFA).
//!
//! DFA is a method for determining the statistical self-affinity of a signal. It is useful for analyzing time series data to detect long-range correlations.
//!
//! The module includes:
//!
//! - `DetrendAlgorithm` trait: Defines the interface for detrending algorithms.
//! - `DetrendStrategy` enum: Provides different strategies for detrending, including linear detrending and custom detrending algorithms.
//! - `LinearDetrend` struct: Implements linear detrending.
//! - `DFAnalysis` struct: Represents the result of DFA or UDFA, including the scaling exponent, intercept, R-squared value, and log-transformed window sizes and fluctuation amplitudes.
//! - `linear_fit` function: Performs linear regression on log-log data to compute the scaling exponent and other statistical properties.
//!
//! # Example
//!
//! ```rust
//! use hrv_algos::analysis::dfa::{DFAnalysis, DetrendStrategy, LinearDetrend};
//! use rand;
//! // your data series
//! let data = (0..128).map(|_| rand::random::<f64>()).collect::<Vec<f64>>();
//! let windows = vec![4, 8, 16, 32];
//! let detrender = DetrendStrategy::Linear;
//!
//! let analysis = DFAnalysis::dfa(&data, &windows, detrender).unwrap();
//! println!("Alpha: {}", analysis.alpha);
//! println!("Intercept: {}", analysis.intercept);
//! println!("R-squared: {}", analysis.r_squared);
//! ```
//!
//! # References
//!
//! - Yuan, Q., Gu, C., Weng, T., Yang, H. (2018). *Unbiased detrended fluctuation analysis: Long-range correlations in very short time series*. Physica A, 505, 179-189.
//! - Bianchi, S. (2020). fathon: A Python package for a fast computation of detrended fluctuation analysis and related algorithms. Journal of Open Source Software, 5(45), 1828. https://doi.org/10.21105/joss.01828
//! - fathon on github: https://github.com/stfbnc/fathon

use anyhow::anyhow;
use anyhow::Result;
use core::f64;
use nalgebra::DMatrix;
use nalgebra::DVector;
use nalgebra::DVectorView;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::slice::ParallelSlice;

/// A trait representing a detrending algorithm for time series data.
///
/// Implementors of this trait should provide a method to remove trends from
/// the given data.
///
/// # Example
///
/// ```
/// use hrv_algos::analysis::dfa::DetrendAlgorithm;
/// use anyhow::Result;
///
/// struct MyDetrendAlgorithm;
///
/// impl DetrendAlgorithm for MyDetrendAlgorithm {
///     fn detrend(&self, data: &[f64]) -> Result<Vec<f64>> {
///         // Implementation goes here
///         Ok(data.to_vec())
///     }
/// }
/// ```
#[cfg_attr(test, mockall::automock)]
pub trait DetrendAlgorithm {
    /// Removes trends from the provided data and returns the detrended data.
    ///
    /// # Parameters
    ///
    /// - `data`: A slice of f64 values representing the time series data to be detrended.
    ///
    /// # Returns
    ///
    /// A `Result` containing a vector of f64 values representing the detrended data on success,
    /// or an error on failure.
    fn detrend(&self, data: &[f64]) -> Result<Vec<f64>>;
}

/// Available detrend strategies for the DFA algorithm.
/// user provided algorithms can be passed via the `Custom` variant.
pub enum DetrendStrategy {
    /// Linear detrending algorithm using least squares regression.
    Linear,
    /// A custom detrending algorithm that implements the `DetrendAlgorithm` trait.
    /// The algorithm is wrapped in a `Box` to allow for dynamic dispatch and must
    /// also implement the `Sync` and `Send` traits to ensure thread safety.
    Custom(Box<dyn DetrendAlgorithm + Sync + Send>),
}

// implement the trait for the enum to dispatch calculation
impl DetrendAlgorithm for DetrendStrategy {
    fn detrend(&self, data: &[f64]) -> Result<Vec<f64>> {
        match self {
            // dispatch to the linear detrend algorithm
            DetrendStrategy::Linear => LinearDetrend.detrend(data),
            // dispatch to the custom detrend algorithm
            DetrendStrategy::Custom(detrender) => detrender.detrend(data),
        }
    }
}

/// A linear detrending algorithm using least squares regression.
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

/// A struct representing Detrended Fluctuation Analysis (DFA) results.
///
/// DFA is a method used to find long-term statistical dependencies in time series data.
///
/// # Fields
///
/// * `alpha` - The scaling exponent, which indicates the presence of long-range correlations.
/// * `intercept` - The intercept of the linear fit in the log-log plot.
/// * `r_squared` - The coefficient of determination, which indicates the goodness of fit.
/// * `log_n` - A vector containing the logarithm of the box sizes used in the analysis.
/// * `log_f` - A vector containing the logarithm of the fluctuation function values corresponding to the box sizes.
#[derive(Debug, Clone)]
pub struct DFAnalysis {
    pub alpha: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub log_n: Vec<f64>,
    pub log_f: Vec<f64>,
}

impl DFAnalysis {
    /// Performs Detrended Fluctuation Analysis (DFA) on the provided data.
    ///
    /// This method computes the scaling exponent (Hurst exponent) and other statistical
    /// properties of a time series using the DFA algorithm. DFA is used to detect long-range
    /// correlations in time series data.
    ///
    /// # Arguments
    ///
    /// - `data`: A slice of `f64` representing the input time series data.
    /// - `windows`: A slice of `usize` specifying the window sizes to use for the fluctuation analysis.
    /// - `detrender`: A `DetrendStrategy` implementation used to remove trends from each segment.
    ///
    /// # Returns
    ///
    /// If successful, this method returns a `DFAnalysis` instance containing:
    /// - `alpha`: The estimated scaling exponent.
    /// - `intercept`: The intercept of the log-log regression.
    /// - `r_squared`: The coefficient of determination for the log-log fit.
    /// - `log_n`: A vector of log-transformed window sizes.
    /// - `log_f`: A vector of log-transformed fluctuation amplitudes.
    ///
    /// # Errors
    ///
    /// This method returns an error if:
    /// - `windows` is empty.
    /// - The length of the input data is less than four times the largest window size.
    /// - The smallest window size is less than 4.
    /// - The `detrender` fails to detrend a segment.
    ///
    /// # Algorithm Overview
    ///
    /// 1. **Integration**:
    ///    - The input data is transformed into a cumulative deviation profile (mean-centered).
    ///
    /// 2. **Segment Detrending**:
    ///    - The profile is divided into overlapping segments, and each segment is detrended using
    ///      the provided `DetrendStrategy`.
    ///
    /// 3. **Variance Calculation**:
    ///    - Variances of the detrended segments are computed.
    ///
    /// 4. **Log-Log Regression**:
    ///    - The log-transformed window sizes (`log_n`) and fluctuations (`log_f`) are
    ///      used to perform a linear regression, yielding the scaling exponent `alpha`.
    pub fn dfa(data: &[f64], windows: &[usize], detrender: DetrendStrategy) -> Result<Self> {
        if windows.is_empty() {
            return Err(anyhow!("Windows must not be empty"));
        }
        let windows = {
            let mut _w = windows.to_owned();
            _w.sort();
            _w
        };
        if data.len() < 4 * *windows.last().unwrap() {
            return Err(anyhow!(
                "Data length must be at least 4x the size of the largest window"
            ));
        }
        if *windows.first().unwrap() < 4 {
            return Err(anyhow!("Minimum window size must be at least 4"));
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

        // Calculate fluctuations for each window size
        let mut log_n: Vec<f64> = Vec::with_capacity(windows.len());
        let mut log_f: Vec<f64> = Vec::with_capacity(windows.len());

        for window in windows {
            let (fluctuation, n_segments) = integrated
                .par_chunks(window)
                .filter_map(|slice| -> Option<Result<f64>> {
                    if slice.len() != window {
                        None
                    } else {
                        match detrender.detrend(slice) {
                            Ok(data) => {
                                let detrended = DVector::from(data);
                                let var = detrended.variance();
                                Some(Ok(var))
                            }
                            Err(e) => Some(Err(e)),
                        }
                    }
                })
                .collect::<Result<Vec<f64>, _>>()?
                .iter()
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
            log_n,
            log_f,
        })
    }

    /// Performs an unbiased Detrended Fluctuation Analysis (DFA) on the provided data.
    /// this algorithm uses overlapping windows.
    ///
    /// This method computes the scaling exponent (Hurst exponent) and other statistical
    /// properties of a time series using the DFA algorithm. DFA is used to detect long-range
    /// correlations in time series data.
    ///
    /// # Arguments
    ///
    /// - `data`: A slice of `f64` representing the input time series data.
    /// - `windows`: A slice of `usize` specifying the window sizes to use for the fluctuation analysis.
    /// - `detrender`: A `DetrendStrategy` implementation used to remove trends from each segment.
    ///
    /// # Returns
    ///
    /// If successful, this method returns a `DFAnalysis` instance containing:
    /// - `alpha`: The estimated scaling exponent.
    /// - `intercept`: The intercept of the log-log regression.
    /// - `r_squared`: The coefficient of determination for the log-log fit.
    /// - `log_n`: A vector of log-transformed window sizes.
    /// - `log_f`: A vector of log-transformed fluctuation amplitudes.
    ///
    /// # Errors
    ///
    /// This method returns an error if:
    /// - `windows` is empty.
    /// - The length of the input data is less than four times the largest window size.
    /// - The smallest window size is less than 4.
    /// - The `detrender` fails to detrend a segment.
    ///
    /// # Algorithm Overview
    ///
    /// 1. **Integration**:
    ///    - The input data is transformed into a cumulative deviation profile (mean-centered).
    ///
    /// 2. **Segment Detrending**:
    ///    - The profile is divided into overlapping segments, and each segment is detrended using
    ///      the provided `DetrendStrategy`.
    ///
    /// 3. **Variance Calculation**:
    ///    - Variances of the detrended segments are computed.
    ///
    /// 4. **Log-Log Regression**:
    ///    - The log-transformed window sizes (`log_n`) and fluctuations (`log_f`) are
    ///      used to perform a linear regression, yielding the scaling exponent `alpha`.
    pub fn udfa(data: &[f64], windows: &[usize], detrender: DetrendStrategy) -> Result<Self> {
        if windows.is_empty() {
            return Err(anyhow!("Windows must not be empty"));
        }
        let windows = {
            let mut _w = windows.to_owned();
            _w.sort();
            _w
        };
        if data.len() < 4 * *windows.last().unwrap() {
            return Err(anyhow!(
                "Data length must be at least 4x the size of the largest window"
            ));
        }
        if *windows.first().unwrap() < 4 {
            return Err(anyhow!("Minimum window size must be at least 4"));
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

        let results: Vec<_> = windows
            .par_iter()
            .map(|&window| -> Result<_> {
                let n_segments = integrated.len() - window + 1;
                if n_segments < 1 {
                    return Ok(None);
                }

                let f: f64 = integrated
                    .as_slice()
                    .windows(window)
                    .map(|slice| -> Result<f64> {
                        let detrended = DVector::from(detrender.detrend(slice)?);

                        let df_sum: f64 = detrended.sum();
                        let df_2_sum: f64 = detrended.iter().map(|&x| x * x).sum();
                        let df_odd_sum: f64 = detrended.iter().step_by(2).sum();
                        let df_even_sum: f64 = detrended.iter().skip(1).step_by(2).sum();
                        let df_shift_sum: f64 =
                            detrended.as_slice().windows(2).map(|w| w[0] * w[1]).sum();

                        let df_neg_mean = (df_odd_sum - df_even_sum) / window as f64;
                        let df_neg_var = df_2_sum / window as f64 - df_neg_mean.powi(2);
                        let df_pos_mean = df_sum / window as f64;
                        let df_pos_var = df_2_sum / window as f64 - df_pos_mean.powi(2);

                        let df_pos_shift = (df_shift_sum
                            + df_pos_mean
                                * (detrended[0] + detrended[window - 1]
                                    - df_pos_mean * (window as f64 + 1.0)))
                            / df_pos_var;

                        let df_neg_shift = (-df_shift_sum
                            + df_neg_mean
                                * (detrended[0]
                                    + detrended[window - 1] * (-1.0_f64).powi(window as i32 + 1)
                                    - df_neg_mean * (window as f64 + 1.0)))
                            / df_neg_var;

                        let rho_a = (window as f64 + df_pos_shift) / (2.0 * window as f64 - 1.0);
                        let rho_b = (window as f64 + df_neg_shift) / (2.0 * window as f64 - 1.0);

                        let rho_a_star = rho_a + (1.0 + 3.0 * rho_a) / (2.0 * window as f64);
                        let rho_b_star = rho_b + (1.0 + 3.0 * rho_b) / (2.0 * window as f64);

                        Ok((rho_a_star + rho_b_star)
                            * (1.0 - 1.0 / (2.0 * window as f64))
                            * df_pos_var)
                    })
                    .collect::<Result<Vec<f64>>>()?
                    .iter()
                    .sum();

                let f_n =
                    (f * ((window - 1) as f64 / window as f64).sqrt() / n_segments as f64).sqrt();
                Ok(Some(((window as f64).ln(), f_n.ln())))
            })
            .collect::<Result<Vec<Option<_>>>>()?
            .into_iter()
            .flatten()
            .collect();

        let (log_n, log_f): (Vec<_>, Vec<_>) =
            results.into_iter().filter(|(_, f)| f.is_finite()).unzip();

        let ((alpha, intercept), r_squared) = linear_fit(&log_n, &log_f)?;

        Ok(DFAnalysis {
            alpha,
            intercept,
            r_squared,
            log_n,
            log_f,
        })
    }
}

/// Performs linear regression on the provided data.
///
/// This function takes two slices of `f64` values representing the x and y coordinates
/// of data points and performs a linear regression to find the best-fit line. It returns
/// the slope and intercept of the line, as well as the coefficient of determination (R-squared value).
///
/// # Arguments
///
/// - `x`: A slice of `f64` values representing the x coordinates of the data points.
/// - `y`: A slice of `f64` values representing the y coordinates of the data points.
///
/// # Returns
///
/// A `Result` containing a tuple with:
/// - A tuple of two `f64` values representing the slope and intercept of the best-fit line.
/// - An `f64` value representing the coefficient of determination (R-squared value).
///
/// # Errors
///
/// This function returns an error if:
/// - The length of `x` is less than 2.
/// - The lengths of `x` and `y` do not match.
///
/// # Example
///
/// ```ignore
/// let x = [1.0, 2.0, 3.0, 4.0, 5.0];
/// let y = [2.0, 4.0, 6.0, 8.0, 10.0];
/// let ((slope, intercept), r_sqr) = linear_fit(&x, &y).unwrap();
/// assert!((slope - 2.0).abs() < 1e-6, "Slope should be approximately 2.0");
/// assert!((intercept - 0.0).abs() < 1e-6, "Intercept should be approximately 0.0");
/// assert!(r_sqr > 0.999, "R-squared should be close to 1.0 for perfect fit.");
/// ```
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
    use rand::{Rng, SeedableRng};

    use super::*;

    fn get_test_data(size: usize) -> Vec<f64> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        (0..size)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect()
    }

    #[test]
    fn invalid_window() {
        let data = get_test_data(4);
        let windows = vec![];
        let detrender = DetrendStrategy::Linear;
        let result = DFAnalysis::dfa(&data, &windows, detrender);
        assert!(result.is_err(), "DFA should fail with empty windows.");
    }

    #[test]
    fn invalid_data_length() {
        let data = get_test_data(3);
        let windows = vec![4];
        let detrender = DetrendStrategy::Linear;
        let result = DFAnalysis::dfa(&data, &windows, detrender);
        assert!(
            result.is_err(),
            "DFA should fail with less than 4x window size data."
        );
    }

    #[test]
    fn detrend_invalid_data() {
        let data = get_test_data(1);
        let result = LinearDetrend.detrend(&data);
        assert!(
            result.is_err(),
            "Detrend should fail with less than 2 elements."
        );
    }

    #[test]
    fn custom_detrend() {
        let mut detrender = MockDetrendAlgorithm::new();
        detrender
            .expect_detrend()
            .times(1..)
            .returning(|data| Ok(data.to_vec()));
        let detrend_strategy = DetrendStrategy::Custom(Box::new(detrender));
        let data = get_test_data(128);
        let windows = vec![4, 5, 6];
        let result = DFAnalysis::dfa(&data, &windows, detrend_strategy);
        assert!(result.is_ok(), "DFA should succeed with custom detrender.");
    }

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
