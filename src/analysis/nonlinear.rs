//! Nonlinear analysis module for HRV algorithms.
//!
//! This module currently only provides functionality to calculate Poincare metrics SD1 and SD2.
//!
//! # Structures
//! - `PoincareAnalysisResult`: Stores results of Poincare plot analysis, including SD1, SD2, and their eigenvectors.
//!
//! # Functions
//! - `calc_poincare_metrics`: Calculates Poincare plot metrics SD1 and SD2 with their eigenvectors.
//!
//! # Usage
//! To use the `calc_poincare_metrics` function, provide a slice of RR intervals in milliseconds. The function returns a `PoincareAnalysisResult` containing SD1, SD2, and their eigenvectors.
//!
//! # Example
//! ```rust
//! use hrv_algos::analysis::nonlinear::calc_poincare_metrics;
//!
//! let data = [1000.0, 1010.0, 1001.0, 1030.0, 1049.0];
//! let poincare = calc_poincare_metrics(&data).unwrap();
//! println!("SD1: {}, SD2: {}", poincare.sd1, poincare.sd2);
//! ```

use anyhow::anyhow;
use anyhow::Result;
use core::f64;
use nalgebra::DMatrix;
use nalgebra::DVectorView;

/// Results of Poincare plot metrics.
/// `PoincareAnalysisResult` structure.
///
/// Stores results of Poincare plot analysis, including SD1, SD2, and their eigenvectors.
#[derive(Clone, Copy, Default)]
pub struct PoincareAnalysisResult {
    pub sd1: f64,
    pub sd1_eigenvector: [f64; 2],
    pub sd2: f64,
    pub sd2_eigenvector: [f64; 2],
}

/// `calc_poincare_metrics` function.
///
/// Calculates Poincare plot metrics SD1 and SD2 with their eigenvectors.
///
/// # Arguments
/// - `data`: A slice of RR intervals in milliseconds.
///
/// # Returns
/// A `PoincareAnalysisResult` containing SD1, SD2, and their eigenvectors.
///
/// # Panics
/// Panics if the input slice has less than 2 elements.
pub fn calc_poincare_metrics(data: &[f64]) -> Result<PoincareAnalysisResult> {
    if data.len() < 2 {
        return Err(anyhow!(
            "Data must contain at least two elements for Poincaré metrics calculation."
        ));
    }

    let rr_points_a = DVectorView::from(&data[0..data.len() - 1]);
    let rr_points_b = DVectorView::from(&data[1..]);

    // Center the data
    let poincare_matrix = {
        let mut centered = DMatrix::from_columns(&[rr_points_a, rr_points_b]);
        let col_means = centered.row_mean();
        for mut row in centered.row_iter_mut() {
            row -= &col_means;
        }
        centered
    };

    // Covariance matrix and eigen decomposition
    let poincare_cov =
        poincare_matrix.transpose() * &poincare_matrix / (poincare_matrix.nrows() as f64 - 1.0);
    let ev = nalgebra::SymmetricEigen::new(poincare_cov);

    // Ensure SD1 < SD2 by convention
    let (sd1, sd2, sd1_vec, sd2_vec) = if ev.eigenvalues[0] < ev.eigenvalues[1] {
        (
            ev.eigenvalues[0].sqrt(),
            ev.eigenvalues[1].sqrt(),
            [ev.eigenvectors.column(0)[0], ev.eigenvectors.column(0)[1]],
            [ev.eigenvectors.column(1)[0], ev.eigenvectors.column(1)[1]],
        )
    } else {
        (
            ev.eigenvalues[1].sqrt(),
            ev.eigenvalues[0].sqrt(),
            [ev.eigenvectors.column(1)[0], ev.eigenvectors.column(1)[1]],
            [ev.eigenvectors.column(0)[0], ev.eigenvectors.column(0)[1]],
        )
    };

    Ok(PoincareAnalysisResult {
        sd1,
        sd1_eigenvector: sd1_vec,
        sd2,
        sd2_eigenvector: sd2_vec,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_poincare_metrics() {
        let data = [1000.0, 1010.0, 1001.0, 1030.0, 1049.0];
        let poincare = calc_poincare_metrics(&data).unwrap();
        assert!(poincare.sd1 < poincare.sd2); // SD1 should always be smaller than SD2
        assert!(poincare.sd1 > 0.0, "SD1 should be positive.");
        assert!(poincare.sd2 > 0.0, "SD2 should be positive.");
        assert!(
            poincare.sd1_eigenvector[0] != 0.0,
            "SD1 eigenvector should not be zero."
        );
        assert!(
            poincare.sd2_eigenvector[0] != 0.0,
            "SD2 eigenvector should not be zero."
        );
    }

    #[test]
    fn test_poincare_metrics_error() {
        let data = [1000.0];
        let result = calc_poincare_metrics(&data);
        assert!(
            result.is_err(),
            "Poincare metrics should fail with less than 2 elements."
        );
    }
}
