//! Time domain heartrate varaibility data analysis algorithms.

use anyhow::anyhow;
use anyhow::Result;
use nalgebra::DVectorView;

/// Calculates the Root Mean Square of Successive Differences (RMSSD) from a slice of RR intervals.
///
/// RMSSD is a time-domain measure of heart rate variability, which is the square root of the mean
/// of the squares of the successive differences between adjacent RR intervals.
///
/// # Arguments
///
/// * `data` - A slice of f64 values representing RR intervals.
///
/// # Returns
///
/// * `Result<f64>` - The RMSSD value if the calculation is successful, otherwise an error.
///
/// # Errors
///
/// This function will return an error if the input slice contains fewer than two elements.
pub fn calc_rmssd(data: &[f64]) -> Result<f64> {
    if data.len() < 2 {
        Err(anyhow!(
            "Data must contain at least two elements for RMSSD calculation."
        ))
    } else {
        let rr_points_a = DVectorView::from(&data[0..data.len() - 1]);
        let rr_points_b = DVectorView::from(&data[1..]);
        let successive_diffs = rr_points_b - rr_points_a;
        Ok((successive_diffs.dot(&successive_diffs) / (successive_diffs.len() as f64)).sqrt())
    }
}

pub fn calc_sdrr(data: &[f64]) -> Result<f64> {
    if data.len() < 2 {
        Err(anyhow!(
            "Data must contain at least two elements for SDRR calculation."
        ))
    } else {
        let variance = DVectorView::from(data).variance();
        Ok(variance.sqrt())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmssd() {
        let data = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0];
        let rmssd = calc_rmssd(&data).unwrap();
        assert!(rmssd > 0.0, "RMSSD should be positive.");
    }

    #[test]
    fn test_sdrr() {
        let data = [1000.0, 1010.0, 1020.0, 1030.0, 1040.0];
        let sdrr = calc_sdrr(&data).unwrap();
        assert!(sdrr > 0.0, "SDRR should be positive.");
    }

    #[test]
    fn test_rmssd_error() {
        let data = [1000.0];
        let result = calc_rmssd(&data);
        assert!(
            result.is_err(),
            "RMSSD should return an error for a single data point."
        );
    }

    #[test]
    fn test_sdrr_error() {
        let data = [1000.0];
        let result = calc_sdrr(&data);
        assert!(
            result.is_err(),
            "SDRR should return an error for a single data point."
        );
    }
}
