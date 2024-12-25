//! This module provides functionality for adding random noise to input data to hide quantization effects.

use anyhow::{anyhow, Result};
use nalgebra::{DVector, DVectorView};
use rand::Rng;

/// Adds random noise to the input data to hide quantization effects.
///
/// # Arguments
///
/// * `data` - A slice of f64 values representing the input data.
/// * `noise_scale` - An optional f64 value representing the scale of the noise to be added.
///                   If not provided, a default value of 1.0 is used.
///
/// # Returns
///
/// A `Result` containing a `Vec<f64>` with the noisy data if successful, or an error if the noise scale is negative.
///
/// # Errors
///
/// This function will return an error if the provided noise scale is negative.
///
/// # Examples
///
/// ```
/// use hrv_algos::preprocessing::noise::hide_quantization;
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = hide_quantization(&data, Some(0.1)).unwrap();
/// data.iter().zip(result.iter()).for_each(|(p, np)| {assert!((p-np).abs() <= 0.1)});
/// ```
pub fn hide_quantization(data: &[f64], noise_scale: Option<f64>) -> Result<Vec<f64>> {
    let scale = noise_scale.unwrap_or(1.0);
    if scale < 0.0 {
        return Err(anyhow!("Noise scale must be positive"));
    }
    let original = DVectorView::from(data);
    let mut rng = rand::thread_rng();
    let noise = DVector::from_fn(original.len(), |_, _| (rng.gen::<f64>() - 0.5) * scale);
    let res = original + noise;
    Ok(res.data.into())
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hide_quantization_with_default_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = hide_quantization(&data, None).unwrap();
        assert_eq!(result.len(), data.len());
        for (p, np) in data.iter().zip(result.iter()) {
            assert!((p - np).abs() <= 1.0);
        }
    }

    #[test]
    fn test_hide_quantization_with_custom_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = hide_quantization(&data, Some(0.1)).unwrap();
        assert_eq!(result.len(), data.len());
        for (p, np) in data.iter().zip(result.iter()) {
            assert!((p - np).abs() <= 0.1);
        }
    }

    #[test]
    fn test_hide_quantization_no_data() {
        let data: Vec<f64> = vec![];
        let result = hide_quantization(&data, None).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_hide_quantization_with_negative_scale() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let scale = -1.0;
        let result = hide_quantization(&data, Some(scale));
        assert!(result.is_err());
    }
}
