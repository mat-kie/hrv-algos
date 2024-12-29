//! This module provides functionality for adding random noise to input data to hide quantization effects.

use rand::distributions::{Distribution, Uniform};

/// Trait to add random noise to an iterator of f64 values to hide quantization effects.
///
/// # Arguments
///
/// * `self` - An iterator of f64 values representing the input data.
/// * `quantization` - An optional f64 value representing the scale of the noise to be added.
///                    If not provided, a default value of 1.0 is used.
///
/// # Returns
///
/// A `DitheringIter` iterator that yields f64 values with added noise.
///
/// # Examples
///
/// ```rust
/// use hrv_algos::preprocessing::noise::ApplyDithering;
/// let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let dithered: Vec<f64> = data.clone().into_iter()
///     .apply_dithering(Some(0.1))
///     .collect();
/// data.iter().zip(dithered.iter()).for_each(|(p, np)| {assert!((p-np).abs() <= 0.1)});
/// ```
pub trait ApplyDithering: Iterator<Item = f64> {
    fn apply_dithering(self, quantization: Option<f64>) -> DitheringIter<Self>
    where
        Self: Sized;
}

impl<I> ApplyDithering for I
where
    I: Iterator<Item = f64>,
{
    fn apply_dithering(self, quantization: Option<f64>) -> DitheringIter<Self> {
        let limit = quantization.unwrap_or(1.0) / 2.0;
        DitheringIter {
            iter: self,
            dist: Uniform::new(-limit, limit),
        }
    }
}

/// Dithering iterator struct that adds random noise to input data.
pub struct DitheringIter<I: Iterator<Item = f64>> {
    iter: I,
    dist: Uniform<f64>,
}

impl<I: Iterator<Item = f64>> Iterator for DitheringIter<I> {
    type Item = f64;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|value| {
            // Generate two uniform random numbers and average them for triangular distribution
            let mut rng = rand::thread_rng();
            let u1 = self.dist.sample(&mut rng);
            let u2 = self.dist.sample(&mut rng);
            value + (u1 + u2) / 2.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dithering_iter() {
        let data = vec![1.0, 2.0, 3.0];
        let gt = data.clone();
        let dithered: Vec<f64> = data.into_iter().apply_dithering(Some(0.1)).collect();
        assert_eq!(dithered.len(), 3);
        // Values should be within ±0.5 of original
        for (orig, dith) in gt.iter().zip(dithered.iter()) {
            assert!((orig - dith).abs() <= 0.05);
        }
    }
    #[test]
    fn test_dithering_iter_default() {
        let data = vec![1.0, 2.0, 3.0];
        let gt = data.clone();
        let dithered: Vec<f64> = data.into_iter().apply_dithering(None).collect();
        assert_eq!(dithered.len(), 3);
        // Values should be within ±0.5 of original
        for (orig, dith) in gt.iter().zip(dithered.iter()) {
            assert!((orig - dith).abs() <= 0.5);
        }
    }
}
