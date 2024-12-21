//! Ectopic beat removal algorithms.

use anyhow::{anyhow, Result};
use nalgebra::DVectorView;

/// Criteria for identifying outliers in a signal.
pub trait OutlierCriterion {
    fn is_acceptable(&self, testvalue: f64, window: &[f64]) -> bool;
}

pub struct ValueRatioCriterion {
    pub ratio: f64,
}

impl OutlierCriterion for ValueRatioCriterion {
    fn is_acceptable(&self, testvalue: f64, window: &[f64]) -> bool {
        let data = DVectorView::from(window);
        let mean = data.mean();
        testvalue >= mean * (1.0 - self.ratio) && testvalue <= mean * (1.0 + self.ratio)
    }
}

// Similar implementations for StdDev, SymmetricLimits, etc.

pub struct StdDevCriterion {
    pub ratio: f64,
}

impl OutlierCriterion for StdDevCriterion {
    fn is_acceptable(&self, testvalue: f64, window: &[f64]) -> bool {
        let data = DVectorView::from(window);
        let mean = data.mean();
        let std_dev = data.variance().sqrt();
        testvalue >= mean - std_dev * self.ratio && testvalue <= mean + std_dev * self.ratio
    }
}

pub struct SymmetricLimitsCriterion {
    pub limit: f64,
}

impl OutlierCriterion for SymmetricLimitsCriterion {
    fn is_acceptable(&self, testvalue: f64, window: &[f64]) -> bool {
        let data = DVectorView::from(window);
        let mean = data.mean();
        testvalue >= mean - self.limit && testvalue <= mean + self.limit
    }
}

pub struct LimitsCriterion {
    pub lower: f64,
    pub upper: f64,
}

impl OutlierCriterion for LimitsCriterion {
    fn is_acceptable(&self, testvalue: f64, _window: &[f64]) -> bool {
        testvalue >= self.lower && testvalue <= self.upper
    }
}

/// Trait for implementing interpolation strategies
pub trait Interpolator {
    fn interpolate(&self, window: &[f64], idx: usize) -> Result<f64>;
}

pub enum InterpolationMethod {
    None,
    Linear,
    Custom(Box<dyn Interpolator>),
}

impl Interpolator for InterpolationMethod {
    fn interpolate(&self, window: &[f64], idx: usize) -> Result<f64> {
        match self {
            InterpolationMethod::None => Err(anyhow!("No interpolation method specified")),
            InterpolationMethod::Linear => LinearInterpolation.interpolate(window, idx),
            InterpolationMethod::Custom(interpolator) => interpolator.interpolate(window, idx),
        }
    }
}

/// Linear interpolation between neighbors
pub struct LinearInterpolation;

impl Interpolator for LinearInterpolation {
    fn interpolate(&self, window: &[f64], idx: usize) -> Result<f64> {
        if window.len() < 3 {
            return Err(anyhow!("Window size must be at least 3"));
        }
        if idx >= window.len() {
            return Err(anyhow!("Index out of bounds"));
        }

        match idx {
            0 => Ok(2.0 * window[1] - window[2]),
            i if i == window.len() - 1 => {
                Ok(2.0 * window[window.len() - 2] - window[window.len() - 3])
            }
            i => Ok((window[i - 1] + window[i + 1]) / 2.0),
        }
    }
}

// Update moving window filter
pub fn moving_window_filter(
    signal: &[f64],
    window_size: usize,
    criterion: impl OutlierCriterion,
    interpolation: InterpolationMethod,
) -> Result<Vec<f64>> {
    if signal.len() < window_size {
        return Err(anyhow::anyhow!(
            "Window size must be less than the signal length."
        ));
    }
    let half_window = window_size / 2;
    let siglen = signal.len();
    (0..signal.len())
        .filter_map(|idx| {
            let (window, window_idx) = if idx < half_window {
                (&signal[0..window_size], idx)
            } else if idx >= siglen - half_window {
                (
                    &signal[siglen - window_size..siglen],
                    window_size - (siglen - idx),
                )
            } else {
                (&signal[idx - half_window..idx + half_window], half_window)
            };
            if criterion.is_acceptable(signal[idx], window) {
                Some(Ok(signal[idx]))
            } else if let InterpolationMethod::None = interpolation {
                None
            } else {
                Some(interpolation.interpolate(window, window_idx))
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use core::f64;

    use super::*;

    #[test]
    fn test_symmetric_limits_acceptable() {
        let criterion = SymmetricLimitsCriterion { limit: 0.2 };
        let window = vec![1.0, 1.1, 0.9, 1.0, 1.0];
        assert!(criterion.is_acceptable(1.15, &window));
        assert!(!criterion.is_acceptable(1.3, &window));
    }

    #[test]
    fn test_limits_acceptable() {
        let criterion = LimitsCriterion {
            lower: 0.8,
            upper: 1.2,
        };
        let window = vec![1.0, 1.1, 0.9, 1.0, 1.0];
        assert!(criterion.is_acceptable(1.1, &window));
        assert!(!criterion.is_acceptable(1.3, &window));
    }

    #[test]
    fn test_moving_window_filter_with_symmetric_limits() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = SymmetricLimitsCriterion { limit: 0.2 };
        let filtered_signal =
            moving_window_filter(&signal, 3, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(filtered_signal, signal);
    }

    #[test]
    fn test_moving_window_filter_with_limits() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = LimitsCriterion {
            lower: 1.0,
            upper: 2.0,
        };
        let filtered_signal =
            moving_window_filter(&signal, 3, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(filtered_signal, signal);
    }

    #[test]
    fn test_moving_window_filter_with_outliers() {
        let signal = vec![1.0, 1.1, 1.2, 5.0, 0.5, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.5 };
        let filtered_signal =
            moving_window_filter(&signal, 5, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(
            filtered_signal,
            vec![1.0, 1.1, 1.2, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9]
        );
    }

    #[test]
    fn test_value_ratio_acceptable() {
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let window = vec![1.0, 1.1, 0.9, 1.0, 1.0];
        assert!(criterion.is_acceptable(1.15, &window));
        assert!(!criterion.is_acceptable(1.5, &window));
    }

    #[test]
    fn test_std_dev_acceptable() {
        let criterion = StdDevCriterion { ratio: 2.0 };
        let window = vec![0.9, 1.1, 0.9, 1.1, 1.0];
        assert!(criterion.is_acceptable(1.15, &window));
        assert!(!criterion.is_acceptable(1.5, &window));
    }

    #[test]
    fn test_moving_window_filter() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let filtered_signal =
            moving_window_filter(&signal, 3, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(filtered_signal, signal);
    }

    #[test]
    fn test_moving_window_filter_even() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let filtered_signal =
            moving_window_filter(&signal, 4, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(filtered_signal, signal);
    }

    #[test]
    fn test_moving_window_filter_with_std_dev() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = StdDevCriterion { ratio: 2.0 };
        let filtered_signal =
            moving_window_filter(&signal, 3, criterion, InterpolationMethod::None).unwrap();
        assert_eq!(filtered_signal, signal);
    }

    #[test]
    fn test_moving_window_filter_with_small_window() {
        let signal = vec![1.0, 1.1, 1.2];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let result = moving_window_filter(&signal, 5, criterion, InterpolationMethod::None);
        assert!(result.is_err());
    }

    #[test]
    fn test_moving_window_filter_with_linear_interpolation() {
        let signal = vec![1.0, 1.1, 1.2, 5.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.5 };
        let filtered_signal =
            moving_window_filter(&signal, 5, criterion, InterpolationMethod::Linear).unwrap();
        let truth = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        assert!(truth
            .iter()
            .zip(filtered_signal)
            .map(|(a, b)| { (a - b).abs() < 2.0 * f64::EPSILON })
            .reduce(|a, b| a && b)
            .unwrap());
    }

    #[test]
    fn test_moving_window_filter_with_linear_interpolation_at_edges() {
        let signal = vec![1.0, 5.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 5.0];
        let criterion = ValueRatioCriterion { ratio: 0.5 };
        let filtered_signal =
            moving_window_filter(&signal, 5, criterion, InterpolationMethod::Linear).unwrap();
        let truth = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        assert!(truth
            .iter()
            .zip(filtered_signal)
            .map(|(a, b)| { (a - b).abs() < 2.0 * f64::EPSILON })
            .reduce(|a, b| a && b)
            .unwrap());
    }

    #[test]
    fn test_linear_interpolation() {
        let window = vec![1.0, 2.0, 3.0];
        let interpolator = LinearInterpolation;
        assert_eq!(interpolator.interpolate(&window, 1).unwrap(), 2.0);
    }

    #[test]
    fn test_linear_interpolation_out_of_bounds() {
        let window = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let interpolation = LinearInterpolation;
        assert!(interpolation.interpolate(&window, 5).is_err());
    }
}
