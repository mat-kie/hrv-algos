//! Ectopic beat removal algorithms.

use anyhow::{anyhow, Result};
use nalgebra::DVectorView;
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

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

fn rolling_quantile(signal: &[f64], window_size: usize, quantile: f64) -> Result<Vec<f64>> {
    if !(0.0..=1.0).contains(&quantile) {
        return Err(anyhow!("Quantile must be between 0 and 1"));
    }
    // ensure window size handles even and odd window sizes
    let back_window = window_size / 2;
    let fwd_window = window_size - back_window;

    signal
        .par_iter()
        .enumerate()
        .map(|(idx, _)| {
            let start = idx.saturating_sub(back_window);
            let end = signal.len().min(idx + fwd_window);
            let mut window = signal[start..end].to_vec();
            window.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let quantile_idx = ((window.len() - 1) as f64 * quantile).round() as usize;
            Ok(window[quantile_idx])
        })
        .collect()
}

pub fn rr_artefacts(
    rr: &[f64],
    c1: Option<f64>,
    c2: Option<f64>,
    alpha: Option<f64>,
) -> Result<Vec<(f64, bool)>> {
    if rr.len() < 2 {
        return Err(anyhow!("RR intervals must have at least 2 elements"));
    }
    let c1 = c1.unwrap_or(0.13);
    let c2 = c2.unwrap_or(0.17);
    let alpha = alpha.unwrap_or(5.2);

    let drr = {
        let mut drr: Vec<f64> = rr.windows(2).map(|w| w[1] - w[0]).collect();
        drr.insert(0, drr.iter().copied().sum::<f64>() / drr.len() as f64);
        drr
    };

    // Rolling quantile calculations (q1 and q3)
    let q1 = rolling_quantile(&drr, 91, 0.25)?;
    let q3 = rolling_quantile(&drr, 91, 0.75)?;
    let th1: Vec<f64> = q1
        .par_iter()
        .zip(&q3)
        .map(|(&q1, &q3)| alpha * (q3 - q1) / 2.0)
        .collect();

    // Calculate median RR (mRR)
    let med_rr = rolling_quantile(rr, 11, 0.5)?;
    let mut mrr: Vec<f64> = rr.iter().zip(&med_rr).map(|(&rr, &med)| rr - med).collect();
    for val in &mut mrr {
        if *val < 0.0 {
            *val *= 2.0;
        }
    }

    // Calculate second threshold (th2)
    let q1_mrr = rolling_quantile(&mrr, 91, 0.25)?;
    let q3_mrr = rolling_quantile(&mrr, 91, 0.75)?;
    let th2: Vec<f64> = q1_mrr
        .iter()
        .zip(&q3_mrr)
        .map(|(&q1, &q3)| alpha * (q3 - q1) / 2.0)
        .collect();

    let normalized_mrr: Vec<f64> = mrr.iter().zip(&th2).map(|(&mrr, &th2)| mrr / th2).collect();

    // Decision
    let mean_rr = rr.iter().copied().sum::<f64>() / rr.len() as f64;
    let result = rr
        .par_iter()
        .enumerate()
        .map(|(idx, &rr_val)| {
            let drr_val = drr[idx];
            let nmrr = normalized_mrr[idx];
            let th1_val = th1[idx];
            let th2_val = th2[idx];

            let s11 = drr_val / th1_val;

            let s12 = if idx == 0 || idx == rr.len() - 1 {
                0.0
            } else {
                let ma = drr[idx - 1].max(drr[idx + 1]);
                let mi = drr[idx - 1].min(drr[idx + 1]);
                if drr_val < 0.0 {
                    mi
                } else {
                    ma
                }
            };

            let s22 = if idx >= rr.len() - 2 {
                0.0
            } else {
                let ma = drr[idx + 1].max(drr[idx + 2]);
                let mi = drr[idx + 1].min(drr[idx + 2]);
                if drr_val >= 0.0 {
                    mi
                } else {
                    ma
                }
            };

            let ectopic =
                (s11 > 1.0 && s12 < (-c1 * s11 - c2)) || (s11 < -1.0 && s12 > (-c1 * s11 + c2));

            let long =
                ((s11 > 1.0 && s22 < -1.0) || (nmrr.abs() > 3.0 && rr_val > mean_rr)) && !ectopic;

            let short =
                ((s11 < -1.0 && s22 > 1.0) || (nmrr.abs() > 3.0 && rr_val <= mean_rr)) && !ectopic;

            let missed = long && ((rr_val / 2.0 - med_rr[idx]).abs() < th2_val);

            let extra =
                short && ((rr_val + rr.get(idx + 1).unwrap_or(&0.0) - med_rr[idx]).abs() < th2_val);

            (rr_val, ectopic || long || short || missed || extra)
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use core::f64;

    use rayon::vec;

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
    fn test_rr_outliers() {
        let signal = vec![1000.0, 1100.0, 1.150e3, 1.50e3, 0.5e3, 1.12e3, 1.15e3, 1.16e3, 1.07e3, 1.08e3, 1.09e3];
        let result = rr_artefacts(&signal, None, None, None).unwrap();
        assert_eq!(
            result,
            vec![
                (1000.0, false),
                (1100.0, false),
                (1.150e3, false),
                (1.50e3, true),
                (0.5e3, true),
                (1.12e3, false),
                (1.15e3, false),
                (1.16e3, false),
                (1.07e3, false),
                (1.08e3, false),
                (1.09e3, false)
            ]
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
