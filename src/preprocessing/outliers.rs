use anyhow::{anyhow, Result};
use nalgebra::{DVector, DVectorView};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

/// Interface for outlier detection criteria used with moving window filters.
pub trait OutlierCriterion {
    /// Determines if a test value is acceptable based on a window of values.
    ///
    /// # Arguments
    ///
    /// * `testvalue` - The value to test for acceptance.
    /// * `window` - A slice of values to use for comparison.
    ///
    /// # Returns
    ///
    /// A boolean indicating if the test value is acceptable.
    fn is_acceptable(&self, testvalue: f64, window: &[f64]) -> bool;
}

/// Trait for implementing interpolation strategies
pub trait Interpolator {
    /// Interpolates a value based on a window of values.
    ///
    /// # Arguments
    ///
    /// * `window` - A slice of values to use for interpolation.
    /// * `idx` - The index with respect to the window of the value to interpolate.
    fn interpolate(&self, window: &[f64], idx: usize) -> Result<f64>;
}

/// Enum representing different types of outliers.
///
/// Outliers can be classified as ectopic, long, short, missed, or extra beats.
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum OutlierType {
    /// No outlier detected
    None,
    /// Ectopic beat
    Ectopic,
    /// Long beat
    Long,
    /// Short beat
    Short,
    /// Missed beat
    Missed,
    /// Extra beat
    Extra,
    /// Unspecified outlier type
    Other,
}

impl OutlierType {
    pub fn is_outlier(&self) -> bool {
        !matches!(self, OutlierType::None)
    }
}

/// Enum representing different interpolation methods.
pub enum InterpolationMethod {
    /// No interpolation
    None,
    /// Linear interpolation
    Linear,
    /// Custom interpolation strategy
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

/// Criterion for outlier detection based on a ratio of the value to the mean of a window.
///
/// The criterion is acceptable if the value is within the mean +/- ratio * mean.
///
/// # Arguments
///
/// * `ratio` - The ratio to use for comparison.
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

/// Criterion for outlier detection based on a ratio of the value to the standard deviation of a window.
///
/// The criterion is acceptable if the value is within the mean +/- ratio * standard deviation.
///
/// # Arguments
///
/// * `ratio` - The ratio to use for comparison.
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

/// Criterion for outlier detection based on symmetric limits around the mean of a window.
///
/// The criterion is acceptable if the value is within the mean +/- limit.
///
/// # Arguments
///
/// * `limit` - The limit to use for comparison.
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

/// Criterion for outlier detection based on upper and lower limits.
///
/// The criterion is acceptable if the value is within the lower and upper limits.
///
/// # Arguments
///
/// * `lower` - The lower limit.
/// * `upper` - The upper limit.
pub struct LimitsCriterion {
    pub lower: f64,
    pub upper: f64,
}

impl OutlierCriterion for LimitsCriterion {
    fn is_acceptable(&self, testvalue: f64, _window: &[f64]) -> bool {
        testvalue >= self.lower && testvalue <= self.upper
    }
}

/// Linear interpolation between neighbors
///
/// The interpolated value is based on the linear interpolation of the two neighbors.
/// If the index is at the beginning or end of the window, the value is extrapolated.
///
/// # Examples
///
/// ```
/// use hrv_algos::preprocessing::outliers::LinearInterpolation;
/// use hrv_algos::preprocessing::outliers::Interpolator;
/// let window = vec![1.0, 2.0, 5.0];
/// let interpolator = LinearInterpolation;
/// // the interpolated value is the average of the neighbors
/// assert_eq!(interpolator.interpolate(&window, 1).unwrap(), 3.0);
/// //rhe extrapolated value is 2.0 - (5.0 - 2.0) = -1.0
/// assert_eq!(interpolator.interpolate(&window, 0).unwrap(), -1.0);
/// // the extrapolated value is 2.0 + (2.0 - 1.0) = 3.0
/// assert_eq!(interpolator.interpolate(&window, 2).unwrap(), 3.0);
/// ```
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

/// Moving window filter for outlier detection.
///
/// The filter processes a signal using a moving window and an outlier detection criterion.
///
/// # Arguments
///
/// * `signal` - The input signal to process.
/// * `window_size` - The size of the moving window.
/// * `criterion` - The outlier detection criterion.
///
/// # Returns
///
/// A vector of `OutlierType` values indicating the presence of outliers.
///
/// # Errors
///
/// Returns an error if the window size is greater than the signal length.
///
/// # Examples
///
/// ```
/// use hrv_algos::preprocessing::outliers::{moving_window_filter, ValueRatioCriterion};
/// let signal = vec![1.0, 1.1, 1.2, 5.0, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
/// let criterion = ValueRatioCriterion { ratio: 0.5 };
/// let filtered_signal = moving_window_filter(&signal, 5, criterion).unwrap();
/// assert!(filtered_signal[3].is_outlier());
/// ```
pub fn moving_window_filter(
    signal: &[f64],
    window_size: usize,
    criterion: impl OutlierCriterion,
) -> Result<Vec<OutlierType>> {
    if signal.len() < window_size {
        return Err(anyhow::anyhow!(
            "Window size must be less than the signal length."
        ));
    }
    let half_window = window_size / 2;
    let siglen = signal.len();
    Ok((0..signal.len())
        .map(|idx| {
            let (window, _window_idx) = if idx < half_window {
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
                OutlierType::None
            } else {
                OutlierType::Other
            }
        })
        .collect())
}

/// Computes a rolling quantile over a 1D time series.
///
/// # Arguments
///
/// * `signal` - Slice of input data to process.
/// * `window_size` - Size of the rolling window. Considers both sides of current index.
/// * `quantile` - Desired quantile in [0.0, 1.0].
///
/// # Returns
///
/// A vector where each element is the quantile value in the local window around that index.
/// Returns an error if `quantile` is not in the range 0.0..=1.0.
///
/// # Errors
///
/// If `quantile` is not between 0.0 and 1.0, returns an error.
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

/// Calculates the threshold for RR interval artefact detection.
///
/// The threshold is calculated as the difference between the 75th and 25th percentiles
/// of the RR interval series.
///
/// # Arguments
///
/// * `signal` - Slice of RR intervals in milliseconds.
/// * `quantile_scale` - Scaling factor for threshold calculations.
fn calc_rr_threshold(signal: &[f64], quantile_scale: f64) -> Result<DVector<f64>> {
    let first_quantile: DVector<f64> = rolling_quantile(signal, 91, 0.25)?.into();
    let third_quantile: DVector<f64> = rolling_quantile(signal, 91, 0.75)?.into();
    let threshold = (third_quantile - first_quantile) * (quantile_scale / 2.0);
    Ok(threshold)
}

/// Detects artefacts (ectopic, long, short, missed, extra beats) in an RR interval series
/// using rolling quantiles and threshold-based classification.
///
/// The algorithm is based on the Systole Python package by Legrand and Allen (2022).
/// Link: https://github.com/embodied-computation-group/systole
///
/// # Arguments
///
/// * `rr` - Slice of consecutive RR intervals (in milliseconds).
/// * `slope` - Optional slope control parameter for threshold lines.
/// * `intersect` - Optional intercept control parameter for threshold lines.
/// * `quantile_scale` - Optional scaling factor for threshold calculations.
///
/// # Returns
///
/// A `Result` containing a vector of tuples `(rr_value, is_artefact)` for each RR interval.
///
/// # Errors
///
/// Returns an error if `rr` has fewer than 2 elements.
///
/// # Example
///
/// ```
/// use hrv_algos::preprocessing::outliers::classify_rr_values;
/// let rr = vec![800.0, 805.0, 790.0];
/// let result = classify_rr_values(&rr, None, None, None).unwrap();
/// for (val, is_artefact) in rr.iter().zip(&result) {
///     println!("{} -> {:?}", val, is_artefact);
/// }
/// ```
///
/// # References
///
///  - Legrand, N. & Allen, M., (2022). Systole: A python package for cardiac signal synchrony and analysis. Journal of Open Source Software, 7(69), 3832, https://doi.org/10.21105/joss.03832
pub fn classify_rr_values(
    rr: &[f64],
    slope: Option<f64>,
    intercept: Option<f64>,
    quantile_scale: Option<f64>,
) -> Result<Vec<OutlierType>> {
    if rr.len() < 2 {
        return Err(anyhow!("RR intervals must have at least 2 elements"));
    }
    let slope = slope.unwrap_or(0.13);
    let intersect = intercept.unwrap_or(0.17);
    let quantile_scale = quantile_scale.unwrap_or(5.2);

    let drr = {
        let drr: DVector<f64> =
            DVector::from_iterator(rr.len() - 1, rr.windows(2).map(|w| w[1] - w[0]));
        let mean_drr = drr.mean();
        drr.insert_row(0, mean_drr)
    };

    // Rolling quantile calculations (q1 and q3)
    let first_threshold = calc_rr_threshold(drr.as_slice(), quantile_scale)?;
    // Calculate median RR (mRR)
    let med_rr = DVector::from(rolling_quantile(rr, 11, 0.5)?);
    let med_rr_deviation: DVector<f64> = DVector::from_iterator(
        rr.len(),
        rr.iter().zip(med_rr.iter()).map(|(&rr, &med)| {
            let val = rr - med;
            if val < 0.0 {
                val * 2.0
            } else {
                val
            }
        }),
    );

    // Calculate second threshold (th2)
    let second_threshold = calc_rr_threshold(med_rr_deviation.as_slice(), quantile_scale)?;

    let normalized_mrr: DVector<f64> = med_rr_deviation.component_div(&second_threshold);

    // Decision
    let mean_rr = rr.iter().copied().sum::<f64>() / rr.len() as f64;
    let result = rr
        .par_iter()
        .enumerate()
        .map(|(idx, &rr_val)| {
            let drr_val = drr[idx];
            let nmrr = normalized_mrr[idx];
            let th1_val = first_threshold[idx];
            let th2_val = second_threshold[idx];

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

            let ectopic = (s11 > 1.0 && s12 < (-slope * s11 - intersect))
                || (s11 < -1.0 && s12 > (-slope * s11 + intersect));

            if ectopic {
                return OutlierType::Ectopic;
            }
            let long =
                ((s11 > 1.0 && s22 < -1.0) || (nmrr.abs() > 3.0 && rr_val > mean_rr)) && !ectopic;
            if long {
                return OutlierType::Long;
            }
            let short =
                ((s11 < -1.0 && s22 > 1.0) || (nmrr.abs() > 3.0 && rr_val <= mean_rr)) && !ectopic;
            if short {
                return OutlierType::Short;
            }
            let missed = long && ((rr_val / 2.0 - med_rr[idx]).abs() < th2_val);
            if missed {
                return OutlierType::Missed;
            }
            let extra =
                short && ((rr_val + rr.get(idx + 1).unwrap_or(&0.0) - med_rr[idx]).abs() < th2_val);
            if extra {
                return OutlierType::Extra;
            }
            OutlierType::None
        })
        .collect();

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_interpolation_method_enum() {
        let window = vec![1.0, 2.0, 3.0];
        let interpolator = InterpolationMethod::None;
        assert!(interpolator.interpolate(&window, 1).is_err());
        let interpolator = InterpolationMethod::Linear;
        assert_eq!(interpolator.interpolate(&window, 1).unwrap(), 2.0);
        let interpolator = InterpolationMethod::Custom(Box::new(LinearInterpolation));
        assert_eq!(interpolator.interpolate(&window, 1).unwrap(), 2.0);
    }

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
        let classes = moving_window_filter(&signal, 3, criterion).unwrap();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_limits() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = LimitsCriterion {
            lower: 1.0,
            upper: 2.0,
        };
        let classes = moving_window_filter(&signal, 3, criterion).unwrap();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_outliers() {
        let signal = vec![1.0, 1.1, 1.2, 5.0, 0.5, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.5 };
        let classes = moving_window_filter(&signal, 5, criterion).unwrap();
        for (idx, class) in classes.iter().enumerate() {
            if idx != 3 && idx != 4 {
                assert!(!class.is_outlier());
            } else {
                assert!(class.is_outlier());
            }
        }
    }

    #[test]
    fn test_rr_outliers() {
        let signal = vec![
            1000.0, 1100.0, 1.150e3, 1.50e3, 0.5e3, 1.12e3, 1.15e3, 1.16e3, 1.07e3, 1.08e3, 1.09e3,
        ];
        let result = classify_rr_values(&signal, None, None, None).unwrap();
        assert_eq!(
            result,
            vec![
                OutlierType::None,
                OutlierType::None,
                OutlierType::None,
                OutlierType::Long,
                OutlierType::Ectopic,
                OutlierType::None,
                OutlierType::None,
                OutlierType::None,
                OutlierType::None,
                OutlierType::None,
                OutlierType::None,
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
    fn test_moving_window_filter_even() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let classes = moving_window_filter(&signal, 4, criterion).unwrap();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_std_dev() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = StdDevCriterion { ratio: 2.0 };
        let classes = moving_window_filter(&signal, 3, criterion).unwrap();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_small_window() {
        let signal = vec![1.0, 1.1, 1.2];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let result = moving_window_filter(&signal, 5, criterion);
        assert!(result.is_err());
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
