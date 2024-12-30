use anyhow::{anyhow, Result};
use nalgebra::{DVector, DVectorView};
use rayon::iter::{IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

pub trait OutlierClassifier {
    fn add_data(&mut self, data: &[f64]) -> Result<()>;
    fn get_data(&self) -> &[f64];
    fn get_classification(&self) -> &[OutlierType];
    fn get_filtered_data(&self) -> Vec<f64> {
        self.get_data()
            .par_iter()
            .zip(self.get_classification())
            .filter_map(
                |(&val, class)| {
                    if class.is_outlier() {
                        None
                    } else {
                        Some(val)
                    }
                },
            )
            .collect()
    }
}

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

pub struct MovingWindowFilter {
    rr_intervals: Vec<f64>,
    rr_classification: Vec<OutlierType>,
    criterion: Box<dyn OutlierCriterion>,
    window_size: usize,
}

impl MovingWindowFilter {
    pub fn new(criterion: Box<dyn OutlierCriterion>, window_size: usize) -> Self {
        Self {
            rr_intervals: Vec::new(),
            rr_classification: Vec::new(),
            criterion,
            window_size: window_size.max(1),
        }
    }

    pub fn update_classification(&mut self) -> Result<()> {
        if self.rr_intervals.len() < self.window_size {
            return Err(anyhow::anyhow!(
                "Window size must be less than the signal length."
            ));
        }
        let half_window = self.window_size / 2;
        let siglen = self.rr_intervals.len();
        self.rr_classification = (0..self.rr_intervals.len())
            .map(|idx| {
                let (window, _window_idx) = if idx < half_window {
                    (&self.rr_intervals[0..self.window_size], idx)
                } else if idx >= siglen - half_window {
                    (
                        &self.rr_intervals[siglen - self.window_size..siglen],
                        self.window_size - (siglen - idx),
                    )
                } else {
                    (
                        &self.rr_intervals[idx - half_window..idx + half_window],
                        half_window,
                    )
                };
                if self.criterion.is_acceptable(self.rr_intervals[idx], window) {
                    OutlierType::None
                } else {
                    OutlierType::Other
                }
            })
            .collect();
        Ok(())
    }
}

impl OutlierClassifier for MovingWindowFilter {
    fn add_data(&mut self, data: &[f64]) -> Result<()> {
        self.rr_intervals.extend_from_slice(data);
        self.update_classification()
    }
    fn get_data(&self) -> &[f64] {
        &self.rr_intervals
    }
    fn get_classification(&self) -> &[OutlierType] {
        &self.rr_classification
    }
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

/// Detects artefacts (ectopic, long, short, missed, extra beats) in an RR interval series
/// using rolling quantiles and threshold-based classification.
///
/// The algorithm is based on the Systole Python package by Legrand and Allen (2022).
/// Link: https://github.com/embodied-computation-group/systole
///
/// # References
///
///  - Legrand, N. & Allen, M., (2022). Systole: A python package for cardiac signal synchrony and analysis. Journal of Open Source Software, 7(69), 3832, https://doi.org/10.21105/joss.03832
pub struct MovingQuantileFilter {
    rr_intervals: Vec<f64>,
    rr_classification: Vec<OutlierType>,
    slope: f64,
    intercept: f64,
    quantile_scale: f64,
    median_window: usize,
    threshold_window: usize,
}

impl MovingQuantileFilter {
    /// Creates a new `MovingQuantileFilter` instance.
    /// The default values for slope, intercept, and quantile scale are 0.13, 0.17, and 5.2, respectively.
    /// The default window sizes for median and threshold calculations are 11 and 91, respectively.
    ///
    /// # Arguments
    ///
    /// * `slope` - The slope value for the threshold calculation. Default is 0.13.
    /// * `intercept` - The intercept value for the threshold calculation. Default is 0.17.
    /// * `quantile_scale` - The scaling factor for the threshold calculation. Default is 5.2.
    ///
    /// # Returns
    ///
    /// A new `MovingQuantileFilter` instance.
    ///
    pub fn new(slope: Option<f64>, intercept: Option<f64>, quantile_scale: Option<f64>) -> Self {
        Self {
            rr_intervals: Vec::new(),
            rr_classification: Vec::new(),
            slope: slope.unwrap_or(0.13),
            intercept: intercept.unwrap_or(0.17),
            quantile_scale: quantile_scale.unwrap_or(5.2),
            median_window: 11,
            threshold_window: 91,
        }
    }

    /// Sets the slope value for the threshold comparison.
    ///
    /// # Arguments
    ///
    /// * `slope` - The slope value for the threshold calculation.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of reclassification.
    pub fn set_slope(&mut self, slope: f64) -> Result<()> {
        self.slope = slope;
        self.rr_classification.clear();
        self.update_classification()
    }

    /// Sets the intercept value for the threshold comparison.
    ///
    /// # Arguments
    ///
    /// * `intercept` - The intercept value for the threshold calculation.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of reclassification.
    pub fn set_intercept(&mut self, intercept: f64) -> Result<()> {
        self.intercept = intercept;
        self.rr_classification.clear();
        self.update_classification()
    }

    /// Sets the quantile scale value for the threshold comparison.
    ///
    /// # Arguments
    ///
    /// * `quantile_scale` - The scaling factor for the threshold calculation.
    ///
    /// # Returns
    ///
    /// A `Result` indicating success or failure of reclassification.
    pub fn set_quantile_scale(&mut self, quantile_scale: f64) -> Result<()> {
        self.quantile_scale = quantile_scale;
        self.rr_classification.clear();
        self.update_classification()
    }

    pub fn get_slope(&self) -> f64 {
        self.slope
    }

    pub fn get_intercept(&self) -> f64 {
        self.intercept
    }
    pub fn get_quantile_scale(&self) -> f64 {
        self.quantile_scale
    }

    fn update_classification(&mut self) -> Result<()> {
        // classification uses a 91 item rolling quantile
        // take 91 last rr, update the last 46 elements classification
        let win_start = self
            .rr_classification
            .len()
            .saturating_sub(self.threshold_window);
        let cutoff = self
            .rr_classification
            .len()
            .saturating_sub(self.threshold_window / 2);
        let added_rr = self
            .rr_intervals
            .len()
            .saturating_sub(self.rr_classification.len());
        let data = &self.rr_intervals[win_start..];
        if data.is_empty() {
            return Ok(());
        }
        let new_class = if data.len() == 1 {
            vec![OutlierType::None]
        } else {
            self.classify_rr_values(data)?
        };

        let mut added_classes = new_class[new_class.len().saturating_sub(added_rr)..].to_vec();
        self.rr_classification.append(&mut added_classes);
        //  update the last 46 elements classification
        for (a, b) in self.rr_classification.iter_mut().skip(cutoff).zip(
            new_class
                .iter()
                .skip(new_class.len().saturating_sub(self.threshold_window / 2)),
        ) {
            *a = *b;
        }
        Ok(())
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
    fn calc_rr_threshold(&self, signal: &[f64]) -> Result<DVector<f64>> {
        let first_quantile: DVector<f64> =
            rolling_quantile(signal, self.threshold_window, 0.25)?.into();
        let third_quantile: DVector<f64> =
            rolling_quantile(signal, self.threshold_window, 0.75)?.into();
        let threshold = (third_quantile - first_quantile) * (self.quantile_scale / 2.0);
        Ok(threshold)
    }

    /// Classifies RR intervals as ectopic, long, short, missed, or extra beats.
    ///
    /// The classification is based on the RR interval series and the calculated thresholds.
    ///
    /// # Arguments
    ///
    /// * `rr` - Slice of RR intervals in milliseconds.
    ///
    /// # Returns
    ///
    /// A vector of `OutlierType` values indicating the presence of outliers.
    fn classify_rr_values(&self, rr: &[f64]) -> Result<Vec<OutlierType>> {
        if rr.len() < 2 {
            return Err(anyhow!("RR intervals must have at least 2 elements"));
        }

        let drr = {
            let drr: DVector<f64> =
                DVector::from_iterator(rr.len() - 1, rr.windows(2).map(|w| w[1] - w[0]));
            let mean_drr = drr.mean();
            drr.insert_row(0, mean_drr)
        };

        // Rolling quantile calculations (q1 and q3)
        let first_threshold = self.calc_rr_threshold(drr.as_slice())?;
        // Calculate median RR (mRR)
        let med_rr = DVector::from(rolling_quantile(rr, self.median_window, 0.5)?);
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
        let second_threshold = self.calc_rr_threshold(med_rr_deviation.as_slice())?;

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

                let ectopic = (s11 > 1.0 && s12 < (-self.slope * s11 - self.intercept))
                    || (s11 < -1.0 && s12 > (-self.slope * s11 + self.intercept));

                if ectopic {
                    return OutlierType::Ectopic;
                }
                let long = ((s11 > 1.0 && s22 < -1.0) || (nmrr.abs() > 3.0 && rr_val > mean_rr))
                    && !ectopic;

                let short = ((s11 < -1.0 && s22 > 1.0) || (nmrr.abs() > 3.0 && rr_val <= mean_rr))
                    && !ectopic;

                let missed = long && ((rr_val / 2.0 - med_rr[idx]).abs() < th2_val);
                if missed {
                    return OutlierType::Missed;
                }
                if long {
                    return OutlierType::Long;
                }
                let extra = short
                    && ((rr_val + rr.get(idx + 1).unwrap_or(&0.0) - med_rr[idx]).abs() < th2_val);
                if extra {
                    return OutlierType::Extra;
                }
                if short {
                    return OutlierType::Short;
                }
                OutlierType::None
            })
            .collect();

        Ok(result)
    }
}

impl OutlierClassifier for MovingQuantileFilter {
    fn add_data(&mut self, data: &[f64]) -> Result<()> {
        self.rr_intervals.extend_from_slice(data);
        self.update_classification()
    }

    fn get_data(&self) -> &[f64] {
        &self.rr_intervals
    }

    fn get_classification(&self) -> &[OutlierType] {
        &self.rr_classification
    }
}

#[cfg(test)]
mod tests {
    use rand::{Rng, SeedableRng};

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
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 3);
        assert!(filter.add_data(&signal).is_ok());
        let classes = filter.get_classification();
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
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 3);
        assert!(filter.add_data(&signal).is_ok());
        let classes = filter.get_classification();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_outliers() {
        let signal = vec![1.0, 1.1, 1.2, 5.0, 0.5, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = ValueRatioCriterion { ratio: 0.5 };
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 5);
        assert!(filter.add_data(&signal).is_ok());
        let classes = filter.get_classification();
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
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());

        assert_eq!(
            filter.get_classification(),
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
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 4);
        assert!(filter.add_data(&signal).is_ok());
        let classes = filter.get_classification();
        assert_eq!(filter.get_data().len(), filter.get_classification().len());
        assert_eq!(signal.len(), filter.get_classification().len());
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_std_dev() {
        let signal = vec![1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9];
        let criterion = StdDevCriterion { ratio: 2.0 };
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 3);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_data().len(), filter.get_classification().len());
        assert_eq!(signal.len(), filter.get_classification().len());
        let classes = filter.get_classification();
        classes.iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_window_filter_with_small_window() {
        let signal = vec![1.0, 1.1, 1.2];
        let criterion = ValueRatioCriterion { ratio: 0.2 };
        let mut filter = MovingWindowFilter::new(Box::new(criterion), 5);
        assert!(filter.add_data(&signal).is_err());
    }
    #[test]
    fn test_linear_interpolation() {
        let window = vec![1.0, 2.0, 3.0];
        let interpolator = LinearInterpolation;
        assert_eq!(interpolator.interpolate(&window, 1).unwrap(), 2.0);
        assert_eq!(interpolator.interpolate(&window, 0).unwrap(), 1.0);
        assert_eq!(interpolator.interpolate(&window, 2).unwrap(), 3.0);
    }

    #[test]
    fn test_linear_interpolation_out_of_bounds() {
        let window = vec![1.0, 1.1, 1.2, 1.3, 1.4];
        let interpolation = LinearInterpolation;
        assert!(interpolation.interpolate(&window, 5).is_err());
    }

    #[test]
    fn test_linear_interpolation_window_too_small() {
        let window = vec![1.0, 1.1];
        let interpolation = LinearInterpolation;
        assert!(interpolation.interpolate(&window, 0).is_err());
    }
    #[test]
    fn test_moving_quantile_filter() {
        let signal = vec![
            1.001e3, 1.012e3, 1.023e3, 1.014e3, 1.005e3, 1.016e3, 1.027e3, 1.018e3, 1.009e3,
            1.02e3, 1.031e3, 1.022e3,
        ];
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }

    #[test]
    fn test_moving_quantile_filter_set_intercept_inf() {
        let signal = vec![
            1.001e3, 1.012e3, 1.023e3, 1.014e3, 1.005e3, 1.016e3, 1.027e3, 1.018e3, 1.009e3,
            1.02e3, 1.031e3, 1.022e3,
        ];
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
        assert_eq!(filter.get_intercept(), 0.17);
        assert!(filter.set_intercept(0.0).is_ok());
        assert_eq!(filter.get_intercept(), 0.0);
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        assert_eq!(filter.get_slope(), 0.13);
        assert!(filter.set_slope(0.0).is_ok());
        assert_eq!(filter.get_slope(), 0.0);
        assert_eq!(filter.get_classification().len(), filter.get_data().len());

        assert_eq!(filter.get_quantile_scale(), 5.2);
        assert!(filter.set_quantile_scale(0.0).is_ok());
        assert_eq!(filter.get_quantile_scale(), 0.0);
        assert_eq!(filter.get_classification().len(), filter.get_data().len());

        filter
            .get_classification()
            .iter()
            .take(filter.get_classification().len() - 1)
            .for_each(|&outlier| {
                assert!(outlier.is_outlier());
            });
    }

    #[test]
    fn test_moving_quantile_filter_add_data() {
        // assure rng is stable
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1000)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
        let mut new_data: Vec<f64> = (0..10)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        new_data[5] = 1500.0;
        assert!(filter.add_data(&new_data).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter
            .get_classification()
            .iter()
            .take(filter.get_classification().len() - 56)
            .for_each(|&outlier| {
                assert!(!outlier.is_outlier());
            });
        assert!(filter.get_classification()[1005].is_outlier());
    }
    #[test]
    fn test_moving_quantile_filter_add_empty_data() {
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&[]).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        assert_eq!(filter.get_classification().len(), 0);
    }
    #[test]
    fn test_moving_quantile_filter_add_single_data() {
        // assure rng is stable
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
        assert!(filter.add_data(&[1000.0]).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        assert_eq!(filter.get_classification().len(), 2);
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
    }
    #[test]
    fn test_moving_quantile_filter_add_data_missed() {
        // assure rng is stable
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1000)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
        let mut new_data: Vec<f64> = (0..10)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        new_data[5] = 2000.0;
        assert!(filter.add_data(&new_data).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter
            .get_classification()
            .iter()
            .take(filter.get_classification().len() - 56)
            .for_each(|&outlier| {
                assert!(!outlier.is_outlier());
            });
        assert!(filter.get_classification()[1005].is_outlier());
        assert!(matches!(
            filter.get_classification()[1005],
            OutlierType::Missed
        ));
        let filtered = filter.get_filtered_data();
        assert!(filtered.iter().all(|&val| (val - 1000.0).abs() <= 20.0));
    }

    #[test]
    fn test_moving_quantile_filter_add_data_extra() {
        // assure rng is stable
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let signal: Vec<f64> = (0..1000)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        let mut filter = MovingQuantileFilter::new(None, None, None);
        assert!(filter.add_data(&signal).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter.get_classification().iter().for_each(|&outlier| {
            assert!(!outlier.is_outlier());
        });
        let mut new_data: Vec<f64> = (0..10)
            .map(|_| 1000.0 + rng.gen_range(-10.0..10.0))
            .collect();
        new_data[5] = 20.0;
        assert!(filter.add_data(&new_data).is_ok());
        assert_eq!(filter.get_classification().len(), filter.get_data().len());
        filter
            .get_classification()
            .iter()
            .take(filter.get_classification().len() - 56)
            .for_each(|&outlier| {
                assert!(!outlier.is_outlier());
            });
        assert!(filter.get_classification()[1005].is_outlier());
        assert!(matches!(
            filter.get_classification()[1005],
            OutlierType::Extra
        ));
        let filtered = filter.get_filtered_data();
        assert!(filtered.iter().all(|&val| (val - 1000.0).abs() <= 20.0));
    }
}
