/// This module contains various submodules for heart rate variability (HRV) analysis.
///
/// The available submodules are:
///
/// - `dfa`: Implements Detrended Fluctuation Analysis (DFA) for HRV.
/// - `nonlinear`: Contains nonlinear analysis methods for HRV.
/// - `time`: Provides time-domain analysis methods for HRV.
pub mod dfa;
pub mod nonlinear;
pub mod time;
