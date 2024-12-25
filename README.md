# hrv-algos

[![Pipeline Status](https://github.com/mat-kie/hrv-algos/actions/workflows/rust.yml/badge.svg)](https://github.com/mat-kie/hrv-algos/actions/workflows/rust.yml)
[![codecov](https://codecov.io/github/mat-kie/hrv-algos/graph/badge.svg?token=2VYIA5LC8M)](https://codecov.io/github/mat-kie/hrv-algos)

Heartrate data processing and analysis algorithms implemented in Rust.

## Overview

`hrv-algos` is a Rust library for Heart Rate Variability (HRV) analysis. It provides a collection of algorithms for preprocessing, time-domain, frequency-domain, and nonlinear HRV metrics.

## Features

- **Preprocessing**: Detect and remove ectopic beats, noise filtering.
- **Time-Domain Analysis**: Compute metrics like RMSSD, SDNN.
- **Nonlinear Analysis**: Detrended Fluctuation Analysis (DFA), Poincaré plot analysis (SD1, SD2).

## Installation

Add `hrv-algos` to your `Cargo.toml`:

```bash
cargo add hrv-algos
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on GitHub.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements

The algorithms implemented in this crate are based on the following python libraries:
- systole: [https://github.com/embodied-computation-group/systole](https://github.com/embodied-computation-group/systole)
- fathon: [https://github.com/stfbnc/fathon](https://github.com/stfbnc/fathon)
