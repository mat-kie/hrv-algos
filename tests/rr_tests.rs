use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

fn read_rr_intervals(file_path: &str) -> io::Result<Vec<f64>> {
    let path = Path::new(file_path);
    let file = File::open(path)?;
    let reader = io::BufReader::new(file);

    let mut rr_intervals = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if let Ok(value) = line.trim().parse::<f64>() {
            rr_intervals.push(value * 1e3);
        }
    }
    Ok(rr_intervals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hrv_algos::analysis::dfa::{DFAnalysis, DetrendStrategy};

    #[test]
    fn test_read_rr_intervals() {
        let rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        assert!(!rr_intervals.is_empty(), "RR intervals should not be empty");
    }

    #[test]
    fn test_rr_intervals_length() {
        let rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        assert!(
            rr_intervals.len() > 120,
            "Expected more than 120 RR intervals"
        );
    }

    #[test]
    fn test_rr_intervals_values() {
        let rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        for &interval in &rr_intervals {
            assert!(interval > 0.0, "RR interval should be positive");
        }
    }

    #[test]
    fn test_dfa_analysis() {
        let rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        let windows = (4..17).collect::<Vec<_>>();
        let dfa_analysis = DFAnalysis::dfa(&rr_intervals[..120], &windows, DetrendStrategy::Linear)
            .expect("Failed to perform DFA analysis");
        println!("{:?}", dfa_analysis);
        assert!(
            dfa_analysis.alpha > 0.0,
            "Expected DFA alpha to be positive"
        );
        assert!(
            dfa_analysis.intercept > 0.0,
            "Expected DFA intercept to be positive"
        );
        assert!(
            dfa_analysis.r_squared > 0.0,
            "Expected DFA R-squared to be positive"
        );
    }

    #[test]
    fn test_udfa_analysis() {
        let rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        let windows = (4..17).collect::<Vec<_>>();
        let dfa_analysis =
            DFAnalysis::udfa(&rr_intervals[..120], &windows, DetrendStrategy::Linear)
                .expect("Failed to perform DFA analysis");
        println!("{:?}", dfa_analysis);
        assert!(
            dfa_analysis.alpha > 0.0,
            "Expected DFA alpha to be positive"
        );
        assert!(
            dfa_analysis.intercept > 0.0,
            "Expected DFA intercept to be positive"
        );
        assert!(
            dfa_analysis.r_squared > 0.0,
            "Expected DFA R-squared to be positive"
        );
    }

    #[test]
    fn test_udfa_analysis_rep() {
        let mut rr_intervals = read_rr_intervals(&format!(
            "{}/tests/resource/Y1.txt",
            env!("CARGO_MANIFEST_DIR")
        ))
        .expect("Failed to read RR intervals");
        rr_intervals[0] = 1000.0;
        rr_intervals[1] = 1000.0;
        rr_intervals[2] = 1000.0;
        rr_intervals[3] = 1000.0;
        rr_intervals[4] = 1000.0;
        rr_intervals[5] = 1000.0;
        rr_intervals[6] = 1000.0;
        rr_intervals[7] = 1000.0;
        rr_intervals[9] = 1000.0;
        rr_intervals[10] = 1000.0;
        rr_intervals[11] = 1000.0;
        rr_intervals[12] = 1000.0;
        rr_intervals[13] = 1000.0;
        rr_intervals[14] = 1000.0;
        rr_intervals[15] = 1000.0;
        rr_intervals[16] = 1000.0;
        rr_intervals[17] = 1000.0;
        rr_intervals[18] = 1000.0;
        let windows = (4..17).collect::<Vec<_>>();
        let dfa_analysis =
            DFAnalysis::udfa(&rr_intervals[..120], &windows, DetrendStrategy::Linear)
                .expect("Failed to perform DFA analysis");
        println!("{:?}", dfa_analysis);
        assert!(
            dfa_analysis.alpha > 0.0,
            "Expected DFA alpha to be positive"
        );
        assert!(
            dfa_analysis.intercept > 0.0,
            "Expected DFA intercept to be positive"
        );
        assert!(
            dfa_analysis.r_squared > 0.0,
            "Expected DFA R-squared to be positive"
        );
    }
}
