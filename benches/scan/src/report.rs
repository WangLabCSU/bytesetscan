use std::{
    env,
    io::{self, IsTerminal, Write},
};

use super::{input::Hit, sampling::Sample};

// Use the same width for both libraries' measurements and separators.
const COLUMN_WIDTH: usize = 26;

pub(super) struct Summary {
    pub(super) median: f64,
    pub(super) min: f64,
    pub(super) max: f64,
}

pub(super) fn header(backend: &str) {
    println!("\n{backend}");
    println!(
        "| needle | haystack | hit    | search | algorithm | {:>COLUMN_WIDTH$} | {:>COLUMN_WIDTH$} |",
        "bytesetscan", "Hyperscan"
    );
    separator();
}

pub(super) fn rows(
    needle: u16,
    len: usize,
    hit: Hit,
    reverse: bool,
    results: [Option<Summary>; 5],
) {
    let hit = hit_name(hit);
    let operation = if reverse { "rfind" } else { "find" };
    let best = results
        .iter()
        .flatten()
        .map(|result| result.median)
        .fold(f64::INFINITY, f64::min);
    let color = io::stdout().is_terminal() && env::var_os("NO_COLOR").is_none();
    let [shufti, truffle, bitset, hs_shufti, hs_truffle] = results;
    for (search, algorithm, ours, reference) in [
        (operation, "Shufti", shufti, hs_shufti),
        ("", "Truffle", truffle, hs_truffle),
        ("", "Bitset", bitset, None),
    ] {
        let pair = [ours, reference];
        let row_best = pair
            .iter()
            .flatten()
            .map(|result| result.median)
            .fold(f64::INFINITY, f64::min);
        print!("| {needle:6} | {len:8} | {hit:6} | {search:6} | {algorithm:9} |");
        for result in pair {
            // The star belongs to the fastest measurement, in either library's
            // column. Missing implementations cannot receive a marker.
            let fastest = result
                .as_ref()
                .is_some_and(|result| (result.median * 100.0).round() == (best * 100.0).round());
            let text = result
                .as_ref()
                .map_or_else(|| "—".to_owned(), |result| format_result(result, fastest));
            // Green compares this row's implementations; equal displayed
            // medians share the highlight, just as they share the star above.
            let lowest = result.is_some_and(|result| {
                (result.median * 100.0).round() == (row_best * 100.0).round()
            });
            if color && lowest {
                print!(" \x1b[32m{text:>COLUMN_WIDTH$}\x1b[0m |");
            } else {
                print!(" {text:>COLUMN_WIDTH$} |");
            }
        }
        println!();
    }
    separator();
}

fn format_result(result: &Summary, fastest: bool) -> String {
    let Summary { median, min, max } = *result;
    let marker = if fastest { "★ " } else { "" };
    let text = format!("{marker}{median:.2} [{min:.2}, {max:.2}]");
    // Rust's string padding counts characters, not the star's UTF-8 bytes.
    if text.chars().count() <= COLUMN_WIDTH {
        text
    } else {
        // Very long scans still fit the same columns; units remain ns/search.
        let text = format!("{marker}{median:.2e} [{min:.2e},{max:.2e}]");
        if text.chars().count() <= COLUMN_WIDTH {
            text
        } else {
            format!("{marker}{median:.1e} [{min:.1e},{max:.1e}]")
        }
    }
}

fn separator() {
    let rule = "-".repeat(COLUMN_WIDTH + 2);
    println!("|--------|----------|--------|--------|-----------|{rule}|{rule}|");
}

pub(super) fn write_samples(
    writer: &mut impl Write,
    backend: &str,
    needle: u16,
    len: usize,
    hit: Hit,
    reverse: bool,
    samples: &[Vec<Sample>; 5],
) -> io::Result<()> {
    const METHODS: [&str; 5] = ["Shufti", "Truffle", "Bitset", "HS Shufti", "HS Truffle"];
    let hit = hit_name(hit);
    let operation = if reverse { "rfind" } else { "find" };
    for (method, batches) in METHODS.iter().zip(samples) {
        for (round, sample) in batches.iter().enumerate() {
            let elapsed = sample.elapsed.as_nanos();
            let count = sample.iterations;
            let ns = elapsed as f64 / count as f64;
            writeln!(
                writer,
                "{backend},{needle},{len},{hit},{operation},{method},{},{count},{elapsed},{ns:.6}",
                round + 1
            )?;
        }
    }
    Ok(())
}

fn hit_name(hit: Hit) -> &'static str {
    match hit {
        Hit::Miss => "miss",
        Hit::First => "first",
        Hit::Middle => "middle",
        Hit::Last => "last",
    }
}

pub(super) fn summarize(samples: &[Sample]) -> Option<Summary> {
    // Missing implementations have no samples. Do not turn them into a zero
    // time, which would incorrectly make an unavailable method look fastest.
    if samples.is_empty() {
        return None;
    }
    let values: Vec<_> = samples
        .iter()
        .map(|sample| sample.elapsed.as_nanos() as f64 / sample.iterations as f64)
        .collect();
    Some(Summary {
        min: values.iter().copied().fold(f64::INFINITY, f64::min),
        max: values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        median: median(values),
    })
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) / 2.0
    } else {
        values[middle]
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{median, summarize, write_samples, Hit, Sample};

    #[test]
    fn median_selects_the_middle_sample() {
        assert_eq!(median(vec![9.0, 1.0, 3.0]), 3.0);
    }

    #[test]
    fn median_averages_the_two_middle_samples() {
        assert_eq!(median(vec![9.0, 1.0, 3.0, 5.0]), 4.0);
    }

    #[test]
    fn summarize_marks_an_unavailable_method_as_missing() {
        assert!(summarize(&[]).is_none());
    }

    #[test]
    fn summarize_normalizes_each_batch_by_its_own_iteration_count() {
        let samples = [
            Sample {
                iterations: 10,
                elapsed: Duration::from_nanos(100),
            },
            Sample {
                iterations: 20,
                elapsed: Duration::from_nanos(200),
            },
        ];
        assert_eq!(summarize(&samples).unwrap().median, 10.0);
    }

    #[test]
    fn summarize_keeps_a_slow_outlier_in_the_reported_range() {
        let samples = [
            Sample {
                iterations: 10,
                elapsed: Duration::from_nanos(100),
            },
            Sample {
                iterations: 10,
                elapsed: Duration::from_nanos(1_000),
            },
            Sample {
                iterations: 10,
                elapsed: Duration::from_nanos(100),
            },
        ];
        assert_eq!(summarize(&samples).unwrap().max, 100.0);
    }

    #[test]
    fn write_samples_preserves_the_raw_batch_measurement() {
        let mut samples = std::array::from_fn(|_| Vec::new());
        samples[0].push(Sample {
            iterations: 4,
            elapsed: Duration::from_nanos(100),
        });
        let mut output = Vec::new();
        write_samples(&mut output, "avx2", 64, 1024, Hit::Miss, false, &samples).unwrap();
        assert_eq!(
            String::from_utf8(output).unwrap(),
            "avx2,64,1024,miss,find,Shufti,1,4,100,25.000000\n"
        );
    }
}
