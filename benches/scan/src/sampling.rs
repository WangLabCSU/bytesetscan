use std::time::Duration;

// One timed batch. Retain its count and elapsed time so a reported ns/search
// value can be checked against the raw measurements.
#[derive(Debug)]
pub(super) struct Sample {
    pub(super) iterations: usize,
    pub(super) elapsed: Duration,
}

// Exponentially increasing batches warm the selected method and provide an
// estimate of its cost. Use the fastest observed rate: a scheduling pause must
// not shrink the next batch and leave it too short to measure reliably.
pub(super) fn warmup(
    mut measure: impl FnMut(usize) -> Duration,
    warmup_time: Duration,
    sample_time: Duration,
) -> usize {
    let mut iterations = 1usize;
    let mut elapsed = Duration::ZERO;
    let mut fastest = f64::INFINITY;
    loop {
        let batch = measure(iterations);
        elapsed = elapsed.saturating_add(batch);
        if !batch.is_zero() {
            fastest = fastest.min(batch.as_nanos() as f64 / iterations as f64);
        }
        if elapsed >= warmup_time {
            return (sample_time.as_nanos() as f64 / fastest).ceil().max(1.0) as usize;
        }
        let next = iterations.saturating_mul(2);
        assert!(next > iterations, "the warmup timer did not advance");
        iterations = next;
    }
}

// A calibration estimate can be stale after the CPU speeds up. If a batch is
// shorter than the requested duration, enlarge it and retry. Long batches,
// including those interrupted by scheduling, remain in the recorded samples.
// A manual iteration count bypasses this duration floor.
pub(super) fn sample(
    mut measure: impl FnMut(usize) -> Duration,
    iterations: &mut usize,
    minimum_time: Option<Duration>,
) -> Sample {
    loop {
        let elapsed = measure(*iterations);
        if minimum_time.is_none_or(|minimum| elapsed >= minimum) {
            return Sample {
                iterations: *iterations,
                elapsed,
            };
        }
        let minimum = minimum_time.unwrap();
        // Add 10% headroom to avoid repeated retries around the threshold.
        // max(1) handles a clock tick too coarse to resolve the initial batch.
        let estimate = (*iterations as f64 * minimum.as_nanos() as f64
            / elapsed.as_nanos().max(1) as f64
            * 1.1)
            .ceil() as usize;
        let next = estimate.max(iterations.saturating_add(1));
        assert!(next > *iterations, "sample iteration count overflowed");
        *iterations = next;
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{sample, warmup};

    #[test]
    fn warmup_runs_for_the_requested_duration() {
        let mut elapsed = Duration::ZERO;
        warmup(
            |iterations| {
                let batch = Duration::from_micros(iterations as u64);
                elapsed += batch;
                batch
            },
            Duration::from_millis(10),
            Duration::from_millis(1),
        );
        assert!(elapsed >= Duration::from_millis(10));
    }

    #[test]
    fn warmup_estimates_the_count_from_the_measured_rate() {
        // Each simulated search takes one microsecond, so a one-millisecond
        // sample needs 1,000 searches regardless of the warmup batch sizes.
        let iterations = warmup(
            |iterations| Duration::from_micros(iterations as u64),
            Duration::from_millis(10),
            Duration::from_millis(1),
        );
        assert_eq!(iterations, 1_000);
    }

    #[test]
    fn sample_enlarges_a_batch_that_is_too_short() {
        let mut iterations = 10;
        let measured = sample(
            |iterations| Duration::from_micros(iterations as u64),
            &mut iterations,
            Some(Duration::from_millis(1)),
        );
        assert!(measured.elapsed >= Duration::from_millis(1));
        assert!(measured.iterations >= 1_000);
    }

    #[test]
    fn sample_keeps_a_long_batch_without_retrying() {
        let mut iterations = 10;
        let mut calls = 0;
        let measured = sample(
            |_| {
                calls += 1;
                Duration::from_millis(50)
            },
            &mut iterations,
            Some(Duration::from_millis(1)),
        );
        assert_eq!(calls, 1);
        assert_eq!(measured.elapsed, Duration::from_millis(50));
    }

    #[test]
    fn sample_recovers_from_an_unresolved_clock_tick() {
        let mut iterations = 1;
        let mut first = true;
        let measured = sample(
            |_| {
                if std::mem::take(&mut first) {
                    Duration::ZERO
                } else {
                    Duration::from_millis(1)
                }
            },
            &mut iterations,
            Some(Duration::from_millis(1)),
        );
        assert_eq!(measured.elapsed, Duration::from_millis(1));
        assert!(measured.iterations > 1);
    }

    #[test]
    fn sample_preserves_a_manual_iteration_count() {
        let mut iterations = 7;
        let measured = sample(
            |count| Duration::from_micros(count as u64),
            &mut iterations,
            None,
        );
        assert_eq!(measured.iterations, 7);
        assert_eq!(measured.elapsed, Duration::from_micros(7));
    }
}
