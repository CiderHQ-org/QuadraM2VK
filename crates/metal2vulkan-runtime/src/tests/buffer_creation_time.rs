//! What creating a large `Shared` buffer costs: its memory's allocation and the zeroing every new
//! buffer gets, on a real device. Prints the times; asserts only that the zeroing is complete.
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use super::blit_holds::{device, Scratch};
use crate::ResourceOptions;
use std::time::Instant;

#[test]
#[ignore = "allocates 3 GiB of host memory and prints timings; run it on purpose"]
fn large_shared_buffers_report_what_their_creation_costs() {
    let scratch = Scratch::new("buffer-creation-time");
    let device = device(&scratch.0);
    for (what, length, count) in [
        ("64 KiB", 64u64 << 10, 2000),
        ("4 MiB", 4 << 20, 100),
        ("64 MiB", 64 << 20, 16),
        ("256 MiB", 256 << 20, 8),
    ] {
        let mut kept = Vec::new();
        let (mut total, mut worst) = (0.0f64, 0.0f64);
        for _ in 0..count {
            let start = Instant::now();
            let buffer = device
                .new_buffer(length, ResourceOptions::shared())
                .unwrap();
            let took = start.elapsed().as_secs_f64() * 1000.0;
            total += took;
            worst = worst.max(took);
            kept.push(buffer);
        }
        let bytes = length * count as u64;
        eprintln!(
            "CREATE: {count} x {what} ({} MiB): {total:.0} ms in all, {:.2} ms each, worst {worst:.1} ms, {:.1} GB/s",
            bytes >> 20,
            total / count as f64,
            bytes as f64 / 1e9 / (total / 1000.0),
        );
        // Every buffer reads zero.
        let sample = kept.last().unwrap().contents().unwrap();
        let first =
            unsafe { std::slice::from_raw_parts(sample.as_ptr(), length.min(4096) as usize) };
        assert!(first.iter().all(|&byte| byte == 0));
        // Release them all and time that too.
        let start = Instant::now();
        drop(kept);
        eprintln!(
            "RELEASE: {count} x {what}: {:.0} ms",
            start.elapsed().as_secs_f64() * 1000.0
        );
    }
}
