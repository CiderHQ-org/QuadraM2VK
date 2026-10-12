//! `command_buffer/completion.rs`: a command buffer's completion is seen as its work ends, while
//! other queues are in use and waited for.
//!
//! The consumer's guest presented at about a quarter of Metal's rate under MoltenVK 1.4.1: the
//! completion thread waited for any of the timelines of the busy queue's first pending batch and
//! of every idle `VkQueue` in use, and MoltenVK's wait for any of several timelines loses a signal
//! that lands while it still registers a later one, so the batch was seen only when the 100 ms
//! wait ran out. This test makes that shape: four queues, three of them in use and idle, and the
//! fourth committing a few fills back to back each round, then waiting for the last. The
//! completion thread finishes the first and waits again while the next one's signal lands, its
//! moment swept across rounds by the fills' sizes, which is when a wait for any loses it.
//!
//! Every command buffer must be seen completed well within the wait's 100 ms slice of the moment
//! it could first be seen, once its work ended and the submission that signals its end returned;
//! a lost wake shows as about 100 ms. That moment is the later of its GPU end time and its kernel
//! end time where the device places its timestamps on the host's clock by a calibration
//! (`gpu_time.rs`), and the round's commit elsewhere, which comes before it: a device that answers
//! its GPU times from the host, or calibrates none, ends them where the completion was seen. The
//! round's own time is no measure of the completion, as neither the GPU's other work nor the
//! driver's submission is the completion thread's: on an RTX 3090 (NVIDIA 615.71.09) beside the
//! lane tests' busy kernels, rounds of 190 to 202 ms were submitted within 0.1 ms, their GPU work
//! begun as long after the commit, as another process's busy kernel runs, and each command buffer
//! seen within 0.7 ms of its GPU end; and rounds of 57 to 64 ms ran their first command buffer's
//! work at the commit while the driver's submission of it returned 57 to 64 ms later, each command
//! buffer seen within 0.3 ms of that. A summary of the rounds' times and of how late each round's
//! command buffers were seen is printed (`ROUNDS`), and each round that took 50 ms or more with
//! its command buffers' kernel start and end, GPU start and end, and the moment each was seen, from
//! the round's commit (`SLOW-ROUND`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    host_time, CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions,
};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

/// The longest a command buffer may take.
const WAIT: Duration = Duration::from_secs(60);

/// How many rounds the busy queue runs.
const ROUNDS: usize = 600;

/// The command buffers of a round, committed back to back.
const IN_FLIGHT: usize = 3;

/// The fills' sizes, in turn over the rounds: their GPU times move each signal's moment.
const SIZES: [u64; 7] = [
    4 << 10,
    16 << 10,
    64 << 10,
    256 << 10,
    1 << 20,
    4 << 20,
    8 << 20,
];

/// The latest a command buffer may be seen completed after it could first be seen: half the
/// completion wait's 100 ms slice.
const STALL: Duration = Duration::from_millis(50);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> Scratch {
        let path = std::env::temp_dir().join(format!("m2v-completion-{}", std::process::id()));
        std::fs::create_dir_all(&path).unwrap();
        Scratch(path)
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// The device the cfg selects: a CPU device under `m2v_cpu_icd`, else a GPU, its shader cache
/// in `cache`.
fn device(cache: &Path) -> Device {
    let options = DeviceOptions {
        kinds: match cfg!(m2v_cpu_icd) {
            true => vec![DeviceKind::Cpu],
            false => vec![
                DeviceKind::Discrete,
                DeviceKind::Integrated,
                DeviceKind::Virtual,
            ],
        },
        shader_cache: ShaderCacheOptions {
            directory: Some(cache.to_owned()),
            ..ShaderCacheOptions::default()
        },
        ..DeviceOptions::default()
    };
    Device::system_default_with(&options).unwrap()
}

/// Whether `device`'s GPU end times are when the GPU ended the work: its timestamps, placed on
/// the host's clock by a calibration (`gpu_time.rs`).
fn calibrated(device: &Device) -> bool {
    let timing = &device.shared().timing;
    !timing.answers_from_host()
        && timing
            .clock()
            .is_some_and(|clock| clock.calibration.is_some())
}

/// A completed command buffer's times, in milliseconds from `from`, a host time: its kernel start
/// and end, GPU start and end, and `seen`, when its wait returned.
fn times(command_buffer: &CommandBuffer, from: f64, seen: f64) -> String {
    let ms = |at: f64| (at - from) * 1e3;
    let gpu = |at: Result<f64, _>| at.map_or("-".to_owned(), |at| format!("{:.1}", ms(at)));
    format!(
        "[ks {:.1} ke {:.1} gs {} ge {} seen {:.1}]",
        ms(command_buffer.kernel_start_time()),
        ms(command_buffer.kernel_end_time()),
        gpu(command_buffer.gpu_start_time()),
        gpu(command_buffer.gpu_end_time()),
        ms(seen)
    )
}

#[test]
fn a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
    let calibrated = calibrated(&device);
    let buffer = device
        .new_buffer(SIZES[SIZES.len() - 1], ResourceOptions::private())
        .unwrap();
    // Four queues, round robin over the device's `VkQueue`s: each is in use.
    let queues: Vec<_> = (0..4)
        .map(|_| device.new_command_queue().unwrap())
        .collect();
    // A short command buffer: one fill of `size` bytes.
    let fill = |queue: &crate::CommandQueue, size: u64, value: u8| {
        let command_buffer = queue.command_buffer().unwrap();
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 0..size, value).unwrap();
        blit.end_encoding().unwrap();
        command_buffer
    };
    // Each idle queue runs one first, so its `VkQueue` has a value its waits name.
    for (value, queue) in queues[1..].iter().enumerate() {
        let command_buffer = fill(queue, SIZES[0], value as u8);
        command_buffer.commit().unwrap();
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    }
    let mut times = Vec::with_capacity(ROUNDS);
    let mut lates = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let size = SIZES[round % SIZES.len()];
        let command_buffers: Vec<_> = (0..IN_FLIGHT)
            .map(|_| fill(&queues[0], size, round as u8))
            .collect();
        let start = Instant::now();
        let committed = host_time();
        for command_buffer in &command_buffers {
            command_buffer.commit().unwrap();
        }
        let mut seen = Vec::with_capacity(IN_FLIGHT);
        for command_buffer in &command_buffers {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            seen.push(host_time());
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        let took = start.elapsed();
        // How late each was seen after it could first be seen: the latest of the round.
        let late = (command_buffers.iter().zip(&seen))
            .map(|(command_buffer, &seen)| {
                let seeable = match calibrated {
                    true => (command_buffer.gpu_end_time().unwrap())
                        .max(command_buffer.kernel_end_time()),
                    false => committed,
                };
                Duration::from_secs_f64((seen - seeable).max(0.0))
            })
            .max()
            .unwrap();
        if took >= STALL || late >= STALL {
            let each: Vec<String> = (command_buffers.iter().zip(&seen))
                .map(|(command_buffer, &seen)| self::times(command_buffer, committed, seen))
                .collect();
            println!(
                "SLOW-ROUND {round} ({size} bytes a fill): {took:?}, seen {late:?} late; ms from \
                 the commit: {}",
                each.join(" ")
            );
        }
        times.push(took);
        lates.push(late);
    }
    let stalls: Vec<(usize, Duration)> = lates
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, late)| late >= STALL)
        .collect();
    let summary = |durations: &[Duration]| {
        let mut sorted = durations.to_vec();
        sorted.sort();
        format!(
            "p50 {:?} p90 {:?} p99 {:?} max {:?}",
            sorted[ROUNDS / 2],
            sorted[ROUNDS * 9 / 10],
            sorted[ROUNDS * 99 / 100],
            sorted[ROUNDS - 1],
        )
    };
    let after = match calibrated {
        true => "the later of its GPU end and kernel end",
        false => "the round's commit",
    };
    println!(
        "ROUNDS {ROUNDS}: {}, {} at or over {STALL:?}; seen after {after}: {}, {} at or over \
         {STALL:?}",
        summary(&times),
        times.iter().filter(|&&took| took >= STALL).count(),
        summary(&lates),
        stalls.len()
    );
    assert!(
        stalls.is_empty(),
        "command buffers seen completed {STALL:?} or more after {after} (round, time): {stalls:?}"
    );
}
