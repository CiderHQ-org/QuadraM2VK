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
//! moment swept across rounds by the fills' sizes, which is when a wait for any loses it. Every
//! round must be seen completed well within the wait's 100 ms slice; a lost wake shows as a round
//! of about 100 ms. A summary of the rounds' times is printed (`ROUNDS`).
//!
//! A device test: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    CommandBufferStatus, Device, DeviceKind, DeviceOptions, ResourceOptions, ShaderCacheOptions,
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

/// The longest a round may take: half the completion wait's 100 ms slice, far above a round's
/// own time (three fills of at most 8 MiB).
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

#[test]
fn a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for() {
    let scratch = Scratch::new();
    let device = device(&scratch.0);
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
    for round in 0..ROUNDS {
        let size = SIZES[round % SIZES.len()];
        let command_buffers: Vec<_> = (0..IN_FLIGHT)
            .map(|_| fill(&queues[0], size, round as u8))
            .collect();
        let start = Instant::now();
        for command_buffer in &command_buffers {
            command_buffer.commit().unwrap();
        }
        for command_buffer in &command_buffers {
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        }
        times.push(start.elapsed());
    }
    let stalls: Vec<(usize, Duration)> = times
        .iter()
        .copied()
        .enumerate()
        .filter(|&(_, time)| time >= STALL)
        .collect();
    let mut sorted = times.clone();
    sorted.sort();
    println!(
        "ROUNDS {ROUNDS}: p50 {:?} p90 {:?} p99 {:?} max {:?}; {} at or over {STALL:?}",
        sorted[ROUNDS / 2],
        sorted[ROUNDS * 9 / 10],
        sorted[ROUNDS * 99 / 100],
        sorted[ROUNDS - 1],
        stalls.len()
    );
    assert!(
        stalls.is_empty(),
        "commits seen completed only after {STALL:?} or more (round, time): {stalls:?}"
    );
}
