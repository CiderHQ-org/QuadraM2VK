//! What encoding a stream of dispatches costs on the device the cfg selects, without the
//! validation layer: report only, in release builds, to weigh the runtime's dispatch path before
//! and after a change to it, as `draw_costs` weighs its draw path.
//!
//! Each stream encodes [`WARM`] command buffers untimed, then [`TIMED`], each one compute encoder
//! of [`DISPATCHES`] dispatches of the owned `a4_index_word` (`out[t] = in[3t + 1]`, [`THREADS`]
//! threads), and is printed as a `DISPATCH-COST` line: the dispatches a second the host encodes
//! (from the encoder's creation through `end_encoding`), the 10th, 50th and 90th percentiles of a
//! command buffer's encode time per dispatch, and the median of the encoding thread's own CPU time
//! per dispatch over the same span.
//!
//! - **`unchanged`**: every dispatch the same, nothing set between dispatches.
//! - **`buffers`**: each dispatch binding its own offsets of the same two buffers.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, as `draw_costs`.

use super::draw_costs::{percentiles, thread_cpu};
use super::indirect_host_steps::state;
use super::{buffer_with, validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{Buffer, CommandBufferStatus, ComputePipelineState, Device, Size};
use std::time::{Duration, Instant};

/// The untimed and the timed command buffers of each stream.
const WARM: usize = 5;
const TIMED: usize = 20;
/// The dispatches of a command buffer's encoder, and the threads of each.
const DISPATCHES: u64 = 500;
const THREADS: u64 = 16;

/// The streams (the module docs).
#[derive(Clone, Copy, Debug)]
enum Stream {
    Unchanged,
    Buffers,
}

impl Stream {
    fn name(self) -> &'static str {
        match self {
            Stream::Unchanged => "unchanged",
            Stream::Buffers => "buffers",
        }
    }
}

/// One command buffer's times.
struct Run {
    encoded: Duration,
    cpu: Duration,
}

/// Encode one command buffer of `stream` with `kernel` over `input` and `out`, commit it and wait
/// for it: the wall time its encoding took and the encoding thread's CPU time over the same span.
fn run(
    device: &Device,
    kernel: &ComputePipelineState,
    stream: Stream,
    input: &Buffer,
    out: &Buffer,
) -> Run {
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let start = Instant::now();
    let cpu_start = thread_cpu();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encoder.set_compute_pipeline_state(kernel).unwrap();
    encoder.set_buffer(Some(input), 0, 0).unwrap();
    encoder.set_buffer(Some(out), 0, 1).unwrap();
    for d in 0..DISPATCHES {
        if matches!(stream, Stream::Buffers) {
            encoder.set_buffer_offset(4 * 3 * THREADS * d, 0).unwrap();
            encoder.set_buffer_offset(4 * THREADS * d, 1).unwrap();
        }
        encoder
            .dispatch_threads(Size::new(THREADS, 1, 1), Size::new(THREADS, 1, 1))
            .unwrap();
    }
    encoder.end_encoding().unwrap();
    let encoded = start.elapsed();
    let cpu = thread_cpu().saturating_sub(cpu_start);
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
    Run { encoded, cpu }
}

/// The dispatch path's costs, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn dispatch_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let name = device.name().to_owned();
    let kernel = state(&device, "a4_index_word", false);
    let input = buffer_with(&device, &vec![0; (4 * 3 * THREADS * DISPATCHES) as usize]).unwrap();
    let out = buffer_with(&device, &vec![0; (4 * THREADS * DISPATCHES) as usize]).unwrap();
    eprintln!(
        "DISPATCH-COST {name}: {DISPATCHES} dispatches of {THREADS} threads an encoder, {WARM} \
         untimed and {TIMED} timed command buffers a stream"
    );
    for stream in [Stream::Unchanged, Stream::Buffers] {
        for _ in 0..WARM {
            run(&device, &kernel, stream, &input, &out);
        }
        let runs: Vec<Run> = (0..TIMED)
            .map(|_| run(&device, &kernel, stream, &input, &out))
            .collect();
        let encoded: Duration = runs.iter().map(|run| run.encoded).sum();
        let rate = (TIMED as u64 * DISPATCHES) as f64 / encoded.as_secs_f64();
        let per = |d: Duration| d.as_secs_f64() * 1e6 / DISPATCHES as f64;
        let [low, median, high] = percentiles(runs.iter().map(|run| per(run.encoded)).collect());
        let [_, cpu, _] = percentiles(runs.iter().map(|run| per(run.cpu)).collect());
        eprintln!(
            "DISPATCH-COST {name}: {}: {rate:.0} dispatches/s encoding; {median:.2} us a dispatch \
             (p10 {low:.2}, p90 {high:.2}), {cpu:.2} us of it the thread's CPU",
            stream.name()
        );
    }
}
