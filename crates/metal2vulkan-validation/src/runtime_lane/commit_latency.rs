//! How long a checked command buffer waits from its commit to its submission under a consumer's
//! load, on the device the cfg selects, without the validation layer: report only, in release
//! builds.
//!
//! A consumer commits every command buffer with a submission check
//! (`CommandBuffer::set_submission_check`), so the device's scheduler thread runs each check and
//! submits each batch, and its host memory is hundreds of no-copy buffers, which every kernel
//! submission lists (`submit_costs`). The load: [`IMPORTS`] live host imports of
//! [`IMPORT_BYTES`] each, and frames of [`PER_FRAME`] command buffers committed at [`PER_MS`] a
//! millisecond, [`FRAMES`] frames a phase. Each command buffer is a blit fill of a buffer of its
//! committing thread's with a check that lets it run and a completed handler that notes the time;
//! at most [`FLIGHT`] of a thread's are in flight. The phases:
//!
//! - **One thread** (`one thread`): one committing thread.
//! - **Threads** (`threads`): [`THREADS`] committing threads at once, of queues of their own, each
//!   at its share of the rate.
//! - **Waiting for the GPU** (`event waits`): one thread whose every other command buffer waits
//!   for an event the one before it signals on another queue: the check waits until that signal
//!   landed, so the completion turns' wakes release it.
//!
//! Printed as `COMMIT-LATENCY` lines, per phase: the commits a millisecond the threads kept; per
//! command buffer, the time from the host's clock just before `commit` to its `kernelStartTime`
//! (the scheduler began submitting it), its `kernelEndTime` (its `vkQueueSubmit` returned) and its
//! completed handler, as their 50th and 95th percentiles and their longest; and per frame, the CPU
//! time of the device's scheduler thread, of its queues' waiters (which make the completion
//! turns), of the process and of the
//! committing threads (and their wall time in `commit`), the runtime's `vkQueueSubmit` calls
//! (`WorkCounts::queue_submissions`), the commits handed to the scheduler's thread
//! (`WorkCounts::commits_handed_off`: every commit), the `VkCommandBuffer`s begun
//! (`WorkCounts::command_buffers_begun`), the barrier command buffers that write timestamps
//! recorded to keep and taken again (`WorkCounts::stamped_buffers_kept`,
//! `WorkCounts::stamped_buffers_reused`), the command pools made and the spare ones taken
//! (`WorkCounts::command_pools_made`, `WorkCounts::spare_command_pools_taken`) and those reset
//! as taken (`WorkCounts::spare_command_pools_reset`), the render pass instances whose draws
//! recording ended unrun (`WorkCounts::unrun_instances_ended`), the batches whose end went into
//! their last command buffer and those whose end was a command buffer of its own
//! (`WorkCounts::finales_folded`, `WorkCounts::finales_apart`), the batches that began with no
//! command buffer of their own and the waits on completed work's timelines their submissions held
//! (`WorkCounts::bare_starts`, `WorkCounts::completed_waits`), the timed command buffers whose
//! first encoder wrote their start (`WorkCounts::prefixed_starts`), and where the
//! kernel-submission counter (`scripts/i915-execbuffers/counter.c`) is preloaded, the kernel's
//! submissions, the buffer objects each lists and the host imports among them, and the time each
//! spent in the ioctl.
//!
//! `commit_latency_without_imports` runs the same phases with no host import alive: the
//! submissions' own cost, without the per-object cost of the kernel's lists.
//!
//! Ignored by default: a measurement, not a check. Run it by name with `--run-ignored all`.

use super::commit_costs::{process_cpu_us, since};
use super::submit_costs::{import, thread_cpu_us, Execbuffers};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use metal2vulkan_runtime::{
    host_time, Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Device, EventRef,
    ResourceOptions, Submission, SubmissionCheck, WorkCounts,
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// The host imports live while the phases run.
const IMPORTS: usize = 400;
/// The bytes of each: a guest page. amdgpu's cost for each userptr object a command submission
/// lists was about the same for one page and for 64 (6.8 and 7.6 us per object, on a Radeon 780M,
/// Linux 7.2, a loaded host): it is per object.
const IMPORT_BYTES: usize = 4 << 10;
/// Command buffers committed a millisecond, of all the phase's threads.
const PER_MS: u32 = 6;
/// Command buffers a frame.
const PER_FRAME: usize = 96;
/// Frames a phase.
const FRAMES: usize = 30;
/// Committing threads at once in `threads`.
const THREADS: usize = 3;
/// A thread's command buffers in flight at most.
const FLIGHT: usize = 2 * PER_FRAME;
/// The linux name (15 bytes) of the device's scheduler thread.
const SCHEDULER: &str = "metal2vulkan-sc";
/// The linux name (15 bytes) of the device's queues' waiters, which make the completion turns.
const WAITERS: &str = "metal2vulkan-qu";

/// A check that lets every submission run.
fn check() -> SubmissionCheck {
    Arc::new(|_: &Submission<'_>| None)
}

/// A checked command buffer of `queue` filling `buffer`'s first 4 KiB with `value`, not
/// committed.
fn checked(queue: &CommandQueue, buffer: &Buffer, value: u8) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..4096, value).unwrap();
    blit.end_encoding().unwrap();
    command_buffer.set_submission_check(check()).unwrap();
    command_buffer
}

/// A committed command buffer, the host time of its commit and the host time its completed
/// handler ran (the bits of an `f64`, 0 before).
struct Committed {
    command_buffer: CommandBuffer,
    at: f64,
    completed: Arc<AtomicU64>,
}

/// Commit `command_buffer`, with a completed handler that notes the time: the committed command
/// buffer.
fn commit(command_buffer: CommandBuffer) -> Committed {
    let completed = Arc::new(AtomicU64::new(0));
    let noted = Arc::clone(&completed);
    let handler = move |_: &CommandBuffer| noted.store(host_time().to_bits(), Ordering::Relaxed);
    command_buffer
        .add_completed_handler(Box::new(handler))
        .unwrap();
    let at = host_time();
    command_buffer.commit().unwrap();
    Committed {
        command_buffer,
        at,
        completed,
    }
}

/// Wait for `committed` and require it completed.
fn completed(committed: &Committed) {
    let command_buffer = &committed.command_buffer;
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
}

/// One committing thread's command buffers and its costs.
struct Thread {
    committed: Vec<Committed>,
    /// The thread's CPU time, in microseconds.
    cpu: f64,
    /// The thread's wall time in `commit`, in microseconds.
    in_commit: f64,
    /// The thread's wall time, in microseconds.
    wall: f64,
}

/// Commit what `encode` makes of each of `units`, one unit every `pace`, with at most
/// [`FLIGHT`] command buffers in flight.
fn paced(
    units: usize,
    pace: Duration,
    mut encode: impl FnMut(usize) -> Vec<CommandBuffer>,
) -> Thread {
    let mut flight = VecDeque::new();
    let mut done = Vec::new();
    let mut in_commit = 0.0;
    let cpu = thread_cpu_us();
    let start = Instant::now();
    for unit in 0..units {
        let due = start + pace * unit as u32;
        if let Some(early) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(early);
        }
        for command_buffer in encode(unit) {
            let begin = Instant::now();
            flight.push_back(commit(command_buffer));
            in_commit += since(begin);
        }
        while flight.len() > FLIGHT {
            let oldest = flight.pop_front().unwrap();
            completed(&oldest);
            done.push(oldest);
        }
    }
    for committed in flight {
        completed(&committed);
        done.push(committed);
    }
    Thread {
        committed: done,
        cpu: thread_cpu_us() - cpu,
        in_commit,
        wall: since(start),
    }
}

/// One committing thread of `count` command buffers at `per_ms` a millisecond.
fn committer(device: &Device, count: usize, per_ms: u32) -> Thread {
    let queue = device.new_command_queue().unwrap();
    let buffer = device.new_buffer(4096, ResourceOptions::private()).unwrap();
    let pace = Duration::from_millis(1) / per_ms;
    paced(count, pace, |at| vec![checked(&queue, &buffer, at as u8)])
}

/// One committing thread whose every other command buffer waits for the event the one before it
/// signals on another queue (the module docs).
fn event_waits(device: &Device) -> Thread {
    let (signaling, waiting) = (
        device.new_command_queue().unwrap(),
        device.new_command_queue().unwrap(),
    );
    let event = device.new_shared_event().unwrap();
    let buffer = device
        .new_buffer(1 << 20, ResourceOptions::private())
        .unwrap();
    let pace = Duration::from_millis(2) / PER_MS;
    paced(FRAMES * PER_FRAME / 2, pace, |at| {
        let value = at as u64 + 1;
        let signal = checked(&signaling, &buffer, value as u8);
        signal
            .encode_signal_event(EventRef::from(&event), value)
            .unwrap();
        let wait = waiting.command_buffer().unwrap();
        wait.encode_wait_for_event(EventRef::from(&event), value)
            .unwrap();
        let mut blit = wait.blit_command_encoder().unwrap();
        blit.fill_buffer(&buffer, 4096..8192, !(value as u8))
            .unwrap();
        blit.end_encoding().unwrap();
        wait.set_submission_check(check()).unwrap();
        vec![signal, wait]
    })
}

/// The CPU time so far of the process's threads named `name`, in microseconds (0 where Linux's
/// `/proc` is not).
pub(super) fn named_cpu_us(name: &str) -> f64 {
    let Ok(tasks) = std::fs::read_dir("/proc/self/task") else {
        return 0.0;
    };
    let named = |task: &std::fs::DirEntry| {
        std::fs::read_to_string(task.path().join("comm")).is_ok_and(|comm| comm.trim_end() == name)
    };
    let on_cpu_ns = |task: std::fs::DirEntry| {
        let stat = std::fs::read_to_string(task.path().join("schedstat")).ok()?;
        stat.split_whitespace().next()?.parse::<f64>().ok()
    };
    let tasks = tasks.flatten().filter(named);
    tasks.filter_map(on_cpu_ns).sum::<f64>() / 1e3
}

/// The 50th and 95th percentiles and the longest of `values`, in milliseconds.
fn spread(mut values: Vec<f64>) -> String {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)] * 1e3;
    let longest = values.last().copied().unwrap_or(0.0) * 1e3;
    format!("p50 {:.3} p95 {:.3} max {longest:.3} ms", at(50), at(95))
}

/// The counts and clocks at a phase's start.
struct Phase {
    counts: WorkCounts,
    execbuffers: Option<Execbuffers>,
    process: f64,
    scheduler: f64,
    waiters: f64,
}

impl Phase {
    fn start(device: &Device) -> Phase {
        Phase {
            counts: device.work_counts(),
            execbuffers: Execbuffers::read(),
            process: process_cpu_us(),
            scheduler: named_cpu_us(SCHEDULER),
            waiters: named_cpu_us(WAITERS),
        }
    }

    /// Print the phase's latencies and costs as `what` (the module docs).
    fn print(&self, device: &Device, what: &str, threads: &[Thread]) {
        let committed = || threads.iter().flat_map(|thread| &thread.committed);
        let count = committed().count();
        let frames = count as f64 / PER_FRAME as f64;
        let wall = threads.iter().map(|thread| thread.wall).fold(0.0, f64::max);
        let latency =
            |time: fn(&Committed) -> Option<f64>| spread(committed().filter_map(time).collect());
        eprintln!(
            "COMMIT-LATENCY {}: {what}: {count} command buffers at {:.2} a ms; commit to \
             submitting {}; to submitted {}; to completed handler {}",
            device.name(),
            count as f64 / (wall / 1e3),
            latency(|c| Some(c.command_buffer.kernel_start_time() - c.at)),
            latency(|c| Some(c.command_buffer.kernel_end_time() - c.at)),
            // A handler that has not run yet is left out.
            latency(|c| {
                let handled = f64::from_bits(c.completed.load(Ordering::Relaxed));
                (handled > 0.0).then_some(handled - c.at)
            }),
        );
        let after = device.work_counts();
        let per_frame = |after: u64, before: u64| (after - before) as f64 / frames;
        let ms = |after: f64, before: f64| (after - before) / 1e3 / frames;
        let threads_ms = |time: fn(&Thread) -> f64| threads.iter().map(time).sum::<f64>() / 1e3;
        let kernel = Execbuffers::read()
            .zip(self.execbuffers)
            .map(|(after, before)| {
                let submissions = after.execbuffers - before.execbuffers;
                let each = |after: u64, before: u64| (after - before) as f64 / submissions as f64;
                format!(
                    "; {:.1} kernel submissions, {:.0} objects each ({:.0} host imports), {:.1} us \
                     each in the ioctl",
                    submissions as f64 / frames,
                    each(after.objects, before.objects),
                    each(after.userptr, before.userptr),
                    each(after.nanos, before.nanos) / 1e3,
                )
            })
            .unwrap_or_default();
        eprintln!(
            "COMMIT-LATENCY {}: {what}: a frame of {PER_FRAME}: scheduler thread cpu {:.2} ms, \
             queue waiters {:.2} ms, process {:.2} ms, committing threads {:.2} ms ({:.2} ms \
             in commit); {:.1} vkQueueSubmit calls, {:.1} commits handed off, {:.1} command \
             buffers begun, {:.1} stamped barrier command buffers kept, {:.1} reused, {:.1} \
             command pools made, {:.1} spare ones taken, {:.1} of them reset, {:.1} unrun render \
             pass instances ended, {:.1} batch ends in the work, {:.1} apart, {:.1} bare starts, \
             {:.1} waits on completed work, {:.1} starts in the first encoder{kernel}",
            device.name(),
            ms(named_cpu_us(SCHEDULER), self.scheduler),
            ms(named_cpu_us(WAITERS), self.waiters),
            ms(process_cpu_us(), self.process),
            threads_ms(|thread| thread.cpu) / frames,
            threads_ms(|thread| thread.in_commit) / frames,
            per_frame(after.queue_submissions, self.counts.queue_submissions),
            per_frame(after.commits_handed_off, self.counts.commits_handed_off),
            per_frame(
                after.command_buffers_begun,
                self.counts.command_buffers_begun
            ),
            per_frame(after.stamped_buffers_kept, self.counts.stamped_buffers_kept),
            per_frame(
                after.stamped_buffers_reused,
                self.counts.stamped_buffers_reused
            ),
            per_frame(after.command_pools_made, self.counts.command_pools_made),
            per_frame(
                after.spare_command_pools_taken,
                self.counts.spare_command_pools_taken
            ),
            per_frame(
                after.spare_command_pools_reset,
                self.counts.spare_command_pools_reset
            ),
            per_frame(
                after.unrun_instances_ended,
                self.counts.unrun_instances_ended
            ),
            per_frame(after.finales_folded, self.counts.finales_folded),
            per_frame(after.finales_apart, self.counts.finales_apart),
            per_frame(after.bare_starts, self.counts.bare_starts),
            per_frame(after.completed_waits, self.counts.completed_waits),
            per_frame(after.prefixed_starts, self.counts.prefixed_starts),
        );
    }
}

/// The latencies, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn commit_latency() {
    latencies(IMPORTS);
}

/// The latencies with no host import alive, printed (the module docs).
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn commit_latency_without_imports() {
    latencies(0);
}

/// The phases with `imports` live host imports, printed (the module docs).
fn latencies(imports: usize) {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
        // A consumer's device keeps no execution record.
        options.record_executions = false;
    })
    .unwrap();
    let live: Vec<Buffer> = (0..imports)
        .map(|_| import(&device, IMPORT_BYTES))
        .collect();
    // Warm: the fills' pipelines and the pools.
    drop(committer(&device, PER_FRAME, PER_MS));
    let phase = Phase::start(&device);
    let one = committer(&device, FRAMES * PER_FRAME, PER_MS);
    phase.print(&device, "one thread", &[one]);
    let phase = Phase::start(&device);
    let (each, per_ms) = (FRAMES * PER_FRAME / THREADS, PER_MS / THREADS as u32);
    let threads: Vec<Thread> = std::thread::scope(|scope| {
        let threads: Vec<_> = (0..THREADS)
            .map(|_| scope.spawn(|| committer(&device, each, per_ms)))
            .collect();
        (threads.into_iter())
            .map(|thread| thread.join().unwrap())
            .collect()
    });
    phase.print(&device, "threads", &threads);
    let phase = Phase::start(&device);
    let waits = event_waits(&device);
    phase.print(&device, "event waits", &[waits]);
    assert_eq!(device.work_counts().host_imports_live, imports as u64);
    drop(live);
}
