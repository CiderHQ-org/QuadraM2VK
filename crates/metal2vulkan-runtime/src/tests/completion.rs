//! `command_buffer/completion.rs`: a command buffer's completion is seen as its work ends, while
//! other queues are in use and waited for; and `commit` returns while they are.
//!
//! The consumer's guest presented at about a quarter of Metal's rate under MoltenVK 1.4.1: the
//! completion waited for any of the timelines of the busy queue's first pending batch and
//! of every idle `VkQueue` in use, and MoltenVK's wait for any of several timelines loses a signal
//! that lands while it still registers a later one, so the batch was seen only when the 100 ms
//! wait ran out. These tests make that shape: four queues, three of them in use and idle, and the
//! fourth committing a few fills back to back each round, then waiting for the last. The
//! completion finishes the first and waits again while the next one's signal lands, its
//! moment swept across rounds by the fills' sizes, which is when a wait for any loses it.
//!
//! **The completion** (`a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for`).
//! No completion waiter misses a wake: none of its waits runs out its 100 ms slice while the fence
//! or the timeline value it waits for is signaled (`WorkCounts::missed_completion_wakes`, read
//! after each round), which is how a lost wake shows, a command buffer then seen about a slice
//! late. A count is the runtime's own observation, which a loaded host does not make. On a device
//! that places its timestamps on the host's clock by a calibration (`gpu_time.rs`), a count in a
//! round whose command buffers were all seen within half a slice of when they could first be
//! seen is a signal in the slice's last moment, not a missed wake: printed, not failed.
//!
//! How late each command buffer was seen after it could first be seen is printed, not judged. It is
//! seen completed when its completed handler runs, which the runtime's threads reach from the
//! completion alone; it could first be seen at the later of its GPU end time and its kernel end
//! time where the device calibrates its timestamps, and at its kernel end time elsewhere (a device
//! that answers its GPU times from the host, or calibrates none, ends them where the completion was
//! seen). Under a test suite's load that delay is the host's scheduling, which holds Metal's own
//! handlers as long: on the M3 Max (macOS 27.0.1, at the utility quality of service the Mac lane
//! runs at) beside the runtime lane's tests at nextest's default parallelism, Metal's completed
//! handlers of this test's shape ran up to 82 ms after their `GPUEndTime` (15 of 1800 at 50 ms or
//! more), and the runtime's up to 57 ms after their kernel end, against 0.47 ms and 0.89 ms alone.
//! Neither the round's own time nor when the test's wait returned measures the completion: the
//! test's thread commits the round's command buffers before it waits for the first, so a commit the
//! driver holds holds that wait too, and neither the GPU's other work nor the driver's submission
//! is the completion's. On an RTX 3090 (NVIDIA 615.71.09) beside the lane tests' busy
//! kernels, rounds of 190 to 202 ms were submitted within 0.1 ms, their GPU work begun as long
//! after the commit, as another process's busy kernel runs, and each command buffer seen within 0.7
//! ms of its GPU end; and rounds of 57 to 64 ms ran their first command buffer's work at the commit
//! while the driver's submission of it returned 57 to 64 ms later, each command buffer seen within
//! 0.3 ms of that.
//!
//! **The commit** (`a_commit_returns_while_other_queues_are_waited_for`). Metal's `commit`
//! returns at once, and so does the runtime's: it hands the command buffer's batches to the
//! scheduler's thread and submits nothing itself (`scheduler/inbox.rs`), so a `vkQueueSubmit` the
//! driver holds never holds it. While commits submitted on the committing thread where the
//! scheduler was free, on the RTX 5080 Laptop GPU (NVIDIA 615.71.09) beside about 20 processes
//! opening devices 1 or 2 of a run's 1,800 commits took 58 to 226 ms, the GPU having run the work
//! within 0.1 ms, and on the M3 Max beside the full suite one took 51.4 ms.
//!
//! A commit's time is its wall time less the committing thread's waits for a CPU meanwhile, where
//! the kernel counts them (`timed_commit`): a wait for a lock, and work, stay in it, and the host's
//! preemption of the thread does not. Measured on lavapipe (Mesa 26.2.4): beside a batch's suites
//! (a load of about 48 on 16 hardware threads) one commit of a run's 1,800 took 117.8 ms of wall
//! time; beside 64 spinning threads on 24 hardware threads (a load up to 40), 20 runs of 20 passed,
//! the commits' waits for a CPU up to 2.9 ms and their times less those at most 85 us; and with
//! the scheduler's state lock held 120 ms by another thread while a round committed (a diagnostic
//! build), that round's first commit read 120.2 ms, none of it a wait for a CPU, and the test
//! failed, as it did beside 48 spinning threads (123.1 ms).
//!
//! A summary of the rounds' times, of how late each round's command buffers were seen, of how long
//! each wait returned after its command buffer was seen (what the wait adds), of the commits' times
//! and their waits for a CPU, and of the missed wakes is printed (`ROUNDS`), and each round that
//! took 50 ms or more, or saw a command buffer late, or held a commit that long, or counted a
//! missed wake, with its command buffers' commit times and waits for a CPU, kernel start and end,
//! GPU start and end, completed handler and wait's return, from the round's commit
//! (`SLOW-ROUND`).
//!
//! Device tests: compiled only under the validation crate's GPU safety cfgs (`m2v_gpu`,
//! `m2v_cpu_icd`), on the kind of device the cfg selects.

use crate::{
    host_time, CommandBuffer, CommandBufferStatus, Device, DeviceKind, DeviceOptions,
    ResourceOptions, ShaderCacheOptions,
};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
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

/// Half the completion wait's 100 ms slice: the longest a commit may take; the time from which a
/// round, or how late a command buffer was seen after it could first be seen, prints as slow; and
/// how late a round that counted a missed wake must have seen a command buffer, at least, to fail
/// on a device that calibrates its timestamps (the module docs).
const STALL: Duration = Duration::from_millis(50);

/// A scratch directory for the device's shader cache, removed when dropped.
struct Scratch(PathBuf);

impl Scratch {
    fn new(test: &str) -> Scratch {
        let path =
            std::env::temp_dir().join(format!("m2v-completion-{test}-{}", std::process::id()));
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

/// One command buffer of a round, as the round saw it: host times in seconds.
struct Seen {
    /// Its `commit` call's time, less the committing thread's wait for a CPU meanwhile
    /// ([`timed_commit`]).
    commit: Duration,
    /// That wait.
    queued: Duration,
    /// When its completed handler ran.
    handled: f64,
    /// When the test's wait for it began.
    asked: f64,
    /// When that wait returned.
    returned: f64,
}

/// One round: its fills' size, its time from the first commit until the last wait returned, how
/// late its latest command buffer was seen completed after it could first be seen, and its line
/// for `SLOW-ROUND`.
struct Round {
    size: u64,
    took: Duration,
    late: Duration,
    /// The completion waiters' missed wakes counted while it ran
    /// (`WorkCounts::missed_completion_wakes`).
    missed: u64,
    /// Each command buffer's, in commit order.
    seen: Vec<Seen>,
    /// Its command buffers' times from the round's commit.
    line: String,
}

impl Round {
    /// Its longest commit.
    fn longest_commit(&self) -> Duration {
        self.seen.iter().map(|seen| seen.commit).max().unwrap()
    }
}

/// A completed command buffer's times as `seen` saw them, in milliseconds from `from`, a host
/// time: its commit's wall time, its kernel start and end, GPU start and end, its completed
/// handler, and its wait's return.
fn times(command_buffer: &CommandBuffer, seen: &Seen, from: f64) -> String {
    let ms = |at: f64| (at - from) * 1e3;
    let gpu = |at: Result<f64, _>| at.map_or("-".to_owned(), |at| format!("{:.1}", ms(at)));
    format!(
        "[commit {:.1} queued {:.1} ks {:.1} ke {:.1} gs {} ge {} handled {:.1} returned {:.1}]",
        seen.commit.as_secs_f64() * 1e3,
        seen.queued.as_secs_f64() * 1e3,
        ms(command_buffer.kernel_start_time()),
        ms(command_buffer.kernel_end_time()),
        gpu(command_buffer.gpu_start_time()),
        gpu(command_buffer.gpu_end_time()),
        ms(seen.handled),
        ms(seen.returned)
    )
}

/// Run the rounds on a device of its own (its shader cache in a scratch directory named for
/// `test`): four queues, three of them idle and in use, the fourth committing [`IN_FLIGHT`] fills
/// a round, each with a completed handler that notes the time, then waiting for each. The rounds,
/// and whether the device calibrates its timestamps.
fn rounds(test: &str) -> (Vec<Round>, bool) {
    let scratch = Scratch::new(test);
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
    let mut rounds = Vec::with_capacity(ROUNDS);
    for round in 0..ROUNDS {
        let size = SIZES[round % SIZES.len()];
        // Each with the slot its completed handler notes the host time in.
        let command_buffers: Vec<(CommandBuffer, Arc<Mutex<Option<f64>>>)> = (0..IN_FLIGHT)
            .map(|_| {
                let command_buffer = fill(&queues[0], size, round as u8);
                let handled = Arc::new(Mutex::new(None));
                let noted = Arc::clone(&handled);
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        *noted.lock().unwrap() = Some(host_time());
                    }))
                    .unwrap();
                (command_buffer, handled)
            })
            .collect();
        let missed_before = device.work_counts().missed_completion_wakes;
        let start = Instant::now();
        let committed = host_time();
        let commits: Vec<(Duration, Duration)> = command_buffers
            .iter()
            .map(|(command_buffer, _)| timed_commit(command_buffer))
            .collect();
        let mut seen = Vec::with_capacity(IN_FLIGHT);
        for ((command_buffer, handled), (commit, queued)) in command_buffers.iter().zip(commits) {
            let asked = host_time();
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            let returned = host_time();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            let handled = handled
                .lock()
                .unwrap()
                .expect("its completed handler ran before its wait returned");
            seen.push(Seen {
                commit,
                queued,
                handled,
                asked,
                returned,
            });
        }
        let took = start.elapsed();
        let missed = device.work_counts().missed_completion_wakes - missed_before;
        // How late each was seen after it could first be seen: the latest of the round.
        let late = (command_buffers.iter().zip(&seen))
            .map(|((command_buffer, _), seen)| {
                let seeable = match calibrated {
                    true => (command_buffer.gpu_end_time().unwrap())
                        .max(command_buffer.kernel_end_time()),
                    false => command_buffer.kernel_end_time().max(committed),
                };
                Duration::from_secs_f64((seen.handled - seeable).max(0.0))
            })
            .max()
            .unwrap();
        let line = (command_buffers.iter().zip(&seen))
            .map(|((command_buffer, _), seen)| self::times(command_buffer, seen, committed))
            .collect::<Vec<String>>()
            .join(" ");
        rounds.push(Round {
            size,
            took,
            late,
            missed,
            seen,
            line,
        });
    }
    let summary = |durations: Vec<Duration>| {
        let mut sorted = durations;
        sorted.sort();
        let at = |fraction: usize| sorted[(sorted.len() - 1) * fraction / 100];
        format!(
            "p50 {:?} p90 {:?} p99 {:?} max {:?}",
            at(50),
            at(90),
            at(99),
            sorted[sorted.len() - 1],
        )
    };
    let after = match calibrated {
        true => "the later of its GPU end and kernel end",
        false => "its kernel end",
    };
    for (n, round) in rounds.iter().enumerate() {
        let slow = round.took >= STALL || round.late >= STALL || round.longest_commit() >= STALL;
        if slow || round.missed > 0 {
            println!(
                "SLOW-ROUND {n} ({} bytes a fill): {:?}, seen {:?} late, {} missed wake(s); ms \
                 from the commit: {}",
                round.size, round.took, round.late, round.missed, round.line
            );
        }
    }
    let seen = || rounds.iter().flat_map(|round| &round.seen);
    // What a wait adds: from when its command buffer was seen, or the wait began if later, until
    // it returned.
    let added = seen()
        .map(|seen| {
            Duration::from_secs_f64((seen.returned - seen.handled.max(seen.asked)).max(0.0))
        })
        .collect();
    println!(
        "ROUNDS {ROUNDS}: {}, {} at or over {STALL:?}; seen (completed handler) after {after}: \
         {}, {} at or over {STALL:?}; wait returned after seen: {}; commit time less the wait for \
         a CPU: {}, {} at or over {STALL:?}; the commits' waits for a CPU: {}; missed wakes: {}",
        summary(rounds.iter().map(|round| round.took).collect()),
        rounds.iter().filter(|round| round.took >= STALL).count(),
        summary(rounds.iter().map(|round| round.late).collect()),
        rounds.iter().filter(|round| round.late >= STALL).count(),
        summary(added),
        summary(seen().map(|seen| seen.commit).collect()),
        seen().filter(|seen| seen.commit >= STALL).count(),
        summary(seen().map(|seen| seen.queued).collect()),
        rounds.iter().map(|round| round.missed).sum::<u64>(),
    );
    (rounds, calibrated)
}

/// `command_buffer`'s `commit`, timed: its wall time less the committing thread's wait for a CPU
/// meanwhile, and that wait. Linux counts each thread's wait: the second field of
/// `/proc/thread-self/schedstat`, "time spent waiting on a runqueue" in nanoseconds (the kernel's
/// `Documentation/scheduler/sched-stats.rst`); where it is not counted, the time is the wall time.
/// The instant is read before the count at the start and after it at the end, so a wait between
/// the two reads stays in the time, which never reads less than the call's time on a CPU and
/// blocked: a wait for a lock stays in it.
fn timed_commit(command_buffer: &CommandBuffer) -> (Duration, Duration) {
    let start = Instant::now();
    let before = run_queue_wait();
    command_buffer.commit().unwrap();
    let after = run_queue_wait();
    let wall = start.elapsed();
    let queued = match (before, after) {
        (Some(before), Some(after)) => after.saturating_sub(before),
        _ => Duration::ZERO,
    };
    (wall.saturating_sub(queued), queued)
}

/// The calling thread's wait for a CPU so far, where the kernel counts it ([`timed_commit`]).
fn run_queue_wait() -> Option<Duration> {
    let stat = std::fs::read_to_string("/proc/thread-self/schedstat").ok()?;
    let waited = stat.split_whitespace().nth(1)?.parse::<u64>().ok()?;
    Some(Duration::from_nanos(waited))
}

#[test]
fn a_completion_on_one_queue_is_seen_while_other_queues_are_waited_for() {
    let (rounds, calibrated) = rounds("seen");
    let missed: Vec<(usize, u64, Duration)> = (rounds.iter().enumerate())
        .filter(|(_, round)| round.missed > 0 && (!calibrated || round.late >= STALL))
        .map(|(n, round)| (n, round.missed, round.late))
        .collect();
    assert!(
        missed.is_empty(),
        "completion waits that ran out their slice while what they waited for was signaled \
         (round, missed wakes, latest seen after it could be): {missed:?}"
    );
}

#[test]
fn a_commit_returns_while_other_queues_are_waited_for() {
    let (rounds, _) = rounds("commit");
    let held: Vec<(usize, Duration)> = (rounds.iter().enumerate())
        .filter(|(_, round)| round.longest_commit() >= STALL)
        .map(|(n, round)| (n, round.longest_commit()))
        .collect();
    assert!(
        held.is_empty(),
        "commits that took {STALL:?} or more, less the committing thread's wait for a CPU, \
         where Metal's returns at once (round, longest): {held:?}. A commit only hands its \
         batches to the scheduler's thread (scheduler/inbox.rs): one this slow waited for a lock \
         or worked that long, not for a submission"
    );
}
