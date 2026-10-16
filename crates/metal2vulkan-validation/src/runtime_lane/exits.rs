//! A process that exits while the runtime still works, on the device the cfg selects, under the
//! validation layer.
//!
//! Each case runs in a child process ([`child`], this test binary again), which ends with `exit`
//! or at the end of `main` while the runtime's threads hold work, run the program's code, or tear
//! a device down. The parent checks that the child exited with status 0, not by a signal, and
//! that its exit handlers ended within [`EXIT_LIMIT`] of its last line, or of the end of the CPU
//! device's work its exit waits for where that is later, less the time the exiting thread waited
//! for a CPU meanwhile ([`exit_end`]): below the exit hook's own bound of 2 s, so a hook that
//! waited out its bound fails too. Each child reports the validation layer's messages until then,
//! which must be none. The parent prints each exit's time, with its wall time, that wait, and how
//! long the process took to end after its handlers. Each test that checks that time runs alone in
//! the suite (`.config/nextest.toml`): an exit's time holds the drivers' exit handlers, which
//! other processes' GPU work slows (NVIDIA's `vkDestroyDevice` took 54 ms alone and up to 2 s
//! beside other lanes on an RTX 3090).
//!
//! Metal's answers (EV `RT4-metal-teardown`, an M4 with macOS 27.0.1, each case without and with
//! `MTL_DEBUG_LAYER=1`): every such exit was clean, in 15 to 45 ms, but one: a dropped command
//! buffer waiting for a plain `MTLEvent` held the exit back 5.03 s, which the runtime does not
//! copy. A dropped command buffer that waits for a shared event stays pending, and completes,
//! its handler running, once the event is signalled
//! ([`a_dropped_command_buffer_completes_once_its_event_is_signalled`]).
//!
//! On a CPU device the work submitted before an exit runs on the process's own threads, so the
//! exit waits for it before the destructors run ([`an_exit_waits_for_a_cpu_device_s_work`]):
//! measured on lavapipe, its queue thread compiling a dispatch's shader during the destructors
//! ended the process in the heap's checks, in 1 to 2 exits of 100 while a host step waited.

use super::indirect_host_steps::Setup;
use super::{run_queue_waits, validated_device, Messages, ScratchCache};
use crate::gpu::{DeviceType, TEST_DEVICE};
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, Deallocator, Device, DeviceKind,
    EventRef, HostMemory, ResourceOptions, SharedEvent, SharedEventListener,
};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::PathBuf;
use std::process::{Command, ExitStatus, Stdio};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::sync::{mpsc, Mutex, OnceLock, PoisonError};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// The longest an exit may take, from the child's last line to the end of its last exit handler
/// ([`exit_end`]), or, where the exit waits for a CPU device's work that ends later, from that
/// work's end: the work runs for as long as the host's load lets it ([`CPU_WORK_TIME`]), and the
/// exit's own part does not. Less the time the exiting thread waited for a CPU meanwhile, which
/// is the host's load. What follows the handlers, the libraries' destructors and the kernel's
/// teardown of the process, is not the exit's work: the parent prints it.
const EXIT_LIMIT: Duration = Duration::from_millis(1500);

/// The longest a child may run, or a command buffer of these tests take.
const WAIT: Duration = Duration::from_secs(60);

/// The environment variable that names the case to [`child`].
const CASE_VAR: &str = "M2V_LANE_EXIT_CASE";

/// The environment variable that names the shader cache directory to [`child`].
const CACHE_VAR: &str = "M2V_LANE_EXIT_CACHE_DIR";

/// The line a child reports on, right before it exits.
const REPORT: &str = "RUNTIME-EXIT";

/// The bytes one fill of the in-flight case writes, and how many fills it encodes: enough work
/// that the device still runs it when a handler exits (4 GiB written on a GPU), and on a CPU
/// device, which writes it at memory speed, 64 MiB.
fn in_flight() -> (u64, usize) {
    match TEST_DEVICE {
        DeviceType::Gpu => (64 << 20, 64),
        DeviceType::Cpu => (16 << 20, 4),
    }
}

/// The bytes one fill of the CPU work case writes.
const CPU_WORK_FILL: u64 = 64 << 20;

/// How long the CPU work case's work runs, as one fill's time measured before it predicts: far
/// longer than the exit's own teardown (about 50 ms on lavapipe), so an exit that did not wait for
/// it finds it running, and far shorter than the hook's bound of 2 s, within which the hook waits
/// for it. A host's load stretches it: at a load of 30 to 45, 400 ms of fills ran for 1.5 s.
const CPU_WORK_TIME: Duration = Duration::from_millis(200);

/// The fill byte of [`fill`].
const FILLED: u8 = 0x5a;

/// The marker the CPU work case fills after its work, which [`read_the_marker`] reads at exit: set
/// on a CPU device only. Registered before the device, the handler runs after the driver's own
/// exit-time destructors as well as the device's hook, and a GPU driver's may unmap what it mapped:
/// on the RTX 3090 (NVIDIA 615.71.09) the read of a marker it had mapped ended the process with
/// `SIGSEGV` in every exit (EV `SWP-exit3090-m2`, 3 of 3, the fault at the mapping's first byte in
/// the handler). A CPU device's memory is the process's own, which no destructor unmaps, and only
/// a CPU device's exit waits for the work the marker follows.
static MARKER: AtomicPtr<u8> = AtomicPtr::new(std::ptr::null_mut());

/// A point the exit's own time may count from ([`exit_end`]): when, in milliseconds since the
/// epoch, and each of the process's threads' wait for a CPU by then, by thread id, where Linux
/// counts it (`super::run_queue_waits`).
struct Since {
    ms: u128,
    waits: BTreeMap<u64, Duration>,
}

impl Since {
    /// Now: the instant read before the waits, so that a wait between the two stays in the exit's
    /// time.
    fn now() -> Since {
        let ms = now_ms();
        Since {
            ms,
            waits: run_queue_waits(),
        }
    }
}

/// The child's last line ([`report`]).
static REPORTED: Mutex<Option<Since>> = Mutex::new(None);

/// When the marker's fill was first seen complete ([`watch_the_marker`], else
/// [`read_the_marker`]): where the CPU work the exit waits for ended.
static WORK_ENDED: Mutex<Option<Since>> = Mutex::new(None);

/// When the CPU work the exit waits for ended ([`WORK_ENDED`]), in milliseconds since the epoch:
/// noted now where it was not yet.
fn work_ended() -> u128 {
    WORK_ENDED
        .lock()
        .unwrap_or_else(PoisonError::into_inner)
        .get_or_insert_with(Since::now)
        .ms
}

/// Watch the marker at `at` from a thread of its own, which goes on through the exit, and note in
/// [`WORK_ENDED`] when its fill is first seen complete: where the work the exit waits for ended,
/// which [`exits_cleanly`] counts the exit's own time from.
fn watch_the_marker(at: NonNull<u8>) {
    let at = at.as_ptr() as usize;
    std::thread::spawn(move || {
        let at = at as *const u8;
        loop {
            // SAFETY: the marker's 16 shared bytes, mapped for the process's life (as
            // `read_the_marker` reads them); read volatile, as the device writes them meanwhile.
            let written = (0..16).all(|i| unsafe { at.add(i).read_volatile() } == FILLED);
            if written {
                work_ended();
                return;
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    });
}

/// An exit handler the CPU work case registers before it makes its device, so that it runs after
/// the device's exit hook: print whether the marker's fill, submitted after the work, completed
/// (always `written=0` without a marker, on a GPU), when the work ended (`seen_ms`: when
/// [`watch_the_marker`] saw the fill complete, else now; 0 where it is not written), and whether
/// the runtime answers that the process is exiting (`process_exiting`, which the hook's run makes
/// true on every device).
extern "C" fn read_the_marker() {
    let at = MARKER.load(Ordering::SeqCst);
    // SAFETY: the marker's 16 shared bytes of a CPU device, mapped for the process's life: the
    // case keeps the buffer, the exit drops nothing, and the driver's memory is the process's.
    let written = !at.is_null()
        && unsafe { std::slice::from_raw_parts(at, 16) }
            .iter()
            .all(|&b| b == FILLED);
    let exiting = u8::from(metal2vulkan_runtime::process_exiting());
    // Where the watcher has not run since the fill completed, the work ended by now at the latest.
    let seen_ms = match written {
        true => work_ended(),
        false => 0,
    };
    let line = format!(
        "{REPORT}-MARKER written={} exiting={exiting} seen_ms={seen_ms}\n",
        u8::from(written)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// What the CPU work case leaves for [`release_the_gate`]: the event its work waits for at 1,
/// which nothing else sets, and the marker's command buffer, committed after the work on its
/// queue.
struct Gate {
    event: SharedEvent,
    marking: CommandBuffer,
}

/// The CPU work case's [`Gate`], set once both command buffers are committed.
static GATE: OnceLock<Gate> = OnceLock::new();

/// The longest [`release_the_gate`] waits for the gate's work to be submitted once it opened the
/// gate: a submission takes milliseconds on a loaded host, so a wait past it is a failure the
/// parent reports (`scheduled=0`), not an exit held until the parent's [`WAIT`] kills the child.
const SUBMISSION_WAIT: Duration = Duration::from_secs(10);

/// An exit handler the CPU work case registers last, right before it exits, so that it runs first
/// in the exit, before the device's hook: note whether the marker's command buffer was still held
/// (`held`: `Committed`, its work behind the closed gate, so neither had run when the exit began),
/// and whether the hook had run (`hook_ran`, which would leave the gate's work pending); then open
/// the gate, wait at most [`SUBMISSION_WAIT`] until the marker's command buffer is submitted, the
/// work's before it (`scheduled`), and note whether it still runs as the hook comes next
/// (`running`).
extern "C" fn release_the_gate() {
    let Some(gate) = GATE.get() else {
        return;
    };
    let held = gate.marking.status() == CommandBufferStatus::Committed;
    let hook_ran = metal2vulkan_runtime::process_exiting();
    gate.event.set_signaled_value(1);
    let deadline = Instant::now() + SUBMISSION_WAIT;
    while gate.marking.status() == CommandBufferStatus::Committed && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(1));
    }
    let status = gate.marking.status();
    let scheduled = matches!(
        status,
        CommandBufferStatus::Scheduled | CommandBufferStatus::Completed
    );
    let running = status == CommandBufferStatus::Scheduled;
    let line = format!(
        "{REPORT}-RELEASE held={} hook_ran={} scheduled={} running={}\n",
        u8::from(held),
        u8::from(hook_ran),
        u8::from(scheduled),
        u8::from(running)
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The command buffers the submitting case commits at once.
const COMMITS: usize = 256;

/// The queues the submitting case's command buffers go round: each holds at most 64 uncompleted,
/// as Metal's default queue does, and these hold 32 each.
const COMMIT_QUEUES: usize = 8;

/// The time now, in milliseconds since the epoch: the clock both processes read.
fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_millis())
}

/// Report the case's end and the layer's messages so far, then exit with status 0.
fn exit_now(case: &str, messages: &Messages) -> ! {
    report(case, messages);
    std::process::exit(0)
}

/// Print the line the parent reads, with how many of the layer's host-wait gap's messages
/// passed (`candidate::runtime_case::Messages`; each printed a `LAYER-GAP` line to stderr).
fn report(case: &str, messages: &Messages) {
    let layer_gaps = messages.layer_gaps();
    let messages = messages.take();
    for message in &messages {
        println!("{REPORT}-MESSAGE {message:?}");
    }
    let reported = Since::now();
    println!(
        "{REPORT} {case} at_ms={} messages={} layer_gaps={layer_gaps}",
        reported.ms,
        messages.len()
    );
    *REPORTED.lock().unwrap_or_else(PoisonError::into_inner) = Some(reported);
}

/// An exit handler every child registers first, before its device, so that it runs last of the
/// exit's handlers: print when the handlers ended (`at_ms`), and how long the exiting thread
/// waited for a CPU since the exit's own time began (`queued_us`), which is the host's load and
/// not the exit's. That time begins at the end of the CPU work the exit waits for where the work
/// ended after the child's last line, else at that line ([`exits_cleanly`]). The wait is Linux's
/// count for the thread (`super::run_queue_waits`), read before the instant, so a wait between the
/// two stays in the exit's time; where Linux gives none, or the thread was not yet running at that
/// line, it is 0.
extern "C" fn exit_end() {
    let queued = exit_thread_queued().unwrap_or_default();
    let line = format!(
        "{REPORT}-END at_ms={} queued_us={}\n",
        now_ms(),
        queued.as_micros()
    );
    // SAFETY: a write of the line's bytes to standard output.
    unsafe { libc::write(1, line.as_ptr().cast(), line.len()) };
}

/// The calling thread's wait for a CPU since the exit's own time began ([`exit_end`]).
fn exit_thread_queued() -> Option<Duration> {
    let reported = REPORTED.lock().unwrap_or_else(PoisonError::into_inner);
    let work_ended = WORK_ENDED.lock().unwrap_or_else(PoisonError::into_inner);
    let since = match (reported.as_ref(), work_ended.as_ref()) {
        (Some(reported), Some(ended)) if ended.ms > reported.ms => ended,
        (Some(reported), _) => reported,
        (None, _) => return None,
    };
    let thread = std::fs::read_link("/proc/thread-self").ok()?;
    let thread: u64 = thread.file_name()?.to_str()?.parse().ok()?;
    let waited = *run_queue_waits().get(&thread)?;
    Some(waited.saturating_sub(*since.waits.get(&thread)?))
}

/// A command buffer of `queue` that waits for `event` at 1, then fills 16 bytes of a new buffer.
fn waiting(queue: &CommandQueue, event: EventRef<'_>) -> CommandBuffer {
    let buffer = queue
        .device()
        .new_buffer(16, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    command_buffer.encode_wait_for_event(event, 1).unwrap();
    fill(&command_buffer, &buffer, 16);
    command_buffer
}

/// Encode a fill of `buffer`'s first `length` bytes.
fn fill(command_buffer: &CommandBuffer, buffer: &Buffer, length: u64) {
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.fill_buffer(buffer, 0..length, FILLED).unwrap();
    blit.end_encoding().unwrap();
}

/// A command buffer of `queue` that fills 16 bytes of a new buffer.
fn small(queue: &CommandQueue) -> CommandBuffer {
    let buffer = queue
        .device()
        .new_buffer(16, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    fill(&command_buffer, &buffer, 16);
    command_buffer
}

/// A command buffer of `queue` that runs `setup`'s writing kernel, then executes the commands it
/// writes: a batch whose host step reads them at its gate.
fn stepping(queue: &CommandQueue, setup: &Setup) -> CommandBuffer {
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    setup.write(&mut encoder);
    setup.execute(&mut encoder);
    encoder.end_encoding().unwrap();
    command_buffer
}

/// Block this thread: the case's exit happens on another one.
fn wait_for_the_exit(case: &str) -> ! {
    std::thread::sleep(WAIT);
    panic!("{case}: the process did not exit");
}

/// One case, run in a child process: the case [`CASE_VAR`] names, on a device whose shader cache
/// is in [`CACHE_VAR`]'s directory, which the parent made and removes.
#[test]
#[ignore = "run in a child process by this module's tests"]
fn child() {
    let case = std::env::var(CASE_VAR).expect("the case to run");
    let cache = PathBuf::from(std::env::var(CACHE_VAR).expect("the cache directory"));
    // SAFETY: a handler with no arguments that never unwinds.
    unsafe { libc::atexit(exit_end) };
    if case == "cpu_work_exit" {
        // SAFETY: a handler with no arguments that never unwinds.
        unsafe { libc::atexit(read_the_marker) };
    }
    let messages = Messages::default();
    let device = validated_device(TEST_DEVICE, &messages, &cache).unwrap();
    let case = case.as_str();
    match case {
        "pending_held_exit" => {
            let queue = device.new_command_queue().unwrap();
            let event = device.new_shared_event().unwrap();
            let command_buffer = waiting(&queue, EventRef::from(&event));
            command_buffer.commit().unwrap();
            assert_eq!(command_buffer.status(), CommandBufferStatus::Committed);
            exit_now(case, &messages);
        }
        "pending_dropped_exit" | "pending_dropped_return" => {
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_shared_event().unwrap();
                waiting(&queue, EventRef::from(&event)).commit().unwrap();
            }
            drop(device);
            if case == "pending_dropped_exit" {
                exit_now(case, &messages);
            }
            report(case, &messages);
        }
        "host_step_pending_exit" => {
            // The setup's work once to completion first: its pipelines are made, so no driver's
            // making of one is in progress as the process exits, which the hook would wait for.
            let setup = Setup::new(&device, 2);
            let queue = device.new_command_queue().unwrap();
            let made = stepping(&queue, &setup);
            made.commit().unwrap();
            made.wait_until_completed_for(WAIT).unwrap();
            assert_eq!(made.status(), CommandBufferStatus::Completed);
            // The second command buffer's batch waits at its gate for the fills before it.
            let (length, fills) = in_flight();
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let filling = queue.command_buffer().unwrap();
            for _ in 0..fills {
                fill(&filling, &buffer, length);
            }
            let stepping = stepping(&queue, &setup);
            filling.commit().unwrap();
            stepping.commit().unwrap();
            exit_now(case, &messages);
        }
        "cpu_work_exit" => {
            // Long work held at a gate, then a fill of a shared marker after it on the same queue;
            // the process exits with both held, and `release_the_gate`, first in the exit, opens
            // the gate and waits until both are submitted: the device runs the work as the
            // process exits, whatever the host's load did before.
            let queue = device.new_command_queue().unwrap();
            let length = CPU_WORK_FILL;
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            // As many fills as run for `CPU_WORK_TIME` by one fill's time, 8 to 64.
            let one = queue.command_buffer().unwrap();
            fill(&one, &buffer, length);
            let start = Instant::now();
            one.commit().unwrap();
            one.wait_until_completed().unwrap();
            let each = start.elapsed().as_nanos().max(1);
            let fills = usize::try_from(CPU_WORK_TIME.as_nanos() / each)
                .unwrap_or(64)
                .clamp(8, 64);
            let marker = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let event = device.new_shared_event().unwrap();
            let work = queue.command_buffer().unwrap();
            work.encode_wait_for_event(EventRef::from(&event), 1)
                .unwrap();
            for _ in 0..fills {
                fill(&work, &buffer, length);
            }
            let marking = queue.command_buffer().unwrap();
            fill(&marking, &marker, 16);
            work.commit().unwrap();
            marking.commit().unwrap();
            let cpu = device.properties().kind == DeviceKind::Cpu;
            if cpu {
                let at = marker.contents().expect("a shared buffer's contents");
                MARKER.store(at.as_ptr(), Ordering::SeqCst);
                watch_the_marker(at);
            }
            println!("{REPORT}-WORK {case} cpu={}", u8::from(cpu));
            assert!(
                GATE.set(Gate { event, marking }).is_ok(),
                "one gate a process"
            );
            // Registered after every runtime call of the case, so after the device's hook.
            // SAFETY: a handler with no arguments that never unwinds.
            unsafe { libc::atexit(release_the_gate) };
            exit_now(case, &messages);
        }
        "pending_plain_event_exit" => {
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_event().unwrap();
                waiting(&queue, EventRef::from(&event)).commit().unwrap();
            }
            drop(device);
            exit_now(case, &messages);
        }
        "completed_handler_exits" => {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = small(&queue);
            let in_handler = messages.clone();
            command_buffer
                .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                    exit_now("completed_handler_exits", &in_handler)
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "scheduled_handler_exits_in_flight" => {
            // The handler runs once the work is submitted, while the GPU runs it and the
            // queue's waiter polls for it.
            let queue = device.new_command_queue().unwrap();
            let (length, fills) = in_flight();
            let buffer = device
                .new_buffer(length, ResourceOptions::private())
                .unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            for _ in 0..fills {
                fill(&command_buffer, &buffer, length);
            }
            let in_handler = messages.clone();
            command_buffer
                .add_scheduled_handler(Box::new(move |_: &CommandBuffer| {
                    exit_now("scheduled_handler_exits_in_flight", &in_handler)
                }))
                .unwrap();
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "deallocator_exits" => {
            let queue = device.new_command_queue().unwrap();
            let page = host_page();
            let memory = anonymous(page);
            let in_deallocator = messages.clone();
            let deallocator: Deallocator = Box::new(move |_: NonNull<u8>, _: u64| {
                exit_now("deallocator_exits", &in_deallocator)
            });
            // SAFETY: the pages stay mapped for the life of the process.
            let made = unsafe {
                device.new_buffer_with_bytes_no_copy(
                    HostMemory::new(memory),
                    page as u64,
                    ResourceOptions::default(),
                    Some(deallocator),
                )
            };
            let buffer = match made {
                Ok(buffer) => buffer,
                Err(error) => {
                    // A device that refuses the import typed: nothing to release.
                    println!("{REPORT}-SKIPPED {case}: {error}");
                    exit_now(case, &messages);
                }
            };
            let command_buffer = queue.command_buffer().unwrap();
            fill(&command_buffer, &buffer, 64);
            // The command buffer holds the last reference: its release runs the deallocator on
            // a handler thread.
            drop(buffer);
            command_buffer.commit().unwrap();
            wait_for_the_exit(case);
        }
        "listener_exits" => {
            let event = device.new_shared_event().unwrap();
            let listener = SharedEventListener::new().unwrap();
            let in_listener = messages.clone();
            event.notify_listener(
                &listener,
                1,
                Box::new(move |_: &SharedEvent, _| exit_now("listener_exits", &in_listener)),
            );
            event.set_signaled_value(1);
            wait_for_the_exit(case);
        }
        "submitting_exit" => {
            // Many command buffers encoded first, then committed at once: the scheduler is still
            // submitting them, and the queues' waiters polling, when the program exits right
            // after. (A commit calls no Vulkan function; the program's own Vulkan calls during
            // its exit would be its own race.)
            let queues: Vec<CommandQueue> = (0..COMMIT_QUEUES)
                .map(|_| device.new_command_queue().unwrap())
                .collect();
            let buffer = device.new_buffer(16, ResourceOptions::shared()).unwrap();
            let command_buffers: Vec<CommandBuffer> = (0..COMMITS)
                .map(|at| {
                    let command_buffer = queues[at % COMMIT_QUEUES].command_buffer().unwrap();
                    fill(&command_buffer, &buffer, 16);
                    command_buffer
                })
                .collect();
            for command_buffer in &command_buffers {
                command_buffer.commit().unwrap();
            }
            exit_now(case, &messages);
        }
        "status_learns_exit" => {
            let command_buffer = small(&device.new_command_queue().unwrap());
            command_buffer.commit().unwrap();
            drop(device);
            let deadline = std::time::Instant::now() + WAIT;
            while command_buffer.status() != CommandBufferStatus::Completed {
                assert!(std::time::Instant::now() < deadline, "never completed");
                std::thread::yield_now();
            }
            drop(command_buffer);
            exit_now(case, &messages);
        }
        "event_learns_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let queue = device.new_command_queue().unwrap();
                let event = device.new_shared_event().unwrap();
                let listener = SharedEventListener::new().unwrap();
                event.notify_listener(
                    &listener,
                    1,
                    Box::new(move |_: &SharedEvent, _| {
                        let _ = told.send(());
                    }),
                );
                let command_buffer = small(&queue);
                command_buffer
                    .encode_signal_event(EventRef::from(&event), 1)
                    .unwrap();
                command_buffer.commit().unwrap();
                drop(device);
                learn.recv_timeout(WAIT).expect("the listener heard");
            }
            exit_now(case, &messages);
        }
        "handler_learns_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let command_buffer = small(&device.new_command_queue().unwrap());
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        let _ = told.send(());
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
                drop(device);
                learn.recv_timeout(WAIT).expect("the handler ran");
            }
            exit_now(case, &messages);
        }
        "handler_drops_last_exit" => {
            let (told, learn) = mpsc::channel();
            {
                let queue = device.new_command_queue().unwrap();
                let command_buffer = small(&queue);
                let (kept_queue, kept_buffer) = (queue.clone(), command_buffer.clone());
                command_buffer
                    .add_completed_handler(Box::new(move |_: &CommandBuffer| {
                        drop((kept_queue, kept_buffer));
                        let _ = told.send(());
                    }))
                    .unwrap();
                command_buffer.commit().unwrap();
            }
            drop(device);
            learn.recv_timeout(WAIT).expect("the handler ran");
            exit_now(case, &messages);
        }
        other => panic!("no case {other}"),
    }
}

/// The host's page size.
fn host_page() -> usize {
    // SAFETY: `sysconf` reads a system constant.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    usize::try_from(page).expect("a page size")
}

/// `len` fresh private anonymous bytes, never unmapped.
fn anonymous(len: usize) -> NonNull<u8> {
    // SAFETY: a fresh mapping.
    let at = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_PRIVATE | libc::MAP_ANON,
            -1,
            0,
        )
    };
    assert_ne!(at, libc::MAP_FAILED, "mmap");
    NonNull::new(at.cast()).expect("a mapping")
}

/// What a child process did.
struct Exited {
    status: ExitStatus,
    stdout: String,
    stderr: String,
    /// When the parent saw it end, in milliseconds since the epoch.
    ended_ms: u128,
}

/// Run `case` in a child process, for at most [`WAIT`] (then it is killed, and this fails).
fn run_child(case: &str) -> Exited {
    let cache = ScratchCache::new().unwrap();
    let mut child = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "runtime_lane::exits::child",
            "--ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CASE_VAR, case)
        .env(CACHE_VAR, cache.path())
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let read = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut text = String::new();
            let _ = pipe.read_to_string(&mut text);
            text
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let pid = child.id();
    let (ended, end) = mpsc::channel();
    std::thread::spawn(move || {
        let status = child.wait();
        let _ = ended.send((status, now_ms()));
    });
    let Ok((status, ended_ms)) = end.recv_timeout(WAIT) else {
        // SAFETY: a signal to the child this test started, which has not been reaped.
        unsafe { libc::kill(pid as libc::pid_t, libc::SIGKILL) };
        panic!("{case}: the child did not end within {WAIT:?}");
    };
    Exited {
        status: status.unwrap(),
        stdout: stdout.join().unwrap(),
        stderr: stderr.join().unwrap(),
        ended_ms,
    }
}

/// The first line of `exited`'s standard output from `start` on (the harness may print the test's
/// name on the same line first).
fn stdout_line<'a>(exited: &'a Exited, start: &str) -> Option<&'a str> {
    exited
        .stdout
        .lines()
        .find_map(|line| Some(&line[line.find(start)?..]))
}

/// Run each of `cases` in a child process, and check each exit: status 0, no signal, no
/// validation message, and its exit handlers ended ([`exit_end`]'s `at_ms`; the process's end
/// where the child printed none) within [`EXIT_LIMIT`] of the child's last line, or of the end of
/// the CPU device's work the exit waits for where that is later (the marker's `seen_ms`), less the
/// time the exiting thread waited for a CPU since (`queued_us`; none where the child gave no
/// count). What each child did, in order.
fn exits_cleanly(cases: &[&str]) -> Vec<Exited> {
    let mut all = Vec::new();
    for &case in cases {
        let exited = run_child(case);
        let Exited {
            status,
            stdout,
            stderr,
            ended_ms,
        } = &exited;
        let output = || format!("stdout:\n{stdout}\nstderr:\n{stderr}");
        #[cfg(unix)]
        {
            use std::os::unix::process::ExitStatusExt;
            assert_eq!(
                status.signal(),
                None,
                "{case}: ended by a signal\n{}",
                output()
            );
        }
        assert!(status.success(), "{case}: {status}\n{}", output());
        let line = stdout_line(&exited, &format!("{REPORT} {case} "))
            .unwrap_or_else(|| panic!("{case}: no report\n{}", output()));
        let field = |name: &str| -> u128 {
            line.split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))
                .and_then(|value| value.parse().ok())
                .unwrap_or_else(|| panic!("{case}: no {name} in {line}"))
        };
        assert_eq!(
            field("messages"),
            0,
            "{case}: the layer reported\n{}",
            output()
        );
        let at_ms = field("at_ms");
        // The work the exit waits for runs as long as the host's load lets it: the exit's own
        // time runs from its end.
        let work_ended_ms = stdout_line(&exited, &format!("{REPORT}-MARKER "))
            .and_then(|line| {
                line.split_whitespace()
                    .find_map(|pair| pair.strip_prefix("seen_ms="))?
                    .parse::<u128>()
                    .ok()
            })
            .unwrap_or(0);
        let from_ms = at_ms.max(work_ended_ms);
        let end = stdout_line(&exited, &format!("{REPORT}-END "));
        let end_field = |name: &str| -> Option<u128> {
            end?.split_whitespace()
                .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
                .parse()
                .ok()
        };
        let handled_ms = end_field("at_ms").unwrap_or(*ended_ms);
        let wall = Duration::from_millis(handled_ms.saturating_sub(from_ms) as u64);
        let queued = end_field("queued_us").map_or(Duration::ZERO, |us| {
            Duration::from_micros(u64::try_from(us).unwrap_or(u64::MAX))
        });
        let took = wall.saturating_sub(queued);
        let after = Duration::from_millis(ended_ms.saturating_sub(handled_ms) as u64);
        let work = Duration::from_millis(from_ms.saturating_sub(at_ms) as u64);
        assert!(
            took < EXIT_LIMIT,
            "{case}: the exit took {took:?} ({wall:?} of wall time, less {queued:?} the exiting \
             thread waited for a CPU; the process ended {after:?} after its handlers){}\n{}",
            match work_ended_ms > at_ms {
                true => format!(
                    " after the CPU device's work it waited for ended ({work:?} after the last \
                     line)"
                ),
                false => String::new(),
            },
            output()
        );
        let skipped = stdout.contains(&format!("{REPORT}-SKIPPED {case}"));
        println!(
            "runtime exit, {case}: {} ms ({} ms of wall time, less {} ms the exiting thread \
             waited for a CPU; the process ended {} ms after its handlers){}{}",
            took.as_millis(),
            wall.as_millis(),
            queued.as_millis(),
            after.as_millis(),
            if work_ended_ms > at_ms {
                format!(
                    " after the work it waited for, which ended {} ms after the last line",
                    work.as_millis()
                )
            } else {
                String::new()
            },
            if skipped {
                " (host memory refused)"
            } else {
                ""
            }
        );
        // The child exits before its message list goes, so its count is printed here.
        for line in stderr.lines().filter(|line| line.starts_with("LAYER-GAP ")) {
            println!("{line}");
        }
        let layer_gaps = field("layer_gaps");
        if layer_gaps != 0 {
            println!("LAYER-GAP count {case}: {layer_gaps} layer gap message(s) passed");
        }
        all.push(exited);
    }
    all
}

/// A command buffer still waiting for an event at exit (held, dropped, or for a plain event that
/// only the GPU can signal) stays pending, and the process exits at once, by `exit` or at the end
/// of `main`. Metal's exits were clean in 15 to 30 ms, and 5.03 s for the plain event.
#[test]
fn pending_work_stays_pending_at_exit() {
    exits_cleanly(&[
        "pending_held_exit",
        "pending_dropped_exit",
        "pending_dropped_return",
        "pending_plain_event_exit",
    ]);
}

/// An exit from the program's code on a runtime thread, which holds no section of the exit
/// guard: a completed handler, a scheduled handler while the GPU still runs its work and the
/// queue's waiter polls for it, a no-copy buffer's deallocator run by a command buffer's
/// release, and a listener's notification. The hook does not wait for the exiting thread.
#[test]
fn exits_from_the_runtime_s_threads_are_clean() {
    exits_cleanly(&[
        "completed_handler_exits",
        "scheduled_handler_exits_in_flight",
        "deallocator_exits",
        "listener_exits",
    ]);
}

/// An exit while a batch waits at its host step's gate for the work before it: the scheduler ends
/// at its next section, and what is pending stays pending. The same work ran once to completion
/// before, so that its pipelines are made and no driver's making of one is in progress at the
/// exit: the hook waits for such a making, which the runtime's
/// `the_hook_waits_past_its_bound_for_a_making_in_progress` tests.
#[test]
fn an_exit_while_a_host_step_waits_is_clean() {
    exits_cleanly(&["host_step_pending_exit"]);
}

/// An exit while a CPU device runs work submitted before it (fills it runs for about
/// [`CPU_WORK_TIME`], then a fill of a shared marker) waits for that work: the marker is written
/// by the time an exit handler registered before the device runs, after the device's hook. That
/// is the interleaving of an exit with a dispatch in flight on lavapipe, made certain: the work
/// waits at a gate until the exit has begun, where a handler that runs before the device's hook
/// ([`release_the_gate`]) opens it and waits until the work is submitted, so it runs as the hook
/// starts, for far longer than the exit's own teardown. Without the wait the marker was unwritten
/// in 4 of 20 exits with a fixed 512 MiB of fills, which lavapipe ran within the teardown's 50 ms
/// in the other 16. The exit's own time counts from the work's end, which a thread watching the
/// marker sees: the work's own time follows the host's load (1.585 s from the last line to the end
/// of an exit beside other lanes at a load of 30 to 45, 47 to 58 ms alone). On a GPU the hook does
/// not wait (as Metal's exit does not), and the exit, with the work started in it, is clean. The
/// gate makes the work's state as the exit begins certain on every device and at any load: before
/// it, the child read the status once the work was submitted, and on an RTX 3090 (NVIDIA
/// 615.71.09) beside 24 CPU spinners the GPU finished the work (15 to 18 fills of 64 MiB, sized by
/// one fill's 11 to 13 ms round trip there) in the 2 to 3 ms the child took to see its submission
/// and read its status, in 2 of 20 runs; and in 2 batches beside other lanes.
#[test]
fn an_exit_waits_for_a_cpu_device_s_work() {
    let case = "cpu_work_exit";
    let exited = exits_cleanly(&[case]).remove(0);
    let field = |line: &str, name: &str| -> Option<u8> {
        stdout_line(&exited, line)?
            .split_whitespace()
            .find_map(|pair| pair.strip_prefix(name)?.strip_prefix('='))?
            .parse()
            .ok()
    };
    let what = || format!("stdout:\n{}", exited.stdout);
    let release = format!("{REPORT}-RELEASE ");
    assert_eq!(
        field(&release, "held"),
        Some(1),
        "the work was not held at its gate when the exit began: {}",
        what()
    );
    assert_eq!(
        field(&release, "hook_ran"),
        Some(0),
        "the device's hook ran before the gate opened: {}",
        what()
    );
    assert_eq!(
        field(&release, "scheduled"),
        Some(1),
        "the work was not submitted once the gate opened: {}",
        what()
    );
    assert_eq!(
        field(&format!("{REPORT}-MARKER "), "exiting"),
        Some(1),
        "the runtime did not answer that the process is exiting: {}",
        what()
    );
    if field(&format!("{REPORT}-WORK {case} "), "cpu") == Some(1) {
        assert_eq!(
            field(&release, "running"),
            Some(1),
            "the work completed before the device's hook began: {}",
            what()
        );
        assert_eq!(
            field(&format!("{REPORT}-MARKER "), "written"),
            Some(1),
            "the exit did not wait for the CPU device's work: {}",
            what()
        );
    }
}

/// An exit right after committing many command buffers: the scheduler is still submitting them,
/// and the queues' waiters polling.
#[test]
fn an_exit_while_submitting_is_clean() {
    exits_cleanly(&["submitting_exit"]);
}

/// A program that learns its work finished from the status, an event's listener or a completed
/// handler (or whose handler drops the last references) and then drops everything and exits at
/// once: the device's teardown runs on a runtime thread, and the exit waits for it or leaves the
/// device to the process's end. Metal's exits were clean in 25 to 45 ms.
#[test]
fn exits_right_after_learning_of_completion_are_clean() {
    exits_cleanly(&[
        "status_learns_exit",
        "event_learns_exit",
        "handler_learns_exit",
        "handler_drops_last_exit",
    ]);
}

/// A command buffer whose program dropped every handle but the shared event it waits for stays
/// pending, and completes once the event is signalled: its completed handler runs, and sees the
/// status `Completed` (Metal's answer: status 4).
#[test]
fn a_dropped_command_buffer_completes_once_its_event_is_signalled() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device: Device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let event = device.new_shared_event().unwrap();
        let (told, learn) = mpsc::channel();
        {
            let queue = device.new_command_queue().unwrap();
            let command_buffer = waiting(&queue, EventRef::from(&event));
            command_buffer
                .add_completed_handler(Box::new(move |command_buffer: &CommandBuffer| {
                    let _ = told.send(command_buffer.status());
                }))
                .unwrap();
            command_buffer.commit().unwrap();
        }
        drop(device);
        assert!(
            learn.recv_timeout(Duration::from_millis(200)).is_err(),
            "the handler ran before the signal"
        );
        event.set_signaled_value(1);
        assert_eq!(learn.recv_timeout(WAIT), Ok(CommandBufferStatus::Completed));
        // The device goes with the event's last handle, once the runtime let go of the command
        // buffer.
        drop(event);
        let deadline = std::time::Instant::now() + WAIT;
        while messages.holders() > 1 {
            assert!(
                std::time::Instant::now() < deadline,
                "the device never went"
            );
            std::thread::sleep(Duration::from_millis(10));
        }
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}

/// Devices made again in a long-lived process register the exit hook no more while what they
/// load stays loaded: with one device held, whose instance keeps the Vulkan loader, the driver and
/// the layers loaded, 16 devices made and dropped after two that warm up add no registration at a
/// device, and the sections' registrations stay within their bound.
#[test]
fn devices_made_again_register_the_exit_hook_no_more() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let held: Device = validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        for _ in 0..2 {
            drop(validated_device(TEST_DEVICE, &messages, cache.path()).unwrap());
        }
        let before = metal2vulkan_runtime::exit_registrations();
        for _ in 0..16 {
            drop(validated_device(TEST_DEVICE, &messages, cache.path()).unwrap());
        }
        let after = metal2vulkan_runtime::exit_registrations();
        println!("runtime exit registrations: {before:?}, then after 16 devices {after:?}");
        assert_eq!(
            after.at_devices, before.at_devices,
            "devices made again registered the hook"
        );
        assert!(
            after.total - after.at_devices <= after.section_bound,
            "the sections registered past their bound: {after:?}"
        );
        drop(held);
    }
    let messages = messages.take();
    assert!(
        messages.is_empty(),
        "the validation layer reported: {messages:#?}"
    );
}
