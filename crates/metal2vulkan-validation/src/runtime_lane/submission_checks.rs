//! Device tests of submission checks (`CommandBuffer::set_submission_check`, a runtime
//! extension) on the device the cfg selects, under the validation layer.
//!
//! The consumer emulates Apple's detach: Metal fails a submission (as it splits the command
//! buffer) that uses a resource whose memory was taken away by the time it is submitted, with
//! code 8, and runs every other (measured by the consumer on an M3 Max, macOS 26.7). Its check
//! fails a submission whose used resources hold a detached object, by `ObjectId`.
//!
//! - **Detached behind a wait.** Each probe shape (`failed_commits::SHAPES`), led by an event
//!   wait, is committed with its resource attached, then detached, then the wait is signalled:
//!   the submission that uses it fails alone, as `commit_failed` with that set fails it. The check
//!   sees each of Metal's submissions once.
//! - **Attached again behind a wait.** The same shapes committed while detached, attached again
//!   before the signal: everything runs, and the command buffer completes.
//! - **Reached through an indirect command alone.** A buffer an indirect command binds, which
//!   no encoder makes resident, is the submission's from the commit on; detached behind a wait,
//!   that submission fails.
//! - **A GPU signal in flight**. A submission whose `SharedEvent` wait another command buffer's
//!   GPU signal meets is checked only once that signal landed (the check reads a companion
//!   `SharedEvent` the signaller signals first); detached while the signaller runs, it fails. A
//!   `SharedEvent` the host signals while a GPU signal of it is in flight is checked without
//!   waiting for the GPU.
//! - **Metal's rule for `Event` waits** (measured by the consumer on an M3 Max, macOS 27.0.1). A
//!   command buffer waiting for `Event`s is scheduled, checked and counted in flight at the first
//!   commit after which each event it waits for has a committed signaller of a value at least the
//!   wait's, or holds such a value: behind a signaller committed first, at its own commit; behind
//!   none, held until one is; behind a lower signal, held until one reaches its value; waiting
//!   for two events, held until each has a signaller; a higher signal counts. Scheduled before
//!   the signal landed, it then runs though detached.

use super::degrade_notices::on_device;
use super::failed_commits::{Shape, HELD, KERNEL, SHAPES};
use super::{
    bound_buffers, buffer_with, compute_descriptor, pipeline_state, public_case, read_buffer,
    CaseResources, PublicCase, WAIT,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use metal2vulkan::bind::Dispatch;
use metal2vulkan::metal::IndirectCommandType;
use metal2vulkan::pipeline::LinkedFunctions;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferError, CommandBufferStatus, ComputePipelineState, Device,
    EncoderKind, EventRef, IndirectCommandBufferDescriptor, ObjectId, PipelineOptions,
    ResourceOptions, SharedEvent, Size, Submission, SubmissionCheck,
};
use std::collections::{BTreeSet, HashSet};
use std::num::NonZeroU64;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// The code a submission that uses a detached resource fails with: `OutOfMemory`, as Metal's.
const DETACHED: u64 = 8;

/// The consumer's detach, as a check sees it: the detached objects, and each submission the
/// check saw.
#[derive(Default)]
struct Detach {
    detached: Mutex<HashSet<ObjectId>>,
    seen: Mutex<Vec<(EncoderKind, Range<usize>)>>,
    /// A companion event's value as each check ran ([`Detach::check_reading`]).
    landed: Mutex<Vec<u64>>,
}

impl Detach {
    /// A check that fails, with [`DETACHED`], a submission whose used resources hold a detached
    /// buffer or heap, noting each submission it sees.
    fn check(self: &Arc<Self>) -> SubmissionCheck {
        let detach = Arc::clone(self);
        Arc::new(move |submission: &Submission<'_>| detach.verdict(submission))
    }

    /// [`Detach::check`], which also notes `landed`'s value as it checks.
    fn check_reading(self: &Arc<Self>, landed: &SharedEvent) -> SubmissionCheck {
        let detach = Arc::clone(self);
        let landed = landed.clone();
        Arc::new(move |submission: &Submission<'_>| {
            detach.landed.lock().unwrap().push(landed.signaled_value());
            detach.verdict(submission)
        })
    }

    fn verdict(&self, submission: &Submission<'_>) -> Option<NonZeroU64> {
        self.seen
            .lock()
            .unwrap()
            .push((submission.kind(), submission.encoders()));
        let used = submission.used_resources();
        let detached = self.detached.lock().unwrap();
        let hit = used
            .buffers
            .iter()
            .map(Buffer::object_id)
            .chain(used.heaps.iter().map(|heap| heap.object_id()))
            .any(|id| detached.contains(&id));
        hit.then(|| NonZeroU64::new(DETACHED).unwrap())
    }

    fn detach(&self, buffer: &Buffer) {
        self.detached.lock().unwrap().insert(buffer.object_id());
    }

    fn attach(&self, buffer: &Buffer) {
        self.detached.lock().unwrap().remove(&buffer.object_id());
    }
}

/// Each probe shape led by a wait, committed with a check while its resource is attached, then
/// detached before the wait is signalled: the submission that uses it fails alone, the others
/// run, its signal fires, its handlers run, and it completes with status `Error` and code 8. The
/// check saw each of Metal's submissions once, in order.
#[test]
fn a_submission_detached_behind_a_wait_fails_alone() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = pipeline_state(device, &kernel).unwrap();
        for (shape, failed) in SHAPES {
            let encoded = Shape::encode(device, &kernel, &pipeline, &format!("w{shape}"));
            let failed = encoded.failed_set(failed);
            let submissions = encoded.submissions();
            let detach = Arc::new(Detach::default());
            encoded
                .command_buffer
                .set_submission_check(detach.check())
                .unwrap();
            let seen = encoded.watch_handlers();
            encoded.command_buffer.commit().unwrap();
            detach.detach(&encoded.detached);
            encoded.release_waits();
            encoded.finished(&seen, Some(DETACHED), &failed, &kernel);
            assert_eq!(*detach.seen.lock().unwrap(), submissions, "{shape}");
        }
    });
}

/// Each probe shape led by a wait, committed with a check while its resource is detached, then
/// attached again before the wait is signalled: everything runs, and it completes.
#[test]
fn a_submission_attached_again_behind_a_wait_runs() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = pipeline_state(device, &kernel).unwrap();
        for (shape, _) in SHAPES {
            let encoded = Shape::encode(device, &kernel, &pipeline, &format!("w{shape}"));
            let submissions = encoded.submissions();
            let detach = Arc::new(Detach::default());
            detach.detach(&encoded.detached);
            encoded
                .command_buffer
                .set_submission_check(detach.check())
                .unwrap();
            let seen = encoded.watch_handlers();
            encoded.command_buffer.commit().unwrap();
            detach.attach(&encoded.detached);
            encoded.release_waits();
            encoded.finished(&seen, None, &BTreeSet::new(), &kernel);
            assert_eq!(*detach.seen.lock().unwrap(), submissions, "{shape}");
        }
    });
}

/// A check is refused once the command buffer is committed, changing nothing.
#[test]
fn a_check_is_refused_once_committed() {
    on_device(|device, _| {
        let queue = device.new_command_queue().unwrap();
        let command_buffer = queue.command_buffer().unwrap();
        command_buffer.commit().unwrap();
        let detach = Arc::new(Detach::default());
        assert!(command_buffer.set_submission_check(detach.check()).is_err());
        command_buffer.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
        assert!(detach.seen.lock().unwrap().is_empty());
    });
}

/// A commit returns at once while another command buffer's check runs on the scheduler thread,
/// held there by this test, as Metal's `commit` does, and submits nothing itself: the scheduler
/// thread makes every submission, so the command buffer committed on another queue stays
/// `Committed` until the check returned, and then runs and completes.
#[test]
fn a_commit_made_while_a_check_runs_returns_at_once_and_runs_after_the_check() {
    use std::sync::mpsc;
    use std::time::Duration;
    on_device(|device, _| {
        let fill = |queue: &metal2vulkan_runtime::CommandQueue, buffer: &Buffer| {
            let command_buffer = queue.command_buffer().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(buffer, 0..buffer.length(), 0x5a).unwrap();
            blit.end_encoding().unwrap();
            command_buffer
        };
        let (entered, check_entered) = mpsc::channel();
        let (release, released) = mpsc::channel::<()>();
        let (entered, released) = (Mutex::new(entered), Mutex::new(released));
        let held_check: SubmissionCheck = Arc::new(move |_: &Submission<'_>| {
            let _ = entered.lock().unwrap().send(());
            let _ = released
                .lock()
                .unwrap()
                .recv_timeout(Duration::from_secs(60));
            None
        });
        let checked = device.new_command_queue().unwrap();
        let free = device.new_command_queue().unwrap();
        let (held_bytes, free_bytes) = (
            device.new_buffer(256, ResourceOptions::shared()).unwrap(),
            device.new_buffer(256, ResourceOptions::shared()).unwrap(),
        );
        let held = fill(&checked, &held_bytes);
        held.set_submission_check(held_check).unwrap();
        held.commit().unwrap();
        check_entered
            .recv_timeout(Duration::from_secs(30))
            .expect("the check runs on the scheduler thread");
        let other = fill(&free, &free_bytes);
        other.commit().unwrap();
        std::thread::sleep(Duration::from_millis(50));
        let status = other.status();
        release.send(()).unwrap();
        held.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(held.status(), CommandBufferStatus::Completed);
        assert_eq!(
            status,
            CommandBufferStatus::Committed,
            "submitted while the scheduler thread ran the check"
        );
        other.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(other.status(), CommandBufferStatus::Completed);
        assert_eq!(read_buffer(&free_bytes, 0, 256).unwrap(), vec![0x5a; 256]);
    });
}

/// [`KERNEL`]'s pipeline state, made with `supportIndirectCommandBuffers`.
fn icb_state(device: &Device, kernel: &PublicCase) -> ComputePipelineState {
    let function = device
        .new_library_with_air(&kernel.bitcode)
        .unwrap()
        .new_function_with_name_constant_values(&kernel.compute.entry, &kernel.compute.constants)
        .unwrap();
    let mut descriptor = compute_descriptor(&function, LinkedFunctions::new(), &kernel.compute);
    descriptor.support_indirect_command_buffers = true;
    device
        .new_compute_pipeline_state(&descriptor, &PipelineOptions::default())
        .unwrap()
}

/// A buffer an indirect command binds as [`KERNEL`]'s output, which no encoder makes resident,
/// is the submission's from the commit on (`ended_encoders` lists it then, not before). Between
/// two blits, behind a wait: detached before the signal, that submission fails alone and the
/// output keeps its poison; left attached, the command runs and writes Metal's output.
#[test]
fn a_buffer_reached_only_through_an_indirect_command_fails_its_submission() {
    let kernel = public_case(KERNEL).unwrap();
    on_device(|device, _| {
        let pipeline = icb_state(device, &kernel);
        let compute = &kernel.compute;
        let ComputeOutput::Buffer {
            binding,
            offset,
            length,
        } = compute.output
        else {
            panic!("{KERNEL}: the case's output is no buffer");
        };
        let size = |[x, y, z]: [u32; 3]| Size::new(x.into(), y.into(), z.into());
        for detached in [true, false] {
            let resources = CaseResources::new(device, &kernel).unwrap();
            let buffers = bound_buffers(device, compute, &resources).unwrap();
            let output = buffers[&binding].clone();
            let initial = read_buffer(&output, offset, length).unwrap();
            assert_ne!(initial, kernel.metal_output, "{KERNEL}: no output to see");
            let mut descriptor = IndirectCommandBufferDescriptor::new();
            descriptor.command_types = IndirectCommandType::CONCURRENT_DISPATCH
                .union(IndirectCommandType::CONCURRENT_DISPATCH_THREADS);
            descriptor.max_kernel_buffer_bind_count = 31;
            let icb = device
                .new_indirect_command_buffer(&descriptor, 1, ResourceOptions::shared())
                .unwrap();
            let command = icb.indirect_compute_command_at(0).unwrap();
            command.set_compute_pipeline_state(&pipeline).unwrap();
            for (&slot, buffer) in &buffers {
                command.set_kernel_buffer(Some(buffer), 0, slot).unwrap();
            }
            for (&slot, &threadgroup) in &compute.threadgroup_lengths {
                command
                    .set_threadgroup_memory_length(threadgroup, slot)
                    .unwrap();
            }
            match compute.dispatch {
                Dispatch::Threads {
                    threads,
                    threads_per_group,
                } => command.concurrent_dispatch_threads(size(threads), size(threads_per_group)),
                Dispatch::Threadgroups {
                    groups,
                    threads_per_group,
                } => {
                    command.concurrent_dispatch_threadgroups(size(groups), size(threads_per_group))
                }
                other => panic!("{other:?} is no dispatch of a command"),
            }
            .unwrap();

            let queue = device.new_command_queue().unwrap();
            let gate = device.new_shared_event().unwrap();
            let words = buffer_with(device, &[0; 8]).unwrap();
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .encode_wait_for_event(EventRef::from(&gate), 1)
                .unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 0..4, 0x11).unwrap();
            blit.end_encoding().unwrap();
            let mut encoder = command_buffer.compute_command_encoder().unwrap();
            encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
            encoder.end_encoding().unwrap();
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.fill_buffer(&words, 4..8, 0x22).unwrap();
            blit.end_encoding().unwrap();

            let lists = |command_buffer: &metal2vulkan_runtime::CommandBuffer| {
                command_buffer.ended_encoders()[1]
                    .used_resources
                    .buffers
                    .contains(&output)
            };
            assert!(!lists(&command_buffer), "listed before the commit");
            let detach = Arc::new(Detach::default());
            command_buffer.set_submission_check(detach.check()).unwrap();
            command_buffer.commit().unwrap();
            assert!(lists(&command_buffer), "not listed once committed");
            assert!(command_buffer.used_resources().buffers.contains(&output));
            if detached {
                detach.detach(&output);
            }
            assert!(command_buffer.wait_until_completed_for(HELD).is_err());
            gate.set_signaled_value(1);
            command_buffer.wait_until_completed_for(WAIT).unwrap();
            if detached {
                assert_eq!(command_buffer.status(), CommandBufferStatus::Error);
                assert_eq!(
                    command_buffer.error(),
                    Some(CommandBufferError::Injected(DETACHED))
                );
            } else {
                assert_eq!(command_buffer.status(), CommandBufferStatus::Completed);
            }
            let want = if detached {
                &initial
            } else {
                &kernel.metal_output
            };
            assert_eq!(
                &read_buffer(&output, offset, length).unwrap(),
                want,
                "detached {detached}"
            );
            let mut both = [0x11u8; 8];
            both[4..].fill(0x22);
            assert_eq!(read_buffer(&words, 0, 8).unwrap(), both);
            assert_eq!(
                *detach.seen.lock().unwrap(),
                [
                    (EncoderKind::Blit, 0..1),
                    (EncoderKind::Compute, 1..2),
                    (EncoderKind::Blit, 2..3)
                ]
            );
            drop(resources);
        }
    });
}

/// The bytes each of a signaller's fills writes.
const FILL: u64 = 64 << 20;

/// How many times a GPU-signal test tries, the signaller's fills doubling each time, for the
/// signaller still to run when the host acts.
const ATTEMPTS: u32 = 7;

/// What a GPU-signal test's command buffers use: the waiter's target, the signaller's buffers,
/// a companion event the signaller signals first, and the waiter's check.
struct Parts {
    target: Buffer,
    landed: SharedEvent,
    detach: Arc<Detach>,
    big: Buffer,
    words: Buffer,
}

impl Parts {
    fn new(device: &Device) -> Parts {
        Parts {
            target: buffer_with(device, &[0; 16]).unwrap(),
            landed: device.new_shared_event().unwrap(),
            detach: Arc::new(Detach::default()),
            big: device.new_buffer(FILL, ResourceOptions::private()).unwrap(),
            words: buffer_with(device, &[0; 4]).unwrap(),
        }
    }

    /// A command buffer of a new queue that waits for each of `waits` and fills `target`'s first
    /// word, with a check that reads `landed` ([`Detach::check_reading`]); not committed.
    fn waiter(&self, device: &Device, waits: &[(EventRef<'_>, u64)]) -> CommandBuffer {
        let waiter = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        for &(event, value) in waits {
            waiter.encode_wait_for_event(event, value).unwrap();
        }
        let mut blit = waiter.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.target, 0..4, 0x5a).unwrap();
        blit.end_encoding().unwrap();
        waiter
            .set_submission_check(self.detach.check_reading(&self.landed))
            .unwrap();
        waiter
    }

    /// A command buffer of a new queue that fills a 64-MiB buffer `fills` times, signals
    /// `landed` to 1, fills a word, and signals each of `signals`; not committed. `landed`
    /// reaches 1 no later than `signals` land (one queue's signals land in order), so it reads 0
    /// only while they have not.
    fn signaller(
        &self,
        device: &Device,
        fills: u64,
        signals: &[(EventRef<'_>, u64)],
    ) -> CommandBuffer {
        let signaller = device
            .new_command_queue()
            .unwrap()
            .command_buffer()
            .unwrap();
        let mut blit = signaller.blit_command_encoder().unwrap();
        for fill in 0..fills {
            blit.fill_buffer(&self.big, 0..FILL, fill as u8).unwrap();
        }
        blit.end_encoding().unwrap();
        signaller
            .encode_signal_event(EventRef::from(&self.landed), 1)
            .unwrap();
        let mut blit = signaller.blit_command_encoder().unwrap();
        blit.fill_buffer(&self.words, 0..4, 0x11).unwrap();
        blit.end_encoding().unwrap();
        for &(event, value) in signals {
            signaller.encode_signal_event(event, value).unwrap();
        }
        signaller
    }

    /// The companion event's value as each check of the waiter ran.
    fn checked(&self) -> Vec<u64> {
        self.detach.landed.lock().unwrap().clone()
    }
}

/// One try of a GPU-signal test: a waiter of `waited`'s value 1, committed with a check, then a
/// signaller of it on another queue ([`Parts::signaller`]), committed and scheduled.
struct InFlight {
    waiter: CommandBuffer,
    signaller: CommandBuffer,
    parts: Parts,
}

impl InFlight {
    fn start(device: &Device, waited: &SharedEvent, fills: u64) -> InFlight {
        let parts = Parts::new(device);
        let waiter = parts.waiter(device, &[(EventRef::from(waited), 1)]);
        waiter.commit().unwrap();
        let signaller = parts.signaller(device, fills, &[(EventRef::from(waited), 1)]);
        signaller.commit().unwrap();
        signaller.wait_until_scheduled().unwrap();
        InFlight {
            waiter,
            signaller,
            parts,
        }
    }

    /// Wait for both command buffers; the companion event's value as the waiter's one check ran.
    fn finish(&self) -> u64 {
        self.signaller.wait_until_completed_for(WAIT).unwrap();
        self.waiter.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(self.signaller.status(), CommandBufferStatus::Completed);
        let landed = self.parts.checked();
        assert_eq!(landed.len(), 1, "the waiter's submission is checked once");
        landed[0]
    }
}

/// A submission whose `SharedEvent` wait another command buffer's GPU signal meets is checked
/// only once that signal landed: the companion event the signaller signals first reads 1 in the
/// check, every try. Its buffer detached once the signaller is scheduled, while its signal is in
/// flight (the companion still reads 0 after the detach), the submission fails with code 8 and
/// its fill does not run. A try whose signaller finished before the detach proves nothing of the
/// detach, and the next doubles the signaller's work; one try at least must catch it in flight.
#[test]
fn a_submission_detached_while_a_gpu_signal_is_in_flight_fails() {
    on_device(|device, _| {
        let mut caught = None;
        for attempt in 0..ATTEMPTS {
            let fills = 4u64 << attempt;
            let waited = device.new_shared_event().unwrap();
            let run = InFlight::start(device, &waited, fills);
            run.parts.detach.detach(&run.parts.target);
            let in_flight = run.parts.landed.signaled_value() == 0;
            let at_check = run.finish();
            assert_eq!(at_check, 1, "checked before the GPU signal landed");
            if !in_flight {
                continue;
            }
            assert_eq!(run.waiter.status(), CommandBufferStatus::Error);
            assert_eq!(
                run.waiter.error(),
                Some(CommandBufferError::Injected(DETACHED))
            );
            assert_eq!(read_buffer(&run.parts.target, 0, 4).unwrap(), [0; 4]);
            caught = Some(fills);
            break;
        }
        let fills = caught
            .unwrap_or_else(|| panic!("the signaller never ran at the detach in {ATTEMPTS} tries"));
        println!("IN-FLIGHT caught with {fills} fills of {FILL} bytes");
    });
}

/// One shape of Metal's rule for a command buffer waiting for `Event`s, over two events (by
/// index into them, each signalled to a value): `before`'s signals, by a command buffer
/// completed before the waiter's commit; the waiter's `waits`; `short`'s, by a command buffer
/// committed and completed after the waiter's commit, which leave it held; and `long`'s, by a
/// long signaller ([`Parts::signaller`]) committed last, or first with `long_first`; with no
/// `long` signals, there is none, and the waiter is scheduled at its own commit.
struct EventRule {
    name: &'static str,
    before: &'static [(usize, u64)],
    waits: &'static [(usize, u64)],
    short: &'static [(usize, u64)],
    long: &'static [(usize, u64)],
    long_first: bool,
}

/// The shapes of Metal's rule for `Event` waits.
const EVENT_RULES: &[EventRule] = &[
    EventRule {
        name: "a signaller committed after the waiter",
        before: &[],
        waits: &[(0, 1)],
        short: &[],
        long: &[(0, 1)],
        long_first: false,
    },
    EventRule {
        name: "a signaller committed before the waiter",
        before: &[],
        waits: &[(0, 1)],
        short: &[],
        long: &[(0, 1)],
        long_first: true,
    },
    EventRule {
        name: "a higher signal committed after the waiter",
        before: &[],
        waits: &[(0, 3)],
        short: &[],
        long: &[(0, 5)],
        long_first: false,
    },
    EventRule {
        name: "a higher signal committed before the waiter",
        before: &[],
        waits: &[(0, 3)],
        short: &[],
        long: &[(0, 5)],
        long_first: true,
    },
    EventRule {
        name: "a lower signal, then one of the value",
        before: &[],
        waits: &[(0, 2)],
        short: &[(0, 1)],
        long: &[(0, 2)],
        long_first: false,
    },
    EventRule {
        name: "two events, one signalled, then the other",
        before: &[],
        waits: &[(0, 1), (1, 1)],
        short: &[(0, 1)],
        long: &[(1, 1)],
        long_first: false,
    },
    EventRule {
        name: "an event already at the value",
        before: &[(0, 1)],
        waits: &[(0, 1)],
        short: &[],
        long: &[],
        long_first: false,
    },
];

/// Let the scheduler act on what is committed: an empty command buffer of a new queue,
/// committed and waited for until scheduled (the scheduler's pass that took it looked at every
/// head committed before it), then a moment for a check that pass found due to run.
fn settle(device: &Device) {
    let probe = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    probe.commit().unwrap();
    probe.wait_until_scheduled().unwrap();
    std::thread::sleep(std::time::Duration::from_millis(10));
}

/// A command buffer of a new queue that only signals each of `signals`, committed and waited
/// for.
fn signal_now(device: &Device, signals: &[(EventRef<'_>, u64)]) {
    let signaller = device
        .new_command_queue()
        .unwrap()
        .command_buffer()
        .unwrap();
    for &(event, value) in signals {
        signaller.encode_signal_event(event, value).unwrap();
    }
    signaller.commit().unwrap();
    signaller.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(signaller.status(), CommandBufferStatus::Completed);
}

/// Check that `waiter`, committed while no signaller of `rule`'s long signals is, is held,
/// unchecked; `rule`'s short signals completed meanwhile.
fn held<'a>(
    device: &Device,
    rule: &EventRule,
    (waiter, parts): (&CommandBuffer, &Parts),
    refs: &impl Fn(&[(usize, u64)]) -> Vec<(EventRef<'a>, u64)>,
) {
    let name = rule.name;
    settle(device);
    if !rule.short.is_empty() {
        signal_now(device, &refs(rule.short));
        settle(device);
    }
    assert_eq!(
        waiter.status(),
        CommandBufferStatus::Committed,
        "{name}: scheduled with no signaller of each value committed"
    );
    assert_eq!(
        parts.checked(),
        Vec::<u64>::new(),
        "{name}: checked while held"
    );
}

/// One try of `rule` with a long signaller of `fills` fills: whether that signaller still ran
/// as the waiter read `Scheduled` (always, with none). The waiter is held, unchecked, until the
/// last signal it needs is committed ([`held`]); then it is scheduled and checked once (before
/// that signal landed, where it still ran), and runs though detached as soon as it read
/// `Scheduled`.
fn event_rule_try(device: &Device, rule: &EventRule, fills: u64) -> bool {
    let name = rule.name;
    let events = [device.new_event().unwrap(), device.new_event().unwrap()];
    let refs = |list: &[(usize, u64)]| -> Vec<(EventRef<'_>, u64)> {
        (list.iter())
            .map(|&(at, value)| (EventRef::from(&events[at]), value))
            .collect()
    };
    if !rule.before.is_empty() {
        signal_now(device, &refs(rule.before));
    }
    let parts = Parts::new(device);
    let waiter = parts.waiter(device, &refs(rule.waits));
    let signaller =
        (!rule.long.is_empty()).then(|| parts.signaller(device, fills, &refs(rule.long)));
    match &signaller {
        Some(signaller) if rule.long_first => {
            signaller.commit().unwrap();
            waiter.commit().unwrap();
        }
        Some(signaller) => {
            waiter.commit().unwrap();
            held(device, rule, (&waiter, &parts), &refs);
            signaller.commit().unwrap();
        }
        None => waiter.commit().unwrap(),
    }
    waiter.wait_until_scheduled().unwrap();
    let in_flight = parts.landed.signaled_value() == 0;
    // Metal checked it as it scheduled it: a detach now fails nothing.
    parts.detach.detach(&parts.target);
    if let Some(signaller) = &signaller {
        signaller.wait_until_completed_for(WAIT).unwrap();
        assert_eq!(signaller.status(), CommandBufferStatus::Completed, "{name}");
    }
    waiter.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(waiter.status(), CommandBufferStatus::Completed, "{name}");
    assert_eq!(
        read_buffer(&parts.target, 0, 4).unwrap(),
        [0x5a; 4],
        "{name}"
    );
    let checked = parts.checked();
    assert_eq!(
        checked.len(),
        1,
        "{name}: the waiter's submission is checked once"
    );
    if in_flight {
        assert_eq!(checked[0], 0, "{name}: checked after the GPU signal landed");
    }
    in_flight
}

/// Metal's rule for a command buffer waiting for `Event`s, each shape of [`EVENT_RULES`]: held,
/// unchecked, until each value it waits for has a committed signaller (or is held), then
/// scheduled and checked while that signaller still runs, in one try at least, the GPU waiting
/// for the signal; detached once scheduled, it runs. Such a waiter is never held for a landing
/// (`WorkCounts::event_holds_landed_event`). A try whose signaller finished before the waiter
/// read `Scheduled` proves nothing of the timing, and the next doubles the signaller's work.
#[test]
fn an_event_waiter_is_scheduled_once_each_value_has_a_committed_signaller() {
    on_device(|device, _| {
        let before = device.work_counts();
        for rule in EVENT_RULES {
            let caught = (0..ATTEMPTS)
                .map(|attempt| 4u64 << attempt)
                .find(|&fills| event_rule_try(device, rule, fills));
            let fills = caught.unwrap_or_else(|| {
                panic!(
                    "{}: the signaller never ran as the waiter was scheduled in {ATTEMPTS} tries",
                    rule.name
                )
            });
            println!(
                "EVENT-RULE {}: caught with {fills} fills of {FILL} bytes",
                rule.name
            );
        }
        let after = device.work_counts();
        assert_eq!(
            after.event_holds_landed_event, before.event_holds_landed_event,
            "a waiter for an Event was held for its signal to land"
        );
    });
}

/// A `SharedEvent` wait that a GPU signal in flight would meet, met by the host first: the check
/// runs at the host's signal, without waiting for the GPU (the companion event reads 0 in it, in
/// one try at least), and the submission runs. A try whose signaller finished first proves
/// nothing, and the next doubles its work.
#[test]
fn a_wait_the_host_meets_is_checked_without_waiting_for_the_gpu() {
    on_device(|device, _| {
        let mut caught = None;
        for attempt in 0..ATTEMPTS {
            let fills = 4u64 << attempt;
            let waited = device.new_shared_event().unwrap();
            let run = InFlight::start(device, &waited, fills);
            waited.set_signaled_value(1);
            let at_check = run.finish();
            assert_eq!(run.waiter.status(), CommandBufferStatus::Completed);
            assert_eq!(read_buffer(&run.parts.target, 0, 4).unwrap(), [0x5a; 4]);
            if at_check == 0 {
                caught = Some(fills);
                break;
            }
        }
        let fills = caught.unwrap_or_else(|| {
            panic!("the check never ran before the signaller finished in {ATTEMPTS} tries")
        });
        println!("HOST-MET caught with {fills} fills of {FILL} bytes");
    });
}
