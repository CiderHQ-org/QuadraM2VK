//! The completion thread: one per device, it finishes each submitted batch once its queue's
//! timeline reached the batch's value, which one waiter thread per `VkQueue` watches.
//!
//! - **In each queue's order.** A `VkQueue`'s batches signal its timeline with rising values, and
//!   Vulkan's signal operation order makes them complete in that order, so each queue's batches
//!   wait in one line, and a value reached finishes every batch up to it.
//! - **What a finished batch does.** Its queue's completed value is published for the tracker
//!   (the host saw it before any later submission), the events it signaled are reached, and the
//!   batch that ends a command buffer posts its transition to `Completed` to its queue's context
//!   ([`contexts`](crate::contexts)), once the thread read the command buffer's GPU
//!   timestamps and gave their slot back (`gpu_time.rs`). The thread runs no handler.
//! - **A command buffer finishes at its fence, where it has one.** Where the driver still reads
//!   a command buffer's objects after the value (MoltenVK 1.4.1), the batch that ends it also
//!   signals a fence ([`fences`](super::fences)), which the driver signals after those reads: the
//!   thread posts the transition, after which what the command buffer used is released, only once
//!   the fence is signaled too, and then resets the fence and gives it back. A wait for fences
//!   holds up the other queues' batches by as long. Elsewhere Vulkan's value is enough, and the
//!   batch has no fence.
//! - **One waiter per `VkQueue`, each waiting for one object** (replacing the wait for any of the
//!   timelines). Each `VkQueue`'s waiter thread waits for its queue's first pending batch past the
//!   value it saw reached: for the batch's fence where it has one, else for its queue's timeline to
//!   reach the batch's value, a wait of that one fence or that one semaphore (`poll`). Then it
//!   reads the timeline, publishes it, and hands the value to the completion thread; a signaled
//!   fence's batch reached its value whatever the read shows (lavapipe can show it later). A batch
//!   submitted to an idle `VkQueue` is waited for by that queue's own waiter, so it completes when
//!   its work does, not when another queue's does. No wait names several objects: MoltenVK 1.4.1's
//!   wait for any of several timelines (`vkWaitSemaphores` with `VK_SEMAPHORE_WAIT_ANY_BIT`,
//!   `mvkWaitSemaphores`) loses a signal that lands while it is still registering a later timeline,
//!   so the thread saw the batch only when its 100 ms wait ran out (the consumer's guest presented
//!   at about a quarter of Metal's rate); its wait for any of several fences has the same flaw. A
//!   wait for one timeline or one fence counts its one signal under a lock, and loses none. The
//!   value is read again after every wait, so the waiter relies on no wake for the value, only for
//!   its timing.
//! - **Waits are bounded.** A waiter waits at most [`WAIT_SLICE`] at a time, so a stop is seen
//!   promptly and a lost device is reported by the next wait.
//! - **Exit.** Each wait is a section of the exit guard ([`exit_guard`]), so an exit waits at most
//!   one slice for it; once the process is exiting, the threads end.
//! - **A drain waits for the waiters.** A failed wait or read finishes every pending batch, which
//!   may destroy their fences (`let_go`); the batches are taken only once no waiter is waiting,
//!   and none arms until they are.
//! - **Let go before the waiters return.** The thread holds a command buffer through a [`Hold`],
//!   and lets go of everything it holds for a batch (its events, then the command buffer, or
//!   hands the hold to the transition it posts) before the next; the command buffer's waiters
//!   return once no hold is left ([`gate`](super::gate)). It reaches the device through a pending
//!   command buffer only for the wait, and for a failed wait's drain.
//! - **A failed wait drains the queues first**. Its command buffers finish with its error, and let
//!   go of what they used, once the device's queues were drained (`Device::drain_queues`): MoltenVK
//!   1.4.1 reports a lost device from the first completed handler of a failed Metal command buffer,
//!   while later ones still read the command buffers' pools.
//! - **No self-join.** The thread's drop of a command buffer's last reference can drop the
//!   device's last handle on this thread; stopping from the thread itself then only asks it to
//!   end.

use super::gate::Hold;
use super::CommandBufferShared;
use crate::device::lock;
use crate::error::{CommandBufferError, Error, MemoryKind, VkResultCode};
use crate::event::EventShared;
use crate::exit_guard::{self, Call};
use crate::gpu_time;
use crate::release;
use crate::Device;
use ash::vk;
use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

/// How long one wait on the timelines lasts.
const WAIT_SLICE: Duration = Duration::from_millis(100);

/// One device's completion thread and its `VkQueue`s' waiters.
pub(crate) struct Completion {
    watch: Arc<Watch>,
    threads: Mutex<Vec<JoinHandle<()>>>,
}

/// What the threads and the scheduler share.
struct Watch {
    pending: Mutex<Pending>,
    /// Notified on every change a thread waits for: a batch watched, a value reached, a wait
    /// ended, batches finished or taken, a failure, a stop, an exit.
    wake: Condvar,
}

struct Pending {
    /// Per `VkQueue`: the submitted batches not yet finished, in value order.
    queues: Vec<VecDeque<Submitted>>,
    /// Per `VkQueue`: the largest value its waiter saw its timeline reach.
    reached: Vec<u64>,
    /// Per `VkQueue`: its waiter is in a wait.
    waiting: Vec<bool>,
    /// A waiter's failed wait or read, which every pending batch finishes with.
    failed: Option<CommandBufferError>,
    /// The pending batches are being taken (`take_pending`): no waiter arms.
    holding: bool,
    /// The process is exiting: every thread ends, and what is pending stays pending.
    exiting: bool,
    stopping: bool,
}

/// A submitted batch the thread waits for.
pub(crate) struct Submitted {
    /// The value its queue's timeline reaches once it completed.
    pub(crate) value: u64,
    pub(crate) buffer: Hold<CommandBufferShared>,
    /// It ends its command buffer: finishing it completes the command buffer.
    pub(crate) last: bool,
    /// The fence its submission signals after the work, when it ends its command buffer where
    /// the driver needs one, or its queue's failures are relayed: the command buffer finishes
    /// once it is signaled ([`fences`](super::fences)).
    pub(crate) fence: Option<vk::Fence>,
    /// The events it signals, with their signal indices.
    pub(crate) signals: Vec<(Arc<EventShared>, u64)>,
}

impl Completion {
    /// Start the threads of a device with `queues` `VkQueue`s: the completion thread and one
    /// waiter per `VkQueue`; refused when the host cannot start one.
    pub(crate) fn start(queues: usize) -> Result<Completion, Error> {
        let watch = Arc::new(Watch {
            pending: Mutex::new(Pending {
                queues: (0..queues).map(|_| VecDeque::new()).collect(),
                reached: vec![0; queues],
                waiting: vec![false; queues],
                failed: None,
                holding: false,
                exiting: false,
                stopping: false,
            }),
            wake: Condvar::new(),
        });
        let completion = Completion {
            watch: Arc::clone(&watch),
            threads: Mutex::new(Vec::new()),
        };
        let spawned: std::io::Result<()> = (|| {
            let thread = {
                let watch = Arc::clone(&watch);
                exit_guard::spawn_user_interactive(
                    std::thread::Builder::new().name("metal2vulkan-completion".into()),
                    move || run(&watch),
                )?
            };
            lock(&completion.threads).push(thread);
            for queue in 0..queues {
                let watch = Arc::clone(&watch);
                let waiter = exit_guard::spawn_user_interactive(
                    std::thread::Builder::new().name(format!("metal2vulkan-queue-wait-{queue}")),
                    move || wait_for_queue(&watch, queue),
                )?;
                lock(&completion.threads).push(waiter);
            }
            Ok(())
        })();
        match spawned {
            Ok(()) => Ok(completion),
            Err(_) => {
                completion.stop();
                Err(Error::OutOfMemory(MemoryKind::Host))
            }
        }
    }

    /// Wait for `batch`, just submitted to `VkQueue` `queue` (or a failed command buffer's end,
    /// at the queue's last submitted value).
    pub(crate) fn watch(&self, queue: usize, batch: Submitted) {
        lock(&self.watch.pending).queues[queue].push_back(batch);
        self.watch.wake.notify_all();
    }

    /// Ask the threads to end once nothing is pending, and wait for each but this one (the
    /// device's last handle can drop on one of them).
    pub(crate) fn stop(&self) {
        lock(&self.watch.pending).stopping = true;
        self.watch.wake.notify_all();
        let current = std::thread::current().id();
        let threads: Vec<JoinHandle<()>> = lock(&self.threads).drain(..).collect();
        for handle in threads {
            if handle.thread().id() != current {
                let _ = handle.join();
            }
        }
    }
}

/// What the completion thread found: batches whose values their queues reached, or a waiter's
/// failure.
enum Turn {
    Finished(Vec<(usize, Submitted)>),
    Failed(CommandBufferError),
}

/// Wait on `watch`'s condition variable up to one slice.
fn wait<'a>(
    watch: &'a Watch,
    pending: std::sync::MutexGuard<'a, Pending>,
) -> std::sync::MutexGuard<'a, Pending> {
    watch
        .wake
        .wait_timeout(pending, WAIT_SLICE)
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .0
}

/// The process is exiting: every thread ends.
fn exit(watch: &Watch) {
    lock(&watch.pending).exiting = true;
    watch.wake.notify_all();
}

/// The completion thread's loop.
fn run(watch: &Watch) {
    loop {
        // The batches whose values their queues' waiters saw reached, or a waiter's failure.
        let turn = {
            let mut pending = lock(&watch.pending);
            loop {
                if pending.exiting {
                    return;
                }
                if let Some(error) = pending.failed.take() {
                    // No waiter arms until the pending batches are taken (`fail`).
                    pending.holding = true;
                    break Turn::Failed(error);
                }
                let finished = reached(&mut pending);
                if !finished.is_empty() {
                    break Turn::Finished(finished);
                }
                if pending.stopping && pending.queues.iter().all(VecDeque::is_empty) {
                    return;
                }
                pending = wait(watch, pending);
            }
        };
        // A waiter whose front finished may arm for its queue's next.
        watch.wake.notify_all();
        let finished = match turn {
            Turn::Finished(finished) => finished,
            Turn::Failed(error) => {
                fail(watch, Vec::new(), error);
                continue;
            }
        };
        // The host time the batches were seen complete: an uncalibrated command buffer's GPU end
        // (`gpu_time.rs`).
        let seen = gpu_time::host_nanos();
        // The device, held while the batches finish (each holds it meanwhile).
        let Some(device) = finished
            .first()
            .map(|(_, batch)| batch.buffer.device().clone())
        else {
            continue;
        };
        // Their event signals are seen before the scheduler looks again: a batch whose submission
        // checks wait for a GPU signal to land may now go.
        for (_, batch) in &finished {
            for (event, index) in &batch.signals {
                event.reached(*index);
            }
        }
        // A batch held at its gate, or for a GPU signal to land, may now go (`scheduler`).
        device.shared().scheduler.completions();
        // Every finished batch's fence is waited for: where a queue's failures are relayed each
        // batch has one (`device/relay.rs`), signaled after the driver reported its failure.
        let mut ending = Vec::new();
        let mut fences = Vec::new();
        for (queue, batch) in finished {
            let Submitted {
                buffer,
                last,
                fence,
                ..
            } = batch;
            fences.extend(fence.map(|fence| (queue, fence)));
            if last {
                lock(&buffer.state).times.completed = seen;
                ending.push(buffer);
            }
        }
        let settled = settle(&device, &fences);
        // Their GPU times' timestamps are read, and their slots given back, and their GPU times
        // placed among their queues' in order, before their transitions are posted: a completed
        // handler reads the times. A section, as each poll.
        if let Settled::Signaled = settled {
            let Some(_busy) = exit_guard::enter() else {
                exit(watch);
                return;
            };
            for buffer in &ending {
                buffer.read_stamps();
                buffer.place_gpu_times();
            }
        }
        drop(device);
        match settled {
            Settled::Signaled => {
                for buffer in ending {
                    CommandBufferShared::post_finished(buffer, Ok(()));
                }
            }
            // What is pending stays pending, as Metal's does.
            Settled::Exiting => {
                exit(watch);
                return;
            }
            Settled::Failed(error) => fail(watch, ending, error),
        }
    }
}

/// Every pending batch, and the command buffers `ending` whose work completed (their fences let
/// go of already), finish with `error`: a wait or a read failed. The pending batches' fences are
/// let go of ([`let_go`]).
///
/// First the device's queues are drained (`Device::drain_queues`): a failed command buffer lets go
/// of its pools as it finishes, and a driver may still read them after the failed wait returned
/// (MoltenVK 1.4.1, from its later completed handlers), until its own completion of the work ran.
fn fail(watch: &Watch, ending: Vec<Hold<CommandBufferShared>>, error: CommandBufferError) {
    let failed = take_pending(watch);
    let held = (ending.first().map(|buffer| buffer.device()))
        .or_else(|| failed.first().map(|(_, batch)| batch.buffer.device()));
    if let Some(device) = held {
        device.drain_queues();
    }
    let failed = failed.into_iter().filter_map(|(queue, batch)| {
        let Submitted {
            buffer,
            last,
            fence,
            ..
        } = batch;
        if let Some(fence) = fence {
            let_go(buffer.device(), queue, fence);
        }
        last.then_some(buffer)
    });
    for buffer in ending.into_iter().chain(failed) {
        CommandBufferShared::post_finished(buffer, Err(error));
    }
}

/// Let go of `fence` of `device`'s `VkQueue` `queue` after a failed wait: released once the
/// device is lost, when every submission counts as completed (out of the queue's submissions in
/// flight first, `device/relay.rs`); otherwise left to the device, since a submission that
/// signals it may still be pending. The release reaches the device's functions inside a section
/// of the exit guard, whether or not the caller is in one ([`fail`] is not: it posts the
/// command buffers' transitions, which a section may not), and once the process is exiting the
/// fence is left to its end.
fn let_go(device: &Device, queue: usize, fence: vk::Fence) {
    if device.is_lost() {
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        let shared = device.shared();
        if let Some(flights) = shared.flights(queue) {
            flights.let_go(fence);
        }
        // SAFETY: the fence is the device's, and the device is lost.
        unsafe { release::object(shared.vk(), fence) };
    }
}

/// How the fences of the command buffers whose work completed ended.
enum Settled {
    /// Each is signaled, reset and given back.
    Signaled,
    /// The process is exiting: the thread ends.
    Exiting,
    /// A wait or a reset failed: the command buffers finish with this error, and the fences are
    /// let go of ([`let_go`]).
    Failed(CommandBufferError),
}

/// Wait for `fences` of `device` (each of a `VkQueue`) to be signaled, a slice at a time, each
/// wait a section of the exit guard; then reset them and give them back to the device's idle
/// fences, with the marks before them where the queue's failures are relayed
/// (`device/relay.rs`), which leave its submissions in flight first.
fn settle(device: &Device, fences: &[(usize, vk::Fence)]) -> Settled {
    if fences.is_empty() {
        return Settled::Signaled;
    }
    let waited_for: Vec<vk::Fence> = fences.iter().map(|&(_, fence)| fence).collect();
    loop {
        let Some(_busy) = exit_guard::enter() else {
            return Settled::Exiting;
        };
        let shared = device.shared();
        let raw = shared.vk();
        // SAFETY: the fences are the device's, each signaled by a submission made.
        let waited =
            unsafe { raw.wait_for_fences(&waited_for, true, WAIT_SLICE.as_nanos() as u64) };
        exit_guard::called(Call::WaitForFences);
        let error = match waited {
            Ok(()) => {
                // Out of the queues' submissions in flight before they go back: every
                // submission before a signaled fence's completed too.
                let mut all = waited_for.clone();
                for (queue, flights) in shared.relayed_queues() {
                    let ours: Vec<vk::Fence> = fences
                        .iter()
                        .filter(|&&(q, _)| q == queue)
                        .map(|&(_, fence)| fence)
                        .collect();
                    all.extend(flights.settled(&ours));
                }
                // SAFETY: each fence is signaled and its submission completed.
                let reset = unsafe { raw.reset_fences(&all) };
                exit_guard::called(Call::ResetFences);
                match reset {
                    Ok(()) => {
                        // SAFETY: each fence is unsignaled and in no pending submission; inside
                        // a section.
                        unsafe { shared.fences.give(raw, all) };
                        return Settled::Signaled;
                    }
                    Err(error) => error,
                }
            }
            Err(vk::Result::TIMEOUT) => continue,
            Err(error) => error,
        };
        let failed = failure(device, error);
        for &(queue, fence) in fences {
            let_go(device, queue, fence);
        }
        return Settled::Failed(failed);
    }
}

/// Each queue's batches whose values its waiter saw its timeline reach, taken from `pending`, in
/// each queue's order.
fn reached(pending: &mut Pending) -> Vec<(usize, Submitted)> {
    let Pending {
        queues, reached, ..
    } = pending;
    let mut finished = Vec::new();
    for (queue, batches) in queues.iter_mut().enumerate() {
        while batches
            .front()
            .is_some_and(|batch| batch.value <= reached[queue])
        {
            finished.extend(batches.pop_front().map(|batch| (queue, batch)));
        }
    }
    finished
}

/// Every pending batch, taken once no waiter is in a wait: a waiter may wait for a pending
/// batch's fence, which the caller may destroy (`let_go`). No waiter arms meanwhile.
fn take_pending(watch: &Watch) -> Vec<(usize, Submitted)> {
    let mut pending = lock(&watch.pending);
    pending.holding = true;
    while pending.waiting.iter().any(|&waiting| waiting) {
        pending = wait(watch, pending);
    }
    let taken = pending
        .queues
        .iter_mut()
        .enumerate()
        .flat_map(|(queue, batches)| batches.drain(..).map(move |batch| (queue, batch)))
        .collect();
    pending.holding = false;
    drop(pending);
    watch.wake.notify_all();
    taken
}

/// A queue's first pending batch, as [`poll`] waits for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Front {
    /// The value its queue's timeline reaches once it completed.
    value: u64,
    /// The fence its submission signals, where it has one.
    fence: Option<vk::Fence>,
}

/// `VkQueue` `queue`'s waiter (the module docs): it waits for its queue's first pending batch
/// past the value it saw reached, a slice at a time, and hands what the timeline reached, or a
/// failure, to the completion thread.
fn wait_for_queue(watch: &Watch, queue: usize) {
    loop {
        // The front to wait for, and the device, reached through it (held only for the wait).
        let (front, device) = {
            let mut pending = lock(&watch.pending);
            loop {
                if pending.exiting {
                    return;
                }
                let armable = !pending.holding && pending.failed.is_none();
                let armed = match pending.queues[queue].front() {
                    Some(batch) if armable && batch.value > pending.reached[queue] => {
                        let front = Front {
                            value: batch.value,
                            fence: batch.fence,
                        };
                        Some((front, batch.buffer.device().clone()))
                    }
                    None if pending.stopping => return,
                    _ => None,
                };
                if let Some(armed) = armed {
                    pending.waiting[queue] = true;
                    break armed;
                }
                pending = wait(watch, pending);
            }
        };
        // The wait is a section of the exit guard; once the process is exiting the threads end,
        // and what is pending stays pending, as Metal's does.
        let polled = match exit_guard::enter() {
            Some(busy) => {
                let polled = poll(&device, queue, front);
                drop(busy);
                Some(polled)
            }
            None => None,
        };
        // Let go of the device before the report: until then the front is pending (nothing
        // finishes or takes it while this waits), and holds the device, so this is never its
        // last handle, which goes on the program's or the handlers' threads (`teardown_threads`).
        drop(device);
        let mut pending = lock(&watch.pending);
        pending.waiting[queue] = false;
        let Some(polled) = polled else {
            pending.exiting = true;
            drop(pending);
            watch.wake.notify_all();
            return;
        };
        match polled {
            Ok(value) => {
                let reached = &mut pending.reached[queue];
                *reached = (*reached).max(value);
            }
            Err(error) => {
                pending.failed.get_or_insert(error);
            }
        }
        drop(pending);
        watch.wake.notify_all();
    }
}

/// Wait up to one slice for `front`, the first pending batch of `device`'s `VkQueue` `queue`: for
/// its fence where it has one, else for the queue's timeline to reach its value, a
/// wait of that one object (the module docs). Then read the timeline and publish it: the value it
/// reached, at least the front's where its fence was signaled. A failed wait or read is the error
/// every pending command buffer finishes with.
///
/// MoltenVK signals a fence from its Metal command buffer's completion handler, where it wakes a
/// timeline wait from a shared event's listener (MoltenVK's `MVKSync.mm`; 6 to 11 us later).
///
/// A signaled fence's batch reached its value whatever its timeline's read shows: a submission's
/// fence signal follows its timeline signal in Vulkan's signal operation order. A driver may show
/// the value later than the fence: lavapipe's emulated timeline shows it once the thread that
/// submitted the batch installed its point, after the batch ran, which a loaded host can hold up
/// past the fence's signal.
fn poll(device: &Device, queue: usize, front: Front) -> Result<u64, CommandBufferError> {
    let shared = device.shared();
    let raw = shared.vk();
    let slot = &shared.queues[queue];
    let signaled = match front.fence {
        Some(fence) => {
            // SAFETY: the fence is the device's, in a submission made and not yet reset (its
            // batch is pending, and is not taken while this waits: `take_pending`), live while
            // the device is (held here).
            let waited =
                unsafe { raw.wait_for_fences(&[fence], true, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitForFences);
            match waited {
                Ok(()) => true,
                Err(vk::Result::TIMEOUT) => false,
                Err(error) => return Err(failure(device, error)),
            }
        }
        None => {
            let semaphores = [slot.timeline];
            let values = [front.value];
            let info = vk::SemaphoreWaitInfo::default()
                .semaphores(&semaphores)
                .values(&values);
            // SAFETY: the semaphore is the device's timeline, live while the device is (held
            // here).
            let waited = unsafe { raw.wait_semaphores(&info, WAIT_SLICE.as_nanos() as u64) };
            exit_guard::called(Call::WaitSemaphores);
            shared.counts.timeline_wait();
            match waited {
                Ok(()) | Err(vk::Result::TIMEOUT) => false,
                Err(error) => return Err(failure(device, error)),
            }
        }
    };
    // SAFETY: the semaphore is the device's timeline, live while the device is (held here).
    let value = unsafe { raw.get_semaphore_counter_value(slot.timeline) };
    exit_guard::called(Call::GetSemaphoreCounterValue);
    let value = value.map_err(|error| failure(device, error))?;
    let value = match signaled {
        true => value.max(front.value),
        false => value,
    };
    slot.completed(value);
    Ok(value)
}

/// The command-buffer error of a failed wait, recording a lost device.
fn failure(device: &Device, error: vk::Result) -> CommandBufferError {
    match error {
        vk::Result::ERROR_DEVICE_LOST => {
            device.mark_lost();
            // A batch held for an event may now be refused.
            device.shared().scheduler.wake();
            CommandBufferError::DeviceLost
        }
        vk::Result::ERROR_OUT_OF_HOST_MEMORY | vk::Result::ERROR_OUT_OF_DEVICE_MEMORY => {
            CommandBufferError::OutOfMemory
        }
        other => CommandBufferError::Internal(VkResultCode(other.as_raw())),
    }
}
