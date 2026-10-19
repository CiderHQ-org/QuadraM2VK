//! A device's resolver: the thread that encodes its captured render encoders' calls and commits
//! their command buffers, while the program's thread goes on with its next calls
//! (`render/captured.rs`).
//!
//! - **Answered at the call.** A captured encoder's call returns the answer the encoding would
//!   return at the call, decided there: by its arguments, the device and the pass's descriptor, or
//!   by an earlier call the encoding answered `Ok` with the same encoder state and the same
//!   objects (`render/captured.rs`). A call whose answer the encoder cannot so decide waits: every
//!   call handed here before it is encoded first ([`ResolverShared::drain`]), and it and the
//!   encoder's later calls are encoded at the call, as without a resolver.
//! - **Encoded here, in order.** Each answered call is handed here as it is made, with the
//!   arguments it was given (handles held, bytes copied), and encoded on this thread by the same
//!   encoding as at the call, in the order the calls were made, so the commands are those the
//!   call would record. A command buffer whose encoders were all captured commits here after
//!   them, so command buffers reach the scheduler in commit order.
//! - **What the encoding reads is as at the call.** A call that changes what the encoding reads, or
//!   reads what it records (a rebind, detach or release of host pages, the CPU's access to a
//!   texture, a resource's or a heap's purgeable state, a heap resource made aliasable, the used
//!   resources' lists, an encoder encoded at the call, a command buffer committed at the call)
//!   first has everything handed here encoded ([`ResolverShared::drain`]), so each encoding reads
//!   what its call saw.
//! - **A drain encodes in the thread's place.** A drain wakes no sleeping thread and waits for no
//!   wake: where the thread is not part-way through a piece, the calling thread takes the pieces
//!   handed before its call off the queue and encodes them itself, in order, as the thread would;
//!   only a piece the thread is part-way through is waited for. The thread takes no piece
//!   meanwhile, so the pieces are encoded one at a time, in the order they were handed. A thread
//!   command buffers' completion depends on (`exit_guard::serving_completion`) encodes nothing in
//!   a drain, which an encoding could wait on: it wakes the thread and waits for it.
//! - **A failure here** (out of memory, a lost device, a runtime defect) fails the command buffer
//!   at its submission, as a host step's does (`CommandBufferError`), counted
//!   (`WorkCounts::resolver_failures`) and announced. A refusal here of a call answered `Ok` at
//!   the call is a defect of the answer, counted apart (`WorkCounts::answered_calls_refused`).
//! - **No notice comes from here.** A captured call is one whose encoding announces nothing: the
//!   memo learns a call only from an encoding at the call that delivered no message
//!   (`render/captured.rs`), so a call that announces (a kind the device announces at every
//!   occurrence, or one it has not announced yet) is encoded at the call, where its notice comes
//!   in the program's call. A message delivered on this thread anyway (a notice the memo's keys
//!   did not foresee, or a failure's below) is counted (`WorkCounts::resolver_notices`), and the
//!   memo forgets every key, so the shape goes back to the call.
//! - **Woken once per burst.** A piece handed before an encoder's end wakes no sleeping thread;
//!   an encoder's end, a commit, or [`WAKE_AT`] pieces waiting wake it, and the thread looks for
//!   more work for [`SPIN`] before it sleeps again.

use crate::command_buffer::{BatchCommand, CommandBuffer, ResolverHold};
use crate::device::{lock, Device};
use crate::exit_guard;
use crate::render::RenderWork;
use std::cell::Cell;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

/// What the resolver encodes, in the order it was handed.
pub(crate) enum Work {
    /// Calls of a captured render encoder (`render/captured.rs`).
    Render(Box<RenderWork>),
    /// An event or binary semaphore command of a command buffer with captured encoders, between
    /// them.
    Batch(ResolverHold, BatchCommand),
    /// The commit of a command buffer whose encoders were all captured.
    Commit(CommandBuffer),
}

impl Work {
    /// The device whose work it is.
    fn device(&self) -> &Device {
        match self {
            Work::Render(work) => work.device(),
            Work::Batch(command_buffer, _) => command_buffer.device(),
            Work::Commit(command_buffer) => command_buffer.device(),
        }
    }
}

/// How many pieces of work the queue holds without growing: more than a frame hands over.
const QUEUE_ROOM: usize = 1024;

/// How long the thread looks for more work before it sleeps: a program's calls come microseconds
/// apart, and a sleeping thread costs each hand-over a wake.
const SPIN: Duration = Duration::from_micros(50);

/// How many pieces waiting wake the sleeping thread whatever handed them.
const WAKE_AT: usize = 16;

/// What a drain waits for the resolver for (`ResolverShared::drain`): the call that needs what
/// was handed encoded first, counted apart (`WorkCounts::resolver_drains_commits` and kin).
#[derive(Clone, Copy)]
pub(crate) enum Drain {
    /// A commit at the call: of a command buffer with an encoder or a command at the call.
    Commit,
    /// An encoder other than a render encoder opened at the call, whose commands follow the
    /// captured ones.
    Encoder,
    /// A render encoder opened at the call: its pass, or the command buffer, not one the memo
    /// keys.
    RenderOpen,
    /// A captured encoder taken back to the call by a call the memo does not answer.
    TakeBack,
    /// The program's last hold of an uncommitted command buffer the resolver holds.
    Drop,
    /// A call on an object whose state an encoding reads: its bytes on the CPU, its pages, its
    /// purgeability or aliasing, a command buffer's used resources.
    Object,
}

/// How many kinds of [`Drain`] there are.
const DRAINS: usize = 6;

/// The resolver's queue.
struct Queue {
    work: VecDeque<Work>,
    /// Pieces handed so far, and encoded so far: a drain waits for the second to reach the first
    /// as it stood.
    handed: u64,
    encoded: u64,
    /// Threads waiting in [`ResolverShared::drain`].
    draining: u32,
    /// The thread is part-way through a piece it took off the queue.
    running: bool,
    /// A draining thread encodes pieces in the thread's place: the thread takes none meanwhile.
    taken_over: bool,
    /// Drains that will encode in the thread's place once it is through its piece: the thread
    /// takes no more meanwhile.
    takers: u32,
    /// The thread sleeps for work.
    parked: bool,
    /// When a hand-over or a drain woke the sleeping thread, until it runs.
    woken: Option<Instant>,
    stop: bool,
}

/// What the device and the resolver's thread share.
pub(crate) struct ResolverShared {
    queue: Mutex<Queue>,
    /// The pieces handed so far, taken off the queue so far (by the thread or a drain), and
    /// encoded so far, read without the lock: by the thread looking for work, and by a drain that
    /// finds nothing to wait for.
    handed: AtomicU64,
    started: AtomicU64,
    encoded_count: AtomicU64,
    /// The time the thread spent encoding.
    busy_ns: AtomicU64,
    /// The messages delivered on the thread while it encoded (`device::heard_here`).
    notices: AtomicU64,
    /// The thread sleeps here for work.
    wake: Condvar,
    /// Drains wait here for work to be encoded.
    encoded: Condvar,
    /// Drains that waited, the time they waited, and those that found nothing to wait for.
    drains: AtomicU64,
    drain_wait_ns: AtomicU64,
    drains_idle: AtomicU64,
    /// Of `drains`, by [`Drain`]: how many, and the time they waited.
    drains_by: [AtomicU64; DRAINS],
    drain_ns_by: [AtomicU64; DRAINS],
    /// The pieces the drains that waited waited for.
    drain_pieces: AtomicU64,
    /// Hand-overs that woke the sleeping thread, and the time from each wake to the thread
    /// running.
    wakes: AtomicU64,
    wake_ns: AtomicU64,
    /// Drains that encoded pieces in the thread's place, the pieces, and the time encoding them;
    /// and the wakes of the sleeping thread that drains waited for.
    drains_encoding: AtomicU64,
    drain_wakes: AtomicU64,
    drain_encoded_pieces: AtomicU64,
    drain_encoding_ns: AtomicU64,
}

/// The resolver's counts so far ([`ResolverShared::counts`]).
#[derive(Clone, Copy, Default)]
pub(crate) struct ResolverCounts {
    pub(crate) drains: u64,
    pub(crate) drain_wait_ns: u64,
    pub(crate) drains_idle: u64,
    pub(crate) drains_by: [u64; DRAINS],
    pub(crate) drain_ns_by: [u64; DRAINS],
    pub(crate) drain_pieces: u64,
    pub(crate) wakes: u64,
    pub(crate) wake_ns: u64,
    pub(crate) busy_ns: u64,
    pub(crate) notices: u64,
    pub(crate) drains_encoding: u64,
    pub(crate) drain_wakes: u64,
    pub(crate) drain_encoded_pieces: u64,
    pub(crate) drain_encoding_ns: u64,
}

thread_local! {
    /// This thread is a resolver's: a drain from it waits for nothing (it encodes in order).
    static RESOLVING: Cell<bool> = const { Cell::new(false) };
}

/// Whether the calling thread is a device's resolver, or encodes pieces in its place.
pub(crate) fn resolving() -> bool {
    RESOLVING.with(Cell::get)
}

/// Run `f` as a resolver's thread runs ([`resolving`]), the mark as before after.
fn as_resolver<T>(f: impl FnOnce() -> T) -> T {
    /// Puts the mark back as `f` returns or unwinds.
    struct Mark(bool);
    impl Drop for Mark {
        fn drop(&mut self) {
            RESOLVING.with(|resolving| resolving.set(self.0));
        }
    }
    let _mark = Mark(RESOLVING.with(|resolving| resolving.replace(true)));
    f()
}

impl ResolverShared {
    fn new() -> ResolverShared {
        ResolverShared {
            queue: Mutex::new(Queue {
                work: VecDeque::with_capacity(QUEUE_ROOM),
                handed: 0,
                encoded: 0,
                draining: 0,
                running: false,
                taken_over: false,
                takers: 0,
                parked: false,
                woken: None,
                stop: false,
            }),
            handed: AtomicU64::new(0),
            started: AtomicU64::new(0),
            encoded_count: AtomicU64::new(0),
            busy_ns: AtomicU64::new(0),
            notices: AtomicU64::new(0),
            wake: Condvar::new(),
            encoded: Condvar::new(),
            drains: AtomicU64::new(0),
            drain_wait_ns: AtomicU64::new(0),
            drains_idle: AtomicU64::new(0),
            drains_by: Default::default(),
            drain_ns_by: Default::default(),
            drain_pieces: AtomicU64::new(0),
            wakes: AtomicU64::new(0),
            wake_ns: AtomicU64::new(0),
            drains_encoding: AtomicU64::new(0),
            drain_wakes: AtomicU64::new(0),
            drain_encoded_pieces: AtomicU64::new(0),
            drain_encoding_ns: AtomicU64::new(0),
        }
    }

    /// Hand `work` to the thread, waking it where it sleeps and `wake` (the end of a burst), or
    /// where [`WAKE_AT`] pieces wait.
    pub(crate) fn hand(&self, work: Work, wake: bool) {
        let mut queue = lock(&self.queue);
        queue.work.push_back(work);
        queue.handed += 1;
        self.handed.store(queue.handed, Ordering::Release);
        if queue.parked && (wake || queue.work.len() >= WAKE_AT) {
            self.wake_parked(queue);
        }
    }

    /// Wake the sleeping thread, `queue` held.
    fn wake_parked(&self, mut queue: std::sync::MutexGuard<'_, Queue>) {
        self.wake_parked_held(&mut queue);
        drop(queue);
    }

    /// Wake the sleeping thread, `queue` held and kept.
    fn wake_parked_held(&self, queue: &mut Queue) {
        queue.parked = false;
        queue.woken = Some(Instant::now());
        self.wakes.fetch_add(1, Ordering::Relaxed);
        self.wake.notify_one();
    }

    /// Whether the thread encoded everything handed to it, so it waits for work.
    pub(crate) fn waits(&self) -> bool {
        self.encoded_count.load(Ordering::Acquire) >= self.handed.load(Ordering::Acquire)
    }

    /// Have everything handed to the thread before the call encoded (the module docs), for
    /// `why`: the calling thread encodes the pieces itself where the thread is not part-way
    /// through one, and waits only for a piece it is. Free on the resolver's own thread, which
    /// encodes in order. The caller holds no lock of the runtime's: the work takes them. A thread
    /// command buffers' completion depends on (`exit_guard::serving_completion`) encodes nothing
    /// here and waits for the thread, woken.
    pub(crate) fn drain(&self, why: Drain) {
        if resolving() {
            return;
        }
        // The program's thread hands every piece it waits for, so what it handed is counted.
        if self.waits() {
            self.drains_idle.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let mut queue = lock(&self.queue);
        let target = queue.handed;
        if queue.encoded >= target {
            drop(queue);
            self.drains_idle.fetch_add(1, Ordering::Relaxed);
            return;
        }
        let start = Instant::now();
        self.drain_pieces
            .fetch_add(target - queue.encoded, Ordering::Relaxed);
        let encodes = !exit_guard::serving_completion();
        let mut encoded_here = false;
        let mut spun = false;
        queue.draining += 1;
        queue.takers += u32::from(encodes);
        while queue.encoded < target {
            if encodes && !queue.running && !queue.taken_over && !queue.work.is_empty() {
                queue = self.encode_in_place(queue, target);
                encoded_here = true;
                continue;
            }
            if !encodes && queue.parked {
                self.wakes.fetch_add(1, Ordering::Relaxed);
                self.drain_wakes.fetch_add(1, Ordering::Relaxed);
                queue.parked = false;
                queue.woken = Some(Instant::now());
                self.wake.notify_one();
            }
            // The thread is part-way through a piece: a piece is short, so look for its end
            // before sleeping until it.
            if encodes && !spun {
                spun = true;
                let seen = queue.encoded;
                drop(queue);
                let looking = Instant::now();
                while self.encoded_count.load(Ordering::Acquire) == seen && looking.elapsed() < SPIN
                {
                    std::hint::spin_loop();
                }
                queue = lock(&self.queue);
                continue;
            }
            spun = false;
            queue = self
                .encoded
                .wait(queue)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
        }
        queue.draining -= 1;
        queue.takers -= u32::from(encodes);
        // Pieces handed after the target are the thread's once no drain will take them: a wake
        // handed meanwhile found the thread kept from them and lost it.
        if queue.parked && queue.takers == 0 && !queue.work.is_empty() {
            self.wake_parked_held(&mut queue);
        }
        drop(queue);
        let waited = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        self.drains.fetch_add(1, Ordering::Relaxed);
        self.drain_wait_ns.fetch_add(waited, Ordering::Relaxed);
        self.drains_by[why as usize].fetch_add(1, Ordering::Relaxed);
        self.drain_ns_by[why as usize].fetch_add(waited, Ordering::Relaxed);
        if encoded_here {
            self.drains_encoding.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Encode on the calling thread, in the thread's place, the pieces the queue holds until
    /// `target` pieces are encoded, `queue` held and given back held: the thread takes none
    /// meanwhile, and other drains look again after.
    fn encode_in_place<'a>(
        &'a self,
        mut queue: std::sync::MutexGuard<'a, Queue>,
        target: u64,
    ) -> std::sync::MutexGuard<'a, Queue> {
        queue.taken_over = true;
        while queue.encoded < target {
            let Some(work) = queue.work.pop_front() else {
                break;
            };
            self.started.fetch_add(1, Ordering::Release);
            drop(queue);
            let busy = as_resolver(|| self.encode(work));
            self.drain_encoded_pieces.fetch_add(1, Ordering::Relaxed);
            self.drain_encoding_ns.fetch_add(busy, Ordering::Relaxed);
            queue = lock(&self.queue);
            queue.encoded += 1;
            self.encoded_count.store(queue.encoded, Ordering::Release);
            // Other drains wait for the count too (this one is among `draining`).
            if queue.draining > 1 {
                self.encoded.notify_all();
            }
        }
        queue.taken_over = false;
        // Another drain waits for the count, or for the pieces to be its to take.
        if queue.draining > 1 {
            self.encoded.notify_all();
        }
        queue
    }

    /// Encode `work` on this thread, inside a section of the exit guard as the program's call
    /// (`exit_guard::on_programs_behalf`); the time it took. A message delivered meanwhile is
    /// counted and makes the memo forget every key (the module docs).
    fn encode(&self, work: Work) -> u64 {
        let start = Instant::now();
        let heard = crate::device::heard_here();
        let device = work.device().clone();
        // Once the process is exiting the work is dropped unencoded.
        let _ = exit_guard::on_programs_behalf(move || match work {
            Work::Render(work) => work.resolve(),
            Work::Batch(command_buffer, command) => {
                command_buffer.resolve_batch_command(command);
            }
            Work::Commit(command_buffer) => command_buffer.resolve_commit(),
        });
        let busy = u64::try_from(start.elapsed().as_nanos()).unwrap_or(u64::MAX);
        // A message the memo's keys did not foresee: counted, and the memo forgets every key, so
        // the shape that announced it is encoded at the call from its next call on, where it
        // announces again or teaches a key that announces nothing.
        let noticed = crate::device::heard_here() - heard;
        if noticed > 0 {
            self.notices.fetch_add(noticed, Ordering::Relaxed);
            lock(&device.shared().answers).forget();
        }
        busy
    }

    /// The counts so far.
    pub(crate) fn counts(&self) -> ResolverCounts {
        ResolverCounts {
            drains: self.drains.load(Ordering::Relaxed),
            drain_wait_ns: self.drain_wait_ns.load(Ordering::Relaxed),
            drains_idle: self.drains_idle.load(Ordering::Relaxed),
            drains_by: self.drains_by.each_ref().map(|n| n.load(Ordering::Relaxed)),
            drain_ns_by: self
                .drain_ns_by
                .each_ref()
                .map(|n| n.load(Ordering::Relaxed)),
            drain_pieces: self.drain_pieces.load(Ordering::Relaxed),
            wakes: self.wakes.load(Ordering::Relaxed),
            wake_ns: self.wake_ns.load(Ordering::Relaxed),
            busy_ns: self.busy_ns.load(Ordering::Relaxed),
            notices: self.notices.load(Ordering::Relaxed),
            drains_encoding: self.drains_encoding.load(Ordering::Relaxed),
            drain_wakes: self.drain_wakes.load(Ordering::Relaxed),
            drain_encoded_pieces: self.drain_encoded_pieces.load(Ordering::Relaxed),
            drain_encoding_ns: self.drain_encoding_ns.load(Ordering::Relaxed),
        }
    }

    /// The next piece of work: what the queue holds, else what is handed while the thread looks
    /// for [`SPIN`], else after a sleep; none while a drain encodes in its place; `None` once
    /// [`stop`](Self::stop) finds the queue empty. The piece is the thread's, part-way through,
    /// until [`run`](Self::run) counts it encoded.
    fn next(&self) -> Option<Work> {
        let start = Instant::now();
        while self.handed.load(Ordering::Acquire) == self.started.load(Ordering::Acquire)
            && start.elapsed() < SPIN
        {
            std::hint::spin_loop();
        }
        let mut queue = lock(&self.queue);
        loop {
            if !queue.taken_over && queue.takers == 0 {
                if let Some(work) = queue.work.pop_front() {
                    queue.running = true;
                    self.started.fetch_add(1, Ordering::Release);
                    return Some(work);
                }
            }
            if queue.stop && queue.work.is_empty() {
                return None;
            }
            queue.parked = true;
            queue = self
                .wake
                .wait(queue)
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            // The time from the wake to running here.
            if let Some(woken) = queue.woken.take() {
                let latency = u64::try_from(woken.elapsed().as_nanos()).unwrap_or(u64::MAX);
                self.wake_ns.fetch_add(latency, Ordering::Relaxed);
            }
        }
    }

    /// The thread's loop: each piece of work encoded in order, as the program's call inside a
    /// section of the exit guard (`exit_guard::on_programs_behalf`), until [`stop`](Self::stop)
    /// finds the queue empty. Once the process exits, work is dropped unencoded: its command
    /// buffers never complete, as the process ends.
    fn run(&self) {
        RESOLVING.with(|resolving| resolving.set(true));
        while let Some(work) = self.next() {
            let busy = self.encode(work);
            self.busy_ns.fetch_add(busy, Ordering::Relaxed);
            let mut queue = lock(&self.queue);
            queue.running = false;
            queue.encoded += 1;
            self.encoded_count.store(queue.encoded, Ordering::Release);
            let waiting = queue.draining > 0;
            drop(queue);
            if waiting {
                self.encoded.notify_all();
            }
        }
    }

    /// End the thread's loop once its queue is empty.
    fn stop(&self) {
        let mut queue = lock(&self.queue);
        queue.stop = true;
        queue.parked = false;
        drop(queue);
        self.wake.notify_one();
    }
}

/// A device's resolver thread and what it shares with the device.
pub(crate) struct Resolver {
    shared: Arc<ResolverShared>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl Resolver {
    /// A device's resolver, its thread started; `None` where the host starts no thread (the
    /// device encodes every call at the call).
    pub(crate) fn start() -> Option<Resolver> {
        let shared = Arc::new(ResolverShared::new());
        let running = Arc::clone(&shared);
        // Apart from the recorder's and the scheduler's in a profile: Linux keeps 15 bytes of a
        // thread's name.
        let builder = std::thread::Builder::new().name("m2v-resolver".to_owned());
        let thread = exit_guard::spawn_user_interactive(builder, move || running.run()).ok()?;
        Some(Resolver {
            shared,
            thread: Some(thread),
        })
    }

    /// What it shares with the device.
    pub(crate) fn shared(&self) -> &ResolverShared {
        &self.shared
    }
}

impl Drop for Resolver {
    /// Stop the thread and wait for it, unless the device's teardown runs on that thread (its
    /// last command buffer let go of there), whose loop ends as this returns.
    fn drop(&mut self) {
        self.shared.stop();
        if let Some(thread) = self.thread.take() {
            if !resolving() {
                let _ = thread.join();
            }
        }
    }
}
