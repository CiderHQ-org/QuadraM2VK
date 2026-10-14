//! `CommandBuffer` (`MTLCommandBuffer`).
//!
//! A command buffer owns a command pool of its own: each encoder records into a `VkCommandBuffer`
//! from it when its commands are encoded ([`encoder`](super::encoder)). The pool is a spare of
//! the device's where it has one, which a command buffer that completed gave back
//! ([`spares`](super::spares)). Its event waits and signals split its encoders into batches
//! ([`batches`]); `commit` hands them to the device's scheduler ([`scheduler`](super::scheduler)),
//! which submits each once the events it waits for are reached ([`submit`]), and the completion
//! thread ([`completion`]) sees them complete. One encoder is open at a time; an encoder dropped
//! without `end_encoding` loses its commands, so the command buffer can no longer be committed.
//!
//! Statuses follow Metal's: `NotEnqueued`, then `Committed` at `commit`, `Scheduled`, and
//! `Completed` or `Error`. The transitions after `Committed` run in the queue's serial context on
//! the device's handler threads ([`contexts`](super::contexts)), in the queue's order: to
//! `Scheduled` once its last batch is submitted (every wait of it satisfied), with its scheduled
//! handlers; to `Completed` once its work completed, with its completed handlers and the release
//! of what it retained. As on Metal, a queue's command buffer whose work completed stays
//! `Committed` while an earlier transition of its queue, a handler, still runs. Its submission
//! handlers (a runtime extension) run before its transition to `Scheduled`, on the thread that
//! submitted its last batch, which the transition waits for.
//!
//! Encoders write their commands' own data (tables, neutral buffers) into the command buffer's
//! upload arena ([`upload`](super::upload)), and the execution status words its commands share; the
//! command buffer decodes them when it completes ([`CommandBuffer::execution_report`]) and releases
//! the arena and what it retained.

pub(super) mod batches;
pub(super) mod checks;
pub(super) mod completion;
pub(super) mod fences;
mod first_submission;
pub(super) mod gate;
pub(super) mod kept_barriers;
pub(super) mod own;
mod reencode;
mod storage_form;
mod submit;

pub(super) use completion::Completion;
pub(super) use first_submission::NotedTable;

use super::contexts::{self, Run};
use super::deferred::{Deferred, Prepared};
use super::degradation::Degradation;
use super::descriptor_arena::DescriptorArena;
use super::device::lock;
use super::device::relay::Relayed;
use super::encoder::{Recorded, Segment};
use super::error::{
    internal, invalid, vulkan, CommandBufferError, DeadlockCause, Error, FailedPipeline,
    MeshRecordsOverflow, PipelineCreationCause, Problem, TimedOut,
};
use super::event::{EventRef, EventShared};
use super::execution::Execution;
use super::exit_guard::{self, Call};
use super::function_table::TableSnapshot;
use super::gpu_pools::{IndexPool, PoolKind, PoolLease};
use super::gpu_time::{self, Stamped, Times};
use super::heap::Heap;
use super::heap_sets::HeapUse;
use super::pipeline::kept::StepNumbers;
use super::predication::Predication;
use super::queue::Slot;
use super::rebind::{Holds, SubmittedHolds, Taken};
use super::replay::Replay;
use super::residency::Residency;
use super::resource::ResourceKey;
use super::sampler_listing::SamplerListing;
use super::scheduler::{BatchSync, StepWaits};
use super::spares::CommandPool;
use super::sparse::encoder::SparseUpdate;
use super::sparse::indirect::{self, ReadFailure};
use super::texture::storage_form::Riding;
use super::texture::Texture;
use super::tracker::Footprint;
use super::upload::{Upload, Uploads};
use super::used_resources::{EncoderKind, EncoderRecord, Reached, ReadRecord};
use super::values::CommandBufferStatus;
use super::{CommandQueue, Device};
use crate::release;
use ash::vk;
use batches::Batches;
use checks::{Checks, SubmissionCheck};
use completion::Submitted;
use first_submission::{Listing, Live};
use gate::{Gate, Hold};
use metal2vulkan::bind::{
    ExecutionStatus, IndexBoundsPass, ReplayFaults, ResidentRange, ResidentSet, RootTable,
    StatusReport,
};
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::DispatchType;
use own::{
    completed_wait, running, scheduled_wait, Answer, Context, Marker, Running, Step, Target,
};
use std::any::Any;
use std::collections::{BTreeSet, HashSet, VecDeque};
use std::ops::Range;
use std::sync::atomic::AtomicU64;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, OnceLock};
use std::time::{Duration, Instant};

/// A scheduled or completed handler (`MTLCommandBufferHandler`): called once with the command buffer, on one of the
/// device's handler threads, in its queue's order. A panic inside it is caught. A submission
/// handler ([`CommandBuffer::add_submission_handler`]) is one too, called on the thread that
/// submits the command buffer's last batch.
///
/// - **macOS.** The handler threads run at the user-interactive quality of service, whatever the
///   class of the thread that committed, as Metal runs its handlers. A thread a handler starts
///   starts at the default class, as a new thread does (`<sys/qos.h>`).
/// - **Linux**, which Metal does not run on. Each handler thread asks the kernel to treat it as
///   fully busy, as the user-interactive class does on macOS: a utilization floor of 1024
///   (`sched_setattr(2)`), its policy and nice value as they were. A kernel built without the
///   clamps refuses it, and the thread runs as before. The thread also has
///   `SCHED_FLAG_RESET_ON_FORK`, so a thread a handler starts begins with no floor, and at
///   `SCHED_OTHER` and nice 0 where the handler thread runs at a real-time policy or a negative
///   nice value; a handler cannot clear the setting without `CAP_SYS_NICE`. The first such floor
///   since the host booted turns the kernel's clamp accounting on for every task until it
///   reboots.
pub type Handler = Box<dyn FnOnce(&CommandBuffer) + Send + 'static>;

/// A Metal command buffer (`MTLCommandBuffer`). Cheap to clone (a retain); `Send + Sync`.
#[derive(Clone)]
pub struct CommandBuffer(Arc<CommandBufferShared>);

impl std::fmt::Debug for CommandBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandBuffer")
            .field("status", &self.status())
            .finish()
    }
}

pub(super) struct CommandBufferShared {
    device: Device,
    queue: CommandQueue,
    state: Mutex<State>,
    /// Signaled when the command buffer's status changes (`wait_until_scheduled` waits on it).
    changed: Condvar,
    /// What `wait_until_completed` waits for: the command buffer finished, and the runtime's
    /// threads let go of it.
    gate: Arc<Gate>,
    /// The `MTLCommandBufferError` code the driver relayed for one of its batches, with Metal's
    /// description, empty until one (`device/relay.rs`).
    relayed: Arc<Relayed>,
    /// The numbers its host steps give the variant instances they make, drawn at its commit in
    /// the program's call (`pipeline/kept.rs`).
    step_numbers: OnceLock<StepNumbers>,
    /// The textures its encoders made for their own commands (`Device::new_planned_texture`),
    /// held from their making until it completes, whichever of its commands use them (its
    /// first batch takes them, `submit.rs`). The device's next submission moves each one's image
    /// from `UNDEFINED` to `GENERAL` (`texture/first_move.rs`). Apart from the state, so a texture
    /// is made under no lock of the command buffer's.
    own_textures: Mutex<Vec<Texture>>,
    /// Rebinds since its commit, of buffers its replayable encoders named, whose encoders are
    /// not yet encoded again (`reencode.rs`): its first submission waits until none is. Raised
    /// under the device's lock of its buffers over host pages (`Rebinds::replace`), which its
    /// first submission reads it under; apart from the state, which that lock comes after.
    reencode: AtomicU64,
    /// Held while its encoders are encoded again, one rebind's turn at a time, and while a host
    /// step records: the recordings after its commit into its pool take turns, as Vulkan
    /// synchronizes a pool's command buffers' recording externally.
    reencoding: Mutex<()>,
}

/// How [`CommandBuffer::commit_failed`] fails a command buffer: a runtime extension, no Metal
/// type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CommitFailure {
    /// The `MTLCommandBufferError` code it fails with: 2 for a GPU hang (`Timeout`), 8 for out
    /// of memory, 1 or 16 for a protection fault, among others. 0 for none: the encoders it names
    /// run nothing, and it completes with status `Completed` and no error, as Metal completes a
    /// command buffer of a queue whose GPU hang is in flight.
    pub code: u64,
    /// The encoders that run nothing, each by its place among those the command buffer ended,
    /// in the order they ended, from 0 (the index of [`CommandBuffer::ended_encoders`]); the
    /// others run.
    pub encoders: FailedEncoders,
    /// Whether its event signals (`encode_signal_event`) fire, as a GPU hang's do; none fires
    /// where false, as for a protection fault.
    pub signals: bool,
}

impl CommitFailure {
    /// A failure of `code` in which no encoder runs and the event signals fire.
    pub fn new(code: u64) -> CommitFailure {
        CommitFailure {
            code,
            encoders: FailedEncoders::From(0),
            signals: true,
        }
    }

    /// Why `commit_failed` refuses the failure, the command buffer having ended `ended`
    /// encoders: a set that names an encoder past them.
    pub(super) fn problem(&self, ended: usize) -> Option<Problem> {
        let FailedEncoders::Set(set) = &self.encoders else {
            return None;
        };
        let past = set.range(ended..).next()?;
        let saturated = |count: usize| u32::try_from(count).unwrap_or(u32::MAX);
        Some(Problem::IndexOutOfRange {
            index: saturated(*past),
            limit: saturated(ended),
        })
    }
}

/// How a committing command buffer fails: as `commit_failed` asks, from an encoder Metal fails
/// it from (`State::fails_from`), or both.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct Failure {
    /// The encoders that run nothing.
    pub(super) encoders: FailedEncoders,
    /// The event signals that fire.
    pub(super) signals: FailedSignals,
    /// The error it completes with, if any.
    pub(super) error: Option<CommandBufferError>,
}

/// Which event signals of a failing command buffer fire.
#[derive(Debug, PartialEq, Eq)]
pub(super) enum FailedSignals {
    /// Every one (`CommitFailure::signals`).
    Every,
    /// None.
    Dropped,
    /// Those encoded before ended encoder `n`: Metal fails a command buffer from a render pass
    /// without attachments past 16384 a side, and signals what was encoded before the pass,
    /// not after it (EV RRS-probe-13b-m4).
    From(usize),
}

impl Failure {
    /// Whether ended encoder `encoder` runs under `failure`: every one does but those it fails.
    pub(super) fn runs(failure: Option<&Failure>, encoder: usize) -> bool {
        failure.is_none_or(|failure| !failure.encoders.fails(encoder))
    }

    /// The failure of a command buffer of `ended` ended encoders that `commit_failed` fails as
    /// `injected` asks (none for `commit`), and that Metal fails from ended encoder `from` with
    /// its error: the encoders either fails, and `injected`'s code where it is given, else
    /// `from`'s error. `None` where neither fails it.
    pub(super) fn of(
        injected: Option<CommitFailure>,
        from: Option<(usize, CommandBufferError)>,
        ended: usize,
    ) -> Option<Failure> {
        let injected_signals = |signals: bool| match signals {
            true => FailedSignals::Every,
            false => FailedSignals::Dropped,
        };
        // A code of 0 is no error: the command buffer completes without one.
        let injected_error = |code: u64| (code != 0).then_some(CommandBufferError::Injected(code));
        Some(match (injected, from) {
            (None, None) => return None,
            (Some(failure), None) => Failure {
                encoders: failure.encoders,
                signals: injected_signals(failure.signals),
                error: injected_error(failure.code),
            },
            (None, Some((from, error))) => Failure {
                encoders: FailedEncoders::From(from),
                signals: FailedSignals::From(from),
                error: Some(error),
            },
            (Some(failure), Some((from, error))) => Failure {
                encoders: match failure.encoders {
                    FailedEncoders::From(first) => FailedEncoders::From(first.min(from)),
                    FailedEncoders::Set(mut set) => {
                        set.extend(from..ended);
                        FailedEncoders::Set(set)
                    }
                },
                signals: match failure.signals {
                    true => FailedSignals::From(from),
                    false => FailedSignals::Dropped,
                },
                error: injected_error(failure.code).or(Some(error)),
            },
        })
    }

    /// `failure` (`Failure::of`'s) with the encoders Metal's GPU faults, `faulted`, of a command
    /// buffer of `ended` ended encoders: they run nothing too, and, where nothing else fails it,
    /// the command buffer completes with `CommandBufferError::PageFault` and every event signal
    /// fires, as Metal's does. Where something else fails it too, its error and its signals
    /// stand (Metal's error there is not measured).
    pub(super) fn with_faults(
        failure: Option<Failure>,
        faulted: &BTreeSet<usize>,
        ended: usize,
    ) -> Option<Failure> {
        if faulted.is_empty() {
            return failure;
        }
        Some(match failure {
            None => Failure {
                encoders: FailedEncoders::Set(faulted.clone()),
                signals: FailedSignals::Every,
                error: Some(CommandBufferError::PageFault),
            },
            Some(failure) => Failure {
                // Under a failure of no error, a faulted encoder it lets run faults it.
                error: failure.error.or_else(|| {
                    faulted
                        .iter()
                        .any(|&encoder| !failure.encoders.fails(encoder))
                        .then_some(CommandBufferError::PageFault)
                }),
                encoders: match failure.encoders {
                    FailedEncoders::From(first) if faulted.iter().all(|&e| e >= first) => {
                        FailedEncoders::From(first)
                    }
                    FailedEncoders::From(first) => {
                        FailedEncoders::Set(faulted.iter().copied().chain(first..ended).collect())
                    }
                    FailedEncoders::Set(mut set) => {
                        set.extend(faulted);
                        FailedEncoders::Set(set)
                    }
                },
                ..failure
            },
        })
    }
}

/// The encoders a [`CommitFailure`] fails, each by its place among those the command buffer
/// ended, from 0: a runtime extension, no Metal type.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FailedEncoders {
    /// This encoder and every one after it, as a GPU hang fails the hung encoder's: the
    /// encoders before it run. One past the ended encoders fails none.
    From(usize),
    /// These encoders alone: every other encoder runs, before and after them, as Metal runs
    /// every submission but the one that failed (a submission that uses a resource whose
    /// memory was taken away, measured by a consumer on an M3 Max, macOS 26.7). Each names an
    /// ended encoder; an empty set fails none.
    Set(BTreeSet<usize>),
}

impl FailedEncoders {
    /// Whether encoder `encoder` runs nothing.
    pub(super) fn fails(&self, encoder: usize) -> bool {
        match self {
            FailedEncoders::From(first) => encoder >= *first,
            FailedEncoders::Set(set) => set.contains(&encoder),
        }
    }
}

/// A command buffer's mutable state.
pub(super) struct State {
    status: CommandBufferStatus,
    error: Option<CommandBufferError>,
    /// An encoder is open.
    pub(super) encoder_open: bool,
    /// An encoder was dropped without `end_encoding`.
    pub(super) abandoned: bool,
    /// `commit` is recording the encoders' segments (`encoder.rs`), outside this lock: nothing
    /// else may encode or commit meanwhile.
    committing: bool,
    /// The pool every `VkCommandBuffer` of this command buffer comes from, with those made from
    /// it; `None` until the first, and once given back to the device's spares.
    pool: Option<CommandPool>,
    /// The ended encoders' recordings, in encoding order.
    recorded: Vec<Recorded>,
    /// The encoders split at the event and binary semaphore waits and signals.
    batches: Batches<Arc<EventShared>, vk::Semaphore>,
    scheduled: VecDeque<Handler>,
    /// Its submission handlers ([`CommandBuffer::add_submission_handler`]): run on the thread
    /// that submits its last batch, before its transition to `Scheduled`.
    submission: VecDeque<Handler>,
    /// The thread that submitted its last batch has its submission handlers to run: its
    /// transition to `Scheduled` waits until they ran.
    submitting: bool,
    /// Its scheduled handlers ran: `wait_until_scheduled` returns, as Metal's does only once
    /// they ran (EV SUB-sched1-m4).
    scheduled_ran: bool,
    /// Why a batch could not be submitted: the command buffer finishes with it, and its
    /// scheduled handlers never run. Or the failure `commit_failed` chose, with which it
    /// finishes once its batches completed, its scheduled handlers run.
    failure: Option<CommandBufferError>,
    completed: VecDeque<Handler>,
    /// The handlers of its executions' records as the runtime reads them
    /// (`executed_commands.rs`).
    pub(crate) executed_handlers: Vec<crate::executed_commands::ExecutedCommandsHandler>,
    /// What the ended encoders' commands use, held until the command buffer completed and its
    /// completed handlers ran. A drop may run the caller's code (a no-copy buffer's deallocator),
    /// which may use this command buffer, so each is released outside this state's lock.
    pub(super) retained: Vec<Box<dyn Any + Send + Sync>>,
    /// What the ended encoders' dispatches and draws executed (`execution.rs`), kept after the
    /// command buffer completed.
    pub(super) executions: Vec<Execution>,
    /// The commands' own data, released when the command buffer completes.
    pub(super) uploads: Uploads,
    /// The device's heap sets its commands bound, and its listing of the argument samplers
    /// (`heap_sets.rs`), released with the uploads.
    pub(super) heaps: HeapUse,
    /// Its commands' function-table snapshots, written again as it commits.
    table_snapshots: Vec<TableSnapshot>,
    /// Its commands' sampler tables, listed again as it commits (`sampler_listing.rs`).
    sampler_listings: Vec<SamplerListing>,
    /// As it committed, some command's plan could not list the sampler states live then
    /// (`sampler_listing.rs`), or a command a host step recorded after the commit left a live
    /// state out of its argument samplers (`State::unlisted_argument_sampler`): it completes with
    /// `CommandBufferError::UnlistedArgumentSampler` unless another error fails it first.
    unlisted_argument_samplers: bool,
    /// The first ended encoder Metal fails the command buffer from, with the error (a render
    /// pass without attachments past 16384 a side, `CommandBufferError::RenderTargetSize`, or
    /// at a default sample count Metal fails it at, `DefaultRasterSampleCount`): as it commits,
    /// that encoder and every one after it run nothing, and it completes with that error.
    fails_from: Option<(usize, CommandBufferError)>,
    /// The ended encoders Metal's GPU faults (an execution of commands that inherit a pipeline
    /// state made without `supportIndirectCommandBuffers`): as it commits, they run nothing, and
    /// it completes with `CommandBufferError::PageFault`.
    faulted: BTreeSet<usize>,
    /// The pools of its commands' set 0 (`descriptor_arena.rs`), given back or released with
    /// the uploads.
    pub(super) descriptors: DescriptorArena,
    /// The execution status words the commands share, once a command reaches them.
    status_words: Option<Upload>,
    /// The captured passes' tile masks, which its completion counts (`render/capture.rs`).
    pub(super) capture_tallies: Vec<CaptureTally>,
    /// The index pool of the commands' index bounds and rewrite passes (`index_bounds.rs`,
    /// `index_rewrite.rs`), once a draw needs it: its counter in the arena, starting at 0, and
    /// the block of the device's its indices lie in (`gpu_pools.rs`).
    index_pool: Option<(Upload, PoolLease)>,
    /// The records pool of the commands' scanned emulated mesh draws (`render/mesh.rs`), once a
    /// draw needs it: a block of the device's (`gpu_pools.rs`); the scans' cursor is a status
    /// word.
    mesh_pool: Option<PoolLease>,
    /// The emulated mesh draws numbered so far, each by its order.
    mesh_draws: u32,
    /// The resident-set misses the segments' recordings found on the host, which the report adds to
    /// the words' count.
    resident_misses: u32,
    /// Why some of the segments' commands ran nothing, which the report states.
    faults: ReplayFaults,
    /// The dispatches of the segments placed so far, each among the executions.
    placed: usize,
    /// Why a host step failed: the batch it recorded for is not submitted, and the command buffer
    /// finishes with it.
    step_failure: Option<CommandBufferError>,
    /// The refusal of a host step that failed so (`CommandBufferError::Refused`).
    refusal: Option<Error>,
    /// The first draw's pipeline whose making failed, which a host step met
    /// (`CommandBufferError::PipelineCreation`).
    failed_pipeline: Option<FailedPipeline>,
    /// Metal's description of the failure the driver relayed, where the command buffer failed
    /// with it (`CommandBuffer::error_description`).
    relayed_description: Option<String>,
    /// The decoded status words, once the command buffer completed.
    report: Option<StatusReport>,
    /// Its GPU and kernel times (`gpu_time.rs`).
    times: Times,
    /// The slot of its queue it holds until it completes, or is dropped uncommitted
    /// (`queue.rs`).
    slot: Option<Slot>,
    /// The heaps its ended encoders' commands used, which its commit marks used
    /// (`Heap::set_purgeable_state`).
    used_heaps: Vec<Heap>,
    /// Its ended encoders' holds on the buffers over host pages their commands used
    /// (`rebind.rs`): let go at its commit, or, where a host step records commands after it, as
    /// it completes.
    pub(super) holds: Vec<Holds>,
    /// What its ended encoders made resident, every encoder's joined (`residency.rs`): the
    /// residency the work its commit and host steps record resolves and binds against. Released
    /// with what it retained, as it completes.
    residency: Option<Residency>,
    /// Its ended encoders, in the order they ended, each with what its commands use
    /// (`used_resources.rs`): the keys its commands' uses and bindings named, its placed
    /// segments' among them, and what it made resident. Their uses are let go as it completes.
    ended: Vec<EncoderRecord>,
    /// Its root tables, until its first submission points them at its resident set as the
    /// live buffers stand then (`first_submission.rs`); the commit points them first.
    root_tables: Vec<NotedTable>,
    /// The ranges of its root tables' own resident sets, each once.
    root_ranges: HashSet<ResidentRange>,
    /// The resident set its root tables search, once made; released as it completes.
    resident: Option<Resident>,
    /// What its commit listed among the live buffers (`LiveBuffers::reach`), and the list's
    /// generation then: listed again at its first submission where the list changed since.
    listing: Option<Listing>,
    /// Its first batch was submitted: from then on it holds what its commands took and the
    /// live buffers' table its resident set was made of until it completes.
    submitted: bool,
    /// Metal's submissions (`checks::metal_submissions`, by their ended encoders) its first
    /// submission found to name a buffer rebound since, or to hold a detached one, each with
    /// its failure: each fails as its own batch is submitted, after its check
    /// (`first_submission.rs`).
    failing: Vec<(Range<usize>, CommandBufferError)>,
    /// The ended encoders whose submissions failed so: no part of one goes to a queue with work
    /// (`submit.rs` holds it to that).
    unrunnable: BTreeSet<usize>,
    /// What its batches' commands took of buffers over host pages, held from its first
    /// submission (`rebind::SubmittedHolds`): let go as its work completes, before its
    /// completed handlers, so a detach waiting for it goes on once its GPU work ended.
    submitted_holds: Vec<SubmittedHolds>,
    /// The encoders its commit's failure failed (`commit_failed`'s, Metal's own from a render
    /// pass, or a faulted encoder's: [`State::fail`]): a submission none of whose encoders
    /// runs so is spared its rebound verdict (`first_submission.rs`).
    failed_at_commit: Option<FailedEncoders>,
    /// The program's submission check (`checks.rs`), released as it completes.
    check: Option<SubmissionCheck>,
    /// Metal's submissions of a committed command buffer with a check, and how far they were
    /// checked.
    checks: Option<Checks>,
    /// The fetched render targets' switch to their storage form riding on its commit
    /// (`texture/storage_form.rs`), until the scheduler settled it and a program thread finished
    /// it.
    storage_form: Option<Arc<Riding>>,
    /// Its units of command predication (`predication.rs`).
    pub(super) predication: Predication,
    /// An encoder of it used command predication: its commit lets go of every log, as a command
    /// encoded again after the commit would run outside its block.
    pub(super) predicated: bool,
}

/// The resident set a command buffer's root tables search (`live_buffers.rs`): the device's live
/// buffers at its commit, and what its encoders bound and made resident beyond them (`merged`,
/// uploaded to its arena): `count` entries at `address` (`None` for none).
struct Resident {
    live: Live,
    merged: Option<ResidentSet>,
    address: Option<u64>,
    count: u64,
}

impl State {
    /// Whether it was committed: a handler added now would not run (`Problem::HandlerAfterCommit`).
    pub(crate) fn enqueued(&self) -> bool {
        self.status != CommandBufferStatus::NotEnqueued
    }

    /// An encoder of `kind` ended, its commands' uses and bindings having named `keys`, having
    /// made `residency` resident (a compute or render encoder's): it joins the ended encoders,
    /// with the event waits and signals encoded before it, and what it made resident joins the
    /// command buffer's residency (`Residency::join_encoder`). Its place among the ended
    /// encoders, which each of its recordings' parts carries.
    pub(super) fn encoder_ended(
        &mut self,
        kind: EncoderKind,
        keys: BTreeSet<ResourceKey>,
        residency: Option<Residency>,
        taken: Taken,
    ) -> usize {
        if let Some(ended) = &residency {
            self.residency
                .get_or_insert_with(Residency::default)
                .join_encoder(ended, kind == EncoderKind::Compute);
        }
        let (event_waits_before, event_signals_before) = self.batches.events();
        self.ended.push(EncoderRecord {
            kind,
            event_waits_before,
            event_signals_before,
            keys,
            residency,
            taken,
            replay: None,
            executions: 0..0,
        });
        self.ended.len() - 1
    }

    /// Keep `replay`, the log of ended encoder `encoder`, whose executions take `executions`
    /// among the command buffer's. One of no such encoder comes back, for the caller to let go
    /// of without this lock (what it holds may be the last hold of a no-copy buffer).
    pub(super) fn keep_replay(
        &mut self,
        encoder: usize,
        replay: Option<Replay>,
        executions: Range<usize>,
    ) -> Option<Replay> {
        let replay = replay?;
        let Some(record) = self.ended.get_mut(encoder) else {
            return Some(replay);
        };
        record.replay = Some(Arc::new(replay));
        record.executions = executions;
        None
    }

    /// Its ended encoders `encoders` (all of them where `None`), each as `used_resources.rs`
    /// lists it: its kind, its event counts, every key its commands use and reach, and what it
    /// made resident. The keys are its own commands' and bindings', its placed segments', and
    /// those of each segment recorded and not yet placed (the commit's recording, or the host
    /// step's that replaced it): what the commands of an indirect command buffer it executes
    /// reach, from the commit on.
    pub(super) fn read_ended(&self, encoders: Option<Range<usize>>) -> Vec<ReadRecord> {
        let range = encoders.unwrap_or(0..self.ended.len());
        let Some(records) = self.ended.get(range.clone()) else {
            return Vec::new();
        };
        let mut read: Vec<ReadRecord> = records
            .iter()
            .map(|record| ReadRecord {
                kind: record.kind,
                event_waits_before: record.event_waits_before,
                event_signals_before: record.event_signals_before,
                keys: record.keys.clone(),
                residency: record.residency.clone(),
            })
            .collect();
        let unplaced = self
            .recorded
            .iter()
            .filter(|part| range.contains(&part.encoder))
            .flat_map(|part| {
                let keys = part
                    .segments
                    .iter()
                    .filter_map(|segment| segment.recorded.as_ref())
                    .flat_map(|done| done.keys.iter().copied());
                keys.map(move |key| (part.encoder - range.start, key))
            });
        for (at, key) in unplaced {
            if let Some(record) = read.get_mut(at) {
                record.keys.insert(key);
            }
        }
        read
    }

    /// What the commands of its ended encoders that `runs` (by their place among them) use and a
    /// command reaches unnamed only once used, gathered weakly under this lock ([`Reached`]): what
    /// their keys name, as [`read_ended`](Self::read_ended) reads them (each segment's recorded
    /// and not yet placed with its encoder's), and what they made resident.
    fn reached(
        &self,
        shared: &super::device::DeviceShared,
        runs: impl Fn(usize) -> bool,
    ) -> Reached {
        let mut reached = Reached::default();
        for (_, record) in (self.ended.iter().enumerate()).filter(|(at, _)| runs(*at)) {
            reached.gather(shared, &record.keys, record.residency.as_ref());
        }
        for part in self.recorded.iter().filter(|part| runs(part.encoder)) {
            let segments = part.segments.iter();
            for done in segments.filter_map(|segment| segment.recorded.as_ref()) {
                reached.gather(shared, &done.keys, None);
            }
        }
        reached
    }

    /// Let go of what its ended encoders' commands use, as it completes: their keys, and what
    /// they made resident, for the caller to release without this lock.
    fn let_go_of_uses(&mut self) -> Vec<Residency> {
        self.ended
            .iter_mut()
            .filter_map(|record| {
                record.keys.clear();
                record.residency.take()
            })
            .collect()
    }

    /// Note a root table its commands read, uploaded at `table` (`upload`), whose own resident
    /// set is `ranges`: the commit points it at the command buffer's resident set. One a host
    /// step uploads after the commit is pointed now, at that set where it holds `ranges`, else
    /// at a set of both made for it.
    pub(super) fn root_table(
        &mut self,
        device: &Device,
        noted: NotedTable,
        ranges: Vec<ResidentRange>,
    ) -> Result<(), Error> {
        if self.status == CommandBufferStatus::NotEnqueued {
            self.root_ranges.extend(ranges);
            self.root_tables.push(noted);
            return Ok(());
        }
        // A host step's before the first submission is noted too, which points it again where
        // the live buffers changed by then; the set it is pointed at now is made of the list as
        // it stands.
        if !self.submitted {
            self.root_ranges.extend(ranges.iter().copied());
            self.root_tables.push(noted.clone());
            self.resident(device)?;
            self.hold_live(device)?;
        }
        let NotedTable {
            upload,
            address: table,
        } = noted;
        let resident = self.resident(device)?;
        // Held from the first submission, and from a host step's before it (above).
        let set = resident
            .merged
            .as_ref()
            .or(resident.live.set())
            .ok_or_else(|| internal("a host step's root table before its live table is held"))?;
        if ranges.iter().all(|range| set.holds(range)) {
            return self.point(&upload, table);
        }
        let both = set.with(ranges);
        let (address, count) = self.upload_set(device, &both)?;
        write_fields(&upload, table, address, count)
    }

    /// Point every root table noted before the commit at the command buffer's resident set.
    fn point_root_tables(&mut self, device: &Device) -> Result<(), Error> {
        if self.root_tables.is_empty() {
            return Ok(());
        }
        self.resident(device)?;
        for noted in &self.root_tables {
            self.point(&noted.upload, noted.address)?;
        }
        Ok(())
    }

    /// The command buffer's resident set, made on first need: the device's live buffers, and
    /// the ranges its root tables noted and its encoders made resident beyond them.
    fn resident(&mut self, device: &Device) -> Result<&Resident, Error> {
        if self.resident.is_none() {
            let resident = self.make_resident(device)?;
            self.resident = Some(resident);
        }
        self.resident
            .as_ref()
            .ok_or_else(|| internal("a command buffer's resident set not made"))
    }

    fn make_resident(&mut self, device: &Device) -> Result<Resident, Error> {
        let live = device.shared().live_buffers.table(device)?;
        let set = live.set();
        let mut beyond: Vec<ResidentRange> = self.root_ranges.iter().copied().collect();
        if let Some(residency) = &self.residency {
            beyond.extend(residency.resident_ranges());
        }
        beyond.retain(|range| !set.holds(range));
        let (merged, address, count) = match beyond.is_empty() {
            true => (None, live.address(), set.len()),
            false => {
                let merged = set.with(beyond);
                let (address, count) = self.upload_set(device, &merged)?;
                (Some(merged), Some(address), count)
            }
        };
        // Held weakly until the first submission, which makes the set again where the list
        // changed since.
        Ok(Resident {
            live: Live::of(live, self.submitted),
            merged,
            address,
            count,
        })
    }

    /// `set` uploaded to the arena at a 32-byte-aligned device address: that address and its
    /// count.
    fn upload_set(&mut self, device: &Device, set: &ResidentSet) -> Result<(u64, u64), Error> {
        let bytes = set.bytes();
        let upload = self.uploads.upload_aligned(
            device,
            &bytes,
            bytes.len() as u64,
            RootTable::ENTRY_ALIGNMENT,
        )?;
        let address = upload
            .device_address()
            .ok_or_else(|| internal("a resident set without a device address"))?;
        Ok((address, set.len()))
    }

    /// Point the root table at `table` (`upload`) at the command buffer's resident set, once
    /// made.
    fn point(&self, upload: &Upload, table: u64) -> Result<(), Error> {
        match &self.resident {
            Some(resident) => write_fields(
                upload,
                table,
                resident.address.unwrap_or(table),
                resident.count,
            ),
            None => Ok(()),
        }
    }

    /// Note `heaps`, which an ended encoder's commands used.
    pub(super) fn use_heaps(&mut self, heaps: Vec<Heap>) {
        for heap in heaps {
            if !self.used_heaps.contains(&heap) {
                self.used_heaps.push(heap);
            }
        }
    }

    /// Why nothing more can be encoded or committed now: the command buffer was committed, an
    /// encoder is open, or one was abandoned.
    pub(super) fn refusal(&self) -> Option<Problem> {
        if self.status != CommandBufferStatus::NotEnqueued || self.committing {
            Some(Problem::AlreadyCommitted)
        } else if self.encoder_open {
            Some(Problem::EncoderOpen)
        } else if self.abandoned {
            Some(Problem::EncoderAbandoned)
        } else {
            None
        }
    }

    /// Keep a command's function-table `snapshot` to write again as the command buffer commits. A
    /// snapshot a host step writes after the commit already holds the words as they stand then.
    pub(super) fn snapshot_tables(&mut self, snapshot: TableSnapshot) {
        if self.status == CommandBufferStatus::NotEnqueued {
            self.table_snapshots.push(snapshot);
        }
    }

    /// Keep a command's sampler tables to list the argument samplers again as the command buffer
    /// commits (`sampler_listing.rs`). A command a host step records after the commit lists the
    /// states as they stand then, and one that leaves a live state out fails the command buffer
    /// (`State::unlisted_argument_sampler`).
    pub(super) fn list_samplers(&mut self, listing: SamplerListing) {
        if self.status == CommandBufferStatus::NotEnqueued {
            self.sampler_listings.push(listing);
        }
    }

    /// A command left a live sampler state out of its argument samplers, its variant's set-1
    /// sampler arrays holding fewer elements than the live states (`heap_sets::fit_samplers`).
    /// Before the commit nothing follows here: the commit lists the command's samplers again and
    /// fails the command buffer where the state still lives (`sampler_listing.rs`). A command a
    /// host step records after the commit is listed by no commit, and its function would read the
    /// neutral sampler where Metal's reaches the state by its ID, so the command buffer fails as it
    /// completes (`CommandBufferError::UnlistedArgumentSampler`), counted once
    /// (`WorkCounts::unlisted_argument_sampler_failures`).
    pub(super) fn unlisted_argument_sampler(&mut self, device: &Device) {
        if self.status != CommandBufferStatus::NotEnqueued && !self.unlisted_argument_samplers {
            self.unlisted_argument_samplers = true;
            device.shared().counts.unlisted_argument_sampler_failure();
        }
    }

    /// The place among the ended encoders the open encoder takes as it ends: one encoder is open
    /// at a time, and one dropped before it ended abandons the command buffer, which then
    /// commits nothing.
    pub(super) fn open_encoder_place(&self) -> usize {
        self.ended.len()
    }

    /// A compute encoder of `dispatch_type` opened: it joins the unit of command predication of
    /// the compute encoder that ended just before it, of its dispatch type with no event wait or
    /// signal between, or starts the next (`predication.rs`).
    pub(super) fn enter_predication(&mut self, dispatch_type: DispatchType) {
        let place = self.open_encoder_place();
        self.predication
            .enter(place, self.batches.events(), dispatch_type);
    }

    /// Ended encoder `encoder` fails the command buffer from it on with `error` (`fails_from`).
    pub(super) fn fail_from(&mut self, encoder: usize, error: CommandBufferError) {
        self.fails_from.get_or_insert((encoder, error));
    }

    /// Metal's GPU faults ended encoder `encoder` (`faulted`).
    pub(super) fn fault(&mut self, encoder: usize) {
        self.faulted.insert(encoder);
    }

    /// The encoders (or parts of them) of batch `index`.
    pub(super) fn batch_range(&self, index: usize) -> Range<usize> {
        let batches = self.batches.all();
        let start = index.checked_sub(1).map_or(0, |before| batches[before].end);
        start..batches[index].end
    }

    /// The next encoder, a part of one that executes commands the GPU may write, starts a batch
    /// of its own, which waits for a host step.
    pub(super) fn host_step(&mut self) {
        self.batches.host_step();
    }

    /// A resource state encoder ended with `recorded`: it starts a batch of its own, unless the
    /// batch so far holds only such encoders, so its updates bind before every encoder after it.
    pub(super) fn end_sparse_encoder(&mut self, recorded: Recorded) {
        self.batches.sparse();
        self.end_encoder(recorded);
    }

    /// An encoder ended with `recorded`: it joins the current batch, or a new one after the
    /// batch's signals.
    pub(super) fn end_encoder(&mut self, recorded: Recorded) {
        self.recorded.push(recorded);
        self.batches.encoder();
        self.encoder_open = false;
    }

    /// Fail the command buffer as `failure` says ([`CommandBuffer::commit_failed`], or an
    /// encoder Metal fails it from, `fails_from`), before `commit` hands its work on the segments
    /// of the encoders that run to them: the recordings of the encoders `failure.encoders` names
    /// (every part of each) are released unrun, each part keeping its place among the batches
    /// with its fences' waits and updates alone; the event signals `failure.signals` drops are
    /// not signaled; and the command buffer finishes with `failure.error`, where it has one
    /// (a code-0 `commit_failed` has none).
    fn fail(&mut self, failure: Failure) {
        let State {
            recorded, retained, ..
        } = self;
        for part in recorded
            .iter_mut()
            .filter(|part| failure.encoders.fails(part.encoder))
        {
            State::fail_part(part, retained);
        }
        match failure.signals {
            FailedSignals::Every => {}
            FailedSignals::Dropped => self.batches.drop_signals(),
            FailedSignals::From(encoder) => {
                // The parts before the encoder's first, which the batches count.
                let parts = self
                    .recorded
                    .iter()
                    .take_while(|part| part.encoder < encoder)
                    .count();
                self.batches.drop_signals_after(parts);
            }
        }
        if let Some(error) = failure.error {
            self.failure = Some(error);
        }
        self.failed_at_commit = Some(failure.encoders);
    }

    /// A failed encoder's recorded `part` runs nothing: its segments (with their recordings)
    /// and sparse updates go to `retained`, released with what the command buffer retained
    /// without this lock, its `VkCommandBuffer`s are left unsubmitted, and it keeps its place
    /// among the batches with its fences' waits and updates alone.
    pub(super) fn fail_part(part: &mut Recorded, retained: &mut Vec<Box<dyn Any + Send + Sync>>) {
        for segment in part.segments.drain(..) {
            retained.push(Box::new((segment.work, segment.inputs)));
            if let Some(mut done) = segment.recorded {
                retained.extend(done.retained.into_entry());
                done.taken.retain_into(retained);
            }
        }
        for update in part.sparse.drain(..) {
            retained.push(Box::new(update));
        }
        part.raws.clear();
        part.summary = part.summary.fences_only();
    }

    /// Hand `commit`'s work on the segments, one per segment in encoding order, to them: the
    /// recording of a segment `commit` made, for good where no host step records it, and what a
    /// host step needs.
    fn prepare(&mut self, prepared: Vec<Prepared>) {
        let segments = self
            .recorded
            .iter_mut()
            .flat_map(|encoder| encoder.segments.iter_mut());
        for (segment, prepared) in segments.zip(prepared) {
            let Prepared {
                recorded,
                for_good,
                inputs,
                retained,
                pipelines,
            } = prepared;
            segment.recorded = recorded;
            segment.stepped |= for_good;
            segment.pipelines |= pipelines;
            segment.inputs = Some(inputs);
            self.retained.extend(retained);
        }
    }

    /// Put the segments of encoders (or parts of them) `range`, each recorded, in their places as
    /// their batch is submitted: each `VkCommandBuffer` among its part's, its uses joined to the
    /// part's summary, its dispatches among the executions in encoding order, and what it used
    /// retained. Refused, changing nothing, where a segment is not recorded.
    fn place(&mut self, range: Range<usize>) -> Result<(), Error> {
        let State {
            recorded,
            executions,
            retained,
            resident_misses,
            faults,
            placed,
            ended,
            ..
        } = self;
        let encoders = &mut recorded[range];
        if encoders
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .any(|segment| segment.recorded.is_none())
        {
            return Err(internal("a segment submitted unrecorded"));
        }
        // A segment that recorded nothing for a command (a buffer it named was detached) is
        // never placed: its submission failed (`State::failing_at`).
        let dead = encoders
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .filter_map(|segment| segment.recorded.as_ref())
            .any(|done| done.taken.dead());
        debug_assert!(!dead, "a segment that named a detached buffer submitted");
        if dead {
            return Err(internal("a segment that named a detached buffer submitted"));
        }
        for encoder in encoders {
            for (before, segment) in encoder.segments.drain(..).enumerate() {
                let Segment {
                    at,
                    executions_at,
                    work,
                    inputs,
                    recorded,
                    ..
                } = segment;
                // Released with what the command buffer retained, without this lock.
                retained.push(Box::new((work, inputs)));
                let Some(mut done) = recorded else {
                    continue;
                };
                // The pages its commands took, held until the command buffer completes.
                done.taken.retain_into(retained);
                encoder.raws.insert(at + before, done.raw);
                encoder.summary.join(done.summary);
                let at = executions_at + *placed;
                *placed += done.executions.len();
                executions.splice(at..at, done.executions);
                retained.extend(done.retained.into_entry());
                // The segment's keys are its encoder's.
                if let Some(record) = ended.get_mut(encoder.encoder) {
                    record.keys.extend(done.keys);
                }
                *resident_misses = resident_misses.wrapping_add(done.resident_misses);
                *faults = ReplayFaults(faults.0 | done.faults.0);
            }
        }
        Ok(())
    }

    /// What the segments a failed command buffer never placed retained, for its release.
    fn unplaced(&mut self) {
        let leftover: Vec<_> = self
            .recorded
            .iter_mut()
            .flat_map(|encoder| encoder.segments.drain(..))
            .collect();
        for segment in leftover {
            let Segment {
                work,
                inputs,
                recorded,
                ..
            } = segment;
            self.retained.push(Box::new((work, inputs)));
            if let Some(mut done) = recorded {
                self.retained.extend(done.retained.into_entry());
                done.taken.retain_into(&mut self.retained);
            }
        }
    }

    /// The device address of the command buffer's execution status words, uploaded (all zero,
    /// `ExecutionStatus::initial`) when a command first reaches them; `None` on a device without
    /// device addresses.
    pub(super) fn status_address(&mut self, device: &Device) -> Result<Option<u64>, Error> {
        Ok(self.status_words(device)?.device_address())
    }

    /// The command buffer's execution status words, uploaded (all zero, `ExecutionStatus::initial`)
    /// when a command first reaches them.
    pub(super) fn status_words(&mut self, device: &Device) -> Result<Upload, Error> {
        if let Some(words) = &self.status_words {
            return Ok(words.clone());
        }
        let words = self.uploads.upload(
            device,
            &ExecutionStatus::initial(),
            ExecutionStatus::BYTES as u64,
        )?;
        self.status_words = Some(words.clone());
        Ok(words)
    }

    /// The command buffer's index pool (`index_bounds.rs`, `index_rewrite.rs`), taken when a
    /// draw first needs it: its counter, `IndexBoundsPass::POOL_HEADER_BYTES` of the arena, zero;
    /// and the block its indices lie in, from the block's
    /// `IndexBoundsPass::POOL_HEADER_BYTES` on, leased from the device (`gpu_pools.rs`): the
    /// block's `VkBuffer` and bytes, the header's included.
    pub(super) fn index_pool(&mut self, device: &Device) -> Result<IndexPool, Error> {
        if let Some((counter, block)) = &self.index_pool {
            return Ok(IndexPool::of(counter, block));
        }
        let header = IndexBoundsPass::POOL_HEADER_BYTES;
        let counter = self
            .uploads
            .upload(device, &vec![0; header as usize], header)?;
        let block = device.lease_pool(PoolKind::Index)?;
        let pool = IndexPool::of(&counter, &block);
        self.index_pool = Some((counter, block));
        Ok(pool)
    }

    /// The command buffer's mesh records pool (`render/mesh.rs`): a block of the device's
    /// budget, leased when a scanned emulated mesh draw first needs it, a host step's replay of
    /// one among them (`gpu_pools.rs`): its `VkBuffer` and bytes. Its words need no value before
    /// the scans reserve them.
    pub(super) fn mesh_pool(&mut self, device: &Device) -> Result<(vk::Buffer, u64), Error> {
        if let Some(pool) = &self.mesh_pool {
            return Ok((pool.raw(), pool.bytes()));
        }
        let pool = device.lease_pool(PoolKind::MeshRecords)?;
        let leased = (pool.raw(), pool.bytes());
        self.mesh_pool = Some(pool);
        Ok(leased)
    }

    /// The number of the command buffer's next emulated mesh draw: its order among them, from 0
    /// (`MeshDispatch::DRAW_ID`).
    pub(super) fn next_mesh_draw(&mut self) -> u32 {
        let number = self.mesh_draws;
        self.mesh_draws = number.saturating_add(1);
        number
    }
}

impl Drop for CommandBufferShared {
    fn drop(&mut self) {
        // A command buffer that still holds its pool was never submitted, or ended before its
        // completion gave the pool back: a committed command buffer is held by the scheduler until
        // its batches were submitted, and by the completion thread until they completed, so
        // nothing of its pool is pending. The device's spares keep it (`spares.rs`), which
        // release it only with the device. A timestamp slot it still holds goes back with its
        // state.
        let pool = lock(&self.state).pool.take();
        if let Some(pool) = pool {
            self.device.shared().spares.give_command_pool(pool);
        }
    }
}

/// Point the root table at `table` (`upload`) at `count` entries of a resident set at `set`.
fn write_fields(upload: &Upload, table: u64, set: u64, count: u64) -> Result<(), Error> {
    let fields = RootTable::resident_fields(table, set, count).ok_or_else(|| {
        internal("a root table not a whole number of entries from its resident set")
    })?;
    upload.write_at(u64::from(RootTable::RESIDENT_FIELDS_OFFSET), &fields);
    Ok(())
}

impl CommandBuffer {
    /// A new command buffer of `queue`, holding `slot` of it.
    pub(super) fn new(queue: &CommandQueue, slot: Slot) -> CommandBuffer {
        CommandBuffer(Arc::new(CommandBufferShared {
            device: queue.device().clone(),
            queue: queue.clone(),
            state: Mutex::new(State {
                status: CommandBufferStatus::NotEnqueued,
                error: None,
                encoder_open: false,
                abandoned: false,
                committing: false,
                pool: None,
                recorded: Vec::new(),
                batches: Batches::default(),
                scheduled: VecDeque::new(),
                submission: VecDeque::new(),
                submitting: false,
                scheduled_ran: false,
                failure: None,
                completed: VecDeque::new(),
                executed_handlers: Vec::new(),
                retained: Vec::new(),
                executions: Vec::new(),
                uploads: Uploads::new(),
                heaps: HeapUse::default(),
                table_snapshots: Vec::new(),
                sampler_listings: Vec::new(),
                unlisted_argument_samplers: false,
                fails_from: None,
                faulted: BTreeSet::new(),
                descriptors: DescriptorArena::default(),
                status_words: None,
                capture_tallies: Vec::new(),
                index_pool: None,
                mesh_pool: None,
                mesh_draws: 0,
                resident_misses: 0,
                faults: ReplayFaults::default(),
                placed: 0,
                step_failure: None,
                refusal: None,
                failed_pipeline: None,
                relayed_description: None,
                report: None,
                times: Times::default(),
                slot: Some(slot),
                used_heaps: Vec::new(),
                holds: Vec::new(),
                residency: None,
                ended: Vec::new(),
                root_tables: Vec::new(),
                root_ranges: HashSet::new(),
                resident: None,
                listing: None,
                submitted: false,
                failing: Vec::new(),
                unrunnable: BTreeSet::new(),
                submitted_holds: Vec::new(),
                failed_at_commit: None,
                check: None,
                checks: None,
                storage_form: None,
                predication: Predication::default(),
                predicated: false,
            }),
            changed: Condvar::new(),
            gate: Arc::default(),
            relayed: Arc::default(),
            step_numbers: OnceLock::new(),
            own_textures: Mutex::new(Vec::new()),
            reencode: AtomicU64::new(0),
            reencoding: Mutex::new(()),
        }))
    }

    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// Hold `texture`, which one of its encoders made for its own commands, as its
    /// `own_textures` says.
    pub(super) fn hold_own_texture(&self, texture: Texture) {
        lock(&self.0.own_textures).push(texture);
    }

    /// `commandQueue`.
    pub fn command_queue(&self) -> &CommandQueue {
        &self.0.queue
    }

    /// The state, locked.
    pub(super) fn state(&self) -> MutexGuard<'_, State> {
        lock(&self.0.state)
    }

    /// The residency a new compute or render (`render`) encoder starts with
    /// (`Residency::shadows`): the shadowed textures its ended encoders made resident, those in
    /// the heaps they used made since included.
    pub(super) fn inherited_residency(&self, render: bool) -> Residency {
        let mut state = self.state();
        let Some(residency) = &mut state.residency else {
            return Residency::default();
        };
        residency.sync_heaps();
        residency.shadows(render)
    }

    /// What its ended encoders made resident, every encoder's joined (`residency.rs`): what the
    /// work its commit and host steps record resolves and binds against.
    pub(super) fn residency(&self) -> Result<Residency, Error> {
        let residency = self.state().residency.clone().unwrap_or_default();
        // With the device's live buffers, which an address in no resource made resident
        // resolves against (`live_buffers.rs`).
        let device = &self.0.device;
        let live = device.shared().live_buffers.table(device)?;
        Ok(residency.with_live(live))
    }

    /// A primary `VkCommandBuffer` from this command buffer's pool, made on first use. The
    /// caller holds `state` (the pool is externally synchronized).
    pub(super) fn allocate_raw(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.0.allocate_raw(state)
    }

    /// A secondary `VkCommandBuffer` from this command buffer's pool (a render encoder's draws),
    /// freed with the pool. The caller holds `state`.
    pub(super) fn allocate_secondary(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.0.allocate(state, vk::CommandBufferLevel::SECONDARY)
    }

    /// `addScheduledHandler:`: call `handler` once the command buffer is scheduled, on one of the
    /// device's handler threads, at the user-interactive class on macOS as in Metal, and with a
    /// utilization floor on Linux ([`Handler`]). Refused once the command buffer is committed.
    pub fn add_scheduled_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.status != CommandBufferStatus::NotEnqueued {
            return Err(invalid(
                "add_scheduled_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.scheduled.push_back(handler);
        Ok(())
    }

    /// Call `handler` once the command buffer's last batch reached its `VkQueue`, on the thread
    /// that submitted it, before the command buffer becomes `Scheduled`: a runtime extension, no
    /// Metal call (docs/API.md, "A command buffer's submission handlers"). Metal submits on its
    /// own thread and tells a program of it only through `Scheduled` (EV SUB-sched1-m4); a
    /// program that models what Metal does as it submits takes that point here.
    ///
    /// - **The thread.** The committing thread, inside `commit`, where the commit submits the
    ///   last batch itself (its every wait met, nothing earlier of its queue held, and no other
    ///   thread submitting for the device as it commits); else the thread that submits it later:
    ///   the device's scheduler thread (after an event's value rose, a host step, a check, or a
    ///   commit that found it submitting), or the thread of another `commit` that submits it in
    ///   its pass (one that released it, or that was submitting as this commit was made). It
    ///   runs with no lock of the runtime's held, after the submission's call returned.
    /// - **Before.** Its handlers run, in the order added, before the status reads `Scheduled`,
    ///   before [`wait_until_scheduled`](Self::wait_until_scheduled) returns, before its
    ///   scheduled handlers run, and before it can complete. The GPU may run the work meanwhile:
    ///   a value of an event its work signals can rise before they ran.
    /// - **Never** for a command buffer that fails before its last batch is submitted (a
    ///   submission or host step that fails, a lost device); they are dropped with it. A command
    ///   buffer committed as failed ([`commit_failed`](Self::commit_failed)) submits its batches
    ///   and runs them.
    /// - **Keep it short.** It holds back its command buffer's scheduling, `commit` where the
    ///   commit submits, and on the scheduler thread the device's other submissions (a commit
    ///   meanwhile still submits its own ready batches). A wait in it for a command buffer, or
    ///   for a slot of a full queue, that has not happened yet is refused (`WaitWouldDeadlock` of
    ///   [`DeadlockCause::SubmissionHandler`]); it must not wait for GPU work otherwise.
    ///
    /// Refused once the command buffer is committed (`Invalid`, `Problem::HandlerAfterCommit`),
    /// as [`add_scheduled_handler`](Self::add_scheduled_handler) is.
    pub fn add_submission_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.status != CommandBufferStatus::NotEnqueued {
            return Err(invalid(
                "add_submission_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.submission.push_back(handler);
        Ok(())
    }

    /// `encodeSignalEvent:value:`: once the work encoded before it completed, the event's value
    /// becomes at least `value`. Refused while an encoder is open (Metal asserts), after commit,
    /// after an abandoned encoder, and for another device's event.
    pub fn encode_signal_event(&self, event: EventRef<'_>, value: u64) -> Result<(), Error> {
        self.event_command("encode_signal_event", event, |batches, event| {
            batches.signal(event, value);
        })
    }

    /// `encodeWaitForEvent:value:`: the work encoded after it starts once the event's value
    /// reached `value`. Refused as [`CommandBuffer::encode_signal_event`] is.
    pub fn encode_wait_for_event(&self, event: EventRef<'_>, value: u64) -> Result<(), Error> {
        self.event_command("encode_wait_for_event", event, |batches, event| {
            batches.wait(event, value);
        })
    }

    /// Wait for the caller's binary semaphore `semaphore` before the work encoded after it: the
    /// acquire of a swapchain image, or a signal of the caller's own submission. It splits the
    /// command buffer's batches as [`encode_wait_for_event`](Self::encode_wait_for_event) does, and
    /// the batch waits in all its stages. The device's scheduler submits the batch in its order:
    /// never before the signals of `semaphore` that command buffers of this device committed before
    /// this one are submitted.
    ///
    /// Refused as [`encode_signal_event`](Self::encode_signal_event) is (an encoder open, after
    /// commit, after an abandoned encoder), and for what the runtime can tell is no binary
    /// semaphore of the device: a null handle, or one of its own timeline semaphores (`Invalid`
    /// with a [`SemaphoreProblem`](super::SemaphoreProblem)).
    ///
    /// # Safety
    ///
    /// - `semaphore` is a binary semaphore of this device, and lives until the command buffer
    ///   completed.
    /// - The signal this wait consumes was submitted to the device before `commit` (by the
    ///   caller: `vkAcquireNextImageKHR`, or its own `vkQueueSubmit`), or is the last one an
    ///   [`encode_signal_vk_semaphore`](Self::encode_signal_vk_semaphore) of a command buffer of
    ///   this device committed before this one encodes. No other wait consumes it.
    pub unsafe fn encode_wait_for_vk_semaphore(
        &self,
        semaphore: vk::Semaphore,
    ) -> Result<(), Error> {
        self.binary_command("encode_wait_for_vk_semaphore", semaphore, |batches| {
            batches.binary_wait(semaphore);
        })
    }

    /// Signal the caller's binary semaphore `semaphore` once the work encoded before it
    /// completed: the semaphore a present waits for. It closes the batch as
    /// [`encode_signal_event`](Self::encode_signal_event) does; a wrapped image the batch uses
    /// is back in its home layout before the signal
    /// ([`Device::new_texture_with_vk_image`](super::Device::new_texture_with_vk_image)).
    ///
    /// The signal is submitted with its batch, by the device's scheduler: a wait the caller
    /// submits for it (`vkQueuePresentKHR`, its own `vkQueueSubmit`) comes after the command
    /// buffer is scheduled ([`wait_until_scheduled`](Self::wait_until_scheduled), or a
    /// scheduled handler), which Vulkan requires of a binary semaphore.
    ///
    /// Refused as [`encode_wait_for_vk_semaphore`](Self::encode_wait_for_vk_semaphore) is.
    ///
    /// # Safety
    ///
    /// `semaphore` is a binary semaphore of this device, unsignaled with no signal pending when
    /// the batch runs (every earlier signal of it waited for), and lives until the command buffer
    /// completed.
    pub unsafe fn encode_signal_vk_semaphore(&self, semaphore: vk::Semaphore) -> Result<(), Error> {
        self.binary_command("encode_signal_vk_semaphore", semaphore, |batches| {
            batches.binary_signal(semaphore);
        })
    }

    /// Add a binary semaphore wait or signal to the batches, after the runtime's checks.
    fn binary_command(
        &self,
        call: &'static str,
        semaphore: vk::Semaphore,
        add: impl FnOnce(&mut Batches<Arc<EventShared>, vk::Semaphore>),
    ) -> Result<(), Error> {
        if let Some(problem) = self.0.device.binary_semaphore_problem(semaphore) {
            return Err(invalid(call, Problem::Semaphore(problem)));
        }
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        add(&mut state.batches);
        Ok(())
    }

    /// Add an event wait or signal to the batches, after the checks Metal makes.
    fn event_command(
        &self,
        call: &'static str,
        event: EventRef<'_>,
        add: impl FnOnce(&mut Batches<Arc<EventShared>, vk::Semaphore>, Arc<EventShared>),
    ) -> Result<(), Error> {
        let event = event.shared();
        if !event.device().same(&self.0.device) {
            return Err(invalid(call, Problem::ForeignObject));
        }
        let mut state = self.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        add(&mut state.batches, Arc::clone(event));
        Ok(())
    }

    /// `addCompletedHandler:`: call `handler` once the command buffer completed or failed, on one
    /// of the device's handler threads, at the user-interactive class on macOS as in Metal, and
    /// with a utilization floor on Linux ([`Handler`]). Refused once the command buffer is
    /// committed.
    pub fn add_completed_handler(&self, handler: Handler) -> Result<(), Error> {
        let mut state = self.state();
        if state.status != CommandBufferStatus::NotEnqueued {
            return Err(invalid(
                "add_completed_handler",
                Problem::HandlerAfterCommit,
            ));
        }
        state.completed.push_back(handler);
        Ok(())
    }

    /// `commit`: hand the batches to the device's scheduler, which submits them in the queue's
    /// commit order, each once the events it waits for are reached.
    ///
    /// The indirect command buffers' commands the encoders execute are recorded first, from
    /// their records as they are now: each through the plan path, as a dispatch of the encoder
    /// would be. A command such a dispatch would refuse refuses the commit with that error, and
    /// nothing is submitted; a command that runs nothing on Metal (no dispatch, an empty or
    /// oversized threadgroup) is left out.
    ///
    /// Refused (and nothing submitted) when the command buffer was committed already, an
    /// encoder is open, or an encoder was abandoned (`Invalid`). A submission the device fails
    /// completes the command buffer with that error ([`CommandBuffer::error`]), as Metal reports
    /// it on the command buffer. A command buffer that encoded a render pass without attachments
    /// past 16384 a side fails from that pass on, as Metal's does
    /// ([`CommandBufferError::RenderTargetSize`]): the encoders before it run and the event
    /// signals encoded before it fire, and nothing from the pass on runs or signals. One with an
    /// encoder Metal's GPU faults fails as Metal's does ([`CommandBufferError::PageFault`]): that
    /// encoder runs nothing, and the others run and every signal fires.
    pub fn commit(&self) -> Result<(), Error> {
        self.commit_as("commit", None)
    }

    /// Commit the command buffer as failed with `failure`: a runtime extension, no Metal call
    /// (docs/API.md, "A command buffer committed as failed"), for layered validation, error
    /// injection and paravirtualization. It fails as Metal's failed command buffers do (measured
    /// by a consumer on an M3 Max, macOS 26.7): the encoders `failure.encoders` names run
    /// nothing, not even a render pass's load, clear or store actions, and every other encoder
    /// runs, before and after them ([`FailedEncoders`]: a GPU hang's suffix, or the set of a
    /// failed submission); its event waits hold it as they hold one committed; its event signals
    /// fire where `failure.signals` says so, and none fires where not (the caller's binary
    /// semaphores are signaled either way); its scheduled handlers run; and it completes in its
    /// queue's order, with status `Error` and [`CommandBufferError::Injected`] of
    /// `failure.code`, which its completed handlers see. A code of 0 completes it with status
    /// `Completed` and no error instead, as Metal completes a command buffer of a queue whose GPU
    /// hang is in flight (measured by the consumer on an M3 Max, 11 hangs at a draw of primitive
    /// type 9: it runs nothing, its event signals fire and its completed handlers see `Completed`
    /// and no error). The indirect command buffers the failing encoders execute are neither
    /// recorded nor checked. [`ended_encoders`](Self::ended_encoders) tells the encoders apart:
    /// each one's kind, its place among the event waits and signals, and what it uses. A command
    /// buffer Metal fails from a render pass on ([`commit`](Self::commit)) fails from there too,
    /// whatever `failure` names, and signals no event after the pass; it completes with
    /// `failure.code`.
    ///
    /// Refused, changing nothing, as [`commit`](Self::commit) is; and for a set naming an
    /// encoder past the ended ones (`Invalid`, `Problem::IndexOutOfRange`, its `limit` the
    /// number ended). (A code of 0 was refused, `Problem::UndefinedValue`.)
    pub fn commit_failed(&self, failure: CommitFailure) -> Result<(), Error> {
        self.commit_as("commit_failed", Some(failure))
    }

    /// [`commit`](Self::commit) for `call`, or, with `failure`,
    /// [`commit_failed`](Self::commit_failed).
    fn commit_as(&self, call: &'static str, failure: Option<CommitFailure>) -> Result<(), Error> {
        // A switch its scheduler settled since finishes first, on this thread; then the waiting
        // fetched render targets take their storage form at this commit where it allows
        // (`storage_form.rs`): two atomic reads where none is under way and none waits.
        let device = &self.0.device;
        device.finish_storage_forms();
        if !self.take_storage_forms(failure.is_some()) {
            return self.commit_now(call, failure);
        }
        let committed = self.commit_now(call, failure);
        match committed {
            // Where the commit submitted it, the switch finishes here, before `commit` returns.
            Ok(()) => device.finish_storage_forms(),
            Err(_) => self.give_up_storage_forms(),
        }
        committed
    }

    /// [`commit_as`](Self::commit_as) itself.
    fn commit_now(&self, call: &'static str, failure: Option<CommitFailure>) -> Result<(), Error> {
        let device = &self.0.device;
        let (segments, failure, reached): (Vec<(Deferred, bool)>, _, _) = {
            let mut state = self.state();
            if let Some(problem) = state.refusal() {
                return Err(invalid(call, problem));
            }
            let ended = state.ended.len();
            if let Some(problem) = failure.as_ref().and_then(|failure| failure.problem(ended)) {
                return Err(invalid(call, problem));
            }
            // Its last unit of command predication ends: a live block left open where a unit
            // ended refuses, and each step's gates are listed for it.
            let held = &mut *state;
            held.predication.commit(&mut held.uploads, device)?;
            let failure = Failure::of(failure, state.fails_from, ended);
            let failure = Failure::with_faults(failure, &state.faulted, ended);
            // The encoders that run: all but those `failure` fails.
            let runs = |encoder: usize| Failure::runs(failure.as_ref(), encoder);
            state.committing = true;
            // Its heaps reach every texture they hold now, for the work recorded below and by its
            // host steps.
            if let Some(residency) = &mut state.residency {
                residency.sync_heaps();
            }
            let segments = state
                .recorded
                .iter()
                .filter(|recorded| runs(recorded.encoder))
                .flat_map(|encoder| &encoder.segments)
                .map(|segment| (segment.work.clone(), segment.needs_step()))
                .collect();
            // What of it the live buffers list once used, held weakly: the device's registries'
            // locks, which take no other, inside this one.
            let reached = state.reached(device.shared(), runs);
            (segments, failure, reached)
        };
        // What the commands of its encoders that run use is resident from its commit on, as
        // Metal's is: a no-copy buffer or a heap among it is reached unnamed from here, by its
        // own commands too, the work recorded below included (`live_buffers.rs`). Listed without
        // the lock: what it upgrades may be a resource's last hold.
        let used = reached.list();
        device.shared().live_buffers.reach(&used);
        // Listed again at its first submission where the list changed since (a rebind ends a
        // use).
        let listing = Listing::new(device, used);
        // Recorded, or settled for their host steps, with the lock free: their dispatches take it,
        // and announce degradations.
        let prepared: Result<Vec<_>, Error> = segments
            .iter()
            .map(|(work, host)| work.prepare(self, *host))
            .collect();
        let prepared = match prepared {
            Ok(prepared) => prepared,
            Err(error) => {
                self.state().committing = false;
                return Err(error);
            }
        };
        // The first segment a host step records, which the device announces.
        let announced = segments
            .iter()
            .zip(&prepared)
            .find(|((_, host), prepared)| *host && !prepared.for_good)
            .map(|((work, _), _)| work.clone());
        // A host step may record commands after the commit (a segment not recorded for good):
        // until the command buffer completes, the buffers they use keep their pages.
        let stepped = prepared.iter().any(|prepared| !prepared.for_good);
        // The segments a host step records once the pipelines they record are made.
        let made_later: Vec<Deferred> = segments
            .iter()
            .zip(&prepared)
            .filter(|(_, prepared)| prepared.pipelines)
            .map(|((work, _), _)| work.clone())
            .collect();
        let (waits, away, reaches, unreplayable) = {
            let mut state = self.state();
            state.committing = false;
            if let Some(failure) = failure {
                state.fail(failure);
            }
            state.prepare(prepared);
            // Until its first submission it holds what its commands took weakly: a rebind or a
            // release meanwhile lets the pages go, and the submission finds them moved.
            state.wait_for_submission(listing);
            state.plan_checks();
            // Its root tables search every live buffer of its own on the device, and what its
            // encoders bound and made resident (`live_buffers.rs`).
            state.point_root_tables(device)?;
            // The function-table snapshots read the tables' words as they stand now.
            for snapshot in &state.table_snapshots {
                snapshot.refresh()?;
            }
            state.table_snapshots.clear();
            // Its commands' argument samplers, the states live now, before its heap sets stop
            // taking its writes (`sampler_listing.rs`).
            if !state.sampler_listings.is_empty() {
                let live = device.argument_samplers();
                let held = &mut *state;
                for listing in &held.sampler_listings {
                    match listing.relist(device, &live, &mut held.heaps)? {
                        Some(samplers) if !samplers.is_empty() => {
                            held.retained.push(Box::new(samplers));
                        }
                        Some(_) => {}
                        // Its plan cannot list them: the command buffer fails as it completes.
                        None => held.unlisted_argument_samplers = true,
                    }
                }
                held.sampler_listings.clear();
                if held.unlisted_argument_samplers {
                    device.shared().counts.unlisted_argument_sampler_failure();
                }
            }
            // Until it completes, an image retired from the device's heap waits for it, where
            // its commands, or its host steps', may read the heap (`heap_sets.rs`). The ticket
            // taken here holds what a rebind or a detach replaces; its first submission takes the
            // one gone textures wait for (`submit.rs`).
            if state.heaps.bound() || stepped {
                let ticket = state.heaps.commit(device);
                state.retained.push(Box::new(ticket));
            }
            state.status = CommandBufferStatus::Committed;
            for heap in state.used_heaps.drain(..) {
                heap.mark_used();
            }
            // A rebind before its first submission of a buffer its replayable encoders named
            // encodes them again (`reencode.rs`): noted while their holds still refuse one.
            let (unreplayable, keys) = state.keep_replays();
            if !keys.is_empty() {
                device.shared().rebinds.wait_for_replay(&self.0, keys);
            }
            // Committed, its commands recorded for good: a rebind of what they used takes
            // effect after them (`rebind.rs`); one a host step records holds until completion.
            if stepped {
                let held = &mut *state;
                let kept = held
                    .holds
                    .drain(..)
                    .map(|holds| Box::new(holds) as Box<dyn Any + Send + Sync>);
                held.retained.extend(kept);
            } else {
                state.holds.clear();
            }
            if let Some(slot) = &mut state.slot {
                slot.committed();
            }
            self.0.time(&mut state);
            // Its submissions record their barriers from its pool, inside sections, where no
            // spare is taken: one that encoded nothing takes a spare now, if one is free.
            self.0.take_spare_pool(&mut state);
            let waits = state
                .batches
                .all()
                .iter()
                .enumerate()
                .map(|(index, batch)| BatchSync {
                    waits: batch.waits.clone(),
                    binary_waits: batch.binary_waits.clone(),
                    binary_signals: batch.binary_signals.clone(),
                    segments: state.recorded[state.batch_range(index)]
                        .iter()
                        .any(|encoder| {
                            !encoder.segments.is_empty()
                                || encoder.sparse.iter().any(SparseUpdate::needs_step)
                        }),
                    checked: state.checks.is_some(),
                })
                .collect();
            // A texture over the caller's image that rests out of `GENERAL`: each batch that
            // uses it moves it.
            let away = device
                .shared()
                .interop
                .wrapped
                .among(state.recorded.iter().flat_map(|r| {
                    let segments = r.segments.iter().filter_map(|s| s.recorded.as_ref());
                    r.summary
                        .keys()
                        .chain(segments.flat_map(|done| done.summary.keys()))
                }))
                .into_iter()
                .find(|image| image.rests_out_of_general());
            // Its commands may reach every texture through its resource ID where one bound a
            // heap set.
            (waits, away, state.heaps.bound(), unreplayable)
        };
        // Each making the batches wait for wakes the scheduler as it finishes, with no lock
        // held (`render/pending.rs`), registered before the scheduler first looks at the batches:
        // a making that finished between that look and a later registration left the batch held
        // with nothing to wake the scheduler (17 lane tests timed out, EV PPC-lane1-mvk4).
        if !made_later.is_empty() {
            let waker = device.shared().scheduler.waker();
            for work in &made_later {
                work.when_made(&waker);
                // Its batch waits for them now: they go before the makings nothing waits for.
                work.hasten();
            }
        }
        drop(unreplayable);
        // The storage-sink degrade's trigger decides, in the program's call, before the
        // scheduler sees the command buffer (`detach.rs`).
        device.reach_storage_sinks(reaches);
        // Its host steps' instances take their numbers from here, in the program's call, before
        // the scheduler can run one.
        let _ = self.0.step_numbers.set(StepNumbers::reserve());
        // The scheduler's lock is taken before a command buffer's, never under it.
        device
            .shared()
            .scheduler
            .commit(&self.0, self.0.queue.id(), waits);
        if let Some(image) = away {
            device.degrade(Degradation::WrappedImageLayoutTransitions, || {
                image.notice()
            });
        }
        if let Some(work) = announced {
            work.degrade(device);
        }
        Ok(())
    }

    /// `status`.
    pub fn status(&self) -> CommandBufferStatus {
        self.state().status
    }

    /// `error`: why the command buffer failed, once its status is `Error`.
    pub fn error(&self) -> Option<CommandBufferError> {
        self.state().error
    }

    /// `error.localizedDescription`, where the driver relayed Metal's failure of the command
    /// buffer's work ([`CommandBufferError::Execution`], and the `PageFault` and `DeviceLost` it
    /// maps codes 3 and 11 to): Metal's description, as MoltenVK's message gave it after the
    /// code (`device/relay.rs`), empty where the message had none. `None` for every other
    /// failure, and before one.
    pub fn error_description(&self) -> Option<String> {
        self.state().relayed_description.clone()
    }

    /// Why a host step refused the command buffer's commands, once it failed with
    /// [`CommandBufferError::Refused`]: the refusal, typed as [`commit`](Self::commit) types it
    /// where the CPU wrote the records (`Invalid`, `Unsupported`, `NotImplemented` and the
    /// others). No Metal counterpart.
    pub fn refusal(&self) -> Option<Error> {
        self.state().refusal.clone()
    }

    /// Once the command buffer failed with [`CommandBufferError::PipelineCreation`]: the draw
    /// whose `VkPipeline` the driver did not make, by its encoder and its order there, the state
    /// and the variant it drew, and the driver's error, which the draw returned where the runtime
    /// made the pipeline at the draw (`FailedPipeline`). The first such draw, in encoding order:
    /// Metal reports one error per command buffer. No Metal counterpart.
    pub fn failed_pipeline(&self) -> Option<FailedPipeline> {
        self.state().failed_pipeline.clone()
    }

    /// Keep `failed`, a draw's pipeline whose making failed, which a host step met
    /// (`render/pending.rs`), unless one was kept already: the steps record in encoding order
    /// and stop at the first failure, so the one kept is the first.
    pub(super) fn fail_pipeline(&self, failed: FailedPipeline) {
        let mut held = self.state();
        if held.failed_pipeline.is_none() {
            held.failed_pipeline = Some(failed);
        }
    }

    /// What the command buffer's execution status words report, once it completed: resident-set
    /// misses of argument-buffer pointers, and the other conditions [`StatusReport`] names. They
    /// are facts of the run, not errors; `None` before completion and when the command buffer
    /// failed. No Metal counterpart.
    pub fn execution_report(&self) -> Option<StatusReport> {
        self.state().report
    }

    /// What the ended encoders' dispatches and draws executed, in encoding order (`execution.rs`):
    /// validation's executor reads it; no Metal call answers it. Empty unless the device was made
    /// with [`DeviceOptions::record_executions`](crate::DeviceOptions::record_executions): its
    /// encoders build no record otherwise.
    #[doc(hidden)]
    pub fn executions(&self) -> Vec<Execution> {
        self.state().executions.clone()
    }

    /// `GPUStartTime`: the host time, in seconds of [`host_time`](super::host_time)'s clock, at
    /// which the GPU began the command buffer's work, once it completed; 0 before, and when it
    /// failed (`gpu_time.rs`).
    ///
    /// On a device with calibrated timestamps (`VK_KHR_calibrated_timestamps` or
    /// `VK_EXT_calibrated_timestamps`, enabled where listed) it is Metal's answer, within the
    /// calibration's deviation. Without, it is the end below less the GPU's duration, which
    /// stays exact, and the device announces `Degradation::UncalibratedGpuTimes` once. A device
    /// whose queue family writes no timestamps refuses it (`Unsupported`, `Feature::GpuTimes`).
    /// So does a command buffer committed while others held every slot of the device's
    /// timestamp pools and no pool more could be made (2048 on MoltenVK; elsewhere only where
    /// the device refused a pool, `gpu_time.rs`), and the device announces
    /// `Degradation::UntimedCommandBuffer` once.
    ///
    /// A command buffer that encoded nothing and waits for nothing writes no timestamps: its GPU
    /// start and end are where its queue reached it, its kernel end time or the GPU end of the
    /// queue's command buffer before it where later, a zero duration, as Metal answers for one.
    /// On a device made with `gpu_times_from_host` ([`DeviceOptions`](crate::DeviceOptions),
    /// [`RawDevice`](crate::RawDevice)), and on a driver whose timestamps wait for every
    /// queue's (MoltenVK 1.4.1 on Apple's GPUs), none writes them: the start is the kernel end
    /// time (or the queue's latest GPU end before it, where later) and the end the host time the
    /// runtime saw it complete, and the device announces `Degradation::GpuTimesFromHost` once.
    pub fn gpu_start_time(&self) -> Result<f64, Error> {
        self.gpu_times().map(|(start, _)| gpu_time::seconds(start))
    }

    /// `GPUEndTime`: the host time at which the GPU finished the command buffer's work, as
    /// [`gpu_start_time`](Self::gpu_start_time) says. Without calibrated timestamps, it is the
    /// host time at which the runtime saw the command buffer complete: late by that latency.
    pub fn gpu_end_time(&self) -> Result<f64, Error> {
        self.gpu_times().map(|(_, end)| gpu_time::seconds(end))
    }

    /// `kernelStartTime`: the host time, in seconds of [`host_time`](super::host_time)'s clock,
    /// at which the device's scheduler began submitting the command buffer's work to Vulkan; 0
    /// before.
    pub fn kernel_start_time(&self) -> f64 {
        gpu_time::seconds(self.state().times.kernel_start)
    }

    /// `kernelEndTime`: the host time at which the device's scheduler finished submitting the
    /// command buffer's work to Vulkan (it is then `Scheduled`); 0 before.
    pub fn kernel_end_time(&self) -> f64 {
        gpu_time::seconds(self.state().times.kernel_end)
    }

    /// The GPU start and end, in nanoseconds of the host's clock: placed from the command
    /// buffer's timestamps once it completed, then kept; 0 and 0 before, when it failed, or when
    /// the GPU wrote none. One that wrote no timestamps answers the times placed as it completed
    /// (`place_gpu_times`, `gpu_time.rs`). Refused where the device writes no timestamps (and
    /// does not answer from the host), and for a command buffer that found no slot of the
    /// device's timestamp pools free at its commit, and no pool more.
    fn gpu_times(&self) -> Result<(u64, u64), Error> {
        let device = &self.0.device;
        let shared = device.shared();
        let untimed = || Error::Unsupported(Unsupported::not_representable(Feature::GpuTimes));
        let clock = shared.timing.clock();
        if clock.is_none() && !shared.timing.answers_from_host() {
            return Err(untimed());
        }
        let mut state = self.state();
        if let Some(times) = state.times.gpu {
            return Ok(times);
        }
        if state.status != CommandBufferStatus::Completed {
            return Ok((0, 0));
        }
        let (start, end) = match state.times.stamped {
            // Placed as it completed (`place_gpu_times`).
            Stamped::NoWork => return Ok(state.times.placed.unwrap_or((0, 0))),
            Stamped::FromHost => {
                let Some(times) = state.times.placed else {
                    return Ok((0, 0));
                };
                state.times.gpu = Some(times);
                drop(state);
                let because = shared
                    .timing
                    .host_clock_reason()
                    .map_or("", |from| from.because());
                device.degrade(Degradation::GpuTimesFromHost, || {
                    (
                        "a command buffer's GPU times".to_owned(),
                        because.to_owned(),
                    )
                });
                return Ok(times);
            }
            Stamped::Read(Some(ticks)) => ticks,
            Stamped::Read(None) | Stamped::Unread => return Ok((0, 0)),
            Stamped::NoSlot => {
                drop(state);
                let slots = shared.timing.slots();
                device.degrade(Degradation::UntimedCommandBuffer, || {
                    let why = if slots == 0 {
                        "the device could make no timestamp pool at its commit".to_owned()
                    } else {
                        format!(
                            "every one of the device's {slots} timestamp slots was held by a \
                             command buffer in flight at its commit, and it could make no pool \
                             more"
                        )
                    };
                    ("a command buffer's GPU times".to_owned(), why)
                });
                return Err(untimed());
            }
            Stamped::Failed(error) => return Err(device.failed(error)),
        };
        // Read timestamps came from the device's clock.
        let clock = clock.ok_or_else(untimed)?;
        let pair = state.times.pair;
        let times = clock.span(pair, start, end, state.times.completed);
        state.times.gpu = Some(times);
        drop(state);
        if pair.is_none() {
            let reason = if clock.calibration.is_some() {
                "the device's calibrated timestamps could not be read at its commit"
            } else {
                "the device calibrates its timestamps against no host clock the runtime reads"
            };
            device.degrade(Degradation::UncalibratedGpuTimes, || {
                (
                    "a command buffer's GPU times".to_owned(),
                    format!(
                        "{reason} ({}-bit timestamps of {} ns)",
                        clock.valid_bits, clock.period
                    ),
                )
            });
        }
        Ok(times)
    }

    /// `waitUntilScheduled`: block until the command buffer's scheduled handlers ran, or it
    /// failed. As Metal's (measured on an M4, macOS 27.0.1, EV SUB-sched1-m4), it returns only
    /// once every scheduled handler returned, though [`status`](Self::status) reads `Scheduled`
    /// just before they start.
    ///
    /// From a handler or a release on the device's handler threads it answers as Metal's does
    /// where measured: once the command buffer's transition to `Scheduled` ran, it returns. Its own
    /// scheduled handlers' wait, and a wait for a later command buffer of the handler's queue, are
    /// refused (`WaitWouldDeadlock` of [`DeadlockCause::OwnQueue`]): that transition comes after
    /// the handler in the queue's order, and Metal's never returns there. A wait for another
    /// queue's command buffer waits, unless it would deadlock (as
    /// [`CommandBuffer::wait_until_completed`] says). From a submission handler
    /// ([`add_submission_handler`](Self::add_submission_handler)), a wait for one not yet scheduled
    /// is refused ([`DeadlockCause::SubmissionHandler`]). Once the process is exiting, it answers
    /// as that one does.
    pub fn wait_until_scheduled(&self) -> Result<(), Error> {
        let scheduled_now = scheduled(&self.state());
        let _blocked = match scheduled_wait(running(), self.0.target(), scheduled_now) {
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Return | Answer::Wait => None,
            Answer::Block(from) => Some(contexts::block(from, self.0.context())?),
        };
        let mut state = self.state();
        while !scheduled(&state) {
            if exit_guard::closed() {
                return Err(DeadlockCause::Exiting.into());
            }
            state = self
                .0
                .changed
                .wait_timeout(state, exit_guard::WAKE)
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .0;
        }
        Ok(())
    }

    /// `waitUntilCompleted`: block until the command buffer completed or failed and its
    /// completed handlers ran. It returns on device loss (the status is then `Error`).
    ///
    /// It returns only once the runtime's threads let go of the command buffer, so dropping the
    /// last handles after it destroys the device on the dropping thread, before the drop
    /// returns. A program that learns of the completion otherwise (a completed handler, an event
    /// the command buffer signals, its status) may drop them while a runtime thread still holds
    /// the command buffer; the device then goes on that thread. An exit meanwhile waits for that
    /// teardown, or leaves the device to the process's end.
    ///
    /// Once the process is exiting (its exit handlers run), the runtime's threads that finish
    /// command buffers have ended, and no Vulkan call is made: a wait for a command buffer that
    /// did not finish answers `WaitWouldDeadlock` of [`DeadlockCause::Exiting`] at once, on any
    /// thread, and one already blocked within 50 ms, so that nothing the exit runs waits for
    /// those threads ([`process_exiting`](crate::process_exiting) answers whether it is so).
    ///
    /// From a handler or a release on the device's handler threads it answers as Metal's does
    /// where measured:
    /// - a deallocator that runs as this command buffer releases what it retained gets `Ok` at
    ///   once (the status is `Completed` or `Error`);
    /// - a handler's wait for its own command buffer, or for a later one of its queue, is
    ///   refused (`WaitWouldDeadlock` of [`DeadlockCause::OwnQueue`]): that transition comes
    ///   after the handler in the queue's order, and Metal's never returns there;
    /// - a wait for another queue's command buffer blocks the handler's thread until it
    ///   completed, as Metal's does. It is refused (`WaitWouldDeadlock`) only where it would
    ///   deadlock: when that queue's handlers wait, through the queues they wait for, for this
    ///   one ([`DeadlockCause::Cycle`]), or when it would leave the device's handler threads
    ///   with none that is not blocked so ([`DeadlockCause::HandlerThreadCap`]: the runtime's
    ///   cap, above the 256 measured on Metal).
    pub fn wait_until_completed(&self) -> Result<(), Error> {
        let _blocked = match self.completed_wait() {
            Answer::Return => return Ok(()),
            Answer::Refuse(cause) => return Err(cause.into()),
            Answer::Wait => None,
            Answer::Block(from) => Some(contexts::block(from, self.0.context())?),
        };
        self.0.gate.wait()?;
        Ok(())
    }

    /// [`CommandBuffer::wait_until_completed`] for at most `timeout`. Where that is refused, it
    /// ends at once with `TimedOut`, as it does once the process is exiting.
    pub fn wait_until_completed_for(&self, timeout: Duration) -> Result<(), TimedOut> {
        let _blocked = match self.completed_wait() {
            Answer::Return => return Ok(()),
            Answer::Refuse(_) => return Err(TimedOut),
            Answer::Wait => None,
            Answer::Block(from) => {
                Some(contexts::block(from, self.0.context()).map_err(|_| TimedOut)?)
            }
        };
        self.0.gate.wait_until(Instant::now() + timeout)
    }

    /// What a wait for completion answers on this thread.
    fn completed_wait(&self) -> Answer {
        completed_wait(running(), self.0.target(), self.0.gate.is_open())
    }
}

impl CommandBufferShared {
    /// The device.
    pub(super) fn device(&self) -> &Device {
        &self.device
    }

    /// This command buffer's address, which names it to [`own`]'s marker.
    fn address(&self) -> usize {
        self as *const CommandBufferShared as usize
    }

    /// Its queue's serial context.
    fn context(&self) -> Context {
        self.queue.context()
    }

    /// This command buffer as the target of a wait.
    fn target(&self) -> Target {
        Target {
            context: self.context(),
            buffer: self.address(),
        }
    }

    /// The marker of `step` of this command buffer.
    fn marker(&self, step: Step) -> Marker {
        Marker {
            context: self.context(),
            buffer: self.address(),
            step,
        }
    }

    /// A reference for a runtime thread: the command buffer's waiters wait until it is gone.
    pub(crate) fn hold(self: &Arc<Self>) -> Hold<CommandBufferShared> {
        Hold::new(self, &self.gate)
    }

    /// Whether batch `index` waits for a host step: a segment of it executes commands the GPU may
    /// write, not recorded since, or an indirect sparse update of it reads regions the GPU may
    /// write, not read since.
    pub(crate) fn needs_host_step(&self, index: usize) -> bool {
        let state = lock(&self.state);
        // After a failed step the batch goes to its submission, which fails.
        if state.step_failure.is_some() {
            return false;
        }
        let range = state.batch_range(index);
        let recorded = &state.recorded[range];
        recorded
            .iter()
            .flat_map(|encoder| &encoder.segments)
            .any(Segment::needs_step)
            || recorded
                .iter()
                .flat_map(|encoder| &encoder.sparse)
                .any(SparseUpdate::needs_step)
    }

    /// What the host step of batch `index` waits for ([`needs_host_step`](Self::needs_host_step)):
    /// the work submitted before it, where the GPU may write what a segment or an indirect
    /// sparse update of it reads; and the pipelines its segments record, while one is being made.
    pub(crate) fn step_waits(&self, index: usize) -> StepWaits {
        let state = lock(&self.state);
        let recorded = &state.recorded[state.batch_range(index)];
        let segments = || recorded.iter().flat_map(|encoder| &encoder.segments);
        StepWaits {
            gpu: segments().any(Segment::steps_after_the_gpu)
                || recorded
                    .iter()
                    .flat_map(|encoder| &encoder.sparse)
                    .any(SparseUpdate::needs_step),
            pipelines: segments().any(Segment::steps_after_pipelines),
        }
    }

    /// What batch `index` uses, for the scheduler's order across queues (`scheduler/order.rs`):
    /// its encoders' summaries and its segments' recordings, and for a segment its host step
    /// records for the pipelines it waits for alone, what its work uses, known at encode time
    /// (`Segment::known_summary`): Metal makes the pipeline at `newRenderPipelineState`, so only a
    /// batch that shares a resource with it waits for it. `None` while some of it is not known: a
    /// segment its host step records (or records again) for what the GPU may write, one its
    /// submission records (a scrub), or a resource state encoder's update, whose uses its
    /// submission adds.
    pub(crate) fn footprint(&self, index: usize) -> Option<Footprint<ResourceKey>> {
        let state = lock(&self.state);
        let recorded = &state.recorded[state.batch_range(index)];
        let mut summaries = Vec::new();
        let mut known = Vec::new();
        for encoder in recorded {
            if !encoder.sparse.is_empty() {
                return None;
            }
            summaries.push(&encoder.summary);
            for segment in &encoder.segments {
                match &segment.recorded {
                    Some(recording) if !segment.needs_step() => summaries.push(&recording.summary),
                    _ => known.push(segment.known_summary(&self.device)?),
                }
            }
        }
        Some(Footprint::of(summaries.into_iter().chain(&known)))
    }

    /// The host step of batch `index` (`deferred.rs`): on the scheduler's thread, without its lock,
    /// inside a section of the exit guard, once every batch submitted before it completed. Each
    /// segment of the batch that needs it is recorded from what its work reads as it stands now and
    /// what `commit` left for the step (`Deferred::step`); the recording `commit` made of one, if
    /// any, is dropped. A failure is kept for the batch's submission, which fails the command
    /// buffer with it.
    pub(crate) fn host_step(self: &Arc<Self>, index: usize) {
        // A rebind's encoding again of its encoders records into its pool too: one at a time.
        let _turn = lock(&self.reencoding);
        // Its instances numbered from its commit's place among the program's calls.
        let step = || exit_guard::host_step(|| self.record_host_step(index));
        match self.step_numbers.get() {
            Some(numbers) => numbers.numbering(step),
            None => step(),
        }
    }

    /// [`host_step`](Self::host_step)'s recording, marked: every notice it announces, those
    /// asked for at every occurrence included, is announced here, on this thread, before the
    /// batch is submitted, completes or is scheduled.
    fn record_host_step(self: &Arc<Self>, index: usize) {
        let device = &self.device;
        let steps: Vec<_> = {
            let state = lock(&self.state);
            let range = state.batch_range(index);
            let first = range.start;
            state.recorded[range]
                .iter()
                .enumerate()
                .flat_map(|(encoder, recorded)| {
                    recorded
                        .segments
                        .iter()
                        .enumerate()
                        .filter(|(_, segment)| segment.needs_step())
                        .map(move |(at, segment)| {
                            (
                                (first + encoder, at),
                                segment.work.clone(),
                                segment.inputs.clone(),
                            )
                        })
                })
                .collect()
        };
        self.read_mappings(index);
        let Some((_, first, _)) = steps.first() else {
            return;
        };
        let announced = first.clone();
        let buffer = CommandBuffer(Arc::clone(self));
        for ((encoder, at), work, inputs) in steps {
            let Some(inputs) = inputs else {
                let error = internal("a committed segment without what its host step needs");
                lock(&self.state).step_failure = Some(recording_error(device, error));
                break;
            };
            let recorded = work.step(&buffer, &inputs);
            let mut state = lock(&self.state);
            match recorded {
                Ok(done) => {
                    let segment = &mut state.recorded[encoder].segments[at];
                    segment.stepped = true;
                    let before = segment.recorded.replace(done);
                    if let Some(mut before) = before {
                        state.retained.extend(before.retained.into_entry());
                        before.taken.retain_into(&mut state.retained);
                    }
                }
                Err(error) => {
                    // A draw's pipeline whose making failed fails the command buffer so.
                    state.step_failure = Some(match &state.failed_pipeline {
                        Some(failed) => CommandBufferError::PipelineCreation(
                            PipelineCreationCause::of(&failed.error),
                        ),
                        None => step_error(device, error, &mut state.refusal),
                    });
                    break;
                }
            }
            drop(state);
            // Dropped without the lock: what it holds may be a state's last handle.
            drop((work, inputs));
        }
        drop(buffer);
        announced.degrade(device);
    }

    /// The host step's reading of batch `index`'s indirect sparse updates (`sparse/indirect.rs`),
    /// before its segments' steps: each is replaced, in its place, by the updates its arguments
    /// state now. A refusal is kept for the batch's submission, which fails the command buffer with
    /// it, and the rest stay unread.
    fn read_mappings(self: &Arc<Self>, index: usize) {
        let device = &self.device;
        let unread: Vec<(usize, usize, SparseUpdate)> = {
            let state = lock(&self.state);
            if state.step_failure.is_some() {
                return;
            }
            let range = state.batch_range(index);
            let first = range.start;
            state.recorded[range]
                .iter()
                .enumerate()
                .flat_map(|(encoder, recorded)| {
                    recorded
                        .sparse
                        .iter()
                        .enumerate()
                        .filter(|(_, update)| update.needs_step())
                        .map(move |(at, update)| (first + encoder, at, update.clone()))
                })
                .collect()
        };
        if unread.is_empty() {
            return;
        }
        // Read without the lock, the last of an encoder first, so the places of those before
        // it stay where they were.
        for (encoder, at, update) in unread.into_iter().rev() {
            let SparseUpdate::Indirect {
                texture,
                mode,
                buffer,
                offset,
            } = &update
            else {
                continue;
            };
            let read = indirect::resolve(device, texture, *mode, buffer, *offset);
            let mut state = lock(&self.state);
            match read {
                Ok(updates) => {
                    state.recorded[encoder].sparse.splice(at..at + 1, updates);
                }
                Err(failure) => {
                    state.step_failure = Some(match failure {
                        ReadFailure::Arguments(error) => error,
                        ReadFailure::Read(error) => recording_error(device, error),
                    });
                    break;
                }
            }
        }
        indirect::degrade(device);
    }

    /// A primary `VkCommandBuffer` from this command buffer's pool, made on first use. The
    /// caller holds `state` (the pool is externally synchronized).
    fn allocate_raw(&self, state: &mut State) -> Result<vk::CommandBuffer, Error> {
        self.allocate(state, vk::CommandBufferLevel::PRIMARY)
    }

    /// Give a command buffer without a pool yet (one that encoded nothing) a spare pool of the
    /// device's where one is free, on the committing thread: its submissions, which record its
    /// barriers inside sections where no spare is taken, then make no `VkCommandPool` of their
    /// own (one made and destroyed per such command buffer before). Where none is free, or the
    /// spare's reset fails, the submission makes one as before and answers any failure then.
    fn take_spare_pool(&self, state: &mut State) {
        if state.pool.is_some() {
            return;
        }
        let shared = self.device.shared();
        let family = shared.queue_family_of(self.queue.vk_queue());
        if let Ok(Some(spare)) = shared.spares.command_pool(shared, family) {
            state.pool = Some(spare);
        }
    }

    /// A `VkCommandBuffer` of `level` from this command buffer's pool, made on first use: a
    /// spare of the device's, reset, or a new one. A command buffer the pool made before it was
    /// reset is handed out again before a new one is made. The caller holds `state` (the pool is
    /// externally synchronized).
    fn allocate(
        &self,
        state: &mut State,
        level: vk::CommandBufferLevel,
    ) -> Result<vk::CommandBuffer, Error> {
        let device = &self.device;
        let raw = device.shared().vk();
        if state.pool.is_none() {
            let family = device.shared().queue_family_of(self.queue.vk_queue());
            let pool = match device
                .shared()
                .spares
                .command_pool(device.shared(), family)?
            {
                Some(spare) => spare,
                None => {
                    let info = vk::CommandPoolCreateInfo::default()
                        .flags(vk::CommandPoolCreateFlags::TRANSIENT)
                        .queue_family_index(family);
                    // SAFETY: the device is live and `info` valid.
                    let pool = unsafe { raw.create_command_pool(&info, None) };
                    exit_guard::called(Call::CreateCommandPool);
                    device.shared().counts.command_pool_made();
                    CommandPool::new(pool.map_err(vulkan)?, family)
                }
            };
            state.pool = Some(pool);
        }
        let Some(pool) = &mut state.pool else {
            return Err(internal("a command buffer without its command pool"));
        };
        if let Some(buffer) = pool.reuse(level) {
            return Ok(buffer);
        }
        let info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool.raw)
            .level(level)
            .command_buffer_count(1);
        let buffer = device.allocate_command_buffers(&info)?[0];
        pool.made(level, buffer);
        Ok(buffer)
    }

    /// Ready `state`'s times at its commit, on the program's thread: where the device writes
    /// timestamps, the slot of the device's timestamp pools the batches write its start and end
    /// into, and the device's calibration pair. Without a slot free (and no pool more), it has
    /// no timestamps. One
    /// that encoded nothing and waits for nothing, and every one where the device answers GPU
    /// times from the host, writes none (`gpu_time.rs`).
    fn time(&self, state: &mut State) {
        let shared = self.device.shared();
        let timing = &shared.timing;
        if timing.clock().is_none() && !timing.answers_from_host() {
            return;
        }
        let waits = state
            .batches
            .all()
            .iter()
            .any(|batch| !batch.waits.is_empty() || !batch.binary_waits.is_empty());
        if state.recorded.is_empty() && !waits {
            state.times.stamped = Stamped::NoWork;
            return;
        }
        if timing.answers_from_host() {
            state.times.stamped = Stamped::FromHost;
            return;
        }
        // SAFETY: the device is live.
        match unsafe { timing.slot(shared.vk()) } {
            Some(slot) => {
                shared.counts.timestamp_slot_taken();
                state.times.slot = Some(slot);
                state.times.pair = timing.pair();
            }
            None => state.times.stamped = Stamped::NoSlot,
        }
    }

    /// Read the command buffer's timestamps, now that its work completed, and give its slot
    /// back: on the completion thread, inside its section.
    pub(super) fn read_stamps(&self) {
        let mut state = lock(&self.state);
        let Some(slot) = state.times.slot.take() else {
            return;
        };
        // Each query's value, then its availability.
        let mut results = [[0u64; 2]; 2];
        // SAFETY: the slot's two queries were written by work that completed; the results are
        // 64-bit with availability, two words a query.
        let read = unsafe {
            self.device.shared().vk().get_query_pool_results(
                slot.queries(),
                slot.first(),
                &mut results,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WITH_AVAILABILITY,
            )
        };
        exit_guard::called(Call::GetQueryPoolResults);
        let [[start, started], [end, ended]] = results;
        state.times.stamped = match read {
            Ok(()) | Err(vk::Result::NOT_READY) => {
                Stamped::Read((started != 0 && ended != 0).then_some((start, end)))
            }
            Err(error) => Stamped::Failed(error),
        };
    }

    /// Place its GPU times among its queue's, now that it completed: on the completion thread,
    /// after its timestamps were read, in its queue's order (`gpu_time.rs`). One that wrote no
    /// timestamps takes its times here: encoder-less, where its queue reached it; answered from
    /// the host, its kernel end (or its queue's latest end) and its completion. Every one moves
    /// its queue's latest GPU end: its own, or its completion where it has none (an upper bound
    /// of its GPU end).
    pub(super) fn place_gpu_times(&self) {
        let shared = self.device.shared();
        let mut state = lock(&self.state);
        let times = &mut state.times;
        self.queue.place_gpu_times(|latest_end| {
            let (kernel_end, completed) = (times.kernel_end, times.completed);
            match times.stamped {
                Stamped::NoWork => {
                    let at = kernel_end.max(latest_end);
                    times.placed = Some((at, at));
                    at
                }
                // One after another, as Metal's serial queue can run them (`gpu_time.rs`).
                Stamped::FromHost => {
                    let start = kernel_end.max(latest_end);
                    let end = completed.max(start);
                    times.placed = Some((start, end));
                    end
                }
                Stamped::Read(Some((start, end))) => match shared.timing.clock() {
                    Some(clock) => clock.span(times.pair, start, end, completed).1,
                    None => completed,
                },
                Stamped::Unread | Stamped::NoSlot | Stamped::Read(None) | Stamped::Failed(_) => {
                    completed
                }
            }
        });
    }

    /// The scheduler submitted the last batch: in the queue's context, once the submitting
    /// thread ran its submission handlers, become `Scheduled`, then run the scheduled handlers on
    /// this thread, in the order added, then let `wait_until_scheduled` return. Metal's status
    /// reads `Scheduled` just before its first scheduled handler starts, and its wait returns
    /// once the last returned (EV SUB-sched1-m4: an M4, macOS 27.0.1).
    fn run_scheduled(self: &Arc<Self>) {
        {
            let mut state = lock(&self.state);
            while state.submitting {
                // The submitting thread runs no more once the process is exiting: what is
                // held stays held.
                if exit_guard::closed() {
                    return;
                }
                state = self
                    .changed
                    .wait_timeout(state, exit_guard::WAKE)
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .0;
            }
            state.status = CommandBufferStatus::Scheduled;
        }
        let buffer = CommandBuffer(Arc::clone(self));
        {
            let _running = Running::enter(self.marker(Step::ScheduledHandlers));
            loop {
                let Some(handler) = lock(&self.state).scheduled.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        drop(buffer);
        lock(&self.state).scheduled_ran = true;
        self.changed.notify_all();
    }

    /// Its last batch was submitted on this thread, which holds no lock of the runtime's: run
    /// its submission handlers here, in the order added, then let its transition to `Scheduled`
    /// run ([`CommandBuffer::add_submission_handler`]).
    pub(crate) fn run_submission_handlers(self: &Arc<Self>) {
        let buffer = CommandBuffer(Arc::clone(self));
        {
            let _running = Running::enter(self.marker(Step::SubmissionHandlers));
            loop {
                let Some(handler) = lock(&self.state).submission.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        drop(buffer);
        lock(&self.state).submitting = false;
        self.changed.notify_all();
    }

    /// A batch could not be submitted (the scheduler dropped the rest): the command buffer
    /// finishes with `error` once what it submitted before completed, at its `VkQueue`'s last
    /// submitted value in `values`, and, where a command buffer ends at a fence
    /// ([`fences::ends_at_fence`]), at one submitted after it, as a command buffer whose last
    /// batch was submitted finishes; at the value alone when the queue takes no such fence (a
    /// lost device). Its submission and scheduled handlers never run; they are dropped with it.
    pub(crate) fn failed(self: &Arc<Self>, error: CommandBufferError, values: &[u64]) {
        let vk_queue = self.queue.vk_queue();
        lock(&self.state).failure = Some(error);
        let fence = fences::ends_at_fence(&self.device, vk_queue)
            .then(|| self.fence_after_submitted(vk_queue))
            .flatten();
        self.device.shared().completion.watch(
            vk_queue,
            Submitted {
                value: values[vk_queue],
                buffer: self.hold(),
                last: true,
                fence,
                signals: Vec::new(),
            },
        );
    }

    /// A fence of the device's, submitted to `VkQueue` `queue` with no work: it is signaled once
    /// everything submitted there before completed. `None` when the process is exiting or the
    /// submission fails. A section of the exit guard of its own.
    fn fence_after_submitted(&self, queue: usize) -> Option<vk::Fence> {
        let _busy = exit_guard::enter()?;
        let shared = self.device.shared();
        let raw = shared.vk();
        // SAFETY: the device's fences, inside a section.
        let fence = unsafe { shared.fences.take(raw) }.ok()?;
        // SAFETY: a submission of no batch with an unsignaled fence in no pending submission;
        // the `VkQueue` is used under its lock.
        shared.counts.queue_submitted(0);
        let submitted =
            shared.submit_logged(queue, Some(fence), &self.relayed, |vk_queue| unsafe {
                raw.queue_submit(vk_queue, &[], fence)
            });
        exit_guard::called(Call::QueueSubmit);
        match submitted {
            Ok(()) => Some(fence),
            Err(error) => {
                // SAFETY: as for a failed batch's fence (`submit.rs`).
                unsafe { release::object(raw, fence) };
                let _ = self.device.failed(error);
                None
            }
        }
    }

    /// The command buffer's last batch is submitted: post its transition to `Scheduled` to its
    /// queue's context.
    pub(crate) fn post_scheduled(self: &Arc<Self>) {
        self.post(Transition {
            buffer: self.hold(),
            to: To::Scheduled,
        });
    }

    /// The last batch of the command buffer `buffer` holds finished with `result` (or a batch
    /// failed before): post its transition to `Completed` or `Error` to its queue's context. The
    /// hold moves into the transition, and the caller keeps no other reference: once the
    /// transition ran, the command buffer may go, with its device, on the handler thread.
    pub(crate) fn post_finished(
        buffer: Hold<CommandBufferShared>,
        result: Result<(), CommandBufferError>,
    ) {
        let (contexts, queue) = (buffer.device.shared().contexts.clone(), buffer.queue.id());
        contexts.post(
            queue,
            Transition {
                buffer,
                to: To::Finished(result),
            },
        );
    }

    /// Append `transition` to this command buffer's queue's context.
    fn post(&self, transition: Transition) {
        self.device
            .shared()
            .contexts
            .post(self.queue.id(), transition);
    }

    /// Finish the command buffer with `result` (or the failure of a batch): set its status,
    /// decode its status words, release its uploads, run its completed handlers on this thread,
    /// release what it retained, then open its gate, which releases its waiters once the
    /// transition's [`Hold`] (and every other) is gone. Waits from inside these steps answer as
    /// [`own`] says.
    fn complete(self: &Arc<Self>, result: Result<(), CommandBufferError>) {
        // A capture replay found a primitive whose reach was not measured: its pass rewrote
        // every tile, announced once the lock is let go (below).
        let mut unmeasured_reach = false;
        {
            let mut state = lock(&self.state);
            // A failure the driver relayed for one of its batches is Metal's answer for its work
            // (`device/relay.rs`).
            let result = match (state.failure, self.relayed.failure()) {
                (Some(error), _) => Err(error),
                (None, Some((code, description))) => {
                    state.relayed_description = description;
                    Err(CommandBufferError::of_metal_code(code))
                }
                (None, None) => result,
            };
            match result {
                Ok(()) => {
                    state.status = CommandBufferStatus::Completed;
                    // The queue's timeline reached the last batch's value, and every batch ends
                    // with a barrier to the host.
                    let words = state
                        .status_words
                        .as_ref()
                        .map_or(ExecutionStatus::initial(), Upload::read);
                    let mut report = ExecutionStatus::decode(&words);
                    if state.resident_misses != 0 {
                        report.resident_miss = true;
                        report.resident_misses =
                            report.resident_misses.wrapping_add(state.resident_misses);
                    }
                    if state.faults != ReplayFaults::default() {
                        report.replay_fault = true;
                        report.replay_faults =
                            ReplayFaults(report.replay_faults.0 | state.faults.0);
                    }
                    // An index bounds pass drew nothing of a draw its pool could not hold
                    // (`index_bounds.rs`): the command buffer failed.
                    if report.index_pool_overflow {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::IndexPoolOverflow);
                    }
                    // A predicated loop would have run past its copies (`predication.rs`): the
                    // command buffer failed.
                    if state.predication.overflowed() && state.error.is_none() {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::PredicatedLoopPastCopies);
                    }
                    // A scanned emulated mesh draw drew nothing of records its pool could not
                    // hold (`render/mesh.rs`): the command buffer failed.
                    if let Some(fault) = report.mesh_records_overflow {
                        let budget = state.mesh_pool.as_ref().map_or(0, PoolLease::bytes);
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::MeshRecordsOverflow(
                            MeshRecordsOverflow {
                                draw: fault.draw,
                                object_threadgroups: fault.object_threadgroups,
                                mesh_threadgroups: fault.mesh_threadgroups,
                                budget,
                            },
                        ));
                    }
                    // A function that read an argument its function constants removed took zero
                    // for it, Metal's answer under its debug layer: the command buffer completes,
                    // and its report states the read. A function sampled through a cubic sampler an
                    // argument buffer named, whose bicubic filter it does not emulate there: no
                    // result is then Metal's.
                    if report.cubic_argument_sampler && state.error.is_none() {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::CubicArgumentSampler);
                    }
                    // A command's plan could not list the sampler states live as it committed:
                    // its function may have missed one Metal's would reach.
                    if state.unlisted_argument_samplers && state.error.is_none() {
                        state.status = CommandBufferStatus::Error;
                        state.error = Some(CommandBufferError::UnlistedArgumentSampler);
                    }
                    unmeasured_reach = report.captured_reach_unmeasured;
                    state.report = Some(report);
                    let counts = &self.device.shared().counts;
                    for tally in &state.capture_tallies {
                        counts.capture_reach_tallied(tally.found(), tally.bound);
                    }
                }
                Err(error) => {
                    state.status = CommandBufferStatus::Error;
                    state.error = Some(error);
                }
            }
            state.status_words = None;
            state.capture_tallies.clear();
            state.index_pool = None;
            state.mesh_pool = None;
            state.uploads.clear();
            state.heaps.clear();
            // Its work ended: a detach waiting for it goes on (`detach.rs`).
            state.submitted_holds.clear();
            // Its work ended (its fences were signaled, or its queues drained after a failed
            // wait), so nothing of its pools is pending: its command pool goes to the device's
            // spares whatever its status, never released on this thread (`spares.rs`); a command
            // buffer that completed gives back its descriptor pools too, one that failed releases
            // them here.
            if let Some(pool) = state.pool.take() {
                self.device.shared().spares.give_command_pool(pool);
            }
            if state.status == CommandBufferStatus::Completed {
                state.descriptors.give_back();
            } else {
                state.descriptors.release();
            }
            state.unplaced();
        }
        self.changed.notify_all();
        if unmeasured_reach {
            self.device
                .degrade(Degradation::UnreachedTilesRewritten, || {
                    (
                        "a render pass whose draws' tile reach capture replay found".to_owned(),
                        "a primitive crossed the near or far plane, or lay past what Metal's tile \
                         rule was measured over, so the pass rewrote every 32x32 tile of its \
                         render area where Metal's rewrites only the tiles its primitives reach"
                            .to_owned(),
                    )
                });
        }
        let buffer = CommandBuffer(Arc::clone(self));
        // Each handler leaves the queue under the lock and runs without it, in the order added
        // (a handler may query the command buffer).
        {
            let _running = Running::enter(self.marker(Step::CompletedHandlers));
            loop {
                let Some(handler) = lock(&self.state).completed.pop_front() else {
                    break;
                };
                exit_guard::program(|| handler(&buffer));
            }
        }
        drop(buffer);
        // What the commands used goes after the completed handlers, as Metal releases it (a
        // no-copy buffer's deallocator runs after them), and before the gate opens, so a waiter
        // returns after it. Each leaves under the lock and drops without it.
        {
            let _running = Running::enter(self.marker(Step::Release));
            loop {
                let Some(resource) = lock(&self.state).retained.pop() else {
                    break;
                };
                drop(resource);
            }
            let residency = lock(&self.state).residency.take();
            drop(residency);
            let residencies = lock(&self.state).let_go_of_uses();
            drop(residencies);
            let check = lock(&self.state).check.take();
            drop(check);
            let resident = lock(&self.state).resident.take();
            drop(resident);
        }
        // Its queue's slot comes back after the completed handlers, as Metal's does: a completed
        // handler's make on its full queue never returns there, and a make after
        // `wait_until_completed` returned never blocks (RT10-qlprobe3).
        let slot = lock(&self.state).slot.take();
        drop(slot);
        self.gate.finish();
    }
}

/// A captured pass's tile mask, copied where the host reads it once the command buffer completes
/// (`render/capture.rs`), and the tiles its draws' viewports and scissors reach: the counts
/// `WorkCounts::capture_reach_tiles_found` and its kin.
pub(super) struct CaptureTally {
    /// The mask's copy: one bit a tile, tile `t` bit `t % 32` of word `t / 32`.
    pub(super) mask: Upload,
    /// The tiles of the pass's grid: the bits past them count nothing.
    pub(super) grid: u64,
    /// The tiles the draws' viewports and scissors reach.
    pub(super) bound: u64,
}

impl CaptureTally {
    /// The tiles the mask names, once the command buffer completed.
    fn found(&self) -> u64 {
        let bytes = self.mask.read_bytes();
        let mut found = 0u64;
        for (word, bits) in bytes.as_chunks::<4>().0.iter().enumerate() {
            let first = word as u64 * 32;
            let within = self.grid.saturating_sub(first).min(32);
            let live = match within {
                32 => u32::MAX,
                n => (1u32 << n) - 1,
            };
            let bits = u32::from_le_bytes([bits[0], bits[1], bits[2], bits[3]]);
            found += u64::from((bits & live).count_ones());
        }
        found
    }
}

/// A command buffer's transition, run in its queue's context.
pub(crate) struct Transition {
    /// The command buffer, held until the transition ran.
    buffer: Hold<CommandBufferShared>,
    to: To,
}

/// The status a transition goes to.
enum To {
    Scheduled,
    /// `Completed`, or `Error`.
    Finished(Result<(), CommandBufferError>),
}

impl Run for Transition {
    fn run(self) {
        match self.to {
            To::Scheduled => self.buffer.run_scheduled(),
            To::Finished(result) => self.buffer.complete(result),
        }
        // The hold goes here, after the transition: the gate opens then.
    }
}

/// Whether `wait_until_scheduled` returns: the command buffer's scheduled handlers ran (Metal's
/// wait returns only then, EV SUB-sched1-m4), or it finished, or failed without being scheduled.
fn scheduled(state: &State) -> bool {
    state.scheduled_ran
        || matches!(
            state.status,
            CommandBufferStatus::Completed | CommandBufferStatus::Error
        )
}

/// The command-buffer error of a barrier command buffer that could not be allocated, recording
/// a lost device. `allocate_raw` fails only with a Vulkan result; any other error is internal.
fn recording_error(device: &Device, error: Error) -> CommandBufferError {
    match error {
        Error::DeviceLost => submission_error(device, vk::Result::ERROR_DEVICE_LOST),
        Error::OutOfMemory(_) => CommandBufferError::OutOfMemory,
        Error::Vulkan(code) => CommandBufferError::Internal(code),
        _ => CommandBufferError::Internal(super::error::VkResultCode(
            vk::Result::ERROR_UNKNOWN.as_raw(),
        )),
    }
}

/// The command-buffer error of a host step whose recording failed with `error`: a refusal, what
/// `commit` refuses where the CPU wrote the records, is [`CommandBufferError::Refused`], kept in
/// `refusal`; an object the step needed and nothing made before it is
/// [`CommandBufferError::HostStepObject`] of its kind; a Vulkan failure, a lost device, memory,
/// and the runtime's own broken invariant fail it as a recording does ([`recording_error`]).
fn step_error(device: &Device, error: Error, refusal: &mut Option<Error>) -> CommandBufferError {
    if let Error::HostStepObject(object) = error {
        return CommandBufferError::HostStepObject(object);
    }
    let refused = match &error {
        Error::Compile(metal2vulkan::error::Error::Internal(_)) => false,
        Error::Invalid(_)
        | Error::Unsupported(_)
        | Error::Bind(_)
        | Error::Compile(_)
        | Error::NotImplemented(_)
        | Error::AppleOnly(_)
        | Error::SamplerLimit(_) => true,
        _ => false,
    };
    if !refused {
        return recording_error(device, error);
    }
    *refusal = Some(error);
    CommandBufferError::Refused
}

/// The command-buffer error of a failed submission, recording a lost device.
fn submission_error(device: &Device, error: vk::Result) -> CommandBufferError {
    match device.failed(error) {
        Error::DeviceLost => CommandBufferError::DeviceLost,
        Error::OutOfMemory(_) => CommandBufferError::OutOfMemory,
        _ => CommandBufferError::Internal(super::error::VkResultCode(error.as_raw())),
    }
}
