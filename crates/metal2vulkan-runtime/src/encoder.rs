//! The core every encoder shares: its command buffer, its own `VkCommandBuffer`, and the rule of
//! one open encoder per command buffer.
//!
//! - **Recording.** `open` allocates a primary `VkCommandBuffer` from the command buffer's pool
//!   and begins it; `end` ends it and appends it to the command buffer's recordings, which
//!   `commit` submits in that order.
//! - **Segments**. An indirect command buffer's commands may change until the command buffer is
//!   committed, so an encoder that executes them [`defer`](EncoderCore::defer)s them: it ends its
//!   `VkCommandBuffer` there and goes on in a new one, and the segment's own is recorded later
//!   ([`EncoderCore::segment`]), behind a barrier from everything before it and with one to
//!   everything after it, and goes between the two as its batch is submitted, its uses joined to
//!   its part's summary. `commit` records it from the records as they stand; a buffer the GPU may
//!   write (`IndirectCommandBuffer::gpu_written`) has its commands known only once the work before
//!   them completed, so its segment starts a part of the encoder of its own, in a batch of its own,
//!   which the scheduler's host step records (`indirect_execute.rs`).
//! - **Ordering** ([`tracker`](super::tracker)). Each command goes through [`EncoderCore::command`]
//!   with its uses: in an encoder whose commands Metal runs in order (blits, serial dispatches) it
//!   gets the barrier those uses need after the earlier commands. `end` hands the encoder's summary
//!   to the command buffer; the scheduler records the barriers between encoders, batches and queues
//!   as it submits each batch.
//! - **Shadowed textures** (`shadow.rs`). The encoder copies a 2D texture or array over a buffer,
//!   or a texture buffer viewed through a staging buffer, from the buffer before the commands
//!   that use it, and back after the ones that write it, as commands of its own; `end` copies
//!   back what is left.
//! - **Abandoned encoders.** An encoder dropped without `end_encoding` marks its command buffer
//!   abandoned: its commands are lost, so `commit` refuses rather than run without them.
//! - Encoders are `Send` but not `Sync`: every method takes `&mut self`, and the core holds a
//!   `!Sync` marker so no encoder is shared between threads by reference.

use super::command_buffer::CommandBuffer;
use super::deferred::{Deferred, StepInputs};
use super::error::{internal, invalid, vulkan, CommandBufferError, Error, Problem};
use super::execution::Execution;
use super::fresh_unwritten::FreshRead;
use super::heap::Heap;
use super::pending_dispatch::PendingDispatch;
use super::rebind::{Holds, Taken};
use super::replay::Replay;
use super::residency::Residency;
use super::resource::ResourceKey;
use super::retained::Retained;
use super::shadow::{ShadowCopy, ShadowSync, ShadowUse, Step};
use super::shadow_merge::ShadowMerge;
use super::sparse::encoder::SparseUpdate;
use super::tracker::{Barrier, EncoderUses, Summary, Use, UseRange};
use super::used_resources::EncoderKind;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{ReplayFaults, ResidentRange};
use std::any::Any;
use std::collections::BTreeSet;
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// The cores made so far in the process: the next one's [`EncoderCore::id`].
static CORES: AtomicU64 = AtomicU64::new(0);

/// One ended encoder's recording, or one part of it (`EncoderUses::split`), in its command
/// buffer's encoding order.
pub(super) struct Recorded {
    /// The encoder's place among its command buffer's ended encoders (`State::encoder_ended`),
    /// which each of its parts carries.
    pub(super) encoder: usize,
    /// Its `VkCommandBuffer`s, in order; once its batch is submitted, each segment's among them.
    pub(super) raws: Vec<vk::CommandBuffer>,
    /// Its segments, in order, until its batch is submitted.
    pub(super) segments: Vec<Segment>,
    /// What the encoder used, for the barriers its submission records; each segment's uses join
    /// it as its batch is submitted.
    pub(super) summary: Summary<ResourceKey>,
    /// A resource state encoder's sparse mapping updates, which its batch binds before its work
    /// (`sparse/`); none for any other encoder.
    pub(super) sparse: Vec<SparseUpdate>,
}

/// Commands an encoder left to record later (`deferred.rs`).
pub(super) struct Segment {
    /// Where its `VkCommandBuffer` goes: after this many of its part's own.
    pub(super) at: usize,
    /// Where its dispatches go among the command buffer's executions: after this many, the
    /// segments' placed before it not counted.
    pub(super) executions_at: usize,
    pub(super) work: Deferred,
    /// The GPU may write what the work reads (`Deferred::gpu_written` at the call): the segment
    /// starts a part of its own, and the host step records it.
    pub(super) host: bool,
    /// What `commit` left for its host step: one it foresaw, or one it did not (what the work
    /// reads written by the GPU only since).
    pub(super) inputs: Option<StepInputs>,
    /// Its recording until its batch is submitted: `commit`'s, or the host step's.
    pub(super) recorded: Option<RecordedSegment>,
    /// The host step recorded it.
    pub(super) stepped: bool,
    /// `commit` left its recording to the host step for the pipelines it records, which were
    /// being made then (`render/pending.rs`).
    pub(super) pipelines: bool,
}

impl Segment {
    /// Whether the host step records it before its batch is submitted: the GPU may write what
    /// its work reads, whether it could at the call or only since (`Deferred::gpu_written` is
    /// for good), or `commit` left it for the pipelines it records.
    pub(super) fn needs_step(&self) -> bool {
        !self.stepped && (self.host || self.pipelines || self.work.gpu_written())
    }

    /// Whether its host step waits for the work submitted before it to complete: the GPU may
    /// write what its work reads. A step for its pipelines alone waits for them alone.
    pub(super) fn steps_after_the_gpu(&self) -> bool {
        !self.stepped && (self.host || self.work.gpu_written())
    }

    /// Whether its host step waits for a pipeline still being made.
    pub(super) fn steps_after_pipelines(&self) -> bool {
        !self.stepped && self.work.making()
    }

    /// Whether it is a dispatch whose pipelines were being made (`pending_dispatch.rs`): an
    /// encoder encoded again (`replay.rs`) records it at once, waiting for them, so it leaves no
    /// segment there.
    pub(super) fn pends(&self) -> bool {
        matches!(self.work, Deferred::Pending(_))
    }

    /// What its recording will use, on `device`, while `commit` left it to the host step for the
    /// pipelines it records alone (and not for what the GPU may write): its work's uses, which
    /// the encoder knew, only the `VkPipeline` missing (`Deferred::known_summary`). `None` where
    /// the step or the submission finds them.
    pub(super) fn known_summary(&self, device: &Device) -> Option<Summary<ResourceKey>> {
        match self.pipelines && !self.steps_after_the_gpu() {
            true => self.work.known_summary(device),
            false => None,
        }
    }
}

/// The `VkCommandBuffer` an encoder goes on in after a segment, begun
/// ([`EncoderCore::continuation`]).
pub(super) struct Continuation(vk::CommandBuffer);

/// A segment recorded: its `VkCommandBuffer`, ended, and what it used and ran.
pub(super) struct RecordedSegment {
    pub(super) raw: vk::CommandBuffer,
    pub(super) summary: Summary<ResourceKey>,
    pub(super) retained: Retained,
    pub(super) executions: Vec<Execution>,
    /// The keys its commands' uses named (`used_resources.rs`).
    pub(super) keys: BTreeSet<ResourceKey>,
    /// What its commands took of buffers over host pages, and which they named (`rebind.rs`).
    pub(super) taken: Taken,
    /// The resident-set misses its recording found (`indirect_execute.rs`), which the command
    /// buffer's execution report counts.
    pub(super) resident_misses: u32,
    /// Why some of its commands ran nothing: records the host step could not run
    /// (`indirect_execute.rs`), which the execution report states.
    pub(super) faults: ReplayFaults,
}

/// The state every open encoder carries.
pub(super) struct EncoderCore {
    open: OpenGuard,
    raw: vk::CommandBuffer,
    /// The encoder's `VkCommandBuffer`s ended before a segment, in order.
    ended: Vec<vk::CommandBuffer>,
    /// The segments it left for commit, `executions_at` counting its own executions.
    segments: Vec<Segment>,
    /// The order of the encoder's commands, and its summary.
    uses: EncoderUses<ResourceKey>,
    /// What the encoder's commands use; `end` hands it to the command buffer.
    retained: Retained,
    /// What its dispatches and draws executed; `end` hands it to the command buffer.
    executions: Vec<Execution>,
    /// Where the shadowed textures the encoder used stand.
    shadows: ShadowSync,
    /// The keys its commands' uses named, and its bindings' (`used_resources.rs`): `end` finds the
    /// heaps among them, and hands them to the command buffer.
    keys: BTreeSet<ResourceKey>,
    /// The heaps its commands used through what the encoder made resident; `end` hands them,
    /// and those of `keys`, to the command buffer, whose commit marks them used.
    heaps: Vec<Heap>,
    /// Its holds on the buffers over host pages its commands used (`rebind.rs`): `end` hands
    /// them to the command buffer, whose commit lets go of them.
    holds: Holds,
    /// The pages its commands took of those buffers, and the keys a command named: `end` hands
    /// them to the command buffer, which holds the pages until it completes (weakly from its
    /// commit to its first submission).
    taken: Taken,
    /// Metal fails the command buffer from this encoder on with this error (a render pass
    /// without attachments past 16384 a side, or at a default sample count Metal fails it at):
    /// `end` says so to the command buffer, whose commit runs nothing from it on
    /// (`State::fails_from`).
    fails_command_buffer: Option<CommandBufferError>,
    /// Metal's GPU faults this encoder (an execution of commands that inherit a pipeline state
    /// made without `supportIndirectCommandBuffers`): `end` says so to the command buffer, whose
    /// commit runs nothing of it (`State::fault`).
    faults: bool,
    /// The core's own number in the process ([`id`](Self::id)).
    id: u64,
    /// It encodes an ended encoder's calls again, its command buffer committed (`replay.rs`):
    /// its commands take what they use without holding it (`Rebinds::pin`).
    replaying: bool,
    _not_sync: PhantomData<std::cell::Cell<()>>,
}

/// An encoder encoded again (`EncoderCore::replaying`), ended: what its place among its command
/// buffer's recordings takes in place of what the encoder recorded (`State::splice_replay`).
pub(super) struct Replayed {
    pub(super) raws: Vec<vk::CommandBuffer>,
    pub(super) summary: Summary<ResourceKey>,
    pub(super) retained: Retained,
    pub(super) executions: Vec<Execution>,
    pub(super) keys: BTreeSet<ResourceKey>,
    pub(super) taken: Taken,
    pub(super) heaps: Vec<Heap>,
}

/// The open encoder's hold on its command buffer: dropped before `end` ran, it marks the command
/// buffer abandoned. It is apart from [`EncoderCore`] so `end` can take the core apart, and the
/// resource state encoder holds one too.
pub(super) struct OpenGuard {
    pub(super) command_buffer: CommandBuffer,
    /// `end` ran: the drop does not abandon the command buffer.
    pub(super) ended: bool,
}

impl Drop for OpenGuard {
    fn drop(&mut self) {
        if !self.ended {
            let mut state = self.command_buffer.state();
            state.encoder_open = false;
            state.abandoned = true;
        }
    }
}

impl EncoderCore {
    /// Open an encoder on `command_buffer` for Metal's `call`, `ordered` when Metal runs its
    /// commands in encoded order (a blit or serial compute encoder). Refused when the command
    /// buffer is committed, has an open encoder, or has an abandoned one.
    pub(super) fn open(
        command_buffer: &CommandBuffer,
        call: &'static str,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        let mut state = command_buffer.state();
        if let Some(problem) = state.refusal() {
            return Err(invalid(call, problem));
        }
        let raw = command_buffer.allocate_raw(&mut state)?;
        let device = command_buffer.device().shared().vk();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, under the state lock.
        unsafe { device.begin_command_buffer(raw, &begin) }.map_err(vulkan)?;
        command_buffer
            .device()
            .shared()
            .counts
            .command_buffer_begun();
        state.encoder_open = true;
        drop(state);
        command_buffer
            .device()
            .shared()
            .identifiers
            .encoder_opened();
        Ok(EncoderCore {
            open: OpenGuard {
                command_buffer: command_buffer.clone(),
                ended: false,
            },
            raw,
            ended: Vec::new(),
            segments: Vec::new(),
            uses: EncoderUses::new(ordered),
            retained: Retained::default(),
            executions: Vec::new(),
            shadows: ShadowSync::default(),
            keys: BTreeSet::new(),
            heaps: Vec::new(),
            holds: Holds::new(command_buffer.device()),
            taken: Taken::default(),
            fails_command_buffer: None,
            faults: false,
            id: CORES.fetch_add(1, Ordering::Relaxed),
            replaying: false,
            _not_sync: PhantomData,
        })
    }

    /// The core a segment of `command_buffer` records into at its commit, `ordered` as the
    /// encoder that left it: a new `VkCommandBuffer` of its pool, begun, its first command a
    /// barrier from everything recorded before. Recorded, it ends with
    /// [`end_segment`](Self::end_segment); dropped, it abandons nothing.
    pub(super) fn segment(
        command_buffer: &CommandBuffer,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        let core = EncoderCore::segment_raw(command_buffer, ordered)?;
        // SAFETY: the command buffer is recording, and only this core records it.
        unsafe { Barrier::everything().record(core.device().shared().vk(), core.raw) };
        Ok(core)
    }

    /// The core an ended encoder of `command_buffer`, committed and not yet first submitted, is
    /// encoded again into (`replay.rs`), `ordered` as that encoder: a new `VkCommandBuffer` of
    /// its pool, begun, as [`open`](Self::open) begins one, without its refusals. Ended with
    /// [`finish_replay`](Self::finish_replay); dropped, it abandons nothing.
    pub(super) fn replaying(
        command_buffer: &CommandBuffer,
        ordered: bool,
    ) -> Result<EncoderCore, Error> {
        let mut core = EncoderCore::segment_raw(command_buffer, ordered)?;
        core.replaying = true;
        Ok(core)
    }

    /// A core of `command_buffer` over a new `VkCommandBuffer` of its pool, begun, that abandons
    /// nothing when dropped.
    fn segment_raw(command_buffer: &CommandBuffer, ordered: bool) -> Result<EncoderCore, Error> {
        let raw = {
            let mut state = command_buffer.state();
            command_buffer.allocate_raw(&mut state)?
        };
        let device = command_buffer.device().shared().vk();
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool, and only this core
        // records it.
        unsafe { device.begin_command_buffer(raw, &begin) }.map_err(vulkan)?;
        command_buffer
            .device()
            .shared()
            .counts
            .command_buffer_begun();
        Ok(EncoderCore {
            open: OpenGuard {
                command_buffer: command_buffer.clone(),
                ended: true,
            },
            raw,
            ended: Vec::new(),
            segments: Vec::new(),
            uses: EncoderUses::new(ordered),
            retained: Retained::default(),
            executions: Vec::new(),
            shadows: ShadowSync::default(),
            keys: BTreeSet::new(),
            heaps: Vec::new(),
            holds: Holds::new(command_buffer.device()),
            taken: Taken::default(),
            fails_command_buffer: None,
            faults: false,
            id: CORES.fetch_add(1, Ordering::Relaxed),
            replaying: false,
            _not_sync: PhantomData,
        })
    }

    /// The device.
    pub(super) fn device(&self) -> &Device {
        self.open.command_buffer.device()
    }

    /// The core's own number, which no other core of the process has: what an argument table
    /// retained into it holds until it changes (`ArgumentTable::retain`).
    pub(super) fn id(&self) -> u64 {
        self.id
    }

    /// Note `ranges`, the resident set a GPU-alone replay resolved as it was recorded
    /// (`Taken::resolve`).
    pub(super) fn resolved(&mut self, ranges: impl IntoIterator<Item = ResidentRange>) {
        self.taken.resolve(ranges);
    }

    /// The command buffer the encoder records into.
    pub(super) fn command_buffer(&self) -> &CommandBuffer {
        &self.open.command_buffer
    }

    /// Refuse an object of `device` in Metal's `call` unless `device` is the encoder's.
    pub(super) fn check_device(&self, call: &'static str, device: &Device) -> Result<(), Error> {
        if self.device().same(device) {
            Ok(())
        } else {
            Err(invalid(call, Problem::ForeignObject))
        }
    }

    /// Hold `object` until the command buffer completes: the encoder's commands use it.
    pub(super) fn retain(&mut self, object: impl Any + Send + Sync) {
        self.retained.boxed(Box::new(object));
    }

    /// Hold `object`, a shared one (`Buffer::held` and its kin), until the command buffer
    /// completes: by another reference to it, without a box of its own (`retained.rs`).
    pub(super) fn retain_shared(&mut self, object: Arc<dyn Any + Send + Sync>) {
        self.retained.shared(object);
    }

    /// Add a dispatch or draw the encoder took to the command buffer's execution record, as
    /// `execution` builds it. Only where the device keeps records
    /// ([`records_executions`](Self::records_executions)): elsewhere nothing is built.
    pub(super) fn executed(&mut self, execution: impl FnOnce() -> Execution) {
        if self.records_executions() {
            self.executions.push(execution());
        }
    }

    /// Whether the device keeps execution records (`DeviceOptions::record_executions`).
    pub(super) fn records_executions(&self) -> bool {
        self.device().shared().record_executions
    }

    /// Record a barrier the program asked for (a compute encoder's memory barriers).
    pub(super) fn barrier(&mut self, barrier: Barrier) {
        // SAFETY: the encoder's command buffer is recording, and only this encoder records it.
        unsafe { barrier.record(self.device().shared().vk(), self.raw) };
    }

    /// Record one command that ran in `stages` with `uses`: the barrier the uses need after the
    /// encoder's earlier commands (in an ordered encoder), then the command through `record`.
    /// Call it once every check of the command passed: recording cannot fail.
    pub(super) fn command(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) {
        self.command_reaching(stages, uses, &BTreeSet::new(), record);
    }

    /// [`command`](Self::command) of a command that reads the boxes `reads` of textures' images
    /// (a blit's sources): a read of a new texture's unwritten pattern there is announced only
    /// where a box reaches an unwritten texel (`fresh_unwritten.rs`); its other reads as
    /// `command`'s.
    pub(super) fn command_reading(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        reads: &[FreshRead],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) {
        self.command_observing(stages, uses, &BTreeSet::new(), reads, record);
    }

    /// `command` of a dispatch or draw whose uses of the keys `resident` reach
    /// their buffers through the resident set alone (`binding::CommandUses`): the command does
    /// not name them.
    pub(super) fn command_reaching(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) {
        self.command_observing(stages, uses, resident, &[], record);
    }

    /// [`command_reaching`](Self::command_reaching) reading the boxes `reads`
    /// ([`command_reading`](Self::command_reading)).
    fn command_observing(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
        reads: &[FreshRead],
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
    ) {
        let Some(barrier) = self.track(stages, uses, resident, reads) else {
            return;
        };
        let device = self.open.command_buffer.device().shared().vk();
        if let Some(barrier) = barrier {
            // SAFETY: as in `barrier`.
            unsafe { barrier.record(device, self.raw) };
        }
        record(device, self.raw);
    }

    /// [`command_reaching`](Self::command_reaching) of a command recorded later, into a segment
    /// of its own (a dispatch whose pipelines are being made, `pending_dispatch.rs`): everything
    /// of it but its recording, here, in the encoder's order. The barrier its uses need after the
    /// encoder's earlier commands is returned for its recording to record first: the segment
    /// lies between the encoder's `VkCommandBuffer`s in submission order, so the barriers of the
    /// encoder's later commands order them after it as they would in one. `None` where it
    /// records nothing, as `command_reaching` (a detached buffer).
    pub(super) fn command_later(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
    ) -> Option<Option<Barrier>> {
        self.track(stages, uses, resident, &[])
    }

    /// Everything of a command [`command_observing`](Self::command_observing) records but its
    /// recording: its reads of new textures announced, its uses tracked, and their keys kept and
    /// held. The barrier the command records first, or `None` where it records nothing.
    fn track(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
        reads: &[FreshRead],
    ) -> Option<Option<Barrier>> {
        // A command that names a detached buffer records nothing, nor does any after it.
        let named = uses
            .iter()
            .map(|used| used.key)
            .filter(|key| !resident.contains(key));
        if self.skips(named) {
            self.touch_reaching(uses, resident);
            return None;
        }
        // A command reading a new texture's levels that read zero where Metal reads its pattern,
        // before a clear covered them, announces it (`fresh_unwritten.rs`).
        self.observe_fresh(uses, reads);
        // Hazard tracking takes the uses with their buffers' host memory's; the keys kept and
        // held are the uses' own.
        let aliased = self.aliased(uses);
        let tracked = aliased.as_deref().unwrap_or(uses);
        let barrier = self.uses.barrier(tracked);
        self.uses.record(stages, tracked, barrier);
        self.touch_reaching(uses, resident);
        Some(barrier)
    }

    /// Announce `fresh-pattern-zeroed` for each texture `uses` read in a level and layer still
    /// unwritten since it was made (`fresh_unwritten.rs`): a blit's, a shader's or an input
    /// attachment's read. An attachment's own reads are its pass's load, which the pass decides
    /// as it begins (`FreshUnwritten::begin_pass`). A use of a texture `reads` names boxes of is
    /// read in those boxes alone.
    fn observe_fresh(&self, uses: &[Use<ResourceKey>], reads: &[FreshRead]) {
        let device = self.device();
        let fresh = &device.shared().fresh;
        if !fresh.any() {
            return;
        }
        let attachment = vk::AccessFlags2::COLOR_ATTACHMENT_READ
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
        let reading = |used: &&Use<ResourceKey>| !(used.read.access & !attachment).is_empty();
        for used in uses.iter().filter(reading) {
            if let UseRange::Image {
                aspects,
                levels,
                layers,
            } = &used.range
            {
                let mut boxes = reads.iter().filter(|read| read.key == used.key).peekable();
                let notice = match boxes.peek().is_some() {
                    true => boxes.find_map(|read| {
                        fresh.observed_within(read.key, read.aspects, &read.texels)
                    }),
                    false => {
                        let (levels, layers) = (levels.clone(), layers.clone());
                        fresh.observed(used.key, *aspects, levels, layers)
                    }
                };
                if let Some(notice) = notice {
                    device.degrade_fresh_pattern(notice);
                }
            }
        }
    }

    /// `uses` with a use of the host memory's key beside each use of a no-copy buffer that
    /// joined another's memory (`host_aliases.rs`), which hazard tracking and the shadows take;
    /// `None` where no use names one.
    fn aliased(&self, uses: &[Use<ResourceKey>]) -> Option<Vec<Use<ResourceKey>>> {
        self.device().shared().aliases.augment(uses)
    }

    /// Keep the keys `uses` name (`keys`); and hold each buffer over host pages among them the
    /// first time (`rebind.rs`), retaining the pages the commands took until the command buffer
    /// completes.
    fn touch(&mut self, uses: &[Use<ResourceKey>]) {
        self.touch_reaching(uses, &BTreeSet::new());
    }

    /// `touch` `uses`, of which those of the keys `resident` reach their buffers
    /// through the resident set alone: every other key is named (`Taken`), where the device has
    /// a buffer over host pages.
    fn touch_reaching(&mut self, uses: &[Use<ResourceKey>], resident: &BTreeSet<ResourceKey>) {
        self.touch_keys(uses.iter().map(|used| used.key));
        if self.device().shared().rebinds.any() {
            for used in uses.iter().filter(|used| !resident.contains(&used.key)) {
                self.taken.name(used.key);
            }
        }
    }

    /// Keep `keys`, those of resources a command uses (its own uses', or its bindings', which
    /// it may not read), holding each buffer over host pages among them as `touch` does.
    pub(super) fn touch_keys(&mut self, keys: impl IntoIterator<Item = ResourceKey>) {
        let rebinds = self.device().shared().rebinds.any();
        for key in keys {
            if self.keys.insert(key) && rebinds {
                self.hold(key);
            }
        }
    }

    /// Hold `key` where it names a buffer over host pages (`rebind.rs`).
    fn hold(&mut self, key: ResourceKey) {
        // Encoded again after the commit: nothing refuses a rebind for it.
        if self.replaying {
            if let Some(pinned) = self.holds.pin(key) {
                self.taken.pin(key, pinned);
            }
            return;
        }
        if let Some(pinned) = self.holds.hold(key) {
            self.taken.pin(key, pinned);
        }
    }

    /// Whether the encoder records nothing for a command that names `keys`: one of them is a
    /// detached buffer's (`detach.rs`), or an earlier command of the encoder named one. From
    /// that command on the encoder records nothing (`Taken::die`): its submission fails at
    /// the command buffer's first submission, `CommandBufferError::DetachedBacking` while the
    /// buffer is detached and `EncodedUseRebound` where it lies in pages again by then, so
    /// nothing it would record runs. The keys are kept, held and named as a recorded command's
    /// are. Free while no buffer is detached. Call it once every check of the command passed,
    /// before any Vulkan call that takes one of the buffers.
    pub(super) fn skips(&mut self, keys: impl IntoIterator<Item = ResourceKey> + Clone) -> bool {
        if !self.taken.dead() {
            let rebinds = &self.device().shared().rebinds;
            if !rebinds.any_detached(keys.clone()) {
                return false;
            }
            self.taken.die();
        }
        self.touch_keys(keys.clone());
        for key in keys {
            self.taken.name(key);
        }
        true
    }

    /// The keys its commands named so far, their uses' and their bindings' (`touch`).
    pub(super) fn named_keys(&self) -> &BTreeSet<ResourceKey> {
        &self.keys
    }

    /// Whether its commands reached a heap so far: one they made resident, or the heap of a
    /// resource they named.
    pub(super) fn reached_heaps(&self) -> bool {
        let heap_keys = &self.device().shared().heap_keys;
        !self.heaps.is_empty() || !heap_keys.among(self.keys.iter().copied()).is_empty()
    }

    /// Note `heaps`, which a command the encoder recorded reached through what it made resident.
    pub(super) fn touch_heaps(&mut self, heaps: impl IntoIterator<Item = Heap>) {
        let rebinds = self.device().shared().rebinds.any();
        for heap in heaps {
            if !self.heaps.contains(&heap) {
                // A heap over host pages is held as its resources' uses are.
                let key = heap.bytes().map(|bytes| bytes.identity().key);
                if let Some(key) = key.filter(|&key| rebinds && self.keys.insert(key)) {
                    self.hold(key);
                }
                self.heaps.push(heap);
            }
        }
    }

    /// Note one command recorded outside the encoder's own `VkCommandBuffer` (a render encoder's
    /// draw, in its secondary command buffer) that runs in `stages` with `uses`. Only an
    /// unordered encoder's commands are recorded so: they need no barrier between them. Call it
    /// before recording the command; it refuses (internal) an ordered encoder.
    pub(super) fn note_reaching(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<ResourceKey>],
        resident: &BTreeSet<ResourceKey>,
    ) -> Result<(), Error> {
        if self.uses.ordered() {
            return Err(super::error::internal(
                "a command recorded apart in an ordered encoder",
            ));
        }
        self.observe_fresh(uses, &[]);
        let aliased = self.aliased(uses);
        self.uses
            .record(stages, aliased.as_deref().unwrap_or(uses), None);
        self.touch_reaching(uses, resident);
        Ok(())
    }

    /// The copies a command that uses the shadowed textures `shadows` and has `uses` needs
    /// before it, which the encoder then counts as recorded: the scratch of a merge (`shadow.rs`,
    /// `shadow_merge.rs`) retained until the command buffer completes. Refused, changing
    /// nothing, as the scratch is. Call it once every other check of the command passed, and
    /// record the copies with [`shadow_copies`](Self::shadow_copies), then the command.
    pub(super) fn plan_shadows(
        &mut self,
        shadows: &[ShadowUse],
        uses: &[Use<ResourceKey>],
    ) -> Result<Vec<ShadowCopy>, Error> {
        let device = self.device().clone();
        let aliased = self.aliased(uses);
        let uses = aliased.as_deref().unwrap_or(uses);
        let copies = self.shadows.before(shadows, uses, |copies| {
            ShadowMerge::new(&device, copies).map(Arc::new)
        })?;
        for copy in &copies {
            if let Step::Keep(merge) = &copy.step {
                self.retain(Arc::clone(merge));
            }
        }
        Ok(copies)
    }

    /// Note a recorded command that used the shadowed textures `shadows` and had `uses`.
    pub(super) fn shadows_used(&mut self, shadows: &[ShadowUse], uses: &[Use<ResourceKey>]) {
        let aliased = self.aliased(uses);
        self.shadows
            .after(shadows, aliased.as_deref().unwrap_or(uses));
    }

    /// Record shadowed textures' `copies` as commands of the encoder: ordered as any command
    /// is, in an encoder Metal orders; with a barrier from everything before and one to
    /// everything after each copy, in one whose commands Metal leaves unordered.
    pub(super) fn shadow_copies(&mut self, copies: &[ShadowCopy]) {
        for copy in copies {
            let stages = copy.stages();
            let uses = copy.uses();
            if self.uses.ordered() {
                self.command(stages, &uses, |device, raw| {
                    // SAFETY: the encoder's command buffer is recording outside a render pass;
                    // its commands retain the shadowed textures until it completes, and every
                    // image rests in the `GENERAL` layout.
                    unsafe { copy.record(device, raw) }
                });
            } else {
                // A copy of a detached buffer's bytes records nothing, nor does what follows.
                if self.skips(uses.iter().map(|used| used.key)) {
                    continue;
                }
                self.barrier(Barrier::everything());
                // SAFETY: as above.
                unsafe { copy.record(self.device().shared().vk(), self.raw) };
                self.barrier(Barrier::everything());
                let aliased = self.aliased(&uses);
                self.uses
                    .record(stages, aliased.as_deref().unwrap_or(&uses), None);
                self.touch(&uses);
            }
        }
    }

    /// Whether it encodes an ended encoder's calls again ([`replaying`](Self::replaying)): its
    /// dispatches wait for their pipelines, leaving no segment (`pending_dispatch.rs`).
    pub(super) fn encodes_again(&self) -> bool {
        self.replaying
    }

    /// The encoder's uses, for its fence waits and updates.
    pub(super) fn uses(&mut self) -> &mut EncoderUses<ResourceKey> {
        &mut self.uses
    }

    /// Leave `work` to record here later: the shadowed textures written since their last copy
    /// back are copied back where the work leaves them (`Deferred::leaves_shadows`), the
    /// encoder's `VkCommandBuffer` ends, and its later commands go into a new one. Where the GPU
    /// may write what the work reads, the segment starts a part of the encoder of its own (the
    /// module's segments). Refused, changing nothing, when the new one cannot be made; a failure
    /// to end the current one abandons the command buffer.
    pub(super) fn defer(&mut self, work: Deferred) -> Result<(), Error> {
        let next = self.continuation()?;
        self.defer_into(next, work)
    }

    /// The new `VkCommandBuffer` the encoder's commands go on in after a segment
    /// ([`defer`](Self::defer)), allocated from the command buffer's pool and begun. Left unused
    /// on a refusal, it goes with the pool.
    pub(super) fn continuation(&self) -> Result<Continuation, Error> {
        let next = {
            let command_buffer = &self.open.command_buffer;
            let mut state = command_buffer.state();
            command_buffer.allocate_raw(&mut state)?
        };
        let begin = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        // SAFETY: the command buffer was just allocated from the pool; only this encoder
        // records it.
        unsafe {
            self.device()
                .shared()
                .vk()
                .begin_command_buffer(next, &begin)
        }
        .map_err(vulkan)?;
        self.device().shared().counts.command_buffer_begun();
        Ok(Continuation(next))
    }

    /// [`defer`](Self::defer) `work`, the encoder going on in `next`: refused only where the
    /// current `VkCommandBuffer` does not end, which abandons the command buffer.
    fn defer_into(&mut self, next: Continuation, work: Deferred) -> Result<(), Error> {
        if work.leaves_shadows() {
            let copies = self.shadows.leave();
            self.shadow_copies(&copies);
        }
        // SAFETY: as in `barrier`.
        if let Err(error) = unsafe { self.device().shared().vk().end_command_buffer(self.raw) } {
            self.open.command_buffer.state().abandoned = true;
            return Err(vulkan(error));
        }
        self.ended.push(self.raw);
        self.raw = next.0;
        let host = work.gpu_written();
        if host {
            self.uses.split();
        }
        self.segments.push(Segment {
            at: self.ended.len(),
            executions_at: self.executions.len(),
            work,
            host,
            inputs: None,
            recorded: None,
            stepped: false,
            pipelines: false,
        });
        Ok(())
    }

    /// Leave `dispatch`, whose pipelines are being made and which the encoder tracked already
    /// ([`command_later`](Self::command_later)), to record later (`pending_dispatch.rs`), the
    /// encoder going on in `next`. Refused only where the current `VkCommandBuffer` does not
    /// end, which abandons the command buffer.
    pub(super) fn pend(
        &mut self,
        next: Continuation,
        dispatch: PendingDispatch,
    ) -> Result<(), Error> {
        self.defer_into(next, Deferred::Pending(Arc::new(dispatch)))
    }

    /// Metal fails the command buffer from this encoder on with `error`, as it commits: the
    /// encoder and every one after it run nothing (`State::fail_from`).
    pub(super) fn fail_command_buffer(&mut self, error: CommandBufferError) {
        self.fails_command_buffer = Some(error);
    }

    /// Metal's GPU faults this encoder, as the command buffer commits: the encoder runs nothing,
    /// and the command buffer completes with `CommandBufferError::PageFault` (`State::fault`).
    pub(super) fn fault(&mut self) {
        self.faults = true;
    }

    /// End the encoder, one of `kind` that made `residency` resident (a compute or render
    /// encoder's): its shadowed textures written since their last copy back are copied back,
    /// it joins the command buffer's ended encoders (`State::encoder_ended`), and its recording
    /// and what it used join the command buffer's, in order, one part at a time, each part
    /// after the first in a batch of its own.
    /// The command buffer keeps `replay`, the encoder's log (`replay.rs`), where a rebind before
    /// its first submission may need the encoder encoded again (a command of it named a buffer
    /// over host pages, detached ones included) and the encoder can be (it left no segment for
    /// later, and fails nothing as it commits); else the log goes here.
    pub(super) fn end_with_replay(
        mut self,
        kind: EncoderKind,
        residency: Option<Residency>,
        replay: Option<Replay>,
    ) -> Result<(), Error> {
        let replay = replay.filter(|_| {
            self.segments.iter().all(Segment::pends)
                && self.fails_command_buffer.is_none()
                && !self.faults
                && !self.taken.named_pins().is_empty()
        });
        let copies = self.shadows.finish();
        self.shadow_copies(&copies);
        // SAFETY: as in `barrier`.
        unsafe { self.device().shared().vk().end_command_buffer(self.raw) }.map_err(vulkan)?;
        let EncoderCore {
            mut open,
            raw,
            mut ended,
            segments,
            uses,
            retained,
            executions,
            keys,
            mut heaps,
            holds,
            taken,
            fails_command_buffer,
            faults,
            ..
        } = self;
        ended.push(raw);
        let keyed = open
            .command_buffer
            .device()
            .shared()
            .heap_keys
            .among(keys.iter().copied());
        heaps.extend(keyed);
        let mut state = open.command_buffer.state();
        state.use_heaps(heaps);
        let encoder = state.encoder_ended(kind, keys, residency, taken);
        if let Some(error) = fails_command_buffer {
            state.fail_from(encoder, error);
        }
        if faults {
            state.fault(encoder);
        }
        let base = state.executions.len();
        // One part, its executions these: a log not kept comes back, to go without the lock.
        let logged = replay.as_ref().map(Replay::bytes);
        let unkept = state.keep_replay(encoder, replay, base..base + executions.len());
        if let (Some(bytes), None) = (logged, &unkept) {
            open.command_buffer
                .device()
                .shared()
                .counts
                .replay_log_kept(bytes);
        }
        let mut summaries = uses.finish_parts().into_iter();
        let mut raws = ended.into_iter();
        // The raws the parts before the current one took.
        let mut taken = 0;
        let mut parts = Vec::new();
        let mut current = Vec::new();
        for mut segment in segments {
            segment.executions_at += base;
            if segment.host {
                let before = segment.at - taken;
                parts.push((raws.by_ref().take(before).collect::<Vec<_>>(), current));
                current = Vec::new();
                taken = segment.at;
            }
            segment.at -= taken;
            current.push(segment);
        }
        parts.push((raws.collect(), current));
        for (at, (raws, segments)) in parts.into_iter().enumerate() {
            let summary = summaries
                .next()
                .ok_or_else(|| internal("an encoder's part without its summary"))?;
            if at > 0 {
                state.host_step();
            }
            state.end_encoder(Recorded {
                encoder,
                raws,
                segments,
                summary,
                sparse: Vec::new(),
            });
        }
        state.retained.extend(retained.into_entry());
        state.executions.extend(executions);
        if !holds.is_empty() {
            state.holds.push(holds);
        }
        drop(state);
        drop(unkept);
        open.ended = true;
        Ok(())
    }

    /// End a core made by [`replaying`](Self::replaying): its shadowed textures written since
    /// their last copy back are copied back, and what it recorded comes back for its encoder's
    /// place (`State::splice_replay`). Refused (internal) where the encoder, encoded again, left
    /// a segment, split into parts, failed or faulted its command buffer, or met a detached
    /// buffer: its first encoding kept none of those, so its place stays as it was.
    pub(super) fn finish_replay(mut self) -> Result<Replayed, Error> {
        let copies = self.shadows.finish();
        self.shadow_copies(&copies);
        // SAFETY: as in `barrier`.
        unsafe { self.device().shared().vk().end_command_buffer(self.raw) }.map_err(vulkan)?;
        let EncoderCore {
            raw,
            mut ended,
            segments,
            uses,
            retained,
            executions,
            keys,
            heaps,
            taken,
            fails_command_buffer,
            faults,
            ..
        } = self;
        if !segments.is_empty() || fails_command_buffer.is_some() || faults || taken.dead() {
            return Err(internal(
                "an encoder encoded again differs from its first encoding",
            ));
        }
        let mut summaries = uses.finish_parts();
        let summary = match (summaries.pop(), summaries.is_empty()) {
            (Some(summary), true) => summary,
            _ => return Err(internal("an encoder encoded again in parts")),
        };
        ended.push(raw);
        Ok(Replayed {
            raws: ended,
            summary,
            retained,
            executions,
            keys,
            taken,
            heaps,
        })
    }
}

impl EncoderCore {
    /// A segment of `command_buffer` that `record` records alone into a new `VkCommandBuffer` of
    /// its pool, retaining `retained`: no barrier around it and nothing tracked, its encoder
    /// having tracked its commands where it left them ([`command_later`](Self::command_later)),
    /// so it uses nothing past its encoder's summary.
    pub(super) fn untracked_segment(
        command_buffer: &CommandBuffer,
        record: impl FnOnce(&ash::Device, vk::CommandBuffer),
        retained: Box<dyn Any + Send + Sync>,
    ) -> Result<RecordedSegment, Error> {
        let mut core = EncoderCore::segment_raw(command_buffer, false)?;
        let device = core.device().shared().vk();
        record(device, core.raw);
        // SAFETY: as in `barrier`.
        unsafe { device.end_command_buffer(core.raw) }.map_err(vulkan)?;
        core.retained.boxed(retained);
        let EncoderCore {
            raw,
            uses,
            retained,
            executions,
            keys,
            taken,
            ..
        } = core;
        Ok(RecordedSegment {
            raw,
            summary: uses.finish(),
            retained,
            executions,
            keys,
            taken,
            resident_misses: 0,
            faults: ReplayFaults::default(),
        })
    }

    /// End a [`segment`](Self::segment): its shadowed textures written since their last copy
    /// back are copied back, then a barrier to everything recorded after it.
    pub(super) fn end_segment(mut self) -> Result<RecordedSegment, Error> {
        let copies = self.shadows.finish();
        self.shadow_copies(&copies);
        self.barrier(Barrier::everything());
        // SAFETY: as in `barrier`.
        unsafe { self.device().shared().vk().end_command_buffer(self.raw) }.map_err(vulkan)?;
        let EncoderCore {
            raw,
            uses,
            retained,
            executions,
            keys,
            taken,
            ..
        } = self;
        Ok(RecordedSegment {
            raw,
            summary: uses.finish(),
            retained,
            executions,
            keys,
            taken,
            resident_misses: 0,
            faults: ReplayFaults::default(),
        })
    }
}
