//! Metal's automatic hazard tracking, as pure functions of what was recorded so far and what comes
//! next.
//!
//! Metal orders accesses on its own at two levels, and the runtime records a Vulkan barrier, or a
//! wait for another `VkQueue`, for each ordering it needs:
//! - **Inside an encoder** ([`EncoderUses`]). A blit encoder's commands and a serial compute
//!   encoder's dispatches run in encoded order: `MTLDispatchTypeSerial` "dispatches are executed
//!   in dispatched order", and a memory barrier on a serial encoder is "allowed, but ignored", so
//!   the order covers untracked resources too (Metal's headers; documented, not measured). Before
//!   each command, [`EncoderUses::barrier`] compares its uses with the encoder's earlier commands:
//!   tracked resources per resource and range, and all untracked memory as one class, since
//!   untracked heap resources may alias.
//! - **Between encoders, command buffers and queues** ([`Tracker`]). The device's scheduler
//!   submits batches (a command buffer's encoders between its event waits and signals) in one
//!   order over all its logical queues, and [`Tracker::plan`] compares each batch's encoder
//!   [`Summary`]s (their tracked uses, the stages they ran in, and their fence waits and updates)
//!   with the device's state in that order. Metal orders tracked resources across queues in the
//!   order it schedules the command buffers, which is this order. Untracked resources are the
//!   program's to order, through fences, barriers and events, and the summary leaves them out.
//!   Metal evaluates a fence at submission, so a wait orders after the fence's last update in
//!   submission order.
//!
//! **Barriers and waits.** Every image rests in `VK_IMAGE_LAYOUT_GENERAL` outside render passes,
//! so nothing here changes a layout. An access after one of the same `VkQueue` needs a
//! [`Barrier`]: one global memory barrier, everything in `src` before everything in `dst`, which
//! orders buffer and image memory alike; Vulkan's submission order on one `VkQueue` makes it cover
//! the earlier batches of every logical queue there. An access after one of another `VkQueue`
//! needs a [`Wait`] of its batch for that queue's timeline semaphore at the earlier batch's
//! [`Point`]: the signal makes the earlier writes available and the wait makes them visible to
//! the waiting stages of this batch and of every later one of its queue. That batch was submitted
//! before, so the wait never waits on an unsubmitted signal. Both may order more than Metal
//! requires, never less.
//!
//! **The state.** Per resource, per range: the last write on each queue that wrote it (one queue's
//! unless entries were merged), with the stages and access types it is already visible to on its
//! own queue and the stages of other queues that waited for it, and the reads since, one set of
//! stages per queue. A read needs a barrier or a wait from a write not yet visible to it. A write
//! needs one from the last writes (unless already visible to it) and from the reads since (an
//! execution dependency). A write drops the older entries its range holds: they are ordered
//! before it, and so before whatever follows it. A resource with more than [`ENTRIES_PER_KEY`]
//! entries has them merged into one per kind of range, which orders more and stays exact. A
//! write or read retires once its batch completed and the host saw it ([`Tracker::completed`]);
//! each `VkQueue`'s batches complete in its timeline's order. A queue submission orders
//! everything that happened before it on the host, and every batch starts with
//! [`Barrier::after_completed`], so later work needs no barrier from it.

use ash::vk;
use std::collections::HashMap;
use std::fmt;
use std::hash::Hash;
use std::ops::Range;

/// The most entries the state keeps for one resource before it merges them.
pub(super) const ENTRIES_PER_KEY: usize = 8;

/// One side of a dependency: pipeline stages, and the access types in them.
#[derive(Clone, Copy, Default, PartialEq, Eq, Hash)]
pub(super) struct Access {
    pub(super) stages: vk::PipelineStageFlags2,
    pub(super) access: vk::AccessFlags2,
}

impl Access {
    /// No stage, and so no access.
    pub(super) const NONE: Access =
        Access::new(vk::PipelineStageFlags2::NONE, vk::AccessFlags2::NONE);

    /// `access` in `stages`.
    pub(super) const fn new(stages: vk::PipelineStageFlags2, access: vk::AccessFlags2) -> Access {
        Access { stages, access }
    }

    /// Whether this side names no stage.
    pub(super) fn is_empty(self) -> bool {
        self.stages.is_empty()
    }

    /// Both sides' stages and access types.
    pub(super) fn union(self, other: Access) -> Access {
        Access::new(self.stages | other.stages, self.access | other.access)
    }

    fn intersection(self, other: Access) -> Access {
        Access::new(self.stages & other.stages, self.access & other.access)
    }

    /// Whether every stage and access type of `other` is in this side.
    fn covers(self, other: Access) -> bool {
        self.stages.contains(other.stages) && self.access.contains(other.access)
    }
}

// `ash` implements `Debug` for its flags only with its `debug` feature: the runtime prints raw
// values.
impl fmt::Debug for Access {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Access {{ stages: {:#x}, access: {:#x} }}",
            self.stages.as_raw(),
            self.access.as_raw()
        )
    }
}

/// The part of a resource a use touches. One resource identity is one kind of storage: a buffer
/// (or heap) is bytes, and an image is aspects, mip levels and array layers. A texture view uses
/// its parent's identity and range; a texture aliasing a buffer uses the buffer's identity and
/// bytes.
#[derive(Clone, PartialEq, Eq, Hash)]
pub(super) enum UseRange {
    /// Bytes `start..end`.
    Bytes(Range<u64>),
    /// Aspects, mip levels and array layers of an image. 3D depth slices are not separate, since
    /// Vulkan's barriers cannot name them.
    Image {
        aspects: vk::ImageAspectFlags,
        levels: Range<u32>,
        layers: Range<u32>,
    },
}

impl UseRange {
    /// The range every untracked use takes inside an encoder: all untracked memory is one class.
    const UNTRACKED: UseRange = UseRange::Bytes(0..u64::MAX);

    /// Whether the range holds nothing.
    pub(super) fn is_empty(&self) -> bool {
        match self {
            UseRange::Bytes(bytes) => bytes.is_empty(),
            UseRange::Image {
                aspects,
                levels,
                layers,
            } => aspects.is_empty() || levels.is_empty() || layers.is_empty(),
        }
    }

    /// Whether the two ranges share memory. Ranges of different kinds on one key are taken to
    /// share it, which is always correct.
    pub(super) fn overlaps(&self, other: &UseRange) -> bool {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => meet(a, b),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => a.intersects(*b) && meet(a_levels, b_levels) && meet(a_layers, b_layers),
            _ => true,
        }
    }

    /// Whether this range holds all of `other`.
    pub(super) fn contains(&self, other: &UseRange) -> bool {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => within(a, b),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => a.contains(*b) && within(a_levels, b_levels) && within(a_layers, b_layers),
            _ => false,
        }
    }

    /// The smallest range of the same kind holding both, or `None` across kinds.
    fn hull(&self, other: &UseRange) -> Option<UseRange> {
        match (self, other) {
            (UseRange::Bytes(a), UseRange::Bytes(b)) => Some(UseRange::Bytes(hull(a, b))),
            (
                UseRange::Image {
                    aspects: a,
                    levels: a_levels,
                    layers: a_layers,
                },
                UseRange::Image {
                    aspects: b,
                    levels: b_levels,
                    layers: b_layers,
                },
            ) => Some(UseRange::Image {
                aspects: *a | *b,
                levels: hull(a_levels, b_levels),
                layers: hull(a_layers, b_layers),
            }),
            _ => None,
        }
    }
}

impl fmt::Debug for UseRange {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UseRange::Bytes(bytes) => write!(f, "Bytes({bytes:?})"),
            UseRange::Image {
                aspects,
                levels,
                layers,
            } => write!(
                f,
                "Image {{ aspects: {:#x}, levels: {levels:?}, layers: {layers:?} }}",
                aspects.as_raw()
            ),
        }
    }
}

fn meet<T: Ord>(a: &Range<T>, b: &Range<T>) -> bool {
    a.start < b.end && b.start < a.end
}

fn within<T: Ord>(outer: &Range<T>, inner: &Range<T>) -> bool {
    outer.start <= inner.start && inner.end <= outer.end
}

fn hull<T: Ord + Copy>(a: &Range<T>, b: &Range<T>) -> Range<T> {
    a.start.min(b.start)..a.end.max(b.end)
}

/// One command's, or one encoder's, access of one resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Use<K> {
    /// The resource's identity: never reused, shared by its views.
    pub(super) key: K,
    /// Whether Metal tracks the resource's hazards (its resolved `hazardTrackingMode`).
    pub(super) tracked: bool,
    pub(super) range: UseRange,
    /// The reads, empty when the use only writes.
    pub(super) read: Access,
    /// The writes, empty when the use only reads.
    pub(super) write: Access,
}

impl<K> Use<K> {
    /// A read of `range`.
    pub(super) fn reads(key: K, tracked: bool, range: UseRange, read: Access) -> Use<K> {
        Use {
            key,
            tracked,
            range,
            read,
            write: Access::NONE,
        }
    }

    /// A write of `range`.
    pub(super) fn writes(key: K, tracked: bool, range: UseRange, write: Access) -> Use<K> {
        Use {
            key,
            tracked,
            range,
            read: Access::NONE,
            write,
        }
    }

    /// Whether the use touches nothing: an empty range, or neither a read nor a write.
    fn is_nothing(&self) -> bool {
        self.range.is_empty() || (self.read.is_empty() && self.write.is_empty())
    }
}

/// One global memory barrier: every access of `src` before every access of `dst`. Without access
/// types on a side it is an execution dependency only (a write after a read).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Barrier {
    pub(super) src: Access,
    pub(super) dst: Access,
}

impl Barrier {
    /// The start of every batch with encoders: what completed batches wrote is visible to every
    /// later command. The tracker forgets a batch once it completed, so its writes are ordered by
    /// completion (its queue's timeline, observed on the host before this submission) and need
    /// only this visibility; the empty source waits for nothing.
    pub(super) fn after_completed() -> Barrier {
        Barrier {
            src: Access::NONE,
            dst: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        }
    }

    /// The end of every batch: every earlier write before the host's reads and writes, so the CPU
    /// sees the GPU's writes once the batch completes, and before the batch's signals.
    pub(super) fn to_host() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::HOST,
                vk::AccessFlags2::HOST_READ | vk::AccessFlags2::HOST_WRITE,
            ),
        }
    }

    /// Everything recorded before, before everything recorded after: a shadowed texture's copy
    /// in an encoder whose commands Metal does not order (`shadow.rs`).
    pub(super) fn everything() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::ALL_COMMANDS,
                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
            ),
        }
    }

    /// A compute encoder's memory barrier (`memoryBarrierWithScope:` and
    /// `memoryBarrierWithResources:`): every shader write of the earlier dispatches before every
    /// shader, uniform and indirect-argument access of the later ones.
    pub(super) fn between_dispatches() -> Barrier {
        Barrier {
            src: Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_WRITE,
            ),
            dst: Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER | vk::PipelineStageFlags2::DRAW_INDIRECT,
                vk::AccessFlags2::SHADER_READ
                    | vk::AccessFlags2::SHADER_WRITE
                    | vk::AccessFlags2::UNIFORM_READ
                    | vk::AccessFlags2::INDIRECT_COMMAND_READ,
            ),
        }
    }

    /// Record the barrier on `command_buffer`: one global `VkMemoryBarrier` through
    /// `vkCmdPipelineBarrier`, with [`Barrier::legacy`]'s masks. The runtime records every
    /// barrier this way, with or without synchronization2: each mask it makes maps to the same or
    /// a broader legacy mask, so one path is exact.
    ///
    /// # Safety
    ///
    /// `command_buffer` is a command buffer of `device` in the recording state, and no other
    /// thread records it.
    pub(super) unsafe fn record(self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let masks = self.legacy();
        let barrier = vk::MemoryBarrier::default()
            .src_access_mask(masks.src_access)
            .dst_access_mask(masks.dst_access);
        // SAFETY: the caller's contract.
        unsafe {
            device.cmd_pipeline_barrier(
                command_buffer,
                masks.src_stages,
                masks.dst_stages,
                vk::DependencyFlags::empty(),
                &[barrier],
                &[],
                &[],
            );
        }
    }

    fn add(&mut self, src: Access, dst: Access) {
        self.src = self.src.union(src);
        self.dst = self.dst.union(dst);
    }

    /// The barrier, when it orders anything.
    fn needed(self) -> Option<Barrier> {
        (!self.src.is_empty() && !self.dst.is_empty()).then_some(self)
    }

    /// The masks of the barrier without synchronization2 (`vkCmdPipelineBarrier` with one
    /// `VkMemoryBarrier`). The two flag families share the bits the older one defines; each newer
    /// bit maps to the older stage or access type that holds it, or to `ALL_COMMANDS` and the
    /// memory access types, which order more.
    pub(super) fn legacy(self) -> LegacyBarrier {
        LegacyBarrier {
            src_stages: legacy_stages(self.src.stages, vk::PipelineStageFlags::TOP_OF_PIPE),
            src_access: legacy_access(self.src.access),
            dst_stages: legacy_stages(self.dst.stages, vk::PipelineStageFlags::BOTTOM_OF_PIPE),
            dst_access: legacy_access(self.dst.access),
        }
    }
}

/// A [`Barrier`]'s masks for `vkCmdPipelineBarrier`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct LegacyBarrier {
    pub(super) src_stages: vk::PipelineStageFlags,
    pub(super) src_access: vk::AccessFlags,
    pub(super) dst_stages: vk::PipelineStageFlags,
    pub(super) dst_access: vk::AccessFlags,
}

impl fmt::Debug for LegacyBarrier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "LegacyBarrier {{ src: {:#x}/{:#x}, dst: {:#x}/{:#x} }}",
            self.src_stages.as_raw(),
            self.src_access.as_raw(),
            self.dst_stages.as_raw(),
            self.dst_access.as_raw()
        )
    }
}

/// The stage bits `VkPipelineStageFlags` defines, the same in both families (bits 0 to 25).
const LEGACY_STAGES: u64 = (1 << 26) - 1;
/// The access bits `VkAccessFlags` defines, the same in both families (bits 0 to 27).
const LEGACY_ACCESS: u64 = (1 << 28) - 1;

fn legacy_stages(
    stages: vk::PipelineStageFlags2,
    none: vk::PipelineStageFlags,
) -> vk::PipelineStageFlags {
    use vk::PipelineStageFlags as Old;
    use vk::PipelineStageFlags2 as New;
    if stages.is_empty() {
        return none;
    }
    let mut old = Old::from_raw((stages.as_raw() & LEGACY_STAGES) as u32);
    let mut high = New::from_raw(stages.as_raw() & !LEGACY_STAGES);
    for (new, into) in [
        (
            New::COPY | New::RESOLVE | New::BLIT | New::CLEAR,
            Old::TRANSFER,
        ),
        (
            New::INDEX_INPUT | New::VERTEX_ATTRIBUTE_INPUT,
            Old::VERTEX_INPUT,
        ),
        // The pre-rasterization stages a device has: naming them one by one would name stages
        // whose features may be off.
        (New::PRE_RASTERIZATION_SHADERS, Old::ALL_GRAPHICS),
    ] {
        if high.intersects(new) {
            old |= into;
            high &= !new;
        }
    }
    if !high.is_empty() {
        old |= Old::ALL_COMMANDS;
    }
    old
}

fn legacy_access(access: vk::AccessFlags2) -> vk::AccessFlags {
    use vk::AccessFlags as Old;
    use vk::AccessFlags2 as New;
    let mut old = Old::from_raw((access.as_raw() & LEGACY_ACCESS) as u32);
    let mut high = New::from_raw(access.as_raw() & !LEGACY_ACCESS);
    for (new, into) in [
        (
            New::SHADER_SAMPLED_READ | New::SHADER_STORAGE_READ,
            Old::SHADER_READ,
        ),
        (New::SHADER_STORAGE_WRITE, Old::SHADER_WRITE),
    ] {
        if high.intersects(new) {
            old |= into;
            high &= !new;
        }
    }
    if !high.is_empty() {
        old |= Old::MEMORY_READ | Old::MEMORY_WRITE;
    }
    old
}

/// A batch's place in the device's submission order: the `VkQueue` it ran on (its index among
/// the device's queues), and the value that queue's timeline semaphore reaches once the batch and
/// every batch submitted before it on that queue completed.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(super) struct Point {
    pub(super) queue: usize,
    pub(super) value: u64,
}

/// A wait of a batch for another `VkQueue`: the batch's `stages` (and those of every later batch
/// of its queue) run once that queue's timeline semaphore reached `at.value`.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) struct Wait {
    pub(super) at: Point,
    pub(super) stages: vk::PipelineStageFlags2,
}

impl fmt::Debug for Wait {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Wait {{ at: {:?}, stages: {:#x} }}",
            self.at,
            self.stages.as_raw()
        )
    }
}

impl Wait {
    /// The wait's stages for `vkQueueSubmit`'s `pWaitDstStageMask`: each maps to the same or a
    /// broader legacy stage, as a barrier's do ([`Barrier::legacy`]).
    pub(super) fn legacy_stages(self) -> vk::PipelineStageFlags {
        legacy_stages(self.stages, vk::PipelineStageFlags::ALL_COMMANDS)
    }
}

/// Add `wait` to `waits`, which hold at most one wait per queue: the latest value, for every
/// stage either names.
fn add_wait(waits: &mut Vec<Wait>, wait: Wait) {
    if wait.stages.is_empty() {
        return;
    }
    match waits
        .iter_mut()
        .find(|known| known.at.queue == wait.at.queue)
    {
        Some(known) => {
            known.at.value = known.at.value.max(wait.at.value);
            known.stages |= wait.stages;
        }
        None => waits.push(wait),
    }
}

/// What a use needs ordered before it: a barrier on its own queue, and waits for other queues.
#[derive(Default)]
struct Needs {
    barrier: Barrier,
    waits: Vec<Wait>,
}

/// One queue's last write over a range.
#[derive(Clone)]
struct Writer {
    /// The batch that wrote.
    at: Point,
    /// The stages and access types of the writer's own queue the write is already visible to.
    visible: Access,
    /// The other queues that waited for the write, with the stages their waits cover: a timeline
    /// wait makes the write visible to every access of those stages.
    waited: Vec<(usize, vk::PipelineStageFlags2)>,
}

impl Writer {
    /// The stages of `queue` a wait already ordered after the write.
    fn waited_by(&self, queue: usize) -> vk::PipelineStageFlags2 {
        self.waited
            .iter()
            .find(|(by, _)| *by == queue)
            .map_or(vk::PipelineStageFlags2::NONE, |&(_, stages)| stages)
    }

    /// Merge the write of the same queue in `other` into this one: what both are visible to.
    fn absorb(&mut self, other: &Writer) {
        self.at.value = self.at.value.max(other.at.value);
        self.visible = self.visible.intersection(other.visible);
        self.waited.retain_mut(|(queue, stages)| {
            *stages &= other.waited_by(*queue);
            !stages.is_empty()
        });
    }
}

/// One queue's reads of a range since the last write: their stages, and the latest batch.
#[derive(Clone, Copy)]
struct Reads {
    at: Point,
    stages: vk::PipelineStageFlags2,
}

/// What the state knows of one range of one resource.
#[derive(Clone)]
struct Entry {
    range: UseRange,
    /// The access types and stages of the last writes, empty when none is known.
    write: Access,
    /// The last write on each queue that wrote the range, one per queue: one queue's unless
    /// entries were merged.
    writers: Vec<Writer>,
    /// The reads since the last write, one per queue.
    reads: Vec<Reads>,
}

impl Entry {
    /// The entry a use makes in a batch at `at`.
    fn of<K>(used: &Use<K>, at: Point) -> Entry {
        let mut entry = Entry {
            range: used.range.clone(),
            write: used.write,
            writers: Vec::new(),
            reads: Vec::new(),
        };
        if !used.write.is_empty() {
            entry.writers.push(Writer {
                at,
                visible: Access::NONE,
                waited: Vec::new(),
            });
        }
        entry.add_reads(at, used.read.stages);
        entry
    }

    /// What a use at `at` of this entry's range needs ordered before it.
    fn depend(&self, read: Access, write: Access, at: Point, needs: &mut Needs) {
        if !read.is_empty() {
            self.after_writes(read, at, needs);
        }
        if !write.is_empty() {
            self.after_writes(write, at, needs);
            for reads in &self.reads {
                if reads.at.queue == at.queue {
                    needs.barrier.add(
                        Access::new(reads.stages, vk::AccessFlags2::NONE),
                        Access::new(write.stages, vk::AccessFlags2::NONE),
                    );
                } else {
                    add_wait(
                        &mut needs.waits,
                        Wait {
                            at: reads.at,
                            stages: write.stages,
                        },
                    );
                }
            }
        }
    }

    /// What `access` at `at` needs after the last writes: a barrier from a write of its own
    /// queue not yet visible to it, and a wait for a write of another queue not yet waited for.
    fn after_writes(&self, access: Access, at: Point, needs: &mut Needs) {
        for writer in &self.writers {
            if writer.at.queue == at.queue {
                if !writer.visible.covers(access) {
                    needs.barrier.add(self.write, access);
                }
            } else if !writer.waited_by(at.queue).contains(access.stages) {
                add_wait(
                    &mut needs.waits,
                    Wait {
                        at: writer.at,
                        stages: access.stages,
                    },
                );
            }
        }
    }

    /// Note what a batch at `at` ordered after the writes before its use: `barrier`, recorded on
    /// `at.queue`, and the batch's `waits` for other queues.
    fn ordered(&mut self, barrier: Option<Barrier>, waits: &[Wait], at: Point) {
        let write = self.write;
        for writer in &mut self.writers {
            if writer.at.queue == at.queue {
                if let Some(barrier) = barrier {
                    if barrier.src.covers(write) {
                        writer.visible = writer.visible.union(barrier.dst);
                    }
                }
            } else if let Some(wait) = waits
                .iter()
                .find(|wait| wait.at.queue == writer.at.queue && wait.at.value >= writer.at.value)
            {
                match writer.waited.iter_mut().find(|(by, _)| *by == at.queue) {
                    Some((_, stages)) => *stages |= wait.stages,
                    None => writer.waited.push((at.queue, wait.stages)),
                }
            }
        }
    }

    /// Add reads in `stages` at `at`.
    fn add_reads(&mut self, at: Point, stages: vk::PipelineStageFlags2) {
        if stages.is_empty() {
            return;
        }
        match self
            .reads
            .iter_mut()
            .find(|reads| reads.at.queue == at.queue)
        {
            Some(reads) => {
                reads.at.value = reads.at.value.max(at.value);
                reads.stages |= stages;
            }
            None => self.reads.push(Reads { at, stages }),
        }
    }

    /// Merge `other` into this entry over `range` (one that holds both): it orders after both.
    fn absorb(&mut self, other: Entry, range: UseRange) {
        self.range = range;
        self.write = self.write.union(other.write);
        for writer in other.writers {
            match self
                .writers
                .iter_mut()
                .find(|mine| mine.at.queue == writer.at.queue)
            {
                Some(mine) => mine.absorb(&writer),
                None => self.writers.push(writer),
            }
        }
        for reads in other.reads {
            self.add_reads(reads.at, reads.stages);
        }
    }

    /// Forget the writes and reads of the batches `completed` holds, by queue: the host saw them
    /// complete before any later submission. Whether nothing is left.
    fn retire(&mut self, completed: &[u64]) -> bool {
        let done = |at: Point| {
            completed
                .get(at.queue)
                .is_some_and(|&through| at.value <= through)
        };
        self.writers.retain(|writer| !done(writer.at));
        if self.writers.is_empty() {
            self.write = Access::NONE;
        }
        self.reads.retain(|reads| !done(reads.at));
        self.writers.is_empty() && self.reads.is_empty()
    }
}

/// What the uses (sorted by key, one per key and range) at `at` need after the accesses in
/// `state`.
fn depend<S: Eq + Hash>(state: &HashMap<S, Vec<Entry>>, uses: &[Use<S>], at: Point) -> Needs {
    let mut needs = Needs::default();
    for used in uses {
        for entry in state.get(&used.key).into_iter().flatten() {
            if entry.range.overlaps(&used.range) {
                entry.depend(used.read, used.write, at, &mut needs);
            }
        }
    }
    needs
}

/// Add the uses (sorted by key, one per key and range) of a batch at `at` to `state`, after
/// `barrier` was recorded before them and with the batch's `waits`.
fn admit<S: Copy + Eq + Hash>(
    state: &mut HashMap<S, Vec<Entry>>,
    uses: &[Use<S>],
    barrier: Option<Barrier>,
    waits: &[Wait],
    at: Point,
) {
    for group in uses.chunk_by(|a, b| a.key == b.key) {
        let entries = state.entry(group[0].key).or_default();
        for entry in entries.iter_mut() {
            entry.ordered(barrier, waits, at);
        }
        // The older entries a write of the group covers are ordered before it.
        entries.retain(|entry| {
            !group
                .iter()
                .any(|used| !used.write.is_empty() && used.range.contains(&entry.range))
        });
        for used in group {
            // An older entry of the same range remains only when this use does not write.
            match entries.iter_mut().find(|entry| entry.range == used.range) {
                Some(entry) => entry.add_reads(at, used.read.stages),
                None => entries.push(Entry::of(used, at)),
            }
        }
        if entries.len() > ENTRIES_PER_KEY {
            coalesce(entries);
        }
    }
}

/// Merge a resource's entries into one per kind of range.
fn coalesce(entries: &mut Vec<Entry>) {
    let mut first = 0;
    while first < entries.len() {
        let mut next = first + 1;
        while next < entries.len() {
            match entries[first].range.hull(&entries[next].range) {
                Some(range) => {
                    let other = entries.swap_remove(next);
                    entries[first].absorb(other, range);
                }
                None => next += 1,
            }
        }
        first += 1;
    }
}

/// The uses one per key and range, sorted by key, without the ones that touch nothing.
fn merged<S: Copy + Ord + Hash>(uses: impl IntoIterator<Item = Use<S>>) -> Vec<Use<S>> {
    let mut merged: Vec<Use<S>> = Vec::new();
    let mut index: HashMap<(S, UseRange), usize> = HashMap::new();
    for used in uses {
        if used.is_nothing() {
            continue;
        }
        match index.get(&(used.key, used.range.clone())) {
            Some(&at) => {
                merged[at].read = merged[at].read.union(used.read);
                merged[at].write = merged[at].write.union(used.write);
            }
            None => {
                index.insert((used.key, used.range.clone()), merged.len());
                merged.push(used);
            }
        }
    }
    merged.sort_by_key(|used| used.key);
    merged
}

/// The key a use takes inside an encoder: its resource when tracked, else the one class of all
/// untracked memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
enum Class<K> {
    Tracked(K),
    Untracked,
}

/// The uses of one encoder: the order of its commands, and the summaries its commit reads, one
/// per part. An encoder is one part until it executes commands the GPU wrote: those run once the
/// work before them completed, so the encoder's commands before and after them go to batches of
/// their own, each with its own summary ([`EncoderUses::split`]).
pub(super) struct EncoderUses<K> {
    /// Whether Metal runs the commands in encoded order (blit encoders, serial compute encoders).
    ordered: bool,
    /// The accesses of the current part's commands so far, for that order.
    inside: HashMap<Class<K>, Vec<Entry>>,
    /// The parts, in order: the last is the current one. Never empty.
    parts: Vec<Part<K>>,
    /// The fences waited for: before the encoder's first part.
    waits: Vec<K>,
    /// The fences updated: after its last part.
    updates: Vec<K>,
}

/// The uses of one part of an encoder.
struct Part<K> {
    /// The tracked uses so far, one per range, at most [`ENTRIES_PER_KEY`] per resource.
    tracked: HashMap<K, Vec<Use<K>>>,
    stages: vk::PipelineStageFlags2,
}

impl<K> Part<K> {
    fn new() -> Part<K> {
        Part {
            tracked: HashMap::new(),
            stages: vk::PipelineStageFlags2::NONE,
        }
    }
}

impl<K: Copy + Ord + Hash> EncoderUses<K> {
    /// The uses of a new encoder, `ordered` when Metal runs its commands in encoded order.
    pub(super) fn new(ordered: bool) -> EncoderUses<K> {
        EncoderUses {
            ordered,
            inside: HashMap::new(),
            parts: vec![Part::new()],
            waits: Vec::new(),
            updates: Vec::new(),
        }
    }

    /// Whether Metal runs the encoder's commands in encoded order.
    pub(super) fn ordered(&self) -> bool {
        self.ordered
    }

    /// The barrier a command with `uses` needs after the encoder's earlier commands: none in an
    /// encoder whose commands are not ordered. Nothing changes until [`EncoderUses::record`].
    pub(super) fn barrier(&self, uses: &[Use<K>]) -> Option<Barrier> {
        if !self.ordered {
            return None;
        }
        // One encoder runs on one queue: its commands need barriers, never waits.
        depend(&self.inside, &inside_uses(uses), Point::default())
            .barrier
            .needed()
    }

    /// Note a recorded command: it ran in `stages`, with `uses`, after `barrier`.
    pub(super) fn record(
        &mut self,
        stages: vk::PipelineStageFlags2,
        uses: &[Use<K>],
        barrier: Option<Barrier>,
    ) {
        let part = self.current();
        // Every stage, untracked work's too: a fence wait orders all of the encoder's work.
        part.stages |= uses.iter().fold(stages, |all, used| {
            all | used.read.stages | used.write.stages
        });
        if self.ordered {
            admit(
                &mut self.inside,
                &inside_uses(uses),
                barrier,
                &[],
                Point::default(),
            );
        }
        let part = self.current();
        for used in uses
            .iter()
            .filter(|used| used.tracked && !used.is_nothing())
        {
            let ranges = part.tracked.entry(used.key).or_default();
            match ranges.iter_mut().find(|known| known.range == used.range) {
                Some(known) => {
                    known.read = known.read.union(used.read);
                    known.write = known.write.union(used.write);
                }
                None => ranges.push(used.clone()),
            }
            if ranges.len() > ENTRIES_PER_KEY {
                coalesce_uses(ranges);
            }
        }
    }

    /// Whether a command the encoder recorded so far writes memory one of `uses` reads: an
    /// overlapping range of a tracked resource. A command recorded before the encoder's earlier
    /// commands run (a render encoder's index bounds pass, recorded before its pass) would read
    /// it before that write. Untracked resources are not recorded, so they are not seen.
    pub(super) fn written(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|read| read.tracked && !read.read.is_empty())
            .any(|read| {
                self.parts.iter().any(|part| {
                    part.tracked.get(&read.key).is_some_and(|known| {
                        known
                            .iter()
                            .any(|k| !k.write.is_empty() && k.range.overlaps(&read.range))
                    })
                })
            })
    }

    /// Whether a command the encoder recorded so far reads or writes memory one of `uses`
    /// writes: an overlapping range of a tracked resource. A command recorded before the
    /// encoder's earlier commands run (a render encoder's emulated mesh dispatch, recorded before
    /// its draw's render pass instance at the latest) would write it before that access.
    pub(super) fn accessed(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|write| write.tracked && !write.write.is_empty())
            .any(|write| {
                self.parts.iter().any(|part| {
                    part.tracked
                        .get(&write.key)
                        .is_some_and(|known| known.iter().any(|k| k.range.overlaps(&write.range)))
                })
            })
    }

    /// The tracked uses of the commands recorded so far that ran in one of `stages`.
    pub(super) fn tracked_in(&self, stages: vk::PipelineStageFlags2) -> Vec<Use<K>> {
        self.parts
            .iter()
            .flat_map(|part| part.tracked.values().flatten())
            .filter(|known| (known.read.stages | known.write.stages).intersects(stages))
            .cloned()
            .collect()
    }

    /// Whether a command the encoder recorded so far touches, in one of `stages`, memory one of
    /// `uses` touches where either writes ([`touches`]).
    pub(super) fn touched_in(&self, uses: &[Use<K>], stages: vk::PipelineStageFlags2) -> bool {
        uses.iter().any(|used| {
            self.parts.iter().any(|part| {
                part.tracked
                    .get(&used.key)
                    .is_some_and(|known| known.iter().any(|k| touches(k, used, stages)))
            })
        })
    }

    /// `wait_for_fence`: the encoder runs after the fence's last update in commit order before
    /// it. Fences are waited for at the start of the encoder and updated at its end, so a wait
    /// after the encoder updated the fence waits for the update before the encoder, as plain
    /// Metal's compute, blit and render encoders do (an M4 read the earlier encoder's write in
    /// three runs of each); Metal's API validation asserts.
    pub(super) fn wait_for_fence(&mut self, fence: K) {
        if !self.waits.contains(&fence) {
            self.waits.push(fence);
        }
    }

    /// `update_fence`: the fence is updated after the encoder's work.
    pub(super) fn update_fence(&mut self, fence: K) {
        if !self.updates.contains(&fence) {
            self.updates.push(fence);
        }
    }

    /// The part commands are recorded into now.
    fn current(&mut self) -> &mut Part<K> {
        let last = self.parts.len() - 1;
        &mut self.parts[last]
    }

    /// End the current part: the commands after this run in a part of their own, after
    /// everything before it completed, so they need no barrier after the commands before.
    pub(super) fn split(&mut self) {
        self.inside.clear();
        self.parts.push(Part::new());
    }

    /// The summary of the ended encoder, its parts joined (an encoder of one part: its part's).
    pub(super) fn finish(self) -> Summary<K> {
        self.finish_parts()
            .into_iter()
            .reduce(|mut summary, part| {
                summary.join(part);
                summary
            })
            .expect("an encoder has a part")
    }

    /// The summaries of the ended encoder's parts, in order: the fence waits go with the first,
    /// and the updates with the last.
    pub(super) fn finish_parts(self) -> Vec<Summary<K>> {
        let EncoderUses {
            parts,
            waits,
            updates,
            ..
        } = self;
        let mut summaries: Vec<Summary<K>> = parts
            .into_iter()
            .map(|part| {
                let mut uses: Vec<Use<K>> = part.tracked.into_values().flatten().collect();
                uses.sort_by_key(|used| used.key);
                Summary {
                    uses,
                    stages: part.stages,
                    waits: Vec::new(),
                    updates: Vec::new(),
                }
            })
            .collect();
        if let Some(first) = summaries.first_mut() {
            first.waits = waits;
        }
        if let Some(last) = summaries.last_mut() {
            last.updates = updates;
        }
        summaries
    }
}

impl<K> fmt::Debug for EncoderUses<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("EncoderUses")
            .field("ordered", &self.ordered)
            .field("parts", &self.parts.len())
            .field("waits", &self.waits.len())
            .field("updates", &self.updates.len())
            .finish_non_exhaustive()
    }
}

/// The tracked writes of an unordered encoder's commands since its last memory barrier (a render
/// encoder's draws and tile dispatches, `render.rs`): the writes no barrier orders before the
/// commands after them. One entry per range, at most [`ENTRIES_PER_KEY`] per resource, merged
/// as [`EncoderUses`] merges its own, which claims more and stays exact.
pub(super) struct UnorderedWrites<K> {
    writes: HashMap<K, Vec<Use<K>>>,
}

impl<K: Copy + Ord + Hash> UnorderedWrites<K> {
    /// No writes.
    pub(super) fn new() -> UnorderedWrites<K> {
        UnorderedWrites {
            writes: HashMap::new(),
        }
    }

    /// Add the tracked writes of a command with `uses`.
    pub(super) fn add(&mut self, uses: &[Use<K>]) {
        for used in uses
            .iter()
            .filter(|used| used.tracked && !used.write.is_empty() && !used.range.is_empty())
        {
            let ranges = self.writes.entry(used.key).or_default();
            ranges.push(Use {
                read: Access::NONE,
                ..used.clone()
            });
            if ranges.len() > ENTRIES_PER_KEY {
                coalesce_uses(ranges);
            }
        }
    }

    /// A memory barrier orders every write so far before the commands after it.
    pub(super) fn clear(&mut self) {
        self.writes.clear();
    }

    /// Whether one of the writes overlaps memory one of `uses` reads, as
    /// [`EncoderUses::written`] asks of every write of the encoder.
    pub(super) fn written(&self, uses: &[Use<K>]) -> bool {
        uses.iter()
            .filter(|read| read.tracked && !read.read.is_empty())
            .any(|read| {
                self.writes
                    .get(&read.key)
                    .is_some_and(|known| known.iter().any(|k| k.range.overlaps(&read.range)))
            })
    }
}

/// Whether `known`, a use of earlier commands, touches in one of `stages` memory `used` touches
/// where either writes: overlapping ranges of one tracked resource, `known` writing in `stages`
/// what `used` reads, or reading or writing in `stages` what `used` writes. `known`'s stages
/// are its commands' together, so it claims more and stays exact.
pub(super) fn touches<K: PartialEq>(
    known: &Use<K>,
    used: &Use<K>,
    stages: vk::PipelineStageFlags2,
) -> bool {
    let reads = !used.read.is_empty() && known.write.stages.intersects(stages);
    let writes =
        !used.write.is_empty() && (known.read.stages | known.write.stages).intersects(stages);
    used.tracked && known.key == used.key && (reads || writes) && known.range.overlaps(&used.range)
}

/// Merge one resource's uses into one per kind of range, which claims more and stays exact.
fn coalesce_uses<K>(uses: &mut Vec<Use<K>>) {
    let mut first = 0;
    while first < uses.len() {
        let mut next = first + 1;
        while next < uses.len() {
            match uses[first].range.hull(&uses[next].range) {
                Some(range) => {
                    let other = uses.swap_remove(next);
                    let merged = &mut uses[first];
                    merged.range = range;
                    merged.read = merged.read.union(other.read);
                    merged.write = merged.write.union(other.write);
                }
                None => next += 1,
            }
        }
        first += 1;
    }
}

/// The uses as the encoder orders them: untracked ones as the one class of untracked memory.
fn inside_uses<K: Copy + Ord + Hash>(uses: &[Use<K>]) -> Vec<Use<Class<K>>> {
    merged(uses.iter().map(|used| Use {
        key: if used.tracked {
            Class::Tracked(used.key)
        } else {
            Class::Untracked
        },
        tracked: used.tracked,
        range: if used.tracked {
            used.range.clone()
        } else {
            UseRange::UNTRACKED
        },
        read: used.read,
        write: used.write,
    }))
}

/// What one ended encoder hands to its queue's tracker at commit.
#[derive(Clone)]
pub(super) struct Summary<K> {
    /// The tracked uses, one per resource and range, sorted by resource.
    uses: Vec<Use<K>>,
    /// Every stage the encoder's commands ran in.
    stages: vk::PipelineStageFlags2,
    /// The fences waited for before the encoder.
    waits: Vec<K>,
    /// The fences updated after the encoder.
    updates: Vec<K>,
}

impl<K> fmt::Debug for Summary<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Summary")
            .field("uses", &self.uses.len())
            .field("stages", &format_args!("{:#x}", self.stages.as_raw()))
            .field("waits", &self.waits.len())
            .field("updates", &self.updates.len())
            .finish()
    }
}

impl<K: Copy + Ord> Summary<K> {
    /// The summary of work outside any encoder that makes `uses` in every stage, with no fence:
    /// a batch's layout transitions of the caller's images, planned among its encoders.
    pub(super) fn outside_encoders(mut uses: Vec<Use<K>>) -> Summary<K> {
        uses.sort_by_key(|used| used.key);
        Summary {
            uses,
            stages: vk::PipelineStageFlags2::ALL_COMMANDS,
            waits: Vec::new(),
            updates: Vec::new(),
        }
    }

    /// The summary of a resource state encoder, which records no command: it waits for the fences
    /// `waits` and updates `updates`, in every stage, so a wait on its queue orders after all of
    /// its batch's work before it, which its sparse binding precedes.
    pub(super) fn of_fences(waits: Vec<K>, updates: Vec<K>) -> Summary<K> {
        Summary {
            uses: Vec::new(),
            stages: vk::PipelineStageFlags2::ALL_COMMANDS,
            waits,
            updates,
        }
    }

    /// The summary of an encoder whose commands do not run (`CommandBuffer::commit_failed`): its
    /// fences' waits and updates alone, as [`of_fences`](Self::of_fences) makes them.
    pub(super) fn fences_only(&self) -> Summary<K> {
        Summary::of_fences(self.waits.clone(), self.updates.clone())
    }

    /// The resources of the tracked uses, in order, each once per range.
    pub(super) fn keys(&self) -> impl Iterator<Item = K> + '_ {
        self.uses.iter().map(|used| used.key)
    }

    /// Join `other`, the summary of work recorded apart that runs inside the same encoder (an
    /// indirect command buffer's commands, recorded at commit): its stages, fences and tracked
    /// uses, one per resource and range. A resource past [`ENTRIES_PER_KEY`] ranges takes a new
    /// range into one of the same kind, which claims more and stays exact.
    pub(super) fn join(&mut self, other: Summary<K>) {
        self.stages |= other.stages;
        for used in other.uses {
            if let Some(known) = self
                .uses
                .iter_mut()
                .find(|known| known.key == used.key && known.range == used.range)
            {
                known.read = known.read.union(used.read);
                known.write = known.write.union(used.write);
                continue;
            }
            let ranges = self
                .uses
                .iter()
                .filter(|known| known.key == used.key)
                .count();
            let hull = (ranges >= ENTRIES_PER_KEY)
                .then(|| {
                    self.uses.iter_mut().find_map(|known| {
                        let range = (known.key == used.key)
                            .then(|| known.range.hull(&used.range))
                            .flatten()?;
                        Some((known, range))
                    })
                })
                .flatten();
            match hull {
                Some((known, range)) => {
                    known.range = range;
                    known.read = known.read.union(used.read);
                    known.write = known.write.union(used.write);
                }
                None => self.uses.push(used),
            }
        }
        self.uses.sort_by_key(|used| used.key);
        for fence in other.waits {
            if !self.waits.contains(&fence) {
                self.waits.push(fence);
            }
        }
        for fence in other.updates {
            if !self.updates.contains(&fence) {
                self.updates.push(fence);
            }
        }
    }
}

impl<K> Summary<K> {
    /// The tracked uses.
    #[cfg(test)]
    pub(super) fn uses(&self) -> &[Use<K>] {
        &self.uses
    }

    /// Every stage the encoder's commands ran in.
    #[cfg(test)]
    pub(super) fn stages(&self) -> vk::PipelineStageFlags2 {
        self.stages
    }
}

/// What a batch's summaries make and wait for, as the scheduler compares a batch held for a
/// reason of the runtime's own with one committed after it on another queue (`scheduler/order.rs`):
/// its tracked uses, sorted by resource, and the fences it waits for and updates.
pub(super) struct Footprint<K> {
    uses: Vec<Use<K>>,
    waits: Vec<K>,
    updates: Vec<K>,
}

impl<K: Copy + Ord> Footprint<K> {
    /// The footprint of a batch whose work these summaries hold.
    pub(super) fn of<'a>(summaries: impl IntoIterator<Item = &'a Summary<K>>) -> Footprint<K>
    where
        K: 'a,
    {
        let mut footprint = Footprint {
            uses: Vec::new(),
            waits: Vec::new(),
            updates: Vec::new(),
        };
        for summary in summaries {
            footprint.uses.extend(summary.uses.iter().cloned());
            footprint.waits.extend(&summary.waits);
            footprint.updates.extend(&summary.updates);
        }
        footprint.uses.sort_by_key(|used| used.key);
        footprint
    }

    /// Whether a tracked use of the batch names a resource `named` answers for.
    pub(super) fn names_any(&self, mut named: impl FnMut(&K) -> bool) -> bool {
        self.uses.iter().any(|used| named(&used.key))
    }

    /// Whether the batch makes no tracked use and names no fence: no order across queues
    /// concerns it.
    fn is_empty(&self) -> bool {
        self.uses.is_empty() && self.waits.is_empty() && self.updates.is_empty()
    }

    /// Whether `self`, committed first, and `later` must run in that order: a tracked use of
    /// each shares a resource's range and one of them writes it, or one updates a fence the
    /// other waits for or updates.
    fn conflicts(&self, later: &Footprint<K>) -> bool {
        let fences = |updates: &[K], other: &Footprint<K>| {
            updates
                .iter()
                .any(|fence| other.waits.contains(fence) || other.updates.contains(fence))
        };
        if fences(&self.updates, later) || fences(&later.updates, self) {
            return true;
        }
        later.uses.iter().any(|used| {
            let first = self.uses.partition_point(|earlier| earlier.key < used.key);
            self.uses[first..]
                .iter()
                .take_while(|earlier| earlier.key == used.key)
                .any(|earlier| {
                    earlier.range.overlaps(&used.range)
                        && !(earlier.write.is_empty() && used.write.is_empty())
                })
        })
    }
}

/// Whether a batch whose footprint is `earlier`, committed first, and one whose footprint is
/// `later` must run in that order across queues; `None` is a batch whose uses are not known yet
/// (work a host step or its submission records), which may conflict with any batch that makes
/// a tracked use or names a fence.
pub(super) fn ordered<K: Copy + Ord>(
    earlier: Option<&Footprint<K>>,
    later: Option<&Footprint<K>>,
) -> bool {
    match (earlier, later) {
        (Some(earlier), Some(later)) => earlier.conflicts(later),
        (None, Some(known)) | (Some(known), None) => !known.is_empty(),
        (None, None) => true,
    }
}

/// A fence's last update in the device's submission order.
#[derive(Clone, Copy)]
struct FenceUpdate {
    /// The stages it covers: the updating encoder's, and those of the fences it waited for.
    stages: vk::PipelineStageFlags2,
    /// The batch of the updating encoder.
    at: Point,
}

/// The device's hazard state, in the scheduler's submission order: every logical queue's batches,
/// on every `VkQueue`.
pub(super) struct Tracker<K> {
    resources: HashMap<K, Vec<Entry>>,
    fences: HashMap<K, FenceUpdate>,
    /// Per `VkQueue`: the timeline value through which every batch completed, as the host saw.
    completed: Vec<u64>,
}

impl<K> Default for Tracker<K> {
    fn default() -> Tracker<K> {
        Tracker {
            resources: HashMap::new(),
            fences: HashMap::new(),
            completed: Vec::new(),
        }
    }
}

impl<K> fmt::Debug for Tracker<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tracker")
            .field("resources", &self.resources.len())
            .field("fences", &self.fences.len())
            .field("completed", &self.completed)
            .finish()
    }
}

/// The barriers before each encoder of a batch submitted next, the batch's waits for other
/// queues, and the state after it. [`Tracker::apply`] keeps the state once the batch is
/// submitted; a batch that is never submitted drops its plan, and the state is as before.
pub(super) struct Plan<K> {
    barriers: Vec<Option<Barrier>>,
    waits: Vec<Wait>,
    resources: HashMap<K, Vec<Entry>>,
    fences: HashMap<K, FenceUpdate>,
}

impl<K> fmt::Debug for Plan<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Plan")
            .field("barriers", &self.barriers)
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

impl<K> Plan<K> {
    /// The barrier before each encoder, in order.
    pub(super) fn barriers(&self) -> &[Option<Barrier>] {
        &self.barriers
    }

    /// The batch's waits for other queues' timelines, at most one per queue.
    pub(super) fn waits(&self) -> &[Wait] {
        &self.waits
    }
}

impl<K: Copy + Ord + Hash> Tracker<K> {
    /// Every batch of `queue` through timeline value `value` completed, and the host saw it: the
    /// state forgets what they did.
    pub(super) fn completed(&mut self, queue: usize, value: u64) {
        if self.completed.len() <= queue {
            self.completed.resize(queue + 1, 0);
        }
        if value <= self.completed[queue] {
            return;
        }
        self.completed[queue] = value;
        let completed = &self.completed;
        self.resources.retain(|_, entries| {
            entries.retain_mut(|entry| !entry.retire(completed));
            !entries.is_empty()
        });
        self.fences.retain(|_, update| {
            completed
                .get(update.at.queue)
                .is_none_or(|&through| update.at.value > through)
        });
    }

    /// The barriers before the `encoders` of the batch submitted next, at `at`, and its waits.
    /// The state is unchanged.
    pub(super) fn plan<'a>(
        &self,
        encoders: impl IntoIterator<Item = &'a Summary<K>>,
        at: Point,
    ) -> Plan<K>
    where
        K: 'a,
    {
        let mut plan = Plan {
            barriers: Vec::new(),
            waits: Vec::new(),
            resources: HashMap::new(),
            fences: HashMap::new(),
        };
        for summary in encoders {
            let mut barrier = Barrier::default();
            let mut waited = vk::PipelineStageFlags2::NONE;
            for fence in &summary.waits {
                let update = plan.fences.get(fence).or_else(|| self.fences.get(fence));
                if let Some(update) = update {
                    waited |= update.stages;
                    if update.at.queue == at.queue {
                        barrier.add(
                            Access::new(update.stages, vk::AccessFlags2::MEMORY_WRITE),
                            Access::new(
                                summary.stages,
                                vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE,
                            ),
                        );
                    } else {
                        add_wait(
                            &mut plan.waits,
                            Wait {
                                at: update.at,
                                stages: summary.stages,
                            },
                        );
                    }
                }
            }
            for used in &summary.uses {
                plan.resources
                    .entry(used.key)
                    .or_insert_with(|| self.resources.get(&used.key).cloned().unwrap_or_default());
            }
            let needs = depend(&plan.resources, &summary.uses, at);
            barrier.add(needs.barrier.src, needs.barrier.dst);
            for wait in needs.waits {
                add_wait(&mut plan.waits, wait);
            }
            let barrier = barrier.needed();
            admit(&mut plan.resources, &summary.uses, barrier, &plan.waits, at);
            for &fence in &summary.updates {
                let stages = summary.stages | waited;
                plan.fences.insert(fence, FenceUpdate { stages, at });
            }
            plan.barriers.push(barrier);
        }
        plan
    }

    /// Keep the state after a planned batch that was submitted.
    pub(super) fn apply(&mut self, plan: Plan<K>) {
        for (key, entries) in plan.resources {
            if entries.is_empty() {
                self.resources.remove(&key);
            } else {
                self.resources.insert(key, entries);
            }
        }
        self.fences.extend(plan.fences);
    }

    /// How many entries the state holds, over all resources and fences.
    #[cfg(test)]
    pub(super) fn entries(&self) -> usize {
        self.resources.values().map(Vec::len).sum::<usize>() + self.fences.len()
    }
}
