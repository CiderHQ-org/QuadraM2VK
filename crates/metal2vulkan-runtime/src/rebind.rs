//! Rebinding a no-copy buffer, or a heap over the caller's host memory, to other host pages
//! (a runtime extension, the companion of `new_buffer_with_bytes_no_copy` and
//! `new_heap_with_bytes_no_copy`).
//!
//! A consumer that gathers its own pages into one contiguous range (Mach VM remapping on macOS,
//! `memfd` mappings on Linux) gives an existing no-copy buffer or heap a new range of the same
//! length. Metal, and MoltenVK, bind a no-copy buffer's physical pages when it is made, so a
//! later remap over its range never reaches the GPU; the runtime instead keeps the object and
//! its Metal address, and binds it to the new range from then on.
//!
//! - **What stays.** The buffer or heap and every resource placed in it, their `gpu_address`
//!   (the device's own Metal address space, `address.rs`: the resident table maps each address
//!   to the `VkBuffer` behind it at encode time), their resource IDs, labels, offsets and
//!   options. `contents` answers the new pointer at once.
//! - **The ordering rule, by first submission** (by commit before). Metal resolves a command
//!   buffer's uses as it submits it (the consumer's rebind probe, tables 2 and 3, the M3 Max): one
//!   first submitted before the rebind keeps the old pages until it completes; one committed before
//!   it and first submitted after reaches the new pages through what it uses or made resident, and
//!   none through what it only reaches unnamed (the rebind ends the use). The runtime resolves both
//!   at the first submission (`command_buffer/first_submission.rs`); a command it encoded that
//!   names the buffer took the pages as it was encoded (`Taken`), so the rebind encodes its
//!   compute, blit and render encoders that named the buffer again, in the rebind's call, before
//!   the command buffer's first submission (`replay.rs`, `command_buffer/reencode.rs`); an encoder
//!   it cannot encode again (one of a command buffer that used command predication, among others)
//!   fails `CommandBufferError::EncodedUseRebound` there, as all did before. Commands encoded after
//!   the rebind use the new pages. A rebind while a command buffer that is not yet committed has
//!   encoded a use of the buffer or heap (one of its resources included), or has an encoder open
//!   that used one (a render encoder uses its attachments' bytes from its open), is refused
//!   (`Invalid`, [`RebindProblem::EncodedUses`]): its commands took the old pages, and committing
//!   them after the rebind would be neither order. A command buffer committed with commands the
//!   runtime records only once earlier work completed (executions of an indirect command buffer
//!   the GPU writes) counts as uncommitted until it completed.
//! - **The new pages' import.** New memory that is the bytes a live no-copy buffer was made over,
//!   of its memory class, lies in that buffer's import, as a buffer made over them does
//!   (`host_aliases.rs`): the rebind imports nothing, pins no page and asks the kernel about no
//!   mapping (`WorkCounts::host_mappings_queried`). A heap's, and a buffer's with a texture's
//!   aliasing image over its bytes, import their new memory themselves.
//! - **The deallocators.** The old range's runs once every command buffer first submitted
//!   before the rebind that used it completed (at the rebind where none holds it: a command
//!   buffer not yet submitted holds its pages weakly); the new range's at the next rebind's turn,
//!   or when the buffer or heap goes.
//! - **Textures.** Nothing of the old contents is copied. A heap texture bound to the heap's
//!   bytes (a no-copy heap's linear image), and a texture over a no-copy buffer (or over a
//!   buffer of a no-copy heap) whose linear image aliases its bytes, has its image made again over
//!   the new pages, at its offset, before the rebind is made; the image swaps in under the texture
//!   and every view of it (`Image::swap_in`), and the old image and its views go with the old
//!   pages, which the command buffers that used them hold. A texture held as a shadow of the bytes
//!   (a heap's, or one over a no-copy buffer) reads its buffer's `VkBuffer` as each copy is
//!   recorded (`ShadowBuffer::Rebound`); a texture buffer's views are emptied, going with the old
//!   pages, and made again at the next bind. The texture slots' generations change, so each command
//!   buffer committed after the rebind writes its set-1 elements again (`heap_sets.rs`). The
//!   copies, views and aliasing images are planned at the bytes' offset in their `VkBuffer`, which
//!   a heap's rebind keeps (its first byte at the import's unit); a no-copy buffer with a texture
//!   over it is rebound only to memory at the same offset in its page
//!   (`NotImplemented(RebindTextures)` else), as is one whose aliasing image the device cannot make
//!   again over the new pages (their memory type is not one it takes). Over new pages imported as
//!   dma-bufs, each such image is made again over a dma-buf of its own new pages before the rebind
//!   settles (`Device::dma_buf_images_again`, `dma_buf.rs`), and refused likewise where it is not.

#[cfg(unix)]
use super::buffer::Imported;
use super::buffer::{Buffer, Detached, HostCell, HostPages, Lying, WeakBuffer};
use super::command_buffer::CommandBufferShared;
use super::device::lock;
use super::error::{invalid, Error, Problem};
use super::heap::Heap;
use super::host_memory::{Deallocator, HostMemory, HostRelease, KeepAlive};
use super::key_hash::KeyMap;
use super::program_objects;
use super::resource::{ObjectId, ResourceKey};
use super::Device;
use metal2vulkan::bind::ResidentRange;
use std::any::Any;
use std::collections::BTreeSet;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};

/// What `rebind_bytes_no_copy` refuses as Metal's validation would a misuse (`Invalid`).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RebindProblem {
    /// The buffer is not a no-copy buffer of its own (a buffer of the runtime's memory, one over
    /// the caller's `VkBuffer`, or a heap's buffer, whose heap is rebound instead), or the heap is
    /// not over the caller's host memory.
    NotHostMemory,
    /// A command buffer that is not yet committed has encoded a use of the buffer or heap, or of
    /// a resource placed in it, or has an encoder open that used one (a render encoder uses its
    /// attachments' bytes from its open), or one committed whose commands the runtime records
    /// only once earlier work completed: commit or drop it first.
    EncodedUses,
}

/// The device's buffers over host pages, by their hazard-tracking key, with the command buffers
/// not yet committed whose commands used each.
#[derive(Default)]
pub(super) struct Rebinds {
    live: AtomicUsize,
    buffers: Mutex<KeyMap<ResourceKey, Entry>>,
    /// How many of them are detached (`detach.rs`): else no command names a detached one.
    detached: AtomicUsize,
    /// Signaled as a submitted command buffer that took one of them completes
    /// (`SubmittedHolds`): a detach of it waits for none to hold it.
    completed: Condvar,
    /// The device's stand-in a detached buffer's commands take (`buffer::Detached`), while one
    /// is detached: made again once every one let go of it.
    pub(super) stand_in: Mutex<Option<WeakBuffer>>,
    /// The stand-ins the images of textures over detached bytes present, by plan
    /// (`texture/stand_in.rs`).
    pub(super) stand_ins: super::texture::stand_in::StandIns,
    /// The textures over detached bytes whose unnamed reads may not be Metal's, not yet
    /// announced (`Device::reach_stale_detached`).
    pub(super) stale: Mutex<Vec<StaleTexture>>,
    /// How many there are: the check at each submission costs one load while none is.
    pub(super) stale_count: AtomicUsize,
    /// The command buffers committed and not yet first submitted that an encoder of theirs can
    /// encode again (`replay.rs`), with the keys of the buffers over host pages those encoders
    /// named: a rebind of one of them has them encoded again before their first submission.
    /// Taken under `buffers`' lock, after it.
    waiting: Mutex<Waiting>,
}

/// The command buffers waiting for their first submission whose encoders a rebind encodes again
/// ([`Rebinds::wait_for_replay`]), each with the keys its replayable encoders named, sorted.
#[derive(Default)]
struct Waiting {
    buffers: Vec<(Weak<CommandBufferShared>, Box<[ResourceKey]>)>,
    /// The length past which the next registration first lets go of the entries whose command
    /// buffer went or was submitted.
    prune_at: usize,
}

/// A texture over detached bytes whose unnamed reads a submission's reach may not answer as
/// Metal's (`Rebinds::stale`, `Device::note_stale`): the texture (by type erased,
/// `buffer::NotedTexture`), its object's ID, the hold on the pages of the bytes, whose rebind
/// ends it, and why.
pub(super) struct StaleTexture {
    pub(super) texture: Weak<dyn Any + Send + Sync>,
    pub(super) id: ObjectId,
    pub(super) cell: Weak<HostCell>,
    pub(super) why: Unexact,
}

/// Why a texture over detached bytes is noted ([`StaleTexture`]), each announced by a kind of
/// its own.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum Unexact {
    /// Its bytes were detached after a submitted command buffer made their pages resident:
    /// Metal's unnamed reads read those pages (`Degradation::StaleDetachedReach`).
    StalePages,
    /// Its stand-in has a sink: a read through a storage-class element reads what an unnamed
    /// write left there, where Metal's reads zero (`Degradation::DetachedStorageKeepsWrites`).
    StorageSink,
}

/// One buffer over host pages.
struct Entry {
    cell: Weak<HostCell>,
    /// The holds of encoders and of command buffers not yet committed on it ([`Holds`]).
    pending: u32,
    /// The submitted command buffers not yet complete whose commands took it
    /// (`SubmittedHolds`): a detach waits for none, or refuses.
    submitted: u32,
    /// It is detached (`detach.rs`), as its cell says.
    detached: bool,
    /// A submitted command buffer took the pages it lies in now and found them its own: Metal
    /// made them resident, so a detach leaves their textures' unnamed reads reading those pages
    /// (`Device::reach_stale_detached`).
    resident: bool,
}

impl Rebinds {
    /// The host addresses of the bytes of the buffer of `key` over host pages
    /// (`HostCell::host_range`); `None` for any other key, and for a buffer gone or detached.
    pub(super) fn host_range(&self, key: ResourceKey) -> Option<std::ops::Range<usize>> {
        let cell = lock(&self.buffers).get(&key)?.cell.upgrade()?;
        cell.host_range()
    }

    /// Whether the device has a buffer over host pages: else no use names one.
    pub(super) fn any(&self) -> bool {
        self.live.load(Ordering::Relaxed) > 0
    }

    /// Note the buffer of `key` over host pages, `cell` its hold on them.
    pub(super) fn register(&self, key: ResourceKey, cell: &Arc<HostCell>) {
        let entry = Entry {
            cell: Arc::downgrade(cell),
            pending: 0,
            submitted: 0,
            detached: false,
            resident: false,
        };
        if lock(&self.buffers).insert(key, entry).is_none() {
            self.live.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// The buffer of `key` is gone.
    pub(super) fn forget(&self, key: ResourceKey) {
        if let Some(entry) = lock(&self.buffers).remove(&key) {
            self.live.fetch_sub(1, Ordering::Relaxed);
            if entry.detached {
                self.detached.fetch_sub(1, Ordering::Relaxed);
            }
        }
    }

    /// Hold `key` for an encoder's commands, which used it: the pages they took, which their
    /// command buffer retains until it completed, or `Pinned::Detached` while it is detached;
    /// `None` where `key` names no buffer over host pages. The hold refuses a rebind until
    /// [`Rebinds::release`].
    fn hold(&self, key: ResourceKey) -> Option<Pinned> {
        let mut buffers = lock(&self.buffers);
        let entry = buffers.get_mut(&key)?;
        let cell = entry.cell.upgrade()?;
        entry.pending += 1;
        Some(match cell.lying() {
            Lying::In(pages) => Pinned::Held(pages),
            Lying::Detached(_) => Pinned::Detached,
        })
    }

    /// What a command an encoder encodes again took of the buffer of `key` (`replay.rs`), as
    /// [`Rebinds::hold`] answers it, without the hold: its command buffer is committed, so
    /// nothing it encodes refuses a rebind.
    fn pin(&self, key: ResourceKey) -> Option<Pinned> {
        let buffers = lock(&self.buffers);
        let cell = buffers.get(&key)?.cell.upgrade()?;
        let pinned = match cell.lying() {
            Lying::In(pages) => Pinned::Held(pages),
            Lying::Detached(_) => Pinned::Detached,
        };
        Some(pinned)
    }

    /// Note `buffer`, committed now and not yet first submitted, whose encoders that can be
    /// encoded again named the buffers of `keys` (sorted): a rebind of one of them before its
    /// first submission has them encoded again ([`Rebinds::replace`]). Its first submission
    /// lets go of it ([`Submitting::submitted_now`]).
    pub(super) fn wait_for_replay(
        &self,
        buffer: &Arc<CommandBufferShared>,
        keys: Box<[ResourceKey]>,
    ) {
        let mut waiting = lock(&self.waiting);
        // A command buffer that went before its first submission (its submission failed
        // first) leaves its entry: let go of here, never upgraded, so none goes under the lock.
        if waiting.buffers.len() >= waiting.prune_at {
            waiting
                .buffers
                .retain(|(buffer, _)| buffer.strong_count() > 0);
            waiting.prune_at = (2 * waiting.buffers.len()).max(16);
        }
        waiting.buffers.push((Arc::downgrade(buffer), keys));
    }

    /// Whether one of `keys` names a detached buffer (`detach.rs`): at no cost while none is.
    pub(super) fn any_detached(&self, keys: impl IntoIterator<Item = ResourceKey>) -> bool {
        if self.detached.load(Ordering::Acquire) == 0 {
            return false;
        }
        let buffers = lock(&self.buffers);
        keys.into_iter()
            .any(|key| buffers.get(&key).is_some_and(|entry| entry.detached))
    }

    /// Detach the buffer of `key`, whose hold on its pages is `cell`, to `detached`, unless a
    /// submitted command buffer not yet complete took it (`SubmittedHolds`): where it lay, and
    /// whether a submitted command buffer made those pages resident, or `None` where such a command
    /// buffer refuses it. Already detached, it stays so.
    pub(super) fn detach(
        &self,
        key: ResourceKey,
        cell: &HostCell,
        detached: Arc<Detached>,
    ) -> Option<(Lying, bool)> {
        let mut buffers = lock(&self.buffers);
        let entry = buffers.get_mut(&key)?;
        if entry.submitted > 0 {
            return None;
        }
        if !entry.detached {
            entry.detached = true;
            self.detached.fetch_add(1, Ordering::AcqRel);
        }
        let resident = entry.resident;
        entry.resident = false;
        Some((cell.replace(Lying::Detached(detached)), resident))
    }

    /// Wait until no submitted command buffer not yet complete holds the buffer of `key`
    /// (`SubmittedHolds`), with no other lock of the runtime's held: the completion path lets
    /// go of each without this one's caller. Settles on a lost device (`Err(DeviceLost)`),
    /// whose command buffers may never complete.
    pub(super) fn wait_unsubmitted(&self, device: &Device, key: ResourceKey) -> Result<(), Error> {
        let mut buffers = lock(&self.buffers);
        while buffers.get(&key).is_some_and(|entry| entry.submitted > 0) {
            if device.is_lost() {
                return Err(Error::DeviceLost);
            }
            buffers = match self.completed.wait_timeout(buffers, LOST_POLL) {
                Ok((buffers, _)) => buffers,
                Err(poisoned) => poisoned.into_inner().0,
            };
        }
        Ok(())
    }

    /// Whether a submitted command buffer not yet complete holds the buffer of `key`.
    pub(super) fn submitted(&self, key: ResourceKey) -> bool {
        lock(&self.buffers)
            .get(&key)
            .is_some_and(|entry| entry.submitted > 0)
    }

    /// The buffers over host pages as a command buffer's batch is submitted, locked: what its
    /// commands took is judged and held at once (`Submitting`), so no detach falls between.
    pub(super) fn submitting(&self) -> Submitting<'_> {
        Submitting {
            buffers: lock(&self.buffers),
        }
    }

    /// Submitted command buffers that took `keys` completed: a detach may go on.
    fn completed(&self, keys: &[ResourceKey]) {
        let mut buffers = lock(&self.buffers);
        for key in keys {
            if let Some(entry) = buffers.get_mut(key) {
                entry.submitted = entry.submitted.saturating_sub(1);
            }
        }
        drop(buffers);
        self.completed.notify_all();
    }

    /// Let go of one hold of `key`.
    fn release(&self, key: ResourceKey) {
        if let Some(entry) = lock(&self.buffers).get_mut(&key) {
            entry.pending = entry.pending.saturating_sub(1);
        }
    }

    /// Whether a hold refuses a rebind of `key`.
    fn pending(&self, key: ResourceKey) -> bool {
        lock(&self.buffers)
            .get(&key)
            .is_some_and(|entry| entry.pending > 0)
    }

    /// Give the buffer of `key`, whose hold on its pages is `cell`, `pages` from now on, unless
    /// a hold refuses it: where it lay (a detached one is attached again), and the command
    /// buffers waiting for their first submission whose replayable encoders named it
    /// ([`Rebinds::wait_for_replay`]), each marked under this lock to encode them again before
    /// that submission (`CommandBufferShared::reencode`), which the caller runs; `Err` gives
    /// `pages` back.
    pub(super) fn replace(
        &self,
        key: ResourceKey,
        cell: &HostCell,
        pages: Arc<HostPages>,
    ) -> Result<(Lying, Vec<Arc<CommandBufferShared>>), Arc<HostPages>> {
        let mut buffers = lock(&self.buffers);
        let Some(entry) = buffers.get_mut(&key) else {
            return Ok((cell.replace(Lying::In(pages)), Vec::new()));
        };
        if entry.pending > 0 {
            return Err(pages);
        }
        if entry.detached {
            entry.detached = false;
            self.detached.fetch_sub(1, Ordering::AcqRel);
        }
        // No command buffer made the new pages resident yet.
        entry.resident = false;
        let lying = cell.replace(Lying::In(pages));
        // Marked before this lock goes, so a first submission either came before the rebind
        // (and keeps the old pages) or waits for the encoders encoded again.
        // Each upgraded one goes to the caller, which lets go of it without this lock.
        let mut again = Vec::new();
        let waiting = lock(&self.waiting);
        for (buffer, keys) in &waiting.buffers {
            if keys.binary_search(&key).is_err() {
                continue;
            }
            if let Some(buffer) = buffer.upgrade() {
                buffer.mark_reencode();
                again.push(buffer);
            }
        }
        Ok((lying, again))
    }
}

/// How often a detach waiting for submitted command buffers looks whether the device was lost.
const LOST_POLL: std::time::Duration = std::time::Duration::from_millis(20);

/// The buffers over host pages, locked as a command buffer's batch is submitted
/// (`Rebinds::submitting`): each `Taken` is judged against where its buffers lie now and its
/// pages held from now on under the one lock, so a detach either came before (and fails the
/// submission) or waits for the command buffer.
pub(super) struct Submitting<'a> {
    buffers: MutexGuard<'a, KeyMap<ResourceKey, Entry>>,
}

/// What a submission found of what an encoder's commands, or a segment's, took.
#[derive(Clone, Copy, Default)]
pub(super) struct Found {
    /// A buffer lies elsewhere than a command took it: rebound, detached, attached again, or
    /// gone.
    pub(super) moved: bool,
    /// Such a buffer a command named: the submission fails `EncodedUseRebound`, unless one is
    /// detached.
    pub(super) rebound: bool,
    /// A buffer a command took is detached now: the submission fails `DetachedBacking`.
    pub(super) detached: bool,
}

impl Found {
    /// The failure it gives the submission holding it, if any.
    pub(super) fn failure(&self) -> Option<crate::error::CommandBufferError> {
        use crate::error::CommandBufferError;
        match (self.detached, self.rebound) {
            (true, _) => Some(CommandBufferError::DetachedBacking),
            (false, true) => Some(CommandBufferError::EncodedUseRebound),
            (false, false) => None,
        }
    }
}

impl Submitting<'_> {
    /// Whether a buffer a command of `taken` named lies elsewhere than the command took it, not
    /// detached: what [`Submitting::submit`] finds `rebound`, judged without taking anything.
    pub(super) fn rebound(&self, taken: &Taken) -> bool {
        taken.pins.iter().any(|(key, pinned)| {
            if !taken.named.contains(key) {
                return false;
            }
            let entry = self.buffers.get(key);
            let cell = entry.and_then(|entry| entry.cell.upgrade());
            match (pinned, cell.as_ref().map(|cell| cell.lying())) {
                (_, None) | (_, Some(Lying::Detached(_))) => false,
                (Pinned::Detached, Some(Lying::In(_))) => true,
                (Pinned::Held(pages), Some(Lying::In(current))) => !Arc::ptr_eq(pages, &current),
                (Pinned::Waiting(pages), Some(Lying::In(current))) => {
                    !std::ptr::eq(Weak::as_ptr(pages), Arc::as_ptr(&current))
                }
            }
        })
    }

    /// `buffer` is first submitted now (or holds back for its encoders encoded again): it waits
    /// for no rebind's replay from here ([`Rebinds::wait_for_replay`]).
    pub(super) fn submitted_now(&self, rebinds: &Rebinds, buffer: &CommandBufferShared) {
        let mut waiting = lock(&rebinds.waiting);
        waiting
            .buffers
            .retain(|(waiting, _)| !std::ptr::eq(Weak::as_ptr(waiting), buffer));
    }

    /// Judge `taken`, and hold every pin whose pages still live in `retained` until the command
    /// buffer completes, counted in `holds` for a detach to wait for. Its named keys and
    /// resolved ranges stay for the caller. (Every key was named where the command buffer's root
    /// tables searched their own sets, past the search's capacity, which a set of any count no
    /// longer meets.)
    pub(super) fn submit(
        &mut self,
        taken: &mut Taken,
        retained: &mut Vec<Box<dyn Any + Send + Sync>>,
        holds: &mut SubmittedHolds,
    ) -> Found {
        let mut found = Found::default();
        for (key, pinned) in taken.pins.drain(..) {
            let entry = self.buffers.get_mut(&key);
            let cell = entry.as_ref().and_then(|entry| entry.cell.upgrade());
            let now = cell.as_ref().map(|cell| cell.lying());
            let detached = matches!(now, Some(Lying::Detached(_)));
            // Compared by identity, so a pin whose pages went still compares; neither is the last
            // hold: the cell holds what it lies in, and the buffer its cell.
            let moved = match (&pinned, &now) {
                (_, None) => true,
                (Pinned::Detached, Some(Lying::Detached(_))) => false,
                (Pinned::Detached, Some(Lying::In(_))) => true,
                (Pinned::Held(_) | Pinned::Waiting(_), Some(Lying::Detached(_))) => true,
                (Pinned::Held(pages), Some(Lying::In(current))) => !Arc::ptr_eq(pages, current),
                (Pinned::Waiting(pages), Some(Lying::In(current))) => {
                    !std::ptr::eq(Weak::as_ptr(pages), Arc::as_ptr(current))
                }
            };
            found.detached |= detached;
            found.moved |= moved || detached;
            found.rebound |= moved && !detached && taken.named.contains(&key);
            let pages = match pinned {
                Pinned::Held(pages) => Some(pages),
                Pinned::Waiting(pages) => pages.upgrade(),
                Pinned::Detached => None,
            };
            if let (Some(pages), Some(entry)) = (pages, entry) {
                // Pages the command buffer finds its own, Metal makes resident.
                entry.resident |= !moved && !detached;
                entry.submitted += 1;
                holds.keys.push(key);
                retained.push(Box::new(pages));
            }
        }
        // A blit's objects of a releasable import, held until it completes. Gone only with their
        // import, as a rebind or a detach replaced its buffer's pages, which a pin above found
        // moved: the recording that named them runs nothing.
        for hold in taken.imports.drain(..) {
            match hold.retained() {
                Some(held) => retained.push(held),
                None => found.moved = true,
            }
        }
        // `cell` and `now` drop here, under the lock: never the last hold of anything (above).
        found
    }
}

/// The holds a submitted command buffer's commands took on buffers over host pages
/// (`Submitting::submit`), counted per buffer until it completes: dropped with what it
/// retained, it lets a detach waiting for them go on.
pub(super) struct SubmittedHolds {
    device: Device,
    keys: Vec<ResourceKey>,
}

impl SubmittedHolds {
    /// None yet, on `device`.
    pub(super) fn new(device: &Device) -> SubmittedHolds {
        SubmittedHolds {
            device: device.clone(),
            keys: Vec::new(),
        }
    }

    /// Whether it counts none.
    pub(super) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

impl Drop for SubmittedHolds {
    fn drop(&mut self) {
        if !self.keys.is_empty() {
            self.device.shared().rebinds.completed(&self.keys);
        }
    }
}

/// The holds an encoder's commands, or a command buffer's ended encoders', took on buffers over
/// host pages ([`Rebinds::hold`]): let go when the command buffer commits (or completes, where
/// commands are recorded later), or goes uncommitted.
pub(super) struct Holds {
    device: Device,
    keys: Vec<ResourceKey>,
}

impl Holds {
    /// No holds yet, on `device`.
    pub(super) fn new(device: &Device) -> Holds {
        Holds {
            device: device.clone(),
            keys: Vec::new(),
        }
    }

    /// Hold `key` where it names a buffer over host pages: the pages the commands took, to
    /// retain until the command buffer completes (none while it is detached).
    pub(super) fn hold(&mut self, key: ResourceKey) -> Option<Pinned> {
        let pinned = self.device.shared().rebinds.hold(key)?;
        self.keys.push(key);
        Some(pinned)
    }

    /// Whether it holds nothing.
    pub(super) fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    /// What a command an encoder encodes again took of `key` ([`Rebinds::pin`]), held by
    /// nothing here.
    pub(super) fn pin(&self, key: ResourceKey) -> Option<Pinned> {
        self.device.shared().rebinds.pin(key)
    }
}

/// The pages a command took of a buffer over host pages (`Holds::hold`): held while it is
/// encoded and from its command buffer's first submission until it completes, and only weakly
/// from the commit to that submission, so a rebind or a release meanwhile lets the pages go
/// where nothing submitted holds them. The submission finds them moved (`Taken::moved`).
pub(super) enum Pinned {
    Held(Arc<HostPages>),
    Waiting(Weak<HostPages>),
    /// Taken while the buffer was detached (`detach.rs`): no pages, and the encoder recorded
    /// nothing from that command on (`EncoderCore::skips`).
    Detached,
}

/// What an encoder's commands, or a segment's, took of buffers over host pages: the pages of each,
/// by its key, and the keys a command named. A named key's commands were resolved as they were
/// encoded: a binding's descriptor or its address in a root table's statics, a blit's operand, an
/// attachment, a vertex, index or indirect buffer, a texture over the pages (its views), an
/// indirect command's resolved buffer. A key no command named is reached through the resident set
/// alone (an encoder's `use_resource`, `use_heap`, a bound buffer no command reads), whose ranges
/// the command buffer resolves again as it is first submitted.
///
/// A GPU-alone replay of an indirect command buffer (`generated_execute.rs`) resolves its
/// resident set as the commit records it, into its scratch: its ranges (`resolved`) hold as the
/// command buffer is first submitted only where the live buffers and what its encoders made
/// resident still lie there.
#[derive(Default)]
pub(super) struct Taken {
    pins: Vec<(ResourceKey, Pinned)>,
    /// The objects of releasable imports its blits took ([`ImportHold`]).
    imports: Vec<ImportHold>,
    named: BTreeSet<ResourceKey>,
    resolved: Vec<ResidentRange>,
    /// A command named a detached buffer, and the encoder recorded nothing from it on
    /// (`EncoderCore::skips`): its submission never runs.
    dead: bool,
}

/// The objects of a releasable import a blit's command took (`Buffer::blit_at`,
/// `host_residency.rs`): held while its encoder records, weakly from the commit until the command
/// buffer's first submission, as its pages are, then until it completes. Held weakly, they keep
/// the device's residency from releasing them (the committed recording names their `VkBuffer`),
/// but not the import from going with its buffer's pages (a rebind, a detach), as Metal holds
/// nothing of a command buffer not yet submitted: their free, and the deallocator after it, run
/// in that call, and the recording that named them never runs (its pin of the pages moved).
pub(super) enum ImportHold {
    #[cfg(unix)]
    Held(Arc<Imported>),
    #[cfg(unix)]
    Waiting(Weak<Imported>),
}

impl ImportHold {
    /// Held weakly from now on.
    fn wait(&mut self) {
        match self {
            #[cfg(unix)]
            ImportHold::Held(held) => *self = ImportHold::Waiting(Arc::downgrade(held)),
            #[cfg(unix)]
            ImportHold::Waiting(_) => {}
        }
    }

    /// The objects, held until the command buffer completes: `None` where they went.
    fn retained(self) -> Option<Box<dyn Any + Send + Sync>> {
        match self {
            #[cfg(unix)]
            ImportHold::Held(held) => Some(Box::new(held)),
            #[cfg(unix)]
            ImportHold::Waiting(held) => held
                .upgrade()
                .map(|held| Box::new(held) as Box<dyn Any + Send + Sync>),
        }
    }
}

impl Taken {
    /// Note `hold`, the objects of a releasable import a blit took.
    pub(super) fn hold_import(&mut self, hold: ImportHold) {
        self.imports.push(hold);
    }

    /// Note `pinned`, what a command took of the buffer of `key`.
    pub(super) fn pin(&mut self, key: ResourceKey, pinned: Pinned) {
        self.pins.push((key, pinned));
    }

    /// Note that a command named `key`.
    pub(super) fn name(&mut self, key: ResourceKey) {
        self.named.insert(key);
    }

    /// Whether a command named a buffer over host pages ([`named_pins`](Self::named_pins) holds
    /// one).
    pub(super) fn names_a_pin(&self) -> bool {
        self.pins.iter().any(|(key, _)| self.named.contains(key))
    }

    /// The keys of the buffers over host pages a command named, sorted, each once.
    pub(super) fn named_pins(&self) -> Vec<ResourceKey> {
        let mut keys: Vec<ResourceKey> = self
            .pins
            .iter()
            .map(|(key, _)| *key)
            .filter(|key| self.named.contains(key))
            .collect();
        keys.sort_unstable();
        keys.dedup();
        keys
    }

    /// Note `ranges`, a resident set its commands resolved as they were recorded.
    pub(super) fn resolve(&mut self, ranges: impl IntoIterator<Item = ResidentRange>) {
        self.resolved.extend(ranges);
    }

    /// The resident ranges its commands resolved as they were recorded.
    pub(super) fn resolved(&self) -> &[ResidentRange] {
        &self.resolved
    }

    /// Join `other`'s pins, names and resolved ranges.
    pub(super) fn join(&mut self, other: Taken) {
        self.pins.extend(other.pins);
        self.named.extend(other.named);
        self.resolved.extend(other.resolved);
        self.dead |= other.dead;
    }

    /// The encoder records nothing more: a command named a detached buffer.
    pub(super) fn die(&mut self) {
        self.dead = true;
    }

    /// Whether the encoder recorded nothing from some command on (`Taken::die`).
    pub(super) fn dead(&self) -> bool {
        self.dead
    }

    /// Hold every pin weakly: its command buffer committed and waits for its first submission.
    /// The pages it held go to `held`, which its deferred command buffers' replays hold until
    /// they are replayed (`recorder`).
    pub(super) fn wait(&mut self, held: &mut Vec<Arc<HostPages>>) {
        for (_, pinned) in &mut self.pins {
            if let Pinned::Held(pages) = pinned {
                let waiting = Pinned::Waiting(Arc::downgrade(pages));
                if let Pinned::Held(pages) = std::mem::replace(pinned, waiting) {
                    held.push(pages);
                }
            }
        }
        for hold in &mut self.imports {
            hold.wait();
        }
    }

    /// Its named keys and resolved ranges let go, once judged (`Submitting::submit`).
    pub(super) fn judged(&mut self) {
        self.named.clear();
        self.resolved.clear();
    }

    /// Hand every pin's pages that still live to `retained`, upgraded, which the command buffer
    /// holds until it completes and releases without its lock: as it is first submitted, and as
    /// a segment recorded later is placed or let go. Nothing is let go here but weak pins, so no
    /// pages go (the caller holds locks).
    pub(super) fn retain_into(&mut self, retained: &mut Vec<Box<dyn Any + Send + Sync>>) {
        let live = self.pins.drain(..).filter_map(|(_, pinned)| match pinned {
            Pinned::Held(pages) => Some(pages),
            Pinned::Waiting(pages) => pages.upgrade(),
            Pinned::Detached => None,
        });
        retained.extend(live.map(|pages| Box::new(pages) as Box<dyn Any + Send + Sync>));
        retained.extend(self.imports.drain(..).filter_map(ImportHold::retained));
        self.named.clear();
        self.resolved.clear();
    }
}

impl Drop for Holds {
    fn drop(&mut self) {
        let rebinds = &self.device.shared().rebinds;
        for &key in &self.keys {
            rebinds.release(key);
        }
    }
}

impl Buffer {
    /// Give the no-copy buffer other host memory of its length from now on (a runtime extension;
    /// the module docs of `rebind.rs` state the rule): `memory` ([`HostMemory`]), under its
    /// constructor's contract for the buffer's [`length`](Buffer::length), imported as
    /// [`Device::new_buffer_with_bytes_no_copy`] imports memory. The buffer, its `gpu_address` and
    /// its options stay; `contents` answers `memory.pointer()` from now on. Command buffers first
    /// submitted before the call keep the old memory until they complete; one committed before it
    /// and first submitted after reaches the new memory through what it made resident and what its
    /// commands reach unnamed, and through what its compute, blit and render encoders named, which
    /// this call encodes again; it fails `CommandBufferError::EncodedUseRebound` where an encoder
    /// this call cannot encode again named the buffer (one of a command buffer that used command
    /// predication, among others); commands encoded after it use the new. The old memory's
    /// deallocator runs once every command buffer submitted before the call that used it completed
    /// (in the call, where none holds it); `deallocator` runs for the new memory at the next
    /// rebind's turn, or when the buffer goes. A refused call runs neither and changes nothing.
    /// `keep_alive` ([`KeepAlive`]) is dropped once the device's import of the new memory is freed:
    /// with one, `deallocator` runs at that turn though something still holds the import (a command
    /// buffer submitted before a texture over the bytes was released); with none, `deallocator`
    /// waits for the import (`host-deallocator-delayed`).
    ///
    /// A texture over the buffer (a shadow of its bytes, or a texture buffer) reads the new
    /// memory from its next use, and one whose linear image aliases its bytes has its image made
    /// again over the new memory before the call returns; nothing of the old contents is copied.
    ///
    /// Refused: a buffer that is not a no-copy buffer of its own (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a command buffer not yet committed whose commands used
    /// the buffer, or an open encoder that did (a render encoder with an attachment over it, from
    /// its open) (`Invalid`, [`RebindProblem::EncodedUses`]); new memory at another offset in its
    /// page than the old where a texture over the buffer lives (`NotImplemented(RebindTextures)`:
    /// the texture's copies, views and aliasing image are planned at that offset), or where the
    /// device cannot make a texture's aliasing image again over the new memory
    /// (`NotImplemented(RebindTextures)`, which says when); memory the device cannot import
    /// exactly, as `new_buffer_with_bytes_no_copy` refuses it; a lost device.
    ///
    /// # Safety
    ///
    /// `memory` meets the contract of the [`HostMemory`] constructor that made it
    /// ([`HostMemory::new`], [`HostMemory::with_dma_buf`]) for the buffer's length. The old memory
    /// stays mapped until its keep-alive is dropped, else until its deallocator runs, or with
    /// neither until every command buffer first submitted before the call completed. No command
    /// that uses the buffer is encoded, and no command buffer that uses it committed, on another
    /// thread during the call.
    pub unsafe fn rebind_bytes_no_copy(
        &self,
        memory: HostMemory,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(), Error> {
        const CALL: &str = "rebind_bytes_no_copy";
        let Some(cell) = self.host_cell() else {
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        // SAFETY: the caller's contract.
        let rebound = unsafe {
            self.device().rebind_host_pages(
                CALL,
                self,
                cell,
                memory,
                HostRelease::new(deallocator, keep_alive),
                None,
            )
        };
        // The buffer a pending command buffer may hold has the program's objects now.
        program_objects::settle();
        rebound
    }
}

impl Heap {
    /// Give the heap over the caller's host memory (`Device::new_heap_with_bytes_no_copy`) other
    /// host memory of its size from now on (a runtime extension), as
    /// [`Buffer::rebind_bytes_no_copy`] gives a no-copy buffer: `memory` ([`HostMemory`]), its
    /// first byte at a multiple of the import's unit as the heap's was. Every buffer and texture
    /// placed in the heap stays, at its offset, with its `gpu_address` and resource ID; a buffer's
    /// `contents` answers its offset from `memory.pointer()` from now on. A texture bound to the
    /// heap's bytes has its image made again over the new memory, at its offset, before the call
    /// returns, as has a texture over one of the heap's buffers whose linear image aliases its
    /// bytes; a texture held as a shadow of the heap's bytes, a texture buffer and a shadowed
    /// texture over one of the heap's buffers read them from their next use. Nothing of the old
    /// contents is copied.
    ///
    /// Refused: a heap not over the caller's host memory (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a command buffer not yet committed whose commands used
    /// the heap or a resource placed in it, or an open encoder that did (a render encoder with an
    /// attachment in it, from its open) (`Invalid`, [`RebindProblem::EncodedUses`]); memory whose
    /// first byte is off the import's unit (`Invalid`, `HeapProblem::HostMemoryAlignment`); memory the device cannot import exactly;
    /// an image the device cannot make again there (as `new_texture_at_offset` refuses it); a
    /// lost device.
    ///
    /// # Safety
    ///
    /// As [`Buffer::rebind_bytes_no_copy`]'s, for the heap's [`size`](Heap::size) and the uses
    /// of the heap and its resources.
    pub unsafe fn rebind_bytes_no_copy(
        &self,
        memory: HostMemory,
        deallocator: Option<Deallocator>,
        keep_alive: Option<KeepAlive>,
    ) -> Result<(), Error> {
        const CALL: &str = "rebind_bytes_no_copy";
        let Some((own, cell)) = self
            .bytes()
            .and_then(|own| own.host_cell().map(|cell| (own, cell)))
        else {
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        // SAFETY: the caller's contract.
        let rebound = unsafe {
            self.device().rebind_host_pages(
                CALL,
                own,
                cell,
                memory,
                HostRelease::new(deallocator, keep_alive),
                Some(self),
            )
        };
        // The heap a pending command buffer may hold has the program's objects now.
        program_objects::settle();
        rebound
    }
}

impl Device {
    /// Whether a hold refuses a rebind of the buffer of `key` (`Rebinds::pending`).
    pub(super) fn rebind_pending(&self, key: ResourceKey) -> bool {
        self.shared().rebinds.pending(key)
    }
}
