//! Detaching a no-copy buffer, or a heap over the caller's host memory, from its host pages
//! (a runtime extension in the shape of Metal's `detachBacking`, the companion of
//! `rebind_bytes_no_copy`, `rebind.rs`).
//!
//! A consumer that moves its pages lets go of the old ones before it has the new: the buffer or
//! heap lies in no pages until a rebind gives it some. Metal's answers (the consumer's rebind
//! probe on the M3 Max, tables 1 to 3), which the runtime keeps:
//!
//! - **The call.** It detaches at once, and answers `true`, unless a command buffer submitted
//!   and not yet complete took the buffer or heap: through a command that named it (one of a
//!   heap's buffers included), or through what it made resident (`use_resource`, `use_heap`).
//!   Then a buffer's call blocks until each such command buffer completed and answers `false`,
//!   detaching nothing; a heap's answers `false` at once. A lost device ends the wait
//!   (`Err(DeviceLost)`). The wait holds no lock of the runtime's: a completion turn lets go
//!   of each command buffer's holds without one (`rebind::SubmittedHolds`).
//! - **Pending work.** A command buffer whose first submission comes after the detach and whose
//!   commands named the buffer or made it resident (a render pass whose attachment lies in the
//!   bytes names them from its open), encoded before the detach or after it, fails at that
//!   submission with `CommandBufferError::DetachedBacking` (Metal's code 8) and runs nothing of it;
//!   one that only reaches the bytes unnamed reads zeros there (`live_buffers.rs` no longer lists
//!   them) and runs. A rebind before that submission gives a command encoded while detached
//!   `EncodedUseRebound` instead, the gap until encoded uses resolve at submission.
//! - **No memory.** A detached buffer takes the device's stand-in, a small buffer of the
//!   runtime's own shared by every detached buffer (`buffer::Detached`), never memory of its
//!   size. No command that takes the stand-in runs: an encoder whose command names a detached
//!   buffer records nothing from that command on (`EncoderCore::skips`), and its submission
//!   fails as above; `contents` answers `None` while the buffer is detached.
//! - **Textures**. A texture over the bytes (over the buffer, or over one of a heap's buffers, or
//!   in the heap) stays, and presents a stand-in of its image's plan while they are detached
//!   (`texture/stand_in.rs`): its size and level queries answer its own, an unnamed read reads zero
//!   bytes and an unnamed write lands where nothing reads. Its images bound to the old pages and
//!   its views go with them, once nothing holds the pages: an open encoder that named them (a
//!   render pass over them from its open, whose first instance may have recorded with those
//!   images, `render/inline_first.rs`) and a submitted command buffer hold them. A command that
//!   names it fails as one naming the buffer does. A texture being made over the bytes refuses
//!   the detach (`NotImplemented(DetachTextures)`), and no texture is made over detached bytes
//!   yet.
//! - **Attaching again.** `rebind_bytes_no_copy` gives a detached buffer or heap pages again.

use super::buffer::{Buffer, Detached, HostCell, Lying};
use super::degradation::Degradation;
use super::device::lock;
use super::error::{invalid, Error, Problem, RuntimeGap};
use super::heap::Heap;
use super::rebind::{RebindProblem, StaleTexture, Unexact};
use super::resource::ObjectId;
use super::texture::stand_in::TextureImages;
use super::Device;
use std::any::Any;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Weak};

/// The bytes of the device's stand-in (`Device::detach_stand_in`): room for the widest element
/// a command could take, which no command that takes it reads.
const STAND_IN_BYTES: u64 = 256;

impl Buffer {
    /// `detachBacking` (a runtime extension; the module docs of `detach.rs` state the rule): let
    /// the no-copy buffer lie in no host pages until
    /// [`rebind_bytes_no_copy`](Buffer::rebind_bytes_no_copy) gives it some. `true` once it is
    /// detached (at once, or it was already). Where a command buffer submitted and not yet
    /// complete took the buffer, the call blocks until each such command buffer completed and
    /// answers `false`, detaching nothing. The old memory's deallocator runs once every command
    /// buffer submitted before the call that used it completed, and every open encoder that named
    /// it was committed or dropped (in the call, where none holds it). The buffer, its
    /// `gpu_address` and its options stay; `contents` answers `None` while it is detached.
    ///
    /// A command buffer first submitted after the call whose commands named the buffer or made
    /// it resident (a render pass whose attachment lies over it names it from its open) fails
    /// there, `CommandBufferError::DetachedBacking`, and runs nothing of it; one that reaches the
    /// buffer only by its address reads zeros there.
    ///
    /// A texture over the buffer's bytes stays, and presents a stand-in until the buffer lies in
    /// pages again: its queries answer its own descriptor, an unnamed read reads zeros, and a
    /// command that names it fails as one that names the buffer.
    ///
    /// Refused: a buffer that is not a no-copy buffer of its own (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a buffer with a texture being made over its bytes
    /// (`NotImplemented(DetachTextures)`); a texture over it whose stand-in the device makes none
    /// of (`NotImplemented(DetachTextures)`, a multi-planar format), or out of memory for one; a
    /// lost device, also while the call waits.
    pub fn detach_backing(&self) -> Result<bool, Error> {
        const CALL: &str = "detach_backing";
        let Some(cell) = self.host_cell() else {
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        self.device().detach_host_pages(self, cell, None)
    }
}

impl Heap {
    /// `detachBacking` of a heap over the caller's host memory
    /// (`Device::new_heap_with_bytes_no_copy`; a runtime extension), as [`Buffer::detach_backing`]
    /// detaches a no-copy buffer: every buffer placed in it stays, at its offset, and lies in no
    /// pages until [`rebind_bytes_no_copy`](Heap::rebind_bytes_no_copy) gives the heap some. Where
    /// a command buffer submitted and not yet complete took the heap or a buffer placed in it, the
    /// call answers `false` at once, detaching nothing.
    ///
    /// Its textures, and those over its buffers, stay and present stand-ins as a buffer's do.
    ///
    /// Refused: a heap not over the caller's host memory (`Invalid`,
    /// [`RebindProblem::NotHostMemory`]); a heap with a texture being made in it, or over one of
    /// its buffers, and a texture whose stand-in the device makes none of
    /// (`NotImplemented(DetachTextures)`), or out of memory for one; a lost device.
    pub fn detach_backing(&self) -> Result<bool, Error> {
        const CALL: &str = "detach_backing";
        let Some((own, cell)) = self
            .bytes()
            .and_then(|own| own.host_cell().map(|cell| (own, cell)))
        else {
            return Err(invalid(CALL, Problem::Rebind(RebindProblem::NotHostMemory)));
        };
        self.device().detach_host_pages(own, cell, Some(self))
    }
}

impl Device {
    /// Detach `buffer`, a no-copy buffer whose hold on its pages is `cell` (`heap`'s own buffer
    /// where given), as the module docs say: `true` once detached; `false` where a command
    /// buffer submitted and not yet complete took it, after waiting for each such command buffer
    /// to complete unless it is a heap's.
    fn detach_host_pages(
        &self,
        buffer: &Buffer,
        cell: &Arc<HostCell>,
        heap: Option<&Heap>,
    ) -> Result<bool, Error> {
        if self.is_lost() {
            return Err(Error::DeviceLost);
        }
        let shared = self.shared();
        let rebinds = &shared.rebinds;
        let key = buffer.identity().key;
        // A submitted command buffer took it: a buffer's call waits for each, a heap's not.
        let refuse = || {
            if heap.is_none() {
                rebinds.wait_unsubmitted(self, key)?;
            }
            Ok(false)
        };
        if rebinds.submitted(key) {
            return refuse();
        }
        let Lying::In(from) = cell.lying() else {
            return Ok(true);
        };
        // No texture is being made over the bytes while this is held, and those over them
        // present stand-ins once they are detached, each made now, before anything changes.
        let no_textures = || Error::NotImplemented(RuntimeGap::DetachTextures);
        let textures = cell.settled().ok_or_else(no_textures)?;
        let images = TextureImages::of(textures.images());
        let stand_ins = images.stand_ins(self)?;
        let detached = Arc::new(Detached::new(self.detach_stand_in()?, &from)?);
        drop(from);
        let Some((old, resident)) = rebinds.detach(key, cell, detached) else {
            drop(textures);
            return refuse();
        };
        // What the textures over the old pages made of them goes: the images bound there and
        // every view (the texel views of texture buffers too, made again over pages the next
        // bind finds).
        let gone = images.present_stand_ins(stand_ins);
        let texels = cell.texels();
        // Pages a submitted command buffer made resident stay Metal's textures' unnamed reads;
        // and a stand-in's sink holds what unnamed writes left.
        let stale = textures.textures().flat_map(|noted| {
            let pages = resident.then_some(Unexact::StalePages);
            let sink = noted.sink().then_some(Unexact::StorageSink);
            let (texture, id) = (noted.texture(), noted.id);
            pages
                .into_iter()
                .chain(sink)
                .map(move |why| (Weak::clone(&texture), id, why))
        });
        self.note_stale(cell, stale);
        drop(textures);
        if let Lying::In(pages) = &old {
            for views in &texels {
                pages.park(views.park());
            }
        }
        // The buffer, or the heap's buffers, are reached unnamed again only once a command buffer
        // uses it after it lies in pages again (`live_buffers.rs`): until then, unnamed reads
        // read zeros.
        shared.live_buffers.rebound(buffer, heap);
        // Later buffers over the memory it was made over no longer join it.
        shared.aliases.leave(key);
        // It lies over no host bytes: its uses take its stand-in.
        shared.aliases.unplace(key);
        // The device's heap sets view the stand-ins of the textures over the bytes, and no other
        // texture's elements change (`heap_sets::renew_over`); the images and views they named
        // before wait for the command buffers that may read them through the sets, and the old
        // pages, which the images bound there read, wait with them.
        if !images.is_empty() || !texels.is_empty() {
            super::heap_sets::renew_over(self, key);
        }
        // What lay in the old pages goes with them, as a rebind's swapped images do: once the
        // command buffers that may read them through the heap sets completed, and once nothing
        // holds the pages: an open encoder that took them (a render pass's attachments from its
        // open, whose first instance may have recorded with their images,
        // `render/inline_first.rs`), and a submitted command buffer until it completes.
        if !gone.is_empty() {
            match &old {
                Lying::In(pages) => {
                    pages.park(gone);
                    super::heap_sets::retire(self, Box::new(Arc::clone(pages)));
                }
                Lying::Detached(_) => super::heap_sets::retire(self, Box::new(gone)),
            }
        }
        // The old pages go now where no command buffer holds them, their deallocator with them.
        drop(old);
        Ok(true)
    }

    /// The device's stand-in, which detached buffers take (`buffer::Detached`): made once
    /// while one is detached, and again once every one let go of it.
    pub(super) fn detach_stand_in(&self) -> Result<Buffer, Error> {
        let mut stand_in = lock(&self.shared().rebinds.stand_in);
        if let Some(buffer) = stand_in.as_ref().and_then(|weak| weak.upgrade()) {
            return Ok(buffer);
        }
        let buffer = self.scratch_buffer(STAND_IN_BYTES)?;
        *stand_in = Some(buffer.downgrade());
        Ok(buffer)
    }
}

impl Device {
    /// Note `textures` (by type erased, with their objects' IDs), over the bytes `cell` holds the
    /// pages of, detached, whose unnamed reads may not be Metal's, each for why (`Unexact`):
    /// detached after a submitted command buffer made their pages resident, or presenting a
    /// stand-in with a sink. Each is announced at the first submission, or commit, that may
    /// reach it (`Device::reach_stale_detached`, `Device::reach_storage_sinks`).
    pub(super) fn note_stale(
        &self,
        cell: &Arc<HostCell>,
        textures: impl Iterator<Item = (Weak<dyn Any + Send + Sync>, ObjectId, Unexact)>,
    ) {
        let rebinds = &self.shared().rebinds;
        let mut stale = lock(&rebinds.stale);
        stale.retain(|held| held.texture.strong_count() > 0);
        stale.extend(textures.map(|(texture, id, why)| StaleTexture {
            texture,
            id,
            cell: Arc::downgrade(cell),
            why,
        }));
        rebinds.stale_count.store(stale.len(), Ordering::Release);
    }

    /// The bytes `cell` holds the pages of lie in pages again (`buffer_descriptor.rs`): their
    /// textures are no longer over detached bytes.
    pub(super) fn forget_stale(&self, cell: &HostCell) {
        let rebinds = &self.shared().rebinds;
        if rebinds.stale_count.load(Ordering::Acquire) == 0 {
            return;
        }
        let mut stale = lock(&rebinds.stale);
        stale.retain(|held| {
            !std::ptr::eq(held.cell.as_ptr(), cell) && held.texture.strong_count() > 0
        });
        rebinds.stale_count.store(stale.len(), Ordering::Release);
    }

    /// The stale-reach degrade's trigger, the one place it is decided: a submission whose command
    /// buffer bound a heap set (`reaches`) may reach every texture through its resource ID unnamed,
    /// so each texture whose bytes were detached after a submitted command buffer made them
    /// resident is announced now, once
    /// ([`Degradation::StaleDetachedReach`](super::Degradation::StaleDetachedReach), with its
    /// ID), and counted (`WorkCounts::stale_detached_reaches`). Metal's unnamed read of such a
    /// texture reads the pages the program gave back, which a correct program never relies on;
    /// the stand-in's reads zero. The trigger is reachability, not a read. On the submitting
    /// thread, with no lock of the command buffer's held; one load while no texture is noted.
    pub(super) fn reach_stale_detached(&self, reaches: bool) {
        for texture in self.reached(reaches, Unexact::StalePages) {
            self.shared().counts.stale_detached_reach();
            self.degrade_detached_reach(Degradation::StaleDetachedReach, texture.id);
        }
    }

    /// The storage-sink degrade's trigger, the one place it is decided: a command buffer that bound
    /// a heap set (`reaches`) may reach every texture through its resource ID unnamed, so each
    /// texture over detached bytes whose stand-in has a sink is announced as it commits, once
    /// ([`Degradation::DetachedStorageKeepsWrites`](super::Degradation::DetachedStorageKeepsWrites),
    /// with its ID), and counted (`WorkCounts::detached_storage_reaches`): a read through a
    /// storage-class element there reads what an unnamed write left in the sink, where Metal's
    /// reads zero. In the program's `commit`, with no lock held, so a handler asked for each
    /// occurrence may end the process there.
    pub(super) fn reach_storage_sinks(&self, reaches: bool) {
        for texture in self.reached(reaches, Unexact::StorageSink) {
            self.shared().counts.detached_storage_reach();
            self.degrade_detached_reach(Degradation::DetachedStorageKeepsWrites, texture.id);
        }
    }

    /// The live textures noted for `why`, where a command buffer `reaches` them, no longer
    /// noted: each is announced once. One load while no texture is noted.
    fn reached(&self, reaches: bool, why: Unexact) -> Vec<StaleTexture> {
        let rebinds = &self.shared().rebinds;
        if !reaches || rebinds.stale_count.load(Ordering::Acquire) == 0 {
            return Vec::new();
        }
        let mut stale = lock(&rebinds.stale);
        let (reached, kept): (Vec<StaleTexture>, Vec<StaleTexture>) = stale
            .drain(..)
            .filter(|held| held.texture.strong_count() > 0)
            .partition(|held| held.why == why);
        stale.extend(kept);
        rebinds.stale_count.store(stale.len(), Ordering::Release);
        reached
    }
}
