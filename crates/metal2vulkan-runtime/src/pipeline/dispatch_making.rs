//! A dispatch's `VkPipeline`s made on the device's pool, as Metal makes a compute state's at its
//! creation.
//!
//! - **One `VkPipeline` per threadgroup size.** The local size is a specialization, so each
//!   threadgroup size, and each edge region of a `dispatchThreads`, is a `VkPipeline` of its own,
//!   which only a dispatch names. The first dispatch of each made it on the program's thread:
//!   0.6 to 1.5 s for a bicubic kernel on NVIDIA's 615 driver, and 14 to 107 ms for a matrix
//!   kernel at a second threadgroup size, where Metal's dispatch waits for no compile.
//! - **A dispatch never waits.** One whose region's `VkPipeline` its variant has not made hands
//!   the making to the device's pool (`pool.rs`) and goes on, as a draw does
//!   (`render/pending.rs`); a making in flight already (another dispatch's) is taken as it stands.
//!   The dispatch is recorded later (`pending_dispatch.rs`); a render encoder's tile dispatch
//!   alike, its pass recorded later (`render/pending_tile.rs`). Its specialized notices are announced
//!   at the dispatch that hands the making over, on the program's thread, naming what it binds: the
//!   pool decides nothing the program sees.
//! - **Nothing at the creation.** A compute state's creation starts no making: one that ran
//!   past the state's release, which never waits, would hold the variant and the device past
//!   the program's last drop, which Metal's never does.
//! - **Kept by the state** (`piece_holds.rs`). What a piece needs (its variant and its making's
//!   ticket) is kept by the state, never by the pool's thread; the state's release stops a piece
//!   not started, its making failing, and a dispatch that took it (of another state of the variant)
//!   makes its own (`pending_dispatch.rs`), as a caller after a failed making does (`making.rs`).

use super::graphics::Notices;
use super::kept;
use super::making::{Begun, Pending, Ticket};
use super::objects::RawPipeline;
use super::piece_holds::PieceHolds;
use super::pool::Place;
use super::{ComputePipelineState, RegionKey, Variant};
use crate::degradation::{BoundFormats, Occurrence};
use crate::error::Error;
use metal2vulkan::bind::DispatchRegion;
use std::sync::Arc;

/// A region's `VkPipeline` for a dispatch that records it later
/// ([`Variant::begin_region_pipeline`]).
pub(crate) enum RegionPipeline {
    Made(Arc<RawPipeline>),
    Making(Pending<RawPipeline>),
}

/// What a compute state keeps for one of its pool pieces (`piece_holds.rs`): each holds the
/// device.
pub(crate) enum RegionHeld {
    /// A making: its variant, its region's key, and its making's ticket.
    Making(Arc<Variant>, RegionKey, Box<Ticket<RegionKey, RawPipeline>>),
    /// What a making gives back once settled, held until let go of: its variant.
    Made { _variant: Arc<Variant> },
}

impl RegionKey {
    /// The key of `region`'s `VkPipeline`.
    pub(crate) fn of(region: &DispatchRegion) -> RegionKey {
        RegionKey {
            specialization: region.specialization.clone(),
            stage_flags: region.stage_flags,
            required_subgroup_size: region.required_subgroup_size,
        }
    }
}

impl Variant {
    /// Announce the specialized notices of `key`'s `VkPipeline` for a dispatch that binds
    /// `formats`.
    fn announce(&self, key: &RegionKey, formats: BoundFormats<'_>) {
        self.objects.device.degrade_specialized(
            &self.objects.specialized,
            &key.specialization,
            Some(formats),
            Occurrence::Made,
        );
    }

    /// The `VkPipeline` of `region` for a dispatch that binds `formats` and records it later
    /// (the module docs), a render encoder's tile dispatch where `tile`: from the memo, the
    /// making in flight for it, or handed now to the device's pool, its making kept by `holds`,
    /// its state's.
    fn begin_region_pipeline(
        self: &Arc<Self>,
        holds: &Arc<PieceHolds<RegionHeld>>,
        (region, tile): (&DispatchRegion, bool),
        formats: BoundFormats<'_>,
    ) -> RegionPipeline {
        let key = RegionKey::of(region);
        match self.pipelines.begin(&key) {
            Begun::Made(made) => RegionPipeline::Made(made),
            // The command that began it announces its kinds.
            Begun::Pending(pending) => RegionPipeline::Making(pending),
            Begun::Yours(ticket) => {
                let counts = &self.objects.device.shared().counts;
                match tile {
                    true => counts.tile_pipeline_pooled(),
                    false => counts.dispatch_pipeline_pooled(),
                }
                self.announce(&key, formats);
                let pending = ticket.pending();
                pending.queued_at(self.pooled(holds, key, ticket));
                RegionPipeline::Making(pending)
            }
        }
    }

    /// The `VkPipeline` of `key`, from the memo, the making in flight for it, or made now on
    /// this thread: what a dispatch whose making failed makes itself (`pending_dispatch.rs`),
    /// its notices announced at the dispatch.
    pub(crate) fn make_region(&self, key: &RegionKey) -> Result<Arc<RawPipeline>, Error> {
        self.pipelines.get_or_make(key, || {
            let made = self.objects.pipeline(key, Notices::Announced)?;
            Ok(made.kept_at(&self.objects, kept::Place::Region(key.clone())))
        })
    }

    /// Hand the making of `key`, `ticket`'s, to the device's pool, kept by `holds`: a
    /// dispatch's, which announced its notices. Where it waits there.
    fn pooled(
        self: &Arc<Self>,
        holds: &Arc<PieceHolds<RegionHeld>>,
        key: RegionKey,
        ticket: Ticket<RegionKey, RawPipeline>,
    ) -> Place {
        let keeping = holds.keep(RegionHeld::Making(Arc::clone(self), key, Box::new(ticket)));
        let device = self.objects.device.clone();
        device.shared().pipeline_pool.run(Box::new(move || {
            // Where the state let go meanwhile, the piece stops: its ticket went with the
            // state, failing the making.
            let Some((lease, held)) = keeping.start() else {
                return;
            };
            let (variant, key, ticket) = match held {
                RegionHeld::Making(variant, key, ticket) => (variant, key, *ticket),
                other => return lease.give_back(other, None),
            };
            let made = variant
                .objects
                .pipeline(&key, Notices::Announced)
                .map(|made| made.kept_at(&variant.objects, kept::Place::Region(key.clone())));
            // Its callers take it from the making: from the settle, the batches that wait for it
            // may run and complete while this piece still holds the variant, and the device with
            // it, until it gave it back. A trail spans that (`trails.rs`).
            let trail = variant.objects.device.shared().trails.open();
            drop(ticket.settle(made));
            #[cfg(all(test, any(m2v_gpu, m2v_cpu_icd)))]
            super::pool::held::settled();
            lease.give_back(RegionHeld::Made { _variant: variant }, Some(trail));
        }))
    }
}

impl ComputePipelineState {
    /// The `VkPipeline` of `region` of `variant`, the state's, for a dispatch that binds
    /// `formats` and records it later ([`Variant::begin_region_pipeline`]), its making kept by
    /// the state: a compute encoder's dispatch, or a render encoder's tile dispatch of a tile
    /// function's state (`render/pending_tile.rs`).
    pub(crate) fn begin_region_pipeline(
        &self,
        variant: &Arc<Variant>,
        region: &DispatchRegion,
        formats: BoundFormats<'_>,
    ) -> RegionPipeline {
        let holds = &self.0.program.pieces;
        variant.begin_region_pipeline(holds, (region, self.is_tile()), formats)
    }
}
