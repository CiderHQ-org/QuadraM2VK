//! The scratch a tiled heap texture's staged moves run through (`heap_tiled.rs`,
//! `HeapTiledCopies`): one `Private` buffer of three parts, the kernel's two shape slots, a
//! staging part the texture's bytes are copied into and out of, and a tight part of the image's
//! elements, with the tiled texel kernel's two sets over them (into the image, then back).
//!
//! - **A queue's, not a texture's.** Each command queue keeps one ([`TiledScratches`]), which
//!   every staged move, keep and merge of its encoders runs through, from every command buffer
//!   of the queue at once: an encoder takes it at its first staged move
//!   (`EncoderCore::tiled_scratch`), and its command buffer holds it until it completes. Each
//!   move records a full barrier first and its dispatch one on each side, so no two moves' steps
//!   overlap in it, in one command buffer or across the queue's: a queue's command buffers go to
//!   one `VkQueue`, and a pipeline barrier recorded outside a render pass orders "all commands
//!   that occur earlier in submission order" before it. No move reads what an earlier one left
//!   there: a move into the image or back copies the texture's bytes into the staging part first,
//!   and a merge's changed texels are moved over a copy of its kept part
//!   (`HeapTiledCopies::record_changed`).
//! - **Grown, not kept per texture.** A move larger than the queue's scratch holds makes a new
//!   one holding both, its parts' sizes rounded up to powers of two, for the queue's later
//!   encoders; the command buffers that took the old one hold it until they complete. So a queue
//!   holds at most its largest texture's parts rounded up, and about as much again while it
//!   grows.
//!
//! On an RTX 5080 laptop GPU, each texture's own staging and tight buffers, held for its life,
//! were 1.04 GiB of 869 staged textures of up to 90 layers, beside the merges' scratch.

use super::super::buffer::Buffer;
use super::super::device::lock;
use super::super::error::Error;
use super::super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::super::values::HazardTrackingMode;
use super::super::Device;
use ash::vk;
use std::sync::{Arc, Mutex};

/// Where a shape lies in the shape part, per direction: past its bytes, and a multiple of every
/// `minStorageBufferOffsetAlignment`.
const SHAPE_SLOT: u64 = 1024;
/// The bytes of the shape part: a slot per direction.
const SHAPES: u64 = 2 * SHAPE_SLOT;
/// What the staging and tight parts start at: a multiple of every
/// `minStorageBufferOffsetAlignment`.
const PART_ALIGNMENT: u64 = 256;
/// The least bytes a new scratch's staging or tight part holds.
const LEAST_PART: u64 = 64 << 10;

/// The bytes of the staging and tight parts a texture's moves need.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ScratchNeed {
    pub(crate) staging: u64,
    pub(crate) tight: u64,
}

impl ScratchNeed {
    /// Both needs met.
    pub(crate) fn max(self, other: ScratchNeed) -> ScratchNeed {
        ScratchNeed {
            staging: self.staging.max(other.staging),
            tight: self.tight.max(other.tight),
        }
    }
}

/// A scratch's buffer, where its parts lie, and the kernel's sets over them.
struct ScratchParts {
    _buffer: Buffer,
    raw: vk::Buffer,
    /// Where the buffer's byte 0 lies in `raw`: the shape part's first byte.
    base: u64,
    /// The staging and tight parts' bytes.
    held: ScratchNeed,
    sets: OwnedSets,
}

impl ScratchParts {
    /// Where the staging part starts in the buffer.
    fn staging(&self) -> u64 {
        SHAPES
    }

    /// Where the tight part starts in the buffer.
    fn tight(&self) -> u64 {
        (SHAPES + self.held.staging).next_multiple_of(PART_ALIGNMENT)
    }

    /// A scratch of `held` bytes of staging and tight parts on `device`, with `kernel`'s sets.
    fn new(
        device: &Device,
        kernel: &OwnedKernel,
        held: ScratchNeed,
    ) -> Result<ScratchParts, Error> {
        let tight_at = (SHAPES + held.staging).next_multiple_of(PART_ALIGNMENT);
        let buffer =
            device.unzeroed_buffer(tight_at + held.tight, HazardTrackingMode::Untracked)?;
        let (raw, base) = buffer.at(0);
        // Each set binds the tight part, the staging part and a shape slot, in binding order.
        let sets = OwnedSets::write(
            device,
            kernel,
            &[0, SHAPE_SLOT]
                .iter()
                .map(|&slot| {
                    vec![
                        (raw, base + tight_at, held.tight),
                        (raw, base + SHAPES, held.staging),
                        (raw, base + slot, SHAPE_SLOT),
                    ]
                })
                .collect::<Vec<_>>(),
        )?;
        Ok(ScratchParts {
            _buffer: buffer,
            raw,
            base,
            held,
            sets,
        })
    }
}

/// A queue's scratch for staged tiled moves (the module docs), as one of its encoders took it.
pub(crate) struct TiledScratch {
    parts: Arc<ScratchParts>,
}

impl TiledScratch {
    /// Its `VkBuffer`.
    pub(crate) fn raw(&self) -> vk::Buffer {
        self.parts.raw
    }

    /// Where its staging part starts in its `VkBuffer`.
    pub(crate) fn staging_at(&self) -> u64 {
        self.parts.base + self.parts.staging()
    }

    /// Where its tight part starts in its `VkBuffer`.
    pub(crate) fn tight_at(&self) -> u64 {
        self.parts.base + self.parts.tight()
    }

    /// Where shape slot `slot` (0 into the image, 1 back) starts in its `VkBuffer`.
    pub(crate) fn shape_at(&self, slot: usize) -> u64 {
        self.parts.base + slot as u64 * SHAPE_SLOT
    }

    /// The kernel's set over its parts and shape slot `slot`.
    pub(crate) fn set(&self, slot: usize) -> vk::DescriptorSet {
        self.parts.sets.sets[slot]
    }

    /// Whether it holds the parts `need` names.
    pub(crate) fn holds(&self, need: ScratchNeed) -> bool {
        self.parts.held.max(need) == self.parts.held
    }
}

/// A command queue's scratch for staged tiled moves (the module docs); none before its first.
#[derive(Default)]
pub(crate) struct TiledScratches(Mutex<Option<Arc<ScratchParts>>>);

impl TiledScratches {
    /// The queue's scratch on `device` holding `need`: its own where that holds it, else a new
    /// one, its own from now on, of both its own parts and `need`, rounded up to powers of two
    /// (refused as `new_buffer` refuses it).
    pub(crate) fn take(&self, device: &Device, need: ScratchNeed) -> Result<TiledScratch, Error> {
        let mut own = lock(&self.0);
        let parts = match own.as_ref() {
            Some(parts) if parts.held.max(need) == parts.held => {
                device.shared().counts.tiled_scratch_reused();
                Arc::clone(parts)
            }
            held => {
                let need = held.map_or(need, |parts| parts.held.max(need));
                let kernel = device.owned_kernel(OwnedModule::TiledTexels)?;
                let rounded = |bytes: u64| bytes.max(LEAST_PART).next_power_of_two();
                let held = ScratchNeed {
                    staging: rounded(need.staging),
                    tight: rounded(need.tight),
                };
                let parts = Arc::new(ScratchParts::new(device, &kernel, held)?);
                device.shared().counts.tiled_scratch_made();
                // The old one goes as the command buffers holding it complete.
                *own = Some(Arc::clone(&parts));
                parts
            }
        };
        Ok(TiledScratch { parts })
    }
}
