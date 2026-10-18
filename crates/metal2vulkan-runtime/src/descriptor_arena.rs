//! A command buffer's descriptor pools: the set-0 set of each dispatch, draw and tile dispatch it
//! encodes comes from them (but one's that pushes its set 0, `pipeline::objects::
//! pushes_set_zero`), and they go when the command buffer completes, with its uploads and heap
//! sets (`heap_sets.rs` keeps set 1, one per variant).
//!
//! Before, each such command made a `VkDescriptorPool` of its own, which its command buffer
//! released as it completed: on RADV, ANV and NVIDIA a pool is device memory of its own, so a
//! pool per command cost more than the rest of its encoding (2000 dispatches in
//! one command buffer took 14.5 to 17.8 us each to encode on a Radeon 780M).
//!
//! - **Sizing.** A pool holds [`FIRST`] sets of the needs (descriptor counts by type, and pool
//!   flags) of the allocation that opened it in a command buffer's first pool, and twice as many
//!   in each pool the command buffer opens after, up to [`MOST`] sets, and never more than
//!   [`MOST_DESCRIPTORS`] descriptors unless one set needs them. So a command buffer of `n`
//!   commands of one variant opens about `log2(n / FIRST)` pools, and one of a single command one
//!   pool, as before.
//! - **Room is counted, not tried.** Each pool counts the sets and the descriptors of each type it
//!   has left; an allocation takes the newest pool with room for all of its needs, so no
//!   allocation relies on `VK_ERROR_OUT_OF_POOL_MEMORY`. A set is never freed alone (no
//!   `FREE_DESCRIPTOR_SET`): a refused command's set stays allocated until its pool goes.
//! - **Spares**. A pool to open is a spare of the device's, reset, where one has room for the set
//!   (`spares.rs`), whatever the size it was made with; a command buffer that completed gives its
//!   pools back to the device's spares, and one that failed or was dropped unsubmitted releases
//!   them.

use super::error::{vulkan, Error};
use super::spares::{release_in_section, DescriptorPool};
use super::Device;
use ash::vk;

/// The sets of a command buffer's first pool.
pub(super) const FIRST: u32 = 16;

/// The most sets a pool is opened for.
pub(super) const MOST: u32 = 1024;

/// The most descriptors a pool is opened for, unless one set needs more.
pub(super) const MOST_DESCRIPTORS: u64 = 1 << 16;

/// What one set of a layout takes from a pool: its descriptors by type (each binding's whole
/// count) and the pool flags its layout needs.
#[derive(Clone, Default, PartialEq, Eq)]
pub(super) struct SetNeeds {
    /// By raw descriptor type, ascending, each count above 0.
    pub(super) counts: Vec<(i32, u32)>,
    pub(super) flags: vk::DescriptorPoolCreateFlags,
}

impl SetNeeds {
    /// Whether the set takes nothing from a pool (a layout of no descriptors): no set is made.
    pub(super) fn is_empty(&self) -> bool {
        self.counts.is_empty()
    }

    /// The pool sizes of `sets` such sets.
    pub(super) fn sizes(&self, sets: u32) -> Vec<vk::DescriptorPoolSize> {
        self.counts
            .iter()
            .map(|&(ty, count)| {
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::from_raw(ty))
                    .descriptor_count(count.saturating_mul(sets))
            })
            .collect()
    }

    /// The descriptors one set takes, of every type.
    fn descriptors(&self) -> u64 {
        self.counts.iter().map(|&(_, count)| u64::from(count)).sum()
    }
}

/// What a pool has left: its sets and descriptors by type, and its flags (`Book`'s pure part).
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Room {
    pub(super) flags: vk::DescriptorPoolCreateFlags,
    pub(super) sets: u32,
    /// By raw descriptor type, ascending.
    pub(super) counts: Vec<(i32, u32)>,
}

impl Room {
    /// The room of a pool of `sets` sets of `needs`.
    pub(super) fn of(needs: &SetNeeds, sets: u32) -> Room {
        Room {
            flags: needs.flags,
            sets,
            counts: needs
                .counts
                .iter()
                .map(|&(ty, count)| (ty, count.saturating_mul(sets)))
                .collect(),
        }
    }

    /// Whether one set of `needs` fits.
    pub(super) fn fits(&self, needs: &SetNeeds) -> bool {
        self.flags == needs.flags
            && self.sets > 0
            && needs.counts.iter().all(|&(ty, count)| {
                self.counts
                    .iter()
                    .any(|&(left_ty, left)| left_ty == ty && left >= count)
            })
    }

    /// The descriptors left, of every type.
    pub(super) fn descriptors(&self) -> u64 {
        self.counts.iter().map(|&(_, count)| u64::from(count)).sum()
    }

    /// Take one set of `needs`, which fits.
    fn take(&mut self, needs: &SetNeeds) {
        self.sets -= 1;
        for &(ty, count) in &needs.counts {
            if let Some((_, left)) = self.counts.iter_mut().find(|(left_ty, _)| *left_ty == ty) {
                *left -= count;
            }
        }
    }
}

/// The pools' rooms, in opening order: which pool an allocation takes, and the size of a pool
/// it opens.
#[derive(Default)]
pub(super) struct Book {
    pub(super) rooms: Vec<Room>,
}

impl Book {
    /// The pool one set of `needs` comes from: `Ok(index)` of the newest with room, or
    /// `Err(sets)`, the sets of the pool to open for it (which [`Book::opened`] then records).
    pub(super) fn place(&self, needs: &SetNeeds) -> Result<usize, u32> {
        match self.rooms.iter().rposition(|room| room.fits(needs)) {
            Some(at) => Ok(at),
            None => {
                // `FIRST << 16` is past `MOST`, and far from overflowing.
                let opened = self.rooms.len().min(16) as u32;
                let doubled = (u64::from(FIRST) << opened).min(u64::from(MOST));
                let by_descriptors = MOST_DESCRIPTORS / needs.descriptors().max(1);
                let sets = doubled.min(by_descriptors).max(1);
                // At most `MOST`, so it fits.
                Err(sets as u32)
            }
        }
    }

    /// A pool of `room` was opened, last.
    pub(super) fn opened(&mut self, room: Room) {
        self.rooms.push(room);
    }

    /// One set of `needs` was allocated from pool `at`, which [`Book::place`] gave.
    pub(super) fn take(&mut self, at: usize, needs: &SetNeeds) {
        self.rooms[at].take(needs);
    }
}

/// A command buffer's pools ([`Book`] and the pools, in the same order), with the device they
/// are of once the first is opened.
#[derive(Default)]
pub(crate) struct DescriptorArena {
    book: Book,
    pools: Vec<DescriptorPool>,
    device: Option<Device>,
}

impl DescriptorArena {
    /// One set of `layout`, whose needs are `needs` (not empty), from the newest pool with room
    /// for it, or a pool opened for it: a spare of the device's with room, else a new one.
    pub(crate) fn allocate(
        &mut self,
        device: &Device,
        layout: vk::DescriptorSetLayout,
        needs: &SetNeeds,
    ) -> Result<vk::DescriptorSet, Error> {
        let raw = device.shared().vk();
        let at = match self.book.place(needs) {
            Ok(at) => at,
            Err(sets) => {
                let shared = device.shared();
                let pool = match shared.spares.descriptor_pool(shared, needs)? {
                    Some(spare) => spare,
                    None => {
                        let sizes = needs.sizes(sets);
                        let info = vk::DescriptorPoolCreateInfo::default()
                            .flags(needs.flags)
                            .max_sets(sets)
                            .pool_sizes(&sizes);
                        // SAFETY: `info` is live for the call.
                        let raw_pool =
                            unsafe { raw.create_descriptor_pool(&info, None) }.map_err(vulkan)?;
                        DescriptorPool {
                            raw: raw_pool,
                            room: Room::of(needs, sets),
                        }
                    }
                };
                self.book.opened(pool.room.clone());
                self.pools.push(pool);
                self.device.get_or_insert_with(|| device.clone());
                self.pools.len() - 1
            }
        };
        let layouts = [layout];
        let info = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.pools[at].raw)
            .set_layouts(&layouts);
        // The set straight into a local: `ash`'s wrapper collects it into a `Vec`.
        let mut set = vk::DescriptorSet::null();
        // SAFETY: the pool has room for the set (counted); its sets go with it. `info` names one
        // layout, so the driver writes one set.
        unsafe { (raw.fp_v1_0().allocate_descriptor_sets)(raw.handle(), &info, &mut set) }
            .result()
            .map_err(vulkan)?;
        self.book.take(at, needs);
        Ok(set)
    }

    /// Give every pool to the device's spares: the command buffer completed, so no command uses
    /// their sets any more.
    pub(crate) fn give_back(&mut self) {
        self.book.rooms.clear();
        let Some(device) = &self.device else {
            return;
        };
        let shared = device.shared();
        for pool in self.pools.drain(..) {
            shared.spares.give_descriptor_pool(shared, pool);
        }
    }

    /// Release every pool: no command uses their sets any more (the command buffer failed, or
    /// was never submitted).
    pub(crate) fn release(&mut self) {
        self.book.rooms.clear();
        let Some(device) = &self.device else {
            return;
        };
        for pool in self.pools.drain(..) {
            // SAFETY: no pending command uses the pool's sets: the arena goes when its command
            // buffer completed, or was dropped unsubmitted; the sets are freed with it.
            unsafe { release_in_section(device.shared(), pool.raw) };
        }
    }
}

impl Drop for DescriptorArena {
    fn drop(&mut self) {
        self.release();
    }
}
