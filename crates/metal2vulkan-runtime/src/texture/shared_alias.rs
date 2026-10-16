//! One `VkImage` shared by the textures of one image over the same bytes of one import of host
//! pages. A consumer makes a texture over the presented surface at each present (a no-copy buffer
//! over the surface's pages, which joins the import of a buffer that lives across presents,
//! `host_aliases.rs`, and a texture of its plane over it), releasing the one before first: each
//! the same image over the same bytes. Each texture stays its own Metal object, with its own
//! resource ID, views and hazard identity (its buffer's, which hazard tracking orders with the
//! others' over the same bytes); the `VkImage`, its first move (which keeps the texels it lies
//! over, `texture/first_move.rs`) and, over dma-buf pages, the dedicated import it is bound to
//! (`dma_buf.rs`) are made once.
//!
//! - **The key** ([`AliasKey`]) is structural: the image's place in the import (in its memory, or
//!   from the dma-buf pages' first byte), its plan, the texels' pitches and slices, and its form
//!   (a linear image of the driver's or at Metal's pitch, holding what beside the texels; a Metal
//!   texture of a type; an image over a dma-buf of its pages).
//! - **Held by the import.** The import of the pages (`buffer.rs`'s `HostImport`) holds its
//!   shared images ([`SharedImages`]): the [`HELD_PER_IMPORT`] made or taken last, the oldest
//!   dropped past them, and all of them as the import goes, before its memory is freed. So a
//!   texture made after the one before went takes its image, while a buffer keeps the import. A
//!   texture holds its shared image too, so one dropped by its import goes with the last texture
//!   over it, once the submissions that used it completed. The shared image holds nothing of its
//!   import: every texture over it holds the import, and so does every submission that takes its
//!   move.
//! - **Its move** waits for the next submission that may reach it, as an image's does. The image
//!   notes weakly the pages of each texture over it ([`SharedAlias::taken_over`]); the submission
//!   that takes the move holds the pages of one that lives then until it completed, so no
//!   deallocator runs under its copies, as an image's own pages are held. Where none lives, no
//!   texture reaches the image, and the move waits, registered still, for a submission while one
//!   lives ([`Fresh::still_waits`]). It keeps the texels under the key of the memory the buffers
//!   over them join (`HostAliases::of`), which every joined buffer's uses carry, so hazard
//!   tracking orders it with the uses of any buffer over those bytes, the one that made the image
//!   gone or not.
//! - **A rebind or a detach** of one texture's buffer gives that texture an image of its own over
//!   the new pages, or a stand-in (`rebind.rs`, `detach.rs`); the others keep the shared one, over
//!   the import their buffers still lie in.
//! - **Counted**: `WorkCounts::shared_alias_images`, a texture that took a shared image rather than
//!   make one, and `shared_alias_moves_waited`, a submission that left a move waiting.

use super::first_move::{FirstMove, Fresh};
use super::{ImageImport, ImagePlan};
use crate::device::lock;
use crate::exit_guard;
use crate::linear_plan::{BufferLayout, Holds};
use crate::metal_texture::MetalOver;
use crate::release;
use crate::resource::ResourceKey;
use crate::Device;
use ash::vk;
use metal2vulkan::metal::TextureType;
use std::any::Any;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, Weak};

/// The most shared images an import holds ([`SharedImages`]): a surface's import carries its
/// presented texture's image, and the few other images a consumer makes of it (another usage, a
/// view format); one past them drops the oldest.
pub(crate) const HELD_PER_IMPORT: usize = 4;

/// What a shared image is, by structure, within its import ([`SharedImages`]): textures of the
/// same key take the same `VkImage`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct AliasKey {
    /// The image's place: in the import's memory, or from the dma-buf pages' first byte.
    at: u64,
    plan: ImagePlan,
    /// The texels' row and slice pitches, and slices (their offset in their buffer aside).
    row_pitch: u64,
    slice_pitch: u64,
    slices: u64,
    form: AliasForm,
}

/// How a shared image is made ([`AliasKey`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum AliasForm {
    /// A linear image made for host allocations, bound in the pages' import, holding what
    /// `Holds` lets it beside the texels.
    Linear(Holds),
    /// A Metal texture over the import's `MTLBuffer` (`metal_texture.rs`), of a type.
    Metal(MetalOver, TextureType),
    /// A `DRM_FORMAT_MOD_LINEAR` image over a dma-buf of its own pages, holding what `Holds`
    /// lets it.
    DmaBuf(Holds),
}

impl AliasKey {
    /// The key of an image of `plan` over texels laid out as `layout`, made as `form`, at `at` in
    /// its import.
    pub(crate) fn new(
        at: u64,
        plan: &ImagePlan,
        layout: &BufferLayout,
        form: AliasForm,
    ) -> AliasKey {
        AliasKey {
            at,
            plan: plan.clone(),
            row_pitch: layout.row_pitch,
            slice_pitch: layout.slice_pitch,
            slices: layout.slices,
            form,
        }
    }
}

/// A `VkImage` the textures of one [`AliasKey`] share, and its first move until a submission
/// takes it.
pub(crate) struct SharedAlias {
    device: Device,
    raw: vk::Image,
    /// The dedicated import `raw` is bound to, over dma-buf pages: freed after `raw` is
    /// destroyed.
    import: Option<ImageImport>,
    key: AliasKey,
    /// The key a host call that makes its move names ([`Fresh::rebindable`]): the memory's its
    /// texels lie in, as the buffers over them join it.
    named: ResourceKey,
    first_move: Mutex<Option<FirstMove>>,
    /// The host pages of the textures over it, noted weakly, those gone dropped as one is noted:
    /// the submission that takes its move holds one that lives.
    takers: Mutex<Vec<Weak<dyn Any + Send + Sync>>>,
}

impl SharedAlias {
    /// The shared image `raw` of `key`, bound to `import` where it is a dedicated import's, whose
    /// move is `first`, which a host call naming `named` makes: registered with the device's
    /// first moves. Its import holds it ([`SharedImages::hold`]).
    pub(crate) fn new(
        device: &Device,
        (raw, import): (vk::Image, Option<ImageImport>),
        (key, named): (AliasKey, ResourceKey),
        first: FirstMove,
    ) -> Arc<SharedAlias> {
        let alias = Arc::new(SharedAlias {
            device: device.clone(),
            raw,
            import,
            key,
            named,
            first_move: Mutex::new(Some(first)),
            takers: Mutex::new(Vec::new()),
        });
        let weak: Weak<SharedAlias> = Arc::downgrade(&alias);
        let fresh: Weak<dyn Fresh> = weak;
        device.shared().textures.first_moves.register(fresh);
        alias
    }

    /// The `VkImage`.
    pub(crate) fn raw(&self) -> vk::Image {
        self.raw
    }

    /// Whether it is bound to a dedicated import of its own pages.
    pub(crate) fn dedicated(&self) -> bool {
        self.import.is_some()
    }

    /// Whether its move waits for a submission, and the key a host call that makes it names.
    pub(crate) fn move_waits(&self) -> Option<ResourceKey> {
        lock(&self.first_move).is_some().then_some(self.named)
    }

    /// A texture over it was made, over the host `pages`: noted weakly, for the submission that
    /// takes its move to hold.
    pub(crate) fn taken_over<T: Any + Send + Sync>(&self, pages: &Arc<T>) {
        let pages: Weak<dyn Any + Send + Sync> = Arc::downgrade(pages) as _;
        let mut takers = lock(&self.takers);
        takers.retain(|taker| taker.strong_count() > 0);
        takers.push(pages);
    }
}

impl Fresh for SharedAlias {
    /// Its move, holding the pages of a texture over it that lives; where none lives, no texture
    /// reaches it, and it waits.
    fn take(&self) -> Vec<FirstMove> {
        let mut waiting = lock(&self.first_move);
        let Some(first) = waiting.take() else {
            return Vec::new();
        };
        let live = lock(&self.takers).iter().find_map(Weak::upgrade);
        match live {
            Some(pages) => vec![first.holding_now(pages)],
            None => {
                *waiting = Some(first);
                self.device.shared().counts.shared_alias_move_waited();
                Vec::new()
            }
        }
    }

    fn still_waits(&self) -> bool {
        lock(&self.first_move).is_some()
    }

    fn give_back(&self, moves: Vec<FirstMove>) {
        let mut waiting = lock(&self.first_move);
        for first in moves {
            if waiting.is_none() {
                *waiting = Some(first);
            }
        }
    }

    fn rebindable(&self) -> Option<ResourceKey> {
        let waiting = lock(&self.first_move);
        (waiting.as_ref())
            .is_some_and(|first| first.rebindable())
            .then_some(self.named)
    }
}

impl Drop for SharedAlias {
    fn drop(&mut self) {
        // Once the process is exiting, the image is left to its end (`exit_guard`).
        let Some(_busy) = exit_guard::enter() else {
            return;
        };
        // SAFETY: nothing holds the image: its import let go of it, every texture over it went,
        // each after the command buffers that used it completed, and so did every submission
        // that took its move; their views went before it. Its dedicated import goes after it, as
        // it drops; the memory it is bound in is freed after it (its import drops its shared
        // images first, and every holder of it holds the import).
        unsafe { release::object(self.device.shared().vk(), self.raw) };
    }
}

/// An import's shared images ([`SharedAlias`]), held while it lives: the [`HELD_PER_IMPORT`] made
/// or taken last.
#[derive(Default)]
pub(crate) struct SharedImages {
    held: Mutex<Recent<Arc<SharedAlias>>>,
}

impl SharedImages {
    /// The shared image of `key`, where the import holds one: held as the one taken last.
    pub(crate) fn find(&self, key: &AliasKey) -> Option<Arc<SharedAlias>> {
        let mut held = lock(&self.held);
        held.find(|alias| alias.key == *key).cloned()
    }

    /// Hold `alias`, made over the import, as the one made last; the oldest goes past
    /// [`HELD_PER_IMPORT`], once the lock is free.
    pub(crate) fn hold(&self, alias: &Arc<SharedAlias>) {
        let dropped = lock(&self.held).hold(Arc::clone(alias), HELD_PER_IMPORT);
        drop(dropped);
    }

    /// Let go of every one, as the import goes: once the lock is free.
    pub(crate) fn clear(&self) {
        let held: Vec<Arc<SharedAlias>> = lock(&self.held).items.drain(..).collect();
        drop(held);
    }
}

/// Items held in the order they were held or found last, the oldest first.
struct Recent<T> {
    items: VecDeque<T>,
}

impl<T> Default for Recent<T> {
    fn default() -> Recent<T> {
        Recent {
            items: VecDeque::new(),
        }
    }
}

impl<T> Recent<T> {
    /// The first item `matches` takes, moved to the end (found last).
    fn find(&mut self, matches: impl Fn(&T) -> bool) -> Option<&T> {
        let at = self.items.iter().position(matches)?;
        let item = self.items.remove(at)?;
        self.items.push_back(item);
        self.items.back()
    }

    /// Hold `item` last, and give back the oldest past `bound`.
    fn hold(&mut self, item: T, bound: usize) -> Vec<T> {
        self.items.push_back(item);
        let past = self.items.len().saturating_sub(bound);
        self.items.drain(..past).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An import holds its last [`HELD_PER_IMPORT`] images, a found one counting as held last,
    /// and gives back the oldest past them.
    #[test]
    fn an_import_holds_its_last_shared_images() {
        let mut recent = Recent::default();
        for item in 0..HELD_PER_IMPORT {
            assert!(recent.hold(item, HELD_PER_IMPORT).is_empty());
        }
        assert_eq!(recent.find(|&item| item == 0), Some(&0));
        assert_eq!(recent.hold(HELD_PER_IMPORT, HELD_PER_IMPORT), vec![1]);
        assert_eq!(recent.find(|&item| item == 1), None);
        let held: Vec<usize> = recent.items.iter().copied().collect();
        assert_eq!(held, [2, 3, 0, 4]);
        assert_eq!(recent.hold(5, 2), vec![2, 3, 0]);
    }
}
