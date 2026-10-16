//! The orderings a program asks for: `Fence` (`MTLFence`) with `Device::new_fence` and the
//! encoders' fence calls, and the compute encoder's resource barrier.
//!
//! - **Fences** order untracked resources between encoders of one queue: `update_fence` records
//!   "after this encoder's work" in the device's submission order, and a later encoder's
//!   `wait_for_fence` becomes a barrier from that work's stages, or a wait for its `VkQueue` when
//!   the update ran on another ([`tracker`](super::tracker)). A fence has no Vulkan object: it is
//!   an identity the tracker keys on.
//! - **Resource barriers** order dispatches of one compute encoder: one global memory barrier,
//!   as for a scope barrier.
//! - **A render encoder's** fences are the same encoder-level orderings: a wait orders all of
//!   the encoder's work after the update, and an update follows all of it, which covers any
//!   stages the calls name (Metal's fences are evaluated at the pass's start and end; "Drivers
//!   may delay fence updates until the end of the encoder"). Its memory barriers order the draws
//!   after them behind those before. Vulkan's barriers inside a render pass instance are limited
//!   to its subpass's dependency on itself, which is by region over the framebuffer-space
//!   stages: it orders the fragment stages' accesses of one pixel. A barrier over the render
//!   targets alone, after the fragment stage, before a stage at or before it (the vertex, object,
//!   mesh or fragment stage: a later draw's fragment stage runs after its earlier stages, so a
//!   barrier before those orders it too), is one of those: the pixels a draw's fragments write as
//!   render targets are the pixels later fragments read as render targets (blending, the depth
//!   and stencil tests, framebuffer fetch), so it records inside the instance
//!   (`render_target::between_draws`); a draw after it that samples an attachment's texture, at
//!   any pixel, records into a new instance (`texture/attached_reads.rs`). Any other
//!   (buffers or textures, which a fragment may write at any address and a draw may read at any
//!   stage) splits the pass: the next draw records into a new instance, after a barrier from
//!   everything before to everything after (`render.rs`, the `render-pass-split` degradation).
//! - **Barriers over the render targets on Apple GPUs.** The Metal Feature Set Tables give that
//!   scope to Mac2 GPUs, not to Apple3 to Apple10 ones: on the M3 Max under macOS 27.0.1 (probe
//!   `rtbarrier`), a draw after one read its own pixel's texels from before the pass, as with no
//!   barrier, and (probe `rtb`) a draw after one from the fragment stage to the fragment or the
//!   vertex stage read other pixels' texels from before the pass too, as an Apple GPU orders none
//!   of them. The runtime orders what each barrier names, as a Mac2 GPU does.
//! - **Barriers after the fragment stage.** On an Apple GPU Metal's validation stops one
//!   ("afterStages (0x2) can not contain MTLRenderStageTile or MTLRenderStageFragment on this
//!   device", M4, EV `CFIT-metal-probe-3`), and Metal without it did not order a fragment's
//!   buffer write before the next draw's vertex read. Its message names the device, and a Mac2
//!   GPU is not measured. The runtime honours each barrier, ordering what it names: a program
//!   for an Apple GPU never records one. A later draw's vertex, object and mesh stages still read
//!   the buffers of their tables a fragment stage of the pass wrote as Metal's do, from a copy
//!   taken before the pass (`render/pass_start_reads.rs`).

use super::blit::BlitCommandEncoder;
use super::compute::ComputeCommandEncoder;
use super::device::{lock, Device};
use super::encoder::EncoderCore;
use super::error::Error;
use super::predication::Event;
use super::render::RenderCall;
use super::render::RenderCommandEncoder;
use super::replay::{BlitCall, ComputeCall, Resource};
use super::resource::{ResourceKey, ResourceRef};
use super::tracker::Barrier;
use metal2vulkan::metal::{BarrierScope, RenderStages};
use std::sync::{Arc, Mutex};

/// A Metal fence (`MTLFence`). Cheap to clone (a retain); `Send + Sync`. Two handles are equal
/// when they are the same fence.
#[derive(Clone)]
pub struct Fence(Arc<FenceShared>);

struct FenceShared {
    device: Device,
    key: ResourceKey,
    label: Mutex<Option<String>>,
}

impl std::fmt::Debug for Fence {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Fence")
            .field("label", &self.label())
            .finish()
    }
}

impl PartialEq for Fence {
    fn eq(&self, other: &Fence) -> bool {
        self.0.key == other.0.key
    }
}

impl Eq for Fence {}

impl std::hash::Hash for Fence {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.key.hash(state);
    }
}

impl Device {
    /// `newFence`.
    pub fn new_fence(&self) -> Fence {
        Fence(Arc::new(FenceShared {
            device: self.clone(),
            key: ResourceKey::new(),
            label: Mutex::new(None),
        }))
    }
}

impl Fence {
    /// `device`.
    pub fn device(&self) -> &Device {
        &self.0.device
    }

    /// `setLabel:`.
    pub fn set_label(&self, label: &str) {
        *lock(&self.0.label) = Some(label.to_owned());
    }

    /// `label`.
    pub fn label(&self) -> Option<String> {
        lock(&self.0.label).clone()
    }

    /// The identity the tracker keys the fence on.
    pub(super) fn key(&self) -> ResourceKey {
        self.0.key
    }
}

impl EncoderCore {
    /// `updateFence:` of `call`: the fence is updated after the encoder's work.
    fn update_fence(&mut self, call: &'static str, fence: &Fence) -> Result<(), Error> {
        self.check_device(call, fence.device())?;
        self.uses().update_fence(fence.0.key);
        Ok(())
    }

    /// `waitForFence:` of `call`: the encoder's work waits for the fence's last update before
    /// the encoder in the queue's commit order, also after the encoder updated the fence itself,
    /// as plain Metal's does (its API validation asserts).
    fn wait_for_fence(&mut self, call: &'static str, fence: &Fence) -> Result<(), Error> {
        self.check_device(call, fence.device())?;
        self.uses().wait_for_fence(fence.0.key);
        Ok(())
    }
}

impl ComputeCommandEncoder {
    /// `memoryBarrierWithResources:count:`: the writes of the encoder's earlier dispatches to
    /// `resources` are visible to its later dispatches. One global memory barrier covers them, as
    /// a scope barrier does; an empty list records nothing.
    ///
    /// Refused (`Invalid`): a resource of another device.
    pub fn memory_barrier_with_resources(
        &mut self,
        resources: &[ResourceRef<'_>],
    ) -> Result<(), Error> {
        self.log(|_| ComputeCall::BarrierWithResources(Resource::all(resources)));
        const CALL: &str = "memory_barrier_with_resources";
        let core = self.core();
        for resource in resources {
            core.check_device(CALL, resource.device())?;
        }
        if !resources.is_empty() {
            core.barrier(Barrier::between_dispatches());
            self.capture(|| Event::Barrier);
        }
        Ok(())
    }

    /// `updateFence:`: the fence is updated after the encoder's dispatches.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        self.log(|_| ComputeCall::UpdateFence(fence.clone()));
        self.core().update_fence("update_fence", fence)
    }

    /// `waitForFence:`: the encoder's dispatches wait for the fence's last update before them in
    /// the queue's commit order.
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        self.log(|_| ComputeCall::WaitForFence(fence.clone()));
        self.core().wait_for_fence("wait_for_fence", fence)
    }
}

impl BlitCommandEncoder {
    /// `updateFence:`: the fence is updated after the encoder's commands.
    pub fn update_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        self.log(|_| BlitCall::UpdateFence(fence.clone()));
        self.core().update_fence("update_fence", fence)
    }

    /// `waitForFence:`: the encoder's commands wait for the fence's last update before them in
    /// the queue's commit order.
    pub fn wait_for_fence(&mut self, fence: &Fence) -> Result<(), Error> {
        self.log(|_| BlitCall::WaitForFence(fence.clone()));
        self.core().wait_for_fence("wait_for_fence", fence)
    }
}

/// The render stages a draw runs before or as its fragment stage: a barrier before one of them
/// orders the draw's fragment stage too.
const UP_TO_FRAGMENT: RenderStages = RenderStages::VERTEX
    .union(RenderStages::OBJECT)
    .union(RenderStages::MESH)
    .union(RenderStages::FRAGMENT);

impl RenderCommandEncoder {
    /// `updateFence:afterStages:`: the fence is updated after the encoder's draws. The update
    /// follows every stage of the encoder's work, which covers `after_stages` (the module docs).
    pub fn update_fence(&mut self, fence: &Fence, after_stages: RenderStages) -> Result<(), Error> {
        self.log(|_| RenderCall::UpdateFence(fence.clone(), after_stages));
        // Every stage is ordered, a superset of the ones named.
        let _ = after_stages;
        // The update follows the encoder's work, its later color groups' passes too, which
        // record in its core after the first group's: their encoders' calls change nothing
        // (`render/color_groups.rs`).
        if self.later_group_encoder() {
            return Ok(());
        }
        self.core().update_fence("update_fence", fence)
    }

    /// `waitForFence:beforeStages:`: the encoder's draws wait for the fence's last update before
    /// the encoder in the queue's commit order, also after the encoder updated the fence, as the
    /// compute encoder's do; every stage of the encoder waits, which covers `before_stages`.
    pub fn wait_for_fence(
        &mut self,
        fence: &Fence,
        before_stages: RenderStages,
    ) -> Result<(), Error> {
        self.log(|_| RenderCall::WaitForFence(fence.clone(), before_stages));
        // Every stage waits, a superset of the ones named.
        let _ = before_stages;
        // All of the encoder's work waits, its later color groups' passes too
        // (`render/color_groups.rs`).
        if self.later_group_encoder() {
            return Ok(());
        }
        self.core().wait_for_fence("wait_for_fence", fence)
    }

    /// `memoryBarrierWithScope:afterStages:beforeStages:`: the writes of the encoder's draws
    /// before the barrier to the kinds of resource `scope` names (buffers, textures, render
    /// targets) are visible to its draws after it (the module docs). A scope of the render
    /// targets alone, after the fragment stage and before a stage at or before it (the vertex,
    /// object, mesh or fragment stage), orders the fragment stage's writes before the later
    /// draws' fragment stages, which run after their earlier stages, inside the render pass
    /// instance; another scope splits the pass there. An empty scope, no stage before or after,
    /// or a scope of the render targets alone after no fragment stage or before none of those
    /// stages orders nothing more than the pass does: only the fragment stage writes a render
    /// target, and a tile dispatch runs between render pass instances
    /// (`render/tile_dispatch.rs`).
    pub fn memory_barrier_with_scope(
        &mut self,
        scope: BarrierScope,
        after_stages: RenderStages,
        before_stages: RenderStages,
    ) {
        self.log(|_| RenderCall::BarrierWithScope(scope, after_stages, before_stages));
        if scope.is_empty() || after_stages.is_empty() || before_stages.is_empty() {
            return;
        }
        if scope == BarrierScope::RENDER_TARGETS {
            if after_stages.contains(RenderStages::FRAGMENT)
                && before_stages.intersects(UP_TO_FRAGMENT)
            {
                self.barrier_between_draws();
            }
            return;
        }
        self.barrier_before_next_draw();
    }

    /// `memoryBarrierWithResources:count:afterStages:beforeStages:`: as
    /// [`memory_barrier_with_scope`](Self::memory_barrier_with_scope) over buffers and textures,
    /// for the writes to `resources`, which splits the pass; an empty list orders nothing.
    ///
    /// Refused, changing nothing (`Invalid`): a resource of another device.
    pub fn memory_barrier_with_resources(
        &mut self,
        resources: &[ResourceRef<'_>],
        after_stages: RenderStages,
        before_stages: RenderStages,
    ) -> Result<(), Error> {
        self.log(|_| {
            RenderCall::BarrierWithResources(Resource::all(resources), after_stages, before_stages)
        });
        const CALL: &str = "memory_barrier_with_resources";
        for resource in resources {
            self.core().check_device(CALL, resource.device())?;
        }
        if !resources.is_empty() && !after_stages.is_empty() && !before_stages.is_empty() {
            self.barrier_before_next_draw();
        }
        Ok(())
    }
}
