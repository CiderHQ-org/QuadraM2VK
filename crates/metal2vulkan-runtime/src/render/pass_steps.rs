//! The steps a pass records around its render pass instances (`end_encoding`), the same whether
//! the encoder records the pass at once or leaves it to record later with its executions of
//! render commands (`render/executions.rs`), in this order:
//!
//! - **Around each instance's render pass instance**: its colors a draw reads through a snapshot
//!   moved into the feedback loop layout just before it begins, and back just after it ends
//!   (`render_target::feedback_barriers`).
//! - **Before each instance**: after a memory barrier, the colors over a buffer's bytes that
//!   render through copied stand-ins merged back and copied in again, or copied back
//!   (`linear_merge.rs`); after the first, the shadowed textures' copies into it, each before a
//!   barrier to what follows (`render/shadows.rs`); its sample planes' steps (`sample_planes.rs`);
//!   where its draws test the stencil in the shader, the shadow's filling (`shadow_stencil.rs`),
//!   and where they test the depth there, the depth stand-in's (`depth_stand_in.rs`); where its
//!   draws first hold an extended-range color's `half` shadow, the shadow's setting
//!   (`render/extended_shadow.rs`); the steps its draws take before their pass that read what an
//!   earlier instance wrote (`Before`); the copies into the stand-ins of the colors it renders in
//!   other formats (`render.rs`'s `Reformat`); and, where the pass has custom sample positions, its
//!   first draw's pipeline bound before each of its begins (`sample_positions.rs`).
//! - **After each instance**: the copies back from those stand-ins; the shadow's copy back into
//!   the stencil where its draws test it in the shader, and the stand-in's into the depth where
//!   they test that; then the tile dispatches that follow it,
//!   each after the shadowed textures' copies it reads (`render/tile_dispatch.rs`).
//! - **After the instances**: the sample planes' steps after the last; the copy of a
//!   depth/stencil attachment the resolves read from a scratch image (`resolve::ResolveSource`);
//!   the instance of no draws that resolves the pass where its last does not; and the depth/stencil
//!   resolves past the first, each a pass of its own (`render_target::later_resolve`).
//!
//! The pass's visibility results are reset before it and copied after it (`visibility.rs`),
//! outside these steps: the copies go after the pass in the encoder's own command buffer, and so
//! does the reset before it where the pass records later.

use super::extended_shadow::ShadowFill;
use super::tile_dispatch::{Staging, TileDispatch};
use super::Before;
use crate::attachment_stand_in::StandInCopies;
use crate::depth_stand_in::StandInCopies as DepthStandInCopies;
use crate::device::functions::Vk;
use crate::render_target::{self, FeedbackImage, FramebufferObject, RenderPassObject};
use crate::resolve::ResolveSource;
use crate::sample_planes::PlanesCommands;
use crate::shadow::ShadowCopy;
use crate::shadow_stencil::ShadowCopies;
use crate::tracker::Barrier;
use ash::vk;

/// What one render pass instance records before and after it, past its draws.
pub(super) struct InstanceSteps {
    /// Its draws test and write the stencil in the shader: the shadow is filled before it and
    /// copied back after it.
    pub(super) shadow: bool,
    /// Its draws test the depth in the shader: the depth stand-in is filled before it and copied
    /// back after it.
    pub(super) shader_tests: bool,
    /// The settings of the extended-range colors' `half` shadows its draws first hold.
    pub(super) shadow_fills: Vec<ShadowFill>,
    /// Its draws' steps before their pass that read what an earlier instance wrote.
    pub(super) before: Vec<Before>,
    /// The pipeline bound before it begins, where the pass has custom sample positions.
    pub(super) positioned: Option<vk::Pipeline>,
    /// The copies of the colors it renders through stand-ins of other formats into them,
    /// before it (`render.rs`'s `Reformat`).
    pub(super) reformat_in: Option<StandInCopies>,
    /// Their copies back after it.
    pub(super) reformat_out: Option<StandInCopies>,
    /// The subresources of its colors in the feedback loop layout (`render/snapshot.rs`): moved
    /// there just before it begins and back just after it ends
    /// (`render_target::feedback_barriers`). Empty for every instance whose draws read no color
    /// through a snapshot.
    pub(super) feedback: Vec<FeedbackImage>,
}

/// A tile dispatch, ready to record after the instance before it: its staged attachments'
/// copies, and the shadowed textures' copies before it.
pub(super) struct StagedDispatch {
    pub(super) dispatch: TileDispatch,
    pub(super) staging: Vec<Staging>,
    pub(super) copies: Vec<ShadowCopy>,
}

/// The steps around a pass's render pass instances (the module docs), planned as the encoder
/// ends. They hold the objects they record until the command buffer completes: the encoder
/// retains them, or the pass that records later does.
pub(super) struct PassSteps {
    pub(super) instances: Vec<InstanceSteps>,
    /// The flags of the queue family that records the pass, for which its image barriers are made
    /// (`image_barrier.rs`).
    pub(super) family: vk::QueueFlags,
    /// The instances that follow a memory barrier (`render.rs`'s `reload_at`): before each, the
    /// colors over a buffer's bytes rendered through copied stand-ins are merged back and copied
    /// in again (`reloads`), and those the pass does not load are copied back (`published`).
    pub(super) reload_at: std::collections::BTreeSet<usize>,
    pub(super) reloads: Vec<std::sync::Arc<crate::linear_merge::LinearMerge>>,
    pub(super) published: Option<StandInCopies>,
    /// The shadowed textures' copies before each instance past the first, in order.
    pub(super) texel_copies: Vec<Vec<ShadowCopy>>,
    pub(super) planes: Option<PlanesCommands>,
    /// The copies between the stencil and its shader-side shadow.
    pub(super) shadow_copies: Option<ShadowCopies>,
    /// The copies between the depth and its stand-in.
    pub(super) stand_in_copies: Option<DepthStandInCopies>,
    pub(super) dispatches: Vec<StagedDispatch>,
    /// The copy of the depth/stencil attachment the resolving instances read, and its scratch
    /// image.
    pub(super) resolve_source: Option<ResolveSource>,
    /// The instance of no draws that resolves the pass after its last.
    pub(super) resolve_instance: Option<(RenderPassObject, FramebufferObject)>,
    /// The depth/stencil resolves past the first, each a pass and framebuffer of its own.
    pub(super) later_resolves: Vec<(RenderPassObject, FramebufferObject)>,
    pub(super) clear_values: Vec<vk::ClearValue>,
    pub(super) render_area: vk::Rect2D,
}

impl PassSteps {
    /// Record what goes before instance `index` (the module docs), after the barrier that
    /// orders it after the instance before.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only the caller records it;
    /// every object the steps name is live until it completes, and every image rests in
    /// `GENERAL`.
    pub(super) unsafe fn before(&self, raw: &Vk, command_buffer: vk::CommandBuffer, index: usize) {
        // SAFETY: the caller's contract.
        unsafe {
            if self.reload_at.contains(&index) {
                for merge in &self.reloads {
                    merge.reload_in(raw, command_buffer);
                }
                if let Some(copies) = &self.published {
                    copies.record_in(raw, command_buffer);
                }
            }
            let texels = index.checked_sub(1).and_then(|i| self.texel_copies.get(i));
            for copy in texels.into_iter().flatten() {
                copy.record(raw, command_buffer);
                Barrier::everything().record(raw, command_buffer);
            }
            if let Some(planes) = &self.planes {
                planes.record(raw, command_buffer, index);
            }
            if let Some(instance) = self.instances.get(index) {
                if let Some(copies) = self.shadow_copies.as_ref().filter(|_| instance.shadow) {
                    copies.record_in(raw, command_buffer, index == 0);
                }
                let tests = instance.shader_tests;
                if let Some(copies) = self.stand_in_copies.as_ref().filter(|_| tests) {
                    copies.record_in(raw, command_buffer, index == 0);
                }
                for fill in &instance.shadow_fills {
                    fill.record_in(raw, command_buffer);
                }
                for step in &instance.before {
                    step.record_in(raw, command_buffer);
                }
                if let Some(copies) = &instance.reformat_in {
                    copies.record_in(raw, command_buffer);
                }
            }
        }
    }

    /// Begin instance `index` in `pass` and `framebuffer` (one of its runs, where an execution
    /// splits it), its subpass running secondary command buffers, after binding its positioned
    /// pipeline where it has one, and after moving its feedback colors into their layout
    /// (`InstanceSteps::feedback`).
    ///
    /// # Safety
    ///
    /// As for [`PassSteps::before`]; `pass` and `framebuffer` are the instance's (or a run's
    /// pass compatible with it) and live until the command buffer completes.
    pub(super) unsafe fn begin(
        &self,
        raw: &ash::Device,
        command_buffer: vk::CommandBuffer,
        index: usize,
        pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
    ) {
        let positioned = self.instances.get(index).and_then(|i| i.positioned);
        let feedback = self
            .instances
            .get(index)
            .map_or(&[][..], |i| &i.feedback[..]);
        // SAFETY: the caller's contract; the feedback colors rest in `GENERAL` until here.
        unsafe {
            render_target::feedback_barriers(raw, command_buffer, feedback, true, self.family);
            if let Some(pipeline) = positioned {
                raw.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
            }
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pass)
                .framebuffer(framebuffer)
                .render_area(self.render_area)
                .clear_values(&self.clear_values);
            raw.cmd_begin_render_pass(
                command_buffer,
                &begin,
                vk::SubpassContents::SECONDARY_COMMAND_BUFFERS,
            );
        }
    }

    /// Move instance `index`'s feedback colors back to `GENERAL` (`InstanceSteps::feedback`),
    /// right after its render pass instance ended, before anything after it.
    ///
    /// # Safety
    ///
    /// As for [`PassSteps::before`], but for the feedback colors, in their layout since the
    /// instance began ([`PassSteps::begin`]).
    pub(super) unsafe fn ended(
        &self,
        raw: &ash::Device,
        command_buffer: vk::CommandBuffer,
        index: usize,
    ) {
        let feedback = self
            .instances
            .get(index)
            .map_or(&[][..], |i| &i.feedback[..]);
        // SAFETY: the caller's contract.
        unsafe {
            render_target::feedback_barriers(raw, command_buffer, feedback, false, self.family)
        };
    }

    /// Record what goes after instance `index` (the module docs), once it ended.
    ///
    /// # Safety
    ///
    /// As for [`PassSteps::before`].
    pub(super) unsafe fn after(&self, raw: &Vk, command_buffer: vk::CommandBuffer, index: usize) {
        // SAFETY: the caller's contract.
        unsafe {
            let reformat = self
                .instances
                .get(index)
                .and_then(|i| i.reformat_out.as_ref());
            if let Some(copies) = reformat {
                copies.record_in(raw, command_buffer);
            }
            let shadow = self.instances.get(index).is_some_and(|i| i.shadow);
            if let Some(copies) = self.shadow_copies.as_ref().filter(|_| shadow) {
                copies.record_out(raw, command_buffer);
            }
            let tests = self.instances.get(index).is_some_and(|i| i.shader_tests);
            if let Some(copies) = self.stand_in_copies.as_ref().filter(|_| tests) {
                copies.record_out(raw, command_buffer);
            }
            for staged in self
                .dispatches
                .iter()
                .filter(|s| s.dispatch.after == index + 1)
            {
                // Its shadowed textures' copies, each after a barrier from what precedes it; the
                // dispatch begins with another.
                for copy in &staged.copies {
                    Barrier::everything().record(raw, command_buffer);
                    copy.record(raw, command_buffer);
                }
                staged.dispatch.record(raw, command_buffer, &staged.staging);
            }
        }
    }

    /// Record what goes after the pass's instances (the module docs).
    ///
    /// # Safety
    ///
    /// As for [`PassSteps::before`].
    pub(super) unsafe fn end(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: the caller's contract.
        unsafe {
            if let Some(planes) = &self.planes {
                planes.record(raw, command_buffer, self.instances.len());
            }
            if let Some(source) = &self.resolve_source {
                source.record(raw, command_buffer);
            }
            let resolves = self.resolve_instance.iter().chain(&self.later_resolves);
            for (pass, framebuffer) in resolves {
                Barrier::everything().record(raw, command_buffer);
                let begin = vk::RenderPassBeginInfo::default()
                    .render_pass(pass.raw)
                    .framebuffer(framebuffer.raw)
                    .render_area(self.render_area)
                    .clear_values(&self.clear_values);
                raw.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
                raw.cmd_end_render_pass(command_buffer);
            }
        }
    }
}
