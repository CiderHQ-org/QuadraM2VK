//! A draw encoded while its `VkPipeline` is still being made.
//!
//! - **Where creation runs.** A draw whose `VkPipeline` its variant has not made yet hands the
//!   making to the device's pool (`pipeline/pool.rs`) and goes on: the draw returns once encoded,
//!   as any draw, and the makings of several draws, of one command buffer or of many, run side
//!   by side. The encode-time checks and refusals all stay at the draw; only the driver's
//!   creation call moves.
//! - **What records later.** The draw's own commands go into a secondary command buffer of their
//!   own, begun at the draw against the instance's pass and recorded once the pipeline is made:
//!   the instance's secondary ends before it and a new one goes on after it, as around an
//!   execution of render commands ([`Piece::Pending`](super::Piece)), and the pass records as a
//!   segment of its encoder (`deferred.rs`). A draw's recording binds everything it uses and
//!   sets every dynamic state, so it records alike in a secondary of its own. Around it: the
//!   barrier from the draws before that wrote storage the pixel interlock orders (the fragment
//!   imageblock's, or an interlocked color's), recorded first in it; and an active visibility query, which ends in the secondary before and continues in
//!   the one after, the draw counting in a query lent to it of the same offset
//!   (`visibility::Lent`), or in none where it counts no sample.
//! - **When it records.** `commit` records the pass where every pipeline it waits for is made
//!   by then; otherwise it records nothing of it, and the scheduler's host step records it once
//!   they are made (`scheduler`), its batch held in its queue's order meanwhile, with no wait for
//!   the GPU. A pass that executes render commands the CPU writes records at commit, which waits
//!   for its pipelines: its commands are read as they stand then.
//! - **A failed making.** The command buffer completes with `CommandBufferError::PipelineCreation`,
//!   its batch and those after it unrun; `CommandBuffer::failed_pipeline` names the draw (its
//!   encoder's place and its order there), the state, its variant and specialization, and the
//!   driver's error. Where creation ran at the draw, the draw returned that error.
//! - **Recorded once.** A pass recorded at commit and again by a host step (an execution's
//!   buffer the GPU may write since) records each such draw once, its secondary executed by the
//!   recording that is submitted.

use super::bound::{Bound, Recorded};
use super::{DrawState, Recording};
use crate::binding::PushWrite;
use crate::command_buffer::CommandBuffer;
use crate::device::lock;
use crate::draw_source::IndexBind;
use crate::dynamic_state::ExtendedDynamicState;
use crate::error::{vulkan, Error, FailedPipeline, FailedState};
use crate::pipeline::{Pending, RawPipeline, RenderPipelineState};
use crate::render_target;
use crate::tracker::LegacyBarrier;
use crate::visibility::Lent;
use ash::vk;
use metal2vulkan::cache::VariantKey;
use metal2vulkan::pipeline::SpecKey;
use std::sync::Mutex;

/// One recording of a draw, owned until its pipeline is made: [`Recording`]'s but for the
/// device, the command buffer and the pipeline.
pub(super) struct OwnedRecording {
    pub(super) layout: vk::PipelineLayout,
    pub(super) shape: u64,
    pub(super) sets: Vec<vk::DescriptorSet>,
    pub(super) pushed: Vec<crate::binding::SetWrite>,
    pub(super) heap: Option<vk::DescriptorSet>,
    pub(super) push: Vec<PushWrite>,
    pub(super) vertex_buffers: Vec<(u32, vk::Buffer, u64, u64)>,
    pub(super) index: Option<IndexBind>,
    pub(super) indirect: Option<vk::Buffer>,
    pub(super) dynamic: Vec<vk::DynamicState>,
    pub(super) topology: metal2vulkan::vk::PrimitiveTopology,
    pub(super) primitive_restart: bool,
    pub(super) state: DrawState,
    pub(super) extended: Option<ExtendedDynamicState>,
    pub(super) draw: metal2vulkan::vk::DrawCommand,
    pub(super) pair: Option<super::snapshot::Pair>,
}

/// A draw whose pipeline is being made (the module docs).
pub(super) struct PendingDraw {
    /// Its secondary command buffer, begun at the draw against its instance's pass.
    pub(super) draws: vk::CommandBuffer,
    pub(super) pipeline: Pending<RawPipeline>,
    /// The state it draws, which a failed making names.
    pub(super) state: RenderPipelineState,
    /// What a failed making names besides (`FailedPipeline`): its encoder's place among the
    /// command buffer's ended encoders, its order among the encoder's draw calls, the variant
    /// of the state it draws, and its pipeline's specialization constants' values.
    pub(super) encoder: usize,
    pub(super) draw: usize,
    pub(super) variant: VariantKey,
    pub(super) specialization: Vec<(SpecKey, u64)>,
    /// The view mask of the storage barrier it records first
    /// (`render_target::between_storage_draws`), where it has one.
    pub(super) storage_barrier: Option<u32>,
    /// The view mask and destination stages of the barrier from its instance's earlier
    /// indirect argument reads it records first (`render_target::after_indirect_reads`), where it
    /// has one.
    pub(super) indirect_barrier: Option<(u32, vk::PipelineStageFlags)>,
    /// The view mask and masks of the barrier from its instance's earlier writes before their
    /// fragments it records first (`render_target::after_geometry_writes`), where it has one.
    pub(super) geometry_barrier: Option<(u32, LegacyBarrier)>,
    /// The visibility query it counts in.
    pub(super) query: Option<Lent>,
    pub(super) recordings: Vec<OwnedRecording>,
    /// Its recording, once made.
    pub(super) recorded: Mutex<Option<Result<(), Error>>>,
}

impl PendingDraw {
    /// Whether its pipeline's making finished, made or failed.
    pub(super) fn made(&self) -> bool {
        self.pipeline.outcome().is_some()
    }

    /// Whether its pipeline's making failed.
    pub(super) fn failed(&self) -> bool {
        matches!(self.pipeline.outcome(), Some(Err(_)))
    }

    /// Run `waker` once its pipeline's making finished.
    pub(super) fn when_made(&self, waker: impl FnOnce() + Send + 'static) {
        self.pipeline.when_finished(Box::new(move |_| waker()));
    }

    /// Its secondary command buffer, recorded and ended once its pipeline is made, waiting for
    /// that: recorded the first time, as `command_buffer`'s, which a failed making fails
    /// (`CommandBuffer::fail_pipeline`).
    pub(super) fn record(
        &self,
        command_buffer: &CommandBuffer,
    ) -> Result<vk::CommandBuffer, Error> {
        let mut recorded = lock(&self.recorded);
        if recorded.is_none() {
            *recorded = Some(self.record_once(command_buffer));
        }
        match recorded.as_ref() {
            Some(Ok(())) => Ok(self.draws),
            Some(Err(error)) => Err(error.clone()),
            None => Ok(self.draws),
        }
    }

    fn record_once(&self, command_buffer: &CommandBuffer) -> Result<(), Error> {
        let pipeline = match self.pipeline.wait() {
            Ok(pipeline) => pipeline,
            Err(error) => {
                command_buffer.fail_pipeline(FailedPipeline {
                    encoder: self.encoder,
                    draw: self.draw,
                    state: FailedState::Render(self.state.clone()),
                    variant: self.variant.clone(),
                    specialization: self.specialization.clone(),
                    error: error.clone(),
                });
                return Err(error);
            }
        };
        let device = command_buffer.device();
        let shared = device.shared();
        let raw = shared.vk();
        let cb = self.draws;
        // SAFETY: the secondary command buffer was begun at the draw inside a pass compatible
        // with the pipeline's, and only this recording records it, once, after its encoder
        // ended (the command buffer's pool is no encoder's then); every object recorded is live
        // until the command buffer completes (the encoder retained them at the draw, and the
        // pipeline is held with the pass); the barrier is the one the draw would record into its
        // instance, valid in the pass's subpass; the lent query is a continuation of the offset
        // the encoder's draws count in, with none active in this secondary before it.
        unsafe {
            if let Some(view_mask) = self.storage_barrier {
                render_target::between_storage_draws(raw, cb, view_mask);
            }
            if let Some((view_mask, stages)) = self.indirect_barrier {
                render_target::after_indirect_reads(raw, cb, view_mask, stages);
            }
            if let Some((view_mask, barrier)) = self.geometry_barrier {
                render_target::after_geometry_writes(raw, cb, view_mask, barrier);
            }
            if let Some(query) = &self.query {
                query.begin(raw, cb);
            }
            let mut bound = Bound::default();
            let mut recorded = Recorded::default();
            for owned in &self.recordings {
                recorded += Recording {
                    raw,
                    command_buffer: cb,
                    layout: owned.layout,
                    shape: owned.shape,
                    pipeline: pipeline.raw,
                    sets: &owned.sets,
                    pushed: &owned.pushed,
                    push_descriptor: shared.push_descriptor,
                    pushes_after_pipeline_binds: shared.pushes_after_pipeline_binds,
                    heap: owned.heap,
                    push: &owned.push,
                    vertex_buffers: &owned.vertex_buffers,
                    index: owned.index,
                    indirect: owned.indirect,
                    dynamic: &owned.dynamic,
                    topology: owned.topology,
                    primitive_restart: owned.primitive_restart,
                    state: &owned.state,
                    extended: owned.extended,
                    mesh: shared.mesh_draws.as_ref(),
                    draw: owned.draw,
                    pair: owned.pair.clone(),
                }
                .record(&mut bound);
            }
            recorded.count(&shared.counts);
            if let Some(query) = &self.query {
                query.end(raw, cb);
            }
            command_buffer.end_raw(cb).map_err(vulkan)?;
        }
        Ok(())
    }
}
