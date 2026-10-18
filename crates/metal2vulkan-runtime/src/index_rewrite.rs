//! The index rewrite pass of an indexed strip that restarts where Vulkan does not (the modules and
//! their contract are `bind::index_rewrite`).
//!
//! After `RenderCommandEncoder::set_primitive_restart_enabled` (Apple's
//! `setPrimitiveRestartEnabled:index:`), Metal restarts an indexed line or triangle strip at the
//! call's index `k'` alone, and draws the index type's all-ones index as a vertex; Vulkan restarts
//! at the all-ones index alone. Where `k'` is not the all-ones index, the render encoder draws the
//! strip as the list this pass writes:
//!
//! - **Before the pass** (hoisted, as the index bounds pass is, `index_bounds.rs`): three
//!   dispatches of the program's modules, one per `RewritePhase` (count, place, write), read
//!   Metal's indices (and an indirect draw's arguments) and write the list and the draw's
//!   `VkDrawIndexedIndirectCommand` into the draw's rewrite words (an upload of
//!   [`IndexRewritePass::BYTES`]), between a barrier from everything before and one to
//!   everything after. The count and write dispatches spread the draw's positions over up to
//!   `IndexRewritePass::MAX_GROUPS` workgroups, as many as its index buffer's whole indices need.
//! - **The list** of a direct draw lies in a buffer of its own, of the program's capacity of the
//!   draw's count, made for the draw; an indirect draw's, whose count only the GPU knows, in the
//!   command buffer's index pool, which the index bounds pass shares (its counter in the arena,
//!   its indices in a block the command buffer leases from the device, `gpu_pools.rs`): the count
//!   and place dispatches bind the counter as the output, the write dispatch the block. A list the
//!   pool cannot hold draws nothing and fails the command buffer with
//!   `CommandBufferError::IndexPoolOverflow`.
//! - **In the pass**, one indexed indirect draw of 32-bit indices over the list, of the class's
//!   list topology with restart off (`render.rs`): every primitive keeps its winding, its
//!   provoking vertex and its place in `primitive_id`'s count.
//! - **The hazard**, as the index bounds pass's: where an earlier draw or tile dispatch of the
//!   encoder writes the indices or arguments behind a memory barrier, the pass records before the
//!   draw's own render pass instance, after the writer's; a write since the last barrier that the
//!   pass would read in place is refused (`NotImplemented(RestartedStrip)`), as is any earlier
//!   write where an execution's command draws. A blit's or another encoder's write is ordered
//!   before the pass by the command buffer's tracking, as any compute read is.

use super::buffer::Buffer;
use super::degradation::Degradation;
use super::draw_source::{DrawSource, IndexBind, SizedIndexBind};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::gpu_pools::IndexPool;
use super::index_bounds;
use super::owned_kernel::{OwnedKernel, OwnedModule, OwnedSets};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::Upload;
use crate::device::functions::Vk;
use ash::vk;
use metal2vulkan::bind::{
    ExecutionStatus, IndexBufferRef, IndexRewriteModule, IndexRewritePass, IndirectRef,
    RewriteOutput, RewritePhase, RewriteProgram, RewriteSource,
};
use metal2vulkan::device::VkFeature;

/// A rewritten draw's Metal arguments: a direct draw's values, or an indirect draw's buffer.
pub(super) enum Arguments<'a> {
    /// A direct draw's index count, instances, base vertex and base instance.
    Direct {
        count: u32,
        instances: u32,
        base_vertex: i32,
        base_instance: u32,
    },
    /// An indirect draw's `MTLDrawIndexedPrimitivesIndirectArguments`.
    Indirect(&'a DrawSource<'a>),
}

/// Where a pass writes its list.
enum Output {
    /// A direct draw's own buffer.
    Owned(Buffer),
    /// The command buffer's index pool.
    Pooled(IndexPool),
}

/// One draw's pass, planned: what records before the render pass instance, and the draw in it.
pub(super) struct RewritePass {
    words: Upload,
    output: Output,
    /// The kernels of its three dispatches, in order (`RewritePhase::ALL`).
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// The set of the count and place dispatches, and the write dispatch's where it differs (a
    /// pooled list's: the block where the others bind the counter).
    sets: (OwnedSets, Option<OwnedSets>),
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
    /// An indirect draw's Metal index offset and the buffer's length, where the offset lies at or
    /// past the length: every position draws index 0 there, where Metal reads memory.
    past_length: Option<(u64, u64)>,
}

impl RewritePass {
    /// The pass of `program` restarting at `restart` (`k'`) over the `index_size`-byte indices of
    /// `index`, with `arguments`, in `core`'s command buffer, whose reads of the Metal buffers are
    /// `uses` ([`reads`]). Refused (`NotImplemented(RestartedStrip)`) where a direct draw's list
    /// is past what one storage range binds or a `u32` counts, or carries a base instance the
    /// device's indirect draws cannot (`drawIndirectFirstInstance`), and where the binding layer
    /// cannot bind the pass's ranges; and as buffers, uploads, kernels and descriptor sets are
    /// refused.
    pub(super) fn new(
        core: &mut EncoderCore,
        (program, restart): (RewriteProgram, u32),
        index: &DrawSource<'_>,
        arguments: &Arguments<'_>,
        index_size: u64,
        uses: Vec<Use<ResourceKey>>,
    ) -> Result<RewritePass, Error> {
        let device = core.device().clone();
        let limits = &device.properties().description.limits;
        let gap = || Error::NotImplemented(RuntimeGap::RestartedStrip);
        let (source, owned) = match *arguments {
            Arguments::Direct {
                count,
                instances,
                base_vertex,
                base_instance,
            } => {
                if base_instance != 0
                    && !device
                        .enabled()
                        .features
                        .contains(&VkFeature::DrawIndirectFirstInstance)
                {
                    return Err(gap());
                }
                let capacity = program.capacity(u64::from(count));
                let bytes = capacity * 4;
                if bytes > u64::from(limits.max_storage_buffer_range) {
                    return Err(gap());
                }
                let capacity = u32::try_from(capacity).map_err(|_| gap())?;
                let source = RewriteSource::Direct {
                    count,
                    instances,
                    base_vertex,
                    base_instance,
                };
                (source, Some((capacity, bytes.max(16))))
            }
            Arguments::Indirect(indirect) => {
                let source = RewriteSource::Indirect {
                    arguments: IndirectRef::new(0, indirect.at),
                };
                (source, None)
            }
        };
        let output = match owned {
            Some((capacity, _)) => RewriteOutput::Owned { capacity },
            None => RewriteOutput::Pooled {
                pool_bytes: u32::try_from(device.shared().gpu_pools.index_bytes)
                    .map_err(|_| internal("an index pool past a u32's bytes"))?,
            },
        };
        // The indices the draw reads before where Metal's fetch stops (the source's bytes).
        let index_end = index.at + index.bound(index.at);
        let pass = IndexRewritePass::new(
            source,
            restart,
            (index_size, index.at, index_end),
            output,
            limits.min_storage_buffer_offset_alignment,
            limits.max_storage_buffer_range,
        )
        .ok_or_else(gap)?;
        let kernel = |phase| device.owned_kernel(OwnedModule::IndexRewrite(program, phase));
        let kernels = [
            kernel(RewritePhase::Count)?,
            kernel(RewritePhase::Place)?,
            kernel(RewritePhase::Write)?,
        ];
        // A direct draw's list holds no value before the pass writes it: a buffer of the
        // device's, made without a submission.
        let output = match owned {
            Some((_, bytes)) => Output::Owned(device.scratch_buffer(bytes)?),
            None => Output::Pooled(core.command_buffer().state().index_pool(&device)?),
        };
        let (words, status) = {
            let mut state = core.command_buffer().state();
            let words =
                state
                    .uploads
                    .upload(&device, &pass.parameter_bytes(), IndexRewritePass::BYTES)?;
            (words, state.status_words(&device)?)
        };
        let region = |upload: &Upload, len: u64| (upload.raw(), upload.offset(), len);
        let words_range = region(&words, IndexRewritePass::BYTES);
        let mut set = [(vk::Buffer::null(), 0, 0); 5];
        set[IndexRewriteModule::WORDS_BINDING as usize] = words_range;
        // A direct draw reads no arguments, and a draw with no index inside the buffer no index:
        // nothing reads the binding, which takes the words.
        set[IndexRewriteModule::ARGUMENTS_BINDING as usize] =
            match (pass.arguments_range, arguments) {
                (Some(range), Arguments::Indirect(indirect)) => {
                    (indirect.raw, range.offset, range.range)
                }
                _ => words_range,
            };
        set[IndexRewriteModule::INDICES_BINDING as usize] = match pass.indices_range {
            Some(range) => (index.raw, range.offset, range.range),
            None => words_range,
        };
        set[IndexRewriteModule::STATUS_BINDING as usize] =
            region(&status, ExecutionStatus::BYTES as u64);
        // A pooled list's count and place dispatches bind the pool's counter, its write dispatch
        // the pool's block, whose indices start where the place dispatch's first word says.
        let (output_range, write_range) = match &output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                ((raw, at, buffer.length()), None)
            }
            Output::Pooled(pool) => (
                region(&pool.counter, IndexRewritePass::POOL_HEADER_BYTES),
                Some((pool.block, 0, pool.bytes)),
            ),
        };
        set[IndexRewriteModule::OUTPUT_BINDING as usize] = output_range;
        if set
            .iter()
            .chain(&write_range)
            .any(|&(buffer, _, _)| buffer == vk::Buffer::null())
        {
            return Err(internal("an index rewrite pass binding without a buffer"));
        }
        // The three's set layouts are defined alike: one set for all, or one for the write.
        let sets = (
            OwnedSets::write(&device, &kernels[0], &[set.to_vec()])?,
            match write_range {
                Some(range) => {
                    set[IndexRewriteModule::OUTPUT_BINDING as usize] = range;
                    Some(OwnedSets::write(&device, &kernels[2], &[set.to_vec()])?)
                }
                None => None,
            },
        );
        let past_length = match arguments {
            Arguments::Indirect(_) => {
                let (buffer, offset) = index.metal();
                (offset >= buffer.length()).then(|| (offset, buffer.length()))
            }
            // A direct draw's positions past the buffer read a padded copy, which the draw
            // announces (`render.rs`).
            Arguments::Direct { .. } => None,
        };
        Ok(RewritePass {
            words,
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
        })
    }

    /// The pass's reads of the Metal arguments and indices ([`reads`]).
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// The draw of the pass: its index binding, the `VkBuffer` of its arguments, and its command,
    /// with `sized` the device's index range binding.
    pub(super) fn draw(
        &self,
        sized: Option<SizedIndexBind>,
    ) -> (
        Option<IndexBind>,
        Option<vk::Buffer>,
        metal2vulkan::vk::DrawCommand,
    ) {
        let (buffer, offset, bytes) = match &self.output {
            Output::Owned(buffer) => {
                let (raw, at) = buffer.at(0);
                (raw, at, buffer.length())
            }
            Output::Pooled(pool) => (
                pool.block,
                IndexRewritePass::POOL_HEADER_BYTES,
                pool.index_bytes(),
            ),
        };
        let index_type = metal2vulkan::vk::IndexType::UINT32;
        let draw = metal2vulkan::vk::DrawCommand::DrawIndexedIndirect {
            index_type,
            index_buffer_offset: offset,
            offset: self.words.offset() + IndexRewritePass::DRAW_OFFSET,
            draw_count: 1,
            stride: 20,
        };
        let bind = IndexBind {
            buffer,
            offset,
            index_type: index_type.into(),
            sized: sized.map(|bind| (bind, bytes)),
        };
        (Some(bind), Some(self.words.raw()), draw)
    }

    /// Hold the pass's set and list in `core`'s command buffer until it completes, and announce
    /// an indirect draw from past its index buffer once; what is left records the dispatch, before
    /// the render pass or before the render pass instance the draw records into (`render.rs`).
    /// Call it once the draw records.
    pub(super) fn into_steps(self, core: &mut EncoderCore) -> RewriteSteps {
        let RewritePass {
            output,
            kernels,
            pass,
            sets,
            uses,
            past_length,
            ..
        } = self;
        let (sets, write) = sets;
        let steps = RewriteSteps {
            kernels,
            pass,
            sets: [
                sets.sets.clone(),
                sets.sets.clone(),
                write.as_ref().unwrap_or(&sets).sets.clone(),
            ],
            uses,
        };
        core.retain(sets);
        if let Some(write) = write {
            core.retain(write);
        }
        if let Output::Owned(buffer) = output {
            core.retain(buffer);
        }
        if let Some((offset, length)) = past_length {
            core.device().degrade(Degradation::ZeroedIndexFetch, || {
                (
                    format!(
                        "an indexed indirect draw from offset {offset} of an index buffer of \
                         {length} bytes"
                    ),
                    "its offset lies at or past the buffer's length, so each position draws \
                     index 0 plus the base vertex"
                        .to_owned(),
                )
            });
        }
        steps
    }
}

/// The pass's reads of the Metal buffers: an indirect draw's, as the index bounds pass's
/// (`index_bounds::reads`); a direct draw's indices from its offset through its count, or to
/// where Metal's fetch stops, in the compute stage.
pub(super) fn reads(
    index: &DrawSource<'_>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    let count = match *arguments {
        Arguments::Indirect(indirect) => return index_bounds::reads(index, indirect, index_size),
        Arguments::Direct { count, .. } => u64::from(count),
    };
    let (indices, offset) = index.metal();
    let end = IndexBufferRef::new(0, indices.length()).fetch_end(offset);
    let whole = end.saturating_sub(offset) / index_size * index_size;
    let bytes = whole.min(count * index_size);
    if bytes == 0 {
        return Vec::new();
    }
    let read = Access::new(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_READ,
    );
    let (identity, range) = indices.tracked(offset..offset + bytes);
    vec![Use::reads(
        identity.key,
        identity.tracked,
        UseRange::Bytes(range),
        read,
    )]
}

/// Of [`reads`], those of the bytes the pass reads in place, which a write since the encoder's
/// last memory barrier races: an indirect draw's as the index bounds pass's
/// (`index_bounds::in_place`); a direct draw's indices all (it copies none for a race).
pub(super) fn in_place(
    index: &DrawSource<'_>,
    arguments: &Arguments<'_>,
    index_size: u64,
) -> Vec<Use<ResourceKey>> {
    match *arguments {
        Arguments::Indirect(indirect) => index_bounds::in_place(index, indirect, index_size),
        Arguments::Direct { .. } => reads(index, arguments, index_size),
    }
}

/// A planned pass's dispatch, its set held by the command buffer ([`RewritePass::into_steps`]).
pub(super) struct RewriteSteps {
    /// The kernels of its three dispatches, in order.
    kernels: [OwnedKernel; 3],
    /// The pass the kernels run: its workgroups.
    pass: IndexRewritePass,
    /// Each dispatch's sets, in order.
    sets: [Vec<vk::DescriptorSet>; 3],
    /// The pass's reads of the Metal arguments and indices.
    uses: Vec<Use<ResourceKey>>,
}

impl RewriteSteps {
    /// The pass's reads of the Metal arguments and indices.
    pub(super) fn uses(&self) -> &[Use<ResourceKey>] {
        &self.uses
    }

    /// Record the dispatches into `core`'s command buffer before the pass, between a barrier from
    /// everything before and one to everything after.
    pub(super) fn record(&self, core: &mut EncoderCore) {
        core.barrier(Barrier::everything());
        core.command(
            vk::PipelineStageFlags2::COMPUTE_SHADER,
            &self.uses,
            |raw, command_buffer| {
                // SAFETY: the encoder's command buffer is recording outside a render pass.
                unsafe { self.dispatch(raw, command_buffer) }
            },
        );
        core.barrier(Barrier::everything());
    }

    /// Record the dispatches into `command_buffer` between a barrier from everything before and one
    /// to everything after: before a render pass instance of the encoder's pass, after the
    /// instances before it (`render.rs`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is the encoder's, recording outside a render pass, and only the encoder
    /// records it.
    pub(super) unsafe fn record_in(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        // SAFETY: as the caller states; barriers need nothing more.
        unsafe {
            Barrier::everything().record(raw, command_buffer);
            self.dispatch(raw, command_buffer);
            Barrier::everything().record(raw, command_buffer);
        }
    }

    /// The pass's three dispatches, in order, each after the one before's writes
    /// (`Barrier::between_dispatches`): the count and write dispatches over the pass's groups,
    /// the place dispatch over one workgroup.
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording outside a render pass, and only one encoder records it.
    unsafe fn dispatch(&self, raw: &Vk, command_buffer: vk::CommandBuffer) {
        let phases = self.kernels.iter().zip(RewritePhase::ALL).zip(&self.sets);
        for (at, ((kernel, phase), sets)) in phases.enumerate() {
            // SAFETY: as the caller states; each kernel is the device's, its sets (whose layout
            // each kernel's is defined as) are live until the command buffer completes (retained
            // by `into_steps`), and so are the uploads, the list and the Metal buffers they name.
            unsafe {
                if at > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                raw.cmd_bind_pipeline(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.pipeline,
                );
                raw.cmd_bind_descriptor_sets(
                    command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    kernel.layout,
                    0,
                    sets,
                    &[],
                );
                raw.cmd_dispatch(command_buffer, phase.workgroups(&self.pass), 1, 1);
            }
        }
    }
}
