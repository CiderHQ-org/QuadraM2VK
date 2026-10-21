//! The indirect dispatches' recording of an execution on the GPU alone (`gpu_execute.rs`), on
//! every device the generated-commands path does not take: core Vulkan 1.2, MoltenVK's among
//! them.
//!
//! - **What runs.** After the replay's preprocess (`bind::replay_preprocess`) and its barrier, per
//!   command of the range in order: a compute barrier before each but the first
//!   (`Barrier::between_dispatches`), where the execution's order needs it (`gpu_execute.rs`), then
//!   per variant, in the variants' order, its pipeline, its
//!   push bytes naming the command's slot table (`ReplayLayout::table_offset`), its region record
//!   (`ReplayLayout::region_offset`) and the command buffer's status words, and one
//!   `vkCmdDispatchIndirect` of the arguments the preprocess wrote for the command and the variant
//!   (`ReplayLayout::argument_offset`): the command's group count for the variant of the state its
//!   record names, zero for every other variant and for a command that runs nothing. So a command
//!   runs at most one variant, the one its record names, as Metal's command runs its state; no
//!   Vulkan pipeline is chosen from GPU data, and nothing is guessed.
//! - **Cost.** Commands times variants indirect dispatches (at most `gpu_execute::MAX_DISPATCHES`)
//!   and a barrier per command. MoltenVK 1.4.1 (an M3 Max on macOS 27) encodes each indirect
//!   dispatch as a Metal one whatever its group count, which its GPU runs in about 3.3 us, and ends
//!   its Metal compute encoder at each barrier (argument buffers and residency sets in use, its
//!   `MVKCmdPipelineBarrier::encode`), about 20 us of the GPU's time more.
//! - **Pipelines.** Each variant's `VkPipeline` of its one region, made once and kept with the
//!   variant (`Variant::region_pipeline`); its layout holds no descriptor binding (an empty set at
//!   most, which nothing binds), and its push bytes are all it reads besides memory by device
//!   address ([`Template::of`]).

use super::binding::PushWrite;
use super::degradation::BoundFormats;
use super::encoder::EncoderCore;
use super::error::{internal, Error};
use super::gpu_execute::{Prepared, Scratch};
use super::pipeline::VariantObjects;
use super::resource::ResourceKey;
use super::tracker::{Barrier, Use};
use ash::vk;
use metal2vulkan::bind::{IndirectArgs, PushHole, ReplayPreprocess};
use std::sync::Arc;
use std::time::Duration;

/// What a dispatch's push bytes take per command.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Hole {
    /// The command's slot table.
    Table,
    /// The command's region record.
    Region,
    /// The command buffer's status words.
    Status,
}

/// A variant's push bytes as its dispatches write them: its push writes, and where in them each
/// command's slot table, region record and the status words go (a write's index, the bytes into
/// it, and what goes there).
#[derive(Clone)]
pub(super) struct Template {
    writes: Vec<PushWrite>,
    holes: Vec<(usize, usize, Hole)>,
}

impl Template {
    /// The template of `p`'s variant: `None` where its commands are not one region whose group
    /// count the replay's arguments give, with no root table and no clamp, whose every hole one of
    /// its push writes of the hole's stages covers and is the command's slot table, region record
    /// or the status words, or where its pipeline layout holds a descriptor binding: the host step
    /// remains for it. An empty set is bound by none: Vulkan asks a bound set only "for each set n
    /// that is statically used by a bound shader" (VUID-vkCmdDispatch-None-08600), and a set of no
    /// binding cannot be.
    pub(super) fn of(p: &Prepared) -> Option<Template> {
        let [region] = p.commands.regions.as_slice() else {
            return None;
        };
        if region.indirect != Some(IndirectArgs::Replay)
            || p.commands.root_table.is_some()
            || p.commands.indirect_clamp.is_some()
            || p.variant
                .pipeline()
                .layout
                .sets
                .iter()
                .any(|set| !set.bindings.is_empty())
        {
            return None;
        }
        let writes: Vec<PushWrite> = region
            .push_constants
            .iter()
            .map(|write| (write.stages.into(), write.offset, write.bytes.clone()))
            .collect();
        let mut holes = Vec::with_capacity(p.commands.push_holes.len());
        for hole in &p.commands.push_holes {
            let (kind, stages, offset) = match *hole {
                PushHole::RootTableAddress { stages, offset } => (Hole::Table, stages, offset),
                PushHole::ReplayRegionAddress { stages, offset } => (Hole::Region, stages, offset),
                PushHole::ExecutionStatusAddress { stages, offset } => {
                    (Hole::Status, stages, offset)
                }
                _ => return None,
            };
            let stages: vk::ShaderStageFlags = stages.into();
            let at = writes.iter().position(|(own, start, bytes)| {
                *own == stages
                    && *start <= offset
                    && offset as usize + 8 <= *start as usize + bytes.len()
            })?;
            holes.push((at, (offset - writes[at].1) as usize, kind));
        }
        Some(Template { writes, holes })
    }

    /// Write `address` into every hole of `kind`.
    fn fill(&mut self, kind: Hole, address: u64) {
        let Template { writes, holes } = self;
        for &(at, offset, hole) in holes.iter() {
            if hole == kind {
                writes[at].2[offset..offset + 8].copy_from_slice(&address.to_le_bytes());
            }
        }
    }
}

/// Record into `core`, after the replay's preprocess `pre` and its barrier, the indirect
/// dispatches of the commands of `prepared`'s variants, each with its `templates`' push bytes
/// (the module docs), reading `scratch`, a barrier between every two commands where `barriers`,
/// with `uses`, what the variants' plans reach. Each variant's objects and its pipeline's creation
/// time, in the variants' order. Refused as the device refuses a variant's pipeline.
pub(super) fn record(
    core: &mut EncoderCore,
    (pre, prepared, templates): (&ReplayPreprocess, &[Prepared], Vec<Template>),
    (scratch, barriers): (&Scratch, bool),
    uses: &[Use<ResourceKey>],
) -> Result<Vec<(Arc<VariantObjects>, Duration)>, Error> {
    let layout = pre.layout;
    if layout.regions != 1 || layout.variants as usize != prepared.len() {
        return Err(internal(
            "a dispatches' replay of other than one region a command, or of other variants",
        ));
    }
    // The commands bind no texture the CPU knows of: the GPU writes their records.
    let unbound: BoundFormats<'_> = &|_| None;
    let mut dispatching = Vec::with_capacity(prepared.len());
    let mut made = Vec::with_capacity(prepared.len());
    for (p, mut template) in prepared.iter().zip(templates) {
        let [region] = p.commands.regions.as_slice() else {
            return Err(internal("a replay variant of other than one region"));
        };
        let pipeline = p.variant.region_pipeline(region, unbound)?;
        let objects = Arc::clone(p.variant.objects());
        template.fill(Hole::Status, scratch.status);
        made.push((Arc::clone(&objects), pipeline.created_in));
        dispatching.push((pipeline.raw, objects.layout(), template));
        core.retain_shared(pipeline);
    }
    let (buffer, base, address) = (
        scratch.upload.raw(),
        scratch.upload.offset(),
        scratch.address,
    );
    let several = dispatching.len() > 1;
    let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
    core.command(compute, uses, move |raw, command_buffer| {
        // SAFETY: the segment's command buffer is recording outside a render pass; every
        // pipeline is held until the command buffer completes, as are the scratch, whose chunk
        // has indirect usage and holds each command's arguments at a multiple of 4 bytes, and the
        // slot tables, region records and status words the push bytes name.
        unsafe {
            for slot in 0..layout.slots {
                if slot > 0 && barriers {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                for (variant, (pipeline, pipeline_layout, template)) in
                    (0u32..).zip(dispatching.iter_mut())
                {
                    if slot == 0 || several {
                        raw.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            *pipeline,
                        );
                    }
                    template.fill(Hole::Table, address + layout.table_offset(slot));
                    template.fill(Hole::Region, address + layout.region_offset(slot, 0));
                    for (stages, offset, bytes) in &template.writes {
                        raw.cmd_push_constants(
                            command_buffer,
                            *pipeline_layout,
                            *stages,
                            *offset,
                            bytes,
                        );
                    }
                    raw.cmd_dispatch_indirect(
                        command_buffer,
                        buffer,
                        base + layout.argument_offset(slot, 0, variant),
                    );
                }
            }
        }
    });
    Ok(made)
}
