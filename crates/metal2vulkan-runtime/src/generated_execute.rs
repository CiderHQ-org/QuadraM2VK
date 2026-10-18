//! Executing an indirect command buffer's kernel-written compute commands on the GPU alone
//! (`executeCommandsInBuffer:withRange:`, the orchestrator's ruling of 2026-10-06).
//!
//! - **Totality, decided at commit.** An execution whose buffer an encoder made resident for
//!   writing (`indirect_execute.rs`) runs here where every record it may hold is covered: the
//!   device enabled `VK_EXT_device_generated_commands` (`DeviceOptions::
//!   withhold_generated_commands` keeps it off); the buffer does not inherit the encoder's
//!   buffers; every pipeline state its records may name (the encoder's where the buffer inherits
//!   it, else every state made with `supportIndirectCommandBuffers` alive at the commit) compiles
//!   as a flat device-replay variant (`ComputePipelineState::replay_variant`: its kernel does not
//!   observe its threadgroup partition) whose plan binds no descriptor, and fetches no
//!   `[[stage_in]]` attribute, whose reach past 4 GiB the host step checks (`fetch_reach.rs`); no
//!   command of the range has a state the CPU set that the program released since, whose program
//!   the command runs while its record names the ID (`IndirectCommandBuffer::name_state`), and the
//!   variants of the states alive run the ID's holder instead; and the product sets the
//!   generated-commands path up for them (`bind::replay_preprocess_generated`). Each flat variant
//!   runs every grid and threadgroup size of its state, so the records' contents cannot leave a
//!   command uncovered. Otherwise the execution takes the host step (`indirect_execute.rs`, part
//!   3a), which the device announces once (`Degradation::HostReplayedIndirectCommands`). Nothing is
//!   decided at run time.
//! - **What runs.** One segment: the replay's preprocess (`bind::replay_preprocess_module`, an
//!   owned kernel) over the range, reading the records where the GPU and the CPU left them and
//!   writing each command's slot table, region record and sequences; a barrier; then per command
//!   of the range one `vkCmdExecuteGeneratedCommandsEXT` of the sequences the preprocess counted
//!   for it, a compute barrier before each but the first. Where the driver's own preprocessing of
//!   an execution after another goes wrong (`generated_commands::preprocess_apart`), the
//!   commands layout is made for explicit preprocessing: every execution of the range is
//!   preprocessed (`vkCmdPreprocessGeneratedCommandsEXT`) after the barrier, then a barrier, then
//!   the executions run preprocessed, so no preprocessing follows an execution. A record the
//!   preprocess cannot run (a header with the producers' fault, a record that does not decode or
//!   names no variant's state) runs nothing, and the command buffer's execution report states
//!   why, from the status words the preprocess wrote (`bind::ReplayFaults`): an invalid record is
//!   a skip on the GPU.
//! - **What the commands reach.** The replay's resident set, which the preprocess resolves each
//!   record's buffer addresses in and the commands' kernels search for the pointers they load,
//!   holds the resources the command buffer's encoders made resident and the device's live
//!   buffers Metal reaches unnamed (`live_buffers.rs`, `BindState::resident_ranges`), as a directly
//!   encoded command's does: every buffer of its own, and a no-copy buffer or a heap's buffer once
//!   a command buffer's commands used it. Where the live buffers are past the search's capacity,
//!   the execution takes the host step.
//! - **Objects.** Per execution: each variant's indirect-bindable pipeline under the pipeline
//!   layout the variants share, the indirect execution set and commands layout, the preprocess
//!   memory, and the scratch (a region of the command buffer's upload arena, `upload.rs`); the
//!   command buffer holds them until it completes. The execution record lists each variant once.

use super::argument_table::ArgumentTable;
use super::binding;
use super::blit::reads;
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::degradation::CompiledPipeline;
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{internal, vulkan, Error};
use super::executed_commands::RecordsRead;
use super::execution::{ExecutedPipeline, Execution};
use super::fetch_reach;
use super::generated_commands::{
    preprocess_apart, Commands, Execution as GeneratedExecution, ExecutionSetCreateInfo,
    ExecutionSetPipelineInfo, ExecutionSetToken, GeneratedCommandsInfo, Handle, LayoutCreateInfo,
    LayoutToken, MemoryRequirementsInfo, PushConstantToken, WriteExecutionSetPipeline,
    EXECUTION_SET_PIPELINES, TOKEN_DISPATCH, TOKEN_EXECUTION_SET, TOKEN_PUSH_CONSTANT,
};
use super::indirect_execute::{Execute, ExecutionRange};
use super::live_buffers::LiveTable;
use super::owned_kernel::OwnedModule;
use super::pipeline::{ComputePipelineState, RawPipeline, Variant, VariantObjects};
use super::release::{self, Generated};
use super::residency::Residency;
use super::tracker::Barrier;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    replay_preprocess_generated, BindState, BindingPlan, DispatchCommands, GeneratedLayout,
    GeneratedVariant, NeutralResource, PushHole, ReplayExecution, ReplayPreprocess, ReplayVariant,
    RootTable, StageTable,
};
use metal2vulkan::error::BindError;
use metal2vulkan::pipeline::Stage;
use std::collections::BTreeMap;
use std::num::NonZeroU64;
use std::sync::Arc;

/// The threads of a flat variant's threadgroup: a wave on most devices, and within every
/// state's maximum but the smallest, which takes its maximum.
const FLAT_THREADS: u32 = 64;

/// One variant of the execution: its compiled variant and plan, its replay commands, and its
/// registration.
struct Prepared {
    variant: Arc<Variant>,
    plan: Arc<BindingPlan>,
    commands: DispatchCommands,
    registered: ReplayVariant,
}

/// What [`total`] found: the variants, the replay's preprocess, the bind state they were
/// planned for, and the live buffers its resident set holds. The command buffer keeps their
/// ranges, not their pages, until its first submission, which finds whether they still lie there
/// and holds the table then current until it completes.
struct Total {
    prepared: Vec<Prepared>,
    pre: ReplayPreprocess,
    state: BindState,
    live: Arc<LiveTable>,
}

/// The recording of `execute` on the GPU alone into a segment of `command_buffer`, where the
/// execution is total on its device (the module's rules); `None` where it takes the host step.
/// Refused as the device refuses an object it makes, or the arena an upload.
pub(super) fn record(
    command_buffer: &CommandBuffer,
    execute: &Execute,
) -> Result<Option<RecordedSegment>, Error> {
    let device = command_buffer.device();
    let Some(commands) = device.shared().generated_commands else {
        return Ok(None);
    };
    let residency = command_buffer.residency()?.kernel();
    match total(device, execute, &residency)? {
        Some(total) => {
            // The GPU reads the records: the handlers learn the states they may name.
            command_buffer.report_executed(&execute.icb, RecordsRead::Gpu, &residency, || {
                let states = execute.states(device);
                let encode = states
                    .iter()
                    .any(ComputePipelineState::can_encode_indirect_commands);
                (Vec::new(), encode)
            });
            record_total(command_buffer, execute, &residency, &commands, total).map(Some)
        }
        None => Ok(None),
    }
}

/// `result`'s value; `None` where the binder refuses what the generated-commands path asks for
/// (`Unsupported`, an invalid replay, or a threadgroup size that needs another variant): the
/// execution takes the host step. Any other error is the runtime's or the binder's.
fn refusal<T>(result: Result<T, BindError>) -> Result<Option<T>, Error> {
    match result {
        Ok(value) => Ok(Some(value)),
        Err(
            BindError::Unsupported(_)
            | BindError::InvalidReplay { .. }
            | BindError::NeedsVariant { .. },
        ) => Ok(None),
        Err(error) => Err(Error::Bind(error)),
    }
}

/// The variants and the replay's preprocess of `execute` on `device`, with `residency`, its
/// command buffer's, where every record it may hold is covered (the module's rules); `None`
/// where it is not.
fn total(
    device: &Device,
    execute: &Execute,
    residency: &Residency,
) -> Result<Option<Total>, Error> {
    let ExecutionRange::Fixed(range) = &execute.range else {
        return Ok(None);
    };
    if execute.icb.descriptor().inherit_buffers || range.is_empty() {
        return Ok(None);
    }
    // A command whose state the CPU set and the program released since runs that state's
    // program while its record names the ID (`IndirectCommandBuffer::name_state`), which no
    // variant here runs: a state of the ID's live holder would.
    if execute.icb.sets_a_released_state(range.clone()) {
        return Ok(None);
    }
    let Ok(capacity) = u32::try_from(execute.icb.size()) else {
        return Ok(None);
    };
    // A replay binds no kernel buffer, and the command buffer's resident resources, every
    // encoder's (`residency.rs`). Its resident set holds the live buffers Metal reaches unnamed
    // besides (`live_buffers.rs`), listed apart from the used resources: a record's buffer address
    // and a pointer its kernel loads reach them, as Metal's do, of any count (Past 2^16 entries the
    // execution took the host step).
    let live = device.shared().live_buffers.table(device)?;
    let mut state = BindState::default();
    state.stages.insert(Stage::Kernel, StageTable::default());
    state.used_resources = residency.used_resources();
    state.resident_ranges = live.ranges().collect();
    let shape = state.shape();
    let states = execute.states(device);
    let mut prepared = Vec::with_capacity(states.len());
    for pipeline in &states {
        match prepare(pipeline, &state, &shape)? {
            Some(p) => prepared.push(p),
            None => return Ok(None),
        }
    }
    let mut execution = ReplayExecution::new(capacity, range.start, range.len() as u32);
    execution.inheritance = execute.icb.inheritance();
    execution.parent.pipeline = execute
        .pipeline
        .as_ref()
        .and_then(|pipeline| NonZeroU64::new(pipeline.gpu_resource_id().0));
    // A flat variant runs each command as one region.
    execution.regions = 1;
    let registered: Vec<ReplayVariant> = prepared.iter().map(|p| p.registered).collect();
    let generated: Vec<GeneratedVariant<'_>> = prepared
        .iter()
        .map(|p| GeneratedVariant::new(p.variant.pipeline(), &p.plan, &p.commands))
        .collect();
    let Some(pre) = refusal(replay_preprocess_generated(
        device.compile_device(),
        &state,
        &execution,
        &registered,
        &generated,
    ))?
    else {
        return Ok(None);
    };
    if pre.generated.is_none() || pre.layout.generated.is_none() {
        return Err(internal(
            "a generated-commands replay without its sequences",
        ));
    }
    Ok(Some(Total {
        prepared,
        pre,
        state,
        live,
    }))
}

/// The flat replay variant of `pipeline` for `state` (of `shape`), instantiated for its
/// threadgroup size; `None` where it has none (`ComputePipelineState::replay_variant`), its
/// plan cannot run that size, or its kernel fetches `[[stage_in]]` attributes.
fn prepare(
    pipeline: &ComputePipelineState,
    state: &BindState,
    shape: &metal2vulkan::bind::BindShape,
) -> Result<Option<Prepared>, Error> {
    // A kernel that fetches `[[stage_in]]` attributes leaves the execution to the host step,
    // which checks each command's fetch past 4 GiB (`fetch_reach.rs`).
    if fetch_reach::fetches_stage_in(pipeline) {
        return Ok(None);
    }
    let Some((variant, plan)) = pipeline.replay_variant(shape)? else {
        return Ok(None);
    };
    // The variant runs threadgroups the device holds; a command runs nothing past Metal's limit
    // for the state alone, since the kernel does not observe Metal's threadgroups.
    let threads = [
        FLAT_THREADS
            .min(pipeline.max_total_threads_per_threadgroup())
            .max(1),
        1,
        1,
    ];
    let max = pipeline.metal_max_total_threads_per_threadgroup();
    let Some(commands) = refusal(plan.instantiate_replay(state, threads))? else {
        return Ok(None);
    };
    let token = NonZeroU64::new(pipeline.gpu_resource_id().0)
        .ok_or_else(|| internal("an indirect command buffer's state without an ID"))?;
    Ok(Some(Prepared {
        variant,
        plan,
        commands,
        registered: ReplayVariant::flat(token, threads, max),
    }))
}

/// An execution's indirect execution set and indirect commands layout, released through the
/// release door (`release.rs`) when the command buffer that holds them completes.
struct GeneratedObjects {
    /// The device whose extension commands made them: held, so they are released while it lives
    /// whichever of their holders goes last.
    device: Device,
    execution_set: Handle,
    commands_layout: Handle,
}

impl Drop for GeneratedObjects {
    /// Release both through the release door (`release.rs`): once the process is exiting, they
    /// are left to its end.
    fn drop(&mut self) {
        // Made only by a device that has the commands.
        let Some(commands) = self.device.shared().generated_commands else {
            return;
        };
        // SAFETY: no pending command uses the objects (a command buffer holds them until it
        // completes); each is this execution's, released once, by its device's commands.
        unsafe {
            if self.execution_set != 0 {
                release::generated(&commands, Generated::ExecutionSet(self.execution_set));
            }
            if self.commands_layout != 0 {
                release::generated(&commands, Generated::CommandsLayout(self.commands_layout));
            }
        }
    }
}

impl GeneratedObjects {
    /// The execution set of `pipelines` (the first its initial pipeline, each at its variant's
    /// index) and the commands layout of `layout`'s three tokens under `pipeline_layout`, which
    /// every pipeline's layout is defined as, made for explicit preprocessing where `explicit`.
    fn create(
        device: &Device,
        pipelines: &[RawPipeline],
        pipeline_layout: vk::PipelineLayout,
        layout: &GeneratedLayout,
        explicit: bool,
    ) -> Result<GeneratedObjects, Error> {
        let commands = device
            .shared()
            .generated_commands
            .ok_or_else(|| internal("generated objects on a device without the commands"))?;
        let mut objects = GeneratedObjects {
            device: device.clone(),
            execution_set: 0,
            commands_layout: 0,
        };
        let (first, rest) = pipelines
            .split_first()
            .ok_or_else(|| internal("an execution set of no pipeline"))?;
        let count = u32::try_from(pipelines.len()).map_err(|_| internal("too many variants"))?;
        let info = ExecutionSetPipelineInfo::new(first.raw, count);
        // SAFETY: `info` and the initial pipeline are live for the call; dropping `objects`
        // releases what was made on a later failure.
        objects.execution_set =
            unsafe { commands.create_execution_set(&ExecutionSetCreateInfo::new(&info)) }
                .map_err(vulkan)?;
        let writes: Vec<WriteExecutionSetPipeline> = (1..)
            .zip(rest)
            .map(|(index, pipeline)| WriteExecutionSetPipeline::new(index, pipeline.raw))
            .collect();
        if !writes.is_empty() {
            // SAFETY: nothing has used the set yet; every written pipeline is live.
            unsafe { commands.update_execution_set(objects.execution_set, &writes) };
        }
        let set_token = ExecutionSetToken {
            kind: EXECUTION_SET_PIPELINES,
            shader_stages: vk::ShaderStageFlags::COMPUTE,
        };
        let push_token = PushConstantToken {
            update_range: vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: layout.push_bytes,
            },
        };
        let tokens = [
            LayoutToken::new(
                TOKEN_EXECUTION_SET,
                std::ptr::from_ref(&set_token).cast(),
                GeneratedLayout::INDEX_OFFSET,
            ),
            LayoutToken::new(
                TOKEN_PUSH_CONSTANT,
                std::ptr::from_ref(&push_token).cast(),
                GeneratedLayout::PUSH_OFFSET,
            ),
            LayoutToken::new(TOKEN_DISPATCH, std::ptr::null(), layout.dispatch_offset()),
        ];
        let info = LayoutCreateInfo::new(layout.stride, pipeline_layout, &tokens);
        let info = match explicit {
            true => info.explicitly_preprocessed(),
            false => info,
        };
        // SAFETY: `info`, its tokens, and their data are live for the call.
        objects.commands_layout =
            unsafe { commands.create_commands_layout(&info) }.map_err(vulkan)?;
        Ok(objects)
    }

    /// Each command's preprocess memory, by slot (its address and size; `(0, 0)` where the
    /// device needs none), in one buffer: `vkGetGeneratedCommandsMemoryRequirementsEXT` for the
    /// most sequences its execution runs (asked once per distinct count), each at its alignment.
    /// No two executions share any of it.
    fn preprocess_memory(
        &self,
        device: &Device,
        layout: &GeneratedLayout,
    ) -> Result<(Option<Buffer>, Vec<(u64, u64)>), Error> {
        let commands = self
            .device
            .shared()
            .generated_commands
            .ok_or_else(|| internal("generated objects on a device without the commands"))?;
        let mut asked: BTreeMap<u32, vk::MemoryRequirements> = BTreeMap::new();
        let mut placed = Vec::with_capacity(layout.slots as usize);
        let mut total = 0u64;
        let mut alignment = 1u64;
        let mut memory_type_bits = u32::MAX;
        for slot in 0..layout.slots {
            let sequences = layout.max_sequences(slot);
            let requirement = *asked.entry(sequences).or_insert_with(|| {
                let info = MemoryRequirementsInfo::new(
                    self.execution_set,
                    self.commands_layout,
                    sequences,
                );
                // SAFETY: the execution set and commands layout are live, made by these commands.
                unsafe { commands.memory_requirements(&info) }
            });
            let align = requirement.alignment.max(1);
            let at = total.next_multiple_of(align);
            placed.push((at, requirement.size));
            total = at + requirement.size;
            alignment = alignment.max(align);
            if requirement.size != 0 {
                memory_type_bits &= requirement.memory_type_bits;
            }
        }
        if total == 0 {
            return Ok((None, vec![(0, 0); placed.len()]));
        }
        let buffer = device.preprocess_buffer(vk::MemoryRequirements {
            size: total,
            alignment,
            memory_type_bits,
        })?;
        let address = buffer
            .device_address()
            .ok_or_else(|| internal("preprocess memory without a device address"))?;
        if address % alignment != 0 {
            return Err(internal("preprocess memory below its alignment"));
        }
        let reserved = placed
            .into_iter()
            .map(|(at, size)| {
                if size == 0 {
                    (0, 0)
                } else {
                    (address + at, size)
                }
            })
            .collect();
        Ok((Some(buffer), reserved))
    }
}

/// The scratch's initial bytes: the parameters, the resident root table with its sink entry's
/// `host` field at `sink`, the resident roots' extents, the variants, the parent's state and
/// the variants' templates, each at its layout offset; the slot tables, region records,
/// sequences and counts are the preprocess's to write.
fn scratch_bytes(pre: &ReplayPreprocess, sink: u64) -> Result<Vec<u8>, Error> {
    let layout = pre.layout;
    let (Some(replay), Some(generated)) = (&pre.generated, &layout.generated) else {
        return Err(internal("a replay without the generated-commands path"));
    };
    let len = usize::try_from(layout.bytes).map_err(|_| internal("a replay scratch too large"))?;
    let mut bytes = vec![0u8; len];
    let mut resident = pre.resident.bytes.clone();
    let at = RootTable::SINK_HOST_OFFSET as usize;
    resident
        .get_mut(at..at + 8)
        .ok_or_else(|| internal("a resident root table without a sink entry"))?
        .copy_from_slice(&sink.to_le_bytes());
    for (offset, part) in [
        (layout.params_offset(), &pre.params),
        (layout.resident_offset, &resident),
        (layout.extents_offset, &pre.extents),
        (layout.variants_offset, &pre.variants),
        (layout.parent_offset, &pre.parent),
        (generated.templates_offset, &replay.templates),
    ] {
        let offset = offset as usize;
        bytes
            .get_mut(offset..offset + part.len())
            .ok_or_else(|| internal("a replay scratch section past the scratch"))?
            .copy_from_slice(part);
    }
    Ok(bytes)
}

/// Record the total execution `execute` (`total`, with `residency`, the command buffer's)
/// through `commands` into a segment of `command_buffer` (the module's rules).
fn record_total(
    command_buffer: &CommandBuffer,
    execute: &Execute,
    residency: &Residency,
    commands: &Commands,
    total: Total,
) -> Result<RecordedSegment, Error> {
    let device = command_buffer.device().clone();
    let Total {
        prepared,
        pre,
        state,
        live,
    } = total;
    let (Some(replay), Some(layout)) = (&pre.generated, pre.layout.generated) else {
        return Err(internal("a replay without the generated-commands path"));
    };
    // Each variant's indirect-bindable pipeline, under a layout defined as the shared one.
    let mut objects = Vec::with_capacity(prepared.len());
    let mut pipelines = Vec::with_capacity(prepared.len());
    for p in &prepared {
        let [region] = p.commands.regions.as_slice() else {
            return Err(internal("a replay variant of other than one region"));
        };
        let made = Arc::new(VariantObjects::create(
            &device,
            &replay.pipeline_layout,
            std::slice::from_ref(&p.variant.pipeline().stage),
            CompiledPipeline::Kernel(p.variant.pipeline()),
        )?);
        pipelines.push(made.indirect_bindable_pipeline(region)?);
        objects.push(made);
    }
    let pipeline_layout = objects
        .first()
        .ok_or_else(|| internal("a replay of no variant"))?
        .layout();
    // Where the driver's own preprocessing of an execution after another runs the wrong shader,
    // every execution is preprocessed apart before the first runs.
    let explicit = preprocess_apart(&device.properties().description);
    let generated =
        GeneratedObjects::create(&device, &pipelines, pipeline_layout, &layout, explicit)?;
    let (memory, reserved) = generated.preprocess_memory(&device, &layout)?;
    let kernel = device.owned_kernel(OwnedModule::ReplayPreprocess)?;
    let no_address = || internal("a replay on a device without device addresses");
    let storage = execute.icb.storage();
    let records = storage.device_address().ok_or_else(no_address)?;
    let (scratch, status) = {
        let mut cb = command_buffer.state();
        let sink = cb
            .uploads
            .upload(&device, &[], u64::from(NeutralResource::SINK_BYTES))?
            .device_address()
            .ok_or_else(no_address)?;
        let bytes = scratch_bytes(&pre, sink)?;
        let scratch = cb.uploads.upload(&device, &bytes, pre.layout.bytes)?;
        let status = cb.status_address(&device)?.ok_or_else(no_address)?;
        (scratch.device_address().ok_or_else(no_address)?, status)
    };
    let mut push = pre.push_constants.clone();
    for hole in &pre.push_holes {
        let (offset, address) = match *hole {
            PushHole::RootTableAddress { offset, .. } => {
                (offset, scratch + pre.layout.resident_offset)
            }
            PushHole::ExecutionStatusAddress { offset, .. } => (offset, status),
            PushHole::CommandRecordsAddress { offset, .. } => (offset, records),
            _ => {
                return Err(internal(
                    "a replay preprocess hole the runtime does not fill",
                ))
            }
        };
        push.get_mut(offset as usize..offset as usize + 8)
            .ok_or_else(|| internal("a replay preprocess hole past its push bytes"))?
            .copy_from_slice(&address.to_le_bytes());
    }
    // What the executions use: the resident resources each variant's plan reaches.
    let empty = ArgumentTable::default();
    let stages = [(Stage::Kernel, &empty)];
    let tables = binding::Tables::new(&stages).with_residency(residency);
    let none = binding::Command {
        descriptor_bytes: &[],
        descriptor_ranges: &[],
        root_tables: Vec::new(),
        push_holes: &[],
        uses: &[],
        blobs: Vec::new(),
        mesh: None,
        tessellation: None,
    };
    let mut uses = Vec::new();
    for p in &prepared {
        // Each names what it uses: the replay's resident set is resolved here, at the commit.
        let command = binding::command_uses(
            "execute_commands_in_buffer",
            &p.plan,
            &none,
            &tables,
            None,
            None,
        )?;
        uses.extend(command.iter().cloned());
    }
    let preprocess_uses = [reads(storage, 0..storage.length())];

    let mut core = EncoderCore::segment(command_buffer, execute.ordered)?;
    let compute = vk::PipelineStageFlags2::COMPUTE_SHADER;
    let group_count = pre.group_count;
    core.command(compute, &preprocess_uses, |raw, command_buffer| {
        // SAFETY: the segment's command buffer is recording; the kernel is the device's, and
        // its push bytes name the scratch, the status words and the records, which live until
        // the command buffer completes.
        unsafe {
            raw.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            raw.cmd_push_constants(
                command_buffer,
                kernel.layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &push,
            );
            let [x, y, z] = group_count;
            raw.cmd_dispatch(command_buffer, x, y, z);
        }
    });
    // The preprocess's writes before the executions' preprocessing, indirect and shader reads.
    core.barrier(Barrier::everything());
    let zeros = vec![0u8; layout.push_bytes as usize];
    let initial = pipelines[0].raw;
    let (execution_set, commands_layout) = (generated.execution_set, generated.commands_layout);
    // Each command's execution: its sequences, its count and its own preprocess memory.
    let execution = |slot: u32| {
        let sequences = layout.max_sequences(slot);
        GeneratedExecution {
            indirect: (
                scratch + layout.sequence_offset(slot, 0),
                u64::from(layout.stride) * u64::from(sequences),
            ),
            preprocess: reserved[slot as usize],
            max_sequences: sequences,
            count_address: scratch + layout.count_offset(slot),
        }
    };
    core.command(compute, &uses, |raw, command_buffer| {
        // The state every preprocessing and execution takes: the execution set's initial
        // pipeline and zeroed push constants (the tokens write every byte). An execution leaves
        // the bound pipeline and push constants undefined.
        let bind = || {
            // SAFETY: the segment's command buffer is recording; the pipeline and its layout are
            // held until the command buffer completes.
            unsafe {
                raw.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, initial);
                raw.cmd_push_constants(
                    command_buffer,
                    pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &zeros,
                );
            }
        };
        // SAFETY: as above; every object the executions name is held until the command buffer
        // completes, and each command's preprocess memory is its own. Where `explicit`, the
        // layout was made for explicit preprocessing, every execution is preprocessed with the
        // state it then runs with, and a barrier orders those writes before every execution.
        unsafe {
            if explicit {
                bind();
                for slot in 0..layout.slots {
                    let info = GeneratedCommandsInfo::new(
                        execution_set,
                        commands_layout,
                        &execution(slot),
                    );
                    commands.preprocess(command_buffer, &info);
                }
                Barrier::everything().record(raw, command_buffer);
            }
            for slot in 0..layout.slots {
                if slot > 0 {
                    Barrier::between_dispatches().record(raw, command_buffer);
                }
                bind();
                let info =
                    GeneratedCommandsInfo::new(execution_set, commands_layout, &execution(slot));
                match explicit {
                    true => commands.execute_preprocessed(command_buffer, &info),
                    false => commands.execute(command_buffer, &info),
                }
            }
        }
    });
    for (p, made) in prepared.iter().zip(&pipelines) {
        core.executed(|| {
            Execution::new(
                ExecutedPipeline::Compute(Arc::clone(p.variant.shared_pipeline())),
                Arc::new(state.shape()),
                vec![made.created_in],
            )
        });
    }
    // The execution binds each variant for its commands, which the GPU writes: the kinds a
    // variant's making announces at every occurrence, once an execution.
    for (p, made) in prepared.iter().zip(&objects) {
        if let [region] = p.commands.regions.as_slice() {
            made.announce_bound(
                (
                    std::slice::from_ref(&p.variant.pipeline().stage),
                    CompiledPipeline::Kernel(p.variant.pipeline()),
                ),
                [(&region.specialization, None)],
            );
        }
        // The GPU's commands bind the textures: every slot whose `half` writes read its word.
        device.degrade_half_texel_writes(&p.variant.pipeline().dependencies, None);
        device.degrade_bound_formats(std::slice::from_ref(&p.variant.pipeline().stage), None);
        // The values the variant's plan pushes for the commands the GPU writes.
        for (n, region) in p.commands.regions.iter().enumerate() {
            if p.commands.regions[..n]
                .iter()
                .all(|earlier| earlier.specialization != region.specialization)
            {
                device.degrade_bound_values(
                    std::slice::from_ref(&p.variant.pipeline().stage),
                    &region.specialization,
                    &|k| p.plan.specialized_value(k),
                    None,
                );
            }
        }
        let stage = &p.variant.pipeline().stage;
        if stage
            .precision_notes
            .contains(&metal2vulkan::pipeline::PrecisionNote::ArgumentPixelSamplerScaling)
        {
            device.degrade_argument_pixel_samplers(
                std::slice::from_ref(stage),
                &device.argument_samplers(),
            );
        }
    }
    core.retain(generated);
    // Its resident set, resolved now: the command buffer's first submission finds whether the
    // live buffers and what its encoders made resident still lie there, holding the table from
    // then on; until then a release or a rebind lets their pages go.
    core.resolved(live.ranges().chain(residency.resident_ranges()));
    drop(live);
    core.retain(memory);
    core.retain(pipelines);
    core.retain(objects);
    let variants: Vec<Arc<Variant>> = prepared.into_iter().map(|p| p.variant).collect();
    core.retain(variants);
    core.end_segment()
}
