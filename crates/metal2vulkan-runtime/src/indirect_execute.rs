//! Executing an indirect command buffer's compute commands in a compute encoder
//! (`executeCommandsInBuffer:withRange:`).
//!
//! - **When.** The CPU may write a command's record until the command buffer is committed, and
//!   an execution runs the records as they stand then. So the encoder records nothing for it at
//!   the call: it leaves an [`Execute`] as a segment ([`EncoderCore::defer`]), recorded into a
//!   `VkCommandBuffer` of its own ([`Execute::record`]), behind a barrier from everything before
//!   it and with one to everything after it, which goes between the encoder's two as its batch
//!   is submitted.
//! - **Records the GPU may write**. A buffer an encoder made resident for writing
//!   (`IndirectCommandBuffer::gpu_written`, for good) is written by functions as they run, so its
//!   records are known only once the work before the execution completed. Its segment starts a part
//!   of the encoder, and a batch, of its own; the scheduler submits the batch once every batch
//!   submitted before it on any queue completed and its host step recorded the segment from the
//!   records then (`scheduler`), which the device announces once
//!   (`Degradation::HostReplayedIndirectCommands`). So is a segment `commit` recorded from the
//!   CPU's records whose buffer an encoder made resident for writing after the commit
//!   ([`Unforeseen`]). The step compiles nothing: `commit`, or for such a segment the encoder's
//!   `use_resource` before the scheduler sees the buffer written, settles on its own thread the
//!   variants every state a record may name needs, and the neutrals their plans bind
//!   ([`Execute::settle`]); the step makes only what a dispatch on the CPU makes (descriptor sets,
//!   upload chunks, views and samplers of bound textures) and the `VkPipeline` of a threadgroup
//!   size or threadgroup lengths a state has not dispatched before, from its compiled variant. A
//!   record names its pipeline state by its ID among the states made with
//!   `supportIndirectCommandBuffers` that lived at the commit (`ComputePipelineState::
//!   indirect_states`); an ID of none runs nothing, as Metal's undefined one may not. A record
//!   naming the ID of the state the CPU set on its command runs that state's program, though the
//!   program released the state since (`IndirectCommandBuffer::step_states`).
//! - **Each command** runs through the plan path as the encoder's dispatch would
//!   ([`record_dispatch`]): with its pipeline state, or the encoder's when the buffer inherits
//!   it; its kernel buffers, or the encoder's bound ones when the buffer inherits them; its
//!   threadgroup lengths, stage-in region (all zero: none set) and imageblock size (zero: 1 by
//!   1, as the encoder's until set); and the command buffer's resident resources, every
//!   encoder's, as the commit leaves them (`residency.rs`). A kernel buffer's attribute stride past
//!   32 bits, at an index a stage-in layout of `MTLBufferLayoutStrideDynamic` reads, is refused
//!   (`Unsupported{KernelStageIn, NotImplemented}`): Metal's answer is not measured. So is a kernel
//!   buffer set without a stride there (`Invalid`,
//!   `IndirectCommandBufferProblem::UnstridedBuffer`): Metal reads it at stride 2^32 - 1 with no
//!   bounds check, so past element 0 its answer is whatever memory lies 4 GiB on, where an
//!   inheriting buffer's command reads the encoder's buffer set without one at stride 0, as the
//!   encoder's own dispatch does. A command with no dispatch, no pipeline state, or a threadgroup
//!   that is empty or holds more threads than the state's `max_total_threads_per_threadgroup` runs
//!   nothing, as Metal's runs nothing, and its neighbours run (ruling a). A
//!   command whose dispatch the plan path refuses refuses the commit, with that error, or fails the
//!   command buffer where the host step records it (`CommandBufferError::Refused`, the same error
//!   typed in `CommandBuffer::refusal`).
//! - **Faults** (the host step's, as the device replay reports them, `bind::ReplayFaults`). A
//!   header that holds the producers' sticky encoding fault runs none of the commands
//!   (`ENCODING`); a record that does not decode, or sets what its descriptor does not allow,
//!   runs nothing (`RECORD`). The command buffer's execution report states them.
//! - **Buffers.** A command names each buffer by a Metal address, as Metal's does. The address
//!   resolves against the resources the command buffer's encoders made resident
//!   (`use_resource(s)`, `use_heap(s)`, before the execution or after it, in its encoder or
//!   another: Metal's residency is the command buffer's, `residency.rs`), then against the
//!   device's live buffers Metal reaches unnamed (`live_buffers.rs`): the buffer it lies in at the
//!   offset it lies at. One in none of them (a no-copy buffer or a heap's buffer no command
//!   buffer's commands used yet, or none) misses as a pointer loaded from memory does: its slot is
//!   unbound, so it reads zero and drops writes, and the command buffer's execution report counts
//!   the miss, one per slot.
//! - **Threadgroup lengths** are kept as set: one past the device's `max_threadgroup_memory_length`
//!   runs clamped to it (ruling c); one that is not a multiple of 16 bytes runs rounded up to one,
//!   as the plan rounds every length.
//! - **Order.** Between commands, the segment orders as the encoder orders its dispatches (a
//!   serial encoder's where their uses meet; a concurrent one's not at all), and a command with
//!   its barrier set runs after every earlier command of the segment.

use super::argument_table::ArgumentTable;
use super::buffer::Buffer;
use super::command_buffer::CommandBuffer;
use super::compute::{
    prepare_variants, record_dispatch, validated_imageblock_size, DispatchTables,
};
use super::degradation::Degradation;
use super::device::lock;
use super::encoder::{EncoderCore, RecordedSegment};
use super::error::{internal, invalid, Error, Problem};
use super::executed_commands::{ExecutedCommand, RecordsRead};
use super::indirect_command_buffer::{IndirectCommandBuffer, IndirectCommandBufferProblem, Reader};
use super::pipeline::{BufferForms, ComputePipelineState, IndirectStates, Variant};
use super::predication::HostGate;
use super::residency::Residency;
use super::tracker::Barrier;
use super::Device;
use metal2vulkan::bind::indirect_commands::{Action, BufferBinding, BufferStage, CommandRecord};
use metal2vulkan::bind::{Dispatch, ImageblockSize, ReplayFaults, StageInRegion};
use metal2vulkan::error::{BindError, Feature, Unsupported};
use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::{Arc, Mutex};

/// The call an execution's refusals name.
pub(super) const CALL: &str = "execute_commands_in_buffer";

/// The call an execution of a range read from a buffer names.
pub(super) const CALL_INDIRECT: &str = "execute_commands_in_buffer_indirect";

/// The bytes of an `MTLIndirectCommandBufferExecutionRange`: `location`, then `length`.
pub(super) const RANGE_BYTES: u64 = 8;

/// Refuse the execution range at `offset` of a buffer of `length` bytes where Metal's reading of
/// it is not defined (`execute_commands_in_buffer_indirect`; where API
/// validation stops neither): an offset that is not a multiple of 4 bytes, at which Metal's GPU
/// faulted the command buffer, and a range whose 8 bytes run past the buffer, which Metal reads
/// from whatever follows it.
pub(super) fn check_range_buffer(
    call: &'static str,
    length: u64,
    offset: u64,
) -> Result<(), Error> {
    let past = offset
        .checked_add(RANGE_BYTES)
        .is_none_or(|end| end > length);
    if !offset.is_multiple_of(4) || past {
        return Err(invalid(
            call,
            Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::RangeBuffer {
                offset,
                length,
            }),
        ));
    }
    Ok(())
}

/// An execution of an indirect command buffer's commands, left to record later.
pub(super) struct Execute {
    pub(super) icb: IndirectCommandBuffer,
    /// The commands it runs.
    pub(super) range: ExecutionRange,
    /// The encoder's pipeline state, where the buffer inherits it.
    pub(super) pipeline: Option<ComputePipelineState>,
    /// The encoder's bound buffers where the buffer inherits them; else empty.
    pub(super) table: ArgumentTable,
    /// Whether the encoder runs its dispatches in encoded order.
    pub(super) ordered: bool,
    /// The check of the block of command predication it was encoded in, if any
    /// (`predication.rs`): a host step records it, and nothing where the counter skips.
    pub(super) gate: Option<HostGate>,
}

/// The commands an execution runs.
pub(super) enum ExecutionRange {
    /// `executeCommandsInBuffer:withRange:`'s, inside the buffer's size.
    Fixed(Range<u32>),
    /// `executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:`'s: a copy of the
    /// `MTLIndirectCommandBufferExecutionRange` (`location`, then `length`, each a `u32`), made
    /// where the execution is encoded behind a barrier from everything before
    /// (`indirect_dispatch::copy_arguments`), which the host step reads.
    Indirect(Buffer),
}

impl ExecutionRange {
    /// Whether the range is read from a buffer, which the GPU may write
    /// ([`ExecutionRange::Indirect`]): a host step records its execution, whoever wrote the
    /// records.
    pub(super) fn indirect(&self) -> bool {
        matches!(self, ExecutionRange::Indirect(_))
    }

    /// The commands it runs of a buffer of `size` commands: the range given, or the range the
    /// copy holds now as Metal's execution takes it (an M3 Max and an M4, for a compute encoder; a
    /// probe, the same chips, for a render encoder): from `location` to `location + length`,
    /// clamped at the buffer's size, so a location of 2 and a length of 5 in a buffer of 4 run
    /// commands 2 and 3, and a length of 0 or a location at the size runs nothing. Where the
    /// location lies past the size, or `location + length` passes `2^32 - 1` (Metal's GPU then
    /// faults the command buffer, or hangs), it runs nothing and reports `ReplayFaults::RANGE`.
    pub(super) fn commands(&self, size: u64) -> Result<(Range<u32>, ReplayFaults), Error> {
        let copy = match self {
            ExecutionRange::Fixed(range) => return Ok((range.clone(), ReplayFaults::default())),
            ExecutionRange::Indirect(copy) => copy,
        };
        let contents = copy
            .contents()
            .ok_or_else(|| internal("an execution range's copy without host bytes"))?;
        let mut bytes = [0u8; RANGE_BYTES as usize];
        // SAFETY: the copy is a `Shared` buffer of at least `RANGE_BYTES`, mapped; the GPU's
        // write to it completed before the host step (its batch ended with the barrier to the
        // host), and nothing writes it after.
        unsafe { std::ptr::copy_nonoverlapping(contents.as_ptr(), bytes.as_mut_ptr(), 8) };
        let [l0, l1, l2, l3, n0, n1, n2, n3] = bytes;
        let location = u64::from(u32::from_le_bytes([l0, l1, l2, l3]));
        let length = u64::from(u32::from_le_bytes([n0, n1, n2, n3]));
        if location > size || location + length > u64::from(u32::MAX) {
            return Ok((0..0, ReplayFaults::RANGE));
        }
        let end = (location + length).min(size);
        // Both lie within the size, which a buffer's `u32` command count holds.
        Ok((location as u32..end as u32, ReplayFaults::default()))
    }
}

/// What `commit` settled for an execution's host step ([`Execute::settle`]).
pub(super) struct Settled {
    /// The pipeline states its records may name.
    pub(super) states: Vec<ComputePipelineState>,
    /// The variants they need, held until the command buffer completes, so they stay live.
    pub(super) variants: Vec<Arc<Variant>>,
}

/// An execution `commit` recorded from the CPU's records. An encoder that makes its buffer resident
/// for writing before its batch is submitted settles its host step then, on that encoder's thread
/// and before the scheduler sees the buffer written (`IndirectCommandBuffer::written_by_gpu`), so
/// the step translates nothing.
pub(super) struct Unforeseen {
    execute: Arc<Execute>,
    /// The states made with `supportIndirectCommandBuffers` at the commit, held weakly: those its
    /// records may name (one made since names none); `None` where the buffer inherits the
    /// encoder's.
    listed: Option<IndirectStates>,
    /// Its command buffer's residency at the commit (`residency.rs`), without the live buffers.
    residency: Residency,
    /// What the settle made, once it ran.
    settled: Mutex<Option<Settled>>,
}

impl Unforeseen {
    /// The execution `execute` on `device`, as its commit leaves it, with its command buffer's
    /// `residency`.
    pub(super) fn new(
        device: &Device,
        execute: &Arc<Execute>,
        residency: Residency,
    ) -> Arc<Unforeseen> {
        let listed = execute
            .pipeline
            .is_none()
            .then(|| IndirectStates::list(device));
        Arc::new(Unforeseen {
            execute: Arc::clone(execute),
            listed,
            // The settle states variants of it alone: it holds no table of the live buffers.
            residency: residency.without_live(),
            settled: Mutex::new(None),
        })
    }

    /// Whether its host step was settled.
    pub(super) fn settled(&self) -> bool {
        lock(&self.settled).is_some()
    }

    /// Settle its host step on `device` ([`Execute::settle`]), unless it was: for the states
    /// listed at the commit that still live, or the encoder's.
    pub(super) fn settle(&self, device: &Device) {
        let mut settled = lock(&self.settled);
        if settled.is_none() {
            let states = self.live(device);
            *settled = Some(self.execute.settle(device, states, &self.residency));
        }
    }

    /// The states its host step lets the records name: those settled, else those listed at the
    /// commit that still live, or the encoder's.
    pub(super) fn states(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &*lock(&self.settled) {
            Some(settled) => settled.states.clone(),
            None => self.live(device),
        }
    }

    fn live(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &self.listed {
            Some(listed) => listed.live(),
            None => self.execute.states(device),
        }
    }
}

/// A command to run: its record, and the pipeline state it names (`None`: none, or the
/// encoder's where the buffer inherits it).
type Command = (CommandRecord, Option<ComputePipelineState>);

impl Execute {
    /// Record the commands into a segment of `command_buffer` (the module's rules): from the
    /// CPU's records as they stand now, or, for the host step, from the records as the GPU and
    /// the CPU left them, each naming its state among `states`.
    pub(super) fn record(
        &self,
        command_buffer: &CommandBuffer,
        states: Option<&[ComputePipelineState]>,
    ) -> Result<RecordedSegment, Error> {
        let mut core = EncoderCore::segment(command_buffer, self.ordered)?;
        // Its host step inside a block of command predication that skips runs nothing.
        if states.is_some() && self.gate.as_ref().is_some_and(HostGate::skips) {
            return core.end_segment();
        }
        let max_length = max_length(&core);
        // The host step's commands take the large-buffer form, whose variants `settle` made.
        let forms = match states {
            Some(_) => BufferForms::Large,
            None => BufferForms::Bound,
        };
        let (commands, faults) = self.commands(states)?;
        // Every encoder's resident resources, as the command buffer's are at the commit.
        let residency = command_buffer.residency()?.kernel();
        // What the commands run, for the command buffer's handlers (`executed_commands.rs`).
        let read = match states {
            Some(_) => RecordsRead::HostStep,
            None => RecordsRead::Commit,
        };
        command_buffer.report_executed(&self.icb, read, &residency, || {
            let first = self.range().map_or(0, |(range, _)| u64::from(range.start));
            let commands = commands
                .iter()
                .enumerate()
                .filter_map(|(at, command)| {
                    let (record, own) = command.as_ref()?;
                    let (pipeline, _) = self.runnable(record, own.as_ref())?;
                    Some(ExecutedCommand {
                        index: first + at as u64,
                        can_encode_indirect_commands: pipeline.can_encode_indirect_commands(),
                    })
                })
                .collect();
            (commands, false)
        });
        let mut misses = 0u32;
        let mut ran = false;
        for (record, own) in commands.iter().flatten() {
            let Some((pipeline, dispatch)) = self.runnable(record, own.as_ref()) else {
                continue;
            };
            if record.barrier && ran {
                core.barrier(Barrier::between_dispatches());
            }
            let bindings = self.bindings(record, max_length, &residency, &mut misses);
            // A stride past 32 bits that a stage-in layout of `MTLBufferLayoutStrideDynamic`
            // reads: Metal's answer is not measured, and the fetch's `u32` displacement cannot
            // place it.
            if bindings
                .wide_strides
                .iter()
                .any(|&index| pipeline.reads_stage_in_stride(index))
            {
                return Err(Error::Bind(BindError::Unsupported(
                    Unsupported::not_implemented(Feature::KernelStageIn),
                )));
            }
            // A buffer set without a stride that such a layout reads: Metal reads it at stride
            // 2^32 - 1, past element 0 whatever memory lies 4 GiB on.
            if let Some(&index) = bindings
                .unstrided
                .iter()
                .find(|&&index| pipeline.reads_stage_in_stride(index))
            {
                return Err(invalid(
                    CALL,
                    Problem::IndirectCommandBuffer(IndirectCommandBufferProblem::UnstridedBuffer {
                        index,
                    }),
                ));
            }
            let tables = bindings.tables(self, pipeline, record, &residency, forms)?;
            record_dispatch(&mut core, &tables, CALL, dispatch)?;
            ran = true;
        }
        let mut segment = core.end_segment()?;
        segment.resident_misses = misses;
        segment.faults = faults;
        Ok(segment)
    }

    /// Settle, on the thread that commits or flags the buffer, what the host step needs to
    /// translate nothing, for `states`, those the records may name: each one's variants in the
    /// large-buffer form (`BufferForms::Large`: one whatever buffers a command binds, and its
    /// emulated-simdgroups twin where the kernel has one), and the device's neutrals their plans
    /// may bind. On a device without device addresses a variant still follows the bound
    /// buffers' lengths, so the step translates the variant of a shape it meets first. What
    /// cannot be settled now (a state the plan path refuses) is left to the step, which meets it
    /// as it is then.
    pub(super) fn settle(
        &self,
        device: &Device,
        states: Vec<ComputePipelineState>,
        residency: &Residency,
    ) -> Settled {
        let empty = ArgumentTable::default();
        let lengths = BTreeMap::new();
        let mut variants = Vec::new();
        for state in &states {
            let tables = DispatchTables {
                pipeline: state,
                table: if self.icb.descriptor().inherit_buffers {
                    &self.table
                } else {
                    &empty
                },
                threadgroup_lengths: &lengths,
                imageblock: ImageblockSize::new(1, 1),
                stage_in_region: None,
                residency,
                forms: BufferForms::Large,
            };
            variants.extend(prepare_variants(device, &tables).into_iter().flatten());
        }
        Settled { states, variants }
    }

    /// Whether it reads its range from a buffer, which the GPU may write
    /// ([`ExecutionRange::indirect`]): a host step records it, whoever wrote its records.
    pub(super) fn indirect(&self) -> bool {
        self.range.indirect()
    }

    /// Whether it lies inside a block of command predication: a host step records it, once the
    /// steps before it left their counter (`predication.rs`).
    pub(super) fn predicated(&self) -> bool {
        self.gate.is_some()
    }

    /// The commands it runs ([`ExecutionRange::commands`]).
    fn range(&self) -> Result<(Range<u32>, ReplayFaults), Error> {
        self.range.commands(self.icb.size())
    }

    /// The pipeline states the records may name now: the encoder's where the buffer inherits
    /// it, else every state of `device` made with `supportIndirectCommandBuffers` alive now.
    pub(super) fn states(&self, device: &Device) -> Vec<ComputePipelineState> {
        match &self.pipeline {
            Some(pipeline) => vec![pipeline.clone()],
            None => ComputePipelineState::indirect_states(device),
        }
    }

    /// The commands of the range, in order, `None` where a record runs nothing for a fault, and
    /// the faults: the CPU's records, or the records as the GPU and the CPU left them, each
    /// naming its state among `states`.
    fn commands(
        &self,
        states: Option<&[ComputePipelineState]>,
    ) -> Result<(Vec<Option<Command>>, ReplayFaults), Error> {
        let (range, faults) = self.range()?;
        // The execution announces the commands it runs otherwise than Metal's may, as it records
        // them.
        let Some(states) = states else {
            let (commands, released) = self.icb.records(range, Reader::Commit)?;
            self.icb.degrade_released(&released);
            return Ok((commands.into_iter().map(Some).collect(), faults));
        };
        let written = self.icb.written_records(range.clone())?;
        // A record naming the ID of the state the CPU set runs its program, else the state of its
        // ID among `states` (`IndirectCommandBuffer::step_states`).
        let (named, released) = self.icb.step_states(range.start, &written.records, states);
        self.icb.degrade_released(&released);
        let commands = written
            .records
            .into_iter()
            .zip(named)
            .map(|(record, state)| record.map(|record| (record, state)))
            .collect();
        Ok((commands, ReplayFaults(written.faults.0 | faults.0)))
    }

    /// The pipeline state and dispatch of `record`, which names `own`, where it runs on Metal:
    /// it dispatches, has a pipeline state, and its threadgroup holds at least one thread and no
    /// more than the state's `max_total_threads_per_threadgroup` (ruling a).
    fn runnable<'a>(
        &'a self,
        record: &CommandRecord,
        own: Option<&'a ComputePipelineState>,
    ) -> Option<(&'a ComputePipelineState, Dispatch)> {
        let dispatch = dispatch_of(record)?;
        let pipeline = self.pipeline.as_ref().or(own)?;
        runs(pipeline, &dispatch).then_some((pipeline, dispatch))
    }

    /// The command's own buffers, resolved against `residency`, the command buffer's (each miss
    /// counted in `misses`), and its threadgroup lengths, clamped to `max_length`.
    fn bindings(
        &self,
        record: &CommandRecord,
        max_length: u32,
        residency: &Residency,
        misses: &mut u32,
    ) -> Bindings {
        let descriptor = *self.icb.descriptor();
        let mut table = ArgumentTable::default();
        let mut wide_strides = Vec::new();
        let mut unstrided = Vec::new();
        if !descriptor.inherit_buffers {
            let bindings = record.bindings.buffers(BufferStage::Kernel);
            for (index, binding) in (0..descriptor.max_kernel_buffer_bind_count).zip(bindings) {
                let Some(binding) = binding else {
                    continue;
                };
                if !binding.names_buffer() {
                    degrade_nil_offset(self.icb.device(), binding, index);
                    continue;
                }
                let Some(bound) = residency.resolve(binding.address) else {
                    *misses = misses.wrapping_add(1);
                    continue;
                };
                table.buffers.insert(index, bound);
                // Only a stage-in layout of `MTLBufferLayoutStrideDynamic` reads the stride;
                // `record` refuses one past 32 bits there.
                match binding.attribute_stride.map(u32::try_from) {
                    Some(Ok(stride)) => {
                        table.attribute_strides.insert(index, stride);
                    }
                    Some(Err(_)) => wide_strides.push(index),
                    None => unstrided.push(index),
                }
            }
        }
        let lengths = (0..descriptor.max_kernel_threadgroup_memory_bind_count)
            .zip(record.threadgroup_memory_lengths)
            .filter(|&(_, length)| length != 0)
            .map(|(index, length)| (index, length.min(max_length)))
            .collect();
        Bindings {
            table,
            lengths,
            wide_strides,
            unstrided,
        }
    }
}

/// Announce `Degradation::IndirectNilBufferOffset` for a command's `binding` at `index`, a nil
/// buffer's, where it was set at an offset whose low 32 bits are not 0: Metal's function sees
/// them as the buffer's address, the host replay's sees no buffer.
pub(super) fn degrade_nil_offset(device: &Device, binding: &BufferBinding, index: u32) {
    if binding.address == 0 {
        return;
    }
    device.degrade(Degradation::IndirectNilBufferOffset, || {
        (
            format!(
                "an indirect command's nil buffer at index {index}, set at offset {:#x}",
                binding.address
            ),
            "Metal's function sees the offset as the buffer's address; the execution, replayed \
             on the host, binds no buffer there"
                .to_owned(),
        )
    });
}

/// Announce, once per device, that an execution is replayed on the host (the module's records
/// the GPU may write, and ranges read from a buffer).
pub(super) fn degrade_host_replay(device: &Device, execute: &Execute) {
    device.degrade(Degradation::HostReplayedIndirectCommands, || {
        let why = if execute.icb.gpu_written() {
            "an encoder made it resident for writing, so the GPU may write the records an \
             execution runs"
        } else if !execute.indirect() {
            "an execution inside a block of command predication, whose condition is GPU memory"
        } else {
            "an execution reads its range from a buffer \
             (executeCommandsInBuffer:indirectBuffer:indirectBufferOffset:), which the GPU may \
             write"
        };
        (
            format!(
                "an indirect command buffer of {} commands",
                execute.icb.size()
            ),
            why.to_owned(),
        )
    });
}

/// A command's own buffers and threadgroup lengths ([`Execute::bindings`]).
struct Bindings {
    table: ArgumentTable,
    lengths: BTreeMap<u32, u32>,
    /// The kernel buffer indices bound with a stride past 32 bits.
    wide_strides: Vec<u32>,
    /// The kernel buffer indices bound without a stride.
    unstrided: Vec<u32>,
}

impl Bindings {
    /// The tables a command of `execute` with `record` dispatches `pipeline` with, its variant
    /// in `forms`: the encoder's buffers where the buffer inherits them, else its own. Refused as
    /// `setImageblockWidth:height:` refuses its imageblock size.
    fn tables<'a>(
        &'a self,
        execute: &'a Execute,
        pipeline: &'a ComputePipelineState,
        record: &CommandRecord,
        residency: &'a Residency,
        forms: BufferForms,
    ) -> Result<DispatchTables<'a>, Error> {
        let table = if execute.icb.descriptor().inherit_buffers {
            &execute.table
        } else {
            &self.table
        };
        Ok(DispatchTables {
            pipeline,
            table,
            threadgroup_lengths: &self.lengths,
            imageblock: imageblock(pipeline, record.imageblock_size)?,
            stage_in_region: stage_in_region(record.stage_in_region),
            residency,
            forms,
        })
    }
}

/// The device's `max_threadgroup_memory_length`, as a record's lengths are clamped to it.
fn max_length(core: &EncoderCore) -> u32 {
    let max_length = core.device().properties().max_threadgroup_memory_length;
    u32::try_from(max_length).unwrap_or(u32::MAX)
}

/// The dispatch `record` makes; `None` for a command that dispatches nothing.
fn dispatch_of(record: &CommandRecord) -> Option<Dispatch> {
    match record.action {
        Action::DispatchThreadgroups { grid, local_size } => Some(Dispatch::Threadgroups {
            groups: grid,
            threads_per_group: local_size,
        }),
        Action::DispatchThreads { grid, local_size } => Some(Dispatch::Threads {
            threads: grid,
            threads_per_group: local_size,
        }),
        Action::Empty
        | Action::DrawPrimitives { .. }
        | Action::DrawIndexedPrimitives { .. }
        | Action::DrawPatches { .. }
        | Action::DrawIndexedPatches { .. }
        | Action::DrawMeshThreadgroups { .. }
        | Action::DrawMeshThreads { .. } => None,
    }
}

/// Whether `dispatch` of `pipeline` runs on Metal: its threadgroup holds at least one thread and
/// no more than Metal's limit for the state (ruling a). One within it but past the state's
/// `max_total_threads_per_threadgroup`, where this device's limit is lower, runs on Metal: its
/// plan refuses it typed as the device's gap.
fn runs(pipeline: &ComputePipelineState, dispatch: &Dispatch) -> bool {
    let threads: u64 = dispatch
        .threads_per_group()
        .iter()
        .map(|&axis| u64::from(axis))
        .product();
    threads != 0 && threads <= u64::from(pipeline.metal_max_total_threads_per_threadgroup())
}

/// The imageblock size a command of `pipeline` runs with: 1 by 1 for a kernel without an
/// explicit imageblock or a record that sets none (both zero); else the size set, refused as
/// Metal's API validation refuses `setImageblockWidth:height:` (an indirect command's plain
/// answer is unmeasured, where the encoder's is measured).
fn imageblock(
    pipeline: &ComputePipelineState,
    [width, height]: [u16; 2],
) -> Result<ImageblockSize, Error> {
    if pipeline.imageblock_cell_bytes().is_none() || (width == 0 && height == 0) {
        return Ok(ImageblockSize::new(1, 1));
    }
    validated_imageblock_size(width.into(), height.into()).map_err(|problem| invalid(CALL, problem))
}

/// The stage-in region a record sets: origin, then size; `None` when all zero (none set).
fn stage_in_region([x, y, z, width, height, depth]: [u32; 6]) -> Option<StageInRegion> {
    let region = [x, y, z, width, height, depth];
    (region != [0; 6]).then(|| StageInRegion::new([x, y, z], [width, height, depth]))
}
