//! Command predication: Metal's private compute-encoder calls
//! `encodeStartIf:offset:comparison:referenceValue:`, `encodeStartElse` and `encodeEndIf`, and its
//! loops, `encodeStartWhile:offset:comparison:referenceValue:` to `encodeEndWhile` and
//! `encodeStartDoWhile` to `encodeEndDoWhile:offset:comparison:referenceValue:`, as Metal runs
//! them (measured on an M4 under macOS 27.0.1).
//!
//! # What Metal does
//!
//! - **A unit** of predication is a run of a command buffer's compute encoders of one dispatch
//!   type with nothing between them. A blit, render or resource state encoder, an event wait or
//!   signal, an encoder of the other dispatch type, or the command buffer's end ends it; a
//!   scheduled handler does not (nor does a binary semaphore wait or signal, which Metal has
//!   not). A block may span its unit's encoders; a record after the unit's end is the next
//!   unit's.
//! - **The condition** is `word OP reference` (`PredicationCompare`, Metal's codes 0 to 5, and
//!   `==` for any other code), the word the `u32` at the offset rounded down to a multiple of 4,
//!   read when the GPU reaches the record: after the work before it, with or without a barrier,
//!   and after a CPU write made before then.
//! - **A skipped region** runs none of its dispatches, of any form, and none of its executions
//!   of an indirect command buffer. Setters, barriers, resource uses and fences take effect in
//!   encode order whatever runs.
//! - **Nesting** has no limit (2^20 levels measured).
//! - **A loop** runs its commands again and again: a While while its condition holds, tested
//!   before each pass, a DoWhile once and again while its End's condition holds, tested after
//!   each pass; each word read when the GPU reaches the test. Each pass runs every command with
//!   the encoder's state it was encoded with (its bindings and bytes), and the state the body's
//!   last setters left stands after the loop. Metal bounds no loop (16 million passes ran).
//! - **An End** of the open block answers YES.
//! - **A misused record kills** (an End of another kind of block, a second Else or an Else on a
//!   loop, a Start or an `encodeEndDoWhile` on a nil buffer inside a block): the outermost open
//!   block, with every block in it, runs nothing from its start on, the commands of the unit's
//!   earlier encoders included; the record answers NO. Every End then answers NO but the one
//!   that closes that outermost block, which answers YES, and the unit runs again after it. A
//!   further misuse inside it changes nothing.
//! - **An unbalanced record** (an End or Else with no open block, a Start or an
//!   `encodeEndDoWhile` on a nil buffer with none) answers NO and drops its unit's work from the
//!   unit's start up to a point the later records decide (the next Else or End of a block opened
//!   after it, or the start of a DoWhile, with exceptions no rule fitted, and one stream that
//!   ends the process): refused, recording nothing
//!   ([`RuntimeGap::PredicationUnbalanced`]).
//! - **A block open where its unit ends**: a killed one is harmless; a live If whose condition
//!   held is too, and one whose condition failed fails the command buffer in a GPU recovery,
//!   its unit's work, from before the If too, not run. Which happens is GPU data, so the commit
//!   refuses such a command buffer ([`RuntimeGap::PredicationOpenAtUnitEnd`]).
//! - **A word past its buffer's end** reads whatever memory lies there (zero past a lone
//!   buffer; the next buffer's bytes in a heap): refused
//!   ([`RuntimeGap::PredicationWordPastBuffer`]).
//!
//! # What the runtime builds
//!
//! - **A skip counter** per unit, in the command buffer's upload arena, and **a step** at each
//!   record that moves it: one invocation of the crate's predication step module
//!   (`bind::predication_step_module`), behind a barrier from everything before it and with one
//!   to the commands after it. A device without device addresses refuses every call
//!   (`Unsupported{CommandPredication}`).
//! - **Gated dispatches.** A dispatch encoded inside a block runs by `vkCmdDispatchIndirect`
//!   from a gate the step before it writes: its group counts where the counter is 0, else zero.
//!   A dispatch on the flat path gates its preprocess, whose command then stays zero. A dispatch
//!   or an execution a host step records reads the counter there, once the work before it
//!   completed, and records nothing where it is not 0; an execution inside a block is always
//!   recorded so (`Degradation::HostReplayedIndirectCommands`).
//! - **A kill** rewrites the outermost open block's start step as `PredicationOp::Dead`, whose
//!   End is then `PredicationOp::EndDead`; this model gives Metal's answers.
//! - **A loop** is laid down as copies of its commands: its first copy is what the encoder encodes
//!   between its Start and its End, which the encoder notes as it goes (each command's public call
//!   with the encoder's state it ran with, `Captures`); at the End the encoder encodes the further
//!   copies, 64 in all, or 16 inside another loop, each behind a copy start step and closed by a
//!   copy end step, and then a step past them. The copies share a flag the first failing test sets,
//!   after which every copy skips; a step past the copies sets the command buffer's overflow word
//!   where the loop would run another, which fails the command buffer at its completion
//!   (`CommandBufferError::PredicatedLoopPastCopies`). A loop inside another's copy is laid down
//!   again in each. A killed loop gets no copies. Refused: a fourth loop open
//!   ([`RuntimeGap::PredicatedLoopDepth`]), and, at the commit, a live loop that started in an
//!   earlier encoder of its unit ([`RuntimeGap::PredicatedLoopAcrossEncoders`]).

use super::buffer::Buffer;
use super::compute::{ComputeCommandEncoder, EncoderState};
use super::encoder::EncoderCore;
use super::error::{internal, Error, RuntimeGap};
use super::indirect_command_buffer::IndirectCommandBuffer;
use super::owned_kernel::{OwnedKernel, OwnedModule};
use super::resource::ResourceKey;
use super::tracker::{Access, Barrier, Use, UseRange};
use super::upload::{Upload, Uploads};
use super::values::Size;
use super::Device;
use ash::vk;
use metal2vulkan::bind::{
    PredicationCompare, PredicationOp, PredicationStep, PredicationStepModule,
};
use metal2vulkan::metal::{BarrierScope, DispatchType};
use std::ops::Range;

/// A command buffer's command predication (the module docs): its current unit, the gated runs of
/// its records, and whether a unit ended with a live block open.
#[derive(Default)]
pub(super) struct Predication {
    unit: Option<Unit>,
    /// Each step's record and the gates of the dispatches it runs, until the commit writes
    /// their tables.
    runs: Vec<Run>,
    /// A unit ended with a live block open: the commit refuses.
    open_at_end: bool,
    /// The word a loop's last copy sets where the loop would run another (made with the
    /// command buffer's first loop): the command buffer fails at its completion
    /// (`CommandBufferError::PredicatedLoopPastCopies`).
    overflow: Option<Upload>,
    /// A live loop spanned its unit's encoders: the commit refuses.
    across_encoders: bool,
}

/// The most loops open at once: each nesting multiplies its body's copies.
const MAX_LOOP_DEPTH: usize = 3;

/// The copies of a loop no other loop holds.
const FIRST_COPIES: usize = 64;

/// The copies of a loop inside another.
const NESTED_COPIES: usize = 16;

/// A record's condition: the `u32` at byte `at` of `buffer` (Metal's offset rounded down to a
/// multiple of 4), at device address `address`, compared so with `reference`.
#[derive(Clone)]
pub(super) struct Test {
    buffer: Buffer,
    at: u64,
    address: u64,
    compare: PredicationCompare,
    reference: u32,
}

impl Test {
    /// The condition of a record on `buffer` at `offset`, of Metal's `comparison` code and
    /// `reference`. Refused: a word past the buffer's end
    /// ([`RuntimeGap::PredicationWordPastBuffer`]).
    fn new(buffer: &Buffer, offset: u64, comparison: u64, reference: u32) -> Result<Test, Error> {
        let at = offset & !3;
        if at.checked_add(4).is_none_or(|end| end > buffer.length()) {
            return Err(Error::NotImplemented(RuntimeGap::PredicationWordPastBuffer));
        }
        let address = buffer
            .device_address()
            .ok_or_else(|| internal("a condition word without a device address"))?;
        Ok(Test {
            buffer: buffer.clone(),
            at,
            address: address + at,
            compare: PredicationCompare::from_metal(comparison),
            reference,
        })
    }

    /// `step` testing this condition.
    fn of(&self, step: PredicationStep) -> PredicationStep {
        step.testing(self.address, self.compare, self.reference)
    }
}

/// One unit of predication.
struct Unit {
    /// The place among the command buffer's ended encoders its last compute encoder takes.
    place: usize,
    /// The event waits and signals encoded before it.
    events: (usize, usize),
    dispatch_type: DispatchType,
    /// Its skip counter, once a record needs it.
    counter: Option<Upload>,
    /// Its open blocks, the outermost first.
    stack: Vec<Block>,
    /// A misuse killed the outermost open block (the module docs).
    dead: bool,
    /// The run (in [`Predication::runs`]) of its last step, which gates the dispatches encoded
    /// now.
    run: Option<usize>,
}

/// An open block.
struct Block {
    /// Its start step's record, whose op a kill rewrites.
    start: Upload,
    kind: Kind,
}

/// What an open block is.
#[derive(Clone)]
enum Kind {
    /// An If, and whether it has had its Else.
    If { elsed: bool },
    /// A loop's copy.
    Loop(Looping),
}

/// A loop as its copies are laid down: a While's condition, or a DoWhile's (`None`: its End names
/// it), and the flag its copies share, which its first failing test sets.
#[derive(Clone)]
pub(super) struct Looping {
    test: Option<Test>,
    flag: Upload,
}

/// An End of a loop: Metal's answer, the step to record, if any, and, where the loop is live (no
/// misused record killed it), the loop, with a DoWhile's condition, whose further copies the
/// encoder lays down.
struct LoopEnd {
    answer: bool,
    record: Option<Upload>,
    copies: Option<(Looping, Option<Test>)>,
}

/// A step's record and the gates (device addresses) of the dispatches it runs.
struct Run {
    record: Upload,
    gates: Vec<u64>,
}

impl Unit {
    fn new(place: usize, events: (usize, usize), dispatch_type: DispatchType) -> Unit {
        Unit {
            place,
            events,
            dispatch_type,
            counter: None,
            stack: Vec::new(),
            dead: false,
            run: None,
        }
    }

    /// Kill the outermost open block (the module docs), once.
    fn kill(&mut self) {
        if self.dead {
            return;
        }
        if let Some(outermost) = self.stack.first() {
            self.dead = true;
            outermost
                .start
                .write_at(PredicationStep::OP_OFFSET, &PredicationOp::Dead.word());
        }
    }
}

impl Predication {
    /// A compute encoder of `dispatch_type` opened at `place` among the ended encoders, after
    /// `events` event waits and signals: it joins the current unit, or that one ends and it
    /// starts the next.
    pub(super) fn enter(
        &mut self,
        place: usize,
        events: (usize, usize),
        dispatch_type: DispatchType,
    ) {
        if let Some(unit) = &mut self.unit {
            if unit.place + 1 == place
                && unit.events == events
                && unit.dispatch_type == dispatch_type
            {
                unit.place = place;
                return;
            }
        }
        self.end_unit();
        self.unit = Some(Unit::new(place, events, dispatch_type));
    }

    /// The current unit ends: one with a live block open refuses the commit.
    fn end_unit(&mut self) {
        if let Some(unit) = self.unit.take() {
            self.open_at_end |= !unit.stack.is_empty() && !unit.dead;
        }
    }

    /// The command buffer commits: its last unit ends, and each run's table is written into its
    /// record. Refused where a unit ended with a live block open
    /// ([`RuntimeGap::PredicationOpenAtUnitEnd`]), or as the arena refuses a table.
    pub(super) fn commit(&mut self, uploads: &mut Uploads, device: &Device) -> Result<(), Error> {
        self.end_unit();
        if self.open_at_end {
            return Err(Error::NotImplemented(RuntimeGap::PredicationOpenAtUnitEnd));
        }
        if self.across_encoders {
            return Err(Error::NotImplemented(
                RuntimeGap::PredicatedLoopAcrossEncoders,
            ));
        }
        for run in self.runs.iter().filter(|run| !run.gates.is_empty()) {
            let bytes = PredicationStep::table(addressed(&run.record)?, &run.gates);
            let table = uploads.upload(device, &bytes, bytes.len() as u64)?;
            let address = addressed(&table)?;
            run.record
                .write_at(PredicationStep::TABLE_OFFSET, &address.to_le_bytes());
        }
        self.runs.clear();
        Ok(())
    }

    /// The open encoder's unit.
    fn unit(&mut self) -> Result<&mut Unit, Error> {
        self.unit
            .as_mut()
            .ok_or_else(|| internal("a predication record outside a unit"))
    }

    /// A step of `step` on the unit's counter (made now where it has none): its record, which
    /// gates the dispatches encoded after it until the next step.
    fn step(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        step: impl FnOnce(u64) -> PredicationStep,
    ) -> Result<Upload, Error> {
        let counter = match self.unit()?.counter.clone() {
            Some(counter) => counter,
            None => uploads.upload(device, &[], 16)?,
        };
        let record = uploads.upload(device, &[], PredicationStep::RECORD_BYTES)?;
        let at = addressed(&record)?;
        record.write(&step(addressed(&counter)?).record(at));
        let run = self.runs.len();
        self.runs.push(Run {
            record: record.clone(),
            gates: Vec::new(),
        });
        let unit = self.unit()?;
        unit.counter = Some(counter);
        unit.run = Some(run);
        Ok(record)
    }

    /// The overflow word's device address, made with the first loop.
    fn overflow(&mut self, uploads: &mut Uploads, device: &Device) -> Result<u64, Error> {
        let overflow = match &self.overflow {
            Some(overflow) => overflow.clone(),
            None => uploads.upload(device, &[], 16)?,
        };
        let address = addressed(&overflow)?;
        self.overflow = Some(overflow);
        Ok(address)
    }

    /// Whether a loop's last copy found the loop would run another: read once the command
    /// buffer completed.
    pub(super) fn overflowed(&self) -> bool {
        self.overflow
            .as_ref()
            .is_some_and(|overflow| u32::from_le_bytes(overflow.read::<4>()) != 0)
    }

    /// A Start or an `encodeEndDoWhile` on a nil buffer: inside a block it kills, answering NO;
    /// with none it is refused ([`RuntimeGap::PredicationUnbalanced`]).
    fn nil(&mut self) -> Result<(), Error> {
        let unit = self.unit()?;
        if unit.stack.is_empty() {
            return Err(Error::NotImplemented(RuntimeGap::PredicationUnbalanced));
        }
        unit.kill();
        Ok(())
    }

    /// `encodeStartIf:...` of `test`: the step to record.
    fn start_if(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        test: &Test,
    ) -> Result<Upload, Error> {
        let record = self.step(uploads, device, |counter| {
            test.of(PredicationStep::new(counter, PredicationOp::If))
        })?;
        self.unit()?.stack.push(Block {
            start: record.clone(),
            kind: Kind::If { elsed: false },
        });
        Ok(record)
    }

    /// `encodeStartElse`: the step to record, if any. An Else on a loop, or an If's second,
    /// kills.
    fn start_else(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<Option<Upload>, Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return Err(Error::NotImplemented(RuntimeGap::PredicationUnbalanced));
        };
        if !matches!(top.kind, Kind::If { elsed: false }) {
            unit.kill();
            return Ok(None);
        }
        let record = self.step(uploads, device, |counter| {
            PredicationStep::new(counter, PredicationOp::Else)
        })?;
        if let Some(top) = self.unit()?.stack.last_mut() {
            top.kind = Kind::If { elsed: true };
        }
        Ok(Some(record))
    }

    /// Close the unit's top block with a step of `op` (inside a killed block, the outermost
    /// block's closes as `EndDead`): Metal's answer, which inside a killed block only the End
    /// that closes it gives YES, the unit running again after it; and the step's record.
    fn close(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        op: PredicationOp,
    ) -> Result<(bool, Upload), Error> {
        let unit = self.unit()?;
        let op = match unit.dead && unit.stack.len() == 1 {
            true => PredicationOp::EndDead,
            false => op,
        };
        let record = self.step(uploads, device, |counter| PredicationStep::new(counter, op))?;
        let unit = self.unit()?;
        unit.stack.pop();
        let answer = match unit.dead {
            true => {
                let closed = unit.stack.is_empty();
                unit.dead = !closed;
                closed
            }
            false => true,
        };
        Ok((answer, record))
    }

    /// `encodeEndIf`: Metal's answer, and the step to record, if any. An End of a loop kills
    /// and answers NO.
    fn end_if(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
    ) -> Result<(bool, Option<Upload>), Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return Err(Error::NotImplemented(RuntimeGap::PredicationUnbalanced));
        };
        if !matches!(top.kind, Kind::If { .. }) {
            unit.kill();
            return Ok((false, None));
        }
        let (answer, record) = self.close(uploads, device, PredicationOp::End)?;
        Ok((answer, Some(record)))
    }

    /// How many loops the unit has open.
    fn loops_open(&self) -> usize {
        self.unit.as_ref().map_or(0, |unit| {
            let loops = |block: &&Block| matches!(block.kind, Kind::Loop(_));
            unit.stack.iter().filter(loops).count()
        })
    }

    /// `encodeStartWhile:...` of `test`, or `encodeStartDoWhile` (`None`): its first copy's
    /// start step to record. Refused past [`MAX_LOOP_DEPTH`] open loops
    /// ([`RuntimeGap::PredicatedLoopDepth`]).
    fn start_loop(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        test: Option<&Test>,
    ) -> Result<Upload, Error> {
        if self.loops_open() >= MAX_LOOP_DEPTH {
            return Err(Error::NotImplemented(RuntimeGap::PredicatedLoopDepth));
        }
        let looping = Looping {
            test: test.cloned(),
            flag: uploads.upload(device, &[], 16)?,
        };
        self.copy_start(uploads, device, &looping)
    }

    /// The start of a copy of `looping`: its step's record, its block open.
    fn copy_start(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
    ) -> Result<Upload, Error> {
        let op = match looping.test {
            Some(_) => PredicationOp::While,
            None => PredicationOp::Do,
        };
        let record = self.loop_step(uploads, device, looping, (op, looping.test.as_ref()))?;
        self.unit()?.stack.push(Block {
            start: record.clone(),
            kind: Kind::Loop(looping.clone()),
        });
        Ok(record)
    }

    /// A step of `op` on a copy of `looping`, testing `test` where `op` tests: its record.
    fn loop_step(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        (op, test): (PredicationOp, Option<&Test>),
    ) -> Result<Upload, Error> {
        let (flag, overflow) = (addressed(&looping.flag)?, self.overflow(uploads, device)?);
        self.step(uploads, device, |counter| {
            let step = PredicationStep::new(counter, op).in_loop(flag, overflow);
            match test {
                Some(test) => test.of(step),
                None => step,
            }
        })
    }

    /// The end of a copy of `looping`, a DoWhile's testing `test`: its step's record, its block
    /// closed.
    fn copy_end(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        test: Option<&Test>,
    ) -> Result<Upload, Error> {
        let op = match looping.test {
            Some(_) => PredicationOp::End,
            None => PredicationOp::DoEnd,
        };
        let record = self.loop_step(uploads, device, looping, (op, test))?;
        self.unit()?.stack.pop();
        Ok(record)
    }

    /// After the last copy of `looping`, a DoWhile's testing `test`: the step's record that sets
    /// the overflow word where the loop would run another.
    fn past(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        looping: &Looping,
        test: Option<&Test>,
    ) -> Result<Upload, Error> {
        let op = match looping.test {
            Some(_) => PredicationOp::WhilePast,
            None => PredicationOp::DoPast,
        };
        let test = looping.test.as_ref().or(test);
        self.loop_step(uploads, device, looping, (op, test))
    }

    /// `encodeEndWhile` (`None`) or `encodeEndDoWhile:...` of `test`: Metal's answer, the step
    /// to record, if any, and the loop to lay further copies of where it is live. An End of
    /// another block kills and answers NO.
    fn end_loop(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        test: Option<&Test>,
    ) -> Result<LoopEnd, Error> {
        let unit = self.unit()?;
        let Some(top) = unit.stack.last() else {
            return Err(Error::NotImplemented(RuntimeGap::PredicationUnbalanced));
        };
        let looping = match &top.kind {
            Kind::Loop(looping) if looping.test.is_some() == test.is_none() => looping.clone(),
            _ => {
                unit.kill();
                return Ok(LoopEnd {
                    answer: false,
                    record: None,
                    copies: None,
                });
            }
        };
        if unit.dead {
            let (answer, record) = self.close(uploads, device, PredicationOp::End)?;
            return Ok(LoopEnd {
                answer,
                record: Some(record),
                copies: None,
            });
        }
        let record = self.copy_end(uploads, device, &looping, test)?;
        Ok(LoopEnd {
            answer: true,
            record: Some(record),
            copies: Some((looping, test.cloned())),
        })
    }

    /// A loop the encoder lays no copies of, as it opened in an earlier encoder of its unit:
    /// the commit refuses ([`RuntimeGap::PredicatedLoopAcrossEncoders`]).
    fn loop_across_encoders(&mut self) {
        self.across_encoders = true;
    }

    /// The gates of a dispatch the open encoder records now, one per command of `counts` (its
    /// group counts), where its unit has a block open; none where not, and it runs as recorded.
    pub(super) fn gates(
        &mut self,
        uploads: &mut Uploads,
        device: &Device,
        counts: &[[u32; 3]],
    ) -> Result<Vec<Upload>, Error> {
        let Some(unit) = &self.unit else {
            return Ok(Vec::new());
        };
        if unit.stack.is_empty() {
            return Ok(Vec::new());
        }
        let run = unit
            .run
            .ok_or_else(|| internal("an open predication block without its step"))?;
        let mut gates = Vec::with_capacity(counts.len());
        let mut addresses = Vec::with_capacity(counts.len());
        for &count in counts {
            let gate = uploads.upload(
                device,
                &PredicationStep::gate(count),
                PredicationStep::GATE_BYTES,
            )?;
            addresses.push(addressed(&gate)?);
            gates.push(gate);
        }
        self.runs[run].gates.extend(addresses);
        Ok(gates)
    }

    /// The check of a command the open encoder leaves to a host step, where its unit has a block
    /// open.
    pub(super) fn host_gate(&self) -> Option<HostGate> {
        let unit = self.unit.as_ref()?;
        if unit.stack.is_empty() {
            return None;
        }
        unit.counter.clone().map(HostGate)
    }
}

/// The device address of `upload`, which every predication region has (its step module needs
/// device addresses).
fn addressed(upload: &Upload) -> Result<u64, Error> {
    upload
        .device_address()
        .ok_or_else(|| internal("a predication region without a device address"))
}

/// The check a host step makes of a command inside a block: its unit's skip counter.
#[derive(Clone)]
pub(super) struct HostGate(Upload);

impl HostGate {
    /// Whether the command runs nothing: the counter the steps before it left is not 0. Read at
    /// the command's host step, once the work before it completed.
    pub(super) fn skips(&self) -> bool {
        u32::from_le_bytes(self.0.read::<4>()) != 0
    }
}

/// Record the step of `record` into `core`, reading the word of `test`, where it tests one: a
/// barrier from everything before, the step, and a barrier from its writes to the commands after
/// it.
fn record_step(
    core: &mut EncoderCore,
    kernel: OwnedKernel,
    record: &Upload,
    test: Option<&Test>,
) -> Result<(), Error> {
    let push = PredicationStepModule::push_constants(addressed(record)?);
    let uses: Vec<Use<ResourceKey>> = test
        .iter()
        .map(|test| {
            let (identity, range) = test.buffer.tracked(test.at..test.at + 4);
            let read = Access::new(
                vk::PipelineStageFlags2::COMPUTE_SHADER,
                vk::AccessFlags2::SHADER_STORAGE_READ,
            );
            Use::reads(identity.key, identity.tracked, UseRange::Bytes(range), read)
        })
        .collect();
    core.barrier(Barrier::everything());
    core.command(
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        &uses,
        |raw, command_buffer| {
            // SAFETY: the encoder's command buffer is recording outside a render pass, and only
            // this encoder records it; the kernel is the device's, and the record lies in the
            // command buffer's arena, which lives until it completes.
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
                raw.cmd_dispatch(command_buffer, 1, 1, 1);
                Barrier::between_dispatches().record(raw, command_buffer);
            }
        },
    );
    if let Some(test) = test {
        core.retain(test.buffer.clone());
    }
    Ok(())
}

/// A command a compute encoder encoded inside a loop, as its copies replay it: the public call's
/// arguments.
#[derive(Clone)]
pub(super) enum Event {
    Threadgroups(Size, Size),
    Threads(Size, Size),
    ThreadgroupsIndirect(Buffer, u64, Size),
    ThreadsIndirect(Buffer, u64),
    /// A memory barrier, of a scope or of resources: one global barrier.
    Barrier,
    Execute(IndirectCommandBuffer, Range<u64>),
    ExecuteIndirect(IndirectCommandBuffer, Buffer, u64),
    StartIf(Option<Buffer>, u64, u64, u32),
    Else,
    EndIf,
    StartWhile(Option<Buffer>, u64, u64, u32),
    StartDoWhile,
    EndWhile,
    EndDoWhile(Option<Buffer>, u64, u64, u32),
}

/// A compute encoder's captures of the bodies of the loops it opened and has not closed, the
/// innermost last: each command with the encoder's state it ran with.
#[derive(Default)]
pub(super) struct Captures {
    open: Vec<Vec<(EncoderState, Event)>>,
    /// The captures below it take no command: those a loop's further copies replay.
    floor: usize,
}

impl ComputeCommandEncoder {
    /// The predication step kernel, which refuses on a device without device addresses
    /// (`Unsupported{CommandPredication}`).
    fn predication_kernel(&mut self) -> Result<OwnedKernel, Error> {
        self.core()
            .device()
            .owned_kernel(OwnedModule::PredicationStep)
    }

    /// Run `record` on the command buffer's predication with its arena, under its lock.
    fn predication<T>(
        &mut self,
        record: impl FnOnce(&mut Predication, &mut Uploads, &Device) -> Result<T, Error>,
    ) -> Result<T, Error> {
        // Its commands, and those of its command buffer's other encoders, run gated inside
        // blocks that end at the commit: none of them is encoded again (`replay.rs`).
        self.stop_logging();
        let core = self.core();
        let device = core.device().clone();
        let mut state = core.command_buffer().state();
        let state = &mut *state;
        state.predicated = true;
        record(&mut state.predication, &mut state.uploads, &device)
    }

    /// The condition of a record of `call` on `buffer` (`None`: a nil buffer), with the step
    /// kernel. Refused: a buffer of another device (`Invalid`), a device without device
    /// addresses (`Unsupported{CommandPredication}`), a word past the buffer's end
    /// (`NotImplemented(PredicationWordPastBuffer)`).
    fn condition(
        &mut self,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        (comparison, reference): (u64, u32),
    ) -> Result<(OwnedKernel, Option<Test>), Error> {
        if let Some(buffer) = buffer {
            self.core().check_device(call, buffer.device())?;
        }
        let kernel = self.predication_kernel()?;
        let test = buffer
            .map(|buffer| Test::new(buffer, offset, comparison, reference))
            .transpose()?;
        Ok((kernel, test))
    }

    /// Note `event`, encoded now, in the captures of the open loops' bodies above the floor.
    pub(super) fn capture(&mut self, event: impl FnOnce() -> Event) {
        let floor = self.captures().floor;
        if self.captures().open.len() <= floor {
            return;
        }
        let captured = (self.encoder_state(), event());
        for body in &mut self.captures().open[floor..] {
            body.push(captured.clone());
        }
    }

    /// Encode `event` again, as its public call.
    fn replay(&mut self, event: &Event) -> Result<(), Error> {
        match event {
            Event::Threadgroups(groups, threads) => self.dispatch_threadgroups(*groups, *threads),
            Event::Threads(threads, per_group) => self.dispatch_threads(*threads, *per_group),
            Event::ThreadgroupsIndirect(buffer, offset, threads) => {
                self.dispatch_threadgroups_indirect(buffer, *offset, *threads)
            }
            Event::ThreadsIndirect(buffer, offset) => {
                self.dispatch_threads_indirect(buffer, *offset)
            }
            Event::Barrier => {
                self.memory_barrier_with_scope(BarrierScope::BUFFERS);
                Ok(())
            }
            Event::Execute(icb, range) => self.execute_commands_in_buffer(icb, range.clone()),
            Event::ExecuteIndirect(icb, buffer, offset) => {
                self.execute_commands_in_buffer_indirect(icb, buffer, *offset)
            }
            Event::StartIf(buffer, offset, comparison, reference) => {
                self.encode_start_if(buffer.as_ref(), *offset, *comparison, *reference)
            }
            Event::Else => self.encode_start_else(),
            Event::EndIf => self.encode_end_if().map(drop),
            Event::StartWhile(buffer, offset, comparison, reference) => {
                self.encode_start_while(buffer.as_ref(), *offset, *comparison, *reference)
            }
            Event::StartDoWhile => self.encode_start_do_while(),
            Event::EndWhile => self.encode_end_while().map(drop),
            Event::EndDoWhile(buffer, offset, comparison, reference) => self
                .encode_end_do_while(buffer.as_ref(), *offset, *comparison, *reference)
                .map(drop),
        }
    }

    /// Open a loop of `test` (`None`: a DoWhile) by the call of `event`, recording its first
    /// copy's start: refused past [`MAX_LOOP_DEPTH`] open loops. Its body is captured from here.
    fn open_loop(
        &mut self,
        kernel: OwnedKernel,
        test: Option<&Test>,
        event: impl FnOnce() -> Event,
    ) -> Result<(), Error> {
        let record = self.predication(|predication, uploads, device| {
            predication.start_loop(uploads, device, test)
        })?;
        record_step(self.core(), kernel, &record, test)?;
        self.capture(event);
        self.captures().open.push(Vec::new());
        Ok(())
    }

    /// Close the open loop by `encodeEndWhile` or `encodeEndDoWhile:...` of `test`: its first
    /// copy's end, then, where it is live, its further copies (`lay_copies`). Metal's answer.
    fn close_loop(
        &mut self,
        kernel: OwnedKernel,
        test: Option<&Test>,
        event: impl FnOnce() -> Event,
    ) -> Result<bool, Error> {
        let end = self.predication(|predication, uploads, device| {
            predication.end_loop(uploads, device, test)
        })?;
        let Some(record) = &end.record else {
            self.capture(event);
            return Ok(end.answer);
        };
        record_step(self.core(), kernel, record, test)?;
        // The loop's body, where this encoder opened it.
        let body = self.captures().open.pop();
        self.capture(event);
        match (end.copies, body) {
            (Some((looping, test)), Some(body)) => {
                self.lay_copies(kernel, (&looping, test.as_ref()), &body)?;
            }
            (Some(_), None) => self.predication(|predication, _, _| {
                predication.loop_across_encoders();
                Ok(())
            })?,
            (None, _) => {}
        }
        Ok(end.answer)
    }

    /// Lay down the further copies of `looping` (with a DoWhile's `test`), each its start, its
    /// `body` replayed with the encoder's state each command ran with, and its end; then the step
    /// that finds whether it would run past them. The encoder's state is then as before.
    fn lay_copies(
        &mut self,
        kernel: OwnedKernel,
        (looping, test): (&Looping, Option<&Test>),
        body: &[(EncoderState, Event)],
    ) -> Result<(), Error> {
        let nested = self.predication(|predication, _, _| Ok(predication.loops_open() > 0))?;
        let copies = if nested { NESTED_COPIES } else { FIRST_COPIES };
        let state = self.encoder_state();
        let floor = self.captures().floor;
        self.captures().floor = self.captures().open.len();
        let laid = self.copies(kernel, (looping, test), body, copies);
        self.captures().floor = floor;
        self.restore_state(state);
        laid
    }

    /// [`lay_copies`](Self::lay_copies)'s copies 2 to `copies`, and its past step.
    fn copies(
        &mut self,
        kernel: OwnedKernel,
        (looping, test): (&Looping, Option<&Test>),
        body: &[(EncoderState, Event)],
        copies: usize,
    ) -> Result<(), Error> {
        for _ in 1..copies {
            let record = self.predication(|predication, uploads, device| {
                predication.copy_start(uploads, device, looping)
            })?;
            record_step(self.core(), kernel, &record, looping.test.as_ref())?;
            for (state, event) in body {
                self.restore_state(state.clone());
                self.replay(event)?;
            }
            let record = self.predication(|predication, uploads, device| {
                predication.copy_end(uploads, device, looping, test)
            })?;
            record_step(self.core(), kernel, &record, test)?;
        }
        let record = self.predication(|predication, uploads, device| {
            predication.past(uploads, device, looping, test)
        })?;
        record_step(self.core(), kernel, &record, looping.test.as_ref().or(test))
    }

    /// `encodeStartIf:offset:comparison:referenceValue:` (SPI): the commands encoded after it, up
    /// to its `encodeStartElse` or `encodeEndIf`, run where the `u32` at `offset` (rounded down
    /// to a multiple of 4) of `buffer` compares so with `reference_value` when the GPU reaches the
    /// call (`comparison`: Metal's codes, `metal2vulkan::bind::PredicationCompare`), and those
    /// after its Else where not (the module docs). A nil `buffer` inside a block kills the
    /// outermost open block, as Metal's does.
    ///
    /// Refused, recording nothing: a buffer of another device (`Invalid`); a word past the
    /// buffer's end (`NotImplemented(PredicationWordPastBuffer)`); a nil buffer with no block
    /// open (`NotImplemented(PredicationUnbalanced)`); a device without device addresses
    /// (`Unsupported{CommandPredication}`).
    pub fn encode_start_if(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        let arguments = (comparison, reference_value);
        let (kernel, test) = self.condition("encode_start_if", buffer, offset, arguments)?;
        match &test {
            Some(test) => {
                let record = self.predication(|predication, uploads, device| {
                    predication.start_if(uploads, device, test)
                })?;
                record_step(self.core(), kernel, &record, Some(test))?;
            }
            None => self.predication(|predication, _, _| predication.nil())?,
        }
        self.capture(|| Event::StartIf(buffer.cloned(), offset, comparison, reference_value));
        Ok(())
    }

    /// `encodeStartElse` (SPI): the commands encoded after it, up to the open If's
    /// `encodeEndIf`, run where its condition failed (the module docs). A second Else in one If, or
    /// an Else on a loop, kills the outermost open block, as Metal's does.
    ///
    /// Refused, recording nothing: no block open (`NotImplemented(PredicationUnbalanced)`); a
    /// device without device addresses (`Unsupported{CommandPredication}`).
    pub fn encode_start_else(&mut self) -> Result<(), Error> {
        let kernel = self.predication_kernel()?;
        let record = self
            .predication(|predication, uploads, device| predication.start_else(uploads, device))?;
        if let Some(record) = record {
            record_step(self.core(), kernel, &record, None)?;
        }
        self.capture(|| Event::Else);
        Ok(())
    }

    /// `encodeEndIf` (SPI): closes the open If, answering Metal's `YES`; on a loop it kills the
    /// outermost open block and answers `NO`; inside a block a misused record killed, `NO` but
    /// for the End that closes that block (the module docs).
    ///
    /// Refused, recording nothing: no block open (`NotImplemented(PredicationUnbalanced)`); a
    /// device without device addresses (`Unsupported{CommandPredication}`).
    pub fn encode_end_if(&mut self) -> Result<bool, Error> {
        let kernel = self.predication_kernel()?;
        let (answer, record) =
            self.predication(|predication, uploads, device| predication.end_if(uploads, device))?;
        if let Some(record) = record {
            record_step(self.core(), kernel, &record, None)?;
        }
        self.capture(|| Event::EndIf);
        Ok(answer)
    }

    /// `encodeStartWhile:offset:comparison:referenceValue:` (SPI): the commands encoded after it,
    /// up to its `encodeEndWhile`, run again and again while the condition (read as
    /// [`encode_start_if`](Self::encode_start_if) reads it, before each run) holds, each run with
    /// the encoder's state each command was encoded with (the module docs). A nil `buffer` inside a
    /// block kills the outermost open block, as Metal's does.
    ///
    /// The runtime lays the loop down as copies of its commands (64, or 16 inside another loop);
    /// a loop that would run past its copies fails the command buffer at its completion
    /// (`CommandBufferError::PredicatedLoopPastCopies`), where Metal runs on.
    ///
    /// Refused, recording nothing: as [`encode_start_if`](Self::encode_start_if) refuses, and
    /// with three loops open (`NotImplemented(PredicatedLoopDepth)`). A loop that spans its unit's
    /// encoders refuses the commit (`NotImplemented(PredicatedLoopAcrossEncoders)`).
    pub fn encode_start_while(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<(), Error> {
        let arguments = (comparison, reference_value);
        let (kernel, test) = self.condition("encode_start_while", buffer, offset, arguments)?;
        let event = || Event::StartWhile(buffer.cloned(), offset, comparison, reference_value);
        match &test {
            Some(test) => self.open_loop(kernel, Some(test), event),
            None => {
                self.predication(|predication, _, _| predication.nil())?;
                self.capture(event);
                Ok(())
            }
        }
    }

    /// `encodeStartDoWhile` (SPI): the commands encoded after it, up to its `encodeEndDoWhile`,
    /// run, and again while that End's condition holds after each run, as
    /// [`encode_start_while`](Self::encode_start_while)'s run, laid down and refused as there.
    pub fn encode_start_do_while(&mut self) -> Result<(), Error> {
        let kernel = self.predication_kernel()?;
        self.open_loop(kernel, None, || Event::StartDoWhile)
    }

    /// `encodeEndWhile` (SPI): closes the open While loop, answering Metal's `YES`; on another
    /// block it kills the outermost open block and answers `NO` (the module docs).
    ///
    /// Refused, recording nothing: no block open (`NotImplemented(PredicationUnbalanced)`); a
    /// device without device addresses (`Unsupported{CommandPredication}`).
    pub fn encode_end_while(&mut self) -> Result<bool, Error> {
        let kernel = self.predication_kernel()?;
        self.close_loop(kernel, None, || Event::EndWhile)
    }

    /// `encodeEndDoWhile:offset:comparison:referenceValue:` (SPI): closes the open DoWhile loop,
    /// whose body runs again where its condition (read as
    /// [`encode_start_if`](Self::encode_start_if) reads it, after each run) holds, answering
    /// Metal's `YES`; on another block, or for a nil `buffer` inside a block, it kills the
    /// outermost open block and answers `NO` (the module docs).
    ///
    /// Refused, recording nothing: as [`encode_start_if`](Self::encode_start_if) refuses.
    pub fn encode_end_do_while(
        &mut self,
        buffer: Option<&Buffer>,
        offset: u64,
        comparison: u64,
        reference_value: u32,
    ) -> Result<bool, Error> {
        let arguments = (comparison, reference_value);
        let (kernel, test) = self.condition("encode_end_do_while", buffer, offset, arguments)?;
        let event = || Event::EndDoWhile(buffer.cloned(), offset, comparison, reference_value);
        match &test {
            Some(test) => self.close_loop(kernel, Some(test), event),
            None => {
                self.predication(|predication, _, _| predication.nil())?;
                self.capture(event);
                Ok(false)
            }
        }
    }
}
