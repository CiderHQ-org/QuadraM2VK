//! Command predication's step: the GPU half of Metal's private compute encoder calls
//! `encodeStartIf:offset:comparison:referenceValue:`, `encodeStartElse` and `encodeEndIf` (and, for
//! a killed block, the ends of the loop calls). Metal reads each block's condition when the GPU
//! reaches it, so the consumer records one invocation of the crate-owned module
//! [`predication_step_module`] at each record, which reads the condition word where it lies and
//! moves a skip counter, then gates the dispatches up to the next record.
//!
//! - **The counter** (one `u32` per unit of predication, in the consumer's memory, 0 at the
//!   unit's start): 0 while the unit's commands run, above 0 while they are skipped. A record's
//!   step moves it by its [`PredicationOp`]: an If inside a skipped region adds 1, and otherwise
//!   leaves 0 where its condition holds and sets 1 where not; an Else turns 0 into 1 and 1 into 0,
//!   leaving a deeper count alone; an End takes 1 off a count above 0. A block a misused record
//!   killed ([`PredicationOp::Dead`]) adds 2 at its start, so its own Else (which sees 2 or more)
//!   turns nothing back on, and takes 2 off at its End ([`PredicationOp::EndDead`]).
//! - **The condition** (an If's): `word OP reference`, unsigned, the word on the left
//!   ([`PredicationCompare`], Metal's codes 0 to 5; any other code compares `==`, as Metal's do).
//!   The word is the `u32` at the record's word address, which the consumer points at Metal's
//!   offset rounded down to a multiple of 4. Every step reads its word, and only an If's
//!   condition is used: a step of another op names its own record as its word.
//! - **The gates** (the dispatches a step's record runs until the next one): each gate is
//!   [`PredicationStep::GATE_BYTES`] of the consumer's memory: a `VkDispatchIndirectCommand` the
//!   consumer wrote (the dispatch's group counts), and at [`PredicationStep::GATE_OUT_OFFSET`] the
//!   one the dispatch runs by `vkCmdDispatchIndirect`, which the step writes: the group counts
//!   where the counter it leaves is 0, else zero. The record's table names them
//!   ([`PredicationStep::table`]), and then the record's own sink gate (at
//!   [`PredicationStep::SINK_GATE_OFFSET`]); a record's own empty table (at
//!   [`PredicationStep::EMPTY_TABLE_OFFSET`], naming only the sink) is its table until the
//!   consumer points it at one.
//! - **Every read lands in memory the record names**, in a branch not taken and in a loop's pass
//!   past its count too: llvmpipe (lavapipe) makes a uniform load under control flow whether or
//!   not an invocation runs it, and a step faulted there reading address 0 (a scalar `u32` load of
//!   address 0 in its JIT code, `SEGV_MAPERR`). So the word is read whatever the op, and the gate
//!   loop's pass past the table's count reads the sink, whose command the step writes, if at all,
//!   where no dispatch reads it.
//!
//! **The record** ([`PredicationStep::record`], [`PredicationStep::RECORD_BYTES`], 16-aligned): the
//! counter's device address, the word's, the table's, the op, the comparison and the reference, an
//! empty table, and the sink gate. The consumer may rewrite the op ([`PredicationStep::OP_OFFSET`],
//! [`PredicationOp::word`]) and the table's address ([`PredicationStep::TABLE_OFFSET`]) until the
//! step runs.
//!
//! **The module** (fixed for every device): one `GLCompute` entry `main` of local size
//! `(1, 1, 1)`, dispatched `(1, 1, 1)`, no descriptor, and [`PredicationStepModule::PUSH_BYTES`] of
//! `COMPUTE` push constants: the record's device address
//! ([`PredicationStepModule::push_constants`]).
//!
//! **Recording** (consumer): a barrier from everything before the step (Metal's word is read
//! after the work before the record, written with or without a barrier); the step; a barrier
//! from its writes to the indirect reads and compute reads after it. The counter, the table and
//! the gates are read and written by the steps and the gated dispatches alone.
//!
//! **Where it runs.** The module reaches memory by device address (`bufferDeviceAddress`,
//! `shaderInt64`). Elsewhere [`predication_step_module`] names what is missing
//! (`CommandPredication`).

use std::sync::Arc;

use super::{at, count_loop, k32, k64, load32, load64, no_binding, store32, wide, Types};
use crate::budget::{Budget, Meter};
use crate::device::profile::BufferAccess;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::layout::{PipelineLayout, PushConstantRange};
use crate::pipeline::sym::PushField;
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::Value;
use crate::spirv::ir::{BinOp, ExecMode, ExecModel, MemAccess, Type, TypedModule, UnOp};
use crate::vk;

/// What a predication record's step does to its unit's skip counter (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicationOp {
    /// `encodeStartIf:...`: inside a skipped region, one deeper; else 0 where the condition
    /// holds and 1 where not.
    If,
    /// `encodeStartElse`: 0 becomes 1 and 1 becomes 0; a deeper count stays.
    Else,
    /// `encodeEndIf` (or another block's end): one less, down to 0.
    End,
    /// The start of a block a misused record killed: two deeper, so nothing inside it runs and
    /// its own Else turns nothing back on.
    Dead,
    /// The end of a killed block: two less.
    EndDead,
    /// A copy's start of `encodeStartWhile:...`'s loop (laid down as copies): inside a
    /// skipped region one deeper; else 1 where the loop has exited (its flag set), 0 where the
    /// condition holds, and 1 where not, which sets the flag.
    While,
    /// A copy's start of `encodeStartDoWhile`'s loop: inside a skipped region one deeper; else 1
    /// where the loop has exited, and 0 where not.
    Do,
    /// A copy's end of a DoWhile loop (`encodeEndDoWhile:...`): where the copy ran (the counter
    /// 0) and the condition fails, the flag is set; then one less, as [`End`](Self::End).
    DoEnd,
    /// After a While loop's last copy: where the loop would run another (the counter 0, the
    /// flag clear, the condition holding), the overflow word is set; the counter stays.
    WhilePast,
    /// After a DoWhile loop's last copy: where the loop would run another (the counter 0, the
    /// flag clear), the overflow word is set; the counter stays.
    DoPast,
}

impl PredicationOp {
    /// The op's `u32` in a record, little-endian (written at [`PredicationStep::OP_OFFSET`]).
    pub fn word(self) -> [u8; 4] {
        self.code().to_le_bytes()
    }

    fn code(self) -> u32 {
        match self {
            PredicationOp::If => OP_IF,
            PredicationOp::Else => OP_ELSE,
            PredicationOp::End => OP_END,
            PredicationOp::Dead => OP_DEAD,
            PredicationOp::EndDead => OP_END_DEAD,
            PredicationOp::While => OP_WHILE,
            PredicationOp::Do => OP_DO,
            PredicationOp::DoEnd => OP_DO_END,
            PredicationOp::WhilePast => OP_WHILE_PAST,
            PredicationOp::DoPast => OP_DO_PAST,
        }
    }

    /// Whether the step reads Metal's word.
    fn tests(self) -> bool {
        matches!(
            self,
            PredicationOp::If
                | PredicationOp::While
                | PredicationOp::DoEnd
                | PredicationOp::WhilePast
        )
    }

    /// Whether the step reads and writes its loop's flag and the overflow word.
    fn loops(self) -> bool {
        matches!(
            self,
            PredicationOp::While
                | PredicationOp::Do
                | PredicationOp::DoEnd
                | PredicationOp::WhilePast
                | PredicationOp::DoPast
        )
    }
}

const OP_IF: u32 = 0;
const OP_ELSE: u32 = 1;
const OP_END: u32 = 2;
const OP_DEAD: u32 = 3;
const OP_END_DEAD: u32 = 4;
const OP_WHILE: u32 = 5;
const OP_DO: u32 = 6;
const OP_DO_END: u32 = 7;
const OP_WHILE_PAST: u32 = 8;
const OP_DO_PAST: u32 = 9;

/// An If's comparison of Metal's word with its reference, unsigned, the word on the left: Metal's
/// `comparison` argument, which is not an `MTLCompareFunction` (measured on an M4).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PredicationCompare {
    /// 0: `word == reference`.
    Equal,
    /// 1: `word != reference`.
    NotEqual,
    /// 2: `word > reference`.
    Greater,
    /// 3: `word >= reference`.
    GreaterEqual,
    /// 4: `word < reference`.
    Less,
    /// 5: `word <= reference`.
    LessEqual,
}

impl PredicationCompare {
    /// Metal's `comparison` argument: 0 to 5 as the variants state, and any other value
    /// (6, 255, `2^32 + 2`, `2^64 - 1`, all measured) [`Equal`](Self::Equal), as Metal compares.
    pub fn from_metal(comparison: u64) -> PredicationCompare {
        match comparison {
            1 => PredicationCompare::NotEqual,
            2 => PredicationCompare::Greater,
            3 => PredicationCompare::GreaterEqual,
            4 => PredicationCompare::Less,
            5 => PredicationCompare::LessEqual,
            _ => PredicationCompare::Equal,
        }
    }

    fn code(self) -> u32 {
        match self {
            PredicationCompare::Equal => 0,
            PredicationCompare::NotEqual => 1,
            PredicationCompare::Greater => 2,
            PredicationCompare::GreaterEqual => 3,
            PredicationCompare::Less => 4,
            PredicationCompare::LessEqual => 5,
        }
    }

    /// Whether `word` compares so with `reference`.
    pub fn holds(self, word: u32, reference: u32) -> bool {
        match self {
            PredicationCompare::Equal => word == reference,
            PredicationCompare::NotEqual => word != reference,
            PredicationCompare::Greater => word > reference,
            PredicationCompare::GreaterEqual => word >= reference,
            PredicationCompare::Less => word < reference,
            PredicationCompare::LessEqual => word <= reference,
        }
    }
}

/// One predication record (the module docs): the step's inputs.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PredicationStep {
    /// The device address of the unit's skip counter (a `u32`, 4-aligned).
    pub counter: u64,
    /// What the step does.
    pub op: PredicationOp,
    /// The device address of Metal's condition word (4-aligned), which an
    /// [`If`](PredicationOp::If), a [`While`](PredicationOp::While), a
    /// [`DoEnd`](PredicationOp::DoEnd) and a [`WhilePast`](PredicationOp::WhilePast) test; 0 for
    /// the others, whose record names itself instead ([`PredicationStep::record`]).
    pub word: u64,
    /// The condition's comparison.
    pub compare: PredicationCompare,
    /// The condition's reference value.
    pub reference: u32,
    /// A loop's step: the device address of the loop's flag (a `u32`, 0 until the loop exits;
    /// one per laying down of the loop). 0 for the others, whose record names a word of its own.
    pub flag: u64,
    /// A loop's step: the device address of the overflow word (a `u32`) a
    /// [`WhilePast`](PredicationOp::WhilePast) or [`DoPast`](PredicationOp::DoPast) sets. 0 for
    /// the others, whose record names a word of its own.
    pub overflow: u64,
}

// The record's fields.
const R_COUNTER: u64 = 0;
const R_WORD: u64 = 8;
const R_COMPARE: u64 = 28;
const R_REFERENCE: u64 = 32;
/// A word of the record's own that a step not of a loop names as its flag and overflow word,
/// which it writes back unchanged.
const R_SCRATCH: u64 = 36;
const R_FLAG: u64 = 40;
const R_OVERFLOW: u64 = 48;

impl PredicationStep {
    /// Bytes of a record.
    pub const RECORD_BYTES: u64 = 112;
    /// The record's table address (`u64`): the record's own empty table until the consumer
    /// points it at one ([`PredicationStep::table`]).
    pub const TABLE_OFFSET: u64 = 16;
    /// The record's op (`u32`, [`PredicationOp::word`]).
    pub const OP_OFFSET: u64 = 24;
    /// The record's own empty table: a count of 0, and the sink's entry.
    pub const EMPTY_TABLE_OFFSET: u64 = 56;
    /// The record's sink gate ([`PredicationStep::GATE_BYTES`]), which every table names after
    /// its gates (the module docs).
    pub const SINK_GATE_OFFSET: u64 = 80;
    /// Bytes of a gate: the dispatch's `VkDispatchIndirectCommand`, then the one it runs.
    pub const GATE_BYTES: u64 = 32;
    /// The gate's command the dispatch runs (12 bytes), which the step writes.
    pub const GATE_OUT_OFFSET: u64 = 16;

    /// The step of `op` on the counter at device address `counter`: for an
    /// [`If`](PredicationOp::If) use [`PredicationStep::condition`].
    pub fn new(counter: u64, op: PredicationOp) -> PredicationStep {
        PredicationStep {
            counter,
            op,
            word: 0,
            compare: PredicationCompare::Equal,
            reference: 0,
            flag: 0,
            overflow: 0,
        }
    }

    /// The If on the counter at device address `counter` of Metal's word at device address
    /// `word`, compared so with `reference`.
    pub fn condition(
        counter: u64,
        word: u64,
        compare: PredicationCompare,
        reference: u32,
    ) -> PredicationStep {
        PredicationStep {
            counter,
            op: PredicationOp::If,
            word,
            compare,
            reference,
            flag: 0,
            overflow: 0,
        }
    }

    /// This step testing Metal's word at device address `word`, compared so with `reference`
    /// (a [`While`](PredicationOp::While), [`DoEnd`](PredicationOp::DoEnd) or
    /// [`WhilePast`](PredicationOp::WhilePast)).
    pub fn testing(self, word: u64, compare: PredicationCompare, reference: u32) -> Self {
        PredicationStep {
            word,
            compare,
            reference,
            ..self
        }
    }

    /// This loop step on the loop's flag and the overflow word at device addresses `flag` and
    /// `overflow`.
    pub fn in_loop(self, flag: u64, overflow: u64) -> Self {
        PredicationStep {
            flag,
            overflow,
            ..self
        }
    }

    /// The record's bytes where it lies at device address `at`, its table its own empty one; a
    /// step that tests no word names the record itself as its word, and one not of a loop a word
    /// of its own as its flag and overflow word, so no step reads or writes address 0.
    pub fn record(&self, at: u64) -> [u8; 112] {
        let mut bytes = [0u8; 112];
        let mut put = |offset: u64, word: &[u8]| {
            let offset = offset as usize;
            bytes[offset..offset + word.len()].copy_from_slice(word);
        };
        put(R_COUNTER, &self.counter.to_le_bytes());
        let word = if self.op.tests() { self.word } else { at };
        put(R_WORD, &word.to_le_bytes());
        let (flag, overflow) = match self.op.loops() {
            true => (self.flag, self.overflow),
            false => (at + R_SCRATCH, at + R_SCRATCH),
        };
        put(R_FLAG, &flag.to_le_bytes());
        put(R_OVERFLOW, &overflow.to_le_bytes());
        put(
            PredicationStep::TABLE_OFFSET,
            &(at + PredicationStep::EMPTY_TABLE_OFFSET).to_le_bytes(),
        );
        put(PredicationStep::OP_OFFSET, &self.op.word());
        put(R_COMPARE, &self.compare.code().to_le_bytes());
        put(R_REFERENCE, &self.reference.to_le_bytes());
        let empty = PredicationStep::table(at, &[]);
        put(PredicationStep::EMPTY_TABLE_OFFSET, &empty);
        bytes
    }

    /// The table, for the record at device address `at`, of the gates at device addresses
    /// `gates` (each 16-aligned): their count (a `u32`), 4 bytes of padding, then each address
    /// (a `u64`), and last the record's sink gate's (the module docs).
    pub fn table(at: u64, gates: &[u64]) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(16 + 8 * gates.len());
        bytes.extend_from_slice(&(gates.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&[0; 4]);
        let sink = at + PredicationStep::SINK_GATE_OFFSET;
        for gate in gates.iter().chain([&sink]) {
            bytes.extend_from_slice(&gate.to_le_bytes());
        }
        bytes
    }

    /// A gate's bytes for a dispatch of `group_count`: the command, and a zero one for the step
    /// to write.
    pub fn gate(group_count: [u32; 3]) -> [u8; 32] {
        let mut bytes = [0u8; 32];
        for (i, n) in group_count.iter().enumerate() {
            bytes[4 * i..4 * i + 4].copy_from_slice(&n.to_le_bytes());
        }
        bytes
    }
}

/// The crate-owned predication step module (the module docs), validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredicationStepModule {
    /// SPIR-V words, entry point `main`, no descriptor.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (device addresses and `shaderInt64`).
    pub requirements: Requirements,
    /// The pipeline layout the consumer creates the module's pipeline with: no descriptor set,
    /// and one `COMPUTE` push-constant range of [`PredicationStepModule::PUSH_BYTES`] at offset 0.
    pub layout: PipelineLayout,
}

impl PredicationStepModule {
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the record's device address.
    pub const PUSH_BYTES: u32 = 8;

    /// The push bytes for the record at device address `record`.
    pub fn push_constants(record: u64) -> [u8; 8] {
        record.to_le_bytes()
    }
}

/// The predication step module of `device` (the module docs): built with the typed builder,
/// checked by the owned verifier against the device's profile (requirements included), and
/// serialized. A device without device addresses is `Unsupported{CommandPredication}` naming
/// what it lacks.
pub fn predication_step_module(device: &Device) -> Result<PredicationStepModule, Error> {
    let profile = device.profile();
    if profile.fields().buffer_access != BufferAccess::Addressed {
        return Err(Unsupported::needs_any_of(
            Feature::CommandPredication,
            vec![profile.addressed_needs()],
        )
        .into());
    }
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the predication step module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &no_binding)?;
    Ok(PredicationStepModule {
        spirv: words.into(),
        requirements: r.requirements,
        layout: PipelineLayout {
            sets: Vec::new(),
            push_constants: vec![PushConstantRange {
                stages: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: PredicationStepModule::PUSH_BYTES,
            }],
            push_descriptor_eligible: false,
        },
    })
}

/// The predication step module as a typed module, before serialization (what
/// [`predication_step_module`] serializes, and what the CPU interpreter runs). Its push field is
/// `ReplayRegionAddress`, the record's address.
pub(super) fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let u64_ = b.ty(Type::Int {
        width: 64,
        signed: false,
    })?;
    let t = Types { u32_, u64_ };
    let m_record = b.push_member(PushField::ReplayRegionAddress, u64_, 0)?;
    let push = b.push_global()?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let fb = &mut f;

    let push_value = fb.module().global_value(push)?;
    let i = k32(fb, m_record.index)?;
    let p = fb.access_chain(push_value, &[i])?;
    let record = fb.load(p, MemAccess::default())?;
    let field = |fb: &mut FunctionBuilder<'_>, offset: u64| -> BResult<Value> {
        let a = at(fb, record, offset)?;
        load32(fb, t, a)
    };
    let counter_at = at(fb, record, R_COUNTER)?;
    let counter_at = load64(fb, t, counter_at)?;
    let c = load32(fb, t, counter_at)?;
    let op = field(fb, PredicationStep::OP_OFFSET)?;
    let compare = field(fb, R_COMPARE)?;
    let reference = field(fb, R_REFERENCE)?;

    // The word: an If's condition word, else the record itself (`PredicationStep::record`).
    let word_at = at(fb, record, R_WORD)?;
    let word_at = load64(fb, t, word_at)?;
    let word = load32(fb, t, word_at)?;

    // The condition: Metal's six comparisons, `==` for any other code.
    let mut holds = fb.binary(BinOp::IEqual, word, reference)?;
    for (code, cmp) in [
        (1, BinOp::INotEqual),
        (2, BinOp::UGreaterThan),
        (3, BinOp::UGreaterThanEqual),
        (4, BinOp::ULessThan),
        (5, BinOp::ULessThanEqual),
    ] {
        let k = k32(fb, code)?;
        let is = fb.binary(BinOp::IEqual, compare, k)?;
        let v = fb.binary(cmp, word, reference)?;
        holds = fb.select(is, v, holds)?;
    }

    // The counter each op leaves.
    let zero = k32(fb, 0)?;
    let one = k32(fb, 1)?;
    let two = k32(fb, 2)?;
    let skipping = fb.binary(BinOp::INotEqual, c, zero)?;
    let up = fb.binary(BinOp::IAdd, c, one)?;
    let fresh = fb.select(holds, zero, one)?;
    let after_if = fb.select(skipping, up, fresh)?;
    let at_one = fb.binary(BinOp::IEqual, c, one)?;
    let flipped = fb.select(at_one, zero, c)?;
    let after_else = fb.select(skipping, flipped, one)?;
    let down = fb.binary(BinOp::ISub, c, one)?;
    let after_end = fb.select(skipping, down, zero)?;
    let after_dead = fb.binary(BinOp::IAdd, c, two)?;
    let deep = fb.binary(BinOp::UGreaterThanEqual, c, two)?;
    let down_two = fb.binary(BinOp::ISub, c, two)?;
    let after_end_dead = fb.select(deep, down_two, zero)?;

    // A loop's flag (set once the loop exits) and the overflow word (set where a loop would run
    // past its last copy); a step not of a loop names a word of its record for both, and writes
    // back what it read.
    let flag_at = at(fb, record, R_FLAG)?;
    let flag_at = load64(fb, t, flag_at)?;
    let overflow_at = at(fb, record, R_OVERFLOW)?;
    let overflow_at = load64(fb, t, overflow_at)?;
    let flag = load32(fb, t, flag_at)?;
    let overflow = load32(fb, t, overflow_at)?;
    let exited = fb.binary(BinOp::INotEqual, flag, zero)?;
    let fails = fb.unary(UnOp::LogicalNot, holds)?;
    let running = fb.unary(UnOp::LogicalNot, skipping)?;
    // While: as an If, but 1 once the loop exited, and a failing test sets the flag.
    let tested = fb.select(exited, one, fresh)?;
    let after_while = fb.select(skipping, up, tested)?;
    let exits = fb.binary(BinOp::LogicalAnd, running, fails)?;
    let flag_while = fb.select(exits, one, flag)?;
    // Do: 1 once the loop exited, else 0.
    let entered = fb.select(exited, one, zero)?;
    let after_do = fb.select(skipping, up, entered)?;
    // DoEnd: a copy that ran sets the flag where the test fails; then as an End.
    let flag_do_end = fb.select(exits, one, flag)?;
    // WhilePast and DoPast: a loop still running sets the overflow word; the counter stays.
    let staying = fb.unary(UnOp::LogicalNot, exited)?;
    let live = fb.binary(BinOp::LogicalAnd, running, staying)?;
    let past_while = fb.binary(BinOp::LogicalAnd, live, holds)?;
    let overflow_while = fb.select(past_while, one, overflow)?;
    let overflow_do = fb.select(live, one, overflow)?;

    let mut next = c;
    let mut flag_next = flag;
    let mut overflow_next = overflow;
    for (code, value, flag_value, overflow_value) in [
        (OP_IF, after_if, flag, overflow),
        (OP_ELSE, after_else, flag, overflow),
        (OP_END, after_end, flag, overflow),
        (OP_DEAD, after_dead, flag, overflow),
        (OP_END_DEAD, after_end_dead, flag, overflow),
        (OP_WHILE, after_while, flag_while, overflow),
        (OP_DO, after_do, flag, overflow),
        (OP_DO_END, after_end, flag_do_end, overflow),
        (OP_WHILE_PAST, c, flag, overflow_while),
        (OP_DO_PAST, c, flag, overflow_do),
    ] {
        let k = k32(fb, code)?;
        let is = fb.binary(BinOp::IEqual, op, k)?;
        next = fb.select(is, value, next)?;
        flag_next = fb.select(is, flag_value, flag_next)?;
        overflow_next = fb.select(is, overflow_value, overflow_next)?;
    }
    store32(fb, t, counter_at, next)?;
    store32(fb, t, flag_at, flag_next)?;
    store32(fb, t, overflow_at, overflow_next)?;

    // The gates: each runs its command where the counter is 0, else nothing.
    let runs = fb.binary(BinOp::IEqual, next, zero)?;
    let table = at(fb, record, PredicationStep::TABLE_OFFSET)?;
    let table = load64(fb, t, table)?;
    let n = load32(fb, t, table)?;
    count_loop(fb, t, n, move |fb, i| {
        let i64_ = wide(fb, t, i)?;
        let eight = k64(fb, 8)?;
        let rel = fb.binary(BinOp::IMul, i64_, eight)?;
        let entries = at(fb, table, 8)?;
        let entry = fb.binary(BinOp::IAdd, entries, rel)?;
        let gate = load64(fb, t, entry)?;
        let zero = k32(fb, 0)?;
        for axis in 0..3 {
            let source = at(fb, gate, 4 * axis)?;
            let count = load32(fb, t, source)?;
            let count = fb.select(runs, count, zero)?;
            let out = at(fb, gate, PredicationStep::GATE_OUT_OFFSET + 4 * axis)?;
            store32(fb, t, out, count)?;
        }
        Ok(())
    })?;
    fb.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}
