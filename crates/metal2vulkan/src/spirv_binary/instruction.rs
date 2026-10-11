//! A parsed SPIR-V module: its id bound and its instructions with their operands.

use crate::spv::{Op, Word};

/// A module as [`super::parse_bytes`] reads it.
pub(crate) struct Module {
    /// The header's bound: every id is below it.
    pub(crate) bound: Word,
    pub(crate) instructions: Vec<Instruction>,
}

/// One instruction. Its result type and result are fields, not operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Instruction {
    pub(crate) opcode: Op,
    pub(crate) result_type: Option<Word>,
    pub(crate) result_id: Option<Word>,
    pub(crate) operands: Vec<Operand>,
}

/// One operand, in the grammar's order. An enumerated operand is its value (or mask) followed by
/// the operands it brings; a pair is two operands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum Operand {
    /// An `IdRef`.
    IdRef(Word),
    /// An `IdScope`.
    IdScope(Word),
    /// An `IdMemorySemantics`.
    IdMemorySemantics(Word),
    /// A one-word literal: an integer, a number of 32 bits or fewer, an enumerant or a mask.
    Word(Word),
    /// A two-word literal number, low word first.
    Wide(u64),
    /// A literal string.
    String(String),
    /// An `OpExtInst`'s instruction number in its set.
    LiteralExtInstInteger(Word),
    /// The opcode an `OpSpecConstantOp` applies.
    SpecConstantOp(Op),
}
