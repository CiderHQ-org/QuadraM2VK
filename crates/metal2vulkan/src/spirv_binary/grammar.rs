//! The shape of the generated instruction table (`table.rs`): each instruction's operands by kind
//! and count, and each enumerated kind's values with the operands a value brings.

use super::table::FORMS;
use crate::spv::{Op, Word};

/// One instruction's operands in order, as the grammar lists them.
pub(super) struct Form {
    pub(super) opcode: Op,
    pub(super) operands: &'static [(Kind, Count)],
}

/// How many times an operand occurs.
#[derive(Clone, Copy)]
pub(super) enum Count {
    One,
    /// Zero or one time.
    Optional,
    /// Zero or more times, to the instruction's end.
    Any,
}

/// What an operand is, by the grammar's operand kinds.
#[derive(Clone, Copy)]
pub(super) enum Kind {
    /// `IdResultType`: the instruction's result type.
    ResultType,
    /// `IdResult`: the instruction's result.
    ResultId,
    /// `IdRef`.
    Id,
    /// `IdScope`.
    Scope,
    /// `IdMemorySemantics`.
    Semantics,
    /// `LiteralInteger` or `LiteralFloat`: one word.
    Word,
    /// `LiteralString`: nul-terminated UTF-8, packed four bytes a word.
    Text,
    /// `LiteralContextDependentNumber`: as wide as the instruction's result type.
    Number,
    /// `PairLiteralIntegerIdRef` (`OpSwitch`'s targets): a literal as wide as the selector, then an
    /// id.
    SwitchTarget,
    /// `PairIdRefLiteralInteger`.
    IdWord,
    /// `PairIdRefIdRef`.
    IdId,
    /// `LiteralExtInstInteger`: an extended instruction's number.
    ExtInstNumber,
    /// `LiteralSpecConstantOpInteger`: an opcode, then that instruction's operands less its result
    /// type and result.
    SpecOp,
    /// A `ValueEnum` kind: one value, then the operands that value brings.
    Value(&'static [Enumerant]),
    /// A `BitEnum` kind: a mask, then the operands of each set bit, lowest bit first.
    Mask(&'static [Enumerant]),
}

/// One value (or, of a mask, one bit) of an enumerated kind, by increasing value.
pub(super) struct Enumerant {
    pub(super) value: Word,
    pub(super) parameters: &'static [Kind],
}

/// The form of the instruction with `opcode`, if the grammar defines one.
pub(super) fn form(opcode: u32) -> Option<&'static Form> {
    let at = FORMS
        .binary_search_by_key(&opcode, |form| form.opcode as u32)
        .ok()?;
    FORMS.get(at)
}

/// Whether `opcode`'s instructions have a result type and a result.
pub(crate) fn instruction_result_shape(opcode: Op) -> Option<(bool, bool)> {
    let form = form(opcode as u32)?;
    let has = |wanted: fn(&Kind) -> bool| form.operands.iter().any(|(kind, _)| wanted(kind));
    Some((
        has(|kind| matches!(kind, Kind::ResultType)),
        has(|kind| matches!(kind, Kind::ResultId)),
    ))
}
