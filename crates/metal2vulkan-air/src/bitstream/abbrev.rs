//! Abbreviation definitions (`DEFINE_ABBREV`) and abbreviated record reading.

use super::cursor::{Cursor, MAX_FIXED_WIDTH, MAX_VBR_WIDTH};
use crate::error::BitstreamError;

/// A scalar operand encoding, usable as an array element.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarOp {
    /// A fixed-width field of 1..=64 bits.
    Fixed(u8),
    /// A VBR field with 2..=32-bit chunks.
    Vbr(u8),
    /// A 6-bit character from `[a-zA-Z0-9._]`, yielded as its ASCII code.
    Char6,
}

/// One operand of an abbreviation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AbbrevOp {
    /// A constant value; reads no bits. A zero-width fixed or VBR field is the literal 0 (as in
    /// LLVM).
    Literal(u64),
    /// A scalar field.
    Scalar(ScalarOp),
    /// A VBR6 length followed by that many elements. Only second to last, followed by its
    /// element encoding (which the definition counts as an operand).
    Array(ScalarOp),
    /// A VBR6 byte length, 32-bit alignment, the bytes, 32-bit alignment. Only last.
    Blob,
}

/// A validated abbreviation: the first operand (the record code) is never an array or a blob,
/// an array is last and a blob is last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Abbrev {
    ops: Box<[AbbrevOp]>,
}

impl Abbrev {
    /// The operands, the record code first.
    pub fn ops(&self) -> &[AbbrevOp] {
        &self.ops
    }
}

/// Encoding numbers of `DEFINE_ABBREV` (LLVM `BitCodeAbbrevOp::Encoding`).
const ENC_FIXED: u64 = 1;
const ENC_VBR: u64 = 2;
const ENC_ARRAY: u64 = 3;
const ENC_CHAR6: u64 = 4;
const ENC_BLOB: u64 = 5;

/// The fewest bits one operand definition takes (a 1-bit flag and a 3-bit encoding).
pub(super) const MIN_OP_DEF_BITS: u64 = 4;

/// Read the operand list of a `DEFINE_ABBREV` whose abbrev id was already consumed. `numops`
/// counts operands as written, including an array's element encoding. `ops` is empty with room
/// for `numops` entries (the caller charged it).
pub(super) fn read_definition(
    cursor: &mut Cursor<'_>,
    numops: u64,
    mut ops: Vec<AbbrevOp>,
) -> Result<Abbrev, BitstreamError> {
    if numops == 0 {
        return Err(BitstreamError::BadAbbrev);
    }
    let mut i = 0u64;
    while i < numops {
        let op = match read_op(cursor)? {
            Def::Literal(v) => AbbrevOp::Literal(v),
            Def::Scalar(s) => AbbrevOp::Scalar(s),
            Def::Blob => {
                if i == 0 || i + 1 != numops {
                    return Err(BitstreamError::BadAbbrev);
                }
                AbbrevOp::Blob
            }
            Def::Array => {
                if i == 0 || i + 2 != numops {
                    return Err(BitstreamError::BadAbbrev);
                }
                i += 1;
                match read_op(cursor)? {
                    Def::Scalar(element) => AbbrevOp::Array(element),
                    Def::Literal(_) | Def::Array | Def::Blob => {
                        return Err(BitstreamError::BadAbbrev)
                    }
                }
            }
        };
        ops.push(op);
        i += 1;
    }
    Ok(Abbrev {
        ops: ops.into_boxed_slice(),
    })
}

enum Def {
    Literal(u64),
    Scalar(ScalarOp),
    Array,
    Blob,
}

fn read_op(cursor: &mut Cursor<'_>) -> Result<Def, BitstreamError> {
    if cursor.fixed(1)? == 1 {
        return Ok(Def::Literal(cursor.vbr(8)?));
    }
    match cursor.fixed(3)? {
        enc @ (ENC_FIXED | ENC_VBR) => {
            let width = cursor.vbr(5)?;
            if width == 0 {
                return Ok(Def::Literal(0));
            }
            let (max, min) = if enc == ENC_FIXED {
                (MAX_FIXED_WIDTH, 1)
            } else {
                (MAX_VBR_WIDTH, 2)
            };
            let width = u8::try_from(width)
                .ok()
                .filter(|w| (min..=max).contains(&u32::from(*w)))
                .ok_or(BitstreamError::BadAbbrev)?;
            Ok(Def::Scalar(if enc == ENC_FIXED {
                ScalarOp::Fixed(width)
            } else {
                ScalarOp::Vbr(width)
            }))
        }
        ENC_ARRAY => Ok(Def::Array),
        ENC_CHAR6 => Ok(Def::Scalar(ScalarOp::Char6)),
        ENC_BLOB => Ok(Def::Blob),
        _ => Err(BitstreamError::BadAbbrev),
    }
}

impl ScalarOp {
    /// Read one field.
    pub(super) fn read(self, cursor: &mut Cursor<'_>) -> Result<u64, BitstreamError> {
        match self {
            ScalarOp::Fixed(w) => cursor.fixed(u32::from(w)),
            ScalarOp::Vbr(w) => cursor.vbr(u32::from(w)),
            ScalarOp::Char6 => cursor.fixed(6).map(decode_char6),
        }
    }

    /// The fewest bits one field takes, which bounds an array length by the remaining bits.
    pub(super) fn min_bits(self) -> u64 {
        match self {
            ScalarOp::Fixed(w) | ScalarOp::Vbr(w) => u64::from(w),
            ScalarOp::Char6 => 6,
        }
    }
}

/// Map a 6-bit value to its character: `a-z`, `A-Z`, `0-9`, `.`, `_`.
fn decode_char6(v: u64) -> u64 {
    match v {
        0..=25 => u64::from(b'a') + v,
        26..=51 => u64::from(b'A') + (v - 26),
        52..=61 => u64::from(b'0') + (v - 52),
        62 => u64::from(b'.'),
        _ => u64::from(b'_'),
    }
}
