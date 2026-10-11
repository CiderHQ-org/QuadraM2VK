//! A record-at-a-time view of the bitstream for the record decoders.
//!
//! [`Stream::next`] copies each record's operands into one reused buffer, so a decoder can read
//! them while it still asks the reader for the error location of that record. The buffer grows
//! only through the meter.

use crate::bitstream::{Entry, Reader};
use crate::budget::Meter;
use crate::error::{BitcodeError, InputError, InputErrorKind, Location};
use crate::error::{Error, Feature};

/// One entry, with a record's operands left in [`Stream::ops`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Item<'a> {
    SubBlock(u32),
    EndBlock,
    Record { code: u32, blob: Option<&'a [u8]> },
    End,
}

/// The reader plus the operands of the last record.
#[derive(Debug)]
pub(crate) struct Stream<'a> {
    reader: Reader<'a>,
    ops: Vec<u64>,
}

impl<'a> Stream<'a> {
    pub(crate) fn new(reader: Reader<'a>) -> Stream<'a> {
        Stream {
            reader,
            ops: Vec::new(),
        }
    }

    /// Read the next entry.
    pub(crate) fn next(&mut self, m: &mut Meter) -> Result<Item<'a>, Error> {
        let Stream { reader, ops } = self;
        match reader.advance(m)? {
            Entry::SubBlock { id } => Ok(Item::SubBlock(id)),
            Entry::EndBlock { .. } => Ok(Item::EndBlock),
            Entry::End => Ok(Item::End),
            Entry::Record(rec) => {
                ops.clear();
                if rec.ops.len() > ops.capacity() {
                    m.reserve(ops, rec.ops.len())?;
                }
                ops.extend_from_slice(rec.ops);
                Ok(Item::Record {
                    code: rec.code,
                    blob: rec.blob,
                })
            }
        }
    }

    /// The operands of the last record read.
    pub(crate) fn ops(&self) -> &[u64] {
        &self.ops
    }

    /// Skip the rest of the block just entered.
    pub(crate) fn skip_block(&mut self, m: &mut Meter) -> Result<(), Error> {
        self.reader.skip_block(m)
    }

    /// The bit position where the entry just read starts.
    pub(crate) fn entry_bit(&self) -> u64 {
        self.reader.entry_bit()
    }

    /// The location of the entry just read.
    pub(crate) fn location(&self) -> Location {
        self.reader.location()
    }

    /// A bitcode error at the entry just read.
    pub(crate) fn err(&self, kind: BitcodeError) -> Error {
        InputError::new(InputErrorKind::Bitcode(kind), self.location()).into()
    }

    /// A malformed-record error for record `code` of `block`.
    pub(crate) fn malformed(&self, block: u32, code: u32) -> Error {
        self.err(BitcodeError::MalformedRecord { block, code })
    }

    /// A typed refusal of valid input at the entry just read.
    pub(crate) fn unsupported(&self, feature: Feature) -> Error {
        InputError::new(InputErrorKind::Unsupported(feature), self.location()).into()
    }

    /// Operand `i` of the last record, or a malformed-record error.
    pub(crate) fn op(&self, block: u32, code: u32, i: usize) -> Result<u64, Error> {
        self.ops
            .get(i)
            .copied()
            .ok_or_else(|| self.malformed(block, code))
    }

    /// Operand `i` as a `u32`.
    pub(crate) fn op32(&self, block: u32, code: u32, i: usize) -> Result<u32, Error> {
        u32::try_from(self.op(block, code, i)?).map_err(|_| self.malformed(block, code))
    }

    /// Operands from `start` as bytes (each must be below 256), charged to the meter.
    pub(crate) fn chars(
        &self,
        m: &mut Meter,
        block: u32,
        code: u32,
        start: usize,
    ) -> Result<Vec<u8>, Error> {
        chars(self.ops.get(start..).unwrap_or(&[]), m)?.ok_or_else(|| self.malformed(block, code))
    }

    /// Consume every remaining entry of the block just entered, requiring it to hold only
    /// records, and hand each to `f`.
    pub(crate) fn records(
        &mut self,
        m: &mut Meter,
        block: u32,
        mut f: impl FnMut(&Stream<'a>, u32, Option<&'a [u8]>, &mut Meter) -> Result<(), Error>,
    ) -> Result<(), Error> {
        loop {
            match self.next(m)? {
                Item::Record { code, blob } => f(self, code, blob, m)?,
                Item::EndBlock => return Ok(()),
                Item::SubBlock(id) => {
                    return Err(self.err(BitcodeError::UnknownBlock { block: id }));
                }
                Item::End => {
                    return Err(self.err(BitcodeError::MalformedRecord { block, code: 0 }));
                }
            }
        }
    }
}

/// Record operands as bytes, if every one is below 256. The copy is charged to the meter first.
pub(crate) fn chars(ops: &[u64], m: &mut Meter) -> Result<Option<Vec<u8>>, Error> {
    let mut out = Vec::new();
    m.reserve(&mut out, ops.len())?;
    for &c in ops {
        match u8::try_from(c) {
            Ok(b) => out.push(b),
            Err(_) => return Ok(None),
        }
    }
    Ok(Some(out))
}

/// Decode LLVM's sign-rotated VBR integer (`v >> 1`, negated when bit 0 is set).
pub(crate) fn signed_vbr(v: u64) -> i64 {
    let magnitude = (v >> 1) as i64;
    if v & 1 == 0 {
        magnitude
    } else if magnitude == 0 {
        // "-0" encodes i64::MIN.
        i64::MIN
    } else {
        -magnitude
    }
}
