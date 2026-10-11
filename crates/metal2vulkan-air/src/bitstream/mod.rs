//! Generic LLVM bitstream reader.
//!
//! LLVM-agnostic: 32-bit words, fixed and VBR fields, `ENTER_SUBBLOCK` / `END_BLOCK`,
//! `DEFINE_ABBREV` (literal, fixed, VBR, array, char6, blob), `UNABBREV_RECORD`, and `BLOCKINFO`
//! (`SETBID` and abbreviations; block and record names are ignored). The meaning of block ids and
//! record codes belongs to `air::bitcode`.
//!
//! [`Reader::advance`] yields one [`Entry`] at a time. Entering a sub-block is implicit; a caller
//! that does not want a block's contents calls [`Reader::skip_block`], which jumps over the rest
//! of it by its length word. `BLOCKINFO` blocks are consumed internally and never yielded.
//!
//! Robustness: every count (abbreviation operands, record operands, array and blob lengths) is
//! checked against the remaining bits of the innermost block before anything is reserved, and
//! every reservation is charged to the [`Meter`] first. Blocks nest at most
//! [`Limit::BitstreamNesting`] deep; a block's contents cannot read past its declared length, and
//! its `END_BLOCK` must land exactly on it. Each entry ticks the meter with one unit plus its
//! operand count. Malformed input is an [`InputError`] at the offending entry, never a panic.

mod abbrev;
mod cursor;
#[cfg(test)]
mod tests;
#[cfg(test)]
pub(crate) mod writer;

pub use abbrev::{Abbrev, AbbrevOp};

use crate::budget::Meter;
use crate::error::{BitstreamError, InputError, InputErrorKind, Location};
use crate::error::{Error, InternalError, InternalKind, Limit};
use cursor::Cursor;

/// The block id of `BLOCKINFO`.
pub const BLOCKINFO_BLOCK_ID: u32 = 0;
/// The abbrev id width outside every block.
pub const TOP_LEVEL_ABBREV_WIDTH: u32 = 2;

const ABBREV_END_BLOCK: u64 = 0;
const ABBREV_ENTER_SUBBLOCK: u64 = 1;
const ABBREV_DEFINE: u64 = 2;
const ABBREV_UNABBREV_RECORD: u64 = 3;
/// The first application-defined abbrev id.
const FIRST_APPLICATION_ABBREV: u64 = 4;
/// `BLOCKINFO` record: set the block the following definitions apply to.
const BLOCKINFO_CODE_SETBID: u32 = 1;
/// Largest abbrev id width.
const MAX_ABBREV_WIDTH: u32 = 32;

/// One record: its code, operands (the code excluded), and optional blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Record<'r, 'a> {
    /// The record code, which the enclosing block's kind gives its meaning.
    pub code: u32,
    /// The operands, each widened to `u64`, borrowed from the reader until it advances.
    pub ops: &'r [u64],
    /// The blob operand's bytes, borrowed from the input, when the abbrev has one.
    pub blob: Option<&'a [u8]>,
    /// The abbrev id it was read with (`None` for `UNABBREV_RECORD`).
    pub abbrev: Option<u32>,
    /// The bit position where the record starts.
    pub bit: u64,
}

/// What [`Reader::advance`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Entry<'r, 'a> {
    /// A sub-block was entered; its entries follow until the matching `EndBlock`.
    SubBlock {
        /// The block id.
        id: u32,
    },
    /// The innermost block ended; the reader is back in its parent.
    EndBlock {
        /// The id of the block that ended.
        id: u32,
    },
    /// A record of the innermost block.
    Record(Record<'r, 'a>),
    /// The stream ended after its last top-level block.
    End,
}

#[derive(Debug)]
struct Scope {
    id: u32,
    abbrev_width: u32,
    /// The bit position where the block's `END_BLOCK` must end.
    end: u64,
    /// `BLOCKINFO` abbreviations visible in this block: those defined before it was entered.
    inherited: usize,
    local: Vec<Abbrev>,
    /// Records read so far (for diagnostics).
    records: u32,
}

/// A streaming reader over one bitstream.
#[derive(Debug)]
pub struct Reader<'a> {
    cursor: Cursor<'a>,
    magic: [u8; 4],
    scopes: Vec<Scope>,
    /// Abbreviations from `BLOCKINFO`, per block id, in definition order.
    blockinfo: Vec<(u32, Vec<Abbrev>)>,
    ops: Vec<u64>,
    /// Start of the entry being read, for error locations.
    entry_bit: u64,
    /// Whether the entry being read is a record (for error locations).
    in_record: bool,
}

impl<'a> Reader<'a> {
    /// A reader over `stream`, whose first 32 bits are the application magic ([`Reader::magic`]).
    /// The stream length must be a multiple of four bytes.
    pub fn new(stream: &'a [u8]) -> Result<Reader<'a>, Error> {
        let at = |kind| input_error(kind, Location::bitstream(0, &[], None));
        if !stream.len().is_multiple_of(4) {
            return Err(at(BitstreamError::UnexpectedEnd));
        }
        let mut magic = [0u8; 4];
        match stream.get(..4) {
            Some(m) => magic.copy_from_slice(m),
            None => return Err(at(BitstreamError::UnexpectedEnd)),
        }
        let mut cursor = Cursor::new(stream);
        cursor.seek(32).map_err(at)?;
        Ok(Reader {
            cursor,
            magic,
            scopes: Vec::new(),
            blockinfo: Vec::new(),
            ops: Vec::new(),
            entry_bit: 32,
            in_record: false,
        })
    }

    /// The first four bytes of the stream.
    pub fn magic(&self) -> [u8; 4] {
        self.magic
    }

    /// The current bit position from the start of the stream.
    #[cfg(test)]
    pub fn bit_position(&self) -> u64 {
        self.cursor.pos()
    }

    /// The bit position where the entry last returned by [`Reader::advance`] starts (for a
    /// sub-block, its `ENTER_SUBBLOCK` abbrev id).
    pub fn entry_bit(&self) -> u64 {
        self.entry_bit
    }

    /// How many blocks are open.
    #[cfg(test)]
    pub fn depth(&self) -> usize {
        self.scopes.len()
    }

    /// The id of the innermost open block.
    #[cfg(test)]
    pub fn block_id(&self) -> Option<u32> {
        self.scopes.last().map(|s| s.id)
    }

    /// A [`Location`] for an error found by a caller in the entry just returned.
    pub fn location(&self) -> Location {
        let blocks: Vec<u32> = self.scopes.iter().map(|s| s.id).collect();
        let record = if self.in_record {
            self.scopes.last().map(|s| s.records.saturating_sub(1))
        } else {
            None
        };
        Location::bitstream(self.entry_bit, &blocks, record)
    }

    /// Read the next entry.
    pub fn advance(&mut self, m: &mut Meter) -> Result<Entry<'_, 'a>, Error> {
        loop {
            self.entry_bit = self.cursor.pos();
            self.in_record = false;
            let Some(width) = self.scopes.last().map(|s| s.abbrev_width) else {
                // Only blocks live at the top level.
                if self.cursor.pos() == self.cursor.len_bits() {
                    return Ok(Entry::End);
                }
                let id = self.stream(|c| c.fixed(TOP_LEVEL_ABBREV_WIDTH))?;
                if id != ABBREV_ENTER_SUBBLOCK {
                    return Err(self.error(BitstreamError::UnknownAbbrevId {
                        id: u32::try_from(id).unwrap_or(u32::MAX),
                    }));
                }
                m.tick(1)?;
                match self.enter_block(m)? {
                    BLOCKINFO_BLOCK_ID => {
                        self.read_blockinfo(m)?;
                        continue;
                    }
                    id => return Ok(Entry::SubBlock { id }),
                }
            };
            let id = self.stream(|c| c.fixed(width))?;
            match id {
                ABBREV_END_BLOCK => {
                    m.tick(1)?;
                    let id = self.end_block()?;
                    return Ok(Entry::EndBlock { id });
                }
                ABBREV_ENTER_SUBBLOCK => {
                    m.tick(1)?;
                    match self.enter_block(m)? {
                        BLOCKINFO_BLOCK_ID => self.read_blockinfo(m)?,
                        id => return Ok(Entry::SubBlock { id }),
                    }
                }
                ABBREV_DEFINE => {
                    let abbrev = self.read_definition(m)?;
                    m.tick(1 + op_units(abbrev.ops().len()))?;
                    let scope = self.scope_mut()?;
                    grow(m, &mut scope.local)?;
                    scope.local.push(abbrev);
                }
                _ => {
                    let (code, blob) = self.read_record(id, m)?;
                    m.tick(1 + op_units(self.ops.len()))?;
                    let abbrev = u32::try_from(id)
                        .ok()
                        .filter(|_| id != ABBREV_UNABBREV_RECORD);
                    return Ok(Entry::Record(Record {
                        code,
                        ops: &self.ops,
                        blob,
                        abbrev,
                        bit: self.entry_bit,
                    }));
                }
            }
        }
    }

    /// Jump over the rest of the innermost block (by its length word) and leave it.
    pub fn skip_block(&mut self, m: &mut Meter) -> Result<(), Error> {
        m.tick(1)?;
        let end = self
            .scopes
            .last()
            .map(|s| s.end)
            .ok_or_else(|| invariant("skip_block outside every block"))?;
        self.pop_scope();
        self.stream(|c| c.seek(end))
    }

    /// Read a block header after `ENTER_SUBBLOCK` and open the block. Returns its id.
    fn enter_block(&mut self, m: &mut Meter) -> Result<u32, Error> {
        let id = self.stream(|c| c.vbr32(8))?;
        let width = self.stream(|c| c.vbr32(4))?;
        if width == 0 || width > MAX_ABBREV_WIDTH {
            return Err(self.error(BitstreamError::ValueOutOfRange));
        }
        self.stream(Cursor::align32)?;
        let words = self.stream(|c| c.fixed(32))?;
        let end = words
            .checked_mul(32)
            .and_then(|bits| bits.checked_add(self.cursor.pos()))
            .filter(|end| *end <= self.cursor.limit())
            .ok_or_else(|| self.error(BitstreamError::BadBlockLength))?;
        m.cap(Limit::BitstreamNesting, self.scopes.len() as u64 + 1)?;
        let inherited = self.blockinfo_for(id).map_or(0, <[Abbrev]>::len);
        grow(m, &mut self.scopes)?;
        self.scopes.push(Scope {
            id,
            abbrev_width: width,
            end,
            inherited,
            local: Vec::new(),
            records: 0,
        });
        self.cursor.set_limit(end);
        Ok(id)
    }

    /// Finish the innermost block after its `END_BLOCK` abbrev id. Returns its id.
    fn end_block(&mut self) -> Result<u32, Error> {
        self.stream(Cursor::align32)?;
        let (id, end) = self
            .scopes
            .last()
            .map(|s| (s.id, s.end))
            .ok_or_else(|| invariant("END_BLOCK outside every block"))?;
        if self.cursor.pos() != end {
            return Err(self.error(BitstreamError::BadBlockLength));
        }
        self.pop_scope();
        Ok(id)
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
        let limit = self.scopes.last().map_or(self.cursor.len_bits(), |s| s.end);
        self.cursor.set_limit(limit);
    }

    /// Consume a `BLOCKINFO` block that was just entered, through its `END_BLOCK`.
    fn read_blockinfo(&mut self, m: &mut Meter) -> Result<(), Error> {
        let mut target: Option<u32> = None;
        loop {
            self.entry_bit = self.cursor.pos();
            self.in_record = false;
            let width = self.scope_mut()?.abbrev_width;
            let id = self.stream(|c| c.fixed(width))?;
            match id {
                ABBREV_END_BLOCK => {
                    m.tick(1)?;
                    self.end_block()?;
                    return Ok(());
                }
                ABBREV_ENTER_SUBBLOCK => return Err(self.error(BitstreamError::BadBlockInfo)),
                ABBREV_DEFINE => {
                    let abbrev = self.read_definition(m)?;
                    m.tick(1 + op_units(abbrev.ops().len()))?;
                    let Some(block) = target else {
                        return Err(self.error(BitstreamError::BadBlockInfo));
                    };
                    self.blockinfo_push(block, abbrev, m)?;
                }
                _ => {
                    let (code, _) = self.read_record(id, m)?;
                    m.tick(1 + op_units(self.ops.len()))?;
                    if code == BLOCKINFO_CODE_SETBID {
                        let block = self
                            .ops
                            .first()
                            .copied()
                            .ok_or_else(|| self.error(BitstreamError::BadBlockInfo))?;
                        target = Some(
                            u32::try_from(block)
                                .map_err(|_| self.error(BitstreamError::ValueOutOfRange))?,
                        );
                    }
                    // BLOCKNAME and SETRECORDNAME are names only; other codes carry nothing we use.
                }
            }
        }
    }

    fn blockinfo_for(&self, block: u32) -> Option<&[Abbrev]> {
        self.blockinfo
            .iter()
            .find(|(id, _)| *id == block)
            .map(|(_, list)| list.as_slice())
    }

    fn blockinfo_push(&mut self, block: u32, abbrev: Abbrev, m: &mut Meter) -> Result<(), Error> {
        let index = match self.blockinfo.iter().position(|(id, _)| *id == block) {
            Some(i) => i,
            None => {
                grow(m, &mut self.blockinfo)?;
                self.blockinfo.push((block, Vec::new()));
                self.blockinfo.len() - 1
            }
        };
        let list = match self.blockinfo.get_mut(index) {
            Some((_, list)) => list,
            None => return Err(invariant("blockinfo index out of range")),
        };
        grow(m, list)?;
        list.push(abbrev);
        Ok(())
    }

    /// Read a `DEFINE_ABBREV` body.
    fn read_definition(&mut self, m: &mut Meter) -> Result<Abbrev, Error> {
        let numops = self.stream(|c| c.vbr(5))?;
        if numops > self.cursor.remaining() / abbrev::MIN_OP_DEF_BITS {
            return Err(self.error(BitstreamError::UnexpectedEnd));
        }
        let mut ops = Vec::new();
        m.reserve(&mut ops, count(numops)?)?;
        self.stream(|c| abbrev::read_definition(c, numops, ops))
    }

    /// Read a record body after its abbrev id into `self.ops`. Returns the code and blob.
    fn read_record(&mut self, id: u64, m: &mut Meter) -> Result<(u32, Option<&'a [u8]>), Error> {
        self.in_record = true;
        self.ops.clear();
        let scope = self.scope_mut()?;
        scope.records = scope.records.saturating_add(1);
        let Reader {
            cursor,
            scopes,
            blockinfo,
            ops,
            ..
        } = self;
        let result = if id == ABBREV_UNABBREV_RECORD {
            read_unabbreviated(cursor, ops, m)
        } else {
            match lookup(scopes, blockinfo, id) {
                Some(abbrev) => read_abbreviated(cursor, ops, abbrev, m),
                None => Err(Step::Stream(BitstreamError::UnknownAbbrevId {
                    id: u32::try_from(id).unwrap_or(u32::MAX),
                })),
            }
        };
        let (code, blob) = result.map_err(|step| match step {
            Step::Stream(kind) => self.error(kind),
            Step::Meter(e) => e,
        })?;
        let code = u32::try_from(code).map_err(|_| self.error(BitstreamError::ValueOutOfRange))?;
        Ok((code, blob))
    }

    fn scope_mut(&mut self) -> Result<&mut Scope, Error> {
        self.scopes
            .last_mut()
            .ok_or_else(|| invariant("no open block"))
    }

    /// Run a cursor operation, locating its error at the current entry.
    fn stream<T>(
        &mut self,
        f: impl FnOnce(&mut Cursor<'a>) -> Result<T, BitstreamError>,
    ) -> Result<T, Error> {
        match f(&mut self.cursor) {
            Ok(v) => Ok(v),
            Err(e) => Err(self.error(e)),
        }
    }

    fn error(&self, kind: BitstreamError) -> Error {
        input_error(kind, self.location())
    }
}

/// A failure inside a record read: a stream error (located by the caller) or a meter outcome.
enum Step {
    Stream(BitstreamError),
    Meter(Error),
}

impl From<BitstreamError> for Step {
    fn from(e: BitstreamError) -> Step {
        Step::Stream(e)
    }
}

impl From<Error> for Step {
    fn from(e: Error) -> Step {
        Step::Meter(e)
    }
}

/// The abbreviation for application abbrev id `id` in the innermost block: first the `BLOCKINFO`
/// abbreviations inherited when the block was entered, then the block's own.
fn lookup<'s>(
    scopes: &'s [Scope],
    blockinfo: &'s [(u32, Vec<Abbrev>)],
    id: u64,
) -> Option<&'s Abbrev> {
    let scope = scopes.last()?;
    let index = usize::try_from(id.checked_sub(FIRST_APPLICATION_ABBREV)?).ok()?;
    if index < scope.inherited {
        blockinfo
            .iter()
            .find(|(block, _)| *block == scope.id)
            .and_then(|(_, list)| list.get(index))
    } else {
        scope.local.get(index - scope.inherited)
    }
}

/// `UNABBREV_RECORD`: VBR6 code, VBR6 operand count, VBR6 operands.
fn read_unabbreviated<'a>(
    cursor: &mut Cursor<'a>,
    out: &mut Vec<u64>,
    m: &mut Meter,
) -> Result<(u64, Option<&'a [u8]>), Step> {
    let code = cursor.vbr(6)?;
    let n = cursor.vbr(6)?;
    if n > cursor.remaining() / 6 {
        return Err(BitstreamError::UnexpectedEnd.into());
    }
    reserve_total(m, out, n)?;
    for _ in 0..n {
        out.push(cursor.vbr(6)?);
    }
    Ok((code, None))
}

/// A record read with `abbrev`.
fn read_abbreviated<'a>(
    cursor: &mut Cursor<'a>,
    out: &mut Vec<u64>,
    abbrev: &Abbrev,
    m: &mut Meter,
) -> Result<(u64, Option<&'a [u8]>), Step> {
    let mut ops = abbrev.ops().iter();
    let code = match ops.next() {
        Some(AbbrevOp::Literal(v)) => *v,
        Some(AbbrevOp::Scalar(s)) => s.read(cursor)?,
        // Definitions never start with an array or blob, and are never empty.
        Some(AbbrevOp::Array(_) | AbbrevOp::Blob) | None => {
            return Err(invariant("validated abbreviation starts with a non-scalar").into())
        }
    };
    reserve_total(m, out, ops.len() as u64)?;
    let mut blob = None;
    for op in ops {
        match *op {
            AbbrevOp::Literal(v) => out.push(v),
            AbbrevOp::Scalar(s) => out.push(s.read(cursor)?),
            AbbrevOp::Array(element) => {
                let n = cursor.vbr(6)?;
                if n > cursor.remaining() / element.min_bits() {
                    return Err(BitstreamError::UnexpectedEnd.into());
                }
                reserve_total(m, out, (out.len() as u64).saturating_add(n))?;
                for _ in 0..n {
                    out.push(element.read(cursor)?);
                }
            }
            AbbrevOp::Blob => {
                let n = cursor.vbr(6)?;
                cursor.align32()?;
                blob = Some(cursor.bytes(n)?);
                cursor.align32()?;
            }
        }
    }
    Ok((code, blob))
}

/// Make room for `total` elements in `v`, charging the meter only when the buffer must grow.
fn reserve_total<T>(m: &mut Meter, v: &mut Vec<T>, total: u64) -> Result<(), Error> {
    let total = count(total)?;
    if total > v.capacity() {
        m.reserve(v, total - v.len())?;
    }
    Ok(())
}

fn input_error(kind: BitstreamError, at: Location) -> Error {
    InputError::new(InputErrorKind::Bitstream(kind), at).into()
}

fn invariant(detail: &str) -> Error {
    Error::Internal(InternalError::new(InternalKind::Invariant, detail))
}

/// Work units for `n` operands.
fn op_units(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// A bounded count as a `usize`.
fn count(n: u64) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| Error::LimitExceeded(Limit::LedgerBytes))
}

/// Charge for and reserve room for one more element when `v` is full (doubling).
fn grow<T>(m: &mut Meter, v: &mut Vec<T>) -> Result<(), Error> {
    if v.len() == v.capacity() {
        let additional = v.len().max(4);
        m.reserve(v, additional)?;
    }
    Ok(())
}
