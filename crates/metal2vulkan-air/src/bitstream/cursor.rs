//! Bit-level reads over a byte slice, least-significant bit first (LLVM bitstream order).
//!
//! Every read is bounded by `limit`, the end of the innermost open block (or of the stream). A
//! read that would cross it fails with [`BitstreamError::UnexpectedEnd`], so a record can never
//! run into its parent's bytes.

use crate::error::BitstreamError;

/// Largest fixed-width field (bits).
pub(super) const MAX_FIXED_WIDTH: u32 = 64;
/// Largest VBR chunk width (bits), as in LLVM.
pub(super) const MAX_VBR_WIDTH: u32 = 32;

#[derive(Debug)]
pub(super) struct Cursor<'a> {
    data: &'a [u8],
    pos: u64,
    limit: u64,
}

impl<'a> Cursor<'a> {
    /// A cursor over all of `data`, starting at bit 0.
    pub(super) fn new(data: &'a [u8]) -> Cursor<'a> {
        let limit = (data.len() as u64).saturating_mul(8);
        Cursor {
            data,
            pos: 0,
            limit,
        }
    }

    /// The current bit position from the start of the stream.
    pub(super) fn pos(&self) -> u64 {
        self.pos
    }

    /// The end of the readable range (bits).
    pub(super) fn limit(&self) -> u64 {
        self.limit
    }

    /// The length of the whole stream (bits).
    pub(super) fn len_bits(&self) -> u64 {
        (self.data.len() as u64).saturating_mul(8)
    }

    /// Bits left before `limit`.
    pub(super) fn remaining(&self) -> u64 {
        self.limit.saturating_sub(self.pos)
    }

    /// Narrow or restore the readable range. Callers keep `limit <= len_bits()`.
    pub(super) fn set_limit(&mut self, limit: u64) {
        self.limit = limit.min(self.len_bits());
    }

    /// Move to `pos`, which must be within the readable range.
    pub(super) fn seek(&mut self, pos: u64) -> Result<(), BitstreamError> {
        if pos > self.limit {
            return Err(BitstreamError::UnexpectedEnd);
        }
        self.pos = pos;
        Ok(())
    }

    /// Read a `width`-bit field (`width <= 64`).
    pub(super) fn fixed(&mut self, width: u32) -> Result<u64, BitstreamError> {
        if width > MAX_FIXED_WIDTH {
            return Err(BitstreamError::BadAbbrev);
        }
        let end = self
            .pos
            .checked_add(u64::from(width))
            .ok_or(BitstreamError::UnexpectedEnd)?;
        if end > self.limit {
            return Err(BitstreamError::UnexpectedEnd);
        }
        let mut value = 0u64;
        let mut got = 0u32;
        let mut pos = self.pos;
        while got < width {
            let byte = usize::try_from(pos / 8)
                .ok()
                .and_then(|i| self.data.get(i))
                .copied()
                .ok_or(BitstreamError::UnexpectedEnd)?;
            // `pos % 8 < 8`, so the conversion is exact.
            let shift = (pos % 8) as u32;
            let take = (8 - shift).min(width - got);
            let bits = (u64::from(byte) >> shift) & ((1u64 << take) - 1);
            value |= bits << got;
            got += take;
            pos += u64::from(take);
        }
        self.pos = end;
        Ok(value)
    }

    /// Read a variable-width integer made of `width`-bit chunks (`2 <= width <= 32`). A value that
    /// does not fit 64 bits is [`BitstreamError::BadVbr`].
    pub(super) fn vbr(&mut self, width: u32) -> Result<u64, BitstreamError> {
        if !(2..=MAX_VBR_WIDTH).contains(&width) {
            return Err(BitstreamError::BadAbbrev);
        }
        let payload_bits = width - 1;
        let high = 1u64 << payload_bits;
        let mut value = 0u64;
        let mut shift = 0u32;
        loop {
            let chunk = self.fixed(width)?;
            let payload = chunk & (high - 1);
            if payload != 0 {
                if shift >= 64 || (payload << shift) >> shift != payload {
                    return Err(BitstreamError::BadVbr);
                }
                value |= payload << shift;
            }
            if chunk & high == 0 {
                return Ok(value);
            }
            shift += payload_bits;
            if shift >= 64 {
                // A further chunk could only carry bits beyond 64.
                return Err(BitstreamError::BadVbr);
            }
        }
    }

    /// A VBR value that must fit `u32`.
    pub(super) fn vbr32(&mut self, width: u32) -> Result<u32, BitstreamError> {
        let value = self.vbr(width)?;
        u32::try_from(value).map_err(|_| BitstreamError::ValueOutOfRange)
    }

    /// Skip to the next multiple of 32 bits.
    pub(super) fn align32(&mut self) -> Result<(), BitstreamError> {
        let aligned = self
            .pos
            .checked_add(31)
            .map(|p| p & !31)
            .ok_or(BitstreamError::UnexpectedEnd)?;
        self.seek(aligned)
    }

    /// Take `len` bytes at the current (byte-aligned) position.
    pub(super) fn bytes(&mut self, len: u64) -> Result<&'a [u8], BitstreamError> {
        if !self.pos.is_multiple_of(8) {
            return Err(BitstreamError::BadBlob);
        }
        let bits = len.checked_mul(8).ok_or(BitstreamError::UnexpectedEnd)?;
        if bits > self.remaining() {
            return Err(BitstreamError::UnexpectedEnd);
        }
        let start = usize::try_from(self.pos / 8).map_err(|_| BitstreamError::UnexpectedEnd)?;
        let len = usize::try_from(len).map_err(|_| BitstreamError::UnexpectedEnd)?;
        let end = start
            .checked_add(len)
            .ok_or(BitstreamError::UnexpectedEnd)?;
        let slice = self
            .data
            .get(start..end)
            .ok_or(BitstreamError::UnexpectedEnd)?;
        self.pos += bits;
        Ok(slice)
    }
}
