//! Metallib tag lists: `fourcc`, `u16` length, bytes; terminated by `ENDT` (no length).

use super::container_error;
use crate::error::ContainerError;
use crate::error::Error;

/// The tag-list terminator.
pub(super) const ENDT: [u8; 4] = *b"ENDT";

/// A tag kept verbatim (unknown to this layer, or recorded for a later cross-check).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawTag<'a> {
    /// The tag's four-character code.
    pub fourcc: [u8; 4],
    /// The tag's content, its length prefix excluded.
    pub bytes: &'a [u8],
}

/// A tag with the file offset of its fourcc (for error locations).
#[derive(Clone, Copy, Debug)]
pub(super) struct Tag<'a> {
    pub(super) raw: RawTag<'a>,
    pub(super) offset: u64,
}

/// Read the tag list starting at `start` inside `region = [.., end)` of `file`. Returns the tags
/// and the offset just past `ENDT`. A tag or terminator crossing `end` is `BadTag`.
pub(super) fn read_list(
    file: &[u8],
    start: usize,
    end: usize,
) -> Result<(Vec<Tag<'_>>, usize), Error> {
    let mut tags = Vec::new();
    let mut at = start;
    loop {
        let fourcc = take(file, at, 4, end)?;
        let fourcc: [u8; 4] = fourcc
            .try_into()
            .map_err(|_| container_error(ContainerError::BadTag, at as u64))?;
        if fourcc == ENDT {
            return Ok((tags, at + 4));
        }
        let len = take(file, at + 4, 2, end)?;
        let len = u16::from_le_bytes(
            len.try_into()
                .map_err(|_| container_error(ContainerError::BadTag, at as u64))?,
        );
        let bytes = take(file, at + 6, usize::from(len), end)?;
        tags.push(Tag {
            raw: RawTag { fourcc, bytes },
            offset: at as u64,
        });
        at += 6 + usize::from(len);
    }
}

/// `file[at..at + len]`, which must end at or before `end`.
fn take(file: &[u8], at: usize, len: usize, end: usize) -> Result<&[u8], Error> {
    at.checked_add(len)
        .filter(|e| *e <= end)
        .and_then(|e| file.get(at..e))
        .ok_or_else(|| container_error(ContainerError::BadTag, at as u64))
}

/// A little-endian reader over one tag's bytes that must be consumed exactly.
pub(super) struct TagReader<'a> {
    bytes: &'a [u8],
    at: usize,
    offset: u64,
}

impl<'a> TagReader<'a> {
    pub(super) fn new(tag: &Tag<'a>) -> TagReader<'a> {
        TagReader {
            bytes: tag.raw.bytes,
            at: 0,
            offset: tag.offset,
        }
    }

    fn bad(&self) -> Error {
        container_error(ContainerError::BadTag, self.offset)
    }

    pub(super) fn array<const N: usize>(&mut self) -> Result<[u8; N], Error> {
        let slice = self
            .at
            .checked_add(N)
            .and_then(|e| self.bytes.get(self.at..e))
            .ok_or_else(|| self.bad())?;
        let array = <[u8; N]>::try_from(slice).map_err(|_| self.bad())?;
        self.at += N;
        Ok(array)
    }

    pub(super) fn u8(&mut self) -> Result<u8, Error> {
        self.array::<1>().map(|[b]| b)
    }

    pub(super) fn u16(&mut self) -> Result<u16, Error> {
        self.array().map(u16::from_le_bytes)
    }

    pub(super) fn u64(&mut self) -> Result<u64, Error> {
        self.array().map(u64::from_le_bytes)
    }

    /// A NUL-terminated UTF-8 string (the NUL is consumed, not returned).
    pub(super) fn cstr(&mut self) -> Result<&'a str, Error> {
        let rest = self.bytes.get(self.at..).ok_or_else(|| self.bad())?;
        let nul = rest
            .iter()
            .position(|&b| b == 0)
            .ok_or_else(|| self.bad())?;
        let s = rest
            .get(..nul)
            .and_then(|b| std::str::from_utf8(b).ok())
            .ok_or_else(|| self.bad())?;
        self.at += nul + 1;
        Ok(s)
    }

    /// Bytes left unread.
    pub(super) fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.at)
    }

    /// Require that every byte was consumed.
    pub(super) fn finish(self) -> Result<(), Error> {
        if self.at == self.bytes.len() {
            Ok(())
        } else {
            Err(self.bad())
        }
    }
}
