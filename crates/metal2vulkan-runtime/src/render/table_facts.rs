//! What a render stage's argument table binds, as a render encoder's checks at the call read it
//! (`render/call_checks.rs`): each buffer index's buffer, inline bytes or visible-function table,
//! and each setter's refusals, the table's own (`argument_table.rs`) made over these facts with
//! its own helpers. The lowering's table holds the same after the same calls.

use crate::argument_table::{
    check_bound_buffer, check_index, check_range, BUFFER_SLOTS, INLINE_BYTES_LIMIT, SAMPLER_SLOTS,
    TEXTURE_SLOTS,
};
use crate::buffer::Buffer;
use crate::device::Device;
use crate::encoder::check_device;
use crate::error::{invalid, Error, Problem};
use crate::function_table::VisibleFunctionTable;
use crate::sampler::SamplerState;
use crate::texture::Texture;

/// What one buffer index binds.
#[derive(Clone, Default)]
enum Bound {
    #[default]
    Nothing,
    /// A buffer at an offset.
    Buffer(Buffer, u64),
    /// Inline bytes.
    Bytes,
    /// A visible-function table.
    Table,
    /// The null table an offset made of a table (`ArgumentTable::null_tables`).
    Null,
}

/// One render stage's table's facts (the module docs).
pub(super) struct TableFacts {
    buffers: [Bound; BUFFER_SLOTS as usize],
}

impl Default for TableFacts {
    fn default() -> TableFacts {
        TableFacts {
            buffers: std::array::from_fn(|_| Bound::Nothing),
        }
    }
}

impl TableFacts {
    /// Let go of every binding: a new pass's table.
    pub(super) fn clear(&mut self) {
        self.buffers.fill(Bound::Nothing);
    }

    /// Whether a buffer bound holds more than `bytes` past its bind offset.
    pub(super) fn holds_past(&self, bytes: u64) -> bool {
        self.buffers.iter().any(|bound| match bound {
            Bound::Buffer(buffer, offset) => buffer.length().saturating_sub(*offset) > bytes,
            _ => false,
        })
    }

    /// Whether a visible-function table, or the null table an offset made of one, is bound.
    pub(super) fn holds_tables(&self) -> bool {
        (self.buffers.iter()).any(|bound| matches!(bound, Bound::Table | Bound::Null))
    }

    /// `setBuffer:offset:atIndex:` and its stride form, as `ArgumentTable::set_buffer_with_stride`
    /// refuses it.
    pub(super) fn set_buffer(
        &mut self,
        device: &Device,
        call: &'static str,
        (buffer, offset): (Option<&Buffer>, u64),
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        if let Some(buffer) = buffer {
            check_bound_buffer(device, call, (buffer, offset), false)?;
        }
        self.buffers[index as usize] = match buffer {
            Some(buffer) => Bound::Buffer(buffer.clone(), offset),
            None => Bound::Nothing,
        };
        Ok(())
    }

    /// `setBuffers:offsets:withRange:` and its stride form (`strides`, their count), as
    /// `ArgumentTable::set_buffers_with_strides` refuses it.
    pub(super) fn set_buffers(
        &mut self,
        device: &Device,
        call: &'static str,
        (buffers, offsets, strides): (&[Option<Buffer>], &[u64], Option<usize>),
        first: u32,
    ) -> Result<(), Error> {
        if buffers.len() != offsets.len() || strides.is_some_and(|s| s != buffers.len()) {
            return Err(invalid(call, Problem::LengthMismatch));
        }
        check_range(call, first, buffers.len(), BUFFER_SLOTS)?;
        for (buffer, &offset) in buffers.iter().zip(offsets) {
            if let Some(buffer) = buffer {
                check_bound_buffer(device, call, (buffer, offset), false)?;
            }
        }
        for ((buffer, &offset), index) in buffers.iter().zip(offsets).zip(first..) {
            self.buffers[index as usize] = match buffer {
                Some(buffer) => Bound::Buffer(buffer.clone(), offset),
                None => Bound::Nothing,
            };
        }
        Ok(())
    }

    /// `setBufferOffset:atIndex:` and its stride form, as
    /// `ArgumentTable::set_buffer_offset_with_stride` refuses it in a render stage's table.
    pub(super) fn set_buffer_offset(
        &mut self,
        call: &'static str,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let bound = &mut self.buffers[index as usize];
        match bound {
            // Inline bytes then read as zeros, at any offset, until the index is bound again.
            Bound::Bytes => Ok(()),
            // A table reads as the null table until the index is bound again.
            Bound::Table => {
                *bound = Bound::Null;
                Ok(())
            }
            Bound::Buffer(buffer, at) => {
                let length = buffer.length();
                if offset > length {
                    return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
                }
                *at = offset;
                Ok(())
            }
            Bound::Nothing | Bound::Null => Err(invalid(call, Problem::NoBufferBound { index })),
        }
    }

    /// `setBytes:length:atIndex:` and its forms, of `length` bytes, as `ArgumentTable::set_bytes`
    /// refuses it.
    pub(super) fn set_bytes(
        &mut self,
        call: &'static str,
        length: usize,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let length = length as u64;
        if length > INLINE_BYTES_LIMIT {
            return Err(invalid(
                call,
                Problem::InlineBytesLength {
                    length,
                    limit: INLINE_BYTES_LIMIT,
                },
            ));
        }
        self.buffers[index as usize] = Bound::Bytes;
        Ok(())
    }

    /// `setVisibleFunctionTables:withBufferRange:` and its one-table form, as
    /// `ArgumentTable::set_function_tables` refuses it.
    pub(super) fn set_function_tables<'a>(
        &mut self,
        device: &Device,
        call: &'static str,
        tables: impl ExactSizeIterator<Item = Option<&'a VisibleFunctionTable>> + Clone,
        first: u32,
    ) -> Result<(), Error> {
        check_range(call, first, tables.len(), BUFFER_SLOTS)?;
        for table in tables.clone().flatten() {
            check_device(device, call, table.device())?;
        }
        for (table, index) in tables.zip(first..) {
            self.buffers[index as usize] = match table {
                Some(_) => Bound::Table,
                None => Bound::Nothing,
            };
        }
        Ok(())
    }
}

/// `setTextures:withRange:` and its one-texture form, as `ArgumentTable::set_textures` refuses
/// it: no table fact changes.
pub(super) fn check_textures<'a>(
    device: &Device,
    call: &'static str,
    textures: impl ExactSizeIterator<Item = Option<&'a Texture>> + Clone,
    first: u32,
) -> Result<(), Error> {
    check_range(call, first, textures.len(), TEXTURE_SLOTS)?;
    for texture in textures.flatten() {
        check_device(device, call, texture.device())?;
        texture.check_not_memoryless(call)?;
    }
    Ok(())
}

/// `setSamplerStates:withRange:` and its forms (`clamps`, the minimums' and maximums' counts), as
/// `ArgumentTable::set_samplers` refuses it: no table fact changes.
pub(super) fn check_samplers<'a>(
    device: &Device,
    call: &'static str,
    samplers: impl ExactSizeIterator<Item = Option<&'a SamplerState>> + Clone,
    clamps: Option<(usize, usize)>,
    first: u32,
) -> Result<(), Error> {
    if let Some((min, max)) = clamps {
        if min != samplers.len() || max != samplers.len() {
            return Err(invalid(call, Problem::LengthMismatch));
        }
    }
    check_range(call, first, samplers.len(), SAMPLER_SLOTS)?;
    for sampler in samplers.flatten() {
        check_device(device, call, sampler.device())?;
    }
    Ok(())
}
