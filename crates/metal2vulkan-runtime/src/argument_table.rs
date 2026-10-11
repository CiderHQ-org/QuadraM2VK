//! One stage's argument table: the buffers with their offsets, inline bytes, textures, and sampler
//! states with their LOD clamps an encoder binds by Metal index. A compute encoder has one, a
//! render encoder one per stage (vertex and fragment); every `set_*` call of either is one of these
//! methods, with the same refusals.
//!
//! Inline bytes (`setBytes:length:atIndex:` and its forms) are copied at the call into the command
//! buffer's upload arena (`upload.rs`), as Metal copies them, so the caller's memory may change at
//! once; the bind layer states them as `BoundBuffer::Bytes`, and the binder binds the copy where
//! the plan writes `WriteSource::InlineBytes` or a vertex buffer.
//!
//! What a bound texture's usage allows, and what a sampler's clamps sample, is checked at the
//! command that uses the slot, against the function's use of it; binding checks only the index
//! range, the object's device, a buffer offset against the buffer's length, and that a texture
//! is not memoryless (Metal's validation stops binding one, RT6 probes 6b and 6c).
//!
//! A compute encoder's table differs in two answers, as plain Metal's compute encoder gave them
//! on an M4 (EV `SWP-misuse1-m4`) where its API validation asserts: it holds a buffer bound past
//! its length, which a dispatch refuses only where the kernel reads the slot (a read there is past
//! the buffer: memory decides it), and it takes an offset set at an index with no buffer as no
//! change. A render stage's table refuses both at the call, as Metal's validation does; plain
//! Metal's render forms are not measured.

use super::address_map::SamplerMaps;
use super::binding::bound_sampler;
use super::buffer::Buffer;
use super::device::lock;
use super::encoder::EncoderCore;
use super::error::{invalid, Error, Problem};
use super::function_table::{FunctionTableProblem, VisibleFunctionTable};
use super::live_bind::Reshape;
use super::sampler::SamplerState;
use super::texture::Texture;
use super::upload::Upload;
use metal2vulkan::bind::{BoundBuffer, BoundSampler, BoundTexture, StageTable};
use metal2vulkan::pipeline::{
    BufferKind, PipelineInterface, SlotClass, SlotInterface, SlotRef, Stage,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, Weak};

/// Entries of a buffer table (`setBuffer:offset:atIndex:` and its stage forms).
pub(super) const BUFFER_SLOTS: u32 = 31;
/// Entries of a texture table (`setTexture:atIndex:` and its stage forms).
pub(super) const TEXTURE_SLOTS: u32 = 128;
/// Entries of a sampler table (`setSamplerState:atIndex:` and its stage forms).
pub(super) const SAMPLER_SLOTS: u32 = 16;
/// `MTLAttributeStrideStatic`: the stride argument that sets no per-bind stride, so the buffer
/// is read at its layout's own stride, as by the plain setters.
pub(super) const ATTRIBUTE_STRIDE_STATIC: u64 = u64::MAX;
/// The most bytes one `setBytes:length:atIndex:` (or a stage form) takes: Metal's documented
/// limit for inline bytes, 4 KiB.
pub(super) const INLINE_BYTES_LIMIT: u64 = 4096;

/// One stage's bound objects, by Metal index.
#[derive(Clone, Default)]
pub(super) struct ArgumentTable {
    /// A compute encoder's table ([`ArgumentTable::compute`]): it holds buffers bound past their
    /// length and takes an offset at an index with no buffer as no change (the module docs).
    compute: bool,
    /// Each buffer with its bind offset; a visible-function table's words at offset 0.
    pub(super) buffers: BTreeMap<u32, (Buffer, u64)>,
    /// The visible-function tables among `buffers`, by the index they share with buffers: a
    /// later buffer at the index replaces the table, and a later table the buffer, as on Metal.
    pub(super) function_tables: BTreeMap<u32, VisibleFunctionTable>,
    /// The indices where a visible-function table was bound and an offset then set
    /// (`setBufferOffset:` and its forms): plain Metal reads the table there as the null table,
    /// of size 0 and every entry null, until a table or a buffer is bound at the index again.
    /// Neither `buffers` nor `function_tables` holds them.
    pub(super) null_tables: BTreeSet<u32>,
    /// The per-bind attribute strides of buffers in `buffers`, by index
    /// (`setBuffer:offset:attributeStride:atIndex:` and its forms): a layout of
    /// `MTLBufferLayoutStrideDynamic` reads its buffer at this stride. The plain setters bind
    /// with `MTLAttributeStrideStatic`, which sets none.
    pub(super) attribute_strides: BTreeMap<u32, u32>,
    pub(super) textures: BTreeMap<u32, Texture>,
    /// Each sampler state with the LOD clamps its bind set
    /// (`setSamplerState:lodMinClamp:lodMaxClamp:atIndex:`), which replace the state's.
    pub(super) samplers: BTreeMap<u32, (SamplerState, Option<[f32; 2]>)>,
    /// The bytes of `setBytes:length:atIndex:` and its forms, by index: copied at the call
    /// ([`set_bytes`](Self::set_bytes)). Neither `buffers` nor `function_tables` holds the index.
    pub(super) bytes: BTreeMap<u32, InlineBytes>,
    /// What the table bound as an encoder's core last retained it ([`retain`](Self::retain)).
    retained: Retained,
}

/// The core an argument table was last retained into ([`EncoderCore::id`]), and the objects it
/// bound then, which that core holds ([`ArgumentTable::retain`]): held weakly, so the table never
/// holds an object longer than the core does.
#[derive(Default)]
struct Retained(Mutex<Option<(u64, Weak<RetainedObjects>)>>);

impl Clone for Retained {
    fn clone(&self) -> Retained {
        Retained(Mutex::new(lock(&self.0).clone()))
    }
}

/// The objects an argument table bound, which an encoder's core holds until its command buffer
/// completes ([`ArgumentTable::retain`]), each in the table's order.
struct RetainedObjects {
    buffers: Vec<Buffer>,
    textures: Vec<Texture>,
    samplers: Vec<SamplerState>,
}

impl RetainedObjects {
    /// Whether `table` binds exactly these objects (by identity), in its order.
    fn bound_by(&self, table: &ArgumentTable) -> bool {
        self.buffers
            .iter()
            .eq(table.buffers.values().map(|(b, _)| b))
            && self.textures.iter().eq(table.textures.values())
            && self
                .samplers
                .iter()
                .eq(table.samplers.values().map(|(s, _)| s))
    }
}

/// One `set*Bytes` call's copy of its bytes: a region of the command buffer's upload arena
/// (`upload.rs`) written at the call, so a later write of the caller's memory reaches no
/// command, as Metal copies the bytes at the call.
#[derive(Clone)]
pub(super) struct InlineBytes {
    /// The region: the bytes, then zeros to a multiple of 16 bytes, at least 16.
    pub(super) region: Upload,
    /// The call's length.
    pub(super) len: u32,
    /// The stride form's stride, its low 32 bits (`MTLAttributeStrideStatic` is `u32::MAX`);
    /// `None` for the plain setters (`BoundBuffer::Bytes`).
    pub(super) attribute_stride: Option<u32>,
}

/// Refuse `index` past a table of `limit` entries.
pub(super) fn check_index(call: &'static str, index: u32, limit: u32) -> Result<(), Error> {
    if index < limit {
        Ok(())
    } else {
        Err(invalid(call, Problem::IndexOutOfRange { index, limit }))
    }
}

/// Refuse `count` entries from `first` past a table of `limit` entries.
pub(super) fn check_range(
    call: &'static str,
    first: u32,
    count: usize,
    limit: u32,
) -> Result<(), Error> {
    if count == 0 {
        return Ok(());
    }
    let count = u32::try_from(count).unwrap_or(u32::MAX);
    check_index(call, first.saturating_add(count - 1), limit)
}

/// The per-bind stride `stride` names: none for `MTLAttributeStrideStatic`, else its low 32
/// bits. Plain Metal reads a stride past 32 bits so, and its validation takes it (an M4 under macOS
/// 27.0.1 and an M3 Max under 26.7, compute stage-in and vertex fetch, both binding forms: `2^32 +
/// 8` and `2^63 + 8` read at 8, `2^33 + 4` at 4, `2^32` at 0; EVs RRP-probe1-m4, RRP-probe1-m3,
/// RRP-probe2-m4).
fn attribute_stride(stride: u64) -> Option<u32> {
    // Truncation is the measured answer.
    (stride != ATTRIBUTE_STRIDE_STATIC).then_some(stride as u32)
}

/// Bind `value` at `index` of `table`, or unbind the index when `value` is `None`.
fn set_entry<T>(table: &mut BTreeMap<u32, T>, index: u32, value: Option<T>) {
    match value {
        Some(value) => {
            table.insert(index, value);
        }
        None => {
            table.remove(&index);
        }
    }
}

/// `buffer` bound at `offset`, read at `attribute_stride`, as the bind layer reads it
/// ([`ArgumentTable::stage_table`]), or `None` where it is bound past its length.
fn bound_buffer(
    buffer: &Buffer,
    offset: u64,
    attribute_stride: Option<u32>,
) -> Option<BoundBuffer> {
    if offset > buffer.length() {
        return None;
    }
    let base = buffer.base();
    Some(BoundBuffer::Buffer {
        length: base + buffer.length(),
        offset: base + offset,
        gpu_address: buffer.gpu_address().0 - base,
        device_address: buffer.device_address().map(|address| address - base),
        attribute_stride,
    })
}

/// Inline bytes as the bind layer reads them.
fn bound_bytes(bytes: &InlineBytes) -> BoundBuffer {
    BoundBuffer::Bytes {
        len: bytes.len,
        attribute_stride: bytes.attribute_stride,
    }
}

/// `texture` as the bind layer reads it.
fn bound_texture(texture: &Texture) -> BoundTexture {
    match texture.texels() {
        Some(texels) => texels.bound(texture.pixel_format(), texture.record().sampled_swizzle()),
        // Its resource ID makes it a member of the handle tables that reach its slot: an argument
        // buffer can name it too.
        None => BoundTexture::Texture {
            facts: texture.facts(),
            resource_id: Some(texture.gpu_resource_id().0),
        },
    }
}

impl ArgumentTable {
    /// A compute encoder's table (the module docs).
    pub(super) fn compute() -> ArgumentTable {
        ArgumentTable {
            compute: true,
            ..ArgumentTable::default()
        }
    }

    /// Refuse `buffer` at `offset` in `call` of an encoder of `core`: a buffer of another device,
    /// and, in a render stage's table, an offset past its length (a compute table holds it).
    fn check_buffer(
        &self,
        core: &EncoderCore,
        call: &'static str,
        buffer: &Buffer,
        offset: u64,
    ) -> Result<(), Error> {
        core.check_device(call, buffer.device())?;
        let length = buffer.length();
        if offset > length && !self.compute {
            return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
        }
        Ok(())
    }

    /// `setBuffer:offset:atIndex:` (`None` unbinds the index) in an encoder of `core`: the
    /// stride form with `MTLAttributeStrideStatic`. Refused: an index past the table, a buffer of
    /// another device, and in a render stage's table an offset past the buffer's length.
    pub(super) fn set_buffer(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer_with_stride(core, call, buffer, offset, ATTRIBUTE_STRIDE_STATIC, index)
    }

    /// `setBuffer:offset:attributeStride:atIndex:` (`None` unbinds the index) in an encoder of
    /// `core`, a stride past 32 bits taken modulo `2^32`. Refused as
    /// [`set_buffer`](Self::set_buffer) is.
    pub(super) fn set_buffer_with_stride(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        buffer: Option<&Buffer>,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let stride = attribute_stride(stride);
        match buffer {
            Some(buffer) => {
                self.check_buffer(core, call, buffer, offset)?;
                self.buffers.insert(index, (buffer.clone(), offset));
            }
            None => {
                self.buffers.remove(&index);
            }
        }
        set_entry(&mut self.attribute_strides, index, stride);
        self.function_tables.remove(&index);
        self.null_tables.remove(&index);
        self.bytes.remove(&index);
        Ok(())
    }

    /// `setBuffers:offsets:withRange:`: `buffers[i]` at offset `offsets[i]` at index `first + i`,
    /// every stride `MTLAttributeStrideStatic`. Refused as [`set_buffer`](Self::set_buffer) is,
    /// and when the slices' lengths differ; a refusal binds none of them.
    pub(super) fn set_buffers(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        first: u32,
    ) -> Result<(), Error> {
        self.set_buffers_with_strides(core, call, buffers, offsets, None, first)
    }

    /// `setBuffers:offsets:attributeStrides:withRange:`: as [`set_buffers`](Self::set_buffers),
    /// each buffer at stride `strides[i]` (modulo `2^32`) where `strides` is given. Refused as
    /// `set_buffers` is.
    pub(super) fn set_buffers_with_strides(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        buffers: &[Option<&Buffer>],
        offsets: &[u64],
        strides: Option<&[u64]>,
        first: u32,
    ) -> Result<(), Error> {
        if buffers.len() != offsets.len() || strides.is_some_and(|s| s.len() != buffers.len()) {
            return Err(invalid(call, Problem::LengthMismatch));
        }
        check_range(call, first, buffers.len(), BUFFER_SLOTS)?;
        for (buffer, &offset) in buffers.iter().zip(offsets) {
            if let Some(buffer) = buffer {
                self.check_buffer(core, call, buffer, offset)?;
            }
        }
        let strides = (0..buffers.len()).map(|i| strides.and_then(|s| attribute_stride(s[i])));
        for (((buffer, &offset), stride), index) in
            buffers.iter().zip(offsets).zip(strides).zip(first..)
        {
            set_entry(
                &mut self.buffers,
                index,
                buffer.map(|buffer| (buffer.clone(), offset)),
            );
            set_entry(&mut self.attribute_strides, index, stride);
            self.function_tables.remove(&index);
            self.null_tables.remove(&index);
            self.bytes.remove(&index);
        }
        Ok(())
    }

    /// `setBufferOffset:atIndex:`: the stride form with `MTLAttributeStrideStatic`, so it clears
    /// the index's stride, as the plain buffer setters do. Measured for the vertex form on an M4
    /// (EV `CFIT-metal-probe-3`): after a stride form, Metal's validation stops the draw for
    /// "providing `MTLAttributeStrideStatic` to `attributeStride` when stride at buffer-layout
    /// index 0 is dynamic", and Metal without it reads the buffer at stride 0. Every stage's form
    /// sets this one table. At an index holding a visible-function table, plain Metal then reads
    /// the null table there, of size 0 and every entry null, at any offset, 0 included, until a
    /// table or buffer is bound again (an M4 under macOS 27.0.1: the compute form at offsets 0 to
    /// 4096, the vertex form at 4 to 24; a second offset keeps it null; EVs RRP-probe1-m4,
    /// RRP-probe2-m4; Metal's debug layer raises an exception there), and so does the runtime. At
    /// an index holding inline bytes, they read as zeros at any offset until the index is bound
    /// again, as plain Metal reads them (an M4, EV `APX-bytes1-m4`). Refused: an index past the
    /// table, no buffer bound at the index in a render stage's table (a compute table's index with
    /// none keeps no offset), and in a render stage's table an offset past the buffer's length.
    pub(super) fn set_buffer_offset(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        offset: u64,
        index: u32,
    ) -> Result<(), Error> {
        self.set_buffer_offset_with_stride(core, call, offset, ATTRIBUTE_STRIDE_STATIC, index)
    }

    /// `setBufferOffset:attributeStride:atIndex:` with `stride` (`MTLAttributeStrideStatic`
    /// clears the index's stride; a stride past 32 bits is taken modulo `2^32`). Refused as
    /// [`set_buffer_offset`](Self::set_buffer_offset) is.
    pub(super) fn set_buffer_offset_with_stride(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        offset: u64,
        stride: u64,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let stride = attribute_stride(stride);
        if let Some(len) = self.bytes.get(&index).map(|bytes| bytes.len) {
            // Inline bytes then read as zeros, at any offset, until the index is bound again:
            // plain Metal on an M4 read 64 zero bytes after `setBytes` of 64 then an offset of
            // 0, 16 or 128, in the compute, vertex and fragment forms (EV `APX-bytes1-m4`,
            // twice each; its debug layer asserts "index(0) must have an existing buffer").
            let zeros = [0u8; INLINE_BYTES_LIMIT as usize];
            let padded = u64::from(len).next_multiple_of(16).max(16);
            let region = core.command_buffer().state().uploads.upload(
                core.device(),
                &zeros[..len as usize],
                padded,
            )?;
            self.bytes.insert(
                index,
                InlineBytes {
                    region,
                    len,
                    attribute_stride: stride,
                },
            );
            return Ok(());
        }
        if self.function_tables.remove(&index).is_some() {
            // The table reads as the null table until the index is bound again.
            self.buffers.remove(&index);
            self.attribute_strides.remove(&index);
            self.null_tables.insert(index);
            return Ok(());
        }
        let compute = self.compute;
        let bound = match self.buffers.get_mut(&index) {
            Some(entry) => entry,
            // Plain Metal's compute encoder keeps no offset where no buffer is bound: a later
            // `setBuffer:offset:atIndex:` states its own (`set_offset_unbound_then_bind`).
            None if compute => return Ok(()),
            None => return Err(invalid(call, Problem::NoBufferBound { index })),
        };
        let length = bound.0.length();
        if offset > length && !compute {
            return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
        }
        bound.1 = offset;
        set_entry(&mut self.attribute_strides, index, stride);
        Ok(())
    }

    /// `setBytes:length:atIndex:` (`stride` `None`) and `setBytes:length:attributeStride:atIndex:`
    /// (and every stage's forms) in an encoder of `core`: `bytes` copied now into the command
    /// buffer's upload arena, then zeros to a multiple of 16 bytes, in place of anything bound at
    /// `index`. A layout of `MTLBufferLayoutStrideDynamic` reads them at the stride form's
    /// stride, its low 32 bits, and at 0 after a plain setter (`BoundBuffer::Bytes`). Refused: an
    /// index past the table, and more than [`INLINE_BYTES_LIMIT`] bytes.
    pub(super) fn set_bytes(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        bytes: &[u8],
        stride: Option<u64>,
        index: u32,
    ) -> Result<(), Error> {
        check_index(call, index, BUFFER_SLOTS)?;
        let length = bytes.len() as u64;
        if length > INLINE_BYTES_LIMIT {
            return Err(invalid(
                call,
                Problem::InlineBytesLength {
                    length,
                    limit: INLINE_BYTES_LIMIT,
                },
            ));
        }
        // The bind layer binds at most the bytes padded to 16, at least 16.
        let padded = length.next_multiple_of(16).max(16);
        let region = core
            .command_buffer()
            .state()
            .uploads
            .upload(core.device(), bytes, padded)?;
        self.buffers.remove(&index);
        self.attribute_strides.remove(&index);
        self.function_tables.remove(&index);
        self.null_tables.remove(&index);
        self.bytes.insert(
            index,
            InlineBytes {
                region,
                // At most `INLINE_BYTES_LIMIT`.
                len: length as u32,
                // Truncation is the measured answer for a buffer's (`attribute_stride`).
                attribute_stride: stride.map(|stride| stride as u32),
            },
        );
        Ok(())
    }

    /// `setTextures:withRange:` (and `setTexture:atIndex:` as a range of one): `textures[i]` at
    /// index `first + i`. Refused: an index past the table, a texture of another device, a
    /// memoryless texture (measured: "texture is Memoryless, and cannot be assigned"); a refusal
    /// binds none of them.
    pub(super) fn set_textures(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        textures: &[Option<&Texture>],
        first: u32,
    ) -> Result<(), Error> {
        check_range(call, first, textures.len(), TEXTURE_SLOTS)?;
        for texture in textures.iter().flatten() {
            core.check_device(call, texture.device())?;
            texture.check_not_memoryless(call)?;
        }
        for (texture, index) in textures.iter().zip(first..) {
            set_entry(&mut self.textures, index, texture.cloned());
        }
        Ok(())
    }

    /// Bind `samplers` from index `first`, with `clamps` (minimums and maximums) when given.
    /// Refused: slices of different lengths, an index past the table, a state of another device;
    /// a refusal binds none of them. Metal takes any clamps; a sampler whose clamps are NaN or
    /// inverted is refused at the command that samples through it.
    pub(super) fn set_samplers(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        samplers: &[Option<&SamplerState>],
        clamps: Option<(&[f32], &[f32])>,
        first: u32,
    ) -> Result<(), Error> {
        if let Some((min, max)) = clamps {
            if min.len() != samplers.len() || max.len() != samplers.len() {
                return Err(invalid(call, Problem::LengthMismatch));
            }
        }
        check_range(call, first, samplers.len(), SAMPLER_SLOTS)?;
        for sampler in samplers.iter().flatten() {
            core.check_device(call, sampler.device())?;
        }
        for (i, (sampler, index)) in samplers.iter().zip(first..).enumerate() {
            let clamp = clamps.map(|(min, max)| [min[i], max[i]]);
            set_entry(
                &mut self.samplers,
                index,
                sampler.map(|sampler| (sampler.clone(), clamp)),
            );
        }
        Ok(())
    }

    /// `setVisibleFunctionTables:withBufferRange:` (and `setVisibleFunctionTable:atBufferIndex:`
    /// as a range of one): `tables[i]` at buffer index `first + i`, in place of any buffer there;
    /// `None` unbinds the index. The table's words are the buffer the slot reads. Refused: an
    /// index past the buffer table, a table of another device; a refusal binds none of them.
    pub(super) fn set_function_tables(
        &mut self,
        core: &EncoderCore,
        call: &'static str,
        tables: &[Option<&VisibleFunctionTable>],
        first: u32,
    ) -> Result<(), Error> {
        check_range(call, first, tables.len(), BUFFER_SLOTS)?;
        for table in tables.iter().flatten() {
            core.check_device(call, table.device())?;
        }
        for (table, index) in tables.iter().zip(first..) {
            set_entry(
                &mut self.buffers,
                index,
                table.map(|table| (table.words().clone(), 0)),
            );
            set_entry(&mut self.function_tables, index, table.cloned());
            self.attribute_strides.remove(&index);
            self.null_tables.remove(&index);
            self.bytes.remove(&index);
        }
        Ok(())
    }

    /// Refuse a command of `call` whose `stage` function writes a buffer, or reads an
    /// intersection function table, where a visible-function table is bound, by `interface`'s
    /// slots: plain Metal's answer was not measured (ray tracing's tables are out of scope). A
    /// table bound where the function only reads a buffer reads as an unbound buffer, and a buffer
    /// bound where it reads a visible-function table as a null table (`stage_table_reading`). A
    /// table an offset made null (`null_tables`) where the function reads a buffer is refused too:
    /// not measured.
    pub(super) fn check_function_tables(
        &self,
        call: &'static str,
        stage: Stage,
        interface: &PipelineInterface,
    ) -> Result<(), Error> {
        for (slot, slot_interface) in &interface.slots {
            let SlotInterface::Buffer(buffer) = slot_interface else {
                continue;
            };
            if slot.stage != stage || slot.class != SlotClass::Buffer {
                continue;
            }
            let index = slot.index;
            let table = self.function_tables.contains_key(&index);
            let unmeasured = match buffer.kind {
                BufferKind::Buffer => buffer.access.write,
                BufferKind::VisibleFunctionTable => false,
                _ => true,
            };
            let nulled = self.null_tables.contains(&index)
                && !matches!(buffer.kind, BufferKind::VisibleFunctionTable);
            if (unmeasured && table) || nulled {
                return Err(invalid(
                    call,
                    Problem::FunctionTable(FunctionTableProblem::TableAtBufferIndex { index }),
                ));
            }
        }
        Ok(())
    }

    /// Refuse a command of `call` whose `stage` function reads a buffer slot bound past its
    /// buffer's length (a compute table holds one, the module docs): Metal's read there is past
    /// the buffer, which memory decides. A slot the function does not read runs as Metal's does.
    pub(super) fn check_offsets(
        &self,
        call: &'static str,
        stage: Stage,
        interface: &PipelineInterface,
    ) -> Result<(), Error> {
        for slot in interface.slots.keys() {
            if slot.stage != stage || slot.class != SlotClass::Buffer {
                continue;
            }
            if let Some((buffer, offset)) = self.buffers.get(&slot.index) {
                let (offset, length) = (*offset, buffer.length());
                if offset > length {
                    return Err(invalid(call, Problem::OffsetOutOfRange { offset, length }));
                }
            }
        }
        Ok(())
    }

    /// The table as the bind layer reads it. A buffer is stated in its space (`Buffer::space`),
    /// which the descriptor ranges, vertex buffer offsets and device addresses the bind layer
    /// derives count from: a no-copy buffer's bytes may start past its space's start, where its
    /// length and offset are as many bytes longer and its addresses as many lower, so every Metal
    /// address the layer derives is the buffer's. A heap buffer's space starts at its own byte 0,
    /// so the layer states its bytes alone. A buffer bound past its length is left out: a command
    /// whose function reads its slot is refused ([`check_offsets`](Self::check_offsets)), and no
    /// other reads it.
    pub(super) fn stage_table(&self) -> StageTable {
        let mut table = StageTable::default();
        for (&index, (buffer, offset)) in &self.buffers {
            let stride = self.attribute_strides.get(&index).copied();
            if let Some(bound) = bound_buffer(buffer, *offset, stride) {
                table.buffers.insert(index, bound);
            }
        }
        for (&index, bytes) in &self.bytes {
            table.buffers.insert(index, bound_bytes(bytes));
        }
        for (&index, texture) in &self.textures {
            table.textures.insert(index, bound_texture(texture));
        }
        for (&index, (sampler, clamps)) in &self.samplers {
            table
                .samplers
                .insert(index, BoundSampler::new(bound_sampler(sampler, *clamps)));
        }
        table
    }

    /// The table as `stage`'s function of `interface` reads it: `stage_table`, but a buffer bound
    /// where the function reads a visible-function table is left out, so the slot reads its
    /// neutral, a null table of size 0, as plain Metal reads such a buffer where its validation
    /// stops the command (probe `RT7-ftable-probe1` `query_buffer_at_index`, an M3 Max and an
    /// M4), and so is a table bound where it reads a buffer, whose slot then reads zeros, as
    /// plain Metal's read (EV RFC-probe1-m4, an M4).
    pub(super) fn stage_table_reading(
        &self,
        stage: Stage,
        interface: &PipelineInterface,
    ) -> StageTable {
        let mut table = self.stage_table();
        table
            .buffers
            .retain(|&index, _| !self.reads_unbound((stage, interface), index));
        table
    }

    /// Whether `stage`'s function of `interface` reads buffer index `index` as unbound
    /// ([`stage_table_reading`](Self::stage_table_reading)): a buffer is bound there and the
    /// function reads a visible-function table, or a table is bound there and the function reads a
    /// buffer it does not write, which plain Metal read as zeros, where its validation stops the
    /// command (EV RFC-probe1-m4 `ft_table_at_buffer_compute`, `ft_table_at_buffer_vertex`, EV
    /// RFC-probe2-m4 `ft_table_at_buffer_wide`).
    fn reads_unbound(&self, (stage, interface): (Stage, &PipelineInterface), index: u32) -> bool {
        let slot = SlotRef {
            stage,
            class: SlotClass::Buffer,
            index,
        };
        let Some(SlotInterface::Buffer(buffer)) = interface.slots.get(&slot) else {
            return false;
        };
        let table_bound = self.function_tables.contains_key(&index);
        match buffer.kind {
            BufferKind::VisibleFunctionTable => !table_bound,
            BufferKind::Buffer => table_bound && !buffer.access.write,
            _ => false,
        }
    }

    /// Bring `kept`, the table as a stage's function read it at an earlier draw, to the table as
    /// `stage`'s function of `interface` reads it now
    /// ([`stage_table_reading`](Self::stage_table_reading)), each sampler as `maps` binds it, in
    /// place (`live_bind.rs`): every slot's binding is derived again and stored, so one whose
    /// object changed since with no call (a buffer's pages rebound or detached) is read as it is
    /// now, and `reshape` is told of each change that may change the shape.
    pub(super) fn refresh_reading(
        &self,
        kept: &mut StageTable,
        (stage, interface): (Stage, &PipelineInterface),
        maps: &SamplerMaps,
        reshape: &mut Reshape,
    ) {
        let read = (stage, interface);
        kept.buffers.retain(|&index, _| {
            let bound = !self.reads_unbound(read, index)
                && (self.bytes.contains_key(&index)
                    || self
                        .buffers
                        .get(&index)
                        .is_some_and(|(buffer, offset)| *offset <= buffer.length()));
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, (buffer, offset)) in &self.buffers {
            if self.bytes.contains_key(&index) || self.reads_unbound(read, index) {
                continue;
            }
            let stride = self.attribute_strides.get(&index).copied();
            if let Some(bound) = bound_buffer(buffer, *offset, stride) {
                reshape.slot(&mut kept.buffers, index, bound, BoundBuffer::shape);
            }
        }
        for (&index, bytes) in &self.bytes {
            if !self.reads_unbound(read, index) {
                reshape.slot(
                    &mut kept.buffers,
                    index,
                    bound_bytes(bytes),
                    BoundBuffer::shape,
                );
            }
        }
        kept.textures.retain(|index, _| {
            let bound = self.textures.contains_key(index);
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, texture) in &self.textures {
            reshape.slot(
                &mut kept.textures,
                index,
                bound_texture(texture),
                BoundTexture::shape,
            );
        }
        kept.samplers.retain(|index, _| {
            let bound = self.samplers.contains_key(index);
            if !bound {
                reshape.unbound();
            }
            bound
        });
        for (&index, (sampler, clamps)) in &self.samplers {
            let slot = SlotRef {
                stage,
                class: SlotClass::Sampler,
                index,
            };
            let desc = maps.sampler(slot, bound_sampler(sampler, *clamps));
            reshape.slot(
                &mut kept.samplers,
                index,
                BoundSampler::new(desc),
                BoundSampler::shape,
            );
        }
        // The table read whole, as before (debug builds): the same bindings, a float's sign and
        // a NaN's included (compared as printed).
        #[cfg(debug_assertions)]
        {
            let mut whole = self.stage_table_reading(stage, interface);
            for (&index, bound) in &mut whole.samplers {
                let slot = SlotRef {
                    stage,
                    class: SlotClass::Sampler,
                    index,
                };
                bound.desc = maps.sampler(slot, bound.desc);
            }
            debug_assert_eq!(
                format!("{kept:?}"),
                format!("{whole:?}"),
                "a stage table brought in place that is not the table read whole"
            );
        }
    }

    /// Hold everything bound until `core`'s command buffer completes: a command uses it, and
    /// the command buffer lists it among its used resources (`used_resources.rs`). Once while it
    /// binds the same objects: where `core` retained the table before and it binds exactly the
    /// objects it bound then (by identity), `core` holds them and keeps their keys, each a function
    /// of its object, until the command buffer completes, so nothing is retained again.
    pub(super) fn retain(&self, core: &mut EncoderCore) {
        let mut retained = lock(&self.retained.0);
        let held = retained
            .as_ref()
            .filter(|(id, _)| *id == core.id())
            .and_then(|(_, objects)| objects.upgrade());
        if held.is_some_and(|objects| objects.bound_by(self)) {
            return;
        }
        let keys = self.buffers.values().map(|(b, _)| b.identity().key);
        let texture_keys = self.textures.values().map(Texture::use_key);
        core.touch_keys(keys.chain(texture_keys));
        let objects = Arc::new(RetainedObjects {
            buffers: self.buffers.values().map(|(b, _)| b.clone()).collect(),
            textures: self.textures.values().cloned().collect(),
            samplers: self.samplers.values().map(|(s, _)| s.clone()).collect(),
        });
        *retained = Some((core.id(), Arc::downgrade(&objects)));
        core.retain(objects);
        core.device().shared().counts.argument_table_retained();
    }
}
