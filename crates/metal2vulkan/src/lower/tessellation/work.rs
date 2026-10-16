//! The words of an emulated tessellation draw's work buffer (`SyntheticBinding::TessellationWork`,
//! `lower::tessellation`'s module docs), each a `u32`: a header of [`HEADER`] words, then one slot
//! of [`SLOT_WORDS`] words per chunk of the draw (its `VkDrawIndirectCommand`, then the first
//! record of the chunk).
//!
//! The draw's records (`SyntheticBinding::TessellationRecords`) are [`RECORD_WORDS`] words each,
//! one per instance and patch, instance-major: record `r = i · patchCount + p` is patch `p` of
//! instance `i` (Metal draws every patch of one instance before the next instance's). A record
//! holds the patch's processed factors (its first six words, the tessellator's; a triangle's
//! first four), the draw's triangles before it modulo `2^32` ([`FIRST_TRIANGLE`]) and its stream
//! state ([`STREAM_STATE`]).
//!
//! A chunk is [`CHUNK_TRIANGLES`] triangles of the draw: chunk `k` draws the triangles
//! `[k · C, (k + 1) · C)` as one `vkCmdDrawIndirect`, so that no draw's vertex count passes a
//! `u32` and no vertex index needs more than 32 bits.
//!
//! The binder, the tessellation scan (`bind::tessellation_scan`), the runtime and the vertex
//! prologue read these words at run time, so this file is a runtime input
//! (`validation/build.rs`).

/// `patchCount`: the binder's for a direct draw; for an indirect one, Metal's arguments
/// (`MTLDrawPatchIndirectArguments`, whose four words are these first four in their order), which
/// the consumer copies here before the scan.
pub const PATCH_COUNT: u32 = 0;
/// `instanceCount`, as [`PATCH_COUNT`].
pub const INSTANCE_COUNT: u32 = 1;
/// `patchStart`, as [`PATCH_COUNT`].
pub const PATCH_START: u32 = 2;
/// `baseInstance`, as [`PATCH_COUNT`].
pub const BASE_INSTANCE: u32 = 3;
/// `R`: the records the scan wrote (`patchCount · instanceCount`, or 0 for a draw that draws
/// nothing).
pub const RECORDS: u32 = 4;
/// The word of the records binding where the draw's records begin: 0 for records of the draw's
/// own, the scan's reservation in the command buffer's records pool otherwise.
pub const RECORDS_BASE: u32 = 5;
/// The words of the command buffer's records pool, for records reserved there; 0 for records of
/// the draw's own.
pub const POOL_WORDS: u32 = 6;
/// The consumer's number for the draw, which a records fault reports.
pub const DRAW_ID: u32 = 7;
/// The factor buffer's residual bind offset, in bytes (even).
pub const FACTOR_OFFSET: u32 = 8;
/// The factor buffer's instance stride, in bytes (a multiple of 4).
pub const FACTOR_STRIDE: u32 = 9;
/// The factor scale's `float` bits, as the binder pushes it to a control stage: the scale rounded
/// to `half`, an infinite one as 2^32 (1.0 for a pipeline that does not enable the scale).
pub const SCALE: u32 = 10;
/// `K`: the chunk slots after the header.
pub const CHUNKS: u32 = 11;
/// The header's words.
pub const HEADER: u32 = 16;

/// The words of one chunk's slot.
pub const SLOT_WORDS: u32 = 8;
/// A slot's `VkDrawIndirectCommand`: four words from the slot's first (its vertex count 0 for a
/// chunk past the draw's triangles).
pub const SLOT_DRAW: u32 = 0;
/// A slot's first record: the record that holds the chunk's first triangle.
pub const SLOT_FIRST_RECORD: u32 = 4;

/// The words of one record.
pub const RECORD_WORDS: u32 = 8;
/// A record's word holding the draw's triangles before it, modulo `2^32`.
pub const FIRST_TRIANGLE: u32 = 6;
/// A record's stream state: the tile word the tessellator's `stream_step` gives the record at the
/// batch fill its fold over the records before it reached, which the decode reads.
pub const STREAM_STATE: u32 = 7;

/// `C`: the triangles of one chunk, `2^30`: `3 · C` vertices lie within a `u32`.
pub const CHUNK_TRIANGLES: u64 = 1 << 30;
/// `log2(C)`.
pub const CHUNK_SHIFT: u32 = 30;

/// The words of a work buffer of `chunks` chunks.
pub fn words(chunks: u32) -> u64 {
    u64::from(HEADER) + u64::from(SLOT_WORDS) * u64::from(chunks)
}

/// The byte of the work buffer where chunk `k`'s `VkDrawIndirectCommand` lies.
pub fn draw_offset(k: u32) -> u64 {
    4 * (u64::from(HEADER) + u64::from(SLOT_WORDS) * u64::from(k) + u64::from(SLOT_DRAW))
}

/// The chunks a draw of at most `triangles` triangles needs: at least 1.
pub fn chunks(triangles: u64) -> u64 {
    triangles.div_ceil(CHUNK_TRIANGLES).max(1)
}
