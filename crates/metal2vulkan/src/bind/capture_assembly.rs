//! The tile reach primitives of a pass's capture draws: the step between capture replay and
//! [`tile_reach_module`](super::tile_reach_module).
//!
//! Capture replay draws each kept draw of a pass again before it, under the capture form of its
//! vertex function, which writes one record a vertex invocation. [`capture_assembly_module`]
//! assembles those records into the pass's primitives as Metal's input assembly does, one tile
//! reach [`primitive`](super::TileReachModule::primitive) record each, which
//! [`tile_reach_module`](super::tile_reach_module) then bins; [`capture_settle_module`] turns the
//! mask it binned into every tile where it counted a primitive near-far or unmeasured (the
//! announced fallback).
//!
//! **The assembly module** (one per device). One `GLCompute` entry `main` with local size
//! ([`CaptureAssemblyModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! - [`RECORDS_BINDING`](CaptureAssemblyModule::RECORDS_BINDING): the capture records, each
//!   [`RECORD_WORDS`](CaptureAssemblyModule::RECORD_WORDS) words
//!   ([`record`](CaptureAssemblyModule::record)): the clip-space position (four `f32`s), the point
//!   size (an `f32`), and a word that is not zero where the vertex's raw index restarts its strip
//!   (an indexed draw's index of the index type's all-ones value), then two unused words. The
//!   capture form writes vertex `p` of instance `n` (counted from the draw's first instance) of a
//!   draw of `count` vertices or indices at the draw's first slot plus `p + n * count`.
//! - [`DRAWS_BINDING`](CaptureAssemblyModule::DRAWS_BINDING): a header of
//!   [`HEADER_WORDS`](CaptureAssemblyModule::HEADER_WORDS) words
//!   ([`header`](CaptureAssemblyModule::header)): the entries' count and the primitives' total;
//!   then the entries, each [`DRAW_WORDS`](CaptureAssemblyModule::DRAW_WORDS) words
//!   ([`draw`](CaptureAssemblyModule::draw)), one per instance of each draw, in order:
//!
//!   | Word | Contents |
//!   |---|---|
//!   | 0 | the entry's first primitive: the primitives of the entries before it |
//!   | 1 | its primitives ([`primitives`](CaptureAssemblyModule::primitives) of its topology and records) |
//!   | 2 | its first record: the slot of the instance's vertex 0 |
//!   | 3 | its records: the draw's vertex or index count |
//!   | 4 | its topology: the raw `MTLPrimitiveType` ([`PrimitiveType`]) |
//!   | 5 | 1 where it restarts: an indexed line or triangle strip (Metal restarts strips at a flagged record; lists and points draw every index), else 0 |
//!   | 6 | 1 where it culls: its cull mode is front or back, else 0 |
//!   | 7 | the index of the tile reach state it was drawn under |
//!
//! - [`PRIMITIVES_BINDING`](CaptureAssemblyModule::PRIMITIVES_BINDING): the pass's primitives,
//!   [`PRIMITIVE_WORDS`](super::TileReachModule::PRIMITIVE_WORDS) words each, which the kernel
//!   writes, the buffer [`tile_reach_module`](super::tile_reach_module) reads.
//!
//! **One entry per instance.** An entry covers one instance, so a primitive's place in its
//! entry is a subtraction: the kernel divides nothing (no integer division by a value it reads,
//! which [`tile_reach_module`](super::tile_reach_module)'s docs explain), and a draw of `n`
//! instances takes `n` entries. The host writes no entry of no primitives, so the entries never
//! outnumber the primitives: the draws' buffer holds at most half the primitives' bytes, plus
//! its header.
//!
//! **Invocation `i`** below the header's total (and the primitives the binding holds) finds its
//! entry by a binary search over the first primitives (the last entry whose first primitive is
//! at most `i`), takes `j = i` less that, and writes primitive `i` from the instance's records at
//! the entry's first record plus:
//!
//! - a point list: `j`, a [`POINT`](super::TileReachModule::POINT) of that record's point size;
//! - a line list: `2j`, `2j + 1`; a line strip: `j`, `j + 1` (each a
//!   [`LINE`](super::TileReachModule::LINE));
//! - a triangle list: `3j`, `3j + 1`, `3j + 2`; a triangle strip: `j`, `j + 1`, `j + 2` where the
//!   triangle's parity is even, `j`, `j + 2`, `j + 1` where it is odd (Vulkan's strip order, the
//!   index rewrite's: every triangle keeps the strip's winding);
//!
//! with the entry's state, layer 0, and the first vertex's point size. A vertex unused by its
//! kind (a point's second and third, a line's third) is written as zeros.
//!
//! **Restart** (an entry that restarts, of a line or triangle strip). A primitive one of whose
//! records is flagged reaches nothing: it is written as a triangle of three equal finite vertices,
//! `(0, 0, 0, 1)`, of size 0. The rule has it reach no tile (zero area) and flags it nothing more
//! than its state flags every primitive drawn under it (inside both depth planes, before the
//! eye, at the viewport's centre; `tile_reach`'s test `assembly`, against the rule and the
//! kernel). A line strip's segment is written so too: a line of no length would count unmeasured.
//! A triangle strip's parity restarts after each flagged record (Metal's, measured,
//! `runtime/src/render/restart.rs`). Where the entry culls, a kept triangle's parity is that of
//! its distance from its piece's first record, the one after the last flagged record before `j`
//! (or the instance's vertex 0 where none is): the kernel scans back at most
//! [`SCAN_LIMIT`](CaptureAssemblyModule::SCAN_LIMIT) records before `j`. Where that passes no
//! flagged record and does not reach vertex 0, the parity is unknown, and the triangle is
//! written with three NaN vertices, which the rule counts unmeasured, so the pass falls back to
//! every tile (the settle module). Where the entry does not cull, a triangle reaches the same
//! tiles in either winding, and its parity is counted from vertex 0, with no scan.
//!
//! **Contained.** Every read and write lies inside its binding. An invocation past the
//! primitives the binding holds writes nothing. A primitive the entries do not describe (no
//! entry, `j` at or past the entry's primitives, a topology that is none) or whose records lie
//! past the entry's records or the records' binding is written as the triangle of NaN vertices:
//! counted unmeasured, never a guess.
//!
//! **The settle module** (one per device). One `GLCompute` entry `main` with local size
//! ([`CaptureSettleModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers in
//! set 0, numbered densely from 0 (as the runtime's owned kernels lay out their set):
//! [`STATUS_BINDING`](CaptureSettleModule::STATUS_BINDING), the command buffer's
//! [`ExecutionStatus`](super::ExecutionStatus) words;
//! [`PASS_BINDING`](CaptureSettleModule::PASS_BINDING), the pass's header and states, read; and
//! [`TILES_BINDING`](CaptureSettleModule::TILES_BINDING), the tile mask (the last two the
//! buffers [`tile_reach_module`](super::tile_reach_module) binds at the same numbers). Where the
//! header's [`NEAR_FAR_WORD`](super::TileReachModule::NEAR_FAR_WORD) or
//! [`UNMEASURED_WORD`](super::TileReachModule::UNMEASURED_WORD) is not zero, invocation `i`, one
//! a word of the mask ([`mask_words`](super::TileReachModule::mask_words) of the header's render
//! area, samples and layers), ORs into word `i` the bits of every tile of the pass it holds (a
//! bit past the pass's last tile is left as it is), and invocation 0 ORs
//! [`CAPTURED_REACH_UNMEASURED`](super::StatusBits::CAPTURED_REACH_UNMEASURED) into the status
//! word, which [`captured_reach_unmeasured`](super::StatusReport::captured_reach_unmeasured)
//! reads back so the runtime can announce the fallback. Where both counts are zero it changes
//! nothing. Dispatch at least one workgroup, so invocation 0 runs.
//!
//! **Arithmetic.** Both kernels copy `f32`s as their bits and hold no float instruction, and
//! compute in `u32` alone: they need nothing of a Vulkan 1.2 device past what every one has.
//!
//! **Synchronization** (consumer): the capture draws' record writes (`SHADER_WRITE` in
//! `VERTEX_SHADER`) and the draws' upload → `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the
//! assembly; its writes → [`tile_reach_module`](super::tile_reach_module)'s reads; the tile
//! reach dispatch's writes → `SHADER_STORAGE_READ` and `SHADER_STORAGE_WRITE` in `COMPUTE_SHADER`
//! before the settle; its writes → the mask's readers, and the status buffer's reader as every
//! writer of it (`bind::status`).

use std::sync::Arc;

use super::index_bounds::finished;
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::Error;
use crate::metal::PrimitiveType;
use crate::pipeline::sym::Symbol;

mod kernel;

/// The word positions within the assembly's records ([`CaptureAssemblyModule`]).
mod layout {
    /// A capture record's: the position (four `f32`s), the point size, the restart word.
    pub(super) const POSITION: u32 = 0;
    pub(super) const POINT_SIZE: u32 = 4;
    pub(super) const RESTART: u32 = 5;
    /// The draws' header's: the entries' count and the primitives' total.
    pub(super) const ENTRIES: u32 = 0;
    pub(super) const TOTAL: u32 = 1;
    /// An entry's (the module docs' table).
    pub(super) const FIRST_PRIMITIVE: u32 = 0;
    pub(super) const PRIMITIVES: u32 = 1;
    pub(super) const FIRST_RECORD: u32 = 2;
    pub(super) const RECORDS: u32 = 3;
    pub(super) const TOPOLOGY: u32 = 4;
    pub(super) const RESTARTS: u32 = 5;
    pub(super) const CULLS: u32 = 6;
    pub(super) const STATE: u32 = 7;
}

/// The clip-space vertex, as `f32` bits, of the triangle a primitive that reaches nothing is
/// written as (the module docs): `(0, 0, 0, 1)`.
pub(in crate::bind) const NOTHING: [u32; 4] = [0, 0, 0, 0x3f80_0000];
/// The clip-space vertex, as `f32` bits, of the triangle a primitive whose reach is unknown is
/// written as (the module docs): four quiet NaNs.
pub(in crate::bind) const UNKNOWN: [u32; 4] = [0x7fc0_0000; 4];

/// The crate-owned capture assembly module of one device (the module docs), validated like any
/// output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureAssemblyModule {
    /// SPIR-V 1.5 words, entry point `main`, its bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: within every Vulkan 1.2 device.
    pub requirements: Requirements,
}

impl CaptureAssemblyModule {
    /// Descriptor set of the bindings.
    pub const SET: u32 = 0;
    /// The capture records, a storage buffer the kernel reads.
    pub const RECORDS_BINDING: u32 = 0;
    /// The draws' header and entries, a storage buffer the kernel reads.
    pub const DRAWS_BINDING: u32 = 1;
    /// The pass's primitives, a storage buffer the kernel writes (the buffer tile reach binds at
    /// [`PRIMITIVES_BINDING`](super::TileReachModule::PRIMITIVES_BINDING)).
    pub const PRIMITIVES_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one primitive each.
    pub const LOCAL_SIZE: u32 = 64;
    /// A capture record's words ([`record`](Self::record)): 32 bytes.
    pub const RECORD_WORDS: u32 = 8;
    /// The draws' header's words ([`header`](Self::header)), before the entries.
    pub const HEADER_WORDS: u32 = 2;
    /// An entry's words ([`draw`](Self::draw)).
    pub const DRAW_WORDS: u32 = 8;
    /// The most records the kernel scans back from a kept triangle of a restarting triangle
    /// strip that culls, for the last flagged record before it (the module docs).
    pub const SCAN_LIMIT: u32 = 256;

    /// A capture record's bytes, as the capture form writes them: the clip-space `position`
    /// (`[[position]]`), the `point_size` (`[[point_size]]`; any value where the function writes
    /// none), and whether the vertex's raw index restarts its strip (1 or 0), then two words of
    /// zeros.
    pub fn record(position: [f32; 4], point_size: f32, restart: bool) -> [u8; 32] {
        let mut words = [0; Self::RECORD_WORDS as usize];
        for (c, value) in position.iter().enumerate() {
            words[layout::POSITION as usize + c] = value.to_bits();
        }
        words[layout::POINT_SIZE as usize] = point_size.to_bits();
        words[layout::RESTART as usize] = u32::from(restart);
        bytes_of(words)
    }

    /// The draws' header's bytes: the count of `entries` that follow it, and the `total` of their
    /// primitives (the invocations that write one).
    pub fn header(entries: u32, total: u32) -> [u8; 8] {
        let mut words = [0; Self::HEADER_WORDS as usize];
        words[layout::ENTRIES as usize] = entries;
        words[layout::TOTAL as usize] = total;
        bytes_of(words)
    }

    /// An entry's bytes, one instance of a draw: its `first_primitive` (the primitives of the
    /// entries before it), the slot of its vertex 0 among the records (`first_record`), its
    /// `records` (the draw's vertex or index count), its `topology`, whether it `restarts` (an
    /// indexed strip, whose flagged records restart it; ignored for a list or points), whether
    /// its cull mode `culls` (front or back), and the index of the tile reach `state` it was drawn
    /// under. Its primitives are [`primitives`](Self::primitives) of its topology and records.
    pub fn draw(
        first_primitive: u32,
        first_record: u32,
        records: u32,
        topology: PrimitiveType,
        restarts: bool,
        culls: bool,
        state: u32,
    ) -> [u8; 32] {
        let mut words = [0; Self::DRAW_WORDS as usize];
        words[layout::FIRST_PRIMITIVE as usize] = first_primitive;
        words[layout::PRIMITIVES as usize] = Self::primitives(topology, records);
        words[layout::FIRST_RECORD as usize] = first_record;
        words[layout::RECORDS as usize] = records;
        words[layout::TOPOLOGY as usize] = topology.as_raw();
        words[layout::RESTARTS as usize] = u32::from(restarts);
        words[layout::CULLS as usize] = u32::from(culls);
        words[layout::STATE as usize] = state;
        bytes_of(words)
    }

    /// The primitives one instance of `records` vertices or indices of `topology` assembles
    /// (with a strip's restarts among them: a primitive a restart drops is written as one that
    /// reaches nothing).
    pub fn primitives(topology: PrimitiveType, records: u32) -> u32 {
        match topology {
            PrimitiveType::Point => records,
            PrimitiveType::Line => records / 2,
            PrimitiveType::LineStrip => records.saturating_sub(1),
            PrimitiveType::Triangle => records / 3,
            PrimitiveType::TriangleStrip => records.saturating_sub(2),
        }
    }
}

/// The crate-owned capture settle module of one device (the module docs), validated like any
/// output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptureSettleModule {
    /// SPIR-V 1.5 words, entry point `main`, its bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device: within every Vulkan 1.2 device.
    pub requirements: Requirements,
}

impl CaptureSettleModule {
    /// Descriptor set of the bindings, numbered densely from 0.
    pub const SET: u32 = 0;
    /// The command buffer's [`ExecutionStatus`](super::ExecutionStatus) words, a storage buffer
    /// the kernel ORs
    /// [`CAPTURED_REACH_UNMEASURED`](super::StatusBits::CAPTURED_REACH_UNMEASURED) into.
    pub const STATUS_BINDING: u32 = 0;
    /// The pass's header and states, a storage buffer the kernel reads (the buffer tile reach
    /// binds at [`PASS_BINDING`](super::TileReachModule::PASS_BINDING)).
    pub const PASS_BINDING: u32 = 1;
    /// The tile mask, a storage buffer the kernel ORs bits into (the buffer tile reach binds at
    /// [`TILES_BINDING`](super::TileReachModule::TILES_BINDING)).
    pub const TILES_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one mask word each.
    pub const LOCAL_SIZE: u32 = 64;
}

/// The little-endian bytes of `N` words.
fn bytes_of<const N: usize, const B: usize>(words: [u32; N]) -> [u8; B] {
    let mut bytes = [0; B];
    for (at, w) in words.iter().enumerate() {
        bytes[4 * at..4 * at + 4].copy_from_slice(&w.to_le_bytes());
    }
    bytes
}

/// The fixed binding numbers of the assembly's symbols.
fn assembly_binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(CaptureAssemblyModule::SET),
        Symbol::Binding(s) if *s == kernel::RECORDS => Some(CaptureAssemblyModule::RECORDS_BINDING),
        Symbol::Binding(s) if *s == kernel::DRAWS => Some(CaptureAssemblyModule::DRAWS_BINDING),
        Symbol::Binding(s) if *s == kernel::PRIMITIVES => {
            Some(CaptureAssemblyModule::PRIMITIVES_BINDING)
        }
        _ => None,
    }
}

/// The fixed binding numbers of the settle's symbols.
fn settle_binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(CaptureSettleModule::SET),
        Symbol::Binding(s) if *s == kernel::PASS => Some(CaptureSettleModule::PASS_BINDING),
        Symbol::Binding(s) if *s == kernel::TILES => Some(CaptureSettleModule::TILES_BINDING),
        Symbol::Binding(s) if *s == kernel::STATUS => Some(CaptureSettleModule::STATUS_BINDING),
        _ => None,
    }
}

/// The capture assembly module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`CaptureAssemblyModule`].
pub fn capture_assembly_module(device: &Device) -> Result<CaptureAssemblyModule, Error> {
    let (spirv, requirements) = finished(
        device,
        kernel::assembly_typed_module()?,
        assembly_binding_of,
        "the capture assembly",
    )?;
    Ok(CaptureAssemblyModule {
        spirv,
        requirements,
    })
}

/// The capture settle module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`CaptureSettleModule`].
pub fn capture_settle_module(device: &Device) -> Result<CaptureSettleModule, Error> {
    let (spirv, requirements) = finished(
        device,
        kernel::settle_typed_module()?,
        settle_binding_of,
        "the capture settle",
    )?;
    Ok(CaptureSettleModule {
        spirv,
        requirements,
    })
}

#[cfg(test)]
mod tests;
