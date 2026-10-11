//! The words of an emulated mesh draw's work buffer (`SyntheticBinding::MeshWork`, `lower::mesh`'s
//! module docs), each a `u32`: a header of [`HEADER`] words, then a slot of `S` words per object
//! threadgroup (its grid's three counts and a zero, then its payload), then `O + 1` offsets: the
//! first mesh threadgroup of each object threadgroup, and `T`.
//!
//! The binder, the mesh scan (`bind::mesh_scan`) and the runtime read these words at run time,
//! so this file is a runtime input (`validation/build.rs`); the lowering writes them.

/// Metal's limit on the threadgroups of one mesh grid, 1048576, to which an emulated object
/// function's grid is held: past it Metal defines no answer.
pub const METAL_MESH_GRID_GROUPS: u32 = 1 << 20;

/// `O`: the object threadgroups (1 for a pipeline without an object function).
pub const OBJECT_GROUPS: u32 = 0;
/// `T`: the mesh threadgroups the draw runs, which its records hold.
pub const MESH_GROUPS: u32 = 1;
/// The word of the records binding where the draw's records begin.
pub const RECORDS_BASE: u32 = 2;
/// The mesh dispatch's `VkDispatchIndirectCommand`, three words.
pub const DISPATCH: u32 = 3;
/// The draw's `VkDrawIndirectCommand`, four words.
pub const DRAW: u32 = 6;
/// `S`: the words of one slot.
pub const SLOT_WORDS: u32 = 10;
/// The words of one mesh threadgroup's records (`MeshRecords::group_words`), which the scan
/// reserves for each.
pub const GROUP_WORDS: u32 = 11;
/// The vertices the draw draws for one mesh threadgroup, `NP · k`.
pub const GROUP_VERTICES: u32 = 12;
/// The words of the records pool the scan reserves the draw's records from.
pub const POOL_WORDS: u32 = 13;
/// The consumer's number for the draw, which a records fault reports.
pub const DRAW_ID: u32 = 14;
/// The most workgroups of the mesh dispatch along x (`maxComputeWorkGroupCount[0]`): the
/// scan folds `T` into rows of it.
pub const ROW: u32 = 15;
/// 0 for a draw of host-known object threadgroups; else 1 plus the word of the bound
/// arguments range where Metal's indirect arguments (the mesh grid of a pipeline without an
/// object function) begin, which the scan reads into slot 0.
pub const ARGUMENTS: u32 = 16;
/// The header's words.
pub const HEADER: u32 = 20;
/// The words of a slot before its payload: the grid's three counts and a zero.
pub const GRID: u32 = 4;

/// The words of a work buffer of `objects` slots of `slot_words` words, with its offsets.
pub fn words(objects: u64, slot_words: u32) -> u64 {
    u64::from(HEADER)
        .saturating_add(objects.saturating_mul(u64::from(slot_words)))
        .saturating_add(objects)
        .saturating_add(1)
}
