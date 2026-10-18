//! A blit between two two-plane extended-range textures, as Metal's blit moves their bytes.
//!
//! An `RGB10A8_2P_XR10` or `RGB10A8_2P_XR10_sRGB` texture of width `w` holds plane 0 (each
//! texel's color word, 4 bytes) in rows `align(4 w, 16)` bytes apart and its alpha plane (a byte
//! a texel) in rows `align(w, 16)` bytes apart: what a shader, `getBytes` and `replaceRegion`
//! read and write. Metal's blits address both planes as if their rows were
//! `⌊4 w / 16⌋ · 16` bytes apart ([`PlaneRows::blit`]; measured on an Apple M4, every width from 1
//! to 21 and 23 to 257 sampled). A copy of `sw x sh` texels from `(sx, sy)` of a source to
//! `(dx, dy)` of a destination moves, for each row `r < sh` and column `c < sw`:
//!
//! - the 4 color bytes at `(dy + r) · Bd + 4 (dx + c)` of the destination's plane 0 from
//!   `(sy + r) · Bs + 4 (sx + c)` of the source's;
//! - the alpha byte at `(dy + r) · Bd + dx + c` of the destination's alpha plane from
//!   `(sy + r) · Bs + sx + c` of the source's,
//!
//! where `Bs` and `Bd` are the two textures' blit rows. A byte of a plane lands on the texel whose
//! row and column hold it, and on none in a row's padding or past the plane's last row; a source
//! byte there reads as 0 here. Where rows of the copy overlap in the destination (`4 sw > Bd`, or
//! `sw > Bd` for the alpha bytes), the later row's bytes are the ones kept. At widths that are a
//! multiple of 4 plane 0's rows are the blit's, so its texels copy as a copy does, and only the
//! alpha bytes move.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`PlaneCopyModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`PlaneCopyModule::SOURCE_BINDING`] | the source ([`PlaneSource`]): its level's texels, two words each, row after row, or its planes' bytes |
//! | [`PlaneCopyModule::DESTINATION_BINDING`] | the destination level's texels, the same, read and written |
//! | [`PlaneCopyModule::SHAPE_BINDING`] | the copy's shape ([`PlaneCopyModule::shape`]) |
//!
//! Invocation `i`, plus the shape's first texel, below the destination's texels takes the
//! destination texel `(i % w, i / w)` (`i` counted with the first texel):
//! the copy's row and column whose color bytes land on it, and those whose alpha byte does, each
//! the later row where two do; and from the source texel those bytes are read from, its first or
//! second word (0 where they lie in padding or past the plane), written over the texel's first or
//! second word. A source given as its planes' bytes ([`PlaneSource::Bytes`], a texture over a
//! surface's planes, which Metal's blit reads as they lie) gives those bytes instead, its blit rows
//! its own, the color word whole and the alpha byte alone, 0 at or past the binding's end. A word
//! no row lands on stays. One past the texels moves nothing. The consumer keeps each binding within
//! `maxStorageBufferRange` and each dispatch within `maxComputeWorkGroupCount[0]` workgroups,
//! dispatching a destination of more texels in batches, each with the shape of its first texel.
//!
//! **Synchronization** (consumer): the copies of both levels (or the source's bytes) into the buffers and the shape's
//! update (`TRANSFER_WRITE`) → `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER`
//! before the dispatch; its writes → `TRANSFER_READ` before the destination's texels are copied
//! back into its image.

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
};

use super::PlaneSource;

/// The rows of a two-plane texture of one width, in bytes (the module docs).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneRows {
    /// Plane 0's rows apart: `align(4 w, 16)`.
    pub color: u32,
    /// The alpha plane's rows apart: `align(w, 16)`.
    pub alpha: u32,
    /// The rows Metal's blits address both planes by: `⌊4 w / 16⌋ · 16`, 0 below a width of 4.
    pub blit: u32,
}

impl PlaneRows {
    /// The rows of a texture `width` texels wide.
    pub fn of(width: u32) -> PlaneRows {
        PlaneRows {
            color: (4 * width).next_multiple_of(16),
            alpha: width.next_multiple_of(16),
            blit: 4 * width / 16 * 16,
        }
    }
}

/// The crate-owned plane copy module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneCopyModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl PlaneCopyModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source: its level's texels, two words each, or its planes' bytes ([`PlaneSource`]):
    /// a storage buffer.
    pub const SOURCE_BINDING: u32 = 0;
    /// The destination level's texels, two words each: a storage buffer, read and written.
    pub const DESTINATION_BINDING: u32 = 1;
    /// The copy's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = 19;
    /// Invocations of one workgroup, along x: one destination texel each.
    pub const LOCAL_SIZE: u32 = 64;

    /// The shape's bytes of a copy of `size` texels (width, height; neither 0) from `source` (a
    /// level's texels and the copy's first texel there, or the planes' bytes from the copy's first
    /// byte of each) to `to` of a destination level of `destination` texels, for the dispatch
    /// whose invocation 0 takes the destination's texel `first`. The copy lies inside both levels.
    pub fn shape(
        source: &PlaneSource,
        destination: [u32; 2],
        to: [u32; 2],
        size: [u32; 2],
        first: u32,
    ) -> [u8; 76] {
        let d = PlaneRows::of(destination[0]);
        // The source's level, rows, blit rows, first color and alpha bytes, form and end.
        let (level, rows, blit, base, texels, end) = match *source {
            PlaneSource::Texels { level, from } => {
                let s = PlaneRows::of(level[0]);
                let at = from[1] * s.blit;
                (
                    level,
                    [s.color, s.alpha],
                    s.blit,
                    [at + 4 * from[0], at + from[0]],
                    1,
                    0,
                )
            }
            PlaneSource::Bytes(bytes) => (
                [0, 0],
                [0, 0],
                bytes.blit,
                [bytes.color, bytes.alpha],
                0,
                bytes.end,
            ),
        };
        let words = [
            destination[0],
            destination[0] * destination[1],
            d.color,
            d.alpha,
            d.blit,
            to[1] * d.blit + 4 * to[0],
            to[1] * d.blit + to[0],
            level[0],
            level[1],
            rows[0],
            rows[1],
            blit,
            base[0],
            base[1],
            size[0],
            size[1],
            first,
            texels,
            end,
        ];
        let mut bytes = [0; 76];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The shape's words, in order (written by [`PlaneCopyModule::shape`]).
mod word {
    pub(super) const DST_WIDTH: u32 = 0;
    pub(super) const DST_TEXELS: u32 = 1;
    pub(super) const DST_COLOR_ROWS: u32 = 2;
    pub(super) const DST_ALPHA_ROWS: u32 = 3;
    pub(super) const DST_BLIT_ROWS: u32 = 4;
    pub(super) const DST_COLOR_BASE: u32 = 5;
    pub(super) const DST_ALPHA_BASE: u32 = 6;
    pub(super) const SRC_WIDTH: u32 = 7;
    pub(super) const SRC_HEIGHT: u32 = 8;
    pub(super) const SRC_COLOR_ROWS: u32 = 9;
    pub(super) const SRC_ALPHA_ROWS: u32 = 10;
    pub(super) const SRC_BLIT_ROWS: u32 = 11;
    pub(super) const SRC_COLOR_BASE: u32 = 12;
    pub(super) const SRC_ALPHA_BASE: u32 = 13;
    pub(super) const WIDTH: u32 = 14;
    pub(super) const HEIGHT: u32 = 15;
    pub(super) const FIRST: u32 = 16;
    pub(super) const SRC_TEXELS: u32 = 17;
    pub(super) const SRC_END: u32 = 18;
}

/// The binding symbol of the source's texels in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the destination's texels in the typed module.
const DESTINATION: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};
/// The binding symbol of the shape in the typed module.
const SHAPE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 2,
};

/// A `Block { u32 words[]; }` storage buffer bound at `sym`.
fn words_buffer(b: &mut ModuleBuilder, sym: ResourceSlot, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(sym, StorageClass::StorageBuffer, block)
}

fn u32_value(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// One plane's side of a copy: its unit (bytes a texel: 4 for plane 0, 1 for the alpha plane),
/// the word of a texel it moves, and the shape's rows and bases of each texture.
struct Plane {
    unit: u32,
    word: u32,
    dst_rows: Value,
    dst_base: Value,
    src_rows: Value,
    src_base: Value,
}

/// The values every plane's side reads.
struct Common {
    zero: Value,
    one: Value,
    x: Value,
    y: Value,
    i: Value,
    dst_blit: Value,
    src_blit: Value,
    src_width: Value,
    src_height: Value,
    width: Value,
    last_row: Value,
    source: Value,
    destination: Value,
    src_texels: Value,
    src_end: Value,
}

/// Move `plane`'s bytes onto the invocation's destination texel (the module docs).
fn move_plane(f: &mut FunctionBuilder<'_>, c: &Common, plane: &Plane) -> BResult<()> {
    let unit = u32_value(f, plane.unit)?;
    // The texel's byte in the destination's plane, and how far past the copy's first byte.
    let row = f.binary(BinOp::IMul, c.y, plane.dst_rows)?;
    let column = f.binary(BinOp::IMul, c.x, unit)?;
    let byte = f.binary(BinOp::IAdd, row, column)?;
    let past_base = f.binary(BinOp::UGreaterThanEqual, byte, plane.dst_base)?;
    let u = f.binary(BinOp::ISub, byte, plane.dst_base)?;
    // The latest row of the copy at or before `u`: ⌊u / Bd⌋ (all rows land on one run where Bd
    // is 0), at most the last row.
    let no_rows = f.binary(BinOp::IEqual, c.dst_blit, c.zero)?;
    let divisor = f.select(no_rows, c.one, c.dst_blit)?;
    let q = f.binary(BinOp::UDiv, u, divisor)?;
    let below_last = f.binary(BinOp::ULessThan, q, c.last_row)?;
    let r = f.select(below_last, q, c.last_row)?;
    let r = f.select(no_rows, c.last_row, r)?;
    let into_row = f.binary(BinOp::IMul, r, c.dst_blit)?;
    let k = f.binary(BinOp::ISub, u, into_row)?;
    let run = f.binary(BinOp::IMul, c.width, unit)?;
    let in_run = f.binary(BinOp::ULessThan, k, run)?;
    let written = f.binary(BinOp::LogicalAnd, past_base, in_run)?;
    // The source byte the copy reads for it: the texel holding it (texels), or the word holding
    // it and its place there (bytes).
    let source_row = f.binary(BinOp::IMul, r, c.src_blit)?;
    let s = f.binary(BinOp::IAdd, plane.src_base, source_row)?;
    let s = f.binary(BinOp::IAdd, s, k)?;
    // The rows are 0 in the bytes form, which divides by 1 instead.
    let rows = f.select(c.src_texels, plane.src_rows, c.one)?;
    let ys = f.binary(BinOp::UDiv, s, rows)?;
    let xs = f.binary(BinOp::UMod, s, rows)?;
    let xs = f.binary(BinOp::UDiv, xs, unit)?;
    let in_x = f.binary(BinOp::ULessThan, xs, c.src_width)?;
    let in_y = f.binary(BinOp::ULessThan, ys, c.src_height)?;
    let texel = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let at = f.binary(BinOp::IMul, ys, c.src_width)?;
    let at = f.binary(BinOp::IAdd, at, xs)?;
    let at = f.binary(BinOp::ShiftLeftLogical, at, c.one)?;
    let word = u32_value(f, plane.word)?;
    let texel_word = f.binary(BinOp::IAdd, at, word)?;
    let two = u32_value(f, 2)?;
    let three = u32_value(f, 3)?;
    let eight = u32_value(f, 8)?;
    let byte_word = f.binary(BinOp::ShiftRightLogical, s, two)?;
    let in_bytes = f.binary(BinOp::ULessThan, s, c.src_end)?;
    let lane = f.binary(BinOp::BitwiseAnd, s, three)?;
    let shift = f.binary(BinOp::IMul, lane, eight)?;
    let index = f.select(c.src_texels, texel_word, byte_word)?;
    let inside = f.select(c.src_texels, texel, in_bytes)?;
    let shift = f.select(c.src_texels, c.zero, shift)?;
    let read = f.binary(BinOp::LogicalAnd, written, inside)?;
    let at = f.select(read, index, c.zero)?;
    let from = f.access_chain(c.source, &[c.zero, at])?;
    let value = f.load(from, MemAccess::default())?;
    let value = f.binary(BinOp::ShiftRightLogical, value, shift)?;
    // The bytes form's alpha byte alone (a texel's second word holds it alone).
    let value = if plane.unit == 1 {
        let ff = u32_value(f, 0xff)?;
        let byte = f.binary(BinOp::BitwiseAnd, value, ff)?;
        f.select(c.src_texels, value, byte)?
    } else {
        value
    };
    let value = f.select(read, value, c.zero)?;
    // The destination texel's word, replaced where a row lands on it.
    let mine = f.binary(BinOp::ShiftLeftLogical, c.i, c.one)?;
    let mine = f.binary(BinOp::IAdd, mine, word)?;
    let to = f.access_chain(c.destination, &[c.zero, mine])?;
    let kept = f.load(to, MemAccess::default())?;
    let value = f.select(written, value, kept)?;
    f.store(to, value, MemAccess::default())
}

/// The module as a typed module, before serialization: the module [`plane_copy_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `SOURCE`,
/// `DESTINATION` and `SHAPE` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let source = words_buffer(&mut b, SOURCE, u32_)?;
    let destination = words_buffer(&mut b, DESTINATION, u32_)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let invocation = f.composite_extract(invocation, &[0])?;
    let source = f.module().global_value(source)?;
    let destination = f.module().global_value(destination)?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = u32_value(f, index)?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let first = field(&mut f, word::FIRST)?;
    let i = f.binary(BinOp::IAdd, invocation, first)?;
    let dst_width = field(&mut f, word::DST_WIDTH)?;
    let texels = field(&mut f, word::DST_TEXELS)?;
    let inside = f.binary(BinOp::ULessThan, i, texels)?;
    let dst_color = field(&mut f, word::DST_COLOR_ROWS)?;
    let dst_alpha = field(&mut f, word::DST_ALPHA_ROWS)?;
    let dst_blit = field(&mut f, word::DST_BLIT_ROWS)?;
    let dst_color_base = field(&mut f, word::DST_COLOR_BASE)?;
    let dst_alpha_base = field(&mut f, word::DST_ALPHA_BASE)?;
    let src_width = field(&mut f, word::SRC_WIDTH)?;
    let src_height = field(&mut f, word::SRC_HEIGHT)?;
    let src_color = field(&mut f, word::SRC_COLOR_ROWS)?;
    let src_alpha = field(&mut f, word::SRC_ALPHA_ROWS)?;
    let src_blit = field(&mut f, word::SRC_BLIT_ROWS)?;
    let src_color_base = field(&mut f, word::SRC_COLOR_BASE)?;
    let src_alpha_base = field(&mut f, word::SRC_ALPHA_BASE)?;
    let width = field(&mut f, word::WIDTH)?;
    let height = field(&mut f, word::HEIGHT)?;
    let src_texels = field(&mut f, word::SRC_TEXELS)?;
    let src_texels = f.binary(BinOp::INotEqual, src_texels, zero)?;
    let src_end = field(&mut f, word::SRC_END)?;
    let moved: ArmFn<'_, ()> = Box::new(move |f, _| {
        let x = f.binary(BinOp::UMod, i, dst_width)?;
        let y = f.binary(BinOp::UDiv, i, dst_width)?;
        let last_row = f.binary(BinOp::ISub, height, one)?;
        let common = Common {
            zero,
            one,
            x,
            y,
            i,
            dst_blit,
            src_blit,
            src_width,
            src_height,
            width,
            last_row,
            source,
            destination,
            src_texels,
            src_end,
        };
        let color = Plane {
            unit: 4,
            word: 0,
            dst_rows: dst_color,
            dst_base: dst_color_base,
            src_rows: src_color,
            src_base: src_color_base,
        };
        let alpha = Plane {
            unit: 1,
            word: 1,
            dst_rows: dst_alpha,
            dst_base: dst_alpha_base,
            src_rows: src_alpha,
            src_base: src_alpha_base,
        };
        move_plane(f, &common, &color)?;
        move_plane(f, &common, &alpha)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, moved, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: PlaneCopyModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(PlaneCopyModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(PlaneCopyModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == DESTINATION => Some(PlaneCopyModule::DESTINATION_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(PlaneCopyModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The plane copy module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`PlaneCopyModule`].
pub fn plane_copy_module(device: &Device) -> Result<PlaneCopyModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the plane copy module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(PlaneCopyModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
