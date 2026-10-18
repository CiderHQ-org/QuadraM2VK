//! A blit into a two-plane extended-range texture whose planes lie in bytes the consumer holds (a
//! heap's, or a surface's), as Metal's blit writes them there.
//!
//! An `RGB10A8_2P_XR10` or `RGB10A8_2P_XR10_sRGB` texture of width `w` holds plane 0 (each
//! texel's color word) in rows `align(4 w, 16)` bytes apart and its alpha plane (a byte a texel)
//! in rows `align(w, 16)` bytes apart. Metal's blits address both planes by rows
//! `⌊4 w / 16⌋ · 16` bytes apart ([`PlaneRows::blit`](super::PlaneRows::blit), `plane_copy`'s
//! model). Where the destination's planes lie in a heap's bytes, the bytes the blit writes are
//! the heap's (an Apple M4 under macOS 27.0.1 and an M3 Max under macOS 26.7, byte for byte alike):
//! a copy of `sw x sh` texels writes, for each row `r < sh` and column `c < sw`, the 4 color bytes
//! at `Dc + r · Bd + 4 c` and the alpha byte at `Da + r · Bd + c`, from the source's at `Sc + r ·
//! Bs + 4 c` and `Sa + r · Bs + c`, where `Dc` and `Da` are the copy's first color and alpha bytes,
//! `Bd` and `Bs` the two textures' blit rows. The alpha bytes land past the alpha plane, on
//! whatever else the bytes hold; the source's bytes are read as they were before the copy. Where
//! the copy's rows overlap in the destination, the later row's bytes are kept here.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`PlaneBytesCopyModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`PlaneBytesCopyModule::SOURCE_BINDING`] | the source: its planes' bytes, or its level's texels, two words each, row after row ([`PlaneSource`]) |
//! | [`PlaneBytesCopyModule::DESTINATION_BINDING`] | the bytes the destination's planes lie in, read and written |
//! | [`PlaneBytesCopyModule::SHAPE_BINDING`] | the copy's shape ([`PlaneBytesCopyModule::shape`]) |
//!
//! Invocation `i`, plus the shape's first invocation, takes:
//!
//! - for `i < sw · sh`, the color word of row `i / sw` and column `i % sw` of the copy: the
//!   source's word written over the destination's, unless a later row of the copy lands on it
//!   (`4 c ≥ Bd` below the last row) or it lies at or past the destination's end;
//! - for `sw · sh ≤ i < sw · sh + n · sh`, word `k` of the `n` words row `r` of the copy's alpha
//!   bytes touch (`i - sw · sh` counted row after row): the bytes of the row that land in it, each
//!   from the source's byte, unless a later row lands on it (`c ≥ Bd` below the last row); the
//!   word's other bytes stay, and a word at or past the destination's end is not written.
//!
//! One past both moves nothing. Two invocations never write one word: rows' alpha bytes lie at
//! least 9 bytes apart at a blit row of 16 or more, and at a blit row of 0 only the last row
//! writes. The consumer keeps each binding within `maxStorageBufferRange` and each dispatch
//! within `maxComputeWorkGroupCount[0]` workgroups, dispatching more invocations in batches, each
//! with the shape of its first invocation.
//!
//! **Synchronization** (consumer): the writes of the source's binding and of the shape
//! (`TRANSFER_WRITE`), and every earlier write of the destination's bytes →
//! `SHADER_STORAGE_READ | SHADER_STORAGE_WRITE` in `COMPUTE_SHADER` before the dispatch; its
//! writes → whatever reads the destination's bytes next.

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

use super::PlaneRows;

/// The planes of one side of a copy in a binding's bytes, as Metal's blit addresses them
/// ([`PlaneBytesCopyModule::shape`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneBytes {
    /// The copy's first color byte, from the binding's start: a multiple of 4.
    pub color: u32,
    /// The copy's first alpha byte, from the binding's start.
    pub alpha: u32,
    /// The texture's blit rows ([`PlaneRows::blit`]).
    pub blit: u32,
    /// The binding's bytes that hold the planes' side, a multiple of 4: a source byte at or past
    /// it reads 0, and a destination word there is not written.
    pub end: u32,
}

impl PlaneBytes {
    /// The planes of a side of a copy (the fields' docs).
    pub fn new(color: u32, alpha: u32, blit: u32, end: u32) -> PlaneBytes {
        PlaneBytes {
            color,
            alpha,
            blit,
            end,
        }
    }
}

/// Where the source of a copy lies in the source's binding.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum PlaneSource {
    /// Its planes' bytes, as Metal lays them.
    Bytes(PlaneBytes),
    /// Its level's texels, two words each (the color word, then the alpha byte in the second
    /// word's low byte), row after row: the level's texels across and down, and the copy's
    /// origin. The copy reads the planes' bytes at Metal's blit rows of the level; one in a row's
    /// padding or past a plane reads 0.
    Texels {
        /// The level's texels across and down.
        level: [u32; 2],
        /// The copy's first texel.
        from: [u32; 2],
    },
}

/// The crate-owned plane bytes copy module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneBytesCopyModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl PlaneBytesCopyModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source: a storage buffer ([`PlaneSource`]).
    pub const SOURCE_BINDING: u32 = 0;
    /// The bytes the destination's planes lie in: a storage buffer, read and written.
    pub const DESTINATION_BINDING: u32 = 1;
    /// The copy's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 2;
    /// The words of a shape.
    pub const SHAPE_WORDS: u32 = 18;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;

    /// The words one row of a copy's alpha bytes touch in the destination: a row of `width`
    /// bytes from `destination`'s first alpha byte.
    pub fn alpha_words(destination: &PlaneBytes, width: u32) -> u32 {
        (destination.alpha % 4 + width).div_ceil(4)
    }

    /// The invocations a copy of `size` texels (width, height) into `destination` takes.
    pub fn invocations(destination: &PlaneBytes, size: [u32; 2]) -> u64 {
        let rows = u64::from(size[1]);
        rows * (u64::from(size[0]) + u64::from(Self::alpha_words(destination, size[0])))
    }

    /// The shape's bytes of a copy of `size` texels (width, height; neither 0) from `source`
    /// into `destination`, for the dispatch whose invocation 0 is the copy's invocation `first`.
    pub fn shape(
        source: &PlaneSource,
        destination: &PlaneBytes,
        size: [u32; 2],
        first: u32,
    ) -> [u8; 72] {
        let [width, height] = size;
        let (texels, src, level) = match *source {
            PlaneSource::Bytes(bytes) => (0, bytes, [0, 0]),
            PlaneSource::Texels { level, from } => {
                let blit = PlaneRows::of(level[0]).blit;
                let at = from[1] * blit;
                (
                    1,
                    PlaneBytes::new(at + 4 * from[0], at + from[0], blit, 0),
                    level,
                )
            }
        };
        let rows = PlaneRows::of(level[0]);
        let words = [
            width * height,
            width,
            height,
            destination.color,
            destination.alpha,
            destination.blit,
            destination.end,
            Self::alpha_words(destination, width),
            texels,
            src.color,
            src.alpha,
            src.blit,
            src.end,
            level[0],
            level[1],
            rows.color,
            rows.alpha,
            first,
        ];
        let mut bytes = [0; 72];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The shape's words, in order (written by [`PlaneBytesCopyModule::shape`]).
mod word {
    pub(super) const COLOR_WORDS: u32 = 0;
    pub(super) const WIDTH: u32 = 1;
    pub(super) const HEIGHT: u32 = 2;
    pub(super) const DST_COLOR: u32 = 3;
    pub(super) const DST_ALPHA: u32 = 4;
    pub(super) const DST_BLIT: u32 = 5;
    pub(super) const DST_END: u32 = 6;
    pub(super) const ALPHA_WORDS: u32 = 7;
    pub(super) const SOURCE_TEXELS: u32 = 8;
    pub(super) const SRC_COLOR: u32 = 9;
    pub(super) const SRC_ALPHA: u32 = 10;
    pub(super) const SRC_BLIT: u32 = 11;
    pub(super) const SRC_END: u32 = 12;
    pub(super) const SRC_WIDTH: u32 = 13;
    pub(super) const SRC_HEIGHT: u32 = 14;
    pub(super) const SRC_COLOR_ROWS: u32 = 15;
    pub(super) const SRC_ALPHA_ROWS: u32 = 16;
    pub(super) const FIRST: u32 = 17;
}

/// The binding symbol of the source in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the destination's bytes in the typed module.
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

/// The shape's source words and the values every read of the source takes.
#[derive(Clone, Copy)]
struct Source {
    zero: Value,
    one: Value,
    two: Value,
    three: Value,
    eight: Value,
    words: Value,
    texels: Value,
    blit: Value,
    end: Value,
    width: Value,
    height: Value,
}

/// One plane's side of the source: its unit (4 for plane 0, 1 for the alpha plane), the word of
/// a texel it lies in, its first byte of the copy, and its level's rows (texel form).
#[derive(Clone, Copy)]
struct Plane {
    unit: u32,
    word: u32,
    base: Value,
    rows: Value,
}

/// The source's bytes of `plane` at row `r` and column `c` of the copy, at the low end of the
/// value: the whole word for plane 0, the byte (with the word's higher bytes, which the caller
/// masks) for the alpha plane; 0 where the source holds none there.
fn read(
    f: &mut FunctionBuilder<'_>,
    s: &Source,
    plane: &Plane,
    r: Value,
    c: Value,
) -> BResult<Value> {
    let unit = u32_value(f, plane.unit)?;
    let row = f.binary(BinOp::IMul, r, s.blit)?;
    let column = f.binary(BinOp::IMul, c, unit)?;
    let at = f.binary(BinOp::IAdd, plane.base, row)?;
    let at = f.binary(BinOp::IAdd, at, column)?;
    // Bytes: the word holding the byte, and the byte's place in it.
    let byte_word = f.binary(BinOp::ShiftRightLogical, at, s.two)?;
    let in_bytes = f.binary(BinOp::ULessThan, at, s.end)?;
    let lane = f.binary(BinOp::BitwiseAnd, at, s.three)?;
    let shift = f.binary(BinOp::IMul, lane, s.eight)?;
    // Texels: the texel the byte lies on at the level's own rows, and its word (the rows are
    // 0 in the bytes form, which divides by 1 instead).
    let rows = f.select(s.texels, plane.rows, s.one)?;
    let y = f.binary(BinOp::UDiv, at, rows)?;
    let x = f.binary(BinOp::UMod, at, rows)?;
    let x = f.binary(BinOp::UDiv, x, unit)?;
    let in_x = f.binary(BinOp::ULessThan, x, s.width)?;
    let in_y = f.binary(BinOp::ULessThan, y, s.height)?;
    let in_texels = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let texel = f.binary(BinOp::IMul, y, s.width)?;
    let texel = f.binary(BinOp::IAdd, texel, x)?;
    let texel_word = f.binary(BinOp::ShiftLeftLogical, texel, s.one)?;
    let word = u32_value(f, plane.word)?;
    let texel_word = f.binary(BinOp::IAdd, texel_word, word)?;
    // One of the two, by the source's form.
    let index = f.select(s.texels, texel_word, byte_word)?;
    let inside = f.select(s.texels, in_texels, in_bytes)?;
    let shift = f.select(s.texels, s.zero, shift)?;
    let index = f.select(inside, index, s.zero)?;
    let from = f.access_chain(s.words, &[s.zero, index])?;
    let value = f.load(from, MemAccess::default())?;
    let value = f.binary(BinOp::ShiftRightLogical, value, shift)?;
    f.select(inside, value, s.zero)
}

/// The values both sides of an invocation take.
#[derive(Clone, Copy)]
struct Common {
    zero: Value,
    two: Value,
    four: Value,
    destination: Value,
    dst_blit: Value,
    dst_end: Value,
    width: Value,
    last_row: Value,
}

/// Whether row `r` of the copy keeps its byte at `k` bytes into the row: no later row lands
/// there (`k < Bd`, or the last row).
fn kept(f: &mut FunctionBuilder<'_>, c: &Common, r: Value, k: Value) -> BResult<Value> {
    let last = f.binary(BinOp::IEqual, r, c.last_row)?;
    let before_next = f.binary(BinOp::ULessThan, k, c.dst_blit)?;
    f.binary(BinOp::LogicalOr, last, before_next)
}

/// Write the color word of copy texel `j` (the module docs).
fn color(
    f: &mut FunctionBuilder<'_>,
    c: &Common,
    s: &Source,
    plane: &Plane,
    base: Value,
    j: Value,
) -> BResult<()> {
    let r = f.binary(BinOp::UDiv, j, c.width)?;
    let x = f.binary(BinOp::UMod, j, c.width)?;
    let k = f.binary(BinOp::IMul, x, c.four)?;
    let keep = kept(f, c, r, k)?;
    let row = f.binary(BinOp::IMul, r, c.dst_blit)?;
    let at = f.binary(BinOp::IAdd, base, row)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let inside = f.binary(BinOp::ULessThan, at, c.dst_end)?;
    let write = f.binary(BinOp::LogicalAnd, keep, inside)?;
    let value = read(f, s, plane, r, x)?;
    let index = f.binary(BinOp::ShiftRightLogical, at, c.two)?;
    let index = f.select(write, index, c.zero)?;
    let to = f.access_chain(c.destination, &[c.zero, index])?;
    let written: ArmFn<'_, ()> = Box::new(move |f, _| {
        f.store(to, value, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), write, written, None)?;
    Ok(())
}

/// Write the alpha bytes of row word `j` (the module docs).
fn alpha(
    f: &mut FunctionBuilder<'_>,
    c: &Common,
    s: &Source,
    plane: &Plane,
    base: Value,
    words: Value,
    j: Value,
) -> BResult<()> {
    let r = f.binary(BinOp::UDiv, j, words)?;
    let q = f.binary(BinOp::UMod, j, words)?;
    let row = f.binary(BinOp::IMul, r, c.dst_blit)?;
    let start = f.binary(BinOp::IAdd, base, row)?;
    let first = f.binary(BinOp::ShiftRightLogical, start, c.two)?;
    let index = f.binary(BinOp::IAdd, first, q)?;
    let word_at = f.binary(BinOp::ShiftLeftLogical, index, c.two)?;
    let ff = u32_value(f, 0xff)?;
    let mut mask = c.zero;
    let mut bits = c.zero;
    for lane in 0..4u32 {
        let lane_v = u32_value(f, lane)?;
        let at = f.binary(BinOp::IAdd, word_at, lane_v)?;
        let past_start = f.binary(BinOp::UGreaterThanEqual, at, start)?;
        let k = f.binary(BinOp::ISub, at, start)?;
        let in_row = f.binary(BinOp::ULessThan, k, c.width)?;
        let in_row = f.binary(BinOp::LogicalAnd, past_start, in_row)?;
        let keep = kept(f, c, r, k)?;
        let mine = f.binary(BinOp::LogicalAnd, in_row, keep)?;
        let k = f.select(in_row, k, c.zero)?;
        let value = read(f, s, plane, r, k)?;
        let value = f.binary(BinOp::BitwiseAnd, value, ff)?;
        let shift = u32_value(f, 8 * lane)?;
        let value = f.binary(BinOp::ShiftLeftLogical, value, shift)?;
        let lane_mask = f.binary(BinOp::ShiftLeftLogical, ff, shift)?;
        let value = f.select(mine, value, c.zero)?;
        let lane_mask = f.select(mine, lane_mask, c.zero)?;
        bits = f.binary(BinOp::BitwiseOr, bits, value)?;
        mask = f.binary(BinOp::BitwiseOr, mask, lane_mask)?;
    }
    let any = f.binary(BinOp::INotEqual, mask, c.zero)?;
    let inside = f.binary(BinOp::ULessThan, word_at, c.dst_end)?;
    let write = f.binary(BinOp::LogicalAnd, any, inside)?;
    let index = f.select(write, index, c.zero)?;
    let to = f.access_chain(c.destination, &[c.zero, index])?;
    let written: ArmFn<'_, ()> = Box::new(move |f, _| {
        let old = f.load(to, MemAccess::default())?;
        let all = u32_value(f, u32::MAX)?;
        let keep = f.binary(BinOp::BitwiseXor, mask, all)?;
        let old = f.binary(BinOp::BitwiseAnd, old, keep)?;
        let value = f.binary(BinOp::BitwiseOr, old, bits)?;
        f.store(to, value, MemAccess::default())?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), write, written, None)?;
    Ok(())
}

/// The module as a typed module, before serialization: the module [`plane_bytes_copy_module`]
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
    let two = u32_value(&mut f, 2)?;
    let three = u32_value(&mut f, 3)?;
    let four = u32_value(&mut f, 4)?;
    let eight = u32_value(&mut f, 8)?;
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
    let color_words = field(&mut f, word::COLOR_WORDS)?;
    let width = field(&mut f, word::WIDTH)?;
    let height = field(&mut f, word::HEIGHT)?;
    let dst_color = field(&mut f, word::DST_COLOR)?;
    let dst_alpha = field(&mut f, word::DST_ALPHA)?;
    let dst_blit = field(&mut f, word::DST_BLIT)?;
    let dst_end = field(&mut f, word::DST_END)?;
    let alpha_words = field(&mut f, word::ALPHA_WORDS)?;
    let source_texels = field(&mut f, word::SOURCE_TEXELS)?;
    let src_color = field(&mut f, word::SRC_COLOR)?;
    let src_alpha = field(&mut f, word::SRC_ALPHA)?;
    let src_blit = field(&mut f, word::SRC_BLIT)?;
    let src_end = field(&mut f, word::SRC_END)?;
    let src_width = field(&mut f, word::SRC_WIDTH)?;
    let src_height = field(&mut f, word::SRC_HEIGHT)?;
    let src_color_rows = field(&mut f, word::SRC_COLOR_ROWS)?;
    let src_alpha_rows = field(&mut f, word::SRC_ALPHA_ROWS)?;
    let texels = f.binary(BinOp::INotEqual, source_texels, zero)?;
    let last_row = f.binary(BinOp::ISub, height, one)?;
    let alpha_count = f.binary(BinOp::IMul, alpha_words, height)?;
    let all = f.binary(BinOp::IAdd, color_words, alpha_count)?;
    let is_color = f.binary(BinOp::ULessThan, i, color_words)?;
    let is_alpha = f.binary(BinOp::ULessThan, i, all)?;
    let s = Source {
        zero,
        one,
        two,
        three,
        eight,
        words: source,
        texels,
        blit: src_blit,
        end: src_end,
        width: src_width,
        height: src_height,
    };
    let c = Common {
        zero,
        two,
        four,
        destination,
        dst_blit,
        dst_end,
        width,
        last_row,
    };
    let color_plane = Plane {
        unit: 4,
        word: 0,
        base: src_color,
        rows: src_color_rows,
    };
    let alpha_plane = Plane {
        unit: 1,
        word: 1,
        base: src_alpha,
        rows: src_alpha_rows,
    };
    let colors: ArmFn<'_, ()> = Box::new(move |f, _| {
        color(f, &c, &s, &color_plane, dst_color, i)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    let alphas: ArmFn<'_, ()> = Box::new(move |f, _| {
        let alpha_arm: ArmFn<'_, ()> = Box::new(move |f, _| {
            let j = f.binary(BinOp::ISub, i, color_words)?;
            alpha(f, &c, &s, &alpha_plane, dst_alpha, alpha_words, j)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), is_alpha, alpha_arm, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), is_color, colors, Some(alphas))?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: PlaneBytesCopyModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(PlaneBytesCopyModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(PlaneBytesCopyModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == DESTINATION => Some(PlaneBytesCopyModule::DESTINATION_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(PlaneBytesCopyModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The plane bytes copy module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`PlaneBytesCopyModule`].
pub fn plane_bytes_copy_module(device: &Device) -> Result<PlaneBytesCopyModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the plane bytes copy module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(PlaneBytesCopyModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
