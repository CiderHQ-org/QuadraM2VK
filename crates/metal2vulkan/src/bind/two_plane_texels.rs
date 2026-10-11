//! A heap's two-plane extended-range texture's texels moved between the image that holds them
//! and its two planes laid out in the heap's bytes as Metal lays them.
//!
//! An `RGB10A8_2P_XR10` or `RGB10A8_2P_XR10_sRGB` texture is one `R32G32_UINT` image
//! (`metal::format::ExtendedRange::Planes`): each texel's color word, then its alpha byte in the
//! second word. In a heap, Metal keeps its texels in the heap's bytes in two planes: plane 0
//! holds each texel's color word, rows apart by a multiple of 16 bytes; plane 1 each texel's
//! alpha byte, rows apart by a multiple of 16, from a multiple of 16 KiB past plane 0's start; an
//! array's slices apart in each plane. A consumer keeps the image a shadow of those bytes: it
//! copies the texture's bytes into a buffer, one dispatch of [`two_plane_texels_module`] moves
//! them into a tight buffer of the image's texels, 8 bytes each, which it copies into the image;
//! and back: the image into the tight buffer, a dispatch into the planes' buffer, and each row
//! of each plane on into the heap.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`TwoPlaneTexelsModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`TwoPlaneTexelsModule::TEXELS_BINDING`] | the image's texels, two words each, row after row, slice after slice |
//! | [`TwoPlaneTexelsModule::PLANES_BINDING`] | the planes' bytes |
//! | [`TwoPlaneTexelsModule::SHAPE_BINDING`] | the move's shape, [`TwoPlaneTexelsModule::SHAPE_WORDS`] words |
//!
//! The shape's words, in order ([`TwoPlaneTexelsModule::shape`]): `width` and `height`, a
//! slice's texels across and down; `texels`, every slice's (`width · height · slices`);
//! `alpha_words`, the words of plane 1's rows (`⌈width / 4⌉` a row, every row of every slice);
//! plane 0's first word, its rows apart and its slices apart, in words; plane 1's first word, its
//! rows apart and its slices apart, in words; `into_texels`, nonzero to move the planes into the
//! texels, else back; and `grid_width`, the invocations of one row of the dispatch's grid (its x
//! groups times the local size). Invocation `(gx, gy)` takes `i = gy · grid_width + gx`:
//!
//! - **Into the texels**, for `i < texels`, the texel `(x, y)` of slice `s` (`i` counted row
//!   after row, slice after slice): its first word the color word at
//!   `color_first + s · color_slice + y · color_row + x` of the planes, its second the alpha byte
//!   at byte `4 (alpha_first + s · alpha_slice + y · alpha_row) + x`, the rest of the word 0.
//! - **Back**, for `i < texels`, that color word from the texel's first word; and for
//!   `texels ≤ i < texels + alpha_words`, word `k` of row `y` of plane 1's slice `s`
//!   (`i - texels` counted row after row, slice after slice): its bytes the low bytes of the
//!   second words of texels `4 k` to `4 k + 3` of the row, 0 for a byte past the row's texels,
//!   which no consumer copies on.
//!
//! One past both moves nothing. The consumer keeps every word the shape names inside the
//! bindings, each binding within `maxStorageBufferRange`, and the grid within
//! `maxComputeWorkGroupCount`.
//!
//! **Synchronization** (consumer): the writes of both buffers and the shape (`TRANSFER_WRITE`)
//! → `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its writes →
//! `TRANSFER_READ` before the texels or the planes are copied on.

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

/// The crate-owned two-plane texel module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TwoPlaneTexelsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

/// Where a plane of a two-plane texture lies in the planes' binding, in words
/// ([`TwoPlaneTexelsModule::shape`]).
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneLayout {
    /// Its first word.
    pub first: u32,
    /// Its rows apart.
    pub row: u32,
    /// Its slices apart.
    pub slice: u32,
}

impl PlaneLayout {
    /// A plane from word `first`, rows `row` and slices `slice` words apart.
    pub const fn new(first: u32, row: u32, slice: u32) -> PlaneLayout {
        PlaneLayout { first, row, slice }
    }
}

impl TwoPlaneTexelsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The image's texels, two words each: a storage buffer.
    pub const TEXELS_BINDING: u32 = 0;
    /// The planes' bytes: a storage buffer.
    pub const PLANES_BINDING: u32 = 1;
    /// The move's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_WORDS: u32 = 13;
    /// The move's shape's binding.
    pub const SHAPE_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x.
    pub const LOCAL_SIZE: u32 = 64;

    /// The words of plane 1's rows of `slices` slices of `[width, height]` texels: `⌈width / 4⌉`
    /// a row.
    pub const fn alpha_words([width, height]: [u32; 2], slices: u32) -> u32 {
        width.div_ceil(4) * height * slices
    }

    /// The shape's bytes of a move of `slices` slices of `[width, height]` texels between the
    /// texels and the planes `color` (plane 0) and `alpha` (plane 1), into the texels where
    /// `into_texels`, else back, in a grid whose rows are `grid_width` invocations wide.
    pub fn shape(
        [width, height]: [u32; 2],
        slices: u32,
        color: PlaneLayout,
        alpha: PlaneLayout,
        into_texels: bool,
        grid_width: u32,
    ) -> [u8; 52] {
        let words = [
            width,
            height,
            width * height * slices,
            Self::alpha_words([width, height], slices),
            color.first,
            color.row,
            color.slice,
            alpha.first,
            alpha.row,
            alpha.slice,
            width.div_ceil(4),
            u32::from(into_texels),
            grid_width,
        ];
        let mut bytes = [0; 52];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the texels in the typed module.
const TEXELS: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the planes in the typed module.
const PLANES: ResourceSlot = ResourceSlot {
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

/// The shape's fields, loaded once.
#[derive(Clone, Copy)]
struct Shape {
    width: Value,
    height: Value,
    texels: Value,
    alpha_words: Value,
    color: [Value; 3],
    alpha: [Value; 3],
    row_alpha_words: Value,
    into_texels: Value,
}

/// The word of row `y` of slice `s` of the plane `[first, row, slice]`, plus `column`.
fn plane_word(
    f: &mut FunctionBuilder<'_>,
    [first, row, slice]: [Value; 3],
    s: Value,
    y: Value,
    column: Value,
) -> BResult<Value> {
    let at_slice = f.binary(BinOp::IMul, s, slice)?;
    let at_row = f.binary(BinOp::IMul, y, row)?;
    let at = f.binary(BinOp::IAdd, first, at_slice)?;
    let at = f.binary(BinOp::IAdd, at, at_row)?;
    f.binary(BinOp::IAdd, at, column)
}

/// The slice and the row in it of row `r` of every slice's rows.
fn slice_row(f: &mut FunctionBuilder<'_>, s: Shape, r: Value) -> BResult<(Value, Value)> {
    let slice = f.binary(BinOp::UDiv, r, s.height)?;
    let y = f.binary(BinOp::UMod, r, s.height)?;
    Ok((slice, y))
}

/// The module as a typed module, before serialization: the module [`two_plane_texels_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `TEXELS`, `PLANES`
/// and `SHAPE` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let texels = words_buffer(&mut b, TEXELS, u32_)?;
    let planes = words_buffer(&mut b, PLANES, u32_)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let gx = f.composite_extract(invocation, &[0])?;
    let gy = f.composite_extract(invocation, &[1])?;
    let texels = f.module().global_value(texels)?;
    let planes = f.module().global_value(planes)?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = u32_value(f, index)?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let s = Shape {
        width: field(&mut f, 0)?,
        height: field(&mut f, 1)?,
        texels: field(&mut f, 2)?,
        alpha_words: field(&mut f, 3)?,
        color: [field(&mut f, 4)?, field(&mut f, 5)?, field(&mut f, 6)?],
        alpha: [field(&mut f, 7)?, field(&mut f, 8)?, field(&mut f, 9)?],
        row_alpha_words: field(&mut f, 10)?,
        into_texels: field(&mut f, 11)?,
    };
    let grid_width = field(&mut f, 12)?;
    let grid_row = f.binary(BinOp::IMul, gy, grid_width)?;
    let i = f.binary(BinOp::IAdd, grid_row, gx)?;
    let is_texel = f.binary(BinOp::ULessThan, i, s.texels)?;
    let into = f.binary(BinOp::INotEqual, s.into_texels, zero)?;
    // Invocation `i < texels`: its texel's color word, one way or the other, and into the
    // texels its alpha byte too.
    let texel: ArmFn<'_, ()> = Box::new(move |f, _| {
        let one = u32_value(f, 1)?;
        let three = u32_value(f, 3)?;
        let two = u32_value(f, 2)?;
        let x = f.binary(BinOp::UMod, i, s.width)?;
        let r = f.binary(BinOp::UDiv, i, s.width)?;
        let (slice, y) = slice_row(f, s, r)?;
        let color = plane_word(f, s.color, slice, y, x)?;
        let at_color = f.access_chain(planes, &[zero, color])?;
        let first = f.binary(BinOp::ShiftLeftLogical, i, one)?;
        let second = f.binary(BinOp::IAdd, first, one)?;
        let at_first = f.access_chain(texels, &[zero, first])?;
        let inward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let mask = u32_value(f, 0xff)?;
            let v = f.load(at_color, MemAccess::default())?;
            f.store(at_first, v, MemAccess::default())?;
            let row = plane_word(f, s.alpha, slice, y, zero)?;
            let row_byte = f.binary(BinOp::ShiftLeftLogical, row, two)?;
            let byte = f.binary(BinOp::IAdd, row_byte, x)?;
            let word = f.binary(BinOp::ShiftRightLogical, byte, two)?;
            let place = f.binary(BinOp::BitwiseAnd, byte, three)?;
            let shift = f.binary(BinOp::ShiftLeftLogical, place, three)?;
            let p = f.access_chain(planes, &[zero, word])?;
            let v = f.load(p, MemAccess::default())?;
            let v = f.binary(BinOp::ShiftRightLogical, v, shift)?;
            let v = f.binary(BinOp::BitwiseAnd, v, mask)?;
            let at_second = f.access_chain(texels, &[zero, second])?;
            f.store(at_second, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let outward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let v = f.load(at_first, MemAccess::default())?;
            f.store(at_color, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), into, inward, Some(outward))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    // Back, invocation `texels ≤ i < texels + alpha_words`: one word of plane 1's rows.
    let alpha_word: ArmFn<'_, ()> = Box::new(move |f, _| {
        let back = f.binary(BinOp::IEqual, s.into_texels, zero)?;
        let j = f.binary(BinOp::ISub, i, s.texels)?;
        let in_rows = f.binary(BinOp::ULessThan, j, s.alpha_words)?;
        let moved = f.binary(BinOp::LogicalAnd, back, in_rows)?;
        let word: ArmFn<'_, ()> = Box::new(move |f, _| {
            let one = u32_value(f, 1)?;
            let two = u32_value(f, 2)?;
            let mask = u32_value(f, 0xff)?;
            let r = f.binary(BinOp::UDiv, j, s.row_alpha_words)?;
            let k = f.binary(BinOp::UMod, j, s.row_alpha_words)?;
            let (slice, y) = slice_row(f, s, r)?;
            let row_texel = f.binary(BinOp::IMul, r, s.width)?;
            let first_x = f.binary(BinOp::ShiftLeftLogical, k, two)?;
            let mut assembled = zero;
            for b in 0..4u32 {
                let offset = u32_value(f, b)?;
                let x = f.binary(BinOp::IAdd, first_x, offset)?;
                let inside = f.binary(BinOp::ULessThan, x, s.width)?;
                let x = f.select(inside, x, zero)?;
                let texel = f.binary(BinOp::IAdd, row_texel, x)?;
                let second = f.binary(BinOp::ShiftLeftLogical, texel, one)?;
                let second = f.binary(BinOp::IAdd, second, one)?;
                let p = f.access_chain(texels, &[zero, second])?;
                let v = f.load(p, MemAccess::default())?;
                let v = f.binary(BinOp::BitwiseAnd, v, mask)?;
                let v = f.select(inside, v, zero)?;
                let place = u32_value(f, 8 * b)?;
                let v = f.binary(BinOp::ShiftLeftLogical, v, place)?;
                assembled = f.binary(BinOp::BitwiseOr, assembled, v)?;
            }
            let at = plane_word(f, s.alpha, slice, y, k)?;
            let p = f.access_chain(planes, &[zero, at])?;
            f.store(p, assembled, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), moved, word, None)?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), is_texel, texel, Some(alpha_word))?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: TwoPlaneTexelsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(TwoPlaneTexelsModule::SET),
        Symbol::Binding(s) if *s == TEXELS => Some(TwoPlaneTexelsModule::TEXELS_BINDING),
        Symbol::Binding(s) if *s == PLANES => Some(TwoPlaneTexelsModule::PLANES_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(TwoPlaneTexelsModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The two-plane texel module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`TwoPlaneTexelsModule`].
pub fn two_plane_texels_module(device: &Device) -> Result<TwoPlaneTexelsModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the two-plane texel module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(TwoPlaneTexelsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, PlaneLayout, TwoPlaneTexelsModule, PLANES, SHAPE, TEXELS};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_two_plane_texel_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    fn words_of(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| u32::from_le_bytes(*c))
            .collect()
    }

    fn bytes_of(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    /// One dispatch of `groups` workgroups, both invocation orders alike: the texels and the
    /// planes after it.
    fn moved(
        texels: &[u32],
        planes: &[u8],
        shape: [u8; 52],
        groups: [u32; 2],
    ) -> (Vec<u32>, Vec<u8>) {
        let m = typed_module().expect("module");
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let part = |slot, bytes: Vec<u8>| Buffer {
                slot: Some(slot),
                address: None,
                bytes,
            };
            let res = Resources {
                buffers: vec![
                    part(TEXELS, bytes_of(texels)),
                    part(PLANES, planes.to_vec()),
                    part(SHAPE, shape.to_vec()),
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups[0], groups[1], 1],
                ..Config::default()
            };
            let buffers = run(&m, &cfg, res).expect("run").buffers;
            assert_eq!(buffers[2], shape.to_vec(), "the shape is only read");
            out.push((words_of(&buffers[0]), buffers[1].clone()));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Two slices of 7 x 3 texels, plane 0 from word 5 with rows 9 words and slices 32 apart,
    /// plane 1 from word 70 with rows 4 words and slices 16 apart, in a grid of two rows of
    /// workgroups: into the texels, each texel's first word is its color word and its second its
    /// alpha byte; back, each color word is the texel's first word, each alpha byte its second
    /// word's low byte, a row's bytes past its texels 0, and no other word of the planes changes.
    #[test]
    fn texels_move_between_the_image_and_metals_two_planes() {
        let ([w, h], slices) = ([7u32, 3u32], 2u32);
        let color = PlaneLayout::new(5, 9, 32);
        let alpha = PlaneLayout::new(70, 4, 16);
        let planes_len = 4 * (alpha.first + alpha.slice * slices) as usize;
        let planes: Vec<u8> = (0..planes_len).map(|k| (k * 7 + 3) as u8).collect();
        let n = w * h * slices;
        let texels: Vec<u32> = (0..2 * n).map(|k| 0x5100_0000 + k * 0x0101).collect();
        let place =
            |plane: PlaneLayout, s: u32, y: u32| plane.first + s * plane.slice + y * plane.row;
        let grid = TwoPlaneTexelsModule::LOCAL_SIZE;
        let alpha_words = TwoPlaneTexelsModule::alpha_words([w, h], slices);
        let rows = (n + alpha_words).div_ceil(grid);
        // Into the texels.
        let shape = TwoPlaneTexelsModule::shape([w, h], slices, color, alpha, true, grid);
        let (t, p) = moved(&texels, &planes, shape, [1, rows]);
        assert_eq!(p, planes, "the planes are only read");
        let plane_layout = super::tests::words_of(&planes);
        for s in 0..slices {
            for y in 0..h {
                for x in 0..w {
                    let i = ((s * h + y) * w + x) as usize;
                    let word = plane_layout[(place(color, s, y) + x) as usize];
                    let byte = planes[(4 * place(alpha, s, y) + x) as usize];
                    assert_eq!(t[2 * i], word, "color of ({x}, {y}) of slice {s}");
                    assert_eq!(t[2 * i + 1], u32::from(byte), "alpha of ({x}, {y}) of {s}");
                }
            }
        }
        // And back.
        let shape = TwoPlaneTexelsModule::shape([w, h], slices, color, alpha, false, grid);
        let (t, p) = moved(&texels, &planes, shape, [1, rows]);
        assert_eq!(t, texels, "the texels are only read");
        let mut want = planes.clone();
        for s in 0..slices {
            for y in 0..h {
                for x in 0..w {
                    let i = ((s * h + y) * w + x) as usize;
                    let at = 4 * (place(color, s, y) + x) as usize;
                    want[at..at + 4].copy_from_slice(&texels[2 * i].to_le_bytes());
                    want[(4 * place(alpha, s, y) + x) as usize] = texels[2 * i + 1] as u8;
                }
                // The last word of the row holds 0 past its texels.
                for x in w..w.next_multiple_of(4) {
                    want[(4 * place(alpha, s, y) + x) as usize] = 0;
                }
            }
        }
        assert_eq!(p, want);
    }
}
