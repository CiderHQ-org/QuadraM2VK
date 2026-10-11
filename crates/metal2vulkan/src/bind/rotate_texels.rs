//! A rotated texture's texels moved between the frame its render passes draw in and the frame
//! its bytes hold them in.
//!
//! Metal stores a framebuffer-only texture of rotation 1, 2 or 3 (and 4, which stores as 1) turned:
//! a pass draws a `W x H` frame whose texel `(x, y)` lands at `(H - 1 - y, x)` of an `H x W`
//! store for rotation 1, at `(W - 1 - x, H - 1 - y)` of a `W x H` one for rotation 2, and at
//! `(y, W - 1 - x)` of an `H x W` one for rotation 3. A consumer renders into an image of the
//! pass's frame and moves its texels to and from the store through two buffers: the image
//! copied tight into one, and the store's bytes laid out as in the texture's buffer in the other.
//! One dispatch of [`rotate_texels_module`] fills one buffer from the other, each destination
//! texel from the source texel an affine map of its position names.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`RotateTexelsModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`RotateTexelsModule::SOURCE_BINDING`] | the source texels, rows `source_pitch` bytes apart from its start |
//! | [`RotateTexelsModule::DESTINATION_BINDING`] | the destination texels, rows `pitch` bytes apart from its start |
//! | [`RotateTexelsModule::SHAPE_BINDING`] | the move's shape, [`RotateTexelsModule::SHAPE_WORDS`] words |
//!
//! The shape's words, in order ([`RotateTexelsModule::shape`]): `texel_bytes` (1, 2, 4, 8 or
//! 16); the destination's `pitch` in bytes (a multiple of 4), `width` in texels and `height` in
//! rows; the source's `source_pitch` in bytes; the map `m00, m01, m02, m10, m11, m12` (two's
//! complement); `words`, the destination words written; and `grid_width`, the invocations of one
//! row of the dispatch's grid (its x groups times the local size). Invocation `(gx, gy)` takes
//! destination word `i = gy · grid_width + gx`, and none at `words` or past it. Each byte `k` of
//! the word (`4 i` to `4 i + 3`) lies in row `y = k / pitch`, at byte `c = k % pitch` of it, so
//! in texel `x = c / texel_bytes`; inside the destination (`x < width`, `y < height`), it is byte
//! `c % texel_bytes` of the source texel `(m00 x + m01 y + m02, m10 x + m11 y + m12)`, which lies
//! at byte `sy · source_pitch + sx · texel_bytes` of the source. Outside, it is 0: the bytes past
//! a row's texels, which no consumer copies on. Where `texel_bytes` is a multiple of 4, a word is
//! a whole word of one texel, moved as one.
//!
//! The consumer keeps every source texel the map names inside the source binding, each binding
//! within `maxStorageBufferRange`, and the grid within `maxComputeWorkGroupCount`.
//!
//! **Synchronization** (consumer): the writes of both buffers and the shape (`TRANSFER_WRITE`)
//! → `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its writes →
//! `TRANSFER_READ` before the destination is copied on.

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

/// The crate-owned texel rotation module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RotateTexelsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl RotateTexelsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The source texels: a storage buffer.
    pub const SOURCE_BINDING: u32 = 0;
    /// The destination texels: a storage buffer.
    pub const DESTINATION_BINDING: u32 = 1;
    /// The move's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 2;
    /// The words of a shape (the module docs).
    pub const SHAPE_WORDS: u32 = 13;
    /// Invocations of one workgroup, along x: one destination word each.
    pub const LOCAL_SIZE: u32 = 64;

    /// The shape's bytes of a move of `texel_bytes` texels into a destination of `width` by
    /// `height` texels whose rows are `pitch` bytes apart, from a source whose rows are
    /// `source_pitch` bytes apart, destination texel `(x, y)` from source texel
    /// `(m[0] x + m[1] y + m[2], m[3] x + m[4] y + m[5])`, in a grid whose rows are `grid_width`
    /// invocations wide; `words` is the destination's words, its last row's last texel included.
    pub fn shape(
        texel_bytes: u32,
        [pitch, width, height]: [u32; 3],
        source_pitch: u32,
        m: [i32; 6],
        words: u32,
        grid_width: u32,
    ) -> [u8; 52] {
        let map = m.map(|v| v as u32);
        let words = [
            texel_bytes,
            pitch,
            width,
            height,
            source_pitch,
            map[0],
            map[1],
            map[2],
            map[3],
            map[4],
            map[5],
            words,
            grid_width,
        ];
        let mut bytes = [0; 52];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }
}

/// The binding symbol of the source in the typed module.
const SOURCE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the destination in the typed module.
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

/// The shape's fields, loaded once.
#[derive(Clone, Copy)]
struct Shape {
    texel_bytes: Value,
    pitch: Value,
    width: Value,
    height: Value,
    source_pitch: Value,
    m: [Value; 6],
}

/// Where destination byte `k` reads: whether it lies in a texel, the source word holding its
/// byte (word 0 for one outside, which every source holds), and the byte's place in that word
/// in bits.
fn source_of(f: &mut FunctionBuilder<'_>, s: Shape, k: Value) -> BResult<(Value, Value, Value)> {
    let zero = u32_value(f, 0)?;
    let two = u32_value(f, 2)?;
    let three = u32_value(f, 3)?;
    let row = f.binary(BinOp::UDiv, k, s.pitch)?;
    let column = f.binary(BinOp::UMod, k, s.pitch)?;
    let x = f.binary(BinOp::UDiv, column, s.texel_bytes)?;
    let in_texel = f.binary(BinOp::UMod, column, s.texel_bytes)?;
    let in_row = f.binary(BinOp::ULessThan, x, s.width)?;
    let in_rows = f.binary(BinOp::ULessThan, row, s.height)?;
    let inside = f.binary(BinOp::LogicalAnd, in_row, in_rows)?;
    let affine = |f: &mut FunctionBuilder<'_>, a: Value, b: Value, c: Value| -> BResult<Value> {
        let ax = f.binary(BinOp::IMul, a, x)?;
        let by = f.binary(BinOp::IMul, b, row)?;
        let sum = f.binary(BinOp::IAdd, ax, by)?;
        f.binary(BinOp::IAdd, sum, c)
    };
    let sx = affine(f, s.m[0], s.m[1], s.m[2])?;
    let sy = affine(f, s.m[3], s.m[4], s.m[5])?;
    let at_row = f.binary(BinOp::IMul, sy, s.source_pitch)?;
    let at_texel = f.binary(BinOp::IMul, sx, s.texel_bytes)?;
    let at = f.binary(BinOp::IAdd, at_row, at_texel)?;
    let at = f.binary(BinOp::IAdd, at, in_texel)?;
    let word = f.binary(BinOp::ShiftRightLogical, at, two)?;
    let word = f.select(inside, word, zero)?;
    let byte = f.binary(BinOp::BitwiseAnd, at, three)?;
    let shift = f.binary(BinOp::ShiftLeftLogical, byte, three)?;
    Ok((inside, word, shift))
}

/// The module as a typed module, before serialization: the module [`rotate_texels_module`]
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
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let gx = f.composite_extract(invocation, &[0])?;
    let gy = f.composite_extract(invocation, &[1])?;
    let source = f.module().global_value(source)?;
    let destination = f.module().global_value(destination)?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = u32_value(f, index)?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let s = Shape {
        texel_bytes: field(&mut f, 0)?,
        pitch: field(&mut f, 1)?,
        width: field(&mut f, 2)?,
        height: field(&mut f, 3)?,
        source_pitch: field(&mut f, 4)?,
        m: [
            field(&mut f, 5)?,
            field(&mut f, 6)?,
            field(&mut f, 7)?,
            field(&mut f, 8)?,
            field(&mut f, 9)?,
            field(&mut f, 10)?,
        ],
    };
    let words = field(&mut f, 11)?;
    let grid_width = field(&mut f, 12)?;
    let row = f.binary(BinOp::IMul, gy, grid_width)?;
    let i = f.binary(BinOp::IAdd, row, gx)?;
    let inside = f.binary(BinOp::ULessThan, i, words)?;
    let moved: ArmFn<'_, ()> = Box::new(move |f, _| {
        let two = u32_value(f, 2)?;
        let three = u32_value(f, 3)?;
        let first = f.binary(BinOp::ShiftLeftLogical, i, two)?;
        let at_word = f.access_chain(destination, &[zero, i])?;
        let partial = f.binary(BinOp::BitwiseAnd, s.texel_bytes, three)?;
        let whole = f.binary(BinOp::IEqual, partial, zero)?;
        // A word of one texel, moved as one.
        let as_word: ArmFn<'_, ()> = Box::new(move |f, _| {
            let (inside, word, _) = source_of(f, s, first)?;
            let p = f.access_chain(source, &[zero, word])?;
            let v = f.load(p, MemAccess::default())?;
            let v = f.select(inside, v, zero)?;
            f.store(at_word, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        // Each byte from its texel's.
        let by_bytes: ArmFn<'_, ()> = Box::new(move |f, _| {
            let mask = u32_value(f, 0xff)?;
            let mut assembled = zero;
            for j in 0..4u32 {
                let offset = u32_value(f, j)?;
                let k = f.binary(BinOp::IAdd, first, offset)?;
                let (inside, word, shift) = source_of(f, s, k)?;
                let p = f.access_chain(source, &[zero, word])?;
                let v = f.load(p, MemAccess::default())?;
                let v = f.binary(BinOp::ShiftRightLogical, v, shift)?;
                let v = f.binary(BinOp::BitwiseAnd, v, mask)?;
                let v = f.select(inside, v, zero)?;
                let place = u32_value(f, 8 * j)?;
                let v = f.binary(BinOp::ShiftLeftLogical, v, place)?;
                assembled = f.binary(BinOp::BitwiseOr, assembled, v)?;
            }
            f.store(at_word, assembled, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), whole, as_word, Some(by_bytes))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, moved, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: RotateTexelsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(RotateTexelsModule::SET),
        Symbol::Binding(s) if *s == SOURCE => Some(RotateTexelsModule::SOURCE_BINDING),
        Symbol::Binding(s) if *s == DESTINATION => Some(RotateTexelsModule::DESTINATION_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(RotateTexelsModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The texel rotation module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`RotateTexelsModule`].
pub fn rotate_texels_module(device: &Device) -> Result<RotateTexelsModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the texel rotation module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(RotateTexelsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, RotateTexelsModule, DESTINATION, SHAPE, SOURCE};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_texel_rotation_kernel_is_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            assert_valid(typed_module().expect("module"), &p);
        }
    }

    /// The destination after one dispatch of `groups` workgroups over `source`, both invocation
    /// orders alike.
    fn moved(source: &[u8], destination_len: usize, shape: [u8; 52], groups: [u32; 2]) -> Vec<u8> {
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
                    part(SOURCE, source.to_vec()),
                    part(DESTINATION, vec![0xee; destination_len]),
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
            assert_eq!(buffers[0], source, "the source is only read");
            assert_eq!(buffers[2], shape.to_vec(), "the shape is only read");
            out.push(buffers[1].clone());
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Where Metal stores the pass's texel `(x, y)` of a `w x h` pass at rotation `r`.
    fn stored(r: u32, (w, h): (u32, u32), (x, y): (u32, u32)) -> (u32, u32) {
        match r {
            1 => (h - 1 - y, x),
            2 => (w - 1 - x, h - 1 - y),
            3 => (y, w - 1 - x),
            _ => (x, y),
        }
    }

    /// The map of a move into the pass's frame from the store (`into`) or back, as the runtime
    /// gives it: destination texel to source texel.
    fn map(r: u32, (w, h): (u32, u32), into: bool) -> [i32; 6] {
        let (w, h) = (w as i32, h as i32);
        match (r, into) {
            (1, true) => [0, -1, h - 1, 1, 0, 0],
            (1, false) => [0, 1, 0, -1, 0, h - 1],
            (2, _) => [-1, 0, w - 1, 0, -1, h - 1],
            (3, true) => [0, 1, 0, -1, 0, w - 1],
            (3, false) => [0, -1, w - 1, 1, 0, 0],
            _ => [1, 0, 0, 0, 1, 0],
        }
    }

    /// Every rotation and texel size, both ways, through odd extents and a store whose rows hold
    /// bytes past their texels: into the pass's frame (tight rows), each texel is the store's
    /// where Metal puts it, and back, each stored texel is the pass's, the bytes past a row's
    /// texels 0 and every texel byte exact.
    #[test]
    fn texels_move_between_the_pass_frame_and_the_store_as_metal_turns_them() {
        let (w, h) = (7u32, 5u32);
        for r in 1..=3u32 {
            for texel_bytes in [1u32, 2, 4, 8, 16] {
                let (sw, sh) = if r == 2 { (w, h) } else { (h, w) };
                let store_pitch = (sw * texel_bytes).next_multiple_of(16) + 16;
                let pass_pitch = w * texel_bytes;
                // Each stored texel's bytes name its position and its byte.
                let mut store = vec![0u8; (store_pitch * sh) as usize];
                for sy in 0..sh {
                    for sx in 0..sw {
                        for b in 0..texel_bytes {
                            let at = (sy * store_pitch + sx * texel_bytes + b) as usize;
                            store[at] = (sx * 37 + sy * 11 + b * 3 + 1) as u8;
                        }
                    }
                }
                // Into the pass's frame.
                let pass_bytes = pass_pitch * h;
                let words = pass_bytes.div_ceil(4);
                let grid = RotateTexelsModule::LOCAL_SIZE;
                let shape = RotateTexelsModule::shape(
                    texel_bytes,
                    [pass_pitch, w, h],
                    store_pitch,
                    map(r, (w, h), true),
                    words,
                    grid,
                );
                let rows = words.div_ceil(grid);
                let pass = moved(&store, 4 * words as usize, shape, [1, rows]);
                for y in 0..h {
                    for x in 0..w {
                        let (sx, sy) = stored(r, (w, h), (x, y));
                        for b in 0..texel_bytes {
                            let got = pass[(y * pass_pitch + x * texel_bytes + b) as usize];
                            let want = store[(sy * store_pitch + sx * texel_bytes + b) as usize];
                            assert_eq!(
                                got, want,
                                "r {r}, {texel_bytes} bytes, ({x}, {y}) byte {b}"
                            );
                        }
                    }
                }
                // And back into a store, from the pass's frame.
                let store_words = (store_pitch * (sh - 1) + sw * texel_bytes).div_ceil(4);
                let shape = RotateTexelsModule::shape(
                    texel_bytes,
                    [store_pitch, sw, sh],
                    pass_pitch,
                    map(r, (w, h), false),
                    store_words,
                    2 * grid,
                );
                let back = moved(
                    &pass,
                    4 * store_words as usize,
                    shape,
                    [2, store_words.div_ceil(2 * grid)],
                );
                for (k, &byte) in back.iter().enumerate() {
                    let (row, column) = (k as u32 / store_pitch, k as u32 % store_pitch);
                    let want = match column < sw * texel_bytes {
                        true => store[(row * store_pitch + column) as usize],
                        false => 0,
                    };
                    assert_eq!(byte, want, "r {r}, {texel_bytes} bytes, back, byte {k}");
                }
            }
        }
    }
}
