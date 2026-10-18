//! The color words of a two-plane extended-range texture moved between its image and a buffer
//! laid out as Metal lays them.
//!
//! An `RGB10A8_2P_XR10` or `RGB10A8_2P_XR10_sRGB` texture holds its two planes side by side in
//! one `R32G32_UINT` image (`metal::format::ExtendedRange::Planes`): each texel's color word
//! (plane 0), then its alpha byte in the second word. Metal's blits between a buffer and such a
//! texture move plane 0 alone, 4 bytes a texel at the buffer's rows, and leave the alpha plane as
//! it is. A consumer copies a band of the region's rows out of the image
//! into a scratch buffer, tight (8 bytes a texel), and one dispatch of [`plane_words_module`]
//! moves each texel's first word into the buffer's bytes or out of them; a copy into the texture
//! then copies the band back, its second words as they were.
//!
//! **The module** (one per device). One `GLCompute` entry `main` with local size
//! ([`PlaneWordsModule::LOCAL_SIZE`], 1, 1), no push constants, and three storage buffers of
//! 4-byte words in set 0:
//!
//! | Binding | Buffer |
//! |---|---|
//! | [`PlaneWordsModule::TEXELS_BINDING`] | the band's texels, two words each, row after row |
//! | [`PlaneWordsModule::WORDS_BINDING`] | the buffer's words the band reaches, from one at or below its first |
//! | [`PlaneWordsModule::SHAPE_BINDING`] | the band's shape, [`PlaneWordsModule::SHAPE_WORDS`] words |
//!
//! The shape's words, in order: `width`, the texels of one row of the band (at least 1);
//! `texels`, the band's (`width` times its rows); `row_words`, the buffer's rows apart in words;
//! `first_word`, the word of the band's first texel in the words binding; and `into_texels`,
//! nonzero to move the buffer's words into the texels, else out of them. Invocation `i` below
//! `texels` takes the texel at `(i % width, i / width)` of the band: word `2 i` of the texels
//! binding and word `first_word + (i / width) · row_words + i % width` of the words binding, one
//! into the other. One past `texels` moves nothing. The consumer keeps each binding within
//! `maxStorageBufferRange` and each band within `maxComputeWorkGroupCount[0]` workgroups, and
//! rows of the buffer that do not overlap.
//!
//! **Synchronization** (consumer): the copy out of the image and the shape's update
//! (`TRANSFER_WRITE`) → `SHADER_STORAGE_READ` in `COMPUTE_SHADER` before the dispatch; its writes
//! → `TRANSFER_READ` before the texels are copied back into the image, or the buffer's next use.

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

/// The crate-owned plane words module of one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlaneWordsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl PlaneWordsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The band's texels, two words each: a storage buffer.
    pub const TEXELS_BINDING: u32 = 0;
    /// The buffer's words the band reaches: a storage buffer.
    pub const WORDS_BINDING: u32 = 1;
    /// The band's shape: a storage buffer of [`SHAPE_WORDS`](Self::SHAPE_WORDS) words.
    pub const SHAPE_BINDING: u32 = 2;
    /// The words of a shape: `width`, `texels`, `row_words`, `first_word`, `into_texels`.
    pub const SHAPE_WORDS: u32 = 5;
    /// Invocations of one workgroup, along x: one texel each.
    pub const LOCAL_SIZE: u32 = 64;

    /// The shape's bytes of a band of `width` texels a row and `texels` in all, the buffer's
    /// rows `row_words` apart, its first texel at word `first_word` of the words binding, moved
    /// into the texels where `into_texels`, else out of them.
    pub fn shape(
        width: u32,
        texels: u32,
        row_words: u32,
        first_word: u32,
        into_texels: bool,
    ) -> [u8; 20] {
        let words = [width, texels, row_words, first_word, u32::from(into_texels)];
        let mut bytes = [0; 20];
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
/// The binding symbol of the buffer's words in the typed module.
const WORDS: ResourceSlot = ResourceSlot {
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

/// The module as a typed module, before serialization: the module [`plane_words_module`]
/// serializes, which the CPU interpreter runs. Its bindings are the symbols `TEXELS`, `WORDS`
/// and `SHAPE` (the module docs).
fn typed_module() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let texels = words_buffer(&mut b, TEXELS, u32_)?;
    let words = words_buffer(&mut b, WORDS, u32_)?;
    let shape = words_buffer(&mut b, SHAPE, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let one = u32_value(&mut f, 1)?;
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let i = f.composite_extract(invocation, &[0])?;
    let texels = f.module().global_value(texels)?;
    let words = f.module().global_value(words)?;
    let shape = f.module().global_value(shape)?;
    let field = |f: &mut FunctionBuilder<'_>, index: u32| -> BResult<Value> {
        let index = u32_value(f, index)?;
        let p = f.access_chain(shape, &[zero, index])?;
        f.load(p, MemAccess::default())
    };
    let width = field(&mut f, 0)?;
    let count = field(&mut f, 1)?;
    let row_words = field(&mut f, 2)?;
    let first_word = field(&mut f, 3)?;
    let into_texels = field(&mut f, 4)?;
    let inside = f.binary(BinOp::ULessThan, i, count)?;
    let moved: ArmFn<'_, ()> = Box::new(move |f, _| {
        let y = f.binary(BinOp::UDiv, i, width)?;
        let x = f.binary(BinOp::UMod, i, width)?;
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let word = f.binary(BinOp::IAdd, first_word, row)?;
        let word = f.binary(BinOp::IAdd, word, x)?;
        let texel = f.binary(BinOp::ShiftLeftLogical, i, one)?;
        let at_word = f.access_chain(words, &[zero, word])?;
        let at_texel = f.access_chain(texels, &[zero, texel])?;
        let into = f.binary(BinOp::INotEqual, into_texels, zero)?;
        let inward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let v = f.load(at_word, MemAccess::default())?;
            f.store(at_texel, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        let outward: ArmFn<'_, ()> = Box::new(move |f, _| {
            let v = f.load(at_texel, MemAccess::default())?;
            f.store(at_word, v, MemAccess::default())?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), into, inward, Some(outward))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, moved, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: PlaneWordsModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(PlaneWordsModule::SET),
        Symbol::Binding(s) if *s == TEXELS => Some(PlaneWordsModule::TEXELS_BINDING),
        Symbol::Binding(s) if *s == WORDS => Some(PlaneWordsModule::WORDS_BINDING),
        Symbol::Binding(s) if *s == SHAPE => Some(PlaneWordsModule::SHAPE_BINDING),
        _ => None,
    }
}

/// The plane words module of `device`: built with the typed builder, checked by the owned
/// verifier against the device's profile (requirements included), serialized, and numbered with
/// the fixed bindings of [`PlaneWordsModule`].
pub fn plane_words_module(device: &Device) -> Result<PlaneWordsModule, Error> {
    let profile = device.profile();
    let m = typed_module()?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the plane words module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(PlaneWordsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, PlaneWordsModule, SHAPE, TEXELS, WORDS};
    use crate::spirv::interp::{run, Buffer, Config, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_plane_words_kernel_is_valid_on_every_committed_profile() {
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

    /// One band through the kernel in `groups` workgroups, both invocation orders alike: the
    /// texels and the buffer's words after it.
    fn moved(texels: &[u32], words: &[u32], shape: [u8; 20], groups: u32) -> (Vec<u32>, Vec<u32>) {
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
                    part(WORDS, bytes_of(words)),
                    part(SHAPE, shape.to_vec()),
                ],
                push: Vec::new(),
                spec: Default::default(),
            };
            let cfg = Config {
                order,
                workgroups: [groups, 1, 1],
                ..Config::default()
            };
            let buffers = run(&m, &cfg, res).expect("run").buffers;
            assert_eq!(buffers[2], shape.to_vec(), "the shape is only read");
            out.push((words_of(&buffers[0]), words_of(&buffers[1])));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// A band of 70 texels, 7 a row, from the third word of the binding with rows 9 words
    /// apart: out of the texels, each texel's first word lands at its row and column, and no
    /// other word of the buffer changes; into them, each texel's first word is the buffer's and
    /// its second word stays.
    #[test]
    fn color_words_move_at_metals_rows_and_the_alpha_words_stay() {
        let (width, rows, row_words, first) = (7u32, 10u32, 9u32, 3u32);
        let n = width * rows;
        let texels: Vec<u32> = (0..2 * n).map(|k| 0x1000_0000 + k).collect();
        let words: Vec<u32> = (0..first + rows * row_words)
            .map(|k| 0xabab_0000 + k)
            .collect();
        // Out of the texels, in two workgroups (64 invocations each, 58 past the texels).
        let shape = PlaneWordsModule::shape(width, n, row_words, first, false);
        let (t, w) = moved(&texels, &words, shape, 2);
        assert_eq!(t, texels, "the texels are only read");
        let mut want = words.clone();
        for i in 0..n {
            let at = first + (i / width) * row_words + i % width;
            want[at as usize] = texels[2 * i as usize];
        }
        assert_eq!(w, want);
        // Into the texels.
        let shape = PlaneWordsModule::shape(width, n, row_words, first, true);
        let (t, w) = moved(&texels, &words, shape, 2);
        assert_eq!(w, words, "the buffer's words are only read");
        let mut want = texels.clone();
        for i in 0..n {
            let at = first + (i / width) * row_words + i % width;
            want[2 * i as usize] = words[at as usize];
        }
        assert_eq!(t, want);
    }
}
