//! A heap texture's elements moved between its image and Metal's tiled layout of them in the
//! heap's bytes by one dispatch a level, through an unsigned-integer storage view of the image.
//!
//! [`tiled_texels_module`](super::tiled_texels_module) moves a tiled heap texture's elements
//! between a tight buffer and a staging buffer of the heap's bytes, which copies move on to and
//! from the image and the heap. Where the image has an unsigned-integer view of its element's
//! width that stores, [`tiled_words_module`] moves them between the heap's bytes and the image
//! itself: each invocation reads or writes one element's words at its tiled place in the heap's
//! bytes and its texel through the view. Vulkan states that "Views of compatible formats will have
//! the same mapping between texel coordinates and memory locations irrespective of the format,
//! with only the interpretation of the bit pattern changing" (`VkImageViewCreateInfo`), and an
//! unsigned-integer view stores and returns its components' bits unconverted, so a texel stored
//! through the view holds the bytes a copy of those words into the image writes, and a texel read
//! through it returns the words a copy out of the image writes ([`TiledTexelView`]).
//!
//! **The module** (one per view). One `GLCompute` entry `main` with local size
//! ([`TiledWordsModule::LOCAL_SIZE`], [`TiledWordsModule::LOCAL_SIZE`], 1),
//! [`TiledWordsModule::PUSH_BYTES`] of `COMPUTE` push constants at offset 0 (the level's shape,
//! [`TiledWordsModule::shape`]), and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`TiledWordsModule::IMAGE_BINDING`] | the image's storage view of the module's format: one level, every layer (a `2D_ARRAY` view), in the `GENERAL` layout |
//! | [`TiledWordsModule::BYTES_BINDING`] | a storage buffer over the heap's bytes, from the texture's first byte less `skip` words |
//! | [`TiledWordsModule::KEPT_BINDING`] | a storage buffer of the kept words, laid out as the texture's bytes from its first |
//!
//! The shape names the step ([`TiledWordsStep`]) and the level ([`TiledWordsLevel`]). Invocation
//! `(x, y, z)` with `x` and `y` inside the level's extent takes element `(x, y)` of layer `z`:
//! `rel`, its first word from the texture's first (the level's first word, `z` chain strides, and
//! its place in its tile row and tile, as `tiled_texels` lays it out), at word `skip + rel` of the
//! bytes and word `rel` of the kept words; its `n` words (`n` the module's texel width):
//!
//! - **In** ([`TiledWordsStep::In`]): its words from the bytes, stored as its texel through the
//!   view; [`TiledWordsStep::InKeep`] also stores them in the kept words.
//! - **Keep** ([`TiledWordsStep::Keep`]): its texel read through the view, its words stored in the
//!   kept words.
//! - **Merge** ([`TiledWordsStep::Merge`]): its texel read through the view; for each of its words
//!   that differs from the kept word, the bytes' word takes the texel's byte in each byte where they
//!   differ and keeps its own in the others; a word equal to the kept one is neither read nor
//!   written.
//! - **Back** ([`TiledWordsStep::Back`]): its texel read through the view, its words stored in the
//!   bytes.
//!
//! One outside the extent does nothing. No byte of the bytes or the kept words but an element's is
//! read or written, and no two invocations touch one word: an element is a whole number of words
//! at a multiple of its size. A step that names no kept word binds any buffer there.
//!
//! **Synchronization** (consumer): every earlier access of the image, the bytes and the kept words
//! → the step's storage reads and writes of them in `COMPUTE_SHADER`; its writes → their next use.

use std::sync::Arc;

use super::shadow_merge::{merge_word, u32_value, words_buffer};
use super::tiled_texels::{low_mask, morton};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, Feature, InternalError, InternalKind, Unsupported};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{TypeId, Value};
use crate::spirv::ir::{
    BinOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, IoDir, MemAccess,
    StorageClass, TexelMemory, Type, TypedModule,
};

/// The unsigned-integer view of a tiled heap texture's image that [`tiled_words_module`] stores
/// and reads its texels through: of the element's width and of its format's channel layout, so
/// it shares the image format's compatibility class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum TiledTexelView {
    /// `R32_UINT`: a 4-byte element's word, in component 0.
    R32Uint,
    /// `R8G8B8A8_UINT`: a 4-byte element's bytes, byte `i` in component `i`.
    R8G8B8A8Uint,
    /// `R16G16B16A16_UINT`: an 8-byte element's 16-bit halves, half `i` in component `i`.
    R16G16B16A16Uint,
    /// `R32G32_UINT`: an 8-byte element's words.
    R32G32Uint,
    /// `R32G32B32A32_UINT`: a 16-byte element's words.
    R32G32B32A32Uint,
}

impl TiledTexelView {
    /// Every view.
    pub const ALL: [TiledTexelView; 5] = [
        TiledTexelView::R32Uint,
        TiledTexelView::R8G8B8A8Uint,
        TiledTexelView::R16G16B16A16Uint,
        TiledTexelView::R32G32Uint,
        TiledTexelView::R32G32B32A32Uint,
    ];

    /// The element's 4-byte words: 1, 2 or 4.
    pub fn words(self) -> u32 {
        match self {
            TiledTexelView::R32Uint | TiledTexelView::R8G8B8A8Uint => 1,
            TiledTexelView::R16G16B16A16Uint | TiledTexelView::R32G32Uint => 2,
            TiledTexelView::R32G32B32A32Uint => 4,
        }
    }

    /// The bits of each of its components: the element's words split into them, lowest first.
    fn component_bits(self) -> u32 {
        match self {
            TiledTexelView::R8G8B8A8Uint => 8,
            TiledTexelView::R16G16B16A16Uint => 16,
            TiledTexelView::R32Uint
            | TiledTexelView::R32G32Uint
            | TiledTexelView::R32G32B32A32Uint => 32,
        }
    }

    /// The view's storage image format, as the module declares it.
    fn storage_format(self) -> crate::spv::ImageFormat {
        match self {
            TiledTexelView::R32Uint => crate::spv::ImageFormat::R32ui,
            TiledTexelView::R8G8B8A8Uint => crate::spv::ImageFormat::Rgba8ui,
            TiledTexelView::R16G16B16A16Uint => crate::spv::ImageFormat::Rgba16ui,
            TiledTexelView::R32G32Uint => crate::spv::ImageFormat::Rg32ui,
            TiledTexelView::R32G32B32A32Uint => crate::spv::ImageFormat::Rgba32ui,
        }
    }
}

/// What one dispatch of [`tiled_words_module`] does (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TiledWordsStep {
    /// The bytes' elements into the image.
    In,
    /// The bytes' elements into the image and the kept words.
    InKeep,
    /// The image's texels into the kept words.
    Keep,
    /// The image's texels merged into the bytes where they differ from the kept words.
    Merge,
    /// The image's texels into the bytes.
    Back,
}

impl TiledWordsStep {
    /// Its number in the shape.
    fn code(self) -> u32 {
        match self {
            TiledWordsStep::In => 0,
            TiledWordsStep::InKeep => 1,
            TiledWordsStep::Keep => 2,
            TiledWordsStep::Merge => 3,
            TiledWordsStep::Back => 4,
        }
    }
}

/// One level of a tiled heap texture, as [`tiled_words_module`] moves it.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TiledWordsLevel {
    /// Its elements across and down.
    pub extent: [u32; 2],
    /// Its layers: the chains it lies in.
    pub images: u32,
    /// Its tile, elements across and down: powers of two.
    pub tile: [u32; 2],
    /// Its tiles a row.
    pub across: u32,
    /// Its first byte from the texture's first, in the first chain: a multiple of 4.
    pub first: u32,
    /// The bytes from one chain to the next: a multiple of 4.
    pub chain_stride: u32,
}

impl TiledWordsLevel {
    /// A level of `extent` elements in `images` layers, tiled in tiles of `tile` elements
    /// `across` a row, from byte `first` of the texture's first chain, chains `chain_stride`
    /// bytes apart.
    pub const fn new(
        extent: [u32; 2],
        images: u32,
        tile: [u32; 2],
        across: u32,
        first: u32,
        chain_stride: u32,
    ) -> TiledWordsLevel {
        TiledWordsLevel {
            extent,
            images,
            tile,
            across,
            first,
            chain_stride,
        }
    }
}

/// The crate-owned tiled word module of one view on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TiledWordsModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device but for the
    /// `R32G32_UINT` view, whose storage format is an extended one).
    pub requirements: Requirements,
}

impl TiledWordsModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The image's storage view of one level, every layer.
    pub const IMAGE_BINDING: u32 = 0;
    /// The heap's bytes, a storage buffer.
    pub const BYTES_BINDING: u32 = 1;
    /// The kept words, a storage buffer.
    pub const KEPT_BINDING: u32 = 2;
    /// Invocations of one workgroup along x, and along y: an 8 x 8 square of elements, which
    /// lie in 64 consecutive places of a tile of 8 x 8 elements or more.
    pub const LOCAL_SIZE: u32 = 8;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the level's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of `step` over `level` whose bytes' binding starts `skip` words before the
    /// texture's first byte (the module docs); `None` where a field does not fit its bits: a
    /// side, or the tiles a row, past 65535, a tile side past 128, `skip` past 63, or a byte that
    /// is no multiple of 4.
    pub fn shape(step: TiledWordsStep, level: &TiledWordsLevel, skip: u32) -> Option<[u8; 16]> {
        let [width, height] = level.extent;
        let [tw, th] = level.tile;
        let fits = width <= 0xffff
            && height <= 0xffff
            && level.across <= 0xffff
            && tw.is_power_of_two()
            && th.is_power_of_two()
            && tw <= 128
            && th <= 128
            && skip <= 63
            && level.first.is_multiple_of(4)
            && level.chain_stride.is_multiple_of(4);
        if !fits {
            return None;
        }
        let words = [
            width | (height << 16),
            step.code()
                | (tw.trailing_zeros() << 3)
                | (th.trailing_zeros() << 6)
                | (skip << 9)
                | (level.across << 16),
            level.first / 4,
            level.chain_stride / 4,
        ];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        Some(bytes)
    }

    /// The workgroups of a dispatch over `level`: its extent in squares of
    /// [`LOCAL_SIZE`](Self::LOCAL_SIZE), by its layers.
    pub fn groups(level: &TiledWordsLevel) -> [u32; 3] {
        let [width, height] = level.extent;
        [
            width.div_ceil(Self::LOCAL_SIZE),
            height.div_ceil(Self::LOCAL_SIZE),
            level.images,
        ]
    }
}

/// The binding symbol of the image's view in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the heap's bytes in the typed module.
const BYTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the kept words in the typed module.
const KEPT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// The words of an element whose texel a read through `view` returned (`texel`, a `uvec4`): its
/// components packed into words, lowest first (the view's components are unsigned integers of
/// their bits).
fn texel_words(
    f: &mut FunctionBuilder<'_>,
    texel: Value,
    view: TiledTexelView,
) -> BResult<Vec<Value>> {
    let bits = view.component_bits();
    let per_word = 32 / bits;
    let mut words = Vec::with_capacity(view.words() as usize);
    for word in 0..view.words() {
        let mut packed = f.composite_extract(texel, &[word * per_word])?;
        for k in 1..per_word {
            let part = f.composite_extract(texel, &[word * per_word + k])?;
            let shift = u32_value(f, k * bits)?;
            let part = f.binary(BinOp::ShiftLeftLogical, part, shift)?;
            packed = f.binary(BinOp::BitwiseOr, packed, part)?;
        }
        words.push(packed);
    }
    Ok(words)
}

/// The `uvec4` a store of an element of `words` through `view` takes, as [`texel_words`] reads it
/// back: each word split into the view's components, lowest first, zero in the components the
/// view lacks.
fn texel_of(
    f: &mut FunctionBuilder<'_>,
    uvec4: TypeId,
    view: TiledTexelView,
    words: &[Value],
) -> BResult<Value> {
    let bits = view.component_bits();
    let mut components = Vec::with_capacity(4);
    for &word in words {
        if bits == 32 {
            components.push(word);
            continue;
        }
        let mask = u32_value(f, (1 << bits) - 1)?;
        for k in 0..32 / bits {
            let shift = u32_value(f, k * bits)?;
            let part = f.binary(BinOp::ShiftRightLogical, word, shift)?;
            components.push(f.binary(BinOp::BitwiseAnd, part, mask)?);
        }
    }
    let zero = u32_value(f, 0)?;
    components.resize(4, zero);
    f.composite_construct(uvec4, &components)
}

/// The module of `view` as a typed module, before serialization: the module
/// [`tiled_words_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `IMAGE`, `BYTES` and `KEPT`, and its push field the shape, a `uvec4` (the module docs).
fn typed_module(view: TiledTexelView) -> BResult<TypedModule> {
    let n = view.words();
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let image_type = b.ty(Type::Image(ImageType {
        sampled_type: u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: true,
        multisampled: false,
        usage: ImageUsage::Storage,
        format: view.storage_format(),
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_type)?;
    let bytes = words_buffer(&mut b, BYTES, u32_)?;
    let kept = words_buffer(&mut b, KEPT, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let mut lane_values = Vec::with_capacity(n as usize);
    for lane in 0..n {
        lane_values.push(u32_value(&mut f, lane)?);
    }
    let lanes = &lane_values[..];
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let z = f.composite_extract(invocation, &[2])?;
    let image = f.module().global_value(image)?;
    let bytes = f.module().global_value(bytes)?;
    let kept = f.module().global_value(kept)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let sides = f.composite_extract(shape, &[0])?;
    let fields = f.composite_extract(shape, &[1])?;
    let first = f.composite_extract(shape, &[2])?;
    let stride = f.composite_extract(shape, &[3])?;
    let field = |f: &mut FunctionBuilder<'_>, shift: u32, bits: u32| -> BResult<Value> {
        let shift = u32_value(f, shift)?;
        let mask = u32_value(f, (1 << bits) - 1)?;
        let v = f.binary(BinOp::ShiftRightLogical, fields, shift)?;
        f.binary(BinOp::BitwiseAnd, v, mask)
    };
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let width = f.binary(BinOp::BitwiseAnd, sides, low)?;
    let height = f.binary(BinOp::ShiftRightLogical, sides, sixteen)?;
    let step = field(&mut f, 0, 3)?;
    let tx = field(&mut f, 3, 3)?;
    let ty = field(&mut f, 6, 3)?;
    let skip = field(&mut f, 9, 6)?;
    let across = f.binary(BinOp::ShiftRightLogical, fields, sixteen)?;
    let in_x = f.binary(BinOp::ULessThan, x, width)?;
    let in_y = f.binary(BinOp::ULessThan, y, height)?;
    let inside = f.binary(BinOp::LogicalAnd, in_x, in_y)?;
    let body: ArmFn<'_, ()> = Box::new(move |f, _| {
        // The element's first word from the texture's first: its level's, its chain's, its tile
        // row's and tile's, and its place in the tile.
        let row = f.binary(BinOp::ShiftRightLogical, y, ty)?;
        let column = f.binary(BinOp::ShiftRightLogical, x, tx)?;
        let tile = f.binary(BinOp::IMul, row, across)?;
        let tile = f.binary(BinOp::IAdd, tile, column)?;
        let tile_bits = f.binary(BinOp::IAdd, tx, ty)?;
        let tile = f.binary(BinOp::ShiftLeftLogical, tile, tile_bits)?;
        let xm = low_mask(f, tx)?;
        let ym = low_mask(f, ty)?;
        let xi = f.binary(BinOp::BitwiseAnd, x, xm)?;
        let yi = f.binary(BinOp::BitwiseAnd, y, ym)?;
        let m = morton(f, [tx, ty], xi, yi)?;
        let element = f.binary(BinOp::IAdd, tile, m)?;
        let words_n = u32_value(f, n)?;
        let element = f.binary(BinOp::IMul, element, words_n)?;
        let chain = f.binary(BinOp::IMul, z, stride)?;
        let rel = f.binary(BinOp::IAdd, first, chain)?;
        let rel = f.binary(BinOp::IAdd, rel, element)?;
        let at = f.binary(BinOp::IAdd, skip, rel)?;
        let coord = f.composite_construct(uvec3, &[x, y, z])?;
        let image = f.load(image, MemAccess::default())?;
        let word_at = |f: &mut FunctionBuilder<'_>, buffer: Value, base: Value, lane: Value| {
            let place = f.binary(BinOp::IAdd, base, lane)?;
            f.access_chain(buffer, &[zero, place])
        };
        let two = u32_value(f, 2)?;
        let inward = f.binary(BinOp::ULessThan, step, two)?;
        // In: the bytes' words, stored through the view and, kept too, into the kept words.
        let into: ArmFn<'_, ()> = Box::new(move |f, _| {
            let mut words = Vec::with_capacity(lanes.len());
            for &lane in lanes {
                let p = word_at(f, bytes, at, lane)?;
                words.push(f.load(p, MemAccess::default())?);
            }
            let texel = texel_of(f, uvec4, view, &words)?;
            f.write(
                image,
                coord,
                texel,
                &ImageOperands::default(),
                TexelMemory::default(),
            )?;
            let one = u32_value(f, 1)?;
            let keeps = f.binary(BinOp::IEqual, step, one)?;
            let keep: ArmFn<'_, ()> = Box::new(move |f, _| {
                for (&lane, &word) in lanes.iter().zip(&words) {
                    let p = word_at(f, kept, rel, lane)?;
                    f.store(p, word, MemAccess::default())?;
                }
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), keeps, keep, None)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        // Out: the texel read through the view, into the kept words, merged into the bytes, or
        // stored in them.
        let out: ArmFn<'_, ()> = Box::new(move |f, _| {
            let texel = f.read(
                image,
                coord,
                &ImageOperands::default(),
                TexelMemory::default(),
            )?;
            let after = texel_words(f, texel, view)?;
            let two = u32_value(f, 2)?;
            let keeps = f.binary(BinOp::IEqual, step, two)?;
            let keep_after = after.clone();
            let keep: ArmFn<'_, ()> = Box::new(move |f, _| {
                for (&lane, &word) in lanes.iter().zip(&keep_after) {
                    let p = word_at(f, kept, rel, lane)?;
                    f.store(p, word, MemAccess::default())?;
                }
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            let to_bytes: ArmFn<'_, ()> = Box::new(move |f, _| {
                let three = u32_value(f, 3)?;
                let merges = f.binary(BinOp::IEqual, step, three)?;
                let merge_after = after.clone();
                let merge: ArmFn<'_, ()> = Box::new(move |f, _| {
                    for (&lane, &word) in lanes.iter().zip(&merge_after) {
                        let p = word_at(f, kept, rel, lane)?;
                        let before = f.load(p, MemAccess::default())?;
                        let differ = f.binary(BinOp::BitwiseXor, before, word)?;
                        let moved = f.binary(BinOp::INotEqual, differ, zero)?;
                        // A word the commands left as it was is neither read nor written.
                        let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                            let to = word_at(f, bytes, at, lane)?;
                            merge_word(f, to, word, differ)?;
                            Ok(ArmEnd::FallThrough(Vec::new()))
                        });
                        f.if_(&mut (), moved, write, None)?;
                    }
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                let back: ArmFn<'_, ()> = Box::new(move |f, _| {
                    for (&lane, &word) in lanes.iter().zip(&after) {
                        let p = word_at(f, bytes, at, lane)?;
                        f.store(p, word, MemAccess::default())?;
                    }
                    Ok(ArmEnd::FallThrough(Vec::new()))
                });
                f.if_(&mut (), merges, merge, Some(back))?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), keeps, keep, Some(to_bytes))?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), inward, into, Some(out))?;
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, body, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: TiledWordsModule::LOCAL_SIZE,
            y: TiledWordsModule::LOCAL_SIZE,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(TiledWordsModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(TiledWordsModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == BYTES => Some(TiledWordsModule::BYTES_BINDING),
        Symbol::Binding(s) if *s == KEPT => Some(TiledWordsModule::KEPT_BINDING),
        _ => None,
    }
}

/// The tiled word module of `device` moving elements through `view`: built with the typed
/// builder, checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`TiledWordsModule`]. Refused,
/// `Unsupported` (feature `Format`) naming what the device lacks, for the `R32G32_UINT` view
/// without `shaderStorageImageExtendedFormats` (its `Rg32ui`).
pub fn tiled_words_module(
    device: &Device,
    view: TiledTexelView,
) -> Result<TiledWordsModule, Error> {
    let profile = device.profile();
    let m = typed_module(view)?;
    profile
        .check(m.requirements())
        .map_err(|c| Error::Unsupported(Unsupported::needs(Feature::Format, c)))?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the tiled word module needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(TiledWordsModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests;
