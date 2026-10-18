//! A shadowed texture's texels copied back into its buffer's rows by a compute dispatch.
//!
//! A texture over a buffer whose rows the device cannot lay as Metal does lives in a shadow, an
//! optimally tiled image copied in from the buffer's bytes before a command and back after it.
//! The copy back is `vkCmdCopyImageToBuffer` on every device. Where a device runs that copy
//! slower than a dispatch that stores the same words (a cost the runtime measures per device), a
//! consumer copies back with one dispatch of [`shadow_copy_back_module`] instead: it reads each
//! texel through an unsigned-integer view of the shadow and stores its words at the texel's place
//! in the buffer, and no other byte. Both forms leave the same bytes: Vulkan states that "Views of
//! compatible formats will have the same mapping between texel coordinates and memory locations
//! irrespective of the format, with only the interpretation of the bit pattern changing"
//! (`VkImageViewCreateInfo`), and an unsigned-integer view returns its components' bits
//! unconverted, so the words a fetch returns are the texel's bytes as the copy writes them. The
//! view is the consumer's choice of [`ShadowTexelView`]: a 4-byte texel's `R32_UINT` view returns
//! its word, and its `R8G8B8A8_UINT` view its bytes in memory order (byte `i` in component `i`),
//! which the kernel packs into the word (byte `i` at bits `8i` onward, [`fetched_words`]); a
//! wider texel's is the 32-bit unsigned format of its width, whose components are its words.
//!
//! **The module** (one per view, [`ShadowTexelView`]). One `GLCompute` entry `main` with local
//! size ([`ShadowCopyBackModule::LOCAL_SIZE`], 1, 1), [`ShadowCopyBackModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 (the region's shape, [`ShadowCopyBackModule::shape`]),
//! and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`ShadowCopyBackModule::IMAGE_BINDING`] | the shadow's sampled view of the module's format ([`ShadowTexelView`]), in the `GENERAL` layout |
//! | [`ShadowCopyBackModule::BYTES_BINDING`] | a storage buffer over the buffer's bytes, a whole number of 4-byte words |
//!
//! The shape's words, in order: `origin`, the region's first texel `x0` in its low 16 bits and
//! `y0` in its high 16; `width`, its texels a row; `row_words`, the buffer's rows apart in words;
//! and `first_word`, the word of the region's first texel in the storage binding. Invocation
//! `(x, y)` with `x` below `width` fetches texel `(x0 + x, y0 + y)` of level 0 and stores its
//! `n` words (`n` the module's texel width) at words `first_word + y · row_words + x · n` onward;
//! one at `x` past `width` does nothing. A dispatch takes `ceil(width / LOCAL_SIZE)` by the
//! region's height workgroups ([`ShadowCopyBackModule::groups`]): a workgroup stores consecutive
//! words of one row. Nothing reads the storage binding, and no word between the region's rows or
//! past its texels is written.
//!
//! **Synchronization** (consumer): the shadow's last writes → `SHADER_SAMPLED_READ` in
//! `COMPUTE_SHADER`, and the buffer's earlier accesses → `SHADER_STORAGE_WRITE` there, before the
//! dispatch; its writes → the buffer's next use (`HOST_READ` in `HOST` before the host reads it).

use std::sync::Arc;

use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::ir::{
    natural_layout, BinOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, IoDir,
    LayoutRule, MemAccess, StorageClass, Type, TypedModule,
};

/// The unsigned-integer view of a shadow that its kernels read its texels through, and a copy in
/// stores them through ([`shadow_copy_back_module`], `shadow_merge_image_module`,
/// `shadow_merge_kept_image_module`, `shadow_copy_in_module`): the format of the texel's width, or for a 4-byte texel either its
/// word's or its bytes'. Every one shares the texel's compatibility class, so a view of any
/// reads and stores the texel's bits unconverted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ShadowTexelView {
    /// `R32_UINT`: a 4-byte texel's word, in component 0.
    R32Uint,
    /// `R8G8B8A8_UINT`: a 4-byte texel's bytes, byte `i` in component `i`.
    R8G8B8A8Uint,
    /// `R32G32_UINT`: an 8-byte texel's words.
    R32G32Uint,
    /// `R32G32B32A32_UINT`: a 16-byte texel's words.
    R32G32B32A32Uint,
}

impl ShadowTexelView {
    /// Every view.
    pub const ALL: [ShadowTexelView; 4] = [
        ShadowTexelView::R32Uint,
        ShadowTexelView::R8G8B8A8Uint,
        ShadowTexelView::R32G32Uint,
        ShadowTexelView::R32G32B32A32Uint,
    ];

    /// The texel's 4-byte words: 1, 2 or 4.
    pub fn words(self) -> u32 {
        match self {
            ShadowTexelView::R32Uint | ShadowTexelView::R8G8B8A8Uint => 1,
            ShadowTexelView::R32G32Uint => 2,
            ShadowTexelView::R32G32B32A32Uint => 4,
        }
    }

    /// The view's storage image format, as a copy in declares it.
    pub(crate) fn storage_format(self) -> crate::spv::ImageFormat {
        match self {
            ShadowTexelView::R32Uint => crate::spv::ImageFormat::R32ui,
            ShadowTexelView::R8G8B8A8Uint => crate::spv::ImageFormat::Rgba8ui,
            ShadowTexelView::R32G32Uint => crate::spv::ImageFormat::Rg32ui,
            ShadowTexelView::R32G32B32A32Uint => crate::spv::ImageFormat::Rgba32ui,
        }
    }
}

/// The crate-owned copy-back module of one view on one device, validated like any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowCopyBackModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ShadowCopyBackModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The shadow's unsigned-integer view, a sampled image.
    pub const IMAGE_BINDING: u32 = 0;
    /// The buffer's bytes, a storage buffer the dispatch writes the texels' words into.
    pub const BYTES_BINDING: u32 = 1;
    /// Invocations of one workgroup, along x: one texel each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the region's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a region whose first texel is `[x0, y0]` (each below 65536), `width`
    /// texels a row, whose rows lie `row_words` apart in the storage binding from its word
    /// `first_word` (the module docs).
    pub fn shape([x0, y0]: [u32; 2], width: u32, row_words: u32, first_word: u32) -> [u8; 16] {
        let words = [(x0 & 0xffff) | (y0 << 16), width, row_words, first_word];
        let mut bytes = [0; 16];
        for (chunk, word) in bytes.as_chunks_mut::<4>().0.iter_mut().zip(words) {
            chunk.copy_from_slice(&word.to_le_bytes());
        }
        bytes
    }

    /// The workgroups of a dispatch over a region `[width, height]` texels.
    pub fn groups([width, height]: [u32; 2]) -> [u32; 2] {
        [width.div_ceil(Self::LOCAL_SIZE), height]
    }
}

/// The binding symbol of the shadow's view in the typed module.
const IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 0,
};
/// The binding symbol of the buffer's bytes in the typed module.
const BYTES: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

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

/// The words of a texel that a fetch through the shadow's `view` returned (`texel`, a `uvec4`):
/// an `R8G8B8A8_UINT` view's components, its bytes, packed into one word (byte `i` at bits `8i`
/// onward); another view's components, its words (the module docs).
pub(super) fn fetched_words(
    f: &mut FunctionBuilder<'_>,
    texel: Value,
    view: ShadowTexelView,
) -> BResult<Vec<Value>> {
    if view != ShadowTexelView::R8G8B8A8Uint {
        return (0..view.words())
            .map(|lane| f.composite_extract(texel, &[lane]))
            .collect();
    }
    let mut word = f.composite_extract(texel, &[0])?;
    for lane in 1..4u32 {
        let byte = f.composite_extract(texel, &[lane])?;
        let shift = u32_value(f, 8 * lane)?;
        let byte = f.binary(BinOp::ShiftLeftLogical, byte, shift)?;
        word = f.binary(BinOp::BitwiseOr, word, byte)?;
    }
    Ok(vec![word])
}

/// The `uvec4` a store of a texel of `words` (its words, in memory order) through the shadow's
/// `view` takes, as [`fetched_words`] reads it back: an `R8G8B8A8_UINT` view's bytes, one a
/// component; another view's words, zero in the components it lacks.
pub(super) fn stored_texel(
    f: &mut FunctionBuilder<'_>,
    uvec4: TypeId,
    view: ShadowTexelView,
    words: &[Value],
) -> BResult<Value> {
    let zero = u32_value(f, 0)?;
    let components = match (view, words) {
        (ShadowTexelView::R8G8B8A8Uint, [word]) => {
            let byte = u32_value(f, 0xff)?;
            let mut bytes = Vec::with_capacity(4);
            for lane in 0..4u32 {
                let shift = u32_value(f, 8 * lane)?;
                let shifted = f.binary(BinOp::ShiftRightLogical, *word, shift)?;
                bytes.push(f.binary(BinOp::BitwiseAnd, shifted, byte)?);
            }
            bytes
        }
        _ => {
            let mut lanes = words.to_vec();
            lanes.resize(4, zero);
            lanes
        }
    };
    f.composite_construct(uvec4, &components)
}

/// The copy-back module of `view` as a typed module, before serialization: the module
/// [`shadow_copy_back_module`] serializes, which the CPU interpreter runs. Its bindings are the
/// symbols `IMAGE` and `BYTES`, and its push field the shape, a `uvec4` (the module docs).
fn typed_module(view: ShadowTexelView) -> BResult<TypedModule> {
    let words = view.words();
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec2 = b.ty(Type::Vector { elem: u32_, len: 2 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let uvec4 = b.ty(Type::Vector { elem: u32_, len: 4 })?;
    let m_shape = b.push_member(SHAPE, uvec4, 0)?;
    let push = b.push_global()?;
    let image_type = b.ty(Type::Image(ImageType {
        sampled_type: u32_,
        dim: Dim::D2,
        depth: false,
        arrayed: false,
        multisampled: false,
        usage: ImageUsage::Sampled,
        format: crate::spv::ImageFormat::Unknown,
    }))?;
    let image = b.resource(IMAGE, StorageClass::UniformConstant, image_type)?;
    let bytes = words_buffer(&mut b, BYTES, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = u32_value(&mut f, 0)?;
    let low = u32_value(&mut f, 0xffff)?;
    let sixteen = u32_value(&mut f, 16)?;
    let n = u32_value(&mut f, words)?;
    let mut lanes = Vec::with_capacity(words as usize);
    for lane in 0..words {
        lanes.push(u32_value(&mut f, lane)?);
    }
    let invocation = f.module().global_value(invocation)?;
    let invocation = f.load(invocation, MemAccess::default())?;
    let x = f.composite_extract(invocation, &[0])?;
    let y = f.composite_extract(invocation, &[1])?;
    let image = f.module().global_value(image)?;
    let bytes = f.module().global_value(bytes)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let origin = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let first_word = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let copy: ArmFn<'_, ()> = Box::new(move |f, _| {
        let x0 = f.binary(BinOp::BitwiseAnd, origin, low)?;
        let y0 = f.binary(BinOp::ShiftRightLogical, origin, sixteen)?;
        let tx = f.binary(BinOp::IAdd, x0, x)?;
        let ty = f.binary(BinOp::IAdd, y0, y)?;
        let coord = f.composite_construct(uvec2, &[tx, ty])?;
        let image = f.load(image, MemAccess::default())?;
        let ops = ImageOperands {
            lod: Some(zero),
            ..ImageOperands::default()
        };
        let texel = f.fetch(image, coord, &ops)?;
        let texel = fetched_words(f, texel, view)?;
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let column = f.binary(BinOp::IMul, x, n)?;
        let at = f.binary(BinOp::IAdd, first_word, row)?;
        let at = f.binary(BinOp::IAdd, at, column)?;
        for (&word, &index) in texel.iter().zip(&lanes) {
            let place = f.binary(BinOp::IAdd, at, index)?;
            let to = f.access_chain(bytes, &[zero, place])?;
            f.store(to, word, MemAccess::default())?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, copy, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ShadowCopyBackModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ShadowCopyBackModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(ShadowCopyBackModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == BYTES => Some(ShadowCopyBackModule::BYTES_BINDING),
        _ => None,
    }
}

/// The copy-back module of `device` reading texels through `view`: built with the typed builder,
/// checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`ShadowCopyBackModule`].
pub fn shadow_copy_back_module(
    device: &Device,
    view: ShadowTexelView,
) -> Result<ShadowCopyBackModule, Error> {
    let profile = device.profile();
    let m = typed_module(view)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the shadow copy back needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ShadowCopyBackModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{typed_module, ShadowCopyBackModule, ShadowTexelView, BYTES, IMAGE};
    use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_copy_back_kernels_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for view in ShadowTexelView::ALL {
                assert_valid(typed_module(view).expect("module"), &p);
            }
        }
    }

    /// A `[width, height]` image of texels whose word `i` at `(x, y)` is
    /// `0x0100_0000 · (i + 1) + 0x1000 · y + x`, as `view` returns it: an `R8G8B8A8_UINT` view's
    /// components its word's bytes, another view's its words (the components a view lacks 0, and
    /// alpha 1).
    fn image([width, height]: [u32; 2], view: ShadowTexelView) -> Image {
        let mut image = Image::new(IMAGE, [width, height, 1], 1, 1, 1);
        for y in 0..height {
            for x in 0..width {
                let word = |i: u32| 0x0100_0000 * (i + 1) + 0x1000 * y + x;
                let mut texel = [0, 0, 0, 1];
                match view {
                    ShadowTexelView::R8G8B8A8Uint => {
                        texel = word(0).to_le_bytes().map(u32::from);
                    }
                    _ => {
                        for (i, c) in texel.iter_mut().enumerate().take(view.words() as usize) {
                            *c = word(i as u32);
                        }
                    }
                }
                image.texels[(y * width + x) as usize] = texel;
            }
        }
        image
    }

    /// The buffer's bytes after one dispatch over `region` (`[x0, y0, width, height]`) of the
    /// `extent` image read through `view` into `bytes` words, rows `row_words` apart from
    /// `first_word`, in both invocation orders.
    fn copied_back(
        extent: [u32; 2],
        view: ShadowTexelView,
        region: [u32; 4],
        (row_words, first_word): (u32, u32),
        bytes: &[u32],
    ) -> Vec<u32> {
        let m = typed_module(view).expect("module");
        let [x0, y0, width, height] = region;
        let shape = ShadowCopyBackModule::shape([x0, y0], width, row_words, first_word);
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let res = Resources {
                buffers: vec![Buffer {
                    slot: Some(BYTES),
                    address: None,
                    bytes: bytes.iter().flat_map(|w| w.to_le_bytes()).collect(),
                }],
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let [gx, gy] = ShadowCopyBackModule::groups([width, height]);
            let cfg = Config {
                order,
                workgroups: [gx, gy, 1],
                ..Config::default()
            };
            let images = vec![image(extent, view)];
            let (outcome, images) = run_with_images(&m, &cfg, res, images).expect("run");
            assert_eq!(images[0], image(extent, view), "the view is only read");
            let words: Vec<u32> = outcome.buffers[0]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| u32::from_le_bytes(*b))
                .collect();
            out.push(words);
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Every texel of a region lands at its row's place as its words, and every word between the
    /// rows, before the first texel and past the last keeps the buffer's own (a padded row's
    /// bytes, and a scissor's outside): through every view, across workgroups.
    #[test]
    fn a_region_lands_at_its_rows_and_no_other_word_changes() {
        for view in ShadowTexelView::ALL {
            let words = view.words();
            let extent = [90, 7];
            // The region's rows 5 + 70 · words words apart, from word 3.
            let region = [10, 2, 70, 4];
            let row_words = 5 + 70 * words;
            let first_word = 3;
            let len = (first_word + 4 * row_words + 2) as usize;
            let bytes: Vec<u32> = (0..len as u32).map(|i| 0xa5a5_0000 | i).collect();
            let got = copied_back(extent, view, region, (row_words, first_word), &bytes);
            let mut want = bytes.clone();
            for y in 0..4 {
                for x in 0..70 {
                    for i in 0..words {
                        let at = first_word + y * row_words + x * words + i;
                        let (tx, ty) = (10 + x, 2 + y);
                        want[at as usize] = 0x0100_0000 * (i + 1) + 0x1000 * ty + tx;
                    }
                }
            }
            assert_eq!(got, want, "{view:?}");
        }
    }

    /// The origin's two halves: a texel past 255 in both coordinates is read from its place.
    #[test]
    fn the_origin_reaches_far_texels() {
        let extent = [300, 260];
        let region = [290, 257, 3, 2];
        let bytes = vec![0u32; 8];
        for view in [ShadowTexelView::R32Uint, ShadowTexelView::R8G8B8A8Uint] {
            let got = copied_back(extent, view, region, (4, 1), &bytes);
            let texel = |x: u32, y: u32| 0x0100_0000 + 0x1000 * y + x;
            let want = vec![
                0,
                texel(290, 257),
                texel(291, 257),
                texel(292, 257),
                0,
                texel(290, 258),
                texel(291, 258),
                texel(292, 258),
            ];
            assert_eq!(got, want, "{view:?}");
        }
    }
}
