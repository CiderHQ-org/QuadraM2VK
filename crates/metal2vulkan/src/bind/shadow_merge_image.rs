//! The merge of a shadowed texture's writes into its buffer's bytes, reading the shadow itself.
//!
//! [`shadow_merge_module`](super::shadow_merge_module) compares a texture's texels kept before a
//! command with the same texels copied out of its shadow after it. Where the shadow has an
//! unsigned-integer view of its texel's width, [`shadow_merge_image_module`] reads the texels
//! after the command through that view instead, so no copy of them is made: Vulkan states that
//! "Views of compatible formats will have the same mapping between texel coordinates and memory
//! locations irrespective of the format, with only the interpretation of the bit pattern
//! changing" (`VkImageViewCreateInfo`), and an unsigned-integer view returns its components' bits
//! unconverted, so the words a fetch returns are the words a copy of the texel writes (a 4-byte
//! texel's `R8G8B8A8_UINT` bytes packed into its word, as
//! [`shadow_copy_back_module`](super::shadow_copy_back_module) packs them).
//!
//! **The module** (one per view, [`ShadowTexelView`]). One `GLCompute` entry `main` with local
//! size ([`ShadowMergeImageModule::LOCAL_SIZE`], 1, 1), [`ShadowMergeImageModule::PUSH_BYTES`] of
//! `COMPUTE` push constants at offset 0 (the region's shape, [`ShadowMergeImageModule::shape`]),
//! and in set 0:
//!
//! | Binding | Resource |
//! |---|---|
//! | [`ShadowMergeImageModule::IMAGE_BINDING`] | the shadow's sampled view of the module's format ([`ShadowTexelView`]), in the `GENERAL` layout |
//! | [`ShadowMergeImageModule::KEPT_BINDING`] | the region's texels before the command, row after row, a storage buffer |
//! | [`ShadowMergeImageModule::CURRENT_BINDING`] | the bytes the region lies in, a storage buffer merged in place |
//!
//! The shape's words, in order: `origin`, the region's first texel `x0` in its low 16 bits and
//! `y0` in its high 16; `width`, its texels a row; `row_words`, the current buffer's rows apart in
//! words; and `first_word`, the word of the region's first texel in the current binding.
//! Invocation `(x, y)` with `x` below `width` fetches texel `(x0 + x, y0 + y)` of level 0 and, for
//! each of its `n` words `k` (`n` the module's texel width), compares it with word
//! `(y · width + x) · n + k` of the kept buffer; where they differ, word
//! `first_word + y · row_words + x · n + k` of the current buffer takes, in each byte where they
//! differ, the fetched word's byte, and keeps its own in the others. Where they are equal it reads
//! and writes nothing of the current buffer; one at `x` past `width` does nothing. A dispatch takes
//! `ceil(width / LOCAL_SIZE)` by the region's height workgroups
//! ([`ShadowMergeImageModule::groups`]). No word between the region's rows or past its texels is
//! read or written.
//!
//! **Kept in an image** ([`shadow_merge_kept_image_module`]): the same module, but for its kept
//! binding, a sampled image of the module's format ([`ShadowTexelView`]) in the `GENERAL` layout,
//! of at least the shadow's extent, which holds the region's texels before the command at their
//! own coordinates (a copy of the shadow's region into it, `vkCmdCopyImage`). Invocation `(x, y)`
//! fetches texel `(x0 + x, y0 + y)` of both images and compares their words as above. Both views
//! being of one format, the kept words are the shadow's bits at the copy: "Copies are done
//! layer-by-layer starting with the baseArrayLayer member of srcSubresource ... the texel data is
//! copied" bit for bit between compatible formats (`vkCmdCopyImage`).
//!
//! **Synchronization** (consumer): the shadow's last writes → `SHADER_SAMPLED_READ` in
//! `COMPUTE_SHADER`, the kept buffer's copy → `SHADER_STORAGE_READ` (the kept image's copy →
//! `SHADER_SAMPLED_READ`), and the bytes' earlier accesses → `SHADER_STORAGE_READ` and
//! `SHADER_STORAGE_WRITE` there, before the dispatch; its writes → the bytes' next use.

use std::sync::Arc;

use super::shadow_copy_back::{fetched_words, ShadowTexelView};
use super::shadow_merge::{merge_word, u32_value, words_buffer};
use crate::budget::{Budget, Meter};
use crate::device::requirements::Requirements;
use crate::device::Device;
use crate::error::{Error, InternalError, InternalKind};
use crate::pipeline::sym::{PushField, ResourceKind, ResourceSlot, Stage, Symbol};
use crate::spirv::builder::constructs::{ArmEnd, ArmFn};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::error::BResult;
use crate::spirv::ir::{
    BinOp, Dim, ExecMode, ExecModel, ImageOperands, ImageType, ImageUsage, IoDir, MemAccess,
    StorageClass, Type, TypedModule,
};

/// The crate-owned image-reading merge module of one texel width on one device, validated like
/// any output.
#[non_exhaustive]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShadowMergeImageModule {
    /// SPIR-V 1.5 words, entry point `main`, bindings numbered as below in set 0.
    pub spirv: Arc<[u32]>,
    /// What the module needs from the device (within every Vulkan 1.2 device).
    pub requirements: Requirements,
}

impl ShadowMergeImageModule {
    /// Descriptor set of every binding.
    pub const SET: u32 = 0;
    /// The shadow's unsigned-integer view, a sampled image.
    pub const IMAGE_BINDING: u32 = 0;
    /// The region's texels before the command: a storage buffer, row after row, or (the
    /// [`shadow_merge_kept_image_module`] form) a sampled image holding them at their own
    /// coordinates.
    pub const KEPT_BINDING: u32 = 1;
    /// The buffer's bytes after the command, a storage buffer the dispatch merges into.
    pub const CURRENT_BINDING: u32 = 2;
    /// Invocations of one workgroup, along x: one texel each, of one row.
    pub const LOCAL_SIZE: u32 = 64;
    /// Bytes of the `COMPUTE` push-constant range at offset 0: the region's shape.
    pub const PUSH_BYTES: u32 = 16;

    /// The push bytes of a region whose first texel is `[x0, y0]` (each below 65536), `width`
    /// texels a row, whose rows lie `row_words` apart in the current binding from its word
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
/// The binding symbol of the kept buffer in the typed module.
const KEPT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 0,
};
/// The binding symbol of the kept image in the typed module of the kept-image form.
const KEPT_IMAGE: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Texture,
    index: 1,
};
/// The binding symbol of the current buffer in the typed module.
const CURRENT: ResourceSlot = ResourceSlot {
    stage: Stage::Kernel,
    kind: ResourceKind::Buffer,
    index: 1,
};

/// The push field the shape is declared under: a crate-owned module's own words, which no
/// selection places.
const SHAPE: PushField = PushField::DispatchRegion;

/// Where the merge reads the texels before the command (the module docs).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kept {
    /// A storage buffer, row after row: the symbol `KEPT`.
    Buffer,
    /// A sampled image, at their own coordinates: the symbol `KEPT_IMAGE`.
    Image,
}

/// The merge module of `view` reading the kept texels from `kept` as a typed module, before
/// serialization: the module [`shadow_merge_image_module`] (or, the kept texels in an image,
/// [`shadow_merge_kept_image_module`]) serializes, which the CPU interpreter runs. Its bindings
/// are the symbols `IMAGE`, `KEPT` (or `KEPT_IMAGE`) and `CURRENT`, and its push field the shape,
/// a `uvec4` (the module docs).
fn typed_module(view: ShadowTexelView, kept: Kept) -> BResult<TypedModule> {
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
    let kept = match kept {
        Kept::Buffer => KeptAt::Buffer(words_buffer(&mut b, KEPT, u32_)?),
        Kept::Image => {
            KeptAt::Image(b.resource(KEPT_IMAGE, StorageClass::UniformConstant, image_type)?)
        }
    };
    let current = words_buffer(&mut b, CURRENT, u32_)?;
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
    let kept = match kept {
        KeptAt::Buffer(kept) => KeptAt::Buffer(f.module().global_value(kept)?),
        KeptAt::Image(kept) => KeptAt::Image(f.module().global_value(kept)?),
    };
    let current = f.module().global_value(current)?;
    let push = f.module().global_value(push)?;
    let member = u32_value(&mut f, m_shape.index)?;
    let shape = f.access_chain(push, &[member])?;
    let shape = f.load(shape, MemAccess::default())?;
    let origin = f.composite_extract(shape, &[0])?;
    let width = f.composite_extract(shape, &[1])?;
    let row_words = f.composite_extract(shape, &[2])?;
    let first_word = f.composite_extract(shape, &[3])?;
    let inside = f.binary(BinOp::ULessThan, x, width)?;
    let merge: ArmFn<'_, ()> = Box::new(move |f, _| {
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
        // The texel's words before the command: in the kept image, at the texel's coordinates;
        // in the kept buffer, from its first word, row after row.
        let before = match kept {
            KeptAt::Image(kept) => {
                let kept = f.load(kept, MemAccess::default())?;
                let texel = f.fetch(kept, coord, &ops)?;
                fetched_words(f, texel, view)?
            }
            KeptAt::Buffer(kept) => {
                let kept_row = f.binary(BinOp::IMul, y, width)?;
                let kept_texel = f.binary(BinOp::IAdd, kept_row, x)?;
                let kept_at = f.binary(BinOp::IMul, kept_texel, n)?;
                let mut words = Vec::with_capacity(lanes.len());
                for &index in &lanes {
                    let place = f.binary(BinOp::IAdd, kept_at, index)?;
                    let p = f.access_chain(kept, &[zero, place])?;
                    words.push(f.load(p, MemAccess::default())?);
                }
                words
            }
        };
        // The texel's first word in the current buffer, at its row's place.
        let row = f.binary(BinOp::IMul, y, row_words)?;
        let column = f.binary(BinOp::IMul, x, n)?;
        let at = f.binary(BinOp::IAdd, first_word, row)?;
        let at = f.binary(BinOp::IAdd, at, column)?;
        for ((&after, &before), &index) in texel.iter().zip(&before).zip(&lanes) {
            let differ = f.binary(BinOp::BitwiseXor, before, after)?;
            let moved = f.binary(BinOp::INotEqual, differ, zero)?;
            // A word the command left as it was is neither read nor written.
            let write: ArmFn<'_, ()> = Box::new(move |f, _| {
                let place = f.binary(BinOp::IAdd, at, index)?;
                let to = f.access_chain(current, &[zero, place])?;
                merge_word(f, to, after, differ)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), moved, write, None)?;
        }
        Ok(ArmEnd::FallThrough(Vec::new()))
    });
    f.if_(&mut (), inside, merge, None)?;
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize {
            x: ShadowMergeImageModule::LOCAL_SIZE,
            y: 1,
            z: 1,
        }],
    )?;
    b.finalize()
}

/// A kept binding: the buffer's or the image's global (or, inside the function, value).
enum KeptAt<T> {
    Buffer(T),
    Image(T),
}

/// The fixed binding numbers of the module's symbols.
fn binding_of(sym: &Symbol) -> Option<u32> {
    match sym {
        Symbol::DescriptorSet(_) => Some(ShadowMergeImageModule::SET),
        Symbol::Binding(s) if *s == IMAGE => Some(ShadowMergeImageModule::IMAGE_BINDING),
        Symbol::Binding(s) if *s == KEPT || *s == KEPT_IMAGE => {
            Some(ShadowMergeImageModule::KEPT_BINDING)
        }
        Symbol::Binding(s) if *s == CURRENT => Some(ShadowMergeImageModule::CURRENT_BINDING),
        _ => None,
    }
}

/// The image-reading merge module of `device` reading texels through `view`: built with the typed
/// builder, checked by the owned verifier against the device's profile (requirements included),
/// serialized, and numbered with the fixed bindings of [`ShadowMergeImageModule`].
pub fn shadow_merge_image_module(
    device: &Device,
    view: ShadowTexelView,
) -> Result<ShadowMergeImageModule, Error> {
    module_of(device, view, Kept::Buffer)
}

/// The image-reading merge module of `device` reading texels through `view`, the kept texels a
/// sampled image of `view`'s format holding them at their own coordinates (the module docs):
/// as [`shadow_merge_image_module`], its [`ShadowMergeImageModule::KEPT_BINDING`] that image.
pub fn shadow_merge_kept_image_module(
    device: &Device,
    view: ShadowTexelView,
) -> Result<ShadowMergeImageModule, Error> {
    module_of(device, view, Kept::Image)
}

/// [`shadow_merge_image_module`] or [`shadow_merge_kept_image_module`], by `kept`.
fn module_of(
    device: &Device,
    view: ShadowTexelView,
    kept: Kept,
) -> Result<ShadowMergeImageModule, Error> {
    let profile = device.profile();
    let m = typed_module(view, kept)?;
    let mut meter = Meter::new(&Budget::default());
    crate::spirv::verify::verify(&m, profile, &mut meter)?;
    let r = crate::spirv::serialize::serialize(m)?;
    profile.check(&r.requirements).map_err(|c| {
        Error::Internal(InternalError::new(
            InternalKind::RequirementsExceedProfile,
            format!("the image-reading shadow merge needs {c:?}"),
        ))
    })?;
    let words = crate::spirv::serialize::relocate(&r, &binding_of)?;
    Ok(ShadowMergeImageModule {
        spirv: words.into(),
        requirements: r.requirements,
    })
}

#[cfg(test)]
mod tests {
    use super::{
        typed_module, Kept, ShadowMergeImageModule, ShadowTexelView, CURRENT, IMAGE, KEPT,
        KEPT_IMAGE,
    };
    use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
    use crate::spirv::testing::{assert_valid, committed_profiles};

    #[test]
    fn the_image_merge_kernels_are_valid_on_every_committed_profile() {
        for (_, p) in committed_profiles().expect("profiles") {
            for view in ShadowTexelView::ALL {
                for kept in [Kept::Buffer, Kept::Image] {
                    assert_valid(typed_module(view, kept).expect("module"), &p);
                }
            }
        }
    }

    /// Texel `(x, y)`'s word `i` of the image the tests read: `0x0100_0000 · (i + 1) +
    /// 0x1000 · y + x`.
    fn texel_word(x: u32, y: u32, i: u32) -> u32 {
        0x0100_0000 * (i + 1) + 0x1000 * y + x
    }

    /// The `slot` image of `[width, height]` texels whose words are `word(x, y, i)`, as `view`
    /// returns them: an `R8G8B8A8_UINT` view's components its word's bytes, another view's its
    /// words (the components a view lacks 0, and alpha 1).
    fn image_of(
        slot: crate::pipeline::sym::ResourceSlot,
        [width, height]: [u32; 2],
        view: ShadowTexelView,
        word: impl Fn(u32, u32, u32) -> u32,
    ) -> Image {
        let mut image = Image::new(slot, [width, height, 1], 1, 1, 1);
        for y in 0..height {
            for x in 0..width {
                let mut texel = [0, 0, 0, 1];
                match view {
                    ShadowTexelView::R8G8B8A8Uint => {
                        texel = word(x, y, 0).to_le_bytes().map(u32::from);
                    }
                    _ => {
                        for (i, c) in texel.iter_mut().enumerate().take(view.words() as usize) {
                            *c = word(x, y, i as u32);
                        }
                    }
                }
                image.texels[(y * width + x) as usize] = texel;
            }
        }
        image
    }

    /// The `[width, height]` image the tests read, of texels whose words are [`texel_word`].
    fn image(extent: [u32; 2], view: ShadowTexelView) -> Image {
        image_of(IMAGE, extent, view, texel_word)
    }

    fn bytes_of(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|w| w.to_le_bytes()).collect()
    }

    fn words_of(bytes: &[u8]) -> Vec<u32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| u32::from_le_bytes(*b))
            .collect()
    }

    /// The current buffer's words after one dispatch over `region` (`[x0, y0, width, height]`) of
    /// the `extent` image read through `view`, `kept` its region's words before (in a buffer, row
    /// after row, or, by `form`, in an image at their own coordinates), rows `row_words` apart in
    /// `current` from `first_word`, in both invocation orders.
    fn merged(
        extent: [u32; 2],
        (view, form): (ShadowTexelView, Kept),
        region: [u32; 4],
        kept: &[u32],
        (row_words, first_word): (u32, u32),
        current: &[u32],
    ) -> Vec<u32> {
        let m = typed_module(view, form).expect("module");
        let [x0, y0, width, height] = region;
        let shape = ShadowMergeImageModule::shape([x0, y0], width, row_words, first_word);
        let words = view.words();
        // The kept image: the region's kept words at their coordinates, and outside it words no
        // texel of the read image has, which the merge must not read.
        let kept_image = || {
            image_of(KEPT_IMAGE, extent, view, |x, y, i| {
                let inside = (x0..x0 + width).contains(&x) && (y0..y0 + height).contains(&y);
                if inside {
                    kept[(((y - y0) * width + (x - x0)) * words + i) as usize]
                } else {
                    0x5a5a_5a5a ^ texel_word(x, y, i)
                }
            })
        };
        let mut out = Vec::new();
        for order in [Order::Forward, Order::Reverse] {
            let buffer = |slot, words: &[u32]| Buffer {
                slot: Some(slot),
                address: None,
                bytes: bytes_of(words),
            };
            let mut buffers = vec![buffer(CURRENT, current)];
            let mut images = vec![image(extent, view)];
            match form {
                Kept::Buffer => buffers.insert(0, buffer(KEPT, kept)),
                Kept::Image => images.push(kept_image()),
            }
            let res = Resources {
                buffers,
                push: shape.to_vec(),
                spec: Default::default(),
            };
            let [gx, gy] = ShadowMergeImageModule::groups([width, height]);
            let cfg = Config {
                order,
                workgroups: [gx, gy, 1],
                ..Config::default()
            };
            let (outcome, images) = run_with_images(&m, &cfg, res, images).expect("run");
            assert_eq!(images[0], image(extent, view), "the view is only read");
            match form {
                Kept::Buffer => {
                    assert_eq!(outcome.buffers[0], bytes_of(kept), "kept is only read");
                }
                Kept::Image => assert_eq!(images[1], kept_image(), "kept is only read"),
            }
            out.push(words_of(outcome.buffers.last().expect("current")));
        }
        assert_eq!(out[0], out[1], "order independent");
        out.swap_remove(0)
    }

    /// Each byte the texture changed since the keep lands at its row's place; a word whose kept
    /// and fetched words are equal keeps the current buffer's (another writer's write there
    /// stays); a word with one byte changed takes that byte alone; and no word between the rows,
    /// before the first texel or past the last changes: through every view, across workgroups,
    /// the kept words in a buffer or in an image.
    #[test]
    fn the_bytes_the_texture_changed_merge_at_their_rows_alone() {
        for (view, form) in ShadowTexelView::ALL
            .into_iter()
            .flat_map(|view| [(view, Kept::Buffer), (view, Kept::Image)])
        {
            let words = view.words();
            let extent = [90, 7];
            let region = [10, 2, 70, 4];
            let [x0, y0, width, height] = region;
            let row_words = 5 + width * words;
            let first_word = 3;
            let len = (first_word + height * row_words + 2) as usize;
            let current: Vec<u32> = (0..len as u32).map(|i| 0xa5a5_0000 | i).collect();
            // Kept: the fetched words, but word 0 of texel (x, y) where (x + y) % 3 == 0, whose
            // low byte alone differs, and every word of texel (x, y) where (x + y) % 3 == 1,
            // which differs in every byte.
            let mut kept = Vec::new();
            for y in 0..height {
                for x in 0..width {
                    for i in 0..words {
                        let word = texel_word(x0 + x, y0 + y, i);
                        kept.push(match (x + y) % 3 {
                            0 if i == 0 => word ^ 0xff,
                            1 => !word,
                            _ => word,
                        });
                    }
                }
            }
            let got = merged(
                extent,
                (view, form),
                region,
                &kept,
                (row_words, first_word),
                &current,
            );
            let mut want = current.clone();
            for y in 0..height {
                for x in 0..width {
                    for i in 0..words {
                        let at = (first_word + y * row_words + x * words + i) as usize;
                        let word = texel_word(x0 + x, y0 + y, i);
                        want[at] = match (x + y) % 3 {
                            0 if i == 0 => (current[at] & !0xff) | (word & 0xff),
                            1 => word,
                            _ => current[at],
                        };
                    }
                }
            }
            assert_eq!(got, want, "{view:?} {form:?}");
        }
    }
}
