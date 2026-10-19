//! A tiled heap texture's elements moved between its image and the heap's bytes through an
//! unsigned-integer view of the image, with no staging (`heap_tiled.rs`: Metal's tiled layout
//! of a texture in a heap over host pages).
//!
//! **The form.** A tiled heap texture held as a shadow of the heap's bytes moves its elements
//! through a staging buffer of the bytes and a tight one of the image's elements, by copies and a
//! dispatch of `bind::tiled_texels_module` each way (`HeapTiledCopies`), and keeps and merges a
//! command's writes as any shadow's, through copies of the image laid out tiled into a scratch
//! (`shadow_merge.rs`). Where its image takes a storage view of its element's width that holds the
//! element's bits ([`view_of`]: a 4-, 8- or 16-byte uncompressed color element whose format's
//! channels an unsigned-integer format of its compatibility class shares, `R8G8B8A8_UINT` for
//! the 8-bit four-channel formats, `R16G16B16A16_UINT`, `R32G32_UINT`, `R32G32B32A32_UINT` and
//! `R32_UINT` for theirs), of a 2D texture, array, cube or cube array, and that view costs the
//! image nothing (`Device::if_free`: its memory requirements are the same with the storage usage
//! and the view's format listed as without them, so the device keeps the image's compression), it
//! moves them with one dispatch of `bind::tiled_words_module` a level instead ([`TiledWords`]):
//!
//! - **Copy in**: each element's words read at its tiled place in the heap's bytes and stored as
//!   its texel through the view, every element of every level and layer.
//! - **Copy in and keep** (`shadow_copy_in.rs`'s pairing, `encoder.rs`): the same dispatch also
//!   stores the words in the merge's kept part, laid out as the texture's bytes.
//! - **Keep**: each texel read through the view, its words stored in the kept part.
//! - **Merge**: each texel read through the view, merged into the heap's bytes in place where its
//!   words differ from the kept ones, byte by byte, as `bind::shadow_merge_module` merges; a word
//!   equal to its kept word is neither read nor written.
//! - **Copy back**: each texel read through the view, its words stored at its place in the heap's
//!   bytes.
//!
//! The texture then holds no staging, tight or shape buffer, and a merge's scratch is the kept part
//! alone, the texture's bytes rounded up, one of the shadow's spares where one is kept
//! (`ScratchSpares`), else new and not zeroed: the keep writes every element's words, and the merge
//! reads no other.
//!
//! **Exact.** Each step reads and writes the bytes the staged form's copies and kernel do, for
//! every element: the view shares the image format's compatibility class, and Vulkan states that
//! "Views of compatible formats will have the same mapping between texel coordinates and memory
//! locations irrespective of the format, with only the interpretation of the bit pattern
//! changing" (`VkImageViewCreateInfo`); an unsigned-integer view stores and returns its
//! components' bits unconverted. A copy in writes every texel of every aspect (a color format's
//! one), as the staged copy does. A copy back writes each element's bytes and no byte between
//! them, where the staged form copies back each level's whole region of each chain, the bytes
//! between the elements as it read them; the bytes it leaves are the same but for another
//! writer's write between them while the copy back runs, which the staged form's copy back
//! undoes. A merge compares and writes the words the staged merge does, the elements' (whose
//! bytes between them its kept and changed copies hold alike).
//!
//! **Synchronization.** Every step's uses are storage reads and writes in `COMPUTE_SHADER`: the
//! heap's bytes, the image (in `GENERAL`, where every image rests) and the kept part, which the
//! encoder's tracker orders as any command's. A merge ends with a barrier to everything after it,
//! as every merge does.
//!
//! **Recording.** Each step's set 0 is pushed (`VK_KHR_push_descriptor`) as the step is recorded,
//! over the heap's `VkBuffer` then (a rebind may give the heap other pages, `shadow.rs`'s
//! `ShadowBuffer`), wherever the encoder records the copy: before a command, at its end, or between
//! a pass's instances. A step makes and holds nothing, so none can be refused once planned; a keep
//! and its merge bind the bytes the merge was planned over, as every merge does.
//!
//! **The views** are the image's own (`Image::own_view`), one a level of every layer, made as the
//! texture is made, which go as the image goes. A device that makes no such view or pushes no set
//! 0, a format of no such view, a texture of another type, or moves that do not fit the device's
//! push constants, storage buffer range or workgroup counts keep the staged form.
//!
//! **Unified memory only.** The steps read and write the heap's bytes from the shader, one word
//! of an element at a time in Metal's tiled order. On a device that shares the host's memory
//! (`DeviceProperties::has_unified_memory`: an integrated GPU or a CPU implementation) those are
//! the host pages the device reads anyway. A device without it (a discrete GPU) reaches the
//! heap's host pages across its bus, a scattered word at a time: on the RTX 5080 a guest's
//! command buffer moving ~610 2D arrays of 12 to 90 layers that way ran 22 s, where the staged
//! form's copies move the same bytes in whole runs into device memory and run its kernel there.
//! Such a device keeps the staged form for every tiled heap texture.

use super::heap_tiled::TiledLayout;
use super::Image;
use crate::binding::{self, Info, SetWrite};
use crate::binding_view::VkView;
use crate::device::functions::Vk;
use crate::error::{internal, Error};
use crate::image_plan::{compatible, ImageFacts, ImagePlan};
use crate::owned_kernel::{OwnedKernel, OwnedModule};
use crate::view_plan::keeps_components;
use crate::Device;
use ash::vk;
use metal2vulkan::bind::{TiledTexelView, TiledWordsLevel, TiledWordsModule, TiledWordsStep};
use metal2vulkan::metal::TextureType;
use metal2vulkan::vk::{
    ComponentMapping, Format, FormatFeatureFlags2, ImageAspectFlags, ImageCreateFlags, ImageType,
    ImageUsageFlags, ImageViewType, SampleCountFlags,
};
use std::sync::OnceLock;

/// The unsigned-integer view of an image of `format` that the moves store and read its texels
/// through, and the kernel's name for it: of the format's element width and channels (the module
/// docs); `None` for any other format.
pub(crate) fn view_of(format: Format) -> Option<(Format, TiledTexelView)> {
    use Format as F;
    Some(match format {
        F::R8G8B8A8_UNORM
        | F::R8G8B8A8_SNORM
        | F::R8G8B8A8_UINT
        | F::R8G8B8A8_SINT
        | F::R8G8B8A8_SRGB
        | F::B8G8R8A8_UNORM
        | F::B8G8R8A8_SRGB => (F::R8G8B8A8_UINT, TiledTexelView::R8G8B8A8Uint),
        F::R16G16B16A16_UNORM
        | F::R16G16B16A16_SNORM
        | F::R16G16B16A16_UINT
        | F::R16G16B16A16_SINT
        | F::R16G16B16A16_SFLOAT => (F::R16G16B16A16_UINT, TiledTexelView::R16G16B16A16Uint),
        F::R32G32_UINT | F::R32G32_SINT | F::R32G32_SFLOAT => {
            (F::R32G32_UINT, TiledTexelView::R32G32Uint)
        }
        F::R32G32B32A32_UINT | F::R32G32B32A32_SINT | F::R32G32B32A32_SFLOAT => {
            (F::R32G32B32A32_UINT, TiledTexelView::R32G32B32A32Uint)
        }
        F::R32_UINT | F::R32_SINT | F::R32_SFLOAT => (F::R32_UINT, TiledTexelView::R32Uint),
        _ => return None,
    })
}

/// Whether a texture of `texture_type` whose image is `plan` takes the moves' views (the module
/// docs): one sample, a 2D image of a 2D texture, array, cube or cube array, of a format with a
/// view ([`view_of`]) whose optimal tiling stores, on a device with unified memory that makes the
/// view and builds its kernel with set 0 pushed; with that view.
fn takes_view(
    device: &Device,
    plan: &ImagePlan,
    texture_type: TextureType,
    facts: &ImageFacts<'_>,
) -> Option<(Format, TiledTexelView)> {
    let planar = matches!(
        texture_type,
        TextureType::Type2D | TextureType::Type2DArray | TextureType::Cube | TextureType::CubeArray
    );
    let (format, view) = view_of(plan.format)?;
    let made = device.shared().view_reinterpretation || keeps_components(plan.format, format);
    let fits = planar
        && device.properties().has_unified_memory
        && plan.samples == SampleCountFlags::COUNT_1
        && plan.image_type == ImageType::TYPE_2D
        && made
        && compatible(plan.format, format, false)
        && facts
            .optimal(format)
            .contains(FormatFeatureFlags2::STORAGE_IMAGE);
    let pushed = || {
        device
            .owned_kernel(OwnedModule::TiledWords(view))
            .is_ok_and(|kernel| kernel.pushed)
    };
    (fits && pushed()).then_some((format, view))
}

/// `plan`, a tiled heap texture's image of `texture_type`, with what the moves' views need (the
/// module docs): storage usage (`EXTENDED_USAGE` where its own format does not store),
/// `MUTABLE_FORMAT` with the view's format listed beside its own (or none listed where it takes
/// any compatible format); `None` where it takes no view ([`takes_view`]). The caller keeps it
/// only where it costs the image nothing (`Device::if_free`).
pub(crate) fn with_views(
    device: &Device,
    plan: &ImagePlan,
    texture_type: TextureType,
    facts: &ImageFacts<'_>,
) -> Option<ImagePlan> {
    let (format, _) = takes_view(device, plan, texture_type, facts)?;
    let mut candidate = plan.clone();
    if !facts
        .optimal(plan.format)
        .contains(FormatFeatureFlags2::STORAGE_IMAGE)
    {
        candidate.flags |= ImageCreateFlags::EXTENDED_USAGE;
    }
    candidate.usage |= ImageUsageFlags::STORAGE;
    let any_format =
        plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT) && plan.view_formats.is_empty();
    candidate.flags |= ImageCreateFlags::MUTABLE_FORMAT;
    if !any_format {
        if candidate.view_formats.is_empty() {
            candidate.view_formats.push(plan.format);
        }
        if !candidate.view_formats.contains(&format) {
            candidate.view_formats.push(format);
        }
    }
    Some(candidate)
}

/// The view `plan` holds for the moves, where it holds one ([`with_views`]): storage usage, and
/// the view's format among those it is viewed as.
fn held_view(
    device: &Device,
    plan: &ImagePlan,
    texture_type: TextureType,
    facts: &ImageFacts<'_>,
) -> Option<(Format, TiledTexelView)> {
    let (format, view) = takes_view(device, plan, texture_type, facts)?;
    let mutable = plan.flags.contains(ImageCreateFlags::MUTABLE_FORMAT);
    let listed = plan.view_formats.is_empty() || plan.view_formats.contains(&format);
    let stores = plan.usage.contains(ImageUsageFlags::STORAGE);
    (stores && (format == plan.format || (mutable && listed))).then_some((format, view))
}

/// A tiled heap texture's moves through its image's views (the module docs): its kernel, whose
/// set 0 is pushed, each level's workgroups and shape of each step, and the views, one a level.
pub(crate) struct TiledWords {
    /// Metal's layout of the texture's elements.
    pub(crate) layout: TiledLayout,
    /// Where the texture's first byte lies in its heap's `VkBuffer`, which a rebind keeps.
    pub(crate) first: u64,
    format: Format,
    kernel: OwnedKernel,
    /// Where the bytes' binding starts in the heap's `VkBuffer`: the texture's first byte rounded
    /// down to `minStorageBufferOffsetAlignment`.
    offset: u64,
    /// Each level's workgroups and its shape of each step, in [`STEPS`]' order.
    levels: Vec<([u32; 3], [[u8; 16]; 5])>,
    /// Each level's storage view of every layer, the image's own (`Image::own_view`): set once,
    /// as the texture is made, before any encoder can name it.
    views: OnceLock<Vec<vk::ImageView>>,
}

impl TiledWords {
    /// The moves of a texture of `texture_type` laid out as `layout`, whose first byte lies at
    /// `first` of its heap's `VkBuffer`, through its image of `plan`, where the plan holds their
    /// views, the device pushes the kernel's set 0, and they fit `device` (the module docs);
    /// `None` where the texture keeps the staged form.
    pub(crate) fn new(
        device: &Device,
        (plan, texture_type, facts): (&ImagePlan, TextureType, &ImageFacts<'_>),
        layout: &TiledLayout,
        first: u64,
    ) -> Option<TiledWords> {
        let (format, view) = held_view(device, plan, texture_type, facts)?;
        let limits = &device.properties().description.limits;
        let narrow = |v: u64| u32::try_from(v).ok();
        if layout.volume
            || layout.unit != 4 * u64::from(view.words())
            || layout.levels.len() != plan.levels as usize
            || layout.chains != u64::from(plan.layers)
        {
            return None;
        }
        // The bytes' binding starts up to 63 words before the texture's first byte
        // (`minStorageBufferOffsetAlignment` is at most 256).
        let range = u64::from(limits.max_storage_buffer_range);
        if layout.size + 256 > range {
            return None;
        }
        let align = limits.min_storage_buffer_offset_alignment.max(4);
        let offset = first - first % align;
        let skip = narrow((first - offset) / 4)?;
        let groups = limits.max_compute_work_group_count;
        let mut levels = Vec::with_capacity(layout.levels.len());
        for (index, level) in layout.levels.iter().enumerate() {
            let side = |axis: usize| (plan.extent[axis] >> index).max(1);
            if level.extent != [u64::from(side(0)), u64::from(side(1))] {
                return None;
            }
            let moved = TiledWordsLevel::new(
                [narrow(level.extent[0])?, narrow(level.extent[1])?],
                plan.layers,
                [narrow(level.tile[0])?, narrow(level.tile[1])?],
                narrow(level.gpu_across)?,
                narrow(level.base)?,
                narrow(layout.stride)?,
            );
            let [gx, gy, gz] = TiledWordsModule::groups(&moved);
            if gx > groups[0] || gy > groups[1] || gz > groups[2] {
                return None;
            }
            let mut shapes = [[0; 16]; 5];
            for (shape, step) in shapes.iter_mut().zip(STEPS) {
                *shape = TiledWordsModule::shape(step, &moved, skip)?;
            }
            levels.push(([gx, gy, gz], shapes));
        }
        // Pushed, its moves make no set as they are recorded: nothing they bind outlives the
        // command that pushes it, so every recorder records them over the bytes' `VkBuffer` then.
        let kernel = device.owned_kernel(OwnedModule::TiledWords(view)).ok()?;
        let bindings = (kernel.images, kernel.storage_images, kernel.buffers);
        if !kernel.pushed || bindings != (0, 1, 2) {
            return None;
        }
        Some(TiledWords {
            layout: layout.clone(),
            first,
            format,
            kernel,
            offset,
            levels,
            views: OnceLock::new(),
        })
    }

    /// Make the views of `image`, the texture's image, one a level of every layer, its own for
    /// its life. Refused as `Image::own_view` refuses one.
    pub(super) fn make_views(&self, image: &Image) -> Result<(), Error> {
        let layers = image.plan().layers;
        let views = (0..self.levels.len() as u32)
            .map(|level| {
                image.own_view(&VkView {
                    view_type: ImageViewType::TYPE_2D_ARRAY,
                    format: self.format,
                    components: ComponentMapping::IDENTITY,
                    aspect: ImageAspectFlags::COLOR,
                    base_level: level,
                    level_count: 1,
                    base_layer: 0,
                    layer_count: layers,
                    usage: ImageUsageFlags::STORAGE,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        self.views
            .set(views)
            .map_err(|_| internal("a tiled heap texture's views made twice"))
    }

    /// Record `step`'s dispatch of every level into `command_buffer`, over the heap's bytes in
    /// `bytes` (its `VkBuffer` as the step is recorded) and, where it keeps or merges, the kept
    /// part at `kept` (its `VkBuffer` and where the part starts there, a multiple of 256): each
    /// level's set 0, pushed, binds its view, the bytes from the texture's first byte rounded
    /// down to `minStorageBufferOffsetAlignment`, and the kept part (the bytes again where there
    /// is none). Nothing over detached bytes (a null `bytes`), which no encoder records
    /// (`EncoderCore::skips`).
    ///
    /// # Safety
    ///
    /// `command_buffer` is recording on the texture's device, outside a render pass, after a
    /// barrier that orders the step after the accesses before it; the image, its views, the
    /// bytes and the kept part are live until it completes, and the image is in the `GENERAL`
    /// layout.
    pub(crate) unsafe fn record(
        &self,
        device: &Vk,
        command_buffer: vk::CommandBuffer,
        (bytes, kept): (vk::Buffer, Option<(vk::Buffer, u64)>),
        step: TiledWordsStep,
    ) {
        // The views are made before the texture is handed out, so before any encoder names it.
        let (Some(at), Some(views), Some(push)) = (
            STEPS.iter().position(|&s| s == step),
            self.views.get(),
            device.push_descriptor(),
        ) else {
            return;
        };
        if bytes == vk::Buffer::null() {
            return;
        }
        let span = self.layout.size.next_multiple_of(4);
        let window = vk::DescriptorBufferInfo {
            buffer: bytes,
            offset: self.offset,
            range: self.first - self.offset + span,
        };
        let kept = kept.map_or(window, |(buffer, offset)| vk::DescriptorBufferInfo {
            buffer,
            offset,
            range: span,
        });
        let point = vk::PipelineBindPoint::COMPUTE;
        // SAFETY: the caller's contract; the kernel's set 0 is pushed (`TiledWords::new`) and
        // each push binds its level's view, the texture's bytes and the kept part, which its
        // shape and workgroups stay inside (`TiledWords::new`).
        unsafe {
            device.cmd_bind_pipeline(command_buffer, point, self.kernel.pipeline);
            for (&view, (groups, shapes)) in views.iter().zip(&self.levels) {
                let writes = [
                    SetWrite {
                        binding: TiledWordsModule::IMAGE_BINDING,
                        element: 0,
                        ty: vk::DescriptorType::STORAGE_IMAGE,
                        info: Info::Image(vk::DescriptorImageInfo {
                            sampler: vk::Sampler::null(),
                            image_view: view,
                            image_layout: vk::ImageLayout::GENERAL,
                        }),
                    },
                    SetWrite {
                        binding: TiledWordsModule::BYTES_BINDING,
                        element: 0,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        info: Info::Buffer(window),
                    },
                    SetWrite {
                        binding: TiledWordsModule::KEPT_BINDING,
                        element: 0,
                        ty: vk::DescriptorType::STORAGE_BUFFER,
                        info: Info::Buffer(kept),
                    },
                ];
                binding::push_set(push, command_buffer, point, self.kernel.layout, &writes);
                device.cmd_push_constants(
                    command_buffer,
                    self.kernel.layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    &shapes[at],
                );
                device.cmd_dispatch(command_buffer, groups[0], groups[1], groups[2]);
            }
        }
    }
}

/// Every step, in the order [`TiledWords`] keeps their shapes.
const STEPS: [TiledWordsStep; 5] = [
    TiledWordsStep::In,
    TiledWordsStep::InKeep,
    TiledWordsStep::Keep,
    TiledWordsStep::Merge,
    TiledWordsStep::Back,
];
