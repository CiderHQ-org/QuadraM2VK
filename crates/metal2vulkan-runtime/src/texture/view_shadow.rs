//! Texture views whose image takes no Vulkan view of their format, which hold their texels apart
//! (the coordinator's ruling, 2026-10-06: plans (a) and (b)):
//!
//! - **Another component layout on a device whose Vulkan views reinterpret no image's
//!   components**: a portability subset without `imageViewFormatReinterpretation`, or an adopted
//!   device whose consumer does not say it enabled it (`RawDevice::view_format_reinterpretation`).
//!   Metal views `RGBA8Unorm` as `R32Uint`; such a device makes no Vulkan view of the image in the
//!   view's format (`view_plan.rs`).
//! - **Another format of a texture made without `PixelFormatView` usage**: plain Metal makes the
//!   view (its API validation stops it) and renders through it the same; the texture's image was
//!   made mutable, if at all, only for its sRGB twin's formats (`image_plan.rs`), which keeps the
//!   driver's compression of it, so a view of another format of its class (`BGRA8Unorm` of an
//!   `RGBA8Unorm` texture) is not a view of that image.
//!
//! Such a view:
//!
//! - **A view of a texture over a buffer** (plan (a)) is a texture over the same bytes in the
//!   view's format: an aliasing image where the device lays one out as Metal does, else a shadow
//!   of the bytes, exactly as such a texture is (`shadow.rs`, `Device::storage_over_buffer`).
//! - **A view of an image** (plan (b)) is an image of the view's format and the viewed image's
//!   shape, in memory of its own, a shadow of the viewed image's texels in the levels and layers
//!   the view presents (`ShadowSource::Image`): the encoders copy them in before a command uses
//!   the view and back after it writes through it, by `vkCmdCopyImage` between formats of one
//!   copy unit, which keeps every bit (measured: `runtime_lane::view_reinterpretation::
//!   copies_between_formats_of_one_unit_keep_every_bit`). A command that writes the image both
//!   through the view and through another alias (the image, or another such view) is refused
//!   (`ShadowSync::before`): a copy back would replace the other alias's writes, and only a
//!   buffer's bytes have a merge (`shadow_merge.rs`). One that writes through one alias and reads
//!   through another reads the texels as they were before it, as a texture over a buffer does.
//!   CPU access through the view reads and writes the viewed image. The copies are announced
//!   once per device (`shadowed-texture-view`, performance), naming the cause.
//! - **A view of a shadowed view** presents the viewed image itself where that image takes the
//!   view's format, else a shadow of it of its own.
//! - **Refused**, typed: a view of a sparse texture, of a heap's texture (whose image aliases the
//!   heap's other resources), and of an image the copies cannot serve (formats whose copy units
//!   differ, an image without transfer usage): `Unsupported` of
//!   `Feature::ViewFormatReinterpretation` (not representable) where the device reinterprets no
//!   components, else of `Feature::TextureView` (not implemented).

use super::{Image, Owner, Storage, Texture};
use crate::degradation::Degradation;
use crate::error::Error;
use crate::image_plan::plan_image;
use crate::image_plan::takes_views_of;
use crate::memory::Placement;
use crate::resource::{Identity, ResourceKey};
use crate::shadow::{ShadowCopies, ShadowSource};
use crate::texture::{copy_units, image_error, texture_aspects};
use crate::texture_descriptor::{TextureDescriptor, TextureLayout};
use crate::tracker::UseRange;
use crate::view_plan::{keeps_components, reinterpretation_refused, ViewRecord};
use crate::Device;
use ash::vk;
use metal2vulkan::error::{Feature, Unsupported};
use metal2vulkan::metal::format::direct_format;
use metal2vulkan::metal::{PixelFormat, TextureType};
use metal2vulkan::vk::ImageUsageFlags;
use std::sync::Arc;

/// What a shadowed view's image keeps beside itself (the module docs): its memory, the image whose
/// texels it copies, and the copies.
pub(crate) struct ViewShadow {
    pub(super) placement: Placement,
    pub(super) base: Arc<Image>,
    pub(super) copies: Arc<ShadowCopies>,
}

/// The image whose texels `image` holds: the viewed image of a shadowed view's, else `image`.
pub(super) fn truth(image: &Arc<Image>) -> &Arc<Image> {
    match &image.owner {
        Owner::ViewShadow(shadow) => &shadow.base,
        _ => image,
    }
}

/// [`truth`] of a borrowed image.
fn truth_of(image: &Image) -> &Image {
    match &image.owner {
        Owner::ViewShadow(shadow) => &shadow.base,
        _ => image,
    }
}

impl Texture {
    /// The storage of a view presenting `record` of this texture in a format its image takes no
    /// Vulkan view of (the module docs), for Metal's `call`.
    pub(super) fn reinterpreted(
        &self,
        call: &'static str,
        record: &ViewRecord,
    ) -> Result<Storage, Error> {
        let device = self.device();
        let refused = |image: Option<&Image>| {
            Error::Unsupported(device.apart_refusal(image, record.pixel_format))
        };
        // A two-plane YCbCr texture's view as its `_sRGB` twin converts the same planes.
        if let Some(storage) = self.converted_view(call, record) {
            return storage;
        }
        match &self.0.storage {
            Storage::Image(image) => {
                let base = match &image.owner {
                    Owner::ViewShadow(shadow) => &shadow.base,
                    Owner::Runtime { .. } | Owner::Caller(_) | Owner::HeapUnaliased(_) => image,
                    Owner::Buffer { .. }
                    | Owner::Sparse(_)
                    | Owner::Heap(_)
                    | Owner::HeapTwin
                    | Owner::HeapShadow(_) => return Err(refused(Some(image.as_ref()))),
                };
                if self.device().image_takes_view(base, record.pixel_format) {
                    return Ok(Storage::Image(Arc::clone(base)));
                }
                let shadow =
                    self.device()
                        .new_view_shadow(call, base, &self.root().0.record, record)?;
                Ok(Storage::Image(shadow.shared()))
            }
            // A rotated texture's view holds the stored frame in its format (`texture/rotation.rs`).
            Storage::Shadow { .. } if record.presented.rotation != 0 => {
                self.rotated_view(call, record)
            }
            Storage::Alias { texels, .. } | Storage::Shadow { texels, .. } => {
                let mut descriptor = TextureDescriptor::texture_2d(
                    record.pixel_format,
                    u64::from(record.image_extent[0]),
                    u64::from(record.image_extent[1]),
                    false,
                );
                descriptor.usage = record.usage;
                descriptor.swizzle = record.swizzle;
                descriptor.resource_options = texels.options();
                let layout = descriptor
                    .check(self.device().properties().texture_rules())
                    .map_err(|_| refused(None))?;
                self.device()
                    .storage_over_buffer(call, Arc::clone(texels), &descriptor, &layout)
            }
            // A texture buffer's views are texel buffer views of its bytes, which take any
            // format of its class (`Texture::view`).
            Storage::Texels(_) => Err(refused(None)),
        }
    }

    /// The texture views are made from, at the root of `parentTexture`.
    pub(super) fn root(&self) -> &Texture {
        let mut root = self;
        while let Some(parent) = &root.0.parent {
            root = parent;
        }
        root
    }
}

impl Device {
    /// Whether `image` takes a Vulkan view of a view of pixel format `pf`: it lists or takes the
    /// view's formats (`image_plan::takes_views_of`), and the device's views reinterpret its
    /// components or the view keeps them (`keeps_components`).
    pub(super) fn image_takes_view(&self, image: &Image, pf: PixelFormat) -> bool {
        takes_views_of(image.plan(), pf, self.compile_device().profile())
            && (self.shared().view_reinterpretation || !self.reinterprets(image, pf))
    }

    /// Whether a view of `pf` reinterprets the components of `image`'s format, by their direct
    /// Vulkan representations.
    fn reinterprets(&self, image: &Image, pf: PixelFormat) -> bool {
        match (direct_format(image.plan().pixel_format), direct_format(pf)) {
            (Some(image), Some(view)) => !keeps_components(image, view),
            _ => false,
        }
    }

    /// Why a view of pixel format `pf` that `image` (where the view has one) does not take is
    /// refused, where it is: the device's views reinterpret no components and the view's would
    /// (`Feature::ViewFormatReinterpretation`, not representable), else the runtime holds no such
    /// view's texels apart (`Feature::TextureView`, not implemented).
    fn apart_refusal(&self, image: Option<&Image>, pf: PixelFormat) -> Unsupported {
        let reinterpreting = image.is_some_and(|image| {
            !self.shared().view_reinterpretation && self.reinterprets(truth_of(image), pf)
        });
        if reinterpreting {
            reinterpretation_refused()
        } else {
            Unsupported::not_implemented(Feature::TextureView)
        }
    }

    /// The image of a view presenting `view` of `base` in a format `base` takes no Vulkan view of
    /// (the module docs): [`image_shadow`](Self::image_shadow)'s, announced once per device.
    /// Refused (`Unsupported`, as `apart_refusal` names it) where the copies cannot serve the
    /// pair, and as `new_texture` refuses the image.
    fn new_view_shadow(
        &self,
        call: &'static str,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
    ) -> Result<Image, Error> {
        let apart =
            || Error::Unsupported(self.apart_refusal(Some(base.as_ref()), view.pixel_format));
        let image = self.image_shadow(call, base, texture, view, apart)?;
        let reinterpreting =
            !self.shared().view_reinterpretation && self.reinterprets(base, view.pixel_format);
        self.degrade(Degradation::ShadowedTextureView, || {
            let object = format!(
                "a {:?} view of a {:?} {:?} texture",
                view.pixel_format, texture.pixel_format, texture.texture_type
            );
            let reason = if reinterpreting {
                "the device's Vulkan views reinterpret no image's components (its portability \
                 subset withholds imageViewFormatReinterpretation, or an adopted device's \
                 consumer did not say it enabled it): the view is an image of its own"
            } else {
                "the texture has no PixelFormatView usage, so its image takes no view of another \
                 format than its own and its sRGB twin's (kept so for the driver's compression): \
                 the view is an image of its own"
            }
            .to_owned();
            (object, reason)
        });
        Ok(image)
    }

    /// An image presenting `view` of `base` in memory of its own, a shadow of `base`'s texels
    /// (the module docs): an image of the view's format and usage and the shape of `texture`, the
    /// record of the texture `base` was made for, whose copies move the levels and layers `view`
    /// presents (`ShadowSource::Image`). Refused `refused()` where the copies cannot serve the pair
    /// (no transfer usage, formats whose copy units differ), and as `new_texture` refuses the
    /// image.
    pub(super) fn image_shadow(
        &self,
        call: &'static str,
        base: &Arc<Image>,
        texture: &ViewRecord,
        view: &ViewRecord,
        refused: impl Fn() -> Error,
    ) -> Result<Image, Error> {
        let layout = TextureLayout {
            texture_type: texture.texture_type,
            pixel_format: view.pixel_format,
            extent: texture.image_extent,
            levels: texture.level_count,
            layers: texture.layer_count,
            samples: texture.samples,
        };
        let facts = self
            .shared()
            .textures
            .facts(self.compile_device().profile());
        let plan = plan_image(&layout, view.usage, false, &facts)
            .map_err(|refusal| image_error(call, refusal))?;
        let transfer = ImageUsageFlags::TRANSFER_SRC | ImageUsageFlags::TRANSFER_DST;
        let units = |image: &Image| copy_units(image.plan().format, image.plan().pixel_format);
        let same_units =
            copy_units(plan.format, plan.pixel_format) == units(base) && !units(base).is_empty();
        if !plan.usage.contains(transfer) || !base.plan().usage.contains(transfer) || !same_units {
            return Err(refused());
        }
        let aspects = texture_aspects(&plan);
        let range = |aspects| UseRange::Image {
            aspects,
            levels: view.base_level..view.base_level + view.level_count,
            layers: view.base_layer..view.base_layer + view.layer_count,
        };
        let volume = texture.texture_type == TextureType::Type3D;
        let regions: Vec<vk::ImageCopy> = (view.base_level..view.base_level + view.level_count)
            .map(|level| {
                let [width, height, depth] =
                    texture.image_extent.map(|axis| (axis >> level).max(1));
                let layers = vk::ImageSubresourceLayers {
                    aspect_mask: aspects,
                    mip_level: level,
                    base_array_layer: view.base_layer,
                    layer_count: view.layer_count,
                };
                vk::ImageCopy {
                    src_subresource: layers,
                    src_offset: vk::Offset3D::default(),
                    dst_subresource: layers,
                    dst_offset: vk::Offset3D::default(),
                    extent: vk::Extent3D {
                        width,
                        height,
                        depth: if volume { depth } else { 1 },
                    },
                }
            })
            .collect();
        let identity = Identity {
            key: ResourceKey::new(),
            tracked: base.identity.tracked,
        };
        let (truth, truth_key, truth_range) = (
            base.own_raw(),
            base.identity.key,
            range(texture_aspects(base.plan())),
        );
        let shadow_range = range(aspects);
        let mut descriptor = TextureDescriptor::texture_2d(view.pixel_format, 1, 1, false);
        descriptor.resource_options = base.options;
        descriptor.allow_gpu_optimized_contents = base.allow_gpu_optimized_contents;
        let image =
            self.new_image_owned(plan, &layout, &descriptor, identity, |placement, raw| {
                let copies = ShadowCopies {
                    source: ShadowSource::Image {
                        image: truth,
                        key: truth_key,
                        range: truth_range,
                        shadow: raw,
                        regions,
                    },
                    shadow_key: identity.key,
                    shadow_range,
                    tracked: identity.tracked,
                    drops_writes: false,
                };
                Owner::ViewShadow(Box::new(ViewShadow {
                    placement,
                    base: Arc::clone(base),
                    copies: Arc::new(copies),
                }))
            })?;
        // A view of an image presenting a stand-in, its texture's bytes detached, presents the
        // stand-in of its own plan too, keeping its own image for its copies.
        if base.presents_stand_in() {
            let stand_in = self.stand_in(image.plan())?;
            drop(image.present_stand_in(stand_in));
        }
        Ok(image)
    }
}
