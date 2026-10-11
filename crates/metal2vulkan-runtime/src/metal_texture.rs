//! Metal textures over the `MTLBuffer` of a Vulkan-on-Metal device's imported host memory.
//!
//! A texture over a no-copy buffer aliases the caller's pages through a linear image made for
//! host allocations where the device makes one that lies on Metal's layout (`texture.rs`,
//! `Device::alias_image`). MoltenVK makes none that renders on macOS: its linear images have no
//! colour-attachment or blend feature there, and it refuses the host import for render-target
//! usage. Metal itself makes a texture over a buffer's bytes that renders
//! (`-[MTLBuffer newTextureWithDescriptor:offset:bytesPerRow:]`), and a device that lists
//! `VK_EXT_metal_objects` gives the `MTLBuffer` behind memory it imported
//! (`vkExportMetalObjectsEXT`, memory allocated for export) and makes an image of an
//! `MTLTexture` the caller gives it (`VkImportMetalTextureInfoEXT`, treated as bound). So on such
//! a device a texture over a no-copy buffer is an image of a Metal texture over the import's
//! `MTLBuffer`, at the texels' offset and row pitch, where Metal makes one there: no shadow, and
//! one `MTLBuffer` for Metal's hazard tracking, as the buffer's own uses name.
//!
//! The texture's description is the image's: a 2D texture of one level, slice and sample, or a
//! 2D array of one level and sample whose slices lie as Metal's linear call lays them out (below),
//! of the Metal pixel format whose Vulkan format the image is directly (`FormatEmulation::Direct`),
//! or for one of Apple's extended-range formats whose image holds Metal's bytes in an unsigned
//! integer format (`FormatEmulation::ExtendedRange`) the integer Metal format of that Vulkan
//! format, which holds the same bytes, and for a 4:2:2 texture, whose image is the carrier's, the
//! carrier's own Metal format (`RG8Unorm`, the same bytes; Metal makes no linear 4:2:2 texture);
//! the `MTLTextureUsage` its Vulkan usage needs on Metal, and the buffer's resource options (Metal
//! makes a texture over a buffer only in the buffer's storage and CPU cache modes). Elsewhere
//! (another platform, or no such texture) the answer is `None`, and the texture stays as it was: a
//! linear alias where one lies on Metal's layout, else a shadow.
//!
//! A 2D array: Metal's public `newTextureWithDescriptor:offset:bytesPerRow:` asserts on any array,
//! and only the call outside the public SDK,
//! `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`, makes one over a buffer (the
//! call a caller's `Buffer::new_linear_texture` stands for), each slice `bytesPerRow` x height,
//! rounded up to 128 bytes, after the one before, whatever `bytesPerImage` is (EV ZCI2-spiprobe-m4
//! over a no-copy buffer too). Private Metal that real Metal has belongs to the runtime (the user's
//! ruling of 2026-10-08), so an array is made through that call where the buffer answers to it
//! (`respondsToSelector:`), at an offset that is a multiple of 128 and rows a multiple of 16 (Metal
//! asserts otherwise), and stays shadowed where it does not.

use super::chroma_pairs;
use super::image_plan::ImagePlan;
use super::linear_plan::ARRAY_ALIGNMENT;
use ash::vk;
use metal2vulkan::metal::format::{is_depth_stencil, ExtendedRange, FormatEmulation};
use metal2vulkan::metal::PixelFormat;
use std::ffi::c_void;

/// `MTLTextureUsage`'s bits.
const SHADER_READ: u64 = 0x1;
const SHADER_WRITE: u64 = 0x2;
const RENDER_TARGET: u64 = 0x4;
const PIXEL_FORMAT_VIEW: u64 = 0x10;

/// A Metal texture over a buffer's bytes, as [`texture_over`] makes it: what a rebind makes again
/// over the new pages (`texture.rs`'s `AliasWatch`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct MetalOver {
    /// The texture's `MTLPixelFormat`.
    pixel_format: u64,
    width: u64,
    height: u64,
    /// Its `MTLTextureUsage`.
    usage: u64,
    bytes_per_row: u64,
    /// The slices of a 2D array (made through the linear call), 1 for a 2D texture.
    slices: u64,
    /// A 2D array's slice stride, the linear call's `bytesPerImage` (Metal's own, which the call
    /// keeps whatever it is given); 0 for a 2D texture.
    bytes_per_image: u64,
}

impl MetalOver {
    /// The Metal texture of `plan`, rows `bytes_per_row` bytes apart and (a 2D array's) slices
    /// `bytes_per_image` apart; for an image no such texture is, why (for the shadow's notice):
    /// not 2D of one level and sample, an array whose slices lie otherwise than Metal's linear
    /// call lays them out, or not holding a Metal pixel format's bytes in that format's own Vulkan
    /// format.
    pub(super) fn of(
        plan: &ImagePlan,
        bytes_per_row: u64,
        bytes_per_image: u64,
    ) -> Result<MetalOver, &'static str> {
        let [width, height, depth] = plan.extent;
        let plane = depth == 1
            && plan.levels == 1
            && vk::SampleCountFlags::from(plan.samples) == vk::SampleCountFlags::TYPE_1
            && vk::ImageType::from(plan.image_type) == vk::ImageType::TYPE_2D;
        if !plane {
            return Err(
                "its image is not 2D of one level and sample, as a Metal texture over a buffer \
                 is",
            );
        }
        // Metal makes no linear texture of a depth or stencil format: asked for one, its device
        // aborts the process ("Linear textures do not support depth/stencil pixel formats", an
        // M4 under MoltenVK). Such a texture (a planar surface's, `linear_plan.rs`) is the aspect
        // shadow (`Device::storage_over_buffer`).
        if is_depth_stencil(plan.pixel_format) {
            return Err("Metal makes no linear texture of a depth or stencil format");
        }
        let slices = u64::from(plan.layers);
        let metal_stride = bytes_per_row
            .saturating_mul(u64::from(height))
            .checked_next_multiple_of(ARRAY_ALIGNMENT);
        if slices > 1 && metal_stride != Some(bytes_per_image) {
            return Err(
                "its slices lie otherwise than those of the 2D array Metal's linear call makes \
                 over a buffer",
            );
        }
        let pixel_format = match plan.mapping.emulation {
            FormatEmulation::Direct if chroma_pairs::holds_pairs(plan) => {
                chroma_pairs::CARRIER_PIXEL_FORMAT
            }
            FormatEmulation::Direct if plan.pixel_format != PixelFormat::Invalid => {
                plan.pixel_format
            }
            FormatEmulation::ExtendedRange | FormatEmulation::Bfloat => {
                integer_twin(plan.pixel_format).ok_or(
                    "its format's image holds the texels' two planes side by side, which no Metal \
                 texture over the buffer's bytes is",
                )?
            }
            _ => {
                return Err(
                    "its format's image holds no Metal pixel format's texels in that \
                            format's own Vulkan format",
                )
            }
        };
        let usage_flags = vk::ImageUsageFlags::from(plan.usage);
        let mut usage = SHADER_READ;
        if usage_flags.contains(vk::ImageUsageFlags::STORAGE) {
            usage |= SHADER_WRITE;
        }
        // MoltenVK clears an image (a transfer write) in a render pass.
        if usage_flags
            .intersects(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST)
        {
            usage |= RENDER_TARGET;
        }
        if vk::ImageCreateFlags::from(plan.flags).contains(vk::ImageCreateFlags::MUTABLE_FORMAT) {
            usage |= PIXEL_FORMAT_VIEW;
        }
        Ok(MetalOver {
            pixel_format: u64::from(pixel_format.as_raw()),
            width: u64::from(width),
            height: u64::from(height),
            usage,
            bytes_per_row,
            slices,
            bytes_per_image: if slices > 1 { bytes_per_image } else { 0 },
        })
    }

    /// The texture's description, for the shadow's notice where Metal makes none.
    pub(super) fn describe(&self) -> String {
        let slices = if self.slices > 1 {
            format!(
                " in {} slices {} bytes apart",
                self.slices, self.bytes_per_image
            )
        } else {
            String::new()
        };
        format!(
            "MTLPixelFormat {} of {}x{}{slices}, usage {:#x}, with rows of {} bytes",
            self.pixel_format, self.width, self.height, self.usage, self.bytes_per_row
        )
    }
}

/// The unsigned integer Metal format of the Vulkan format an image of Apple's extended-range
/// format `pf` holds Metal's bytes in (`ExtendedRange::raw_format`), which holds the same bytes:
/// `R32Uint` for `BGR10_XR` and its sRGB twin (`R32_UINT`), `RGBA16Uint` for `BGRA10_XR` and its
/// twin (`R16G16B16A16_UINT`), `R16Uint` and `RG16Uint` for the padded 10-bit formats, `R16Uint`,
/// `RG16Uint` and `RGBA16Uint` for the bfloat formats. `None` for the two-plane formats, whose
/// image holds both planes side by side (not Metal's bytes), and any other format.
pub(super) fn integer_twin(pf: PixelFormat) -> Option<PixelFormat> {
    match ExtendedRange::of(pf)? {
        ExtendedRange::Bgr10 { .. } => Some(PixelFormat::R32Uint),
        ExtendedRange::Bgra10 { .. } => Some(PixelFormat::RGBA16Uint),
        ExtendedRange::R10X6 | ExtendedRange::R10X6Srgb => Some(PixelFormat::R16Uint),
        ExtendedRange::Rg10X12 | ExtendedRange::Rg10X12Srgb => Some(PixelFormat::RG16Uint),
        ExtendedRange::R16Bfloat => Some(PixelFormat::R16Uint),
        ExtendedRange::Rg16Bfloat => Some(PixelFormat::RG16Uint),
        ExtendedRange::Rgba16Bfloat => Some(PixelFormat::RGBA16Uint),
        _ => None,
    }
}

/// A Metal texture the runtime made (+1), released when this goes.
pub(super) struct MetalTexture(*mut c_void);

impl MetalTexture {
    /// The `id<MTLTexture>`, for `VkImportMetalTextureInfoEXT`.
    pub(super) fn id(&self) -> vk::MTLTexture_id {
        self.0
    }
}

impl Drop for MetalTexture {
    fn drop(&mut self) {
        // SAFETY: the texture is the runtime's own reference, released once.
        unsafe { objc::release(self.0) };
    }
}

/// The Metal texture `over` describes over `buffer`'s bytes from `offset`, in the buffer's
/// storage and CPU cache modes; where Metal makes none there, why: no Metal, off its alignment
/// for the format (or an array's), a buffer that has no linear call for an array, or refused.
///
/// # Safety
///
/// `buffer` is a live `id<MTLBuffer>` holding at least the texture's bytes from `offset`.
pub(super) unsafe fn texture_over(
    buffer: vk::MTLBuffer_id,
    over: &MetalOver,
    offset: u64,
) -> Result<MetalTexture, &'static str> {
    let words = [
        over.pixel_format,
        over.width,
        over.height,
        over.usage,
        over.bytes_per_row,
        offset,
        over.slices,
        over.bytes_per_image,
    ];
    // SAFETY: the caller's.
    let made = unsafe { objc::texture_over(buffer, words) }?;
    Ok(MetalTexture(made))
}

#[cfg(target_os = "macos")]
mod objc {
    //! The few Objective-C messages a Metal texture over a buffer takes, sent through the
    //! Objective-C runtime the Metal framework (loaded by the Vulkan driver) runs on.

    use std::ffi::{c_char, c_void, CStr};

    /// The words of a texture over a buffer's bytes: `super::MetalOver`'s and the offset.
    struct Texture {
        pixel_format: u64,
        width: u64,
        height: u64,
        usage: u64,
        bytes_per_row: u64,
        offset: u64,
        slices: u64,
        bytes_per_image: u64,
    }

    type Id = *mut c_void;
    type Sel = *const c_void;

    #[link(name = "objc", kind = "dylib")]
    extern "C" {
        fn objc_getClass(name: *const c_char) -> Id;
        fn sel_registerName(name: *const c_char) -> Sel;
        fn objc_msgSend();
        fn objc_autoreleasePoolPush() -> *mut c_void;
        fn objc_autoreleasePoolPop(pool: *mut c_void);
    }

    /// `objc_msgSend` as a function of the message's arguments and answer.
    macro_rules! send {
        ($receiver:expr, $selector:expr $(, $argument:expr => $kind:ty)* ; $answer:ty) => {{
            // SAFETY: `objc_msgSend` is called with the receiver's method's exact signature.
            let send = unsafe {
                std::mem::transmute::<
                    unsafe extern "C" fn(),
                    unsafe extern "C" fn(Id, Sel $(, $kind)*) -> $answer,
                >(objc_msgSend)
            };
            // SAFETY: the selector names a method of the receiver taking those arguments.
            unsafe { send($receiver, selector($selector) $(, $argument)*) }
        }};
    }

    fn selector(name: &CStr) -> Sel {
        // SAFETY: a NUL-terminated name; registering a selector has no other precondition.
        unsafe { sel_registerName(name.as_ptr()) }
    }

    /// `[object release]`.
    ///
    /// # Safety
    ///
    /// `object` is a live object the caller holds a reference to, which this gives up.
    pub(super) unsafe fn release(object: Id) {
        send!(object, c"release"; ());
    }

    /// The `+1` texture over `buffer`'s bytes (`newTextureWithDescriptor:offset:bytesPerRow:`,
    /// or for an array `newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:`), or
    /// why none is made.
    ///
    /// # Safety
    ///
    /// `buffer` is a live `id<MTLBuffer>`.
    pub(super) unsafe fn texture_over(buffer: Id, words: [u64; 8]) -> Result<Id, &'static str> {
        let [pixel_format, width, height, usage, bytes_per_row, offset, slices, bytes_per_image] =
            words;
        let texture = Texture {
            pixel_format,
            width,
            height,
            usage,
            bytes_per_row,
            offset,
            slices,
            bytes_per_image,
        };
        // SAFETY: no precondition; popped below.
        let pool = unsafe { objc_autoreleasePoolPush() };
        // SAFETY: the caller's.
        let made = unsafe { texture_in_pool(buffer, &texture) };
        // SAFETY: the pool pushed above, the innermost.
        unsafe { objc_autoreleasePoolPop(pool) };
        made
    }

    /// [`texture_over`] inside an autorelease pool.
    ///
    /// # Safety
    ///
    /// As [`texture_over`]'s.
    unsafe fn texture_in_pool(buffer: Id, over: &Texture) -> Result<Id, &'static str> {
        let refused = "Metal answers no texture there";
        // SAFETY: a NUL-terminated name.
        let class = unsafe { objc_getClass(c"MTLTextureDescriptor".as_ptr()) };
        if class.is_null() || buffer.is_null() {
            return Err(refused);
        }
        let device: Id = send!(buffer, c"device"; Id);
        if device.is_null() {
            return Err(refused);
        }
        let alignment: usize = send!(
            device,
            c"minimumLinearTextureAlignmentForPixelFormat:",
            over.pixel_format as usize => usize;
            usize
        );
        let alignment = alignment.max(1) as u64;
        if !over.offset.is_multiple_of(alignment) || !over.bytes_per_row.is_multiple_of(alignment) {
            return Err("the offset or the rows are off Metal's linear texture alignment");
        }
        let array = over.slices > 1;
        let linear = c"newLinearTextureWithDescriptor:offset:bytesPerRow:bytesPerImage:";
        if array {
            // Metal asserts on an array's offset off 128 (`ARRAY_ALIGNMENT`) or rows off 16.
            if !over.offset.is_multiple_of(super::ARRAY_ALIGNMENT)
                || !over.bytes_per_row.is_multiple_of(16)
            {
                return Err("the offset or the rows are off Metal's 2D array alignment");
            }
            let answers: i8 = send!(buffer, c"respondsToSelector:", selector(linear) => Sel; i8);
            if answers == 0 {
                return Err(
                    "Metal's buffer has no newLinearTextureWithDescriptor:offset:bytesPerRow:\
                     bytesPerImage:, the only call that makes a 2D array over a buffer",
                );
            }
        }
        let options: usize = send!(buffer, c"resourceOptions"; usize);
        let allocated: Id = send!(class, c"alloc"; Id);
        let descriptor: Id = send!(allocated, c"init"; Id);
        if descriptor.is_null() {
            return Err(refused);
        }
        // MTLTextureType2D, or MTLTextureType2DArray.
        let texture_type: usize = if array { 3 } else { 2 };
        send!(descriptor, c"setTextureType:", texture_type => usize; ());
        send!(descriptor, c"setPixelFormat:", over.pixel_format as usize => usize; ());
        send!(descriptor, c"setWidth:", over.width as usize => usize; ());
        send!(descriptor, c"setHeight:", over.height as usize => usize; ());
        if array {
            send!(descriptor, c"setArrayLength:", over.slices as usize => usize; ());
        }
        send!(descriptor, c"setUsage:", over.usage as usize => usize; ());
        send!(descriptor, c"setResourceOptions:", options => usize; ());
        let texture: Id = if array {
            send!(
                buffer,
                linear,
                descriptor => Id,
                over.offset as usize => usize,
                over.bytes_per_row as usize => usize,
                over.bytes_per_image as usize => usize;
                Id
            )
        } else {
            send!(
                buffer,
                c"newTextureWithDescriptor:offset:bytesPerRow:",
                descriptor => Id,
                over.offset as usize => usize,
                over.bytes_per_row as usize => usize;
                Id
            )
        };
        // SAFETY: the descriptor this made (+1).
        unsafe { release(descriptor) };
        if texture.is_null() {
            Err(refused)
        } else {
            Ok(texture)
        }
    }
}

#[cfg(not(target_os = "macos"))]
mod objc {
    //! No Metal: no texture is made.

    use std::ffi::c_void;

    /// Never reached: no texture is made here.
    pub(super) unsafe fn release(_object: *mut c_void) {}

    /// No texture: there is no Metal here.
    pub(super) unsafe fn texture_over(
        _buffer: *mut c_void,
        _words: [u64; 8],
    ) -> Result<*mut c_void, &'static str> {
        Err("there is no Metal here")
    }
}
