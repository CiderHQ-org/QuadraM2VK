//! What making textures over no-copy buffers costs on the device the cfg selects, the way a
//! consumer's guest browser makes them each frame: report only, in release builds, to weigh the
//! runtime's texture-over-buffer path before and after a change to it.
//!
//! The workload is a guest browser's while it scrolls, as the consumer replays it: each frame, for
//! each of a few IOSurfaces, a new no-copy buffer over the surface's pages and a texture of its
//! plane over that buffer (`new_texture_with_iosurface_plane`, the consumer's
//! `newTextureWithDescriptor:iosurface:plane:`), a render pass that loads and stores the texture
//! and a blit that reads it into a private texture, one command buffer for the frame, waited for;
//! then the textures and buffers are released. The surfaces are `BGRA8Unorm`, of the layouts in
//! [`LAYOUTS`]: the browser's 1307x512 tile with rows padded to 64 bytes (5248, its tight row
//! 5228), a tight-row tile, and other padded rows. Their pages are:
//!
//! - **`anonymous`**: private anonymous memory, whose pointer every measured driver imports;
//! - **`guest-ram`**: a sealed `memfd` mapped shared, as QEMU's guest RAM is, with a dma-buf
//!   source (`HostMemory::with_dma_buf`, `udmabuf`s of the file's pages): RADV refuses the pointer
//!   and imports dma-bufs, ANV and NVIDIA import the pointer. Skipped where `/dev/udmabuf` cannot
//!   be opened.
//!
//! Each case also runs with the buffer kept across frames (`kept`), a new texture over it each
//! frame. Printed per case (`SURFTEX-COST`): the texture's form (aliased or shadowed), the median
//! and 90th percentile of the host's time making the buffer, making the texture and releasing
//! both, per texture; per frame, the encoding and the commit-to-completion wall time and the
//! process's CPU time; and the device's work counts per frame that the path moves (dma-buf images
//! and fallbacks, shadows and those zeroed as made, images at Metal's row pitch over host pages
//! and the driver's refusals of them, asked and recalled, images started on the host, internal
//! submissions). `SURFTEX-QUERY` lines print what the driver answers for a `DRM_FORMAT_MOD_LINEAR`
//! image of host allocations.
//!
//! [`pitch_classes`] asks which row pitches the device lays a texture over a no-copy buffer's
//! host pages at: for each of a few formats and usages (a device each), widths, and pitches of
//! each power-of-two class from 16 bytes up (a pitch's class is the largest power of two dividing
//! it; two pitches a class, the smallest odd multiples of the class that cover the row), each
//! texture made twice, the form its making took (the driver's own linear image, an image at
//! Metal's row pitch, a shadow where the driver refused that image, asked or recalled, or a shadow
//! for another reason) and the median time of each form's making. Printed per format, usage and
//! class (`SURFTEX-PITCH`): the forms, and whether every texture of the class that the driver's
//! own linear image does not alias took one answer; per format and usage (`SURFTEX-PITCH-RULE`)
//! the classes at Metal's pitch, shadowed and mixed, and the reason the first shadow announced.
//!
//! Both are ignored by default: measurements, not checks. Run them by name with `--run-ignored all`, in a
//! build with `--cfg m2v_system_alloc`, whose lib test binary keeps the system allocator as a
//! consumer's process does.

use super::dma_buf_imports::{udmabuf_openable, Calls, Memfd, View};
use super::host_memory::{host_page, Mapping, GUEST_PAGE};
use super::{validated_device_with, Messages, ScratchCache, WAIT};
use crate::gpu::TEST_DEVICE;
use ash::vk;
use metal2vulkan::metal::{PixelFormat, TextureUsage};
use metal2vulkan_runtime::{
    Buffer, CommandQueue, Device, HostMemory, IoSurfacePlane, LoadAction, Origin,
    RenderPassDescriptor, ResourceOptions, Size, StoreAction, Texture, TextureDescriptor,
    WorkCounts,
};
use std::time::Instant;

/// The untimed and the timed frames of each case.
const WARM: usize = 10;
const FRAMES: usize = 100;

/// A surface's plane: its name, width, height and bytes per row (`BGRA8Unorm`, 4 bytes a texel).
const LAYOUTS: [(&str, u64, u64, u64); 5] = [
    // The browser's tile: rows padded to 64 bytes.
    ("1307x512/5248", 1307, 512, 5248),
    // Tight rows, a multiple of 256 bytes.
    ("1024x512/4096", 1024, 512, 4096),
    // Rows padded to 256 bytes.
    ("1307x512/5376", 1307, 512, 5376),
    // A small tile, rows padded to 64 bytes.
    ("300x200/1216", 300, 200, 1216),
    // A full-width strip, rows padded to 64 bytes.
    ("1913x96/7680", 1913, 96, 7680),
];

/// Where a case's surfaces' pages lie.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Pages {
    Anonymous,
    GuestRam,
}

/// The process's CPU time so far, every thread's, in microseconds.
fn process_cpu_us() -> f64 {
    let mut now = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: `now` is a valid timespec for the call to fill.
    let read = unsafe { libc::clock_gettime(libc::CLOCK_PROCESS_CPUTIME_ID, &mut now) };
    assert_eq!(read, 0, "the process's CPU clock");
    now.tv_sec as f64 * 1e6 + now.tv_nsec as f64 / 1e3
}

/// Microseconds since `at`.
fn since(at: Instant) -> f64 {
    at.elapsed().as_secs_f64() * 1e6
}

/// The median and 90th percentile of `values`.
fn median_p90(mut values: Vec<f64>) -> (f64, f64) {
    values.sort_by(f64::total_cmp);
    let at = |p: usize| values[(values.len() * p / 100).min(values.len() - 1)];
    (at(50), at(90))
}

/// A surface's pages, and how a no-copy buffer is made over them.
enum Surface {
    Anonymous(Mapping),
    GuestRam {
        view: View,
        /// The memfd the view maps (it outlives the view's buffers).
        _memfd: std::rc::Rc<Memfd>,
        calls: Calls,
    },
}

impl Surface {
    fn len(width_bytes: u64, height: u64) -> usize {
        ((width_bytes * height) as usize).next_multiple_of(GUEST_PAGE)
    }

    /// A no-copy buffer of `len` bytes over the surface's first byte, of default options, as the
    /// consumer makes one over a surface's pages for each texture of it.
    fn buffer(&self, device: &Device, len: usize, memfd: Option<&Memfd>) -> Buffer {
        let memory = match self {
            // SAFETY: the mapping outlives every buffer over it (the test drops them first).
            Surface::Anonymous(mapping) => unsafe { HostMemory::new(mapping.pointer()) },
            Surface::GuestRam { view, calls, .. } => {
                let memfd = memfd.expect("the guest RAM's file");
                // SAFETY: the view outlives every buffer over it, and the source makes dma-bufs of
                // its pages.
                unsafe { HostMemory::with_dma_buf(view.pointer(0), memfd.source(view, calls)) }
            }
        };
        // SAFETY: as above.
        unsafe {
            device.new_buffer_with_bytes_no_copy(
                memory,
                len as u64,
                ResourceOptions::default(),
                None,
            )
        }
        .unwrap()
    }
}

/// The texture of a surface's plane, `width`x`height` with rows `bpr` bytes apart, over `buffer`.
fn plane_texture(buffer: &Buffer, width: u64, height: u64, bpr: u64) -> Texture {
    let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, width, height, false);
    d.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
    d.resource_options = buffer.resource_options();
    buffer
        .new_texture_with_iosurface_plane(&d, IoSurfacePlane::new(0, bpr, false))
        .unwrap()
}

/// Whether `texture`'s texels are its own image over the buffer's memory: aliased, not shadowed.
fn aliased(texture: &Texture) -> bool {
    // SAFETY: the handle is only compared, while the texture lives.
    unsafe { texture.vk_image() }.is_some()
}

/// One frame's work over `textures`: a pass loading and storing each, and a blit reading each into
/// its private texture of `privates`, in one command buffer, committed and waited for. The host's
/// time encoding, and from the commit until the wait returned.
fn frame(queue: &CommandQueue, textures: &[Texture], privates: &[Texture]) -> (f64, f64) {
    let began = Instant::now();
    let command_buffer = queue.command_buffer().unwrap();
    for texture in textures {
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.color_attachments[0];
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Load;
        attachment.store_action = StoreAction::Store;
        let encoder = command_buffer.render_command_encoder(&pass).unwrap();
        encoder.end_encoding().unwrap();
    }
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    for (texture, private) in textures.iter().zip(privates) {
        let size = Size::new(texture.width(), texture.height(), 1);
        let origin = Origin::new(0, 0, 0);
        blit.copy_from_texture_to_texture(texture, 0, 0, origin, size, private, 0, 0, origin)
            .unwrap();
    }
    blit.end_encoding().unwrap();
    let encoded = since(began);
    let committed = Instant::now();
    command_buffer.commit().unwrap();
    command_buffer.wait_until_completed_for(WAIT).unwrap();
    assert_eq!(command_buffer.error(), None);
    (encoded, since(committed))
}

/// The work counts the texture path moves, `after` less `before`, per frame of `frames`.
fn counts_per_frame(before: &WorkCounts, after: &WorkCounts, frames: usize) -> String {
    let per = |a: u64, b: u64| (a - b) as f64 / frames as f64;
    format!(
        "shadowed textures {:.2} (zeroed {:.2}), pitched host images {:.2} (refused {:.2}, \
         recalled {:.2}), dma-buf images {:.2}, dma-buf image fallbacks {:.2}, dma-buf imports \
         {:.2}, images started on the host {:.2}, internal submissions {:.2}, staging submissions \
         {:.2}",
        per(after.shadowed_textures, before.shadowed_textures),
        per(after.shadows_zeroed, before.shadows_zeroed),
        per(after.pitched_host_images, before.pitched_host_images),
        per(
            after.pitched_host_image_refusals,
            before.pitched_host_image_refusals
        ),
        per(
            after.pitched_host_image_refusals_recalled,
            before.pitched_host_image_refusals_recalled
        ),
        per(after.dma_buf_images, before.dma_buf_images),
        per(
            after.dma_buf_image_fallbacks,
            before.dma_buf_image_fallbacks
        ),
        per(after.dma_buf_imports, before.dma_buf_imports),
        per(
            after.images_started_on_the_host,
            before.images_started_on_the_host
        ),
        per(after.internal_submissions, before.internal_submissions),
        per(after.staging_submissions, before.staging_submissions),
    )
}

/// What the driver answers for a `B8G8R8A8_UNORM` image of `DRM_FORMAT_MOD_LINEAR` tiling made for
/// host allocations, at each usage the runtime may ask of a surface's texture: whether it makes
/// one, and whether it imports host memory for it (`SURFTEX-QUERY`).
fn print_host_modifier_queries(device: &Device) {
    // SAFETY: read while the device lives; nothing is made.
    let raw = unsafe { device.raw() };
    let sampled = vk::ImageUsageFlags::SAMPLED
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    for (name, usage) in [
        ("sampled", sampled),
        (
            "sampled+color",
            sampled | vk::ImageUsageFlags::COLOR_ATTACHMENT,
        ),
        (
            "sampled+color+storage",
            sampled | vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::STORAGE,
        ),
    ] {
        for (tiling, modifier, tiling_name) in [
            (vk::ImageTiling::LINEAR, false, "LINEAR"),
            (
                vk::ImageTiling::DRM_FORMAT_MODIFIER_EXT,
                true,
                "DRM_FORMAT_MOD_LINEAR",
            ),
        ] {
            let mut drm = vk::PhysicalDeviceImageDrmFormatModifierInfoEXT::default()
                .drm_format_modifier(0)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);
            let mut host = vk::PhysicalDeviceExternalImageFormatInfo::default()
                .handle_type(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
            let mut info = vk::PhysicalDeviceImageFormatInfo2::default()
                .format(vk::Format::B8G8R8A8_UNORM)
                .ty(vk::ImageType::TYPE_2D)
                .tiling(tiling)
                .usage(usage)
                .push_next(&mut host);
            if modifier {
                info = info.push_next(&mut drm);
            }
            let mut external = vk::ExternalImageFormatProperties::default();
            let mut properties = vk::ImageFormatProperties2::default().push_next(&mut external);
            // SAFETY: the physical device is the instance's; the chains live for the call.
            let answer = unsafe {
                raw.instance.get_physical_device_image_format_properties2(
                    raw.physical_device,
                    &info,
                    &mut properties,
                )
            };
            let limits = properties.image_format_properties;
            let features = external.external_memory_properties.external_memory_features;
            println!(
                "SURFTEX-QUERY on {}: B8G8R8A8_UNORM {tiling_name} {name} for host allocations: \
                 {answer:?}, max extent {}x{}, external features {:#x}",
                device.name(),
                limits.max_extent.width,
                limits.max_extent.height,
                features.as_raw(),
            );
        }
    }
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn surface_texture_costs() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
        options.layers.clear();
    })
    .unwrap();
    messages.take_notices();
    print_host_modifier_queries(&device);
    let queue = device.new_command_queue().unwrap();
    let page = host_page();
    let lens: Vec<usize> = LAYOUTS
        .iter()
        .map(|&(_, _, h, bpr)| Surface::len(bpr, h))
        .collect();
    let privates: Vec<Texture> = LAYOUTS
        .iter()
        .map(|&(_, w, h, _)| {
            let mut d = TextureDescriptor::texture_2d(PixelFormat::BGRA8Unorm, w, h, false);
            d.usage = TextureUsage::SHADER_READ;
            d.resource_options = ResourceOptions::private();
            device.new_texture(&d).unwrap()
        })
        .collect();
    for pages in [Pages::Anonymous, Pages::GuestRam] {
        let memfd = match pages {
            Pages::GuestRam if !udmabuf_openable("surface_texture_costs") => continue,
            Pages::GuestRam => Some(std::rc::Rc::new(Memfd::new(
                lens.iter().sum::<usize>() + page,
            ))),
            Pages::Anonymous => None,
        };
        let mut at = 0;
        let surfaces: Vec<Surface> = lens
            .iter()
            .map(|&len| match &memfd {
                None => Surface::Anonymous(Mapping::anonymous(len)),
                Some(memfd) => {
                    let view = memfd.view(&[(at, len)]);
                    at += len;
                    Surface::GuestRam {
                        view,
                        _memfd: std::rc::Rc::clone(memfd),
                        calls: Calls::default(),
                    }
                }
            })
            .collect();
        for kept in [false, true] {
            // Each layout's buffer, kept across frames where `kept`.
            let mut held: Vec<Option<Buffer>> = vec![None; LAYOUTS.len()];
            let mut times = vec![(Vec::new(), Vec::new(), Vec::new()); LAYOUTS.len()];
            let mut forms = vec![false; LAYOUTS.len()];
            let (mut encodes, mut waits, mut cpus) = (Vec::new(), Vec::new(), Vec::new());
            let mut before = device.work_counts();
            for k in 0..WARM + FRAMES {
                if k == WARM {
                    before = device.work_counts();
                }
                let cpu = process_cpu_us();
                let mut textures = Vec::new();
                let mut buffers = Vec::new();
                for (index, &(_, w, h, bpr)) in LAYOUTS.iter().enumerate() {
                    let began = Instant::now();
                    let buffer = match &held[index] {
                        Some(buffer) => buffer.clone(),
                        None => surfaces[index].buffer(&device, lens[index], memfd.as_deref()),
                    };
                    let made_buffer = since(began);
                    if kept {
                        held[index] = Some(buffer.clone());
                    }
                    let began = Instant::now();
                    let texture = plane_texture(&buffer, w, h, bpr);
                    let made_texture = since(began);
                    // Asked once, untimed: handing the image out moves it to `GENERAL` by a
                    // waited submission of its own.
                    if k == 0 {
                        forms[index] = aliased(&texture);
                    }
                    if k >= WARM {
                        times[index].0.push(made_buffer);
                        times[index].1.push(made_texture);
                    }
                    textures.push(texture);
                    buffers.push(buffer);
                }
                let (encoded, waited) = frame(&queue, &textures, &privates);
                for (index, (texture, buffer)) in textures.into_iter().zip(buffers).enumerate() {
                    let began = Instant::now();
                    drop((texture, buffer));
                    if k >= WARM {
                        times[index].2.push(since(began));
                    }
                }
                if k >= WARM {
                    encodes.push(encoded);
                    waits.push(waited);
                    cpus.push(process_cpu_us() - cpu);
                }
            }
            let after = device.work_counts();
            drop(held);
            let mode = if kept { "kept buffer" } else { "new buffer" };
            for (index, (buffers, textures, releases)) in times.into_iter().enumerate() {
                let name = LAYOUTS[index].0;
                let (buffer, buffer90) = median_p90(buffers);
                let (texture, texture90) = median_p90(textures);
                let (release, release90) = median_p90(releases);
                println!(
                    "SURFTEX-COST on {}: {pages:?} pages, {mode}, {name}: {}; per texture: buffer \
                     {buffer:.0} us (p90 {buffer90:.0}), texture {texture:.0} us (p90 \
                     {texture90:.0}), release {release:.0} us (p90 {release90:.0})",
                    device.name(),
                    if forms[index] { "aliased" } else { "shadowed" },
                );
            }
            let (encode, encode90) = median_p90(encodes);
            let (wait, wait90) = median_p90(waits);
            let (cpu, cpu90) = median_p90(cpus);
            println!(
                "SURFTEX-FRAME on {}: {pages:?} pages, {mode}, {} textures a frame: encode \
                 {encode:.0} us (p90 {encode90:.0}), commit to completion {wait:.0} us (p90 \
                 {wait90:.0}), process CPU {cpu:.0} us (p90 {cpu90:.0}); per frame: {}",
                device.name(),
                LAYOUTS.len(),
                counts_per_frame(&before, &after, FRAMES),
            );
        }
        let calls: usize = surfaces
            .iter()
            .map(|surface| match surface {
                Surface::GuestRam { calls, .. } => calls.lock().unwrap().len(),
                Surface::Anonymous(_) => 0,
            })
            .sum();
        println!(
            "SURFTEX-SOURCE on {}: {pages:?} pages: the dma-buf sources were called {calls} times",
            device.name()
        );
        drop(surfaces);
    }
    let notices = super::degrade_notices::counts(&messages.take_notices());
    println!("SURFTEX-NOTICES on {}: {notices:?}", device.name());
}

/// The formats [`pitch_classes`] asks about, with their texel bytes.
const SWEPT_FORMATS: [(PixelFormat, u64); 9] = [
    (PixelFormat::BGRA8Unorm, 4),
    (PixelFormat::RGBA8Unorm, 4),
    (PixelFormat::RGB10A2Unorm, 4),
    (PixelFormat::RGBA16Float, 8),
    (PixelFormat::R8Unorm, 1),
    (PixelFormat::RG8Unorm, 2),
    (PixelFormat::R16Float, 2),
    (PixelFormat::R32Float, 4),
    (PixelFormat::RGBA32Float, 16),
];

/// The widths [`pitch_classes`] makes textures of, and their height.
const SWEPT_WIDTHS: [u64; 5] = [5, 64, 300, 1307, 1913];
const SWEPT_HEIGHT: u64 = 16;

/// The largest power-of-two class [`pitch_classes`] asks about: 2^12 bytes.
const SWEPT_CLASSES: std::ops::RangeInclusive<u32> = 4..=12;

/// The form a texture over host pages took, by what its making counted.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Form {
    /// The driver's own linear image lies at the pitch.
    Own,
    /// A `DRM_FORMAT_MOD_LINEAR` image at Metal's row pitch.
    Pitched,
    /// A shadow: the driver refused the image at Metal's row pitch, asked.
    Refused,
    /// A shadow: the driver refused the same image before, and was not asked again.
    Recalled,
    /// A shadow for another reason (the modifier's features or the format query).
    Shadowed,
    /// The texture was refused.
    Failed,
}

/// The two smallest odd multiples of `class` (a power of two) at least `row`: two pitches whose
/// class is exactly `class`.
fn pitches_of_class(row: u64, class: u64) -> [u64; 2] {
    let mut first = row.next_multiple_of(class);
    if (first / class).is_multiple_of(2) {
        first += class;
    }
    [first, first + 2 * class]
}

/// A texture of `d` over a new no-copy buffer of `len` bytes of `mapping`, rows `bpr` apart: the
/// form its making took, and the host's time making it.
fn swept(
    device: &Device,
    mapping: &Mapping,
    len: usize,
    d: &TextureDescriptor,
    bpr: u64,
) -> (Form, f64) {
    // SAFETY: the mapping outlives the buffer, dropped here.
    let buffer = unsafe {
        device.new_buffer_with_bytes_no_copy(
            HostMemory::new(mapping.pointer()),
            len as u64,
            ResourceOptions::default(),
            None,
        )
    }
    .unwrap();
    let mut d = *d;
    d.resource_options = buffer.resource_options();
    let before = device.work_counts();
    let began = Instant::now();
    let made = buffer.new_texture(&d, 0, bpr);
    let took = since(began);
    let after = device.work_counts();
    let moved = |count: fn(&WorkCounts) -> u64| count(&after) > count(&before);
    let form = match made {
        Err(_) => Form::Failed,
        Ok(_) if moved(|c| c.pitched_host_images) => Form::Pitched,
        Ok(_) if moved(|c| c.pitched_host_image_refusals) => Form::Refused,
        Ok(_) if moved(|c| c.pitched_host_image_refusals_recalled) => Form::Recalled,
        Ok(_) if moved(|c| c.shadowed_textures) => Form::Shadowed,
        Ok(_) => Form::Own,
    };
    (form, took)
}

#[test]
#[ignore = "a release-mode measurement: run by name"]
fn pitch_classes() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let largest = SWEPT_FORMATS.iter().map(|&(_, texel)| texel).max().unwrap();
    let widest = SWEPT_WIDTHS.iter().max().unwrap() * largest + (4 << *SWEPT_CLASSES.end());
    let len = Surface::len(widest, SWEPT_HEIGHT);
    let mapping = Mapping::anonymous(len);
    let usages = [
        ("read", TextureUsage::SHADER_READ),
        (
            "read+render",
            TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
        ),
        (
            "read+write",
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
        ),
        (
            "read+write+render",
            TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET,
        ),
    ];
    let mut times: std::collections::BTreeMap<Form, Vec<f64>> = Default::default();
    let mut name = String::new();
    for &(pf, texel) in &SWEPT_FORMATS {
        for &(usage_name, usage) in &usages {
            // A device for each format and usage, which announces its first shadow's reason.
            let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
                options.layers.clear();
            })
            .unwrap();
            messages.take_notices();
            name = device.name().to_owned();
            let (mut pitched_classes, mut refused_classes, mut mixed) =
                (Vec::new(), Vec::new(), Vec::new());
            for class in SWEPT_CLASSES.map(|bits| 1u64 << bits) {
                let mut forms = Vec::new();
                for &w in &SWEPT_WIDTHS {
                    for bpr in pitches_of_class(w * texel, class) {
                        let mut d = TextureDescriptor::texture_2d(pf, w, SWEPT_HEIGHT, false);
                        d.usage = usage;
                        // Made twice: the second making of a refused image recalls the refusal.
                        for _ in 0..2 {
                            let (form, took) = swept(&device, &mapping, len, &d, bpr);
                            times.entry(form).or_default().push(took);
                            forms.push((w, bpr, form));
                        }
                    }
                }
                // Every texture the driver's own linear image does not alias took one answer
                // (a recalled refusal is the driver's earlier one).
                let answers: std::collections::BTreeSet<Form> = forms
                    .iter()
                    .map(|&(_, _, form)| match form {
                        Form::Recalled => Form::Refused,
                        form => form,
                    })
                    .filter(|&form| form != Form::Own)
                    .collect();
                let one = answers.len() <= 1;
                match answers.iter().next() {
                    _ if !one => mixed.push(class),
                    Some(Form::Pitched) => pitched_classes.push(class),
                    Some(_) => refused_classes.push(class),
                    None => {}
                }
                let count = |form: Form| forms.iter().filter(|&&(_, _, f)| f == form).count();
                println!(
                    "SURFTEX-PITCH on {}: {} {usage_name} class {class}: own linear {}, pitched \
                     {}, refused {} (recalled {}), shadowed {}, failed {}; one answer {one}{}",
                    device.name(),
                    pf.name(),
                    count(Form::Own),
                    count(Form::Pitched),
                    count(Form::Refused),
                    count(Form::Recalled),
                    count(Form::Shadowed),
                    count(Form::Failed),
                    if one {
                        String::new()
                    } else {
                        format!(" ({forms:?})")
                    },
                );
            }
            let reason = messages
                .take_notices()
                .into_iter()
                .find_map(|message| message.degradation.map(|notice| notice.reason))
                .unwrap_or_else(|| "no shadow".to_owned());
            println!(
                "SURFTEX-PITCH-RULE on {}: {} {usage_name}: at Metal's pitch {pitched_classes:?}, \
                 shadowed {refused_classes:?}, mixed {mixed:?}; the first shadow: {reason}",
                device.name(),
                pf.name(),
            );
        }
    }
    for (form, took) in times {
        let count = took.len();
        let (median, p90) = median_p90(took);
        println!(
            "SURFTEX-PITCH-TIME on {name}: {form:?}: {count} textures, making {median:.1} us (p90 \
             {p90:.1})"
        );
    }
}
