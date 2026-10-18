//! Textures in automatic and placement heaps on the device the cfg selects, under the validation
//! layer:
//!
//! - **Answers.** `heap_texture_size_and_align` answers each probe shape the same each time, a
//!   power-of-two alignment and a size that holds the texels; `HEAPTEX` lines print it beside
//!   Metal's own answer (an M4 on macOS 27.0.1 and an M3 Max on 26.7 answer alike). On MoltenVK,
//!   whose image requirement is Metal's, the two are equal, but for a depth or stencil texture with
//!   `allowGPUOptimizedContents` off. A texture with the property off takes the image of one of
//!   `PixelFormatView` usage, on every device the answer of that texture (for a depth or stencil
//!   format, at least that answer: every answer holds Metal's where it is measured, and Metal keeps
//!   such a texture of `PixelFormatView` usage compressed), which MoltenVK makes a
//!   `PixelFormatView` Metal texture where it is a color one: Metal's answer for the texture with
//!   the property off. A depth or stencil texture's metadata no Vulkan request drops there, so its
//!   answer is Metal's for the texture with the property on.
//! - **Placement.** A texture fits at every multiple of the answered alignment of a placement
//!   heap, its `heap_offset` that offset; an offset off the alignment, bytes past the heap's end,
//!   another storage or CPU cache mode and the other heap type's call are refused `Invalid`, as
//!   Metal's validation stops them. An automatic heap places first fit, its
//!   `used_size` the sum of its textures' answered sizes, and takes a range back when its
//!   texture is made aliasable or goes.
//! - **Aliasing.** A heap's buffer and texture over the same bytes alias them, as Metal's do: a
//!   texture write changes the buffer's bytes, and a write through the buffer
//!   changes every texel the texture reads, ordered by the heap's tracking within one command
//!   buffer. The bytes' layout is each device's own (Metal's is Apple's tiled or compressed
//!   layout), so the test asserts the change, not the bytes.

use super::degrade_notices::counts;
use super::heap_metal_sizes::stacked_nil_alignment;
use super::host_memory::{gpu_fill, gpu_read, on_device, run};
use super::presenting::Adopted;
use super::textures::{no_texture_alone, pattern, texel_bytes};
use super::{
    bound_buffers, buffer_with, encode_case, pipeline_state, public_case, read_buffer, run_case,
    texture_descriptor, validated_device, validated_device_with, CaseResources, Messages,
    PublicCase, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::candidate::engine::ComputeOutput;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::device::{DriverFact, HeapCompressionPolicy};
use metal2vulkan::error::{Feature, LinearNeed};
use metal2vulkan::metal::format::is_depth_stencil;
use metal2vulkan::metal::{
    IndirectCommandType, PixelFormat, PrimitiveType, RenderStages, ResourceUsage, TextureType,
    TextureUsage,
};
use metal2vulkan::pipeline::{RenderPipelineDescriptor, StageSource};
use metal2vulkan::vk::{DriverId, NumericFormat};
use metal2vulkan_devtools::{engine_dir, llvm_assemble};
use metal2vulkan_runtime::{
    BlitOption, Buffer, CommandQueue, CpuCacheMode, Degradation, Device, Error, HazardTrackingMode,
    Heap, HeapDescriptor, HeapProblem, HeapType, IndirectCommandBufferDescriptor, Invalid,
    LoadAction, Nil, Origin, PipelineOptions, Problem, Region, RenderPassDescriptor,
    ResourceOptions, RuntimeGap, Size, SizeAndAlign, StorageMode, StoreAction, Texture,
    TextureDescriptor, TextureProblem,
};
use std::collections::{BTreeMap, BTreeSet};

fn invalid(call: &'static str, problem: Problem) -> Error {
    Error::Invalid(Invalid::new(call, problem))
}

/// What `call` answers where plain Metal answers nil.
fn nil(call: &'static str, problem: Problem) -> Error {
    Error::Nil(Nil::new(call, problem))
}

/// A probe shape and Metal's answer for it, `Private`.
struct Shape {
    name: &'static str,
    texture_type: TextureType,
    pixel_format: PixelFormat,
    extent: [u64; 3],
    array_length: u64,
    levels: u64,
    samples: u64,
    usage: TextureUsage,
    /// `allowGPUOptimizedContents`.
    optimized: bool,
    /// Metal's `(size, align)`.
    metal: (u64, u64),
}

impl Shape {
    fn descriptor(&self, options: ResourceOptions) -> TextureDescriptor {
        let [width, height, depth] = self.extent;
        let mut descriptor = TextureDescriptor::default();
        descriptor.texture_type = self.texture_type;
        descriptor.pixel_format = self.pixel_format;
        (descriptor.width, descriptor.height, descriptor.depth) = (width, height, depth);
        descriptor.mipmap_level_count = self.levels;
        descriptor.sample_count = self.samples;
        descriptor.array_length = self.array_length;
        descriptor.usage = self.usage;
        descriptor.resource_options = options;
        descriptor.allow_gpu_optimized_contents = self.optimized;
        descriptor
    }

    /// The bytes level 0 of one slice holds at least.
    fn texel_bytes(&self) -> u64 {
        let texel = match self.pixel_format {
            PixelFormat::R8Unorm => 1,
            PixelFormat::RG8Unorm => 2,
            _ => 4,
        };
        let [width, height, depth] = self.extent;
        width * height * depth * texel * self.samples
    }
}

const READ: TextureUsage = TextureUsage::SHADER_READ;

/// Metal's answers for the probe shapes (the M4 and the M3 Max alike).
fn shapes() -> Vec<Shape> {
    let plain = |name, pixel_format, side: u64, levels, usage, optimized, metal| Shape {
        name,
        texture_type: TextureType::Type2D,
        pixel_format,
        extent: [side, side, 1],
        array_length: 1,
        levels,
        samples: 1,
        usage,
        optimized,
        metal,
    };
    let write = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
    let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
    let rgba = PixelFormat::RGBA8Unorm;
    vec![
        plain("rgba8-16", rgba, 16, 1, READ, true, (1152, 128)),
        plain("rgba8-16-write", rgba, 16, 1, write, true, (1024, 128)),
        plain("rgba8-16-target", rgba, 16, 1, target, true, (1152, 128)),
        plain(
            "rgba8-16-unoptimized",
            rgba,
            16,
            1,
            READ,
            false,
            (1024, 128),
        ),
        plain("rgba8-17", rgba, 17, 1, READ, true, (4224, 128)),
        plain("rgba8-64", rgba, 64, 1, READ, true, (16512, 128)),
        plain("rgba8-64-write", rgba, 64, 1, write, true, (16384, 128)),
        plain("rgba8-100", rgba, 100, 1, READ, true, (66048, 128)),
        plain("rgba8-256-mips", rgba, 256, 9, READ, true, (352768, 128)),
        plain(
            "r8-64",
            PixelFormat::R8Unorm,
            64,
            1,
            READ,
            true,
            (4224, 128),
        ),
        plain(
            "rg8-64",
            PixelFormat::RG8Unorm,
            64,
            1,
            READ,
            true,
            (8320, 128),
        ),
        Shape {
            name: "rgba8-1d-128",
            texture_type: TextureType::Type1D,
            extent: [128, 1, 1],
            ..plain("", rgba, 1, 1, READ, true, (512, 128))
        },
        Shape {
            name: "rgba8-array3-33",
            texture_type: TextureType::Type2DArray,
            extent: [33, 33, 1],
            array_length: 3,
            ..plain("", rgba, 1, 1, READ, true, (49536, 16384))
        },
        Shape {
            name: "rgba8-cube-33",
            texture_type: TextureType::Cube,
            extent: [33, 33, 1],
            ..plain("", rgba, 1, 1, READ, true, (99072, 16384))
        },
        Shape {
            name: "rgba8-3d-33",
            texture_type: TextureType::Type3D,
            extent: [33, 33, 33],
            ..plain("", rgba, 1, 1, READ, true, (544896, 16384))
        },
        Shape {
            name: "rgba8-ms4-33",
            texture_type: TextureType::Type2DMultisample,
            extent: [33, 33, 1],
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 1, 1, READ, true, (66048, 128))
        },
        // The property off (an M3 Max and an M4 alike).
        plain("rgba8-256", rgba, 256, 1, READ, true, (264192, 128)),
        plain(
            "rgba8-256-unoptimized",
            rgba,
            256,
            1,
            READ,
            false,
            (262144, 128),
        ),
        Shape {
            name: "bgra8-512x128-target",
            extent: [512, 128, 1],
            ..plain(
                "",
                PixelFormat::BGRA8Unorm,
                1,
                1,
                target,
                true,
                (264192, 128),
            )
        },
        Shape {
            name: "bgra8-512x128-target-unoptimized",
            extent: [512, 128, 1],
            ..plain(
                "",
                PixelFormat::BGRA8Unorm,
                1,
                1,
                target,
                false,
                (262144, 128),
            )
        },
        Shape {
            name: "rgba8-ms4-256",
            texture_type: TextureType::Type2DMultisample,
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 256, 1, READ, true, (1056768, 128))
        },
        Shape {
            name: "rgba8-ms4-256-unoptimized",
            texture_type: TextureType::Type2DMultisample,
            samples: 4,
            usage: TextureUsage::RENDER_TARGET,
            ..plain("", rgba, 256, 1, READ, false, (1048576, 128))
        },
        plain(
            "depth32-64-target",
            PixelFormat::Depth32Float,
            64,
            1,
            target,
            true,
            (16512, 128),
        ),
        plain(
            "depth32-64-target-unoptimized",
            PixelFormat::Depth32Float,
            64,
            1,
            target,
            false,
            (16384, 128),
        ),
    ]
}

/// A `side`-square `RGBA8Unorm` texture's descriptor, read by shaders, with `options`.
fn rgba8(side: u64, options: ResourceOptions) -> TextureDescriptor {
    let mut descriptor = TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, side, side, false);
    descriptor.resource_options = options;
    descriptor
}

/// Metal's answer for `shape` with `allowGPUOptimizedContents` on: its twin's among `shapes`.
fn metal_optimized(shapes: &[Shape], shape: &Shape) -> (u64, u64) {
    shapes
        .iter()
        .find(|twin| {
            twin.optimized
                && (
                    twin.texture_type,
                    twin.pixel_format,
                    twin.extent,
                    twin.usage,
                ) == (
                    shape.texture_type,
                    shape.pixel_format,
                    shape.extent,
                    shape.usage,
                )
                && (twin.array_length, twin.levels, twin.samples)
                    == (shape.array_length, shape.levels, shape.samples)
        })
        .map(|twin| twin.metal)
        .unwrap_or_else(|| panic!("{}: no optimized twin", shape.name))
}

/// A heap of `heap_type`, `size` bytes, and `options`.
fn heap(device: &Device, heap_type: HeapType, size: u64, options: ResourceOptions) -> Heap {
    device
        .new_heap(&HeapDescriptor::new(heap_type, size, options))
        .unwrap()
}

/// The kinds a heap texture's size answer announces where it is not Metal's: more bytes or a
/// coarser alignment than Metal's modelled answer, which its device's image needs; a depth or
/// stencil texture whose driver keeps it GPU-optimized; a shape whose Metal answer the runtime does
/// not model.
const SIZE_NOTICES: [Degradation; 3] = [
    Degradation::HeapTextureLargerThanMetal,
    Degradation::OptimizedDepthStencilTexture,
    Degradation::UnmeasuredHeapTextureSize,
];

/// The answers for each probe shape, printed beside Metal's, each answered alike twice, a
/// power-of-two alignment and a size holding the texels, with `allowGPUOptimizedContents` off the
/// answer for the shape of `PixelFormatView` usage too: Metal's own on MoltenVK. On every device
/// each answer is Metal's, announcing nothing, or more bytes or a coarser alignment than Metal's
/// with one of [`SIZE_NOTICES`] announced for it (the device announces each occurrence). Then the
/// placement and automatic heaps' rules.
#[test]
fn heap_textures_answer_and_place_as_metal_s_do() {
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    {
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.announce_each = SIZE_NOTICES.into();
        })
        .unwrap();
        let device = &device;
        let announced =
            || -> BTreeSet<Degradation> { counts(&messages.take_notices()).into_keys().collect() };
        let moltenvk = device.properties().description.driver_id == Some(DriverId::MOLTENVK);
        let mut differ = Vec::new();
        let shapes = shapes();
        for shape in &shapes {
            let descriptor = shape.descriptor(ResourceOptions::private());
            let answer = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer,
                Err(error @ (Error::Unsupported(_) | Error::NotImplemented(_))) => {
                    eprintln!("HEAPTEX {}: refused {error:?}", shape.name);
                    let kinds = announced();
                    assert!(
                        kinds.is_empty(),
                        "{}: refused, announced {kinds:?}",
                        shape.name
                    );
                    continue;
                }
                Err(error) => panic!("{}: {error:?}", shape.name),
            };
            let kinds = announced();
            let again = device.heap_texture_size_and_align(&descriptor).unwrap();
            assert_eq!(answer, again, "{}: one answer", shape.name);
            assert_eq!(announced(), kinds, "{}: one answer's notices", shape.name);
            if !shape.optimized {
                let mut viewed = descriptor;
                viewed.allow_gpu_optimized_contents = true;
                viewed.usage |= TextureUsage::PIXEL_FORMAT_VIEW;
                let viewed = device.heap_texture_size_and_align(&viewed).unwrap();
                if is_depth_stencil(shape.pixel_format) {
                    assert!(
                        viewed.size >= answer.size && viewed.align >= answer.align,
                        "{}: a PixelFormatView texture's {viewed:?} holds {answer:?}",
                        shape.name
                    );
                } else {
                    assert_eq!(
                        answer, viewed,
                        "{}: a PixelFormatView texture's",
                        shape.name
                    );
                }
                // Its image is made, in a heap of the answered size and on the device.
                let options = ResourceOptions::private();
                let placement = heap(device, HeapType::Placement, answer.size, options);
                let placed = placement.new_texture_at_offset(&descriptor, 0).unwrap();
                let made = device.new_texture(&descriptor).unwrap();
                for texture in [&placed, &made] {
                    assert!(!texture.allow_gpu_optimized_contents(), "{}", shape.name);
                    assert_eq!(texture.usage(), shape.usage, "{}", shape.name);
                }
                // The view's answer announces only a size's kinds, and the textures' makings
                // the heap texture's route where the device's driver facts force one: an image
                // the driver may compress one of `ALIASES`; a depth or stencil texture's image
                // the driver lays apart from its memory a shadow of the heap's bytes
                // (`shadowed-texture`, as the heap depth lane measures on MoltenVK) or, where no
                // shadow serves it, `heap-depth-stencil-apart`.
                let facts = &device.properties().driver_facts;
                let apart = is_depth_stencil(shape.pixel_format)
                    && facts.contains(&DriverFact::DepthStencilImagesLieApart);
                let routes = facts
                    .contains(&DriverFact::CompressedImagesHideAliasedWrites)
                    .then_some(&ALIASES[..])
                    .into_iter()
                    .chain(apart.then_some(&DEPTH_STENCIL_ROUTES[..]))
                    .flatten();
                let allowed: BTreeSet<Degradation> =
                    SIZE_NOTICES.iter().chain(routes).copied().collect();
                let more = announced();
                assert!(
                    more.is_subset(&allowed),
                    "{}: announced {more:?}, beyond {allowed:?}",
                    shape.name
                );
            }
            assert!(answer.align.is_power_of_two(), "{}", shape.name);
            assert!(answer.size >= shape.texel_bytes(), "{}", shape.name);
            let SizeAndAlign { size, align, .. } = answer;
            eprintln!(
                "HEAPTEX {}: size {size} align {align}; Metal's size {} align {}",
                shape.name, shape.metal.0, shape.metal.1
            );
            // MoltenVK makes a `PixelFormatView` Metal texture of a color texture's image with
            // `allowGPUOptimizedContents` off, laid out as Metal lays out the texture. A depth or
            // stencil texture's it makes optimized (no Vulkan request reaches the property), so its
            // answer is Metal's for the texture with it on (`metal_optimized`).
            let expected = if shape.optimized || !is_depth_stencil(shape.pixel_format) {
                shape.metal
            } else {
                metal_optimized(&shapes, shape)
            };
            if (size, align) != expected {
                differ.push(shape.name);
            }
            // Metal's answer for the shape announces nothing; any other is past it, with its
            // reason announced.
            if (size, align) == shape.metal {
                assert!(
                    kinds.is_empty(),
                    "{}: Metal's answer, announced {kinds:?}",
                    shape.name
                );
            } else {
                assert!(
                    size >= shape.metal.0 && align >= shape.metal.1,
                    "{}: size {size} align {align} below Metal's {:?}",
                    shape.name,
                    shape.metal
                );
                assert!(
                    kinds.len() == 1 && SIZE_NOTICES.iter().any(|kind| kinds.contains(kind)),
                    "{}: size {size} align {align} past Metal's {:?}, announced {kinds:?}",
                    shape.name,
                    shape.metal
                );
                eprintln!(
                    "HEAPTEX-PAST {}: size {size} align {align}, Metal's {:?}: {kinds:?}",
                    shape.name, shape.metal
                );
            }
        }
        if moltenvk {
            assert!(
                differ.is_empty(),
                "MoltenVK answers otherwise than Metal: {differ:?}"
            );
        }
        placement_rules(device);
        automatic_rules(device);
    }
    assert_eq!(messages.take(), Vec::new(), "the validation layer reported");
}

/// A placement heap's textures: at every multiple of the alignment, refused off it and past the
/// end (Metal's nil), with other options (Metal's stop), and through the
/// automatic heap's call.
fn placement_rules(device: &Device) {
    const CALL: &str = "new_texture_at_offset";
    let options = ResourceOptions::private();
    let heap = heap(device, HeapType::Placement, 1 << 20, options);
    let descriptor = rgba8(64, options);
    let SizeAndAlign { size, align, .. } = device.heap_texture_size_and_align(&descriptor).unwrap();
    let last = (heap.size() - size) / align * align;
    let mut placed = Vec::new();
    for offset in [0, align, 3 * align, last] {
        let texture = heap.new_texture_at_offset(&descriptor, offset).unwrap();
        assert_eq!(texture.heap_offset(), offset);
        assert_eq!(texture.heap(), Some(&heap));
        assert_eq!(texture.storage_mode(), StorageMode::Private);
        placed.push(texture);
    }
    assert_eq!(heap.used_size(), 0, "a placement heap's used size is 0");
    if align > 1 {
        let offset = align / 2;
        assert_eq!(
            heap.new_texture_at_offset(&descriptor, offset).err(),
            Some(nil(CALL, Problem::Heap(HeapProblem::Offset { offset })))
        );
    }
    let past = last + align;
    assert_eq!(
        heap.new_texture_at_offset(&descriptor, past).err(),
        Some(nil(
            CALL,
            Problem::Heap(HeapProblem::PastEnd {
                offset: past,
                length: size,
                heap_size: heap.size(),
            })
        ))
    );
    let shared = rgba8(64, ResourceOptions::shared());
    assert_eq!(
        heap.new_texture_at_offset(&shared, 0).err(),
        Some(invalid(
            CALL,
            Problem::Heap(HeapProblem::StorageMode {
                heap: StorageMode::Private,
                resource: StorageMode::Shared,
            })
        ))
    );
    let combined = rgba8(
        64,
        ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::WriteCombined,
            HazardTrackingMode::Default,
        ),
    );
    assert_eq!(
        heap.new_texture_at_offset(&combined, 0).err(),
        Some(invalid(
            CALL,
            Problem::Heap(HeapProblem::CpuCacheMode {
                heap: CpuCacheMode::DefaultCache,
                resource: CpuCacheMode::WriteCombined,
            })
        ))
    );
    assert_eq!(
        heap.new_texture(&descriptor).err(),
        Some(invalid(
            "new_texture",
            Problem::Heap(HeapProblem::HeapType {
                heap_type: HeapType::Placement,
            })
        ))
    );
    // A placement heap's textures are aliasable from the start, as Metal's, and `make_aliasable`
    // changes nothing.
    assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
    assert_eq!(placed[0].make_aliasable(), Ok(()));
    assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
}

/// An automatic heap's textures: first fit at multiples of the alignment, `heap_offset` 0 as
/// Metal answers, `used_size` the sum of the answered sizes, and a range back when a texture is
/// made aliasable or goes.
fn automatic_rules(device: &Device) {
    let options = ResourceOptions::private();
    let heap = heap(device, HeapType::Automatic, 1 << 20, options);
    let small = rgba8(16, options);
    let large = rgba8(64, options);
    let small_size = device.heap_texture_size_and_align(&small).unwrap().size;
    let SizeAndAlign {
        size: large_size,
        align: large_align,
        ..
    } = device.heap_texture_size_and_align(&large).unwrap();
    let first = heap.new_texture(&small).unwrap();
    let second = heap.new_texture(&large).unwrap();
    assert_eq!(first.heap_offset(), 0, "an automatic heap's offset is 0");
    assert_eq!(second.heap_offset(), 0);
    assert_eq!(heap.used_size(), small_size + large_size);
    second.make_aliasable().unwrap();
    assert!(second.is_aliasable());
    assert_eq!(heap.used_size(), small_size);
    drop(first);
    assert_eq!(heap.used_size(), 0);
    assert_eq!(
        heap.new_texture_at_offset(&small, 0).err(),
        Some(invalid(
            "new_texture_at_offset",
            Problem::Heap(HeapProblem::HeapType {
                heap_type: HeapType::Automatic,
            })
        ))
    );
    // Outside a heap, the call changes nothing, as plain Metal.
    let not_in_heap = device.new_texture(&small).unwrap();
    assert_eq!(not_in_heap.make_aliasable(), Ok(()));
    assert!(!not_in_heap.is_aliasable());
    // A heap full of textures has no room for another: first fit at multiples of the alignment
    // puts one every `stride` bytes, the last one's bytes inside the heap (Metal's rule over the
    // device's answered size and alignment, which may leave more than a texture's bytes free).
    let mut filled = Vec::new();
    loop {
        match heap.new_texture(&large) {
            Ok(texture) => filled.push(texture),
            Err(Error::OutOfMemory(_)) => break,
            Err(other) => panic!("filling an automatic heap: {other:?}"),
        }
    }
    assert_eq!(heap.used_size(), filled.len() as u64 * large_size);
    let stride = large_size.next_multiple_of(large_align);
    assert_eq!(filled.len() as u64, (heap.size() - large_size) / stride + 1);
}

/// A `side`-square `RGBA8Unorm` texture's texels `(x, y, 0xAB, 0xCD)`, rows packed.
fn texel_pattern(side: u64) -> Vec<u8> {
    (0..side * side)
        .flat_map(|i| [(i % side) as u8, (i / side) as u8, 0xAB, 0xCD])
        .collect()
}

/// A Shared buffer holding `bytes`.
fn staged(device: &Device, bytes: &[u8]) -> Buffer {
    let buffer = device
        .new_buffer(bytes.len() as u64, ResourceOptions::shared())
        .unwrap();
    let contents = buffer.contents().unwrap();
    // SAFETY: the buffer holds `bytes.len()` bytes, and no command uses them yet.
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), contents.as_ptr(), bytes.len()) };
    buffer
}

/// The texels of a `side`-square texture, read back through a blit.
fn texels(queue: &CommandQueue, texture: &Texture, side: u64) -> Vec<u8> {
    let device = queue.device();
    let bytes = side * side * 4;
    let copy = device.new_buffer(bytes, ResourceOptions::shared()).unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &copy,
        0,
        side * 4,
        bytes,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    // SAFETY: the buffer holds `bytes` bytes the completed copy wrote.
    unsafe { std::slice::from_raw_parts(copy.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// The aliasing of a placement heap's texture of `descriptor` (`RGBA8Unorm`, square) at offset 0
/// and a buffer of its answered size over the same bytes, `what` naming it: texels
/// blitted into the texture's first level read back unchanged and change some of the buffer's
/// bytes, which this asserts; then bytes copied into the buffer, and the first level's texels read
/// back, in one command buffer ordered by the heap's tracking. Gives how many of those texels
/// the buffer's write left unchanged; `None`, printed, where the heap refuses the texture for its
/// tiling (`tiling_refusal`).
fn alias_round_trip(
    queue: &CommandQueue,
    heap: &Heap,
    descriptor: &TextureDescriptor,
    what: &str,
) -> Option<usize> {
    let device = queue.device();
    let (side, options) = (descriptor.width, descriptor.resource_options);
    let texture = match heap.new_texture_at_offset(descriptor, 0) {
        Ok(texture) => texture,
        Err(error) if tiling_refusal(&error).is_some() => {
            eprintln!("HEAPTEX alias {what}: refused {error:?}");
            return None;
        }
        Err(error) => panic!("{what}: {error:?}"),
    };
    let size = device.heap_texture_size_and_align(descriptor).unwrap().size;
    let buffer = heap.new_buffer_at_offset(size, options, 0).unwrap();
    gpu_fill_zero(queue, &buffer, size);
    // Texels into the texture: the buffer's bytes change.
    let pattern = texel_pattern(side);
    let source = staged(device, &pattern);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_texture(
        &source,
        0,
        side * 4,
        side * side * 4,
        Size::new(side, side, 1),
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    assert_eq!(texels(queue, &texture, side), pattern, "{what}");
    let bytes = gpu_read(queue, &buffer, 0..size);
    let changed = bytes.iter().filter(|&&byte| byte != 0).count();
    assert!(
        changed > 0,
        "{what}: the texture's write left the buffer zero"
    );
    // Bytes through the buffer, then the texture's texels, in one command buffer.
    let fill: Vec<u8> = (0..size as usize)
        .map(|i| (0x11 * (i % 4 + 1) + i / 4 % 7) as u8)
        .collect();
    let fill = staged(device, &fill);
    let read = device
        .new_buffer(side * side * 4, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&fill, 0, &buffer, 0, size)
        .unwrap();
    blit.end_encoding().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        &texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &read,
        0,
        side * 4,
        side * side * 4,
        BlitOption::None,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    let after = gpu_read(queue, &read, 0..side * side * 4);
    let unchanged = after
        .as_chunks::<4>()
        .0
        .iter()
        .zip(pattern.as_chunks::<4>().0.iter())
        .filter(|(after, before)| after == before)
        .count();
    eprintln!(
        "HEAPTEX alias {what}: size {size}, {changed} buffer bytes changed by the texture's \
         write, {unchanged} texels unchanged by the buffer's"
    );
    Some(unchanged)
}

/// A heap's buffer and texture over the same bytes alias them, as Metal's do: in
/// a tracked `Shared` and a tracked `Private` placement heap, for `RGBA8Unorm` textures of 16,
/// 64 and 1024 texels a side of storage, render-target and sampled-only usage, texels blitted into
/// the texture change the buffer's bytes and read back unchanged; bytes copied into the buffer then
/// change every texel the texture reads, the copy and the read encoded in one command buffer and
/// ordered by the heap's tracking.
///
/// Where the device's driver may keep a compressed image's texels apart from its memory
/// (`DriverFact::CompressedImagesHideAliasedWrites`: ANV, whose texture's write changed 234 of
/// the buffer's 1024 bytes and the buffer's write none of its 256 texels), the texture takes
/// host-transfer usage or a linear image: the device announces one of `heap-texture-host-transfer`
/// and `heap-texture-linear` once, never `heap-texture-hides-aliased-writes`, whose texels would
/// fail the test. Elsewhere none of the three is announced. `HEAPTEX alias` lines print the counts.
#[test]
fn a_buffer_write_shows_in_an_overlapping_texture() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Whether the driver hides such writes, and whether a texture was made in a heap.
    let (hides, made) = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let hides = device
            .properties()
            .driver_facts
            .contains(&DriverFact::CompressedImagesHideAliasedWrites);
        let mut made = false;
        let queue = device.new_command_queue().unwrap();
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = ResourceOptions::new(
                storage,
                CpuCacheMode::DefaultCache,
                HazardTrackingMode::Tracked,
            );
            // Storage usage often keeps a driver's image uncompressed, so render-target and
            // sampled-only usage, and a side past the small sizes a compressor may skip, are
            // measured too.
            let heap = heap(device, HeapType::Placement, 1 << 23, options);
            for side in [16, 64, 1024] {
                for usage in [
                    TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    TextureUsage::SHADER_READ,
                ] {
                    let mut descriptor = rgba8(side, options);
                    descriptor.usage = usage;
                    let what = format!("{storage:?} {side}x{side} {usage:?}");
                    let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what)
                    else {
                        continue;
                    };
                    made = true;
                    assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                }
            }
        }
        (hides, made)
    };
    let notices = counts(&messages.take_notices());
    let [transfer, linear, hidden] = ALIASES.map(|kind| notices.get(&kind).copied().unwrap_or(0));
    eprintln!(
        "HEAPTEX alias: {transfer} heap-texture-host-transfer, {linear} heap-texture-linear, \
         {hidden} heap-texture-hides-aliased-writes notices"
    );
    assert_eq!(hidden, 0);
    assert_eq!(transfer + linear, usize::from(hides && made));
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}

/// Under the default `HeapCompressionPolicy::Compressed`, where the device's driver may keep a
/// compressed image's texels apart from its memory
/// (`DriverFact::CompressedImagesHideAliasedWrites`: ANV), a heap's `RGBA8Unorm` texture that Metal
/// keeps losslessly compressed (render-target and sampled-only usage, 64 and 1024 texels a side)
/// takes the driver's compressed image, announced once (`heap-texture-compressed`); its texels the
/// buffer's write may leave alone, as Metal's compressed texture shows such a write otherwise than
/// as written: printed (`HEAPTEX compressed` lines), not asserted. One Metal keeps uncompressed
/// (storage usage, or 8 texels a side) takes the uncompressed image it takes under
/// `HeapCompressionPolicy::Uncompressed`, so the buffer's write changes every texel it reads
/// ([`alias_round_trip`]). Elsewhere no route is announced.
#[test]
fn a_heap_texture_metal_compresses_keeps_the_driver_s_compression_by_default() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let (hides, compressed_made, uncompressed_made) = {
        let device = &validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.policy.heap_compression = HeapCompressionPolicy::Compressed;
        })
        .unwrap();
        let hides = device
            .properties()
            .driver_facts
            .contains(&DriverFact::CompressedImagesHideAliasedWrites);
        let (mut compressed_made, mut uncompressed_made) = (false, false);
        let queue = device.new_command_queue().unwrap();
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let heap = heap(device, HeapType::Placement, 1 << 23, options);
            for (side, usage, compressed) in [
                (
                    64,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    true,
                ),
                (1024, TextureUsage::SHADER_READ, true),
                (
                    64,
                    TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE,
                    false,
                ),
                (
                    8,
                    TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET,
                    false,
                ),
            ] {
                let mut descriptor = rgba8(side, options);
                descriptor.usage = usage;
                let what = format!("{storage:?} {side}x{side} {usage:?}");
                let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what) else {
                    continue;
                };
                if compressed {
                    compressed_made = true;
                    eprintln!("HEAPTEX compressed {what}: {unchanged} texels unchanged");
                } else {
                    uncompressed_made = true;
                    assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                }
            }
        }
        (hides, compressed_made, uncompressed_made)
    };
    let notices = counts(&messages.take_notices());
    let count = |kind: Degradation| notices.get(&kind).copied().unwrap_or(0);
    let [transfer, linear, hidden] = ALIASES.map(count);
    let [compressed] = COMPRESSED.map(count);
    eprintln!(
        "HEAPTEX default: {compressed} heap-texture-compressed, {transfer} \
         heap-texture-host-transfer, {linear} heap-texture-linear, {hidden} \
         heap-texture-hides-aliased-writes notices"
    );
    assert_eq!(compressed, usize::from(hides && compressed_made));
    assert_eq!(hidden, 0);
    assert_eq!(transfer + linear, usize::from(hides && uncompressed_made));
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}

/// What a 2048x2048 `RGBA8Unorm` render target in a `Private` placement heap costs the GPU under
/// each `HeapCompressionPolicy`, without the validation layer: a pass that clears it and stores,
/// and a blit copying it into another such texture, each its own command buffer, medians and the
/// fastest of 40 after 8 untimed. On a driver whose compressed images hide aliased writes (ANV)
/// the two policies give the texture its compressed and its uncompressed image; elsewhere the
/// same image. Ignored by default: a measurement, not a check.
#[test]
#[ignore = "a measurement, not a check: run by name with --run-ignored all"]
fn heap_compression_costs() {
    const SIDE: u64 = 2048;
    for policy in [
        HeapCompressionPolicy::Compressed,
        HeapCompressionPolicy::Uncompressed,
    ] {
        let messages = Messages::default();
        let cache = ScratchCache::new().unwrap();
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options.layers.clear();
            options.policy.heap_compression = policy;
        })
        .unwrap();
        let options = ResourceOptions::private();
        let mut descriptor = rgba8(SIDE, options);
        descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let size = device
            .heap_texture_size_and_align(&descriptor)
            .unwrap()
            .size;
        let heap = heap(&device, HeapType::Placement, 2 * size, options);
        let target = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let copy = heap.new_texture_at_offset(&descriptor, size).unwrap();
        let notices = counts(&messages.take_notices());
        let queue = device.new_command_queue().unwrap();
        let time = |encode: &dyn Fn(&metal2vulkan_runtime::CommandBuffer)| {
            let mut gpu = Vec::new();
            for run in 0..48 {
                let command_buffer = queue.command_buffer().unwrap();
                encode(&command_buffer);
                command_buffer.commit().unwrap();
                command_buffer
                    .wait_until_completed_for(super::WAIT)
                    .unwrap();
                assert_eq!(command_buffer.error(), None);
                let start = command_buffer.gpu_start_time().unwrap();
                let end = command_buffer.gpu_end_time().unwrap();
                if run >= 8 {
                    gpu.push((end - start) * 1e6);
                }
            }
            gpu.sort_by(f64::total_cmp);
            (gpu[gpu.len() / 2], gpu[0])
        };
        let clear = time(&|command_buffer| {
            let mut pass = RenderPassDescriptor::new();
            let color = &mut pass.color_attachments[0];
            color.texture = Some(target.clone());
            color.load_action = LoadAction::Clear;
            color.store_action = StoreAction::Store;
            let encoder = command_buffer.render_command_encoder(&pass).unwrap();
            encoder.end_encoding().unwrap();
        });
        let blit = time(&|command_buffer| {
            let mut blit = command_buffer.blit_command_encoder().unwrap();
            blit.copy_texture(&target, &copy).unwrap();
            blit.end_encoding().unwrap();
        });
        eprintln!(
            "HEAP-COMPRESSION on {} ({policy:?}, notices {notices:?}): clear median {:.1} us, \
             fastest {:.1} us; copy median {:.1} us, fastest {:.1} us",
            device.name(),
            clear.0,
            clear.1,
            blit.0,
            blit.1
        );
    }
}

/// An adopted device whose consumer enabled `hostImageCopy` and says so
/// (`RawDevice::host_image_copy`) gives its heaps' colour textures Metal's aliasing
/// ([`alias_round_trip`]), a mipmapped one included: on ANV, whose compressed images hide writes
/// through an alias, their images take host-transfer usage, as on a device the runtime creates. The
/// same device adopted without saying so takes a linear image where one serves, else the optimal
/// image, the recorded near-equivalent (`heap-texture-hides-aliased-writes`), whose texels the
/// buffer's write may leave alone: printed (`HEAPTEX adopted` lines), not asserted. An adopted
/// device announces on stderr, so the routes are not counted here.
#[test]
fn an_adopted_device_stating_host_image_copy_aliases_its_heap_textures() {
    let cache = ScratchCache::new().unwrap();
    for states in [true, false] {
        let adopted = if states {
            Adopted::open_host_copying()
        } else {
            Adopted::open(false)
        }
        .unwrap();
        let copies = adopted.copies_host_images();
        {
            let device = adopted.runtime(cache.path());
            let hides = device
                .properties()
                .driver_facts
                .contains(&DriverFact::CompressedImagesHideAliasedWrites);
            if states && hides {
                assert!(
                    copies,
                    "{}: compressed images hide aliased writes, and the device has no \
                     hostImageCopy",
                    device.name()
                );
            }
            let queue = device.new_command_queue().unwrap();
            for storage in [StorageMode::Shared, StorageMode::Private] {
                let options = tracked(storage);
                let heap = heap(&device, HeapType::Placement, 1 << 20, options);
                for mipmapped in [false, true] {
                    let mut descriptor =
                        TextureDescriptor::texture_2d(PixelFormat::RGBA8Unorm, 16, 16, mipmapped);
                    descriptor.resource_options = options;
                    descriptor.usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
                    let what = format!(
                        "adopted {} (hostImageCopy enabled: {copies}) {storage:?} 16x16 of {} \
                         levels",
                        device.name(),
                        descriptor.mipmap_level_count
                    );
                    let Some(unchanged) = alias_round_trip(&queue, &heap, &descriptor, &what)
                    else {
                        continue;
                    };
                    println!("HEAPTEX {what}: {unchanged} texels unchanged by the buffer's write");
                    if copies || !hides {
                        assert_eq!(unchanged, 0, "{what}: texels the buffer's write left alone");
                    }
                }
            }
        }
        let reported = adopted.close();
        assert!(
            reported.is_empty(),
            "the validation layer reported: {reported:#?}"
        );
    }
}

/// The kind a heap's depth or stencil texture laid apart from the heap announces.
pub(super) const APART: [Degradation; 1] = [Degradation::HeapDepthStencilApart];

/// The kinds a heap's depth or stencil texture announces where the driver lays such images apart
/// from their memory (`DriverFact::DepthStencilImagesLieApart`): its image a shadow of the heap's
/// bytes (one sample, its texels inside its bytes), else laid apart.
const DEPTH_STENCIL_ROUTES: [Degradation; 2] = [
    Degradation::ShadowedTexture,
    Degradation::HeapDepthStencilApart,
];

/// The kind a depth or stencil texture without GPU-optimized contents announces where the driver
/// keeps its layout optimized.
pub(super) const OPTIMIZED: [Degradation; 1] = [Degradation::OptimizedDepthStencilTexture];

/// A depth texture without GPU-optimized contents, sized, made in a placement heap and on the
/// device, announces `optimized-depth-stencil-texture` once where the driver keeps depth and
/// stencil images GPU-optimized (`DriverFact::DepthStencilImagesStayGpuOptimized`: MoltenVK on
/// Apple's GPUs), its answer then Metal's with the property on; elsewhere nothing. Made on the
/// device first, it announces nothing there: nothing shows a device's own texture's layout. A
/// color texture without them, and a depth texture with them, announce nothing anywhere.
#[test]
fn a_depth_texture_without_optimized_contents_is_announced_where_it_stays_optimized() {
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    let optimized = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let optimized = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesStayGpuOptimized);
        let options = ResourceOptions::private();
        let target = TextureUsage::SHADER_READ | TextureUsage::RENDER_TARGET;
        let make = |pixel_format, allow| {
            let mut descriptor = TextureDescriptor::texture_2d(pixel_format, 64, 64, false);
            descriptor.usage = target;
            descriptor.resource_options = options;
            descriptor.allow_gpu_optimized_contents = allow;
            let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
            let placement = heap(device, HeapType::Placement, answer.size, options);
            placement.new_texture_at_offset(&descriptor, 0).unwrap();
            device.new_texture(&descriptor).unwrap();
            answer
        };
        make(PixelFormat::RGBA8Unorm, false);
        let with = make(PixelFormat::Depth32Float, true);
        assert!(!counts(&messages.take_notices()).contains_key(&OPTIMIZED[0]));
        let mut own = TextureDescriptor::texture_2d(PixelFormat::Depth32Float, 64, 64, false);
        own.usage = target;
        own.resource_options = options;
        own.allow_gpu_optimized_contents = false;
        device.new_texture(&own).unwrap();
        assert!(
            !counts(&messages.take_notices()).contains_key(&OPTIMIZED[0]),
            "a device's own depth texture without optimized contents announced"
        );
        let without = make(PixelFormat::Depth32Float, false);
        eprintln!(
            "HEAPTEX optimized depth: {} bytes with, {} without (fact {optimized})",
            with.size, without.size
        );
        if optimized {
            assert_eq!(with, without);
        }
        optimized
    };
    let notices = counts(&messages.take_notices());
    let announced = notices.get(&OPTIMIZED[0]).copied().unwrap_or(0);
    assert_eq!(announced, usize::from(optimized));
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}

/// The kinds a heap's colour texture announces on a driver whose compressed images hide writes
/// through an alias, by the route its image takes: host-transfer usage, a linear image, or the
/// optimal image, a near-equivalent.
pub(super) const ALIASES: [Degradation; 3] = [
    Degradation::HeapTextureHostTransfer,
    Degradation::HeapTextureLinear,
    Degradation::HeapTextureHidesAliasedWrites,
];

/// The kind a heap's colour texture Metal compresses announces on such a driver, where it keeps
/// the driver's compression under the default.
pub(super) const COMPRESSED: [Degradation; 1] = [Degradation::HeapTextureCompressed];

/// The usages the `Shared` heap sweep makes each format's texture with.
const SWEPT_USAGES: [(&str, TextureUsage); 4] = [
    ("read", TextureUsage::SHADER_READ),
    (
        "view",
        TextureUsage::SHADER_READ.union(TextureUsage::PIXEL_FORMAT_VIEW),
    ),
    (
        "write",
        TextureUsage::SHADER_READ.union(TextureUsage::SHADER_WRITE),
    ),
    (
        "render",
        TextureUsage::SHADER_READ.union(TextureUsage::RENDER_TARGET),
    ),
];

/// Whether `error` refuses a heap texture for what the device's linear images lack.
fn tiling_refusal(error: &Error) -> Option<LinearNeed> {
    match error {
        Error::Unsupported(u) => match u.feature {
            Feature::HeapTextureTiling { missing } => Some(missing),
            _ => None,
        },
        _ => None,
    }
}

/// Every pixel format's 16x16 2D texture of a `Shared` placement heap, read, viewed as other pixel
/// formats, written or rendered to, is made or refused typed, the query and the call alike: where
/// the device's optimal images take no memory a `Shared` heap lies in (the RTX 3090's), the texture
/// is a linear image in the heap's memory, refused `HeapTextureTiling` only for what the device's
/// linear images lack, which a `Private` heap's texture never is. Each uncompressed color format's
/// texture made that is not floating point takes bytes through `replace_region` and gives them back
/// through `get_bytes`, and again through a second `get_bytes`: a copy the driver makes after the
/// first read's submission must not change them (MoltenVK 1.4.1 copied a texture into its heap's
/// memory, which is the texture's own, once a command buffer with a barrier asking for host reads
/// completed; `image_barrier.rs` in the runtime). A texture of several levels, a cube, a 3D texture
/// and an array are made or refused alike. `HEAPTEX shared` lines print each refusal for linear
/// tiling and the counts, the second reads that differ among them, and the submissions on the
/// device's staging queue (a copy engine's, where texture copies and first moves run on a family
/// of transfers alone, `device/staging.rs` in the runtime). No colour texture is refused
/// `NotImplemented(HeapTextureMemory)`: one whose image needs extended usage, such as a written
/// sRGB texture, is checked format by format. `Unspecialized` takes Metal's answers: 0 bytes from
/// the query, at 128 for a 2D texture and the device's level's alignment for a cube, a 3D texture
/// or an array (`stacked_nil_alignment`), and no texture from the call
/// (`TextureProblem::Unspecialized`, Metal's nil; an M4 under macOS 27.0.1, every
/// usage). A format Metal makes no texture of from a descriptor alone, such as a two-plane YCbCr
/// one, takes Metal's answers too: 0 bytes from the query, at 16 for a two-plane YCbCr 2D texture
/// and 128 for another, and nil from the call (`Error::Nil`, `TextureProblem::NoTextureOfFormat`).
#[test]
fn shared_heap_textures_are_made_or_refused_for_linear_tiling() {
    on_device(|device| {
        let options = ResourceOptions::new(
            StorageMode::Shared,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Tracked,
        );
        let private = ResourceOptions::new(
            StorageMode::Private,
            CpuCacheMode::DefaultCache,
            HazardTrackingMode::Tracked,
        );
        let heap = heap(device, HeapType::Placement, 4 << 20, options);
        let unspecialized = nil(
            "new_texture_at_offset",
            Problem::Texture(TextureProblem::Unspecialized),
        );
        let no_texture = nil(
            "new_texture_at_offset",
            Problem::Texture(TextureProblem::NoTextureOfFormat),
        );
        let staged = device.work_counts().staging_submissions;
        let (mut made, mut round_trips, mut refused) = (0usize, 0usize, Vec::new());
        let mut changed = Vec::new();
        let mut changed_again = Vec::new();
        let mut gaps = Vec::new();
        let mut check = |what: String, mut descriptor: TextureDescriptor| {
            descriptor.resource_options = options;
            let query = device.heap_texture_size_and_align(&descriptor);
            let call = heap.new_texture_at_offset(&descriptor, 0);
            match (query, call) {
                (Ok(_), Ok(texture)) => {
                    made += 1;
                    let Some(bytes) = texel_bytes(descriptor.pixel_format) else {
                        return;
                    };
                    if descriptor.texture_type != TextureType::Type2D
                        || floating(descriptor.pixel_format)
                    {
                        return;
                    }
                    let row = 16 * bytes;
                    let region = Region::new_2d(0, 0, 16, 16);
                    let src = pattern(row * 16, descriptor.pixel_format.as_raw() as u8);
                    texture
                        .replace_region(region, 0, 0, &src, row as u64, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let mut back = vec![0xeeu8; row * 16];
                    texture
                        .get_bytes(&mut back, row as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    let mut again = vec![0xeeu8; row * 16];
                    texture
                        .get_bytes(&mut again, row as u64, 0, region, 0, 0)
                        .unwrap_or_else(|e| panic!("{what}: {e:?}"));
                    if again != src {
                        changed_again.push(what.clone());
                    }
                    if back != src {
                        changed.push(what);
                    }
                    round_trips += 1;
                }
                // Metal's nil, and its query's 0 bytes at the type's alignment.
                (Ok(answer), Err(error)) if error == unspecialized => {
                    let align = match descriptor.texture_type {
                        TextureType::Type1D | TextureType::Type2D => 128,
                        TextureType::TextureBuffer => 16,
                        _ => stacked_nil_alignment(device),
                    };
                    assert_eq!(
                        (answer.size, answer.align, descriptor.pixel_format),
                        (0, align, PixelFormat::Unspecialized),
                        "{what}"
                    )
                }
                // Metal's nil for a format it makes no texture of from a descriptor alone, and
                // its query's 0 bytes at 16 for a two-plane YCbCr one, 128 for another.
                (Ok(answer), Err(error)) if error == no_texture => {
                    let pf = descriptor.pixel_format;
                    assert!(no_texture_alone(pf), "{what}: {error:?}");
                    let align = if metal2vulkan::metal::format::ycbcr_format(pf).is_some() {
                        16
                    } else {
                        128
                    };
                    assert_eq!((answer.size, answer.align), (0, align), "{what}")
                }
                (Ok(answer), Err(error)) => panic!("{what}: answered {answer:?}, made {error:?}"),
                (Err(error), Ok(_)) => panic!("{what}: refused {error:?} by the query alone"),
                (Err(query), Err(call)) => {
                    let colour = !is_depth_stencil(descriptor.pixel_format);
                    if colour && call == Error::NotImplemented(RuntimeGap::HeapTextureMemory) {
                        gaps.push(format!("{what}: {call:?}"));
                    }
                    let Some(missing) = tiling_refusal(&query) else {
                        return;
                    };
                    assert_eq!(query, call, "{what}");
                    let mut privately = descriptor;
                    privately.resource_options = private;
                    let answer = device.heap_texture_size_and_align(&privately);
                    assert!(
                        answer.as_ref().err().and_then(tiling_refusal).is_none(),
                        "{what}: a Private heap's texture refused {answer:?}"
                    );
                    refused.push(format!("{what}: {missing:?}"));
                }
            }
        };
        for &pf in PixelFormat::ALL {
            for (label, usage) in SWEPT_USAGES {
                let mut descriptor = TextureDescriptor::texture_2d(pf, 16, 16, false);
                descriptor.usage = usage;
                check(format!("{pf:?} {label}"), descriptor);
            }
        }
        let rgba8 = PixelFormat::RGBA8Unorm;
        let mut mipmapped = TextureDescriptor::texture_2d(rgba8, 16, 16, true);
        mipmapped.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm with 5 levels".to_owned(), mipmapped);
        let mut cube = TextureDescriptor::texture_cube(rgba8, 16, false);
        cube.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm cube".to_owned(), cube);
        let mut volume = TextureDescriptor::texture_2d(rgba8, 16, 16, false);
        volume.texture_type = TextureType::Type3D;
        volume.depth = 4;
        volume.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm 3D".to_owned(), volume);
        let mut array = TextureDescriptor::texture_2d(rgba8, 16, 16, false);
        array.texture_type = TextureType::Type2DArray;
        array.array_length = 4;
        array.usage = TextureUsage::SHADER_READ;
        check("RGBA8Unorm array of 4".to_owned(), array);
        for line in &refused {
            eprintln!("HEAPTEX shared refused {line}");
        }
        eprintln!(
            "HEAPTEX shared on {}: {made} made ({round_trips} round trips, {} second reads \
             differ, {} staging queue submissions), {} refused for linear tiling",
            device.name(),
            changed_again.len(),
            device.work_counts().staging_submissions - staged,
            refused.len()
        );
        assert!(made > 0, "no Shared heap texture was made");
        assert!(
            changed.is_empty(),
            "bytes changed by the round trip: {changed:?}"
        );
        assert!(
            changed_again.is_empty(),
            "bytes changed between the round trip and a second read: {changed_again:?}"
        );
        assert!(
            gaps.is_empty(),
            "colour textures refused as gaps: {gaps:#?}"
        );
    });
}

/// Whether `pf`'s texels are floating point, whose bytes a round trip need not keep: a NaN's
/// payload is the device's to canonicalize (MoltenVK's `RG11B10Float` in a heap does, at
/// `8b98df40`).
fn floating(pf: PixelFormat) -> bool {
    metal2vulkan::metal::format::direct_format(pf)
        .and_then(|format| format.block())
        .and_then(|block| block.color)
        .is_some_and(|color| matches!(color, NumericFormat::Sfloat | NumericFormat::Ufloat))
}

/// Zero `size` bytes of `buffer` with a blit.
fn gpu_fill_zero(queue: &CommandQueue, buffer: &Buffer, size: u64) {
    super::host_memory::gpu_fill(queue, buffer, 0..size, 0);
}

/// Reads four texture buffers of 256 texels each into its output buffer.
const FOUR_TEXEL_READS: &str = "half-conversion-by-texture-buffer-reads";

/// Writes eight `rgba8_uint` texels through a texture buffer (texture 1, its output).
const TEXEL_WRITER: &str = "write-four-bytes-into-a-storage-texel-buffer";

/// Tracked options of `storage`, a heap's and its resources'.
fn tracked(storage: StorageMode) -> ResourceOptions {
    ResourceOptions::new(
        storage,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Tracked,
    )
}

/// A texture buffer of a heap takes Metal's answer, its row of texels at a multiple of 16 bytes (a
/// Metal heap probe, an M3 Max and an M4 alike, plain and validated): a placement heap places it at
/// every multiple of 16 with its bytes inside the heap, and refuses an offset off it and bytes past
/// the end as Metal's validation stops them (`Invalid`); an automatic heap places it first fit,
/// answers `heap_offset` 0, counts its row in `used_size` (400 and 400 and a 16-byte buffer, 816,
/// as Metal counted) and takes the range back when it is made aliasable or goes. The texture
/// answers its heap, in a `Shared` and a `Private` heap.
#[test]
fn heap_texture_buffers_answer_and_place_as_metal_s_do() {
    const CALL: &str = "new_texture_at_offset";
    on_device(|device| {
        let usage = TextureUsage::SHADER_READ | TextureUsage::SHADER_WRITE;
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let formats = [
                (PixelFormat::R8Unorm, 1),
                (PixelFormat::RGBA8Unorm, 4),
                (PixelFormat::R32Uint, 4),
                (PixelFormat::RGBA32Float, 16),
            ];
            for (pixel_format, texel) in formats {
                for width in [1, 3, 100, 4096] {
                    let descriptor =
                        TextureDescriptor::texture_buffer(pixel_format, width, options, usage);
                    let answer = device.heap_texture_size_and_align(&descriptor).unwrap();
                    assert_eq!(
                        (answer.size, answer.align),
                        (width * texel, 16),
                        "{storage:?} {pixel_format:?} x {width}"
                    );
                }
            }
            let placement = heap(device, HeapType::Placement, 1 << 16, options);
            let descriptor =
                TextureDescriptor::texture_buffer(PixelFormat::R32Uint, 64, options, usage);
            let size = 256;
            let last = placement.size() - size;
            let mut placed = Vec::new();
            for offset in [0, 16, 64, 256, 4096 + 16, last] {
                let texture = placement
                    .new_texture_at_offset(&descriptor, offset)
                    .unwrap();
                assert_eq!(texture.heap_offset(), offset);
                assert_eq!(texture.heap(), Some(&placement));
                assert_eq!(texture.texture_type(), TextureType::TextureBuffer);
                assert_eq!(texture.storage_mode(), storage);
                placed.push(texture);
            }
            assert_eq!(placement.used_size(), 0);
            assert_eq!(
                placement.new_texture_at_offset(&descriptor, 4).err(),
                Some(nil(CALL, Problem::Heap(HeapProblem::Offset { offset: 4 })))
            );
            let past = last + 16;
            assert_eq!(
                placement.new_texture_at_offset(&descriptor, past).err(),
                Some(nil(
                    CALL,
                    Problem::Heap(HeapProblem::PastEnd {
                        offset: past,
                        length: size,
                        heap_size: placement.size(),
                    })
                ))
            );
            assert!(placed[0].is_aliasable() && placed[1].is_aliasable());
            placed[0].make_aliasable().unwrap();
            assert!(placed[0].is_aliasable() && placed[1].is_aliasable());

            let automatic = heap(device, HeapType::Automatic, 1 << 20, options);
            let descriptor =
                TextureDescriptor::texture_buffer(PixelFormat::RGBA8Unorm, 100, options, usage);
            let first = automatic.new_texture(&descriptor).unwrap();
            assert_eq!(first.heap_offset(), 0, "an automatic heap's offset is 0");
            assert_eq!(first.heap(), Some(&automatic));
            assert_eq!(automatic.used_size(), 400);
            let second = automatic.new_texture(&descriptor).unwrap();
            let buffer = automatic.new_buffer(16, options).unwrap();
            assert_eq!(automatic.used_size(), 816);
            first.make_aliasable().unwrap();
            assert_eq!(automatic.used_size(), 416);
            let third = automatic.new_texture(&descriptor).unwrap();
            assert_eq!(automatic.used_size(), 816);
            drop((first, second, third, buffer));
            assert_eq!(automatic.used_size(), 0);
        }
    });
}

/// Write `bytes` into `buffer` from its first byte with a blit from a staged copy.
fn gpu_write(queue: &CommandQueue, buffer: &Buffer, bytes: &[u8]) {
    let source = staged(queue.device(), bytes);
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_buffer_to_buffer(&source, 0, buffer, 0, bytes.len() as u64)
        .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
}

/// `case`'s one dispatch over `resources`, and its buffers after it.
fn dispatch_case(
    device: &Device,
    case: &PublicCase,
    resources: &CaseResources,
) -> BTreeMap<u32, Buffer> {
    let buffers = bound_buffers(device, &case.compute, resources).unwrap();
    let pipeline = pipeline_state(device, case).unwrap();
    let queue = device.new_command_queue().unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut encoder = command_buffer.compute_command_encoder().unwrap();
    encode_case(&mut encoder, &pipeline, &case.compute, &buffers, resources).unwrap();
    encoder.end_encoding().unwrap();
    run(&command_buffer);
    buffers
}

/// `case`'s texture at `index`, a texture buffer of `heap` at `offset`.
fn heap_texture_buffer(case: &PublicCase, index: u32, heap: &Heap, offset: u64) -> Texture {
    let literal = &case.compute.textures[&index];
    let mut descriptor = texture_descriptor(literal, case.texture_usage[&index]).unwrap();
    descriptor.resource_options = heap.resource_options();
    heap.new_texture_at_offset(&descriptor, offset).unwrap()
}

/// A texture buffer of a heap is the heap's bytes from its offset, in order (a Metal heap probe on
/// the M3 Max and an M4: a kernel read what a buffer over the bytes held, and its writes showed in
/// the buffer there and nowhere else): the four texture buffers `FOUR_TEXEL_READS` reads, packed at
/// 16-byte steps of a placement heap whose bytes a buffer over the heap wrote, give its own run's
/// output; and the texture buffer `TEXEL_WRITER` writes, at byte 48, writes there what its own run
/// writes and leaves the bytes around it; in a `Shared` and a `Private` heap.
#[test]
fn a_heap_texture_buffer_is_the_heap_s_bytes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let reader = public_case(FOUR_TEXEL_READS).unwrap();
        let read = run_case(device, &reader).unwrap().output;
        let writer = public_case(TEXEL_WRITER).unwrap();
        let written = run_case(device, &writer).unwrap().output;
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let placement = heap(device, HeapType::Placement, 1 << 16, options);
            let bytes = placement
                .new_buffer_at_offset(placement.size(), options, 0)
                .unwrap();
            let mut image = vec![0x5a_u8; placement.size() as usize];
            let mut at = 16;
            let mut placed = Vec::new();
            for (&index, literal) in &reader.compute.textures {
                let length = literal.bytes.len();
                image[at..at + length].copy_from_slice(&literal.bytes);
                placed.push((index, at as u64));
                at = (at + length).next_multiple_of(16);
            }
            gpu_write(&queue, &bytes, &image);
            let mut resources = CaseResources::new(device, &reader).unwrap();
            for &(index, offset) in &placed {
                let texture = heap_texture_buffer(&reader, index, &placement, offset);
                resources.textures.insert(index, texture);
            }
            let buffers = dispatch_case(device, &reader, &resources);
            let ComputeOutput::Buffer {
                binding,
                offset,
                length,
            } = reader.compute.output
            else {
                panic!("{FOUR_TEXEL_READS}: a buffer output");
            };
            let output = read_buffer(&buffers[&binding], offset, length).unwrap();
            assert_eq!(output, read, "{storage:?}: the texels at {placed:?}");

            // The writer's texture buffer at byte 48, between bytes of 0x5a.
            let literal = &writer.compute.textures[&1];
            let (start, end) = (48, 48 + literal.bytes.len());
            let mut image = vec![0x5a_u8; 4096];
            image[start..end].copy_from_slice(&literal.bytes);
            gpu_write(&queue, &bytes, &image);
            let mut resources = CaseResources::new(device, &writer).unwrap();
            let texture = heap_texture_buffer(&writer, 1, &placement, start as u64);
            resources.textures.insert(1, texture);
            dispatch_case(device, &writer, &resources);
            let after = gpu_read(&queue, &bytes, 0..4096);
            assert_eq!(
                after[start..end],
                written[..],
                "{storage:?}: the texels written"
            );
            let around = after[..start].iter().chain(&after[end..]);
            assert!(
                around.copied().all(|byte| byte == 0x5a),
                "{storage:?}: a byte around the texture buffer changed"
            );
        }
    });
}

/// Two `Depth32Float` textures a shader reads, side by side in a placement heap, keep their texels
/// (a Metal heap probe, an M3 Max and an M4 alike: each read back as uploaded, then a buffer filled
/// over the first one's answered bytes changed every texel of it and none of the second's). Where
/// the device holds such a texture with a depth twin (RADV, ANV and unmeasured drivers) the
/// answered size takes the twin's bytes too, so the second texture, uploaded before the first,
/// keeps its texels when the first's upload writes its twin; the comparisons of both compare as a
/// texture's outside a heap does; in a `Shared` and a `Private` heap. `HEAPTEX depth` lines print
/// the answer and the device's twin notices.
///
/// Where the device's driver lays depth and stencil images apart from their memory
/// (`DriverFact::DepthStencilImagesLieApart`: MoltenVK, which makes such an image's Metal texture
/// from the device, never from the memory's heap; an M4, MoltenVK 1.4.1, where
/// the fill changed none), each texture's image is a shadow of the heap's bytes: the fill changes
/// every texel of the first, Metal's answer, and the device announces `shadowed-texture` once,
/// never `heap-depth-stencil-apart`. Where no memory a `Shared` heap lies in takes the device's
/// optimal depth image and no linear one serves (the RTX 3090), the `Shared` heap's texture is a
/// shadow of the heap's bytes too: the fill changes every texel of the first, announced
/// `shadowed-texture` once. Elsewhere neither is announced.
#[test]
fn heap_depth_textures_keep_their_bytes_with_their_twins() {
    use super::depth_compares::{pipeline, run as compare, SIDE, VALUES};
    let messages = Messages::default();
    let cache = ScratchCache::new().unwrap();
    // Whether the driver lays the textures apart, whether one was made in a heap, and whether
    // the fill changed every texel of the first in each storage made.
    let (apart, made, all_changed) = {
        let device = &validated_device(TEST_DEVICE, &messages, cache.path()).unwrap();
        let apart = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let mut made = false;
        let mut all_changed = Vec::new();
        let queue = device.new_command_queue().unwrap();
        let side = u64::from(SIDE);
        let first: Vec<u8> = VALUES.iter().flat_map(|v| v.to_le_bytes()).collect();
        let second: Vec<u8> = VALUES.iter().rev().flat_map(|v| v.to_le_bytes()).collect();
        // A sample_compare at each texel's centre against references in and outside [0, 1].
        let step = 1.0 / SIDE as f32;
        let jobs: Vec<[f32; 4]> = (0..SIDE * SIDE)
            .flat_map(|i| {
                let (x, y) = ((i % SIDE) as f32, (i / SIDE) as f32);
                [-0.5, 0.25, 0.5, 0.75, 1.5, 1000.0]
                    .map(|r| [(x + 0.5) * step, (y + 0.5) * step, r, 0.0])
            })
            .collect();
        let state = pipeline(device, "msf_cmp_sample2d", false);
        let plain = |bytes: &[u8]| {
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
            descriptor.resource_options = ResourceOptions::private();
            let texture = device.new_texture(&descriptor).unwrap();
            upload(&queue, &[(&texture, bytes)], side);
            compare(device, (&texture, None), &state, &jobs, 1)
        };
        let (first_compares, second_compares) = (plain(&first), plain(&second));
        for storage in [StorageMode::Shared, StorageMode::Private] {
            let options = tracked(storage);
            let mut descriptor =
                TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::SHADER_READ;
            let what = format!("{storage:?}");
            let answer = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer,
                Err(error) if tiling_refusal(&error).is_some() => {
                    eprintln!("HEAPTEX depth {what}: refused {error:?}");
                    continue;
                }
                Err(error) => panic!("{what}: {error:?}"),
            };
            let SizeAndAlign { size, align, .. } = answer;
            let placement = heap(device, HeapType::Placement, 1 << 20, options);
            let next = size.next_multiple_of(align);
            let a = placement.new_texture_at_offset(&descriptor, 0).unwrap();
            let b = placement.new_texture_at_offset(&descriptor, next).unwrap();
            made = true;
            // The second first: the first's upload writes its twin, if it has one, after.
            upload(&queue, &[(&b, &second), (&a, &first)], side);
            assert_eq!(texels(&queue, &a, side), first, "{what}: the first");
            assert_eq!(texels(&queue, &b, side), second, "{what}: the second");
            let compared = compare(device, (&a, None), &state, &jobs, 1);
            assert_eq!(bits(&compared), bits(&first_compares), "{what}: the first");
            let compared = compare(device, (&b, None), &state, &jobs, 1);
            assert_eq!(
                bits(&compared),
                bits(&second_compares),
                "{what}: the second"
            );
            // A buffer over the first's bytes, filled: the second keeps its texels.
            let over = placement.new_buffer_at_offset(size, options, 0).unwrap();
            gpu_fill(&queue, &over, 0..size, 0x3e);
            assert_eq!(texels(&queue, &b, side), second, "{what}: after the fill");
            let compared = compare(device, (&b, None), &state, &jobs, 1);
            assert_eq!(
                bits(&compared),
                bits(&second_compares),
                "{what}: after the fill"
            );
            let changed = texels(&queue, &a, side)
                .as_chunks::<4>()
                .0
                .iter()
                .zip(first.as_chunks::<4>().0.iter())
                .filter(|(after, before)| after != before)
                .count();
            eprintln!(
                "HEAPTEX depth {what}: size {size} align {align}; the fill changed {changed} of \
                 the first's {} texels",
                side * side
            );
            all_changed.push((storage, changed == (side * side) as usize));
            if apart {
                assert_eq!(
                    changed,
                    (side * side) as usize,
                    "{what}: a shadow of the heap's bytes kept texels the fill covered"
                );
            } else {
                assert!(
                    changed > 0,
                    "{what}: the fill over the first changed none of its texels"
                );
            }
        }
        (apart, made, all_changed)
    };
    let notices = counts(&messages.take_notices());
    let count = |kind| notices.get(&kind).copied().unwrap_or(0);
    let twins = count(Degradation::DepthTwin);
    let aparts = count(APART[0]);
    let shadowed = count(Degradation::ShadowedTexture);
    eprintln!(
        "HEAPTEX depth: {twins} depth-twin notices, {aparts} heap-depth-stencil-apart, \
         {shadowed} shadowed-texture"
    );
    assert_eq!(aparts, 0);
    // A shadow of a `Shared` heap's bytes where its memory takes no optimal depth image: every
    // texel the fill covered changed, as in a shadow where the driver lays the image apart.
    let shared_shadow =
        !apart && shadowed == 1 && all_changed.contains(&(StorageMode::Shared, true));
    if shared_shadow {
        eprintln!("HEAPTEX depth: the Shared heap's texture is a shadow of the heap's bytes");
    }
    assert_eq!(shadowed, usize::from((apart && made) || shared_shadow));
    let failing = messages.take();
    assert!(
        failing.is_empty(),
        "the validation layer reported: {failing:#?}"
    );
}

/// One aspect of a `side`-square depth or stencil texture, read back through a blit with
/// `option`: `texel` bytes each.
fn aspect_texels(
    queue: &CommandQueue,
    texture: &Texture,
    side: u64,
    texel: u64,
    option: BlitOption,
) -> Vec<u8> {
    let bytes = side * side * texel;
    let copy = queue
        .device()
        .new_buffer(bytes, ResourceOptions::shared())
        .unwrap();
    let command_buffer = queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    blit.copy_from_texture_to_buffer(
        texture,
        0,
        0,
        Origin::new(0, 0, 0),
        Size::new(side, side, 1),
        &copy,
        0,
        side * texel,
        bytes,
        option,
    )
    .unwrap();
    blit.end_encoding().unwrap();
    run(&command_buffer);
    // SAFETY: the buffer holds `bytes` bytes the completed copy wrote.
    unsafe { std::slice::from_raw_parts(copy.contents().unwrap().as_ptr(), bytes as usize) }
        .to_vec()
}

/// A heap's depth and stencil textures a render pass clears and stores, in a tracked `Private`
/// placement heap: a `Depth32Float`, a `Stencil8` and a `Depth32Float_Stencil8` texture, each
/// under a buffer over its answered bytes filled with zeros, cleared by a pass (depth 0.25,
/// stencil 0x5a) that stores them, then the buffer filled with 0x3e. Where the device's driver
/// lays depth and stencil images apart from their memory (`DriverFact::DepthStencilImagesLieApart`:
/// MoltenVK), each texture is a shadow of the heap's bytes, in its planes: the pass's clear lands
/// in the depth plane, from the texture's first byte, and in the stencil plane after it at a
/// multiple of 4 bytes, and the fill changes every texel of each aspect read back by a blit
/// (D32_S8's two regions). Elsewhere the texture aliases the heap's memory in the driver's layout,
/// which Metal's answer for a pass's clear is not measured against, so `HEAPTEX attach` lines print
/// the counts only.
#[test]
fn a_heap_depth_attachment_s_clear_reaches_the_heap_s_bytes() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let shadows = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let side = 16u64;
        let texels = (side * side) as usize;
        let formats = [
            PixelFormat::Depth32Float,
            PixelFormat::Stencil8,
            PixelFormat::Depth32FloatStencil8,
        ];
        for format in formats {
            let mut descriptor = TextureDescriptor::texture_2d(format, side, side, false);
            descriptor.resource_options = options;
            descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
            let what = format!("{format:?}");
            let size = match device.heap_texture_size_and_align(&descriptor) {
                Ok(answer) => answer.size,
                Err(error) => {
                    eprintln!("HEAPTEX attach {what}: refused {error:?}");
                    continue;
                }
            };
            let heap = heap(device, HeapType::Placement, 1 << 20, options);
            let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
            let buffer = heap.new_buffer_at_offset(size, options, 0).unwrap();
            gpu_fill_zero(&queue, &buffer, size);
            let depth = format != PixelFormat::Stencil8;
            let stencil = format != PixelFormat::Depth32Float;
            let mut pass = RenderPassDescriptor::new();
            if depth {
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                attachment.clear_depth = 0.25;
            }
            if stencil {
                let attachment = &mut pass.stencil_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = LoadAction::Clear;
                attachment.store_action = StoreAction::Store;
                attachment.clear_stencil = 0x5a;
            }
            let command_buffer = queue.command_buffer().unwrap();
            command_buffer
                .render_command_encoder(&pass)
                .unwrap()
                .end_encoding()
                .unwrap();
            run(&command_buffer);
            let bytes = gpu_read(&queue, &buffer, 0..size);
            let changed = bytes.iter().filter(|&&byte| byte != 0).count();
            // The aspects' options of a combined texture.
            let (depth_option, stencil_option) = if depth && stencil {
                (
                    BlitOption::DepthFromDepthStencil,
                    BlitOption::StencilFromDepthStencil,
                )
            } else {
                (BlitOption::None, BlitOption::None)
            };
            gpu_fill(&queue, &buffer, 0..size, 0x3e);
            let kept_depth = if depth {
                let read = aspect_texels(&queue, &texture, side, 4, depth_option);
                read.as_chunks::<4>()
                    .0
                    .iter()
                    .filter(|texel| **texel == 0.25f32.to_le_bytes())
                    .count()
            } else {
                0
            };
            let kept_stencil = if stencil {
                let read = aspect_texels(&queue, &texture, side, 1, stencil_option);
                read.iter().filter(|&&texel| texel == 0x5a).count()
            } else {
                0
            };
            eprintln!(
                "HEAPTEX attach {what}: size {size}; the clear changed {changed} bytes; after the \
                 fill {kept_depth} depth and {kept_stencil} stencil texels kept the clear"
            );
            if shadows {
                let mut at = 0;
                if depth {
                    let plane = &bytes[..texels * 4];
                    assert!(
                        plane
                            .as_chunks::<4>()
                            .0
                            .iter()
                            .all(|t| *t == 0.25f32.to_le_bytes()),
                        "{what}: the depth plane does not hold the clear"
                    );
                    at = texels * 4;
                }
                if stencil {
                    at = at.next_multiple_of(4);
                    assert!(
                        bytes[at..at + texels].iter().all(|&t| t == 0x5a),
                        "{what}: the stencil plane does not hold the clear"
                    );
                }
                assert_eq!(
                    (kept_depth, kept_stencil),
                    (0, 0),
                    "{what}: texels the fill covered"
                );
            }
        }
    });
}

/// A render pass that executes indirect render commands keeps its heap depth attachment. A
/// `Private` placement heap's 16x16 `Depth32Float` texture is the depth attachment, cleared to 0.25
/// and stored, of a pass whose one command draws the owned `icb_render_replay_template` fixture's
/// vertex function (rasterization disabled: vertex `vid` writes `params[0] * (vid + 1) + params[1]`
/// to word `vid` of `out`), with the heap resident. The pass runs on every device and its command
/// draws; where the texture is a shadow of the heap's bytes (MoltenVK), its copies go before the
/// pass and as the encoder ends, and the clear reaches the heap's bytes, as for a pass of the
/// encoder's own draws. The runtime once refused such a pass there.
#[test]
fn a_pass_executing_render_commands_keeps_its_heap_depth_attachment() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let shadows = device
            .properties()
            .driver_facts
            .contains(&DriverFact::DepthStencilImagesLieApart);
        let side = 16u64;
        let texels = (side * side) as usize;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
        descriptor.resource_options = options;
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        let size = match device.heap_texture_size_and_align(&descriptor) {
            Ok(answer) => answer.size,
            Err(error) => {
                eprintln!("HEAPTEX icb: refused {error:?}");
                return;
            }
        };
        let heap = heap(device, HeapType::Placement, 1 << 20, options);
        let texture = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let bytes = heap.new_buffer_at_offset(size, options, 0).unwrap();
        gpu_fill_zero(&queue, &bytes, size);
        let bitcode = llvm_assemble(&engine_dir().join("icb_render_replay_template.ll")).unwrap();
        let library = device.new_library_with_air(&bitcode).unwrap();
        let function = library
            .new_function_with_name("icb_render_replay_template")
            .unwrap();
        let mut pipeline =
            RenderPipelineDescriptor::new(StageSource::Function(function.function_descriptor()));
        pipeline.rasterization_enabled = false;
        pipeline.support_indirect_command_buffers = true;
        pipeline.depth_format = Some(PixelFormat::Depth32Float);
        let state = device
            .new_render_pipeline_state(&pipeline, &PipelineOptions::default())
            .unwrap();
        let params: Vec<u8> = [3u32, 11].iter().flat_map(|w| w.to_le_bytes()).collect();
        let params = buffer_with(device, &params).unwrap();
        let out = buffer_with(device, &[0; 64]).unwrap();
        let mut commands = IndirectCommandBufferDescriptor::new();
        commands.command_types = IndirectCommandType::DRAW;
        commands.max_vertex_buffer_bind_count = 2;
        commands.max_fragment_buffer_bind_count = 1;
        let icb = device
            .new_indirect_command_buffer(&commands, 1, ResourceOptions::shared())
            .unwrap();
        let command = icb.indirect_render_command_at(0).unwrap();
        command.set_render_pipeline_state(&state).unwrap();
        command.set_vertex_buffer(Some(&params), 0, 0).unwrap();
        command.set_vertex_buffer(Some(&out), 0, 1).unwrap();
        command
            .draw_primitives(PrimitiveType::Point, 0, 4, 1, 0)
            .unwrap();
        let mut pass = RenderPassDescriptor::new();
        let attachment = &mut pass.depth_attachment;
        attachment.texture = Some(texture.clone());
        attachment.load_action = LoadAction::Clear;
        attachment.store_action = StoreAction::Store;
        attachment.clear_depth = 0.25;
        let command_buffer = queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(&pass).unwrap();
        let resident = [(&params, ResourceUsage::READ), (&out, ResourceUsage::WRITE)];
        for (buffer, usage) in resident {
            encoder
                .use_resource(buffer.as_resource(), usage, RenderStages::VERTEX)
                .unwrap();
        }
        encoder.use_heap(&heap, RenderStages::VERTEX).unwrap();
        encoder.execute_commands_in_buffer(&icb, 0..1).unwrap();
        encoder.end_encoding().unwrap();
        run(&command_buffer);
        let drawn: Vec<u32> = read_buffer(&out, 0, 16)
            .unwrap()
            .as_chunks::<4>()
            .0
            .iter()
            .map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]]))
            .collect();
        let held = gpu_read(&queue, &bytes, 0..size);
        let cleared = held[..texels * 4]
            .as_chunks::<4>()
            .0
            .iter()
            .filter(|texel| **texel == 0.25f32.to_le_bytes())
            .count();
        eprintln!(
            "HEAPTEX icb: the command drew {drawn:?}; {cleared} of the heap's first {texels} \
             texels hold the clear"
        );
        assert_eq!(drawn, [14, 17, 20, 23], "the command's draw");
        if shadows {
            assert_eq!(cleared, texels, "the clear in the heap's bytes");
        }
    });
}

/// What a heap shadow's copies cost a render pass: a 1024x1024 `Depth32Float` texture, made outside
/// a heap and in a `Private` placement heap, as the depth attachment of 9 passes each in a command
/// buffer of its own, which clear it and do not store it (a frame's depth buffer), clear and store
/// it, or load and store it. Where the heap's texture is a shadow of its bytes (MoltenVK), a pass
/// that loads it copies its texels in before it, and one that stores it copies them back after it;
/// one that clears the whole texture takes no copy in. Elsewhere the two are alike. `HEAPTEX cost`
/// lines print each one's median GPU time per pass; nothing is asserted.
#[test]
fn a_heap_depth_attachment_s_copies_cost() {
    on_device(|device| {
        let queue = device.new_command_queue().unwrap();
        let options = tracked(StorageMode::Private);
        let side = 1024;
        let mut descriptor =
            TextureDescriptor::texture_2d(PixelFormat::Depth32Float, side, side, false);
        descriptor.usage = TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ;
        descriptor.resource_options = options;
        let plain = device.new_texture(&descriptor).unwrap();
        let size = match device.heap_texture_size_and_align(&descriptor) {
            Ok(answer) => answer.size,
            Err(error) => {
                eprintln!("HEAPTEX cost: refused {error:?}");
                return;
            }
        };
        let heap = heap(device, HeapType::Placement, size, options);
        let placed = heap.new_texture_at_offset(&descriptor, 0).unwrap();
        let actions = [
            (
                "clears and does not store",
                LoadAction::Clear,
                StoreAction::DontCare,
            ),
            ("clears and stores", LoadAction::Clear, StoreAction::Store),
            ("loads and stores", LoadAction::Load, StoreAction::Store),
        ];
        let made = [("plain", &plain), ("heap", &placed)];
        for ((how, load, store), (what, texture)) in actions
            .into_iter()
            .flat_map(|action| made.map(|made| (action, made)))
        {
            let mut times = Vec::new();
            for _ in 0..9 {
                let mut pass = RenderPassDescriptor::new();
                let attachment = &mut pass.depth_attachment;
                attachment.texture = Some(texture.clone());
                attachment.load_action = load;
                attachment.store_action = store;
                let command_buffer = queue.command_buffer().unwrap();
                command_buffer
                    .render_command_encoder(&pass)
                    .unwrap()
                    .end_encoding()
                    .unwrap();
                run(&command_buffer);
                let (start, end) = match (
                    command_buffer.gpu_start_time(),
                    command_buffer.gpu_end_time(),
                ) {
                    (Ok(start), Ok(end)) => (start, end),
                    (start, end) => {
                        eprintln!("HEAPTEX cost: no GPU times ({start:?}, {end:?})");
                        return;
                    }
                };
                times.push(end - start);
            }
            times.sort_by(f64::total_cmp);
            eprintln!(
                "HEAPTEX cost {what}: a {side}x{side} Depth32Float pass that {how}, median \
                 {:.1} us on the GPU (of {:.1} to {:.1})",
                times[4] * 1e6,
                times[0] * 1e6,
                times[8] * 1e6
            );
        }
    });
}

/// The bits of each `float`.
fn bits(values: &[f32]) -> Vec<u32> {
    values.iter().map(|v| v.to_bits()).collect()
}

/// Upload each `(texture, bytes)` (a `side`-square texture of 4-byte texels) with a blit, in order,
/// in one command buffer.
fn upload(queue: &CommandQueue, textures: &[(&Texture, &[u8])], side: u64) {
    let sources: Vec<Buffer> = textures
        .iter()
        .map(|(_, bytes)| staged(queue.device(), bytes))
        .collect();
    let command_buffer = queue.command_buffer().unwrap();
    for ((texture, _), source) in textures.iter().zip(&sources) {
        let mut blit = command_buffer.blit_command_encoder().unwrap();
        blit.copy_from_buffer_to_texture(
            source,
            0,
            side * 4,
            side * side * 4,
            Size::new(side, side, 1),
            texture,
            0,
            0,
            Origin::new(0, 0, 0),
            BlitOption::None,
        )
        .unwrap();
        blit.end_encoding().unwrap();
    }
    run(&command_buffer);
}
