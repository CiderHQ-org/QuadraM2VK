//! Device tests of memoryless textures (its fourth part) on the device the cfg selects, under the
//! validation layer.
//!
//! A memoryless texture is a transient render attachment. Metal probes measured (an M4 with macOS
//! 27.0.1, each case with and without Metal's API validation) what Metal renders through one and
//! which uses its validation stops. Each pass here is one the probes ran, drawn with the owned
//! fixture `rt6_resolves` and compared with the texels Metal left (a resolve Metal's, as
//! `resolves.rs` holds every unorm, sRGB and float resolve). Each refusal is one Metal's validation
//! stopped, refused typed before anything records; each use Metal took is taken.

use super::resolves::{
    color_pass, lane, near, quiet, repeat, Channels, Lane, Shape, FULL, MSAA, ONE, SAMPLES, SIZE,
    WHITE,
};
use crate::runtime_lane::buffer_with;
use metal2vulkan::device::{VkExtension, VkFeature};
use metal2vulkan::error::{Capability, Needs};
use metal2vulkan::metal::{PixelFormat, TextureType, TextureUsage};
use metal2vulkan_runtime::{
    Attachment, BlitOption, CpuCacheMode, Error, HazardTrackingMode, Invalid, LoadAction, Origin,
    Problem, Region, RenderCommandEncoder, RenderPassDescriptor, RenderPassProblem,
    ResourceOptions, Size, StorageMode, StoreAction, Texture, TextureDescriptor,
};
use std::cell::RefCell;

/// `MTLResourceStorageModeMemoryless` with the other modes at their defaults.
pub(super) fn memoryless_options() -> ResourceOptions {
    ResourceOptions::new(
        StorageMode::Memoryless,
        CpuCacheMode::DefaultCache,
        HazardTrackingMode::Default,
    )
}

/// A memoryless texture of `ty`, `pf` and `shape`, with `usage`.
pub(super) fn memoryless(
    lane: &Lane,
    ty: TextureType,
    pf: PixelFormat,
    shape: Shape,
    usage: TextureUsage,
) -> Result<Texture, Error> {
    let mut descriptor = TextureDescriptor::texture_2d(pf, shape.size, shape.size, false);
    descriptor.texture_type = ty;
    descriptor.sample_count = shape.samples;
    descriptor.array_length = shape.slices;
    descriptor.mipmap_level_count = shape.levels;
    descriptor.usage = usage;
    descriptor.resource_options = memoryless_options();
    lane.device.new_texture(&descriptor)
}

/// A memoryless 4x 4² render target of `pf`.
pub(super) fn memoryless_msaa(lane: &Lane, pf: PixelFormat) -> Texture {
    memoryless(
        lane,
        TextureType::Type2DMultisample,
        pf,
        MSAA,
        TextureUsage::RENDER_TARGET,
    )
    .unwrap()
}

/// A memoryless 4² render target of `pf`.
fn memoryless_single(lane: &Lane, pf: PixelFormat) -> Texture {
    memoryless(
        lane,
        TextureType::Type2D,
        pf,
        ONE,
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    )
    .unwrap()
}

/// The passes the probes drew through memoryless attachments: Metal's texels.
#[test]
fn memoryless_attachments_render_as_metal() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let texels = (SIZE * SIZE) as usize;
    let differ = RefCell::new(Vec::new());
    // Probe 6's `res_memoryless`: a memoryless 4x target drawn with the per-sample colors and
    // resolved, as `res_unorm_samples` resolves a stored one.
    let msaa = memoryless_msaa(&lane, rgba8);
    let resolve = lane.single(rgba8);
    let samples = lane.pipeline((FULL, SAMPLES), &[rgba8], (None, None), 4);
    lane.run(
        &color_pass(
            &msaa,
            Some(&resolve),
            StoreAction::MultisampleResolve,
            (0, 0),
        ),
        |l, e, h| l.draw(e, h, &samples, 0.5),
    )
    .unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    let metal = repeat(&[0x80, 0x40, 0x80, 0xff], texels);
    near(
        &lane,
        "res_memoryless",
        &resolved,
        &metal,
        Channels::Unorm8,
        &differ,
    );
    // Probe 6d's `late_resolve`: the store action left `Unknown`, then `MultisampleResolve`.
    let white = lane.pipeline((FULL, WHITE), &[rgba8], (None, None), 4);
    let resolve = lane.single(rgba8);
    lane.run(
        &color_pass(&msaa, Some(&resolve), StoreAction::Unknown, (0, 0)),
        |l, e, h| {
            l.draw(e, h, &white, 0.5);
            e.set_color_store_action(StoreAction::MultisampleResolve, 0)
                .unwrap();
        },
    )
    .unwrap();
    let resolved = lane.read(&resolve, (0, 0), 4, BlitOption::None);
    near(
        &lane,
        "late_resolve",
        &resolved,
        &repeat(&[0xff; 4], texels),
        Channels::Unorm8,
        &differ,
    );
    // Probe 6b's `pass_memoryless_depth`: a stored color drawn beside a memoryless depth
    // attachment, cleared and not stored.
    let depth = PixelFormat::Depth32Float;
    let color = lane.single(rgba8);
    let mut descriptor = color_pass(&color, None, StoreAction::Store, (0, 0));
    descriptor.depth_attachment.texture = Some(memoryless_single(&lane, depth));
    let white = lane.pipeline((FULL, WHITE), &[rgba8], (Some(depth), None), 1);
    lane.run(&descriptor, |l, e, h| l.draw(e, h, &white, 0.5))
        .unwrap();
    assert_eq!(
        lane.read(&color, (0, 0), 4, BlitOption::None),
        repeat(&[0xff; 4], texels),
        "pass_memoryless_depth"
    );
    let differ = differ.into_inner();
    assert!(differ.is_empty(), "resolves not Metal's: {differ:#?}");
    quiet(lane, &messages);
}

/// The memoryless texture a pass attaches: its type, its width, height, depth, array length and
/// level count, and the level, slice and depth plane attached.
struct Attached {
    ty: TextureType,
    extent: (u64, u64, u64),
    array_length: u64,
    levels: u64,
    at: (u64, u64, u64),
}

/// A memoryless texture of `t` is made and attached beside a stored color, as plain Metal makes
/// and renders through one (an M4 under macOS 27.0.1: a pass of the memoryless texture's slice or
/// depth plane beside a stored color; Metal's validation stops the descriptors of all but 2D, 2D
/// array and multisample types): the pass draws its stored color white. Each shape is its own test,
/// so a driver that fails one fails it alone.
fn renders_through(t: Attached) {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let white = lane.pipeline((FULL, WHITE), &[rgba8, rgba8], (None, None), 1);
    let (width, height, depth) = t.extent;
    let ty = t.ty;
    let mut descriptor = TextureDescriptor::texture_2d(rgba8, width, height, false);
    descriptor.texture_type = ty;
    descriptor.depth = depth;
    descriptor.array_length = t.array_length;
    descriptor.mipmap_level_count = t.levels;
    descriptor.usage = TextureUsage::RENDER_TARGET;
    descriptor.resource_options = memoryless_options();
    let texture = lane
        .device
        .new_texture(&descriptor)
        .unwrap_or_else(|error| panic!("{ty:?}: {error:?}"));
    assert_eq!(texture.storage_mode(), StorageMode::Memoryless, "{ty:?}");
    let (level, slice, plane) = t.at;
    let (columns, rows) = ((width >> level).max(1), (height >> level).max(1));
    let mut stored = TextureDescriptor::texture_2d(rgba8, columns, rows, false);
    stored.usage = TextureUsage::RENDER_TARGET;
    stored.resource_options = ResourceOptions::private();
    let stored = lane.device.new_texture(&stored).unwrap();
    let mut pass = color_pass(&stored, None, StoreAction::Store, (0, 0));
    let target = &mut pass.color_attachments[1];
    target.texture = Some(texture);
    target.level = level;
    target.slice = slice;
    target.depth_plane = plane;
    target.load_action = LoadAction::Clear;
    target.store_action = StoreAction::DontCare;
    lane.run(&pass, |l, e, h| l.draw(e, h, &white, 0.5))
        .unwrap_or_else(|error| panic!("{ty:?}: {error:?}"));
    assert_eq!(
        lane.read(&stored, (0, 0), 4, BlitOption::None),
        repeat(&[0xff; 4], (columns * rows) as usize),
        "{ty:?}"
    );
    quiet(lane, &messages);
}

#[test]
fn a_memoryless_1d_texture_renders() {
    renders_through(Attached {
        ty: TextureType::Type1D,
        extent: (SIZE, 1, 1),
        array_length: 1,
        levels: 1,
        at: (0, 0, 0),
    });
}

#[test]
fn a_memoryless_1d_array_slice_renders() {
    renders_through(Attached {
        ty: TextureType::Type1DArray,
        extent: (SIZE, 1, 1),
        array_length: 2,
        levels: 1,
        at: (0, 1, 0),
    });
}

/// A memoryless 3D texture's depth plane: on MoltenVK, whose 2D view of a 3D image in lazily
/// allocated memory crashed this test's process (`SIGSEGV` in `MVKImageViewPlane::newMTLTexture`,
/// 1.4.1 on an M4), its image takes device-local memory
/// (`RuntimeDriverDefect::LazilyAllocatedVolumePlaneViews`).
#[test]
fn a_memoryless_3d_depth_plane_renders() {
    renders_through(Attached {
        ty: TextureType::Type3D,
        extent: (SIZE, SIZE, SIZE),
        array_length: 1,
        levels: 1,
        at: (0, 0, 2),
    });
}

#[test]
fn a_memoryless_cube_face_renders() {
    renders_through(Attached {
        ty: TextureType::Cube,
        extent: (SIZE, SIZE, 1),
        array_length: 1,
        levels: 1,
        at: (0, 3, 0),
    });
}

#[test]
fn a_memoryless_cube_array_face_renders() {
    renders_through(Attached {
        ty: TextureType::CubeArray,
        extent: (SIZE, SIZE, 1),
        array_length: 2,
        levels: 1,
        at: (0, 9, 0),
    });
}

#[test]
fn a_memoryless_2d_array_slice_renders() {
    renders_through(Attached {
        ty: TextureType::Type2DArray,
        extent: (SIZE, SIZE, 1),
        array_length: 2,
        levels: 1,
        at: (0, 1, 0),
    });
}

#[test]
fn a_memoryless_2d_level_renders() {
    renders_through(Attached {
        ty: TextureType::Type2D,
        extent: (2 * SIZE, 2 * SIZE, 1),
        array_length: 1,
        levels: 2,
        at: (1, 0, 0),
    });
}

/// What Metal makes memoryless and what its validation stops of a memoryless texture (probes 6b,
/// 6c, 6d and 6f), refused typed; the uses it takes are taken.
#[test]
fn memoryless_textures_are_refused_as_metal_refuses_them() {
    let (messages, _cache, lane) = lane();
    let rgba8 = PixelFormat::RGBA8Unorm;
    let c0 = Attachment::Color(0);
    let invalid = |call, problem| Err(Error::Invalid(Invalid::new(call, problem)));
    let pass_problem = |call, problem| invalid(call, Problem::RenderPass(problem));
    let storage = |call| invalid(call, Problem::StorageMode);

    // Made: 2D, 2D arrays and 2D multisample textures, with any usage, mipmapped or not, of a
    // color or a depth format (`make_*`), and a multisample array (`ms_array`). The other types
    // are made too, and each renders (`renders_through`'s tests).
    let read_only = TextureUsage::SHADER_READ;
    let made = [
        (TextureType::Type2D, rgba8, ONE, read_only),
        (
            TextureType::Type2D,
            rgba8,
            ONE,
            TextureUsage::SHADER_WRITE | TextureUsage::RENDER_TARGET,
        ),
        (
            TextureType::Type2D,
            rgba8,
            Shape { levels: 3, ..ONE },
            TextureUsage::RENDER_TARGET,
        ),
        (
            TextureType::Type2DArray,
            rgba8,
            Shape { slices: 2, ..ONE },
            TextureUsage::RENDER_TARGET,
        ),
        (
            TextureType::Type2DMultisample,
            rgba8,
            MSAA,
            TextureUsage::RENDER_TARGET,
        ),
        (
            TextureType::Type2D,
            PixelFormat::Depth32Float,
            ONE,
            TextureUsage::RENDER_TARGET,
        ),
    ];
    for (ty, pf, shape, usage) in made {
        let texture = memoryless(&lane, ty, pf, shape, usage)
            .unwrap_or_else(|error| panic!("{ty:?} {pf:?}: {error:?}"));
        assert_eq!(texture.storage_mode(), StorageMode::Memoryless);
    }
    // The multisample array, which a portability subset may withhold: refused there, naming
    // `multisampleArrayImage`, as any multisample array is.
    let enabled = lane.device.compile_device().profile().enabled();
    let withheld = enabled
        .extensions
        .contains(&VkExtension::KhrPortabilitySubset)
        && !enabled.features.contains(&VkFeature::MultisampleArrayImage);
    let ms_array = memoryless(
        &lane,
        TextureType::Type2DMultisampleArray,
        rgba8,
        Shape { slices: 2, ..MSAA },
        TextureUsage::RENDER_TARGET,
    );
    if withheld {
        let Err(Error::Unsupported(refused)) = ms_array else {
            panic!("ms_array: {ms_array:?}");
        };
        assert_eq!(
            refused.needs,
            Needs::AnyOf(vec![vec![Capability::Feature(
                VkFeature::MultisampleArrayImage
            )]])
        );
        eprintln!("refused: ms_array: the portability subset withholds multisampleArrayImage");
    } else {
        let texture = ms_array.unwrap_or_else(|error| panic!("ms_array: {error:?}"));
        assert_eq!(texture.storage_mode(), StorageMode::Memoryless);
    }
    // Passes: `Load` (`load_load`, `depth_load`), a store (`default_store`,
    // `pass_memoryless_depth_store`, `stencil_store`, `msaa_store_and_resolve`) and a memoryless
    // resolve texture (`pass_memoryless_resolve_target`) are refused when the encoder opens.
    let open = |descriptor: &RenderPassDescriptor| {
        lane.queue
            .command_buffer()
            .unwrap()
            .render_command_encoder(descriptor)
            .map(|_| ())
    };
    let call = "render_command_encoder";
    let target = memoryless_single(&lane, rgba8);
    let mut load = color_pass(&target, None, StoreAction::DontCare, (0, 0));
    load.color_attachments[0].load_action = LoadAction::Load;
    assert_eq!(
        open(&load),
        pass_problem(call, RenderPassProblem::MemorylessLoad(c0))
    );
    let mut default_store = RenderPassDescriptor::new();
    default_store.color_attachments[0].texture = Some(target.clone());
    default_store.color_attachments[0].load_action = LoadAction::Clear;
    assert_eq!(
        open(&default_store),
        pass_problem(call, RenderPassProblem::MemorylessStore(c0))
    );
    assert_eq!(
        open(&color_pass(&target, None, StoreAction::DontCare, (0, 0))),
        Ok(())
    );
    let msaa = memoryless_msaa(&lane, rgba8);
    let resolve = lane.single(rgba8);
    assert_eq!(
        open(&color_pass(
            &msaa,
            Some(&resolve),
            StoreAction::StoreAndMultisampleResolve,
            (0, 0)
        )),
        pass_problem(call, RenderPassProblem::MemorylessStore(c0))
    );
    assert_eq!(
        open(&color_pass(
            &lane.msaa(rgba8),
            Some(&target),
            StoreAction::MultisampleResolve,
            (0, 0)
        )),
        pass_problem(call, RenderPassProblem::MemorylessResolve(c0))
    );
    let depth = memoryless_single(&lane, PixelFormat::Depth32Float);
    let stencil = memoryless_single(&lane, PixelFormat::Depth32FloatStencil8);
    let color = lane.single(rgba8);
    let with_depth = |load, store| {
        let mut descriptor = color_pass(&color, None, StoreAction::Store, (0, 0));
        descriptor.depth_attachment.texture = Some(depth.clone());
        descriptor.depth_attachment.load_action = load;
        descriptor.depth_attachment.store_action = store;
        descriptor
    };
    assert_eq!(
        open(&with_depth(LoadAction::Load, StoreAction::DontCare)),
        pass_problem(call, RenderPassProblem::MemorylessLoad(Attachment::Depth))
    );
    assert_eq!(
        open(&with_depth(LoadAction::Clear, StoreAction::Store)),
        pass_problem(call, RenderPassProblem::MemorylessStore(Attachment::Depth))
    );
    let mut with_stencil = color_pass(&color, None, StoreAction::Store, (0, 0));
    with_stencil.stencil_attachment.texture = Some(stencil.clone());
    with_stencil.stencil_attachment.store_action = StoreAction::Store;
    assert_eq!(
        open(&with_stencil),
        pass_problem(
            call,
            RenderPassProblem::MemorylessStore(Attachment::Stencil)
        )
    );

    // Late store actions (`late_store`, `late_store_and_resolve`, `late_depth_store`) are
    // refused when set; `DontCare` (`late_dontcare`, `late_depth_dontcare`) is taken.
    let late = |descriptor: &RenderPassDescriptor,
                set: &dyn Fn(&mut RenderCommandEncoder) -> Result<(), Error>|
     -> Result<(), Error> {
        let command_buffer = lane.queue.command_buffer().unwrap();
        let mut encoder = command_buffer.render_command_encoder(descriptor).unwrap();
        set(&mut encoder)
    };
    let unknown = color_pass(&target, None, StoreAction::Unknown, (0, 0));
    assert_eq!(
        late(&unknown, &|e| e
            .set_color_store_action(StoreAction::Store, 0)),
        pass_problem(
            "set_color_store_action",
            RenderPassProblem::MemorylessStore(c0)
        )
    );
    assert_eq!(
        late(&unknown, &|e| e
            .set_color_store_action(StoreAction::DontCare, 0)),
        Ok(())
    );
    let unknown_msaa = color_pass(&msaa, Some(&resolve), StoreAction::Unknown, (0, 0));
    assert_eq!(
        late(&unknown_msaa, &|e| e.set_color_store_action(
            StoreAction::StoreAndMultisampleResolve,
            0
        )),
        pass_problem(
            "set_color_store_action",
            RenderPassProblem::MemorylessStore(c0)
        )
    );
    let unknown_depth = with_depth(LoadAction::Clear, StoreAction::Unknown);
    assert_eq!(
        late(&unknown_depth, &|e| e
            .set_depth_store_action(StoreAction::Store)),
        pass_problem(
            "set_depth_store_action",
            RenderPassProblem::MemorylessStore(Attachment::Depth)
        )
    );
    assert_eq!(
        late(&unknown_depth, &|e| e
            .set_depth_store_action(StoreAction::DontCare)),
        Ok(())
    );

    // CPU access (`get_bytes`, `replace_region`) and views (`view`).
    let region = Region::new_2d(0, 0, SIZE, SIZE);
    let mut bytes = vec![0; (SIZE * SIZE * 4) as usize];
    assert_eq!(
        target.get_bytes(&mut bytes, SIZE * 4, 0, region, 0, 0),
        storage("get_bytes")
    );
    assert_eq!(
        target.replace_region(region, 0, 0, &bytes, SIZE * 4, 0),
        storage("replace_region")
    );
    assert_eq!(
        target
            .new_texture_view(PixelFormat::RGBA8UnormSRGB)
            .map(|_| ()),
        storage("new_texture_view")
    );
    assert_eq!(
        target
            .new_texture_view_with_levels_slices(rgba8, TextureType::Type2D, 0..1, 0..1)
            .map(|_| ()),
        storage("new_texture_view_with_levels_slices")
    );

    // Blits (`blit_from`, `blit_to`, `copy_to_buffer`, `copy_from_buffer`, `mipmaps`) are
    // refused; the hints (`optimize_gpu`, `optimize_gpu_slice_level`, `optimize_cpu`) taken.
    let buffer = buffer_with(&lane.device, &bytes).unwrap();
    let other = lane.texture(
        rgba8,
        Shape {
            shared: true,
            ..ONE
        },
        TextureUsage::SHADER_READ,
    );
    let mipmapped = memoryless(
        &lane,
        TextureType::Type2D,
        rgba8,
        Shape { levels: 3, ..ONE },
        TextureUsage::RENDER_TARGET | TextureUsage::SHADER_READ,
    )
    .unwrap();
    let command_buffer = lane.queue.command_buffer().unwrap();
    let mut blit = command_buffer.blit_command_encoder().unwrap();
    let origin = Origin::new(0, 0, 0);
    let size = Size::new(SIZE, SIZE, 1);
    assert_eq!(
        blit.copy_from_texture_to_buffer(
            &target,
            0,
            0,
            origin,
            size,
            &buffer,
            0,
            SIZE * 4,
            0,
            BlitOption::None
        ),
        storage("copy_from_texture_to_buffer")
    );
    assert_eq!(
        blit.copy_from_buffer_to_texture(
            &buffer,
            0,
            SIZE * 4,
            0,
            size,
            &target,
            0,
            0,
            origin,
            BlitOption::None
        ),
        storage("copy_from_buffer_to_texture")
    );
    assert_eq!(
        blit.copy_from_texture_to_texture(&target, 0, 0, origin, size, &other, 0, 0, origin),
        storage("copy_from_texture_to_texture")
    );
    assert_eq!(
        blit.copy_from_texture_to_texture(&other, 0, 0, origin, size, &target, 0, 0, origin),
        storage("copy_from_texture_to_texture")
    );
    assert_eq!(
        blit.generate_mipmaps(&mipmapped),
        storage("generate_mipmaps")
    );
    assert_eq!(blit.optimize_contents_for_gpu_access(&target), Ok(()));
    assert_eq!(
        blit.optimize_contents_for_gpu_access_slice_level(&target, 0, 0),
        Ok(())
    );
    assert_eq!(blit.optimize_contents_for_cpu_access(&target), Ok(()));
    blit.end_encoding().unwrap();

    // Bindings (`bind_fragment`, `bind_vertex`, `bind_unset`, `bind_compute`) are refused; a
    // compute memory barrier naming one (`compute_barrier`) is taken.
    let mut compute = command_buffer.compute_command_encoder().unwrap();
    assert_eq!(
        compute.set_texture(Some(&target), 0),
        storage("set_texture")
    );
    assert_eq!(
        compute.memory_barrier_with_resources(&[target.as_resource()]),
        Ok(())
    );
    compute.end_encoding().unwrap();
    let mut render = command_buffer
        .render_command_encoder(&color_pass(&color, None, StoreAction::Store, (0, 0)))
        .unwrap();
    assert_eq!(
        render.set_fragment_texture(Some(&target), 0),
        storage("set_fragment_texture")
    );
    assert_eq!(
        render.set_vertex_texture(Some(&target), 0),
        storage("set_vertex_texture")
    );
    render.end_encoding().unwrap();
    command_buffer.commit().unwrap();
    command_buffer
        .wait_until_completed_for(super::WAIT)
        .unwrap();
    assert_eq!(command_buffer.error(), None);
    quiet(lane, &messages);
}
