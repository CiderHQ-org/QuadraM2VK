//! The check of `DriverDeviation::GraphicsCubeGradientRoute` and
//! `DriverDeviation::GraphicsCubeGradientTangents`, one program, on a device whose kernels hand a
//! cube's explicit gradients to Metal's own reading (`DriverDeviation::CubeGradientRoute`, which
//! MoltenVK takes): [`LANES`] directions, [`PER_FACE`] on each face at a major component of 0.5, 1
//! or 2, each with random gradients, sample a [`SIDE`]-square `RGBA8Unorm` cube of [`LEVELS`] levels
//! of random texels and layer 1 of a cube array of two such cubes, through an anisotropic sampler
//! (16 probes) and an isotropic one, both filtering linearly with a linear mip filter and clamping
//! to the edge. A kernel samples with the gradients as given
//! (`metal2vulkan::bind::cube_gradients_check_kernel`): Metal's sample on the device, the
//! reference. A fragment stage samples with them in each graphics route's form
//! (`cube_gradients_check_fragment`).
//!
//! - `GraphicsCubeGradientRoute` (the graphics fixup as macOS 26.7 reads it) is Metal's answer
//!   where every lane's sample in that route's forms equals the kernel's bit for bit: its
//!   anisotropic form through the anisotropic sampler, its isotropic form through the isotropic
//!   one.
//! - `GraphicsCubeGradientTangents` (the fixup as macOS 27 reads it) is Metal's answer where, in
//!   the projected form, every lane on an `x` or `z` face equals the kernel's through the
//!   anisotropic sampler and every lane does through the isotropic one. A `y` face's anisotropic
//!   sample differs on that route by its own rule (`PrecisionNote::CubeGradientFootprint`), so it
//!   is not compared.
//!
//! Otherwise each is the other answer; a sample left unwritten answers neither.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Graphics, Triangle, UNWRITTEN};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{RasterCheck, SamplerDescriptor};
use metal2vulkan::device::{CheckAnswer, VkFeature};
use metal2vulkan::error::DriverDeviation;
use metal2vulkan::metal::sampler::{OpaqueBlackBorder, SamplerUse};
use metal2vulkan::metal::{SamplerMinMagFilter, SamplerMipFilter};

/// A face's side, in texels, at level 0.
const SIDE: u32 = 16;

/// The cubes' levels: 16 texels down to 1.
const LEVELS: u32 = 5;

/// The cubes' format.
const FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;

/// The lanes on each face.
const PER_FACE: u32 = 16;

/// The lanes, face by face in Vulkan's layer order (`+x`, `-x`, `+y`, `-y`, `+z`, `-z`): a
/// multiple of the kernel's workgroup.
const LANES: u32 = 6 * PER_FACE;

/// The anisotropic sampler's probes.
const ANISOTROPY: u32 = 16;

/// A pseudo-random sequence (xorshift32), fixed by its seed.
struct Draws(u32);

impl Draws {
    fn word(&mut self) -> u32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        self.0 = x;
        x
    }

    /// A float in `[-1, 1)`.
    fn signed(&mut self) -> f32 {
        (self.word() >> 8) as f32 / (1u32 << 23) as f32 - 1.0
    }
}

/// The lanes' records (`RasterCheck::CUBE_LANE_RECORDS` `vec4`s each, as words): lane `l` lies on
/// face `l / PER_FACE`, its major component 0.5, 1 or 2 and its others within 0.7 of it, and each
/// gradient's three components within an eighth to all of the major component, so its footprint
/// spans up to 1 to 8 texels of level 0.
fn lanes() -> Vec<u32> {
    let mut draws = Draws(0x9e37_79b9);
    let mut words = Vec::new();
    for lane in 0..LANES {
        let face = lane / PER_FACE;
        let k = lane % PER_FACE;
        let major = [0.5f32, 1.0, 2.0][(k % 3) as usize];
        let scale = [0.125f32, 0.25, 0.5, 1.0][(k % 4) as usize];
        let axis = (face / 2) as usize;
        let mut p = [0.0f32; 3];
        for (c, value) in p.iter_mut().enumerate() {
            *value = match c == axis {
                true if face.is_multiple_of(2) => major,
                true => -major,
                false => 0.7 * major * draws.signed(),
            };
        }
        let mut gradient = || [0; 3].map(|_: u32| major * scale * draws.signed());
        let (dx, dy) = (gradient(), gradient());
        for record in [p, dx, dy] {
            words.extend(record.iter().map(|v| v.to_bits()));
            words.push(0);
        }
    }
    words
}

/// Whether lane `l` lies on a `y` face.
fn on_y_face(lane: u32) -> bool {
    lane / PER_FACE / 2 == 1
}

/// The check's answers on `target`, `[GraphicsCubeGradientRoute, GraphicsCubeGradientTangents]`:
/// not run where the device's kernels do not hand a cube's gradients to Metal's own reading, or
/// where it lacks a fragment stage's storage writes, cube arrays or anisotropic filtering.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    if let Some(why) = unsupported(target) {
        return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)];
    }
    match samples(kit, target) {
        Ok((fragment, kernel)) => answers(&fragment, &kernel).to_vec(),
        Err(error) => vec![
            CheckAnswer::Unanswered(error.0.clone()),
            CheckAnswer::Unanswered(error.0),
        ],
    }
}

/// Why the device cannot run the check, where it cannot.
fn unsupported(target: &Target<'_>) -> Option<String> {
    if !(target.compile.profile()).takes_driver_deviation(DriverDeviation::CubeGradientRoute) {
        return Some(
            "the device's kernels do not hand a cube's gradients to Metal's own reading \
             (CubeGradientRoute), so no sample is Metal's to compare with"
                .to_owned(),
        );
    }
    let features = &target.enabled.features;
    [
        VkFeature::FragmentStoresAndAtomics,
        VkFeature::ImageCubeArray,
        VkFeature::SamplerAnisotropy,
    ]
    .into_iter()
    .find(|feature| !features.contains(feature))
    .map(|feature| format!("{feature:?} is not enabled"))
}

/// The answers of the fragment stage's and the kernel's words (the module docs).
fn answers(fragment: &[u32], kernel: &[u32]) -> [CheckAnswer; 2] {
    if fragment.iter().chain(kernel).any(|&w| w == UNWRITTEN) {
        let why = "a lane's sample was left unwritten".to_owned();
        return [
            CheckAnswer::Unanswered(why.clone()),
            CheckAnswer::Unanswered(why),
        ];
    }
    let fragment_words = 4 * RasterCheck::CUBE_FRAGMENT_SAMPLES as usize;
    let kernel_words = 4 * RasterCheck::CUBE_KERNEL_SAMPLES as usize;
    // A lane's samples `k..k + 2` (the cube's and the array's) of the fragment stage, against the
    // kernel's through the anisotropic sampler (`anisotropic`) or the isotropic one.
    let equal = |lane: u32, k: usize, anisotropic: bool| {
        let at = lane as usize;
        let f = &fragment[at * fragment_words + 4 * k..][..8];
        let reference = if anisotropic { 0 } else { 2 };
        let r = &kernel[at * kernel_words + 4 * reference..][..8];
        f == r
    };
    let differ = |same: &dyn Fn(u32) -> bool| (0..LANES).filter(|&l| !same(l)).count();
    let verdict = |n: usize, forms: &str| match n {
        0 => Ok(()),
        n => Err(format!(
            "{n} of {LANES} lanes' samples in {forms} differ from the kernel's"
        )),
    };
    let route = differ(&|l| equal(l, 0, true) && equal(l, 2, false));
    let tangents = differ(&|l| (on_y_face(l) || equal(l, 4, true)) && equal(l, 6, false));
    [
        raster::answer(verdict(route, "macOS 26.7's forms")),
        raster::answer(verdict(tangents, "the projected form")),
    ]
}

/// A cube image of `layers` layers (6 a cube) and its view of `view_type`, all its levels.
fn cube(
    kit: &mut Kit<'_>,
    layers: u32,
    view_type: vk::ImageViewType,
) -> Result<(vk::Image, vk::ImageView), CheckError> {
    let info = vk::ImageCreateInfo::default()
        .flags(vk::ImageCreateFlags::CUBE_COMPATIBLE)
        .image_type(vk::ImageType::TYPE_2D)
        .format(FORMAT)
        .extent(vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        })
        .mip_levels(LEVELS)
        .array_layers(layers)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    let image = kit.image(&info)?;
    let view = kit.view(
        &vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(view_type)
            .format(FORMAT)
            .subresource_range(levels(layers)),
    )?;
    Ok((image, view))
}

/// Every level and the first `layers` layers of a cube image.
fn levels(layers: u32) -> vk::ImageSubresourceRange {
    vk::ImageSubresourceRange {
        aspect_mask: vk::ImageAspectFlags::COLOR,
        base_mip_level: 0,
        level_count: LEVELS,
        base_array_layer: 0,
        layer_count: layers,
    }
}

/// The random texels of every level and layer of the cube (6 layers) and the cube array (12),
/// level by level, each level's layers in order, and each image's copies from them.
fn texels() -> (Vec<u32>, [Vec<vk::BufferImageCopy>; 2]) {
    let mut draws = Draws(0x2545_f491);
    let mut words = Vec::new();
    let copies = [6, 12].map(|layers| {
        (0..LEVELS)
            .map(|level| {
                let side = SIDE >> level;
                let copy = vk::BufferImageCopy {
                    buffer_offset: 4 * words.len() as vk::DeviceSize,
                    buffer_row_length: 0,
                    buffer_image_height: 0,
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: level,
                        base_array_layer: 0,
                        layer_count: layers,
                    },
                    image_offset: vk::Offset3D::default(),
                    image_extent: vk::Extent3D {
                        width: side,
                        height: side,
                        depth: 1,
                    },
                };
                words.extend((0..layers * side * side).map(|_| draws.word()));
                copy
            })
            .collect()
    });
    (words, copies)
}

/// The sampler of `target`'s device for an `MTLSamplerDescriptor` filtering linearly with a
/// linear mip filter, clamping to the edge, of `anisotropy` probes, as a draw's binding makes it
/// (`bind::sampler_desc`).
fn sampler(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    anisotropy: u32,
) -> Result<vk::Sampler, CheckError> {
    let mut desc = SamplerDescriptor::new();
    desc.min_filter = SamplerMinMagFilter::Linear;
    desc.mag_filter = SamplerMinMagFilter::Linear;
    desc.mip_filter = SamplerMipFilter::Linear;
    desc.max_anisotropy = anisotropy;
    let desc = metal2vulkan::bind::sampler_desc(
        target.compile,
        &desc,
        SamplerUse::new(false, false),
        OpaqueBlackBorder::Stated,
        Some(metal2vulkan::vk::Format::R8G8B8A8_UNORM),
    )
    .map_err(|e| {
        CheckError(format!(
            "its sampler of {anisotropy} probes is refused: {e}"
        ))
    })?;
    let mut reduction = vk::SamplerReductionModeCreateInfo::default();
    let mut border = vk::SamplerBorderColorComponentMappingCreateInfoEXT::default();
    let info = metal2vulkan::ash_interop::sampler_create_info(&desc, &mut reduction, &mut border);
    kit.sampler(&info)
}

/// The fragment stage's words and the kernel's, lane by lane.
fn samples(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<(Vec<u32>, Vec<u32>), CheckError> {
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::cube_gradients_check_fragment(target.compile),
        "fragment stage",
    )?;
    let kernel = raster::shader_of(
        kit,
        metal2vulkan::bind::cube_gradients_check_kernel(target.compile),
        "kernel",
    )?;
    let whole = (0.0, 0.0, LANES as f32, 1.0);
    let vertices = raster::vertex_buffer(kit, &[Triangle::covering([0.5; 3], whole)])?;
    let lanes = kit.host_buffer(&lanes(), vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let fragment_words = (LANES * 4 * RasterCheck::CUBE_FRAGMENT_SAMPLES) as usize;
    let kernel_words = (LANES * 4 * RasterCheck::CUBE_KERNEL_SAMPLES) as usize;
    let drawn = kit.host_buffer(
        &vec![UNWRITTEN; fragment_words],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let dispatched = kit.host_buffer(
        &vec![UNWRITTEN; kernel_words],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let (words, copies) = texels();
    let staged = kit.host_buffer(&words, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let (cube_image, cube_view) = cube(kit, 6, vk::ImageViewType::CUBE)?;
    let (array_image, array_view) = cube(kit, 12, vk::ImageViewType::CUBE_ARRAY)?;
    let anisotropic = sampler(kit, target, ANISOTROPY)?;
    let isotropic = sampler(kit, target, 1)?;
    let images = |stages: vk::ShaderStageFlags| {
        [
            (
                RasterCheck::CUBE_ANISOTROPIC_BINDING,
                cube_view,
                anisotropic,
            ),
            (RasterCheck::CUBE_ISOTROPIC_BINDING, cube_view, isotropic),
            (
                RasterCheck::CUBE_ARRAY_ANISOTROPIC_BINDING,
                array_view,
                anisotropic,
            ),
            (
                RasterCheck::CUBE_ARRAY_ISOTROPIC_BINDING,
                array_view,
                isotropic,
            ),
        ]
        .map(|(binding, view, sampler)| SetBinding {
            binding,
            stages,
            resource: SetResource::Combined(view, sampler),
        })
    };
    let pass = raster::pass(kit, (LANES, 1, 1), true, None)?;
    let fragment_stage = vk::ShaderStageFlags::FRAGMENT;
    let mut draw_bindings = vec![
        SetBinding {
            binding: RasterCheck::VERTICES_BINDING,
            stages: vk::ShaderStageFlags::VERTEX,
            resource: SetResource::Buffer(vertices),
        },
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages: fragment_stage,
            resource: SetResource::Buffer(drawn),
        },
        SetBinding {
            binding: RasterCheck::PARAMS_BINDING,
            stages: fragment_stage,
            resource: SetResource::Buffer(lanes),
        },
    ];
    draw_bindings.extend(images(fragment_stage));
    let compute = vk::ShaderStageFlags::COMPUTE;
    let mut read_bindings = vec![
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages: compute,
            resource: SetResource::Buffer(dispatched),
        },
        SetBinding {
            binding: RasterCheck::PARAMS_BINDING,
            stages: compute,
            resource: SetResource::Buffer(lanes),
        },
    ];
    read_bindings.extend(images(compute));
    let (draw_layout, draw_set) = kit.descriptor_set(&draw_bindings)?;
    let (read_layout, read_set) = kit.descriptor_set(&read_bindings)?;
    let draw_pipeline_layout = kit.layout(&[draw_layout], None)?;
    let read_pipeline_layout = kit.layout(&[read_layout], None)?;
    let draw = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: Some(fragment),
            layout: draw_pipeline_layout,
            depth: false,
            bias: false,
        },
    )?;
    let read = kit.compute_pipeline(kernel, read_pipeline_layout)?;
    let groups = LANES / RasterCheck::CUBE_KERNEL_LOCAL_SIZE;
    kit.submit(|device, command| {
        let barrier = |image, layers, (old, new), (src, dst)| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(levels(layers))
        };
        let into_transfer = (
            vk::ImageLayout::UNDEFINED,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        let written = (vk::AccessFlags::empty(), vk::AccessFlags::TRANSFER_WRITE);
        let into_reads = (
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
        let read_after = (
            vk::AccessFlags::TRANSFER_WRITE,
            vk::AccessFlags::SHADER_READ,
        );
        let shaders =
            vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER;
        // SAFETY: recording; every object named is live. The texels' copies lie outside the pass,
        // after a barrier that makes each image a transfer's destination, and before one that
        // orders the fragment stage's and the kernel's reads after them, in the layout the
        // descriptors name; the host's reads follow both stages' writes.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(cube_image, 6, into_transfer, written),
                    barrier(array_image, 12, into_transfer, written),
                ],
            );
            for (image, copies) in [(cube_image, &copies[0]), (array_image, &copies[1])] {
                device.cmd_copy_buffer_to_image(
                    command,
                    staged.buffer,
                    image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    copies,
                );
            }
            device.cmd_pipeline_barrier(
                command,
                vk::PipelineStageFlags::TRANSFER,
                shaders,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[
                    barrier(cube_image, 6, into_reads, read_after),
                    barrier(array_image, 12, into_reads, read_after),
                ],
            );
            device.cmd_begin_render_pass(command, &pass.begin(), vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, draw);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                draw_pipeline_layout,
                0,
                &[draw_set],
                &[],
            );
            raster::draw(device, command, 0, (0, 0, LANES, 1));
            device.cmd_end_render_pass(command);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, read);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                read_pipeline_layout,
                0,
                &[read_set],
                &[],
            );
            device.cmd_dispatch(command, groups, 1, 1);
            raster::host_reads(device, command, shaders);
        }
    })?;
    Ok((kit.read(&drawn)?, kit.read(&dispatched)?))
}
