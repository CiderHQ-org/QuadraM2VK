//! The check of `DriverDeviation::CenterFragCoordDepth` and `DriverDeviation::SampleFragCoordDepth`,
//! one program: a 4-sample pass of a `D32_SFLOAT` depth tested `ALWAYS` and written, three
//! triangles each wholly inside the clip volume of its own [`SIDE`]-square region, its viewport
//! and scissor (a flat one, a sloped one, and a sloped one under a constant depth bias), so the
//! device clips none into pieces (`raster::Triangle::inside`), drawn through a fragment stage that
//! runs per sample (`metal2vulkan::bind::fragcoord_depth_check_fragment`) and stores, at each
//! sample, `FragCoord.z` and the sample's depth computed as translation computes it where the
//! device's `FragCoord.z` is the pixel centre's. A kernel then reads back the depth the device's
//! own test stored at each sample (`depth_samples_check_kernel`), in a submission after the
//! pass's. In the pass's submission, ordered after the pass by its dependency alone, the kernel on
//! MoltenVK 1.4.1 on the M3 Max (macOS 27.0.1) read most samples before the pass stored them,
//! where the submission wrote timestamps.
//!
//! - `SampleFragCoordDepth` is Metal's answer (the function's depth is its sample's, so it tests
//!   `FragCoord.z` itself) where `FragCoord.z` equals the stored depth bit for bit at every
//!   sample; otherwise the other answer.
//! - `CenterFragCoordDepth` is Metal's answer (the computed depth is each sample's) where
//!   `FragCoord.z` is one value at every sample of a pixel (the centre's) and the computed depth
//!   equals the stored depth bit for bit at every sample; otherwise the other answer.
//!
//! Each region's pixels whose every sample its triangle covers are read (`raster::COVERED`); a
//! sample of those the fragment stage left unwritten, or whose stored depth reads as the unwritten
//! word, answers neither. The other answer names how many samples differ and the first, with its
//! three words.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, DepthTarget, Graphics, Triangle, UNWRITTEN};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::{CheckAnswer, VkExtension, VkFeature};

/// A region's side, in pixels: a multiple of the kernel's workgroup.
const SIDE: u32 = raster::CELL;

/// The regions, side by side: flat, sloped, biased.
const REGIONS: u32 = 3;

/// The target's width.
const WIDTH: u32 = SIDE * REGIONS;

/// The samples of the pass.
const SAMPLES: u32 = 4;

/// The depth.
const DEPTH: vk::Format = vk::Format::D32_SFLOAT;

/// The constant depth bias of the third triangle, in the device's unit of a float depth.
const BIAS: f32 = 3.0;

/// Each region's triangle's corner depths.
const DEPTHS: [[f32; 3]; REGIONS as usize] = [[0.5, 0.5, 0.5], [0.2, 0.6, 0.4], [0.3, 0.5, 0.7]];

/// What the stores held at one sample: sample `sample` of the pixel `pixel`.
struct AtSample {
    pixel: (u32, u32),
    sample: u32,
    frag_z: u32,
    computed: u32,
    stored: u32,
}

impl AtSample {
    /// The sample and its three words, as an answer's detail names them.
    fn named(&self) -> String {
        let word = |bits: u32| format!("{:?} ({bits:08x})", f32::from_bits(bits));
        format!(
            "sample {} of pixel ({}, {}): FragCoord.z {}, computed {}, stored {}",
            self.sample,
            self.pixel.0,
            self.pixel.1,
            word(self.frag_z),
            word(self.computed),
            word(self.stored)
        )
    }
}

/// The check's answers on `target`, `[CenterFragCoordDepth, SampleFragCoordDepth]`: not run
/// where the device lacks a per-sample fragment stage, its storage writes, a 4-sample sampled
/// `D32_SFLOAT` depth, or on a portability-subset device its interpolation at a sample.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let unanswered = |why: String| {
        vec![
            CheckAnswer::Unanswered(why.clone()),
            CheckAnswer::Unanswered(why),
        ]
    };
    match unsupported(kit, target) {
        Ok(None) => {}
        Ok(Some(why)) => return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)],
        Err(error) => return unanswered(error.0),
    }
    match samples(kit, target) {
        Ok(samples) => answers(&samples).to_vec(),
        Err(error) => unanswered(error.0),
    }
}

/// Why the device cannot run the check, where it cannot.
fn unsupported(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<Option<String>, CheckError> {
    let features = &target.enabled.features;
    for feature in [
        VkFeature::SampleRateShading,
        VkFeature::FragmentStoresAndAtomics,
    ] {
        if !features.contains(&feature) {
            return Ok(Some(format!("{feature:?} is not enabled")));
        }
    }
    // The fragment stage reads the window depth at each sample (`InterpolateAtSample`), which a
    // portability-subset device states apart.
    let portability = (target.enabled.extensions).contains(&VkExtension::KhrPortabilitySubset);
    if portability && !features.contains(&VkFeature::ShaderSampleRateInterpolationFunctions) {
        return Ok(Some(
            "ShaderSampleRateInterpolationFunctions is not enabled".to_owned(),
        ));
    }
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT | vk::ImageUsageFlags::SAMPLED;
    Ok((!raster::takes_samples(kit, DEPTH, usage, SAMPLES)?)
        .then(|| "no 4-sample sampled D32_SFLOAT depth".to_owned()))
}

/// The answers of the stores at every sample (the module docs).
fn answers(samples: &[AtSample]) -> [CheckAnswer; 2] {
    let unwritten = |why: &str| {
        [
            CheckAnswer::Unanswered(why.to_owned()),
            CheckAnswer::Unanswered(why.to_owned()),
        ]
    };
    if (samples.iter()).any(|s| s.frag_z == UNWRITTEN || s.computed == UNWRITTEN) {
        return unwritten("the fragment stage left a sample unwritten");
    }
    if samples.iter().any(|s| s.stored == UNWRITTEN) {
        return unwritten("a sample's stored depth reads as the unwritten word");
    }
    let total = samples.len();
    // How many samples `differs` holds at, and the first.
    let off = |differs: &dyn Fn(&AtSample) -> bool| {
        let mut off = samples.iter().filter(|s| differs(s));
        match off.next() {
            None => (0, String::new()),
            Some(first) => (1 + off.count(), first.named()),
        }
    };
    let at_sample = match off(&|s| s.frag_z != s.stored) {
        (0, _) => Ok(()),
        (n, first) => Err(format!(
            "FragCoord.z is not the stored depth at {n} of {total} samples, first {first}"
        )),
    };
    let varying = samples
        .chunks(SAMPLES as usize)
        .filter(|pixel| pixel.iter().any(|s| s.frag_z != pixel[0].frag_z))
        .count();
    let at_centre = match (varying, off(&|s| s.computed != s.stored)) {
        (0, (0, _)) => Ok(()),
        (0, (n, first)) => Err(format!(
            "the computed depth is not the stored one at {n} of {total} samples, first {first}"
        )),
        (pixels, _) => Err(format!(
            "FragCoord.z differs among the samples of {pixels} pixels"
        )),
    };
    [raster::answer(at_centre), raster::answer(at_sample)]
}

/// The stores at every sample of every pixel read (the module docs), pixel by pixel.
fn samples(kit: &mut Kit<'_>, target: &Target<'_>) -> Result<Vec<AtSample>, CheckError> {
    let slots = (WIDTH * SIDE * RasterCheck::SAMPLE_SLOTS) as usize;
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::fragcoord_depth_check_fragment(target.compile, WIDTH),
        "fragment stage",
    )?;
    let kernel = raster::shader_of(
        kit,
        metal2vulkan::bind::depth_samples_check_kernel(target.compile, WIDTH),
        "stored depths kernel",
    )?;
    let triangles: Vec<Triangle> = (0..REGIONS)
        .map(|r| {
            let region = ((r * SIDE) as f32, 0.0, SIDE as f32, SIDE as f32);
            Triangle::inside(DEPTHS[r as usize], region)
        })
        .collect();
    let vertices = raster::vertex_buffer(kit, &triangles)?;
    let stores = kit.host_buffer(
        &vec![UNWRITTEN; 2 * slots],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let params = kit.host_buffer(
        &[0f32.to_bits(), 1f32.to_bits()],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let stored = kit.host_buffer(
        &vec![UNWRITTEN; slots],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let pass = raster::pass(
        kit,
        (WIDTH, SIDE, SAMPLES),
        false,
        Some(DepthTarget {
            format: DEPTH,
            usage: vk::ImageUsageFlags::SAMPLED,
            final_layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            clear: 1.0,
        }),
    )?;
    let Some((_, depth_view)) = pass.depth else {
        return Err(CheckError("the pass has no depth".to_owned()));
    };
    let graphics_stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let (draw_layout, draw_set) = kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::VERTICES_BINDING,
            stages: graphics_stages,
            resource: SetResource::Buffer(vertices),
        },
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages: graphics_stages,
            resource: SetResource::Buffer(stores),
        },
        SetBinding {
            binding: RasterCheck::PARAMS_BINDING,
            stages: graphics_stages,
            resource: SetResource::Buffer(params),
        },
    ])?;
    let (read_layout, read_set) = kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::Buffer(stored),
        },
        SetBinding {
            binding: RasterCheck::IMAGE_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::SampledImage(
                depth_view,
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            ),
        },
    ])?;
    let draw_pipeline_layout = kit.layout(&[draw_layout], None)?;
    let read_pipeline_layout = kit.layout(&[read_layout], None)?;
    let draw = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: Some(fragment),
            layout: draw_pipeline_layout,
            depth: true,
            bias: true,
        },
    )?;
    let read = kit.compute_pipeline(kernel, read_pipeline_layout)?;
    let groups = SIDE / RasterCheck::DEPTH_SAMPLES_LOCAL_SIZE;
    kit.submit(|device, command| {
        // SAFETY: recording; every object named is live, the draws inside the pass.
        unsafe {
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
            for r in 0..REGIONS {
                let bias = if r == REGIONS - 1 { BIAS } else { 0.0 };
                device.cmd_set_depth_bias(command, bias, 0.0, 0.0);
                raster::draw(device, command, r, (r * SIDE, 0, SIDE, SIDE));
            }
            device.cmd_end_render_pass(command);
        }
    })?;
    // The stored depths read in a submission after the pass's (the module docs).
    kit.submit(|device, command| {
        // SAFETY: recording; every object named is live. The pass's dependency orders the
        // dispatch after its depth writes, of the submission before, which the kit waited for;
        // the host's reads follow the fragment stage's writes there and the dispatch's here.
        unsafe {
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, read);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::COMPUTE,
                read_pipeline_layout,
                0,
                &[read_set],
                &[],
            );
            device.cmd_dispatch(command, groups * REGIONS, groups, SAMPLES);
            raster::host_reads(
                device,
                command,
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
            );
        }
    })?;
    let stores = kit.read(&stores)?;
    let stored = kit.read(&stored)?;
    let mut out = Vec::new();
    let covered = (0..REGIONS).flat_map(|r| {
        raster::COVERED.flat_map(move |y| raster::COVERED.map(move |x| (r * SIDE + x, y)))
    });
    for (x, y) in covered {
        let pixel = (y * WIDTH + x) as usize;
        for s in 0..SAMPLES {
            let slot = pixel * RasterCheck::SAMPLE_SLOTS as usize + s as usize;
            out.push(AtSample {
                pixel: (x, y),
                sample: s,
                frag_z: stores[2 * slot],
                computed: stores[2 * slot + 1],
                stored: stored[slot],
            });
        }
    }
    Ok(out)
}
