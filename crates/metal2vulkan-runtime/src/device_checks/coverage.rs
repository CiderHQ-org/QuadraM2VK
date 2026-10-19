//! The check of `DriverWorkaround::MemoryWritingFragmentsRunPerSample` and
//! `DriverWorkaround::ClippedTrianglesSplitPixelFragments`, one program: a fragment stage that
//! reads `SampleMask` and writes memory, without sample shading
//! (`metal2vulkan::bind::coverage_count_check_fragment`), counts its invocations at each pixel and
//! ORs and ANDs their masks, in [`SIDE`]-square regions of a pass:
//!
//! - **Per-sample runs**, at 4 samples and at 2 where the device renders 2: a triangle wholly
//!   inside the clip volume, so the device clips nothing. At a pixel it covers whole (its masks'
//!   OR is every sample), one invocation whose mask is every sample is a device that runs the
//!   stage once per pixel, as Vulkan's rasterization does without sample shading: no defect,
//!   Metal's answer through translation's loop over the covered samples. One invocation per
//!   sample, each reading every sample, is the defect (Metal's own count, which translation then
//!   leaves to the device). Anything else answers neither.
//! - **Split triangles**, at 4 samples: the triangle over the viewport, clip `(-1, -1)`, `(3, -1)`
//!   and `(-1, 3)`, which the device clips to the viewport, and one reaching [`FAR`] past it,
//!   which every device clips. Every invocation reading every sample of its pixel is no defect;
//!   an invocation of a pixel the triangle covers whole reading only some (the pixel run once per
//!   piece of the clipped triangle) is the defect. A pixel left uncovered answers neither. Not
//!   drawn where the device holds no `DriverWorkaround::ClippedCoverageNotMetals`, whose rasterizer
//!   splits a clipped triangle as Apple's does (MoltenVK on an Apple GPU draws through Metal): its
//!   pieces are Metal's own. Metal itself runs 29 pixels of the two triangles once per piece, each
//!   with part of its mask (a Metal program of this draw on the M3 Max, macOS 27.0.1), as MoltenVK
//!   1.4.1 does there.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Graphics, Triangle, COLOR, COLOR_USAGE};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::{CheckAnswer, DriverWorkaround, VkFeature};

/// A region's side, in pixels.
const SIDE: u32 = 16;

/// The regions of a pass, side by side: inside the clip volume, over the viewport, far past it.
const REGIONS: u32 = 3;

/// The target's width.
const WIDTH: u32 = SIDE * REGIONS;

/// How far past the viewport the third triangle reaches, in clip units: beyond every device's
/// guard band.
const FAR: f32 = 16384.0;

/// The words a pixel holds: the count, the masks' OR and their AND.
const WORDS: usize = 3;

/// The fewest pixels a triangle must cover whole for its count to be read.
const FEWEST: usize = 16;

/// What one pixel's invocations left.
#[derive(Clone, Copy, Debug)]
struct Pixel {
    count: u32,
    or: u32,
    and: u32,
}

/// How a device ran the stage at the pixels a triangle covers whole.
#[derive(Debug, PartialEq, Eq)]
enum Runs {
    PerPixel,
    PerSample,
    Neither(String),
}

/// The check's answers on `target`, `[MemoryWritingFragmentsRunPerSample,
/// ClippedTrianglesSplitPixelFragments]`: not run where the device lacks a fragment stage's
/// storage writes or 4-sample color, the second not where the device's rasterizer splits a clipped
/// triangle as Apple's does (the module docs).
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    if !target
        .enabled
        .features
        .contains(&VkFeature::FragmentStoresAndAtomics)
    {
        let why = "fragmentStoresAndAtomics is not enabled".to_owned();
        return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)];
    }
    let unanswered = |why: String| {
        vec![
            CheckAnswer::Unanswered(why.clone()),
            CheckAnswer::Unanswered(why),
        ]
    };
    match raster::takes_samples(kit, COLOR, COLOR_USAGE, 4) {
        Ok(true) => {}
        Ok(false) => {
            let why = "no 4-sample color".to_owned();
            return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)];
        }
        Err(error) => return unanswered(error.0),
    }
    // The device's clipped coverage is Metal's (the module docs).
    let metals_pieces = !(target.compile.profile().workarounds())
        .contains(&DriverWorkaround::ClippedCoverageNotMetals);
    let drawn = if metals_pieces { 1 } else { REGIONS };
    let four = match pixels(kit, target, 4, drawn) {
        Ok(pixels) => pixels,
        Err(error) => return unanswered(error.0),
    };
    let mut passes = vec![(4, runs(&four, 4))];
    match raster::takes_samples(kit, COLOR, COLOR_USAGE, 2) {
        Ok(true) => match pixels(kit, target, 2, 1) {
            Ok(two) => passes.push((2, runs(&two, 2))),
            Err(error) => return unanswered(error.0),
        },
        Ok(false) => {}
        Err(error) => return unanswered(error.0),
    }
    let clipped = match metals_pieces {
        true => CheckAnswer::NotRun(
            "the device's rasterizer splits a clipped triangle as Apple's does (it holds no \
             ClippedCoverageNotMetals), so its pieces are Metal's own"
                .to_owned(),
        ),
        false => split(&four),
    };
    vec![per_sample(&passes), clipped]
}

/// How the stage ran, in a pass of `samples` samples, at the pixels of the first region its
/// triangle covers whole.
fn runs(pixels: &[Pixel], samples: u32) -> Runs {
    let every = (1u32 << samples) - 1;
    let whole: Vec<Pixel> = region(pixels, 0)
        .filter(|p| p.or & every == every)
        .collect();
    if whole.len() < FEWEST {
        return Runs::Neither(format!("the triangle covered {} pixels whole", whole.len()));
    }
    let all = |count: u32| {
        whole
            .iter()
            .all(|p| p.count == count && p.and & every == every)
    };
    match () {
        () if all(1) => Runs::PerPixel,
        () if all(samples) => Runs::PerSample,
        () => Runs::Neither(format!("the pixels covered whole ran {whole:?}")),
    }
}

/// The pixels of region `r`.
fn region(pixels: &[Pixel], r: u32) -> impl Iterator<Item = Pixel> + '_ {
    (0..SIDE * SIDE).map(move |i| {
        let (x, y) = (r * SIDE + i % SIDE, i / SIDE);
        pixels[(y * WIDTH + x) as usize]
    })
}

/// `MemoryWritingFragmentsRunPerSample`'s answer of every pass's runs (the module docs): Metal's
/// answer is no defect.
fn per_sample(runs: &[(u32, Runs)]) -> CheckAnswer {
    if let Some((samples, Runs::Neither(why))) =
        runs.iter().find(|(_, r)| matches!(r, Runs::Neither(_)))
    {
        return CheckAnswer::Unanswered(format!("at {samples} samples {why}"));
    }
    match () {
        () if runs.iter().all(|(_, r)| *r == Runs::PerPixel) => raster::answer(Ok(())),
        () if runs.iter().all(|(_, r)| *r == Runs::PerSample) => raster::answer(Err(
            "the stage ran once per covered sample, each invocation reading every sample"
                .to_owned(),
        )),
        () => CheckAnswer::Unanswered(format!("the passes ran {runs:?}")),
    }
}

/// `ClippedTrianglesSplitPixelFragments`' answer of the 4-sample pass (the module docs): Metal's
/// answer is no defect.
fn split(pixels: &[Pixel]) -> CheckAnswer {
    let every = 0xf;
    let mut split = 0;
    for r in 1..REGIONS {
        for p in region(pixels, r) {
            if p.count == 0 || p.or & every != every {
                return CheckAnswer::Unanswered(format!(
                    "region {r} left a pixel uncovered: {p:?}"
                ));
            }
            split += usize::from(p.and & every != every);
        }
    }
    raster::answer(match split {
        0 => Ok(()),
        n => Err(format!(
            "{n} pixels the clipped triangles cover whole ran with part of their mask"
        )),
    })
}

/// The pixels a pass of `samples` samples leaves, its first `drawn` regions drawn.
fn pixels(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    samples: u32,
    drawn: u32,
) -> Result<Vec<Pixel>, CheckError> {
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::coverage_count_check_fragment(target.compile, WIDTH),
        "fragment stage",
    )?;
    let triangles = [
        Triangle::at([[-0.75, -0.75], [0.75, -0.75], [-0.75, 0.75]]),
        Triangle::at([[-1.0, -1.0], [3.0, -1.0], [-1.0, 3.0]]),
        Triangle::at([[-1.0, -1.0], [FAR, -1.0], [-1.0, FAR]]),
    ];
    let vertices = raster::vertex_buffer(kit, &triangles)?;
    let start: Vec<u32> = (0..WIDTH * SIDE).flat_map(|_| [0, 0, u32::MAX]).collect();
    let counts = kit.host_buffer(&start, vk::BufferUsageFlags::STORAGE_BUFFER)?;
    let pass = raster::pass(kit, (WIDTH, SIDE, samples), true, None)?;
    let stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let (set_layout, set) = kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::VERTICES_BINDING,
            stages,
            resource: SetResource::Buffer(vertices),
        },
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages,
            resource: SetResource::Buffer(counts),
        },
    ])?;
    let layout = kit.layout(&[set_layout], None)?;
    let pipeline = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: Some(fragment),
            layout,
            depth: false,
            bias: false,
        },
    )?;
    kit.submit(|device, command| {
        // SAFETY: recording; every object named is live.
        unsafe {
            device.cmd_begin_render_pass(command, &pass.begin(), vk::SubpassContents::INLINE);
            device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, pipeline);
            device.cmd_bind_descriptor_sets(
                command,
                vk::PipelineBindPoint::GRAPHICS,
                layout,
                0,
                &[set],
                &[],
            );
            for r in 0..drawn {
                raster::draw(device, command, r, (r * SIDE, 0, SIDE, SIDE));
            }
            device.cmd_end_render_pass(command);
            raster::host_reads(device, command, vk::PipelineStageFlags::FRAGMENT_SHADER);
        }
    })?;
    Ok((kit.read(&counts)?)
        .chunks(WORDS)
        .map(|w| Pixel {
            count: w[0],
            or: w[1],
            and: w[2],
        })
        .collect())
}
