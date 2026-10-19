//! The check of `DriverDeviation::MetalCenterVaryings`: at 4 samples, and at 2 where the device
//! renders 2, one triangle wholly inside the clip volume of a [`SIDE`]-square target, so the device
//! clips it into no pieces (`raster::Triangle::inside`), through a fragment stage that runs per
//! sample (`metal2vulkan::bind::center_varyings_check_fragment`) and stores, at each sample, an
//! attribute read as it is, smooth and without perspective, whose value at a point is the point's
//! window `x` and `y`.
//!
//! Metal's answer, reading such a varying at the pixel centre in every invocation, a per-sample
//! one's included: at every pixel read, those whose every sample the triangle covers
//! (`raster::COVERED`), each of the four values is one value at every sample, within
//! [`TOLERANCE`] of the pixel centre's coordinate. The other answer: a value differs among the
//! samples of a pixel (the device reads it at each sample). Anything else, or a sample left
//! unwritten, answers neither.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, Graphics, Triangle, COLOR, COLOR_USAGE, UNWRITTEN};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::{CheckAnswer, VkFeature};

/// The target's side, in pixels.
const SIDE: u32 = raster::CELL;

/// The words a sample stores: smooth `x`, `y`, then no-perspective `x`, `y`.
const WORDS: usize = 4;

/// The farthest a centre value may lie from the pixel centre's coordinate: well under the nearest
/// sample position's distance from it in a 2- or 4-sample pattern (1/8).
const TOLERANCE: f32 = 1.0 / 64.0;

/// How one pass's values read.
#[derive(Debug, PartialEq, Eq)]
enum Read {
    Centre,
    /// A value differs among the samples of this many pixels.
    PerSample(usize),
    Neither(String),
}

/// The check's answer on `target`: not run where the device lacks a per-sample fragment stage or
/// its storage writes.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let features = &target.enabled.features;
    for feature in [
        VkFeature::SampleRateShading,
        VkFeature::FragmentStoresAndAtomics,
    ] {
        if !features.contains(&feature) {
            return vec![CheckAnswer::NotRun(format!("{feature:?} is not enabled"))];
        }
    }
    let mut reads = Vec::new();
    for samples in [4, 2] {
        match raster::takes_samples(kit, COLOR, COLOR_USAGE, samples) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(error) => return vec![CheckAnswer::Unanswered(error.0)],
        }
        match values(kit, target, samples) {
            Ok(values) => reads.push((samples, read(&values, samples))),
            Err(error) => return vec![CheckAnswer::Unanswered(error.0)],
        }
    }
    vec![answer(&reads)]
}

/// The answer of every pass's reads (the module docs).
fn answer(reads: &[(u32, Read)]) -> CheckAnswer {
    if reads.is_empty() {
        return CheckAnswer::Unanswered("no pass of 2 or 4 samples ran".to_owned());
    }
    if let Some((samples, Read::Neither(why))) =
        reads.iter().find(|(_, r)| matches!(r, Read::Neither(_)))
    {
        return CheckAnswer::Unanswered(format!("at {samples} samples {why}"));
    }
    let per_sample = reads.iter().find_map(|(samples, r)| match r {
        Read::PerSample(pixels) => Some(format!(
            "at {samples} samples a varying differs among the samples of {pixels} pixels"
        )),
        _ => None,
    });
    raster::answer(per_sample.map_or(Ok(()), Err))
}

/// How the values of a pass of `samples` samples read (the module docs).
fn read(values: &[u32], samples: u32) -> Read {
    let mut per_sample = 0;
    let covered = raster::COVERED.flat_map(|y| raster::COVERED.map(move |x| (x, y)));
    for (x, y) in covered {
        let pixel = (y * SIDE + x) as usize;
        let (x, y) = (x as f32, y as f32);
        let at = |s: usize, word: usize| {
            values[(pixel * RasterCheck::SAMPLE_SLOTS as usize + s) * WORDS + word]
        };
        let mut varies = false;
        for word in 0..WORDS {
            let first = at(0, word);
            if (0..samples as usize).any(|s| at(s, word) == UNWRITTEN) {
                return Read::Neither(format!("pixel {pixel} has a sample left unwritten"));
            }
            if (0..samples as usize).any(|s| at(s, word) != first) {
                varies = true;
                continue;
            }
            let centre = if word % 2 == 0 { x } else { y } + 0.5;
            let value = f32::from_bits(first);
            if value.is_nan() || (value - centre).abs() > TOLERANCE {
                return Read::Neither(format!(
                    "pixel {pixel} reads {value} at every sample, its centre {centre}"
                ));
            }
        }
        per_sample += usize::from(varies);
    }
    match per_sample {
        0 => Read::Centre,
        pixels => Read::PerSample(pixels),
    }
}

/// The values a pass of `samples` samples leaves, slot by slot.
fn values(kit: &mut Kit<'_>, target: &Target<'_>, samples: u32) -> Result<Vec<u32>, CheckError> {
    let slots = (SIDE * SIDE * RasterCheck::SAMPLE_SLOTS) as usize;
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        metal2vulkan::bind::center_varyings_check_fragment(target.compile, SIDE),
        "fragment stage",
    )?;
    let whole = (0.0, 0.0, SIDE as f32, SIDE as f32);
    let vertices = raster::vertex_buffer(kit, &[Triangle::inside([0.5; 3], whole)])?;
    let stores = kit.host_buffer(
        &vec![UNWRITTEN; WORDS * slots],
        vk::BufferUsageFlags::STORAGE_BUFFER,
    )?;
    let pass = raster::pass(kit, (SIDE, SIDE, samples), true, None)?;
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
            resource: SetResource::Buffer(stores),
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
            raster::draw(device, command, 0, (0, 0, SIDE, SIDE));
            device.cmd_end_render_pass(command);
            raster::host_reads(device, command, vk::PipelineStageFlags::FRAGMENT_SHADER);
        }
    })?;
    kit.read(&stores)
}
