//! The check of `DriverDeviation::MetalDepth16Stores` and `DriverDeviation::MetalDepth16BiasedStores`,
//! one program: a depth-only pipeline (no fragment stage) draws flat triangles into a `D16_UNORM`
//! depth tested `ALWAYS` and written, each scissored to its own pixel of a [`SIDE`]-square pass, at
//! the depths of the Metal probes behind the facts (`runtime_lane::unorm_depth_bias`): probe 7's
//! [`PROBE_7`] under its [`PROBE_7_FACTORS`] constants, probe 7b's [`PROBE_7B`] under
//! [`PROBE_7B_FACTORS`], and [`SLOPE_ALONE`]'s under a slope factor alone. Each draw's bias is
//! recorded as a render encoder's `setDepthBias:slopeScale:clamp:` is on the device
//! (`metal::state::depth_bias`, `render_state::DepthBiasRecord`).
//!
//! Metal's answer is Metal's code (`metal::state::depth16_code`): `MetalDepth16Stores` where every
//! unbiased draw (constant and slope 0) stores it, `MetalDepth16BiasedStores` where every biased
//! one does; otherwise the other answer.

use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, DepthTarget, Graphics, Triangle};
use super::Target;
use ash::vk;
use metal2vulkan::bind::RasterCheck;
use metal2vulkan::device::CheckAnswer;
use metal2vulkan::metal::state::{depth16_code, depth_bias};
use metal2vulkan::metal::PixelFormat;

/// The pass's side: a pixel per draw.
const SIDE: u32 = 32;

/// Probe 7's constants.
const PROBE_7_FACTORS: [f32; 27] = [
    -30000.0, -1000.0, -100.0, -8.0, -7.0, -4.0, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, -0.25, 0.0,
    0.25, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 7.0, 8.0, 100.0, 1000.0, 30000.0,
];

/// Probe 7's depths (their bits): 0, near the ends, at codes and between them.
const PROBE_7: [u32; 16] = [
    0x00000000, 0x37800080, 0x38000080, 0x3dcccccd, 0x3e800000, 0x3e800080, 0x3e99999a, 0x3effff00,
    0x3f000000, 0x3f000080, 0x3f3fffc0, 0x3f400000, 0x3f666666, 0x3f7d70a4, 0x3f7fff00, 0x3f800000,
];

/// Probe 7b's constants.
const PROBE_7B_FACTORS: [f32; 7] = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];

/// Probe 7b's depths (their bits): around the midpoints of codes 100 and 40000.
const PROBE_7B: [u32; 39] = [
    0x3ac90046, 0x3ac900bc, 0x3ac900c9, 0x3ac900ca, 0x3ac900cd, 0x3ac900d6, 0x3ac900e3, 0x3ac900fd,
    0x3ac90118, 0x3ac90132, 0x3ac9014c, 0x3ac90166, 0x3ac9019b, 0x3ac901cf, 0x3ac90252, 0x3ac902d5,
    0x3ac903db, 0x3ac904e2, 0x3ac906ee, 0x3ac908fa, 0x3ac90d13, 0x3ac9112b, 0x3ac91a63, 0x3ac920c9,
    0x3ac933fc, 0x3ac940c9, 0x3ac96730, 0x3f1c411c, 0x3f1c411d, 0x3f1c411e, 0x3f1c411f, 0x3f1c4120,
    0x3f1c4122, 0x3f1c4124, 0x3f1c4129, 0x3f1c412c, 0x3f1c4136, 0x3f1c413c, 0x3f1c414f,
];

/// Depths (their bits) under a slope factor alone, a constant of 0: a slope brings Metal's one
/// unit.
const SLOPE_ALONE: [(u32, f32); 28] = [
    (0x3ea474a5, 0.5),
    (0x3f2da090, 0.5),
    (0x3e85acd5, 0.5),
    (0x3f12f307, 0.5),
    (0x3f225df6, 1.0),
    (0x3f283eb5, 1.0),
    (0x3f0ff56e, 1.0),
    (0x3f1ec515, 1.0),
    (0x3f1040c8, 2.0),
    (0x3efe92f1, 2.0),
    (0x3ea76d9f, 2.0),
    (0x3f1f9cf1, 2.0),
    (0x3ea5c404, -1.0),
    (0x3f110eb8, -1.0),
    (0x3e8edc5e, -1.0),
    (0x3f3549b8, -1.0),
    (0x3ef3da1c, 100.0),
    (0x3f27c8a5, 100.0),
    (0x3ebc23a2, 100.0),
    (0x3ed43cd2, 100.0),
    (0x3e91cded, -100.0),
    (0x3efb372e, -100.0),
    (0x3f1bb82e, -100.0),
    (0x3f316b5a, -100.0),
    (0x3f14a090, 1000.0),
    (0x3ee65c02, 1000.0),
    (0x3f12edfd, 1000.0),
    (0x3ebc19cf, 1000.0),
];

/// One draw: its depth, constant and slope factor.
#[derive(Clone, Copy, Debug)]
struct Case {
    depth: f32,
    constant: f32,
    slope: f32,
}

impl Case {
    fn unbiased(&self) -> bool {
        self.constant == 0.0 && self.slope == 0.0
    }
}

/// Every draw of the check, in pixel order.
fn cases() -> Vec<Case> {
    let grid = |depths: &'static [u32], factors: &'static [f32]| {
        depths.iter().flat_map(move |&z| {
            factors.iter().map(move |&constant| Case {
                depth: f32::from_bits(z),
                constant,
                slope: 0.0,
            })
        })
    };
    grid(&PROBE_7, &PROBE_7_FACTORS)
        .chain(grid(&PROBE_7B, &PROBE_7B_FACTORS))
        .chain(SLOPE_ALONE.iter().map(|&(z, slope)| Case {
            depth: f32::from_bits(z),
            constant: 0.0,
            slope,
        }))
        .collect()
}

/// The check's answers on `target`, `[MetalDepth16Stores, MetalDepth16BiasedStores]`: not run
/// where `D16_UNORM` is no copyable depth attachment.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    let unanswered = |why: String| {
        vec![
            CheckAnswer::Unanswered(why.clone()),
            CheckAnswer::Unanswered(why),
        ]
    };
    let format = match kit.physical_query(|instance, physical| {
        // SAFETY: the physical device is the instance's.
        unsafe { instance.get_physical_device_format_properties(physical, vk::Format::D16_UNORM) }
    }) {
        Ok(format) => format,
        Err(error) => return unanswered(error.0),
    };
    let needed =
        vk::FormatFeatureFlags::DEPTH_STENCIL_ATTACHMENT | vk::FormatFeatureFlags::TRANSFER_SRC;
    if !format.optimal_tiling_features.contains(needed) {
        let why = "D16_UNORM is no copyable depth attachment".to_owned();
        return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)];
    }
    let cases = cases();
    match codes(kit, target, &cases) {
        Ok(codes) => {
            let verdict = |biased: bool| {
                let drawn: Vec<(&Case, u16)> = cases
                    .iter()
                    .zip(codes.iter().copied())
                    .filter(|(c, _)| c.unbiased() != biased)
                    .collect();
                let off: Vec<&(&Case, u16)> = drawn
                    .iter()
                    .filter(|(c, code)| *code != depth16_code(c.depth, c.constant, c.slope))
                    .collect();
                match off.first() {
                    None => Ok(()),
                    Some((c, code)) => Err(format!(
                        "{} of {} draws store another code than Metal's, the first {c:?} \
                         {code} for {}",
                        off.len(),
                        drawn.len(),
                        depth16_code(c.depth, c.constant, c.slope)
                    )),
                }
            };
            vec![
                raster::answer(verdict(false)),
                raster::answer(verdict(true)),
            ]
        }
        Err(error) => unanswered(error.0),
    }
}

/// The code each case's draw stores on the device.
fn codes(kit: &mut Kit<'_>, target: &Target<'_>, cases: &[Case]) -> Result<Vec<u16>, CheckError> {
    let profile = target.compile.profile();
    // The device's depth bias command, which each record carries: only called in the recording.
    let command = kit.depth_bias_command()?;
    let records = cases
        .iter()
        .map(|c| {
            let bias = depth_bias(
                metal2vulkan::vk::PrimitiveTopology::TRIANGLE_LIST,
                c.constant,
                c.slope,
                0.0,
                Some(PixelFormat::Depth16Unorm),
                profile,
            )
            .map_err(|e| CheckError(format!("its bias of {c:?} does not record: {e}")))?;
            crate::render_state::DepthBiasRecord::of(&bias, command)
                .map_err(|e| CheckError(format!("its bias of {c:?} does not record: {e}")))
        })
        .collect::<Result<Vec<_>, CheckError>>()?;
    let vertex = raster::vertex_shader(kit, target)?;
    let triangles: Vec<Triangle> = cases
        .iter()
        .map(|c| Triangle::covering([c.depth; 3], (0.0, 0.0, SIDE as f32, SIDE as f32)))
        .collect();
    let vertices = raster::vertex_buffer(kit, &triangles)?;
    let texels = (SIDE * SIDE) as usize;
    let codes = kit.host_buffer(&vec![0; texels / 2], vk::BufferUsageFlags::TRANSFER_DST)?;
    let pass = raster::pass(
        kit,
        (SIDE, SIDE, 1),
        false,
        Some(DepthTarget {
            format: vk::Format::D16_UNORM,
            usage: vk::ImageUsageFlags::TRANSFER_SRC,
            final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            clear: 1.0,
        }),
    )?;
    let Some((depth, _)) = pass.depth else {
        return Err(CheckError("the pass has no depth".to_owned()));
    };
    let (set_layout, set) = kit.descriptor_set(&[SetBinding {
        binding: RasterCheck::VERTICES_BINDING,
        stages: vk::ShaderStageFlags::VERTEX,
        resource: SetResource::Buffer(vertices),
    }])?;
    let layout = kit.layout(&[set_layout], None)?;
    let pipeline = raster::graphics(
        kit,
        &pass,
        Graphics {
            vertex,
            fragment: None,
            layout,
            depth: true,
            bias: true,
        },
    )?;
    let copy = vk::BufferImageCopy {
        buffer_offset: 0,
        buffer_row_length: 0,
        buffer_image_height: 0,
        image_subresource: vk::ImageSubresourceLayers {
            aspect_mask: vk::ImageAspectFlags::DEPTH,
            mip_level: 0,
            base_array_layer: 0,
            layer_count: 1,
        },
        image_offset: vk::Offset3D::default(),
        image_extent: vk::Extent3D {
            width: SIDE,
            height: SIDE,
            depth: 1,
        },
    };
    kit.submit(|device, command| {
        // SAFETY: recording; every object named is live, each bias recorded under the pipeline's
        // dynamic depth bias, and the copy after the pass, whose dependency orders it after the
        // depth writes and leaves the depth in `TRANSFER_SRC_OPTIMAL`.
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
            device.cmd_set_viewport(command, 0, &[raster::viewport(0, 0, SIDE, SIDE)]);
            for (k, record) in (0u32..).zip(&records) {
                device.cmd_set_scissor(command, 0, &[raster::rect(k % SIDE, k / SIDE, 1, 1)]);
                record.record(device, command);
                device.cmd_draw(command, 3, 1, 0, k);
            }
            device.cmd_end_render_pass(command);
            device.cmd_copy_image_to_buffer(
                command,
                depth,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                codes.buffer,
                &[copy],
            );
            raster::host_reads(device, command, vk::PipelineStageFlags::TRANSFER);
        }
    })?;
    Ok((kit.read(&codes)?)
        .into_iter()
        .flat_map(|w| [w as u16, (w >> 16) as u16])
        .take(cases.len())
        .collect())
}
