//! The check of `DriverFact::FragmentDepthStoresUnchanged` and
//! `DriverFact::FragmentDepthKeepsDepth16Codes`, settled as the device is made: each decides
//! whether the runtime copies a multisampled depth of its format by `FragDepth` draws or refuses
//! the copy, and whether it draws a `Min` or `Max` depth resolve the device has no mode for
//! (`depth_copy::drawn`, `resolve.rs`), each read as the device is assembled.
//!
//! One program per format, as the runtime's depth copy draws write a depth: a 4-sample pass of
//! the depth tested `ALWAYS` and written, sample shading on, cleared to [`CLEAR`] and drawn whole
//! by a fragment stage that writes a word a sample as its `FragDepth`
//! (`metal2vulkan::bind::depth_writes_check_fragment`), every sample then fetched as `float`
//! (`depth_samples_check_kernel`); then the fetched words written again into a second such pass
//! and fetched alike. Both in one submission, the words the stages read and write in
//! device-local buffers, copied in from the host by a submission of their own before it and out
//! to the host by one after it. Each fetch follows its pass behind a barrier of the pass's depth
//! writes before its reads, beside the pass's dependency: by the dependency alone, a kernel
//! recorded after a 4-sample pass on MoltenVK 1.4.1 on the M3 Max (macOS 27.0.1) read depths the
//! pass had not stored yet, where the submission wrote timestamps (as the kit's do), which left
//! this check's first pass with 2497 different codes of 65536.
//!
//! - `FragmentDepthStoresUnchanged`, a [`FLOAT_SIDE`]-square `D32_SFLOAT` depth whose samples
//!   cycle through [`PROBED`]: Metal's answer, as a texture's copy keeps it, where every sample
//!   of a value in `[0, 1]` (signed zero, the least and greatest denormal, the least normal, the
//!   greatest below 1, 1 and 0.2) is stored as written, and every sample of the rest (2, -1, a
//!   negative denormal, both infinities, NaNs with and without payloads) stored again as it was
//!   first stored. The defect's answer otherwise: MoltenVK's stores -0 as +0, NVIDIA's, RADV's
//!   and ANV's flush denormals. A NaN that writes nothing leaves [`CLEAR`], which it then stores
//!   alike.
//! - `FragmentDepthKeepsDepth16Codes`, a [`CODES_SIDE`]-square `D16_UNORM` depth whose 65536
//!   samples are written each code over 65535: Metal's answer where every sample's fetched `float`
//!   is stored again as the same code (each fetched again alike), the other where one is not. A
//!   first pass that stored fewer than every code answers neither.
//!
//! Neither runs where the device enabled no `sampleRateShading`, which the copy draws need; each
//! does not where the device takes no 4-sample sampled depth of its format.

use super::depth_draws::{self, DepthState, Stages};
use super::kit::{CheckError, Kit, SetBinding, SetResource};
use super::raster::{self, DepthTarget, Triangle, UNWRITTEN};
use super::Target;
use ash::vk;
use metal2vulkan::bind::{depth_samples_check_kernel, depth_writes_check_fragment, RasterCheck};
use metal2vulkan::device::{CheckAnswer, VkFeature};
use std::collections::BTreeSet;

/// The `float` bit patterns the 32-bit depth's samples cycle through: first those a depth texture
/// holds in `[0, 1]`, then the rest.
const PROBED: [u32; 16] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x007f_ffff,
    0x0080_0000,
    0x3f7f_ffff,
    0x3f80_0000,
    0x3e4c_cccd,
    0x4000_0000,
    0xbf80_0000,
    0x8000_0001,
    0x7f80_0000,
    0xff80_0000,
    0x7fc0_0000,
    0x7fc0_0001,
    0x7f80_0001,
];

/// How many of [`PROBED`] lie in `[0, 1]`, first.
const PROBED_IN_RANGE: usize = 8;

/// The 32-bit depth's side: every [`PROBED`] value at 16 samples.
const FLOAT_SIDE: u32 = 8;

/// The 16-bit depth's side: its samples hold each of the 65536 codes once.
const CODES_SIDE: u32 = 128;

/// The samples of each pass.
const SAMPLES: u32 = RasterCheck::SAMPLE_SLOTS;

/// The depth each pass clears to before its draw: 0.25.
const CLEAR: f32 = 0.25;

/// The check's answers on `target`, `[FragmentDepthStoresUnchanged,
/// FragmentDepthKeepsDepth16Codes]`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    if !(target.enabled.features).contains(&VkFeature::SampleRateShading) {
        let why = "sampleRateShading is not enabled".to_owned();
        return vec![CheckAnswer::NotRun(why.clone()), CheckAnswer::NotRun(why)];
    }
    vec![float_answer(kit, target), codes_answer(kit, target)]
}

/// The answer of `FragmentDepthStoresUnchanged` (the module docs).
fn float_answer(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let format = vk::Format::D32_SFLOAT;
    let given: Vec<u32> = (0..(FLOAT_SIDE * FLOAT_SIDE * SAMPLES) as usize)
        .map(|slot| PROBED[slot % PROBED.len()])
        .collect();
    let [once, twice] = match stored_twice(kit, target, format, FLOAT_SIDE, &given) {
        Ok(Some(stored)) => stored,
        Ok(None) => return CheckAnswer::NotRun("no 4-sample sampled D32_SFLOAT depth".to_owned()),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    let mut changed: Vec<(u32, u32, u32)> = Vec::new();
    for (slot, ((&value, &first), &second)) in given.iter().zip(&once).zip(&twice).enumerate() {
        if first == UNWRITTEN || second == UNWRITTEN {
            return CheckAnswer::Unanswered(format!("the fetch left slot {slot} unwritten"));
        }
        let kept = match slot % PROBED.len() < PROBED_IN_RANGE {
            true => first == value && second == first,
            false => second == first,
        };
        if !kept && !changed.iter().any(|&(v, ..)| v == value) {
            changed.push((value, first, second));
        }
    }
    match changed.is_empty() {
        true => CheckAnswer::Metals,
        false => CheckAnswer::NotMetals(format!(
            "the fragment depth changed {}",
            (changed.iter())
                .map(|(value, first, second)| format!(
                    "{value:#010x} (stored as {first:#010x}, then {second:#010x})"
                ))
                .collect::<Vec<_>>()
                .join(", ")
        )),
    }
}

/// The answer of `FragmentDepthKeepsDepth16Codes` (the module docs).
fn codes_answer(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let format = vk::Format::D16_UNORM;
    let codes = CODES_SIDE * CODES_SIDE * SAMPLES;
    let given: Vec<u32> = (0..codes)
        .map(|code| (code as f32 / 65535.0).to_bits())
        .collect();
    let [once, twice] = match stored_twice(kit, target, format, CODES_SIDE, &given) {
        Ok(Some(stored)) => stored,
        Ok(None) => return CheckAnswer::NotRun("no 4-sample sampled D16_UNORM depth".to_owned()),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    if let Some(slot) = (once.iter().chain(&twice)).position(|&word| word == UNWRITTEN) {
        let slot = slot % once.len();
        return CheckAnswer::Unanswered(format!("the fetch left slot {slot} unwritten"));
    }
    let changed: Vec<(usize, u32, u32)> = (once.iter().zip(&twice).enumerate())
        .filter(|(_, (first, second))| first != second)
        .map(|(slot, (&first, &second))| (slot, first, second))
        .collect();
    if let Some(&(slot, first, second)) = changed.first() {
        return CheckAnswer::NotMetals(format!(
            "{} of {codes} samples stored again as another code, the first written as code \
             {slot}: fetched as {first:#010x}, then {second:#010x}",
            changed.len()
        ));
    }
    let distinct = once.iter().collect::<BTreeSet<_>>().len();
    match distinct == codes as usize {
        true => CheckAnswer::Metals,
        false => CheckAnswer::Unanswered(format!(
            "the first pass stored {distinct} different codes of {codes}"
        )),
    }
}

/// The words each sample of a 4-sample `format` depth `side` pixels square fetches as after a
/// pass writes `given` (one a sample, slot `4 (y · side + x) + s`) as its fragment depths, then
/// after a second pass writes those words alike (the module docs); `None` where the device takes
/// no such depth.
fn stored_twice(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    format: vk::Format,
    side: u32,
    given: &[u32],
) -> Result<Option<[Vec<u32>; 2]>, CheckError> {
    let usage = vk::ImageUsageFlags::SAMPLED;
    if !depth_draws::takes_depth_of(kit, format, usage, SAMPLES)? {
        return Ok(None);
    }
    let vertex = raster::vertex_shader(kit, target)?;
    let fragment = raster::shader_of(
        kit,
        depth_writes_check_fragment(target.compile, side),
        "fragment stage",
    )?;
    let kernel = raster::shader_of(
        kit,
        depth_samples_check_kernel(target.compile, side),
        "stored depths kernel",
    )?;
    let covering = Triangle::covering([0.5; 3], (0.0, 0.0, side as f32, side as f32));
    let vertices = raster::vertex_buffer(kit, &[covering])?;
    // The stages' words in device memory, copied in from and out to the host's: a discrete
    // GPU's stages would read and write host memory across its bus.
    use vk::BufferUsageFlags as B;
    let count = given.len();
    let unwritten = kit.host_buffer(&vec![UNWRITTEN; count], B::TRANSFER_SRC)?;
    let given_in = kit.host_buffer(given, B::TRANSFER_SRC)?;
    let read_back = [
        kit.host_buffer(&vec![UNWRITTEN; count], B::TRANSFER_DST)?,
        kit.host_buffer(&vec![UNWRITTEN; count], B::TRANSFER_DST)?,
    ];
    let stored = B::STORAGE_BUFFER | B::TRANSFER_DST;
    let given = kit.device_buffer(count, stored)?;
    let once = kit.device_buffer(count, stored | B::TRANSFER_SRC)?;
    let twice = kit.device_buffer(count, stored | B::TRANSFER_SRC)?;
    let sampled = vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL;
    let depth = DepthTarget {
        format,
        usage,
        final_layout: sampled,
        clear: CLEAR,
    };
    let draw_stages = vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT;
    let mut steps = Vec::with_capacity(2);
    for (words, fetched) in [(given, once), (once, twice)] {
        let pass = raster::pass(kit, (side, side, SAMPLES), false, Some(depth))?;
        let Some((_, view)) = pass.depth else {
            return Err(CheckError("the pass has no depth".to_owned()));
        };
        let draw = kit.descriptor_set(&[
            SetBinding {
                binding: RasterCheck::VERTICES_BINDING,
                stages: draw_stages,
                resource: SetResource::Buffer(vertices),
            },
            SetBinding {
                binding: RasterCheck::PARAMS_BINDING,
                stages: draw_stages,
                resource: SetResource::DeviceBuffer(words),
            },
        ])?;
        let read = fetch_set(kit, SetResource::DeviceBuffer(fetched), view, sampled)?;
        steps.push((pass, draw, read));
    }
    let draw_layout = kit.layout(&[steps[0].1 .0], None)?;
    let read_layout = kit.layout(&[steps[0].2 .0], None)?;
    let stages = Stages {
        vertex,
        fragment: Some(fragment),
        layout: draw_layout,
    };
    let state = DepthState {
        compare: vk::CompareOp::ALWAYS,
        write: true,
        clamp: false,
    };
    let extent = (side, side, SAMPLES);
    let draw = depth_draws::pipeline(kit, stages, steps[0].0.pass, extent, state)?;
    let read = kit.compute_pipeline(kernel, read_layout)?;
    let groups = side / RasterCheck::DEPTH_SAMPLES_LOCAL_SIZE;
    let whole = [vk::BufferCopy {
        src_offset: 0,
        dst_offset: 0,
        size: (count * 4) as vk::DeviceSize,
    }];
    use vk::AccessFlags as A;
    use vk::PipelineStageFlags as S;
    let barrier = |src, dst| {
        vk::MemoryBarrier::default()
            .src_access_mask(src)
            .dst_access_mask(dst)
    };
    let no_flags = vk::DependencyFlags::empty();
    // The words in, waited for before the passes' submission.
    kit.submit(|device, command| {
        // SAFETY: recording; the buffers are the kit's and live, no submission pending.
        unsafe {
            device.cmd_copy_buffer(command, given_in.buffer, given, &whole);
            device.cmd_copy_buffer(command, unwritten.buffer, once, &whole);
            device.cmd_copy_buffer(command, unwritten.buffer, twice, &whole);
        }
    })?;
    kit.submit(|device, command| {
        let copied = barrier(A::TRANSFER_WRITE, A::SHADER_READ | A::SHADER_WRITE);
        let fetched = barrier(A::SHADER_WRITE, A::SHADER_READ);
        let stored = barrier(A::DEPTH_STENCIL_ATTACHMENT_WRITE, A::SHADER_READ);
        let shaders = S::FRAGMENT_SHADER | S::COMPUTE_SHADER;
        let tests = S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS;
        // SAFETY: recording; every object named is the kit's and live. The device buffers took
        // their words in the submission before, which the first barrier orders before the
        // stages' reads and writes. Both passes are alike, so the draw's pipeline is compatible
        // with each; each dispatch follows its pass, whose dependency orders it after the pass's
        // depth writes, as does the barrier after the pass, and the second pass's fragment reads
        // follow the first dispatch's writes behind the barrier.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                shaders,
                no_flags,
                &[copied],
                &[],
                &[],
            );
            for (step, (pass, (_, draw_set), (_, read_set))) in steps.iter().enumerate() {
                if step > 0 {
                    device.cmd_pipeline_barrier(
                        command,
                        vk::PipelineStageFlags::COMPUTE_SHADER,
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::DependencyFlags::empty(),
                        &[fetched],
                        &[],
                        &[],
                    );
                }
                device.cmd_begin_render_pass(command, &pass.begin(), vk::SubpassContents::INLINE);
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::GRAPHICS, draw);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::GRAPHICS,
                    draw_layout,
                    0,
                    &[*draw_set],
                    &[],
                );
                device.cmd_set_scissor(command, 0, &[raster::rect(0, 0, side, side)]);
                device.cmd_draw(command, 3, 1, 0, 0);
                device.cmd_end_render_pass(command);
                // The pass's dependency alone let MoltenVK 1.4.1's dispatch read the depth before
                // the pass stored it, where the submission wrote timestamps.
                device.cmd_pipeline_barrier(
                    command,
                    tests,
                    S::COMPUTE_SHADER,
                    no_flags,
                    &[stored],
                    &[],
                    &[],
                );
                device.cmd_bind_pipeline(command, vk::PipelineBindPoint::COMPUTE, read);
                device.cmd_bind_descriptor_sets(
                    command,
                    vk::PipelineBindPoint::COMPUTE,
                    read_layout,
                    0,
                    &[*read_set],
                    &[],
                );
                device.cmd_dispatch(command, groups, groups, SAMPLES);
            }
        }
    })?;
    // The words out, in a submission after the passes'.
    kit.submit(|device, command| {
        let done = barrier(A::SHADER_WRITE, A::TRANSFER_READ);
        // SAFETY: recording; the buffers are the kit's and live. The barrier orders the copies
        // after the dispatches' writes of the submission before, and the host's reads after the
        // copies.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                S::COMPUTE_SHADER,
                S::TRANSFER,
                no_flags,
                &[done],
                &[],
                &[],
            );
            device.cmd_copy_buffer(command, once, read_back[0].buffer, &whole);
            device.cmd_copy_buffer(command, twice, read_back[1].buffer, &whole);
            raster::host_reads(device, command, S::TRANSFER);
        }
    })?;
    Ok(Some([kit.read(&read_back[0])?, kit.read(&read_back[1])?]))
}

/// The set of the stored depths kernel over the sampled `view` in `layout`, its words into
/// `fetched` (a storage buffer), and the set's layout.
pub(super) fn fetch_set(
    kit: &mut Kit<'_>,
    fetched: SetResource,
    view: vk::ImageView,
    layout: vk::ImageLayout,
) -> Result<(vk::DescriptorSetLayout, vk::DescriptorSet), CheckError> {
    kit.descriptor_set(&[
        SetBinding {
            binding: RasterCheck::RESULTS_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: fetched,
        },
        SetBinding {
            binding: RasterCheck::IMAGE_BINDING,
            stages: vk::ShaderStageFlags::COMPUTE,
            resource: SetResource::SampledImage(view, layout),
        },
    ])
}
