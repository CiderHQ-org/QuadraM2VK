//! The check of `DriverFact::DepthTestKeepsDenormals`, settled as the device is made: it decides
//! how a 32-bit float depth's writes are announced and whether a render target's denormal depth
//! switches its magnitude stand-in on (`image_plan.rs`, `texture/signed_depth.rs`), each read as
//! the device is assembled. Each pixel of a 1-pixel-high `D32_SFLOAT` target of the values posed
//! ([`STORED`]), copied in from a buffer (`vkCmdCopyBufferToImage`) and read back the same way, is
//! depth-tested by a draw of a triangle at depth 0 under an occlusion query, depth writes off,
//! its compare the pixel's: `Less` against a stored positive denormal and the smallest positive
//! normal, `NotEqual` against a stored negative denormal and the smallest negative normal.
//!
//! Vulkan's test compares the fragment's depth `zf` with the stored `za` as they are ("Depth
//! Comparison": `zf` and `za` are its reference and test values); the clamp of "Depth Clamping and
//! Range Adjustment" applies to `zf` alone, here 0 inside the viewport's `[0, 1]`. So each normal's
//! test passes whatever the sign of what is stored, and its control holds on every device: under
//! Vulkan's comparison, and under Metal's, which compares a stored value by its magnitude
//! (`DriverFact::DepthTestComparesMagnitudes`, MoltenVK's by construction), where a `Greater`
//! against a stored negative value fails. Metal's answer: the test passes against each one, a
//! denormal compared as a value other than zero (a negative one by its magnitude). The defect's
//! answer: a test that fails against a denormal, taking it as zero (NVIDIA's, ANV's and lavapipe's
//! of a positive one).
//!
//! A negative value lies outside `[0, 1]`, which a copy into a depth aspect may hold only where
//! the device enables `VK_EXT_depth_range_unrestricted` (`VUID-vkCmdCopyBufferToImage-pRegions-07931`:
//! without it the buffer's data must lie in `[0, 1]`). Elsewhere the negative half is not posed: a
//! positive denormal compared otherwise still answers the defect's, and Metal's answer is not
//! run. A stored value read back otherwise (the device's copy flushed it, RADV's and ANV's) is not
//! run either: the test of the value cannot be posed. A normal's outcome otherwise answers
//! neither: the check went wrong.

use super::depth_draws::{self, DepthState, Stages, DEPTH};
use super::kit::{CheckError, Kit};
use super::raster::{self, Triangle};
use super::Target;
use ash::vk;
use metal2vulkan::device::{CheckAnswer, VkExtension};

/// The stored values, each with the compare that tests it and whether it is negative, a value
/// outside `[0, 1]` (the module docs): the least and greatest denormal of each sign, then the least
/// normal of each sign. Metal's test passes against each.
const STORED: [(u32, vk::CompareOp, bool); 6] = [
    (0x0000_0001, vk::CompareOp::LESS, false),
    (0x007F_FFFF, vk::CompareOp::LESS, false),
    (0x8000_0001, vk::CompareOp::NOT_EQUAL, true),
    (0x807F_FFFF, vk::CompareOp::NOT_EQUAL, true),
    (0x0080_0000, vk::CompareOp::LESS, false),
    (0x8080_0000, vk::CompareOp::NOT_EQUAL, true),
];

/// Whether `value` is a denormal: the rest of [`STORED`] are the normals every test compares
/// alike.
fn denormal(value: u32) -> bool {
    value & 0x7F80_0000 == 0
}

/// What the read-back holds before the copy.
const UNWRITTEN: u32 = 0x7FC0_DEAD;

/// The check's answer on `target`.
pub(super) fn run(kit: &mut Kit<'_>, target: &Target<'_>) -> Vec<CheckAnswer> {
    vec![run_one(kit, target)]
}

/// The check's one answer.
fn run_one(kit: &mut Kit<'_>, target: &Target<'_>) -> CheckAnswer {
    let usage = vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::TRANSFER_DST;
    match depth_draws::takes_depth(kit, usage) {
        Ok(true) => {}
        Ok(false) => return CheckAnswer::NotRun("no D32_SFLOAT depth target".to_owned()),
        Err(error) => return CheckAnswer::Unanswered(error.0),
    }
    let unrestricted =
        (target.enabled.extensions).contains(&VkExtension::ExtDepthRangeUnrestricted);
    let posed: Vec<(u32, vk::CompareOp)> = (STORED.iter())
        .filter(|&&(.., negative)| unrestricted || !negative)
        .map(|&(value, compare, _)| (value, compare))
        .collect();
    let (stored, passed) = match tested(kit, target, &posed) {
        Ok(found) => found,
        Err(error) => return CheckAnswer::Unanswered(error.0),
    };
    for (&(value, _), &read) in posed.iter().zip(&stored) {
        if read != value {
            return CheckAnswer::NotRun(format!(
                "the depth target stored {read:#010x} for {value:#010x}: the device's copy does \
                 not keep the value its test would compare"
            ));
        }
    }
    let mut other = Vec::new();
    for (&(value, compare), &pass) in posed.iter().zip(&passed) {
        if pass {
            continue;
        }
        let compare = match compare {
            vk::CompareOp::LESS => "Less",
            _ => "NotEqual",
        };
        let what = format!("{compare} against {value:#010x} failed");
        if !denormal(value) {
            return CheckAnswer::Unanswered(format!("the test of a normal {what}"));
        }
        other.push(what);
    }
    match (other.is_empty(), unrestricted) {
        (false, _) => CheckAnswer::NotMetals(format!("a test at depth 0 {}", other.join(", "))),
        (true, true) => CheckAnswer::Metals,
        (true, false) => CheckAnswer::NotRun(
            "a negative denormal's test needs VK_EXT_depth_range_unrestricted, without which a \
             copy into a depth aspect may hold no value outside [0, 1] \
             (VUID-vkCmdCopyBufferToImage-pRegions-07931)"
                .to_owned(),
        ),
    }
}

/// The values `posed` the target holds, as read back, and whether each pixel's test, by its
/// compare, passed.
fn tested(
    kit: &mut Kit<'_>,
    target: &Target<'_>,
    posed: &[(u32, vk::CompareOp)],
) -> Result<(Vec<u32>, Vec<bool>), CheckError> {
    // One pixel per value posed.
    let width = posed.len() as u32;
    let vertex = raster::vertex_shader(kit, target)?;
    let flat = Triangle::covering([0.0; 3], (0.0, 0.0, width as f32, 1.0));
    let vertices = raster::vertex_buffer(kit, &[flat])?;
    let (set, layout) = depth_draws::vertex_set(kit, vertices)?;
    let usage = vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
        | vk::ImageUsageFlags::TRANSFER_SRC
        | vk::ImageUsageFlags::TRANSFER_DST;
    let aspect = vk::ImageAspectFlags::DEPTH;
    let (image, view) = raster::image(kit, DEPTH, usage, (width, 1, 1), aspect)?;
    let attachment = vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL;
    let attachments = [vk::AttachmentDescription2::default()
        .format(DEPTH)
        .samples(vk::SampleCountFlags::TYPE_1)
        .load_op(vk::AttachmentLoadOp::LOAD)
        .store_op(vk::AttachmentStoreOp::STORE)
        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
        .initial_layout(attachment)
        .final_layout(attachment)];
    let depth_ref = vk::AttachmentReference2::default()
        .attachment(0)
        .layout(attachment)
        .aspect_mask(aspect);
    let subpasses = [vk::SubpassDescription2::default()
        .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
        .depth_stencil_attachment(&depth_ref)];
    let pass = kit.render_pass(
        &vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses),
    )?;
    let views = [view];
    let framebuffer = kit.framebuffer(
        &vk::FramebufferCreateInfo::default()
            .render_pass(pass)
            .attachments(&views)
            .width(width)
            .height(1)
            .layers(1),
    )?;
    // A pipeline per compare posed.
    let mut pipelines = Vec::with_capacity(2);
    for &(_, compare) in posed {
        if pipelines.iter().any(|&(made, _)| made == compare) {
            continue;
        }
        let state = DepthState {
            compare,
            write: false,
            clamp: false,
        };
        let stages = Stages {
            vertex,
            fragment: None,
            layout,
        };
        let made = depth_draws::pipeline(kit, stages, pass, (width, 1, 1), state)?;
        pipelines.push((compare, made));
    }
    let values: Vec<u32> = posed.iter().map(|&(value, _)| value).collect();
    let upload = kit.host_buffer(&values, vk::BufferUsageFlags::TRANSFER_SRC)?;
    let unwritten = vec![UNWRITTEN; posed.len()];
    let read_back = kit.host_buffer(&unwritten, vk::BufferUsageFlags::TRANSFER_DST)?;
    let queries = kit.query_pool(
        &vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::OCCLUSION)
            .query_count(width),
    )?;
    kit.submit(|device, command| {
        use vk::AccessFlags as A;
        use vk::ImageLayout as L;
        use vk::PipelineStageFlags as S;
        let barrier = |old, new, src, dst| {
            vk::ImageMemoryBarrier::default()
                .src_access_mask(src)
                .dst_access_mask(dst)
                .old_layout(old)
                .new_layout(new)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .image(image)
                .subresource_range(raster::subresources(aspect))
        };
        let fresh = barrier(
            L::UNDEFINED,
            L::TRANSFER_DST_OPTIMAL,
            A::empty(),
            A::TRANSFER_WRITE,
        );
        let copied = barrier(
            L::TRANSFER_DST_OPTIMAL,
            L::TRANSFER_SRC_OPTIMAL,
            A::TRANSFER_WRITE,
            A::TRANSFER_READ,
        );
        let tests = A::DEPTH_STENCIL_ATTACHMENT_READ | A::DEPTH_STENCIL_ATTACHMENT_WRITE;
        let to_test = barrier(
            L::TRANSFER_SRC_OPTIMAL,
            attachment,
            A::TRANSFER_WRITE,
            tests,
        );
        let fragment_tests = S::EARLY_FRAGMENT_TESTS | S::LATE_FRAGMENT_TESTS;
        let copy = depth_draws::depth_copy(width, 1);
        let written = vk::MemoryBarrier::default()
            .src_access_mask(A::TRANSFER_WRITE)
            .dst_access_mask(A::HOST_READ);
        let no_flags = vk::DependencyFlags::empty();
        let begin = vk::RenderPassBeginInfo::default()
            .render_pass(pass)
            .framebuffer(framebuffer)
            .render_area(raster::rect(0, 0, width, 1));
        let graphics = vk::PipelineBindPoint::GRAPHICS;
        // SAFETY: recording; the image moves through the layouts its barriers name, the pass
        // loads it in the attachment layout it is left in, every pipeline is of the pass and of
        // `layout`, whose set 0 is `set`, and the query pool holds a query per pixel, reset
        // before any begins.
        unsafe {
            device.cmd_pipeline_barrier(
                command,
                S::TOP_OF_PIPE,
                S::TRANSFER,
                no_flags,
                &[],
                &[],
                &[fresh],
            );
            let dst = L::TRANSFER_DST_OPTIMAL;
            device.cmd_copy_buffer_to_image(command, upload.buffer, image, dst, &[copy]);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                S::TRANSFER,
                no_flags,
                &[],
                &[],
                &[copied],
            );
            let src = L::TRANSFER_SRC_OPTIMAL;
            device.cmd_copy_image_to_buffer(command, image, src, read_back.buffer, &[copy]);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                fragment_tests,
                no_flags,
                &[],
                &[],
                &[to_test],
            );
            device.cmd_reset_query_pool(command, queries, 0, width);
            device.cmd_begin_render_pass(command, &begin, vk::SubpassContents::INLINE);
            device.cmd_bind_descriptor_sets(command, graphics, layout, 0, &[set], &[]);
            for (x, &(_, compare)) in (0u32..).zip(posed) {
                let Some(&(_, pipeline)) = pipelines.iter().find(|(c, _)| *c == compare) else {
                    continue;
                };
                device.cmd_bind_pipeline(command, graphics, pipeline);
                device.cmd_set_scissor(command, 0, &[raster::rect(x, 0, 1, 1)]);
                device.cmd_begin_query(command, queries, x, vk::QueryControlFlags::empty());
                device.cmd_draw(command, 3, 1, 0, 0);
                device.cmd_end_query(command, queries, x);
            }
            device.cmd_end_render_pass(command);
            device.cmd_pipeline_barrier(
                command,
                S::TRANSFER,
                S::HOST,
                no_flags,
                &[written],
                &[],
                &[],
            );
        }
    })?;
    let stored = kit.read(&read_back)?;
    let counted = kit.query_results(queries, posed.len())?;
    Ok((stored, counted.iter().map(|&n| n != 0).collect()))
}
