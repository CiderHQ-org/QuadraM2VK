//! Metal's stop of a render pipeline state whose attachment is in a pixel format the device's
//! Metal describes none of.
//!
//! - **What Metal stops** (an Apple M4 under macOS 27.0.1, plain).
//!   `newRenderPipelineStateWithDescriptor:`'s descriptor validation asserts where a color
//!   attachment's pixel format is one the release describes none of (the three-channel formats
//!   and 646 to 650: "pixelFormat, for color render target(0), is not a valid MTLPixelFormat"),
//!   blended or not, and where the depth or stencil attachment's is `Depth24Unorm_Stencil8` or
//!   `X24_Stencil8`, which an Apple GPU has neither of ("depthAttachmentPixelFormat is not a
//!   valid MTLPixelFormat", "stencilAttachmentPixelFormat ..."). Its texture validation stops a
//!   texture of each alike (`TextureProblem::NotAFormat`).
//!   Plain Metal under macOS 26.7 on an Apple M3 Max stops alike every format 26.7 describes
//!   none of, 630 to 650 among them, in every mode (the consumer's run of the same probe).
//! - **The answer.** [`Error::Invalid`] with [`Problem::AttachmentNotAFormat`], naming the first
//!   such attachment (the colors by index, then depth, then stencil), before anything else the
//!   creation checks; `new_mesh_render_pipeline_state` alike. The device's formats are its
//!   Metal level's ([`MetalLevel::describes`](crate::MetalLevel::describes)), which leaves out
//!   the 24-bit depth formats at every level, on every device.
//! - **Not checked here.** A tile pipeline's color attachments (not measured).

use crate::error::{invalid, Error, Problem};
use crate::render_pass::Attachment;
use crate::Device;
use metal2vulkan::metal::PixelFormat;
use metal2vulkan::pipeline::ColorAttachmentDescriptor;

/// Stopped in `call` (the module docs): the first of the color attachments `color`, the depth
/// attachment's `depth` and the stencil attachment's `stencil` in a format `device`'s Metal
/// describes none of. `Invalid` is no attachment.
pub(super) fn check(
    device: &Device,
    call: &'static str,
    color: &[Option<ColorAttachmentDescriptor>],
    depth: Option<PixelFormat>,
    stencil: Option<PixelFormat>,
) -> Result<(), Error> {
    let level = device.properties().metal_level;
    let colors = color.iter().enumerate().filter_map(|(index, attachment)| {
        let format = attachment.as_ref()?.pixel_format;
        Some((Attachment::Color(index as u32), format))
    });
    let attachments = colors
        .chain(depth.map(|format| (Attachment::Depth, format)))
        .chain(stencil.map(|format| (Attachment::Stencil, format)));
    for (attachment, format) in attachments {
        if format != PixelFormat::Invalid && !level.describes(format) {
            return Err(invalid(
                call,
                Problem::AttachmentNotAFormat { attachment, format },
            ));
        }
    }
    Ok(())
}
