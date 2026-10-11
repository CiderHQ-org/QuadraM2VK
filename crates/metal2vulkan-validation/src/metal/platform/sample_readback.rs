//! The Metal oracle's per-sample readback of a multisampled attachment
//! (`executor_contract::multisample`).
//!
//! Metal copies no multisampled texture to a buffer, so a compute kernel reads every sample: a
//! color attachment through an unsigned-integer view of its texel size (`newTextureViewWith
//! PixelFormat:`, raw bits, no conversion), a `Depth32Float` one as `depth2d_ms<float>` whose
//! value is bit-cast, a `Stencil8` one as `texture2d_ms<uint>` whose first component is the
//! stencil value. Each sample's texel is written as whole words at its place in the sample-minor
//! attachment layout, which [`crate::executor_contract::multisample::unpack_words`] turns into the
//! attachment's bytes. Nothing is resolved.

use super::*;
use crate::executor_contract::multisample::{
    readback_bytes, unpack_words, ReadbackView, SampleRead,
};
use objc2_metal::MTLTexture;

/// The bytes of every sample of the `dimensions` multisampled attachment `texture` of `samples`
/// samples, read as `read`, in the attachment layout. Submitted and waited for on `queue`.
pub(super) fn read_samples(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    texture: &ProtocolObject<dyn MTLTexture>,
    read: SampleRead,
    dimensions: [u32; 2],
    samples: u32,
) -> Result<Vec<u8>, String> {
    let [width, height] = dimensions;
    let source = kernel_source(read, dimensions, samples);
    let library = compile::bounded("sample readback library compilation", || {
        device.newLibraryWithSource_options_error(&NSString::from_str(&source), None)
    })
    .map_err(|error| format!("compile the sample readback kernel: {error}"))?;
    let function = library
        .newFunctionWithName(&NSString::from_str(KERNEL))
        .ok_or("the sample readback library has no kernel")?;
    let pipeline = compile::bounded("sample readback pipeline creation", || {
        device.newComputePipelineStateWithFunction_error(&function)
    })
    .map_err(|error| format!("create the sample readback pipeline: {error}"))?;
    let view = match read {
        SampleRead::Color(view) => Some(
            texture
                .newTextureViewWithPixelFormat(view_format(view))
                .ok_or("create the sample readback's integer view")?,
        ),
        SampleRead::Depth | SampleRead::Stencil => None,
    };
    let length = readback_bytes(dimensions, samples, read)?;
    let words = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .ok_or("create the sample readback buffer")?;
    let command = queue
        .commandBuffer()
        .ok_or("MTLCommandQueue::commandBuffer returned nil")?;
    let encoder = command
        .computeCommandEncoder()
        .ok_or("computeCommandEncoder returned nil")?;
    encoder.setComputePipelineState(&pipeline);
    unsafe {
        encoder.setTexture_atIndex(Some(view.as_deref().unwrap_or(texture)), 0);
        encoder.setBuffer_offset_atIndex(Some(&words), 0, 0);
    }
    // One thread per sample, one per threadgroup: no grid edge to guard.
    encoder.dispatchThreadgroups_threadsPerThreadgroup(
        MTLSize {
            width: width as usize,
            height: height as usize,
            depth: samples as usize,
        },
        MTLSize {
            width: 1,
            height: 1,
            depth: 1,
        },
    );
    encoder.endEncoding();
    commit_and_wait(&command, "Metal per-sample attachment readback")?;
    // SAFETY: the command completed, and the buffer is shared and `length` bytes long.
    let bytes =
        unsafe { std::slice::from_raw_parts(words.contents().as_ptr().cast::<u8>(), length) };
    unpack_words(bytes, dimensions, samples, read)
}

const KERNEL: &str = "metal2vulkan_sample_readback";

/// The kernel that writes every sample of a `dimensions` attachment of `samples` samples, read as
/// `read`, as words at its place in the attachment layout.
fn kernel_source(read: SampleRead, [width, _]: [u32; 2], samples: u32) -> String {
    let (texture, value) = match read {
        SampleRead::Color(_) => (
            "texture2d_ms<uint, access::read>",
            "uint4 texel = source.read(position.xy, position.z);",
        ),
        SampleRead::Depth => (
            "depth2d_ms<float, access::read>",
            "uint4 texel = uint4(as_type<uint>(source.read(position.xy, position.z)));",
        ),
        // Metal leaves a stencil texture's other components undefined: only the first is read.
        SampleRead::Stencil => (
            "texture2d_ms<uint, access::read>",
            "uint4 texel = uint4(source.read(position.xy, position.z).x);",
        ),
    };
    let words = read.words();
    let stores = (0..words)
        .map(|word| format!("    words[base + {word}u] = texel[{word}];\n"))
        .collect::<String>();
    format!(
        r#"#include <metal_stdlib>
using namespace metal;
kernel void {KERNEL}({texture} source [[texture(0)]],
                     device uint *words [[buffer(0)]],
                     uint3 position [[threadgroup_position_in_grid]]) {{
    {value}
    uint base = ((position.y * {width}u + position.x) * {samples}u + position.z) * {words}u;
{stores}}}
"#
    )
}

/// The Metal pixel format of `view`.
fn view_format(view: ReadbackView) -> MTLPixelFormat {
    match view {
        ReadbackView::R8Uint => MTLPixelFormat::R8Uint,
        ReadbackView::R16Uint => MTLPixelFormat::R16Uint,
        ReadbackView::R32Uint => MTLPixelFormat::R32Uint,
        ReadbackView::Rg32Uint => MTLPixelFormat::RG32Uint,
        ReadbackView::Rgba32Uint => MTLPixelFormat::RGBA32Uint,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::executor_contract::multisample::{attachment_transfer, AttachmentTransfer};

    /// A pass of one sample is drawn as before: the pipeline's `rasterSampleCount` is not set, an
    /// attachmentless pass's default count is 1, and its attachments move by copies (shared
    /// textures, `replaceRegion` and `getBytes`), so no new command reaches it. A multisampled pass
    /// sets its count on both and reads its attachments sample by sample.
    #[test]
    fn a_pass_of_one_sample_issues_the_commands_it_always_did() {
        let mut case = crate::case::attachmentless_fragment_test_case("11".repeat(32), "f".into());
        assert_eq!(
            super::super::pipelines::pipeline_raster_sample_count(&case),
            None
        );
        assert_eq!(super::super::attachmentless_raster_sample_count(&case), 1);
        assert_eq!(attachment_transfer(&case), AttachmentTransfer::Copy);
        case.raster_sample_count = Some(4);
        assert_eq!(
            super::super::pipelines::pipeline_raster_sample_count(&case),
            Some(4)
        );
        assert_eq!(super::super::attachmentless_raster_sample_count(&case), 4);
        assert_eq!(
            attachment_transfer(&case),
            AttachmentTransfer::PerSample { samples: 4 }
        );
    }

    /// The kernel stores each word of a texel at its place in the sample-minor layout, and reads
    /// a depth attachment's float as its bits.
    #[test]
    fn the_kernel_writes_each_samples_words_in_the_attachment_layout() {
        let color = kernel_source(SampleRead::Color(ReadbackView::Rg32Uint), [5, 3], 4);
        assert!(
            color.contains("texture2d_ms<uint, access::read>"),
            "{color}"
        );
        assert!(
            color.contains("((position.y * 5u + position.x) * 4u + position.z) * 2u"),
            "{color}"
        );
        assert!(color.contains("words[base + 1u] = texel[1];"), "{color}");
        assert!(!color.contains("words[base + 2u]"), "{color}");
        let depth = kernel_source(SampleRead::Depth, [2, 2], 2);
        assert!(depth.contains("depth2d_ms<float, access::read>"), "{depth}");
        assert!(depth.contains("as_type<uint>"), "{depth}");
        let stencil = kernel_source(SampleRead::Stencil, [2, 2], 4);
        assert!(
            stencil.contains("texture2d_ms<uint, access::read>"),
            "{stencil}"
        );
        assert!(
            stencil.contains(".read(position.xy, position.z).x"),
            "{stencil}"
        );
        assert!(
            stencil.contains("words[base + 0u] = texel[0];"),
            "{stencil}"
        );
    }
}
