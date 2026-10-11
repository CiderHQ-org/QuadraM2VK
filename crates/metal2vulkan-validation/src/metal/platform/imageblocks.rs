//! Fragment imageblock helper shaders, pipelines, and plane resources.

use super::attachments::DepthStencilTexture;
use super::buffers::new_buffer_from_slice;
use super::textures::Texture;
use super::Buffer;
use crate::case::{AuthoredCase, OutputSelection};
use crate::gpu::compile;
use crate::literal::LiteralResources;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLColorWriteMask, MTLDevice, MTLFunction, MTLLibrary, MTLRenderPipelineDescriptor,
    MTLRenderPipelineState, MTLResourceOptions, MTLTexture,
};

pub(super) struct FragmentImageblockExecution {
    pub(super) initialize: Retained<ProtocolObject<dyn MTLRenderPipelineState>>,
    pub(super) resolve: Option<Retained<ProtocolObject<dyn MTLRenderPipelineState>>>,
    pub(super) inputs: Vec<(usize, Buffer)>,
    pub(super) output: Option<Buffer>,
    pub(super) output_pixel_size: Option<usize>,
    pub(super) dimensions: [u32; 2],
}

/// The name a generated imageblock helper must give a member for Metal to alias it onto the
/// AIR entry's own imageblock.
///
/// Reflection reports one `semantic` per member in two spellings: `user(name)` when the Metal
/// source gave the member an explicit `[[user(...)]]` attribute, and the member's own declared
/// name when it did not. Both name the same thing to Metal, which matches imageblock members
/// by their user attribute where one exists and by the declared name otherwise. Accepting only
/// the wrapped form refused every custom fragment imageblock in the corpus, since none of them
/// carries an explicit attribute -- an authoring gap, not a Metal one.
fn fragment_imageblock_semantic_name(semantic: &str) -> Result<&str, String> {
    let name = semantic
        .strip_prefix("user(")
        .and_then(|value| value.strip_suffix(')'))
        .unwrap_or(semantic);
    if name.is_empty()
        || !name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
    {
        return Err(format!("invalid fragment imageblock semantic {semantic}"));
    }
    Ok(name)
}

pub(super) fn make_fragment_imageblock_execution(
    device: &ProtocolObject<dyn MTLDevice>,
    vertex: &ProtocolObject<dyn MTLFunction>,
    render_targets: &[(u32, Texture)],
    depth_stencil: Option<&DepthStencilTexture>,
    case: &AuthoredCase,
    resources: &LiteralResources,
    reflection: &crate::air::reflect::ShaderInterface,
) -> Result<Option<FragmentImageblockExecution>, String> {
    let (Some(authored), Some(reflected)) = (
        resources.fragment_imageblock.as_ref(),
        reflection.fragment_imageblock.as_ref(),
    ) else {
        return Ok(None);
    };
    let mut fields = String::new();
    let mut parameters = Vec::new();
    let mut assignments = Vec::new();
    let mut inputs = Vec::new();
    for (index, member) in reflected.members.iter().enumerate() {
        let format =
            crate::case::FragmentImageblockFormat::from_air_type(&member.type_name, member.size)
                .ok_or_else(|| {
                    format!(
                        "Metal fragment imageblock member {} has unsupported type {} size {}",
                        member.semantic, member.type_name, member.size
                    )
                })?;
        let metal_type = format.air_type_name();
        let plane_len =
            authored.dimensions[0] as usize * authored.dimensions[1] as usize * format.byte_size();
        let semantic = fragment_imageblock_semantic_name(&member.semantic)?;
        fields.push_str(&format!(
            "    {metal_type} m{index} [[user({semantic}), raster_order_group({})]];\n",
            member.raster_order_group,
        ));
        parameters.push(format!(
            "const device {metal_type}* input{index} [[buffer({index})]]"
        ));
        assignments.push(format!("    result.data.m{index} = input{index}[linear];"));
        let bytes = authored
            .members
            .iter()
            .find(|resource| resource.semantic == member.semantic)
            .map(|resource| resource.bytes.clone())
            .unwrap_or_else(|| vec![0; plane_len]);
        inputs.push((
            index,
            new_buffer_from_slice(
                device,
                &bytes,
                &format!("fragment imageblock member {}", member.semantic),
            )?,
        ));
    }
    let selected = match &case.output {
        OutputSelection::FragmentImageblock { semantic, .. } => {
            let index = reflected
                .members
                .iter()
                .position(|member| member.semantic == *semantic)
                .ok_or_else(|| {
                    format!("selected fragment imageblock semantic {semantic} is not reflected")
                })?;
            let member = &reflected.members[index];
            let format = crate::case::FragmentImageblockFormat::from_air_type(
                    &member.type_name,
                    member.size,
                )
                .ok_or_else(|| {
                    format!(
                        "selected fragment imageblock member {semantic} has unsupported type {} size {}",
                        member.type_name, member.size
                    )
                })?;
            Some((index, format))
        }
        _ => None,
    };
    let resolver = selected.map(|(index, format)| {
            let metal_type = format.air_type_name();
            format!(
                "fragment void metal2vulkan_imageblock_resolve(Metal2VulkanImageblock data [[imageblock_data]], float4 position [[position]], device {metal_type}* output [[buffer(0)]]) {{\n    uint linear = uint(position.y) * {}u + uint(position.x);\n    output[linear] = data.m{index};\n}}\n",
                authored.dimensions[0]
            )
        });
    let source = format!(
            "#include <metal_stdlib>\nusing namespace metal;\nstruct Metal2VulkanImageblock {{\n{fields}}};\nstruct Metal2VulkanImageblockOutput {{ Metal2VulkanImageblock data [[imageblock_data]]; }};\nfragment Metal2VulkanImageblockOutput metal2vulkan_imageblock_initialize(float4 position [[position]], {}) {{\n    uint linear = uint(position.y) * {}u + uint(position.x);\n    Metal2VulkanImageblockOutput result;\n{}\n    return result;\n}}\n{}",
            parameters.join(", "),
            authored.dimensions[0],
            assignments.join("\n"),
            resolver.as_deref().unwrap_or("")
        );
    let library = compile::bounded("imageblock helper library compilation", || {
        device.newLibraryWithSource_options_error(&NSString::from_str(&source), None)
    })
    .map_err(|error| format!("compile generated fragment imageblock helpers: {error}"))?;
    let initialize = library
        .newFunctionWithName(&NSString::from_str("metal2vulkan_imageblock_initialize"))
        .ok_or_else(|| "generated imageblock library has no initializer".to_string())?;
    let resolve = resolver
        .as_ref()
        .map(|_| {
            library
                .newFunctionWithName(&NSString::from_str("metal2vulkan_imageblock_resolve"))
                .ok_or_else(|| "generated imageblock library has no resolver".to_string())
        })
        .transpose()?;
    let make_pipeline = |fragment: &ProtocolObject<dyn MTLFunction>| {
        let descriptor = MTLRenderPipelineDescriptor::new();
        descriptor.setVertexFunction(Some(vertex));
        descriptor.setFragmentFunction(Some(fragment));
        let attachments = descriptor.colorAttachments();
        for (index, texture) in render_targets {
            let attachment = unsafe { attachments.objectAtIndexedSubscript(*index as usize) };
            attachment.setPixelFormat(texture.pixelFormat());
            attachment.setWriteMask(MTLColorWriteMask::None);
        }
        if let Some(depth_stencil) = depth_stencil {
            if let Some(depth) = &depth_stencil.depth {
                descriptor.setDepthAttachmentPixelFormat(depth.pixelFormat());
            }
            if let Some(stencil) = &depth_stencil.stencil {
                descriptor.setStencilAttachmentPixelFormat(stencil.pixelFormat());
            }
        }
        compile::bounded("imageblock helper pipeline creation", || {
            device.newRenderPipelineStateWithDescriptor_error(&descriptor)
        })
        .map_err(|error| format!("create fragment imageblock helper pipeline: {error}"))
    };
    let initialize = make_pipeline(&initialize)?;
    let resolve = resolve.as_deref().map(make_pipeline).transpose()?;
    let output = selected
        .map(|(_, format)| {
            let plane_len = authored.dimensions[0] as usize
                * authored.dimensions[1] as usize
                * format.byte_size();
            device
                .newBufferWithLength_options(
                    plane_len.max(1),
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "create fragment imageblock output buffer".to_string())
        })
        .transpose()?;
    Ok(Some(FragmentImageblockExecution {
        initialize,
        resolve,
        inputs,
        output,
        output_pixel_size: selected.map(|(_, format)| format.byte_size()),
        dimensions: authored.dimensions,
    }))
}
/// Metal names an imageblock member by its `[[user(...)]]` attribute when it has one and by
/// its declared name otherwise, and AIR reports both through the same reflected `semantic`.
/// Every custom fragment imageblock in the corpus is the second form.
#[cfg(test)]
#[test]
fn an_imageblock_semantic_is_a_user_attribute_or_a_plain_member_name() {
    use self::fragment_imageblock_semantic_name as name_of;
    assert_eq!(name_of("user(depth)"), Ok("depth"));
    assert_eq!(name_of("color"), Ok("color"));
    assert_eq!(name_of("gBuffer_2"), Ok("gBuffer_2"));
    for rejected in ["", "user()", "user(depth", "a.b", "user(a b)"] {
        assert!(
            name_of(rejected).is_err(),
            "{rejected:?} is not an MSL member name"
        );
    }
}
