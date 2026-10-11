//! Argument-buffer resource allocation, residency, and handle encoding. Each member's argument
//! index and declared access come from the AIR layout of its argument buffer.

use super::{
    make_texture, texture_shader_usage, Buffer, IndirectCommands, MetalFunctionTables,
    MetalTextureLiteral, Texture,
};
use crate::case::AuthoredCase;
use crate::check::interface::{declared_writable, AirInterface};
use crate::executor_contract::out_of_bounds::{argument_buffer_reach, GUARD_FILL};
use crate::literal::LiteralResources;
use core::ffi::c_void;
use core::ptr::NonNull;
use metal2vulkan_air::abi::{Access, ParamRole};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal::{
    MTLArgumentEncoder, MTLBuffer, MTLCommandQueue, MTLDevice, MTLFunction,
    MTLRenderCommandEncoder, MTLRenderStages, MTLResource, MTLResourceOptions, MTLResourceUsage,
};

pub(super) type ArgumentBufferTexture = ((u32, u32), Texture);
pub(super) type ArgumentBufferBuffer = ((u32, u32), Buffer);

/// The case's argument-buffer buffers. One `case` declares an out-of-bounds target is created
/// inside its shadow (as a buffer target is): one allocation of the target's `reach_before` bytes,
/// its bytes, and its `reach_after` bytes, the shadow filled with [`GUARD_FILL`]. Its argument
/// buffer holds its address at offset `reach_before` ([`embedded_shadow_offset`]), so the shader
/// sees the target as before, and every access the case makes past it through the loaded pointer
/// stays inside this allocation.
pub(super) fn make_argument_buffer_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    resources: &LiteralResources,
) -> Result<Vec<ArgumentBufferBuffer>, String> {
    resources
        .argument_buffer_buffers
        .iter()
        .map(|resource| {
            let key = (resource.buffer_binding, resource.field_offset);
            let label = format!("argument-buffer buffer {}+{}", key.0, key.1);
            let reach = argument_buffer_reach(case, key);
            // A map's member buffer holds Metal's parameter data of the map, at offset 0 of its own
            // allocation, so it has no out-of-bounds shadow.
            if let Some(map) = &resource.rasterization_rate_map {
                if reach.is_some() {
                    return Err(format!(
                        "{label}: a rasterization-rate map's buffer is no out-of-bounds target"
                    ));
                }
                return Ok((key, super::buffers::rate_map_buffer(device, map, &label)?));
            }
            let shadowed = reach.map(|(before, after)| {
                let (before, after) = (before as usize, after as usize);
                let mut bytes = vec![GUARD_FILL; before + resource.bytes.len() + after];
                bytes[before..before + resource.bytes.len()].copy_from_slice(&resource.bytes);
                bytes
            });
            let bytes = shadowed.as_deref().unwrap_or(&resource.bytes);
            let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
                .ok_or_else(|| format!("{label} bytes pointer is null"))?;
            // SAFETY: `pointer` addresses `bytes.len()` initialized bytes, live for the call,
            // which copies them.
            let buffer = unsafe {
                device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| format!("create Metal {label}"))?;
            Ok((key, buffer))
        })
        .collect()
}

/// The offset of the argument-buffer buffer at `key` inside its Metal allocation: its shadow's
/// `reach_before` when `case` declares it an out-of-bounds target, else 0.
pub(super) fn embedded_shadow_offset(case: &AuthoredCase, key: (u32, u32)) -> usize {
    argument_buffer_reach(case, key).map_or(0, |(before, _)| before as usize)
}

fn argument_buffer_resources<'a>(
    textures: &'a [ArgumentBufferTexture],
    buffers: &'a [ArgumentBufferBuffer],
    commands: &'a [IndirectCommands],
) -> impl Iterator<Item = &'a ProtocolObject<dyn MTLResource>> {
    textures
        .iter()
        .map(|(_, resource)| ProtocolObject::from_ref(&**resource))
        .chain(
            buffers
                .iter()
                .map(|(_, resource)| ProtocolObject::from_ref(&**resource)),
        )
        .chain(
            commands
                .iter()
                .map(|(_, resource)| ProtocolObject::from_ref(&**resource)),
        )
}

/// A resource an argument buffer names is reached through a handle, not through a binding, so
/// Metal does not know the encoder needs it until it is told. Until then the access is not
/// merely slow, it is invalid: `MTL_SHADER_VALIDATION=1` drops the write and the case reads
/// back its own authored bytes with a `qualified` status, while the plain path happens to
/// work. Both are the same bug; only one of them says so.
///
/// `Read | Write` because the manifest's role describes what the *case* does with the
/// resource, not what the shader does with it -- the same reason
/// [`texture_shader_usage`] takes the access from AIR rather than from the role.
pub(super) fn argument_buffer_resources_resident_on_compute(
    encoder: &ProtocolObject<dyn objc2_metal::MTLComputeCommandEncoder>,
    textures: &[ArgumentBufferTexture],
    buffers: &[ArgumentBufferBuffer],
    commands: &[IndirectCommands],
) {
    for resource in argument_buffer_resources(textures, buffers, commands) {
        encoder.useResource_usage(resource, MTLResourceUsage::Read | MTLResourceUsage::Write);
    }
}

/// The same residency on a render encoder, which needs the stages that will reach the handle.
pub(super) fn argument_buffer_resources_resident_on_render(
    encoder: &ProtocolObject<dyn MTLRenderCommandEncoder>,
    textures: &[ArgumentBufferTexture],
    buffers: &[ArgumentBufferBuffer],
    commands: &[IndirectCommands],
    stages: MTLRenderStages,
) {
    for resource in argument_buffer_resources(textures, buffers, commands) {
        encoder.useResource_usage_stages(
            resource,
            MTLResourceUsage::Read | MTLResourceUsage::Write,
            stages,
        );
    }
}

/// The argument index Metal's argument encoder takes for the AIR member at `field_offset` of the
/// argument buffer bound at `buffer_binding` (its `[[id]]`), with what `select` takes from its
/// role; a member `select` refuses disagrees with the resource the case binds there.
pub(super) fn argument_member<R>(
    interface: &AirInterface,
    buffer_binding: u32,
    field_offset: u32,
    what: &str,
    select: impl Fn(&ParamRole) -> Option<R>,
) -> Result<(u32, R), String> {
    let (_, member) = interface
        .member_at(buffer_binding, field_offset)
        .ok_or_else(|| {
            format!("{what} {buffer_binding}+{field_offset} is no AIR argument-buffer member")
        })?;
    let selected = select(&member.role).ok_or_else(|| {
        format!("{what} {buffer_binding}+{field_offset} disagrees with its AIR member role")
    })?;
    let id = member.id.ok_or_else(|| {
        format!("{what} {buffer_binding}+{field_offset} has no constant AIR argument id")
    })?;
    Ok((id, selected))
}

fn texture_access(role: &ParamRole) -> Option<Access> {
    match role {
        ParamRole::Texture(texture) => Some(texture.access),
        _ => None,
    }
}

pub(super) fn make_argument_buffer_textures(
    device: &ProtocolObject<dyn MTLDevice>,
    queue: &ProtocolObject<dyn MTLCommandQueue>,
    resources: &LiteralResources,
    interface: &AirInterface,
) -> Result<Vec<ArgumentBufferTexture>, String> {
    resources
        .argument_buffer_textures
        .iter()
        .map(|resource| {
            let label = resource.label();
            let (_, access) = argument_member(
                interface,
                resource.buffer_binding,
                resource.field_offset,
                "argument-buffer texture",
                texture_access,
            )?;
            let texture = make_texture(
                device,
                queue,
                MetalTextureLiteral {
                    label: &label,
                    usage: texture_shader_usage(resource.role, Some(declared_writable(access))),
                    texture_type: resource.texture_type,
                    format: resource.format,
                    dimensions: resource.dimensions,
                    sample_count: resource.sample_count,
                    mip_levels: 1,
                    bytes: &resource.bytes,
                },
            )?;
            Ok(((resource.buffer_binding, resource.field_offset), texture))
        })
        .collect()
}

pub(super) fn argument_encoder(
    function: &ProtocolObject<dyn MTLFunction>,
    buffers: &[(u32, Buffer)],
    binding: u32,
) -> Result<Retained<ProtocolObject<dyn MTLArgumentEncoder>>, String> {
    let buffer = buffers
        .iter()
        .find_map(|(index, buffer)| (*index == binding).then_some(buffer))
        .ok_or_else(|| format!("argument buffer {binding} was not allocated"))?;
    let encoder = unsafe { function.newArgumentEncoderWithBufferIndex(binding as usize) };
    if buffer.length() < encoder.encodedLength() {
        return Err(format!(
            "argument buffer {binding} has {} bytes, but Metal requires {}",
            buffer.length(),
            encoder.encodedLength()
        ));
    }
    unsafe { encoder.setArgumentBuffer_offset(Some(&**buffer), 0) };
    Ok(encoder)
}

pub(super) fn encode_argument_buffer_textures(
    function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    buffers: &[(u32, Buffer)],
    textures: &[ArgumentBufferTexture],
    interface: &AirInterface,
) -> Result<(), String> {
    let buffer_bindings = textures
        .iter()
        .map(|((buffer_binding, _), _)| *buffer_binding)
        .collect::<std::collections::BTreeSet<_>>();
    for buffer_binding in buffer_bindings {
        let encoder = argument_encoder(function, buffers, buffer_binding)?;
        for ((_, field_offset), texture) in textures
            .iter()
            .filter(|((binding, _), _)| *binding == buffer_binding)
        {
            let (index, _) = argument_member(
                interface,
                buffer_binding,
                *field_offset,
                "argument-buffer texture",
                texture_access,
            )?;
            unsafe { encoder.setTexture_atIndex(Some(&**texture), index as usize) };
        }
    }
    Ok(())
}

/// Make each argument-buffer sampler of `case` and encode it into its owner with the owner's
/// argument encoder, at the argument index of its AIR member, which must be a sampler. A sampler
/// state is no `MTLResource`, so it takes no `useResource`; the returned states must live until the
/// command completes.
pub(super) fn encode_argument_buffer_samplers(
    device: &ProtocolObject<dyn MTLDevice>,
    function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    case: &AuthoredCase,
    buffers: &[(u32, Buffer)],
    interface: &AirInterface,
) -> Result<Vec<super::samplers::Sampler>, String> {
    let mut made = Vec::with_capacity(case.argument_buffer_samplers.len());
    for resource in &case.argument_buffer_samplers {
        let (index, ()) = argument_member(
            interface,
            resource.buffer_binding,
            resource.field_offset,
            "argument-buffer sampler",
            |role| matches!(role, ParamRole::Sampler { .. }).then_some(()),
        )?;
        let sampler = super::samplers::make_argument_buffer_sampler(device, resource)?;
        let encoder = argument_encoder(function, buffers, resource.buffer_binding)?;
        unsafe { encoder.setSamplerState_atIndex(Some(&*sampler), index as usize) };
        made.push(sampler);
    }
    Ok(made)
}

/// Encode each argument-buffer buffer's address into its owner, at its shadow's offset when it
/// is an out-of-bounds target of `case` ([`embedded_shadow_offset`]).
pub(super) fn encode_argument_buffer_buffers(
    function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    case: &AuthoredCase,
    owners: &[(u32, Buffer)],
    resources: &[ArgumentBufferBuffer],
    interface: &AirInterface,
) -> Result<(), String> {
    for ((buffer_binding, field_offset), resource) in resources {
        let (index, ()) = argument_member(
            interface,
            *buffer_binding,
            *field_offset,
            "argument-buffer buffer",
            |role| matches!(role, ParamRole::Buffer(_)).then_some(()),
        )?;
        let encoder = argument_encoder(function, owners, *buffer_binding)?;
        unsafe {
            encoder.setBuffer_offset_atIndex(
                Some(&**resource),
                embedded_shadow_offset(case, (*buffer_binding, *field_offset)),
                index as usize,
            );
        }
    }
    Ok(())
}

pub(super) fn encode_argument_buffer_function_tables(
    function: &ProtocolObject<dyn objc2_metal::MTLFunction>,
    buffers: &[(u32, Buffer)],
    tables: &MetalFunctionTables,
    interface: &AirInterface,
) -> Result<(), String> {
    for (location, table) in &tables.intersection {
        let crate::library_module::ResolvedIntersectionFunctionTableLocation::ArgumentBuffer {
            buffer_binding,
            field_offset,
        } = location
        else {
            continue;
        };
        let (index, ()) = argument_member(
            interface,
            *buffer_binding,
            *field_offset,
            "argument-buffer intersection table",
            |role| matches!(role, ParamRole::IntersectionTable(_)).then_some(()),
        )?;
        let encoder = argument_encoder(function, buffers, *buffer_binding)?;
        unsafe {
            encoder.setIntersectionFunctionTable_atIndex(Some(&**table), index as usize);
        }
    }
    Ok(())
}
