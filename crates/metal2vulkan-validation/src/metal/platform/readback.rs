//! Declared output selection from completed native Metal resources.

use super::{
    ArgumentBufferBuffer, ArgumentBufferTexture, Buffer, DepthStencilTexture,
    DeviceBufferArrayElement, Texture, TextureArray,
};
use crate::case::{AuthoredCase, OutputSelection, TextureFormat, TextureType};
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLOrigin, MTLRegion, MTLSize, MTLTexture};

#[derive(Clone, Copy)]
pub(super) struct MetalOutputResources<'a> {
    pub(super) device_buffer_array_elements: &'a [DeviceBufferArrayElement],
    pub(super) colors: &'a [(u32, Texture)],
    pub(super) depth_stencil: Option<&'a DepthStencilTexture>,
    pub(super) fragment_imageblock: Option<(&'a Buffer, [u32; 2], usize)>,
}

pub(super) fn selected_output(
    case: &AuthoredCase,
    buffers: &[(u32, Buffer)],
    argument_buffer_buffers: &[ArgumentBufferBuffer],
    textures: &[(u32, Texture)],
    texture_arrays: &[TextureArray],
    argument_buffer_textures: &[ArgumentBufferTexture],
    output_resources: MetalOutputResources<'_>,
) -> Result<Vec<u8>, String> {
    let device_buffer_array_elements = output_resources.device_buffer_array_elements;
    let render_targets = output_resources.colors;
    let depth_stencil = output_resources.depth_stencil;
    match case.output {
        OutputSelection::None => Ok(Vec::new()),
        OutputSelection::Buffer {
            binding,
            offset,
            length,
        } => {
            let buffer = buffers
                .iter()
                .find_map(|(candidate, buffer)| (*candidate == binding).then_some(buffer))
                .ok_or_else(|| format!("output buffer {binding} was not bound"))?;
            // An out-of-bounds target is bound inside its shadow.
            let start = super::shadow_offset(case, binding) + offset as usize;
            let length = length as usize;
            if start.saturating_add(length) > buffer.length() {
                return Err(format!("selected output exceeds Metal buffer {binding}"));
            }
            unsafe {
                let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                Ok(std::slice::from_raw_parts(pointer, length).to_vec())
            }
        }
        OutputSelection::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
            offset,
            length,
        } => {
            let buffer = argument_buffer_buffers
                    .iter()
                    .find_map(|((owner, field), buffer)| {
                        (*owner == buffer_binding && *field == field_offset).then_some(buffer)
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer buffer {buffer_binding}+{field_offset} was not bound"
                        )
                    })?;
            // An out-of-bounds target is created inside its shadow.
            let start = super::argument_buffers::embedded_shadow_offset(
                case,
                (buffer_binding, field_offset),
            ) + offset as usize;
            let length = length as usize;
            if start.saturating_add(length) > buffer.length() {
                return Err("selected output exceeds Metal argument-buffer buffer".into());
            }
            unsafe {
                let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                Ok(std::slice::from_raw_parts(pointer, length).to_vec())
            }
        }
        OutputSelection::DeviceBufferArrayElement {
            binding,
            element,
            offset,
            length,
        } => {
            let buffer = device_buffer_array_elements
                .iter()
                .find_map(|((owner, index), buffer)| {
                    (*owner == binding && *index == element).then_some(buffer)
                })
                .ok_or_else(|| {
                    format!("output device-buffer-array {binding} element {element} was not bound")
                })?;
            let start = offset as usize;
            let length = length as usize;
            if start.saturating_add(length) > buffer.length() {
                return Err("selected output exceeds Metal device-buffer-array element".into());
            }
            unsafe {
                let pointer = buffer.contents().as_ptr().cast::<u8>().add(start);
                Ok(std::slice::from_raw_parts(pointer, length).to_vec())
            }
        }
        OutputSelection::Texture {
            binding,
            origin,
            dimensions,
        } => {
            let texture = textures
                .iter()
                .find_map(|(candidate, texture)| (*candidate == binding).then_some(texture))
                .ok_or_else(|| format!("output texture {binding} was not bound"))?;
            let resource = case
                .textures
                .iter()
                .find(|resource| resource.binding == binding)
                .ok_or_else(|| format!("output texture {binding} is not declared"))?;
            read_texture_region(
                texture,
                resource.texture_type,
                resource.format,
                origin,
                dimensions,
            )
        }
        OutputSelection::TextureArrayElement {
            binding,
            element,
            origin,
            dimensions,
        } => {
            let texture = texture_arrays
                .iter()
                .find(|(candidate, _)| *candidate == binding)
                .and_then(|(_, elements)| elements.get(element as usize))
                .ok_or_else(|| {
                    format!("output texture-array {binding} element {element} was not bound")
                })?;
            let resource = case
                .texture_arrays
                .iter()
                .find(|resource| resource.binding == binding)
                .ok_or_else(|| format!("output texture-array {binding} is not declared"))?;
            read_texture_region(
                texture,
                resource.texture_type,
                resource.format,
                origin,
                dimensions,
            )
        }
        OutputSelection::ArgumentBufferTexture {
            buffer_binding,
            field_offset,
            origin,
            dimensions,
        } => {
            let texture = argument_buffer_textures
                    .iter()
                    .find_map(|((buffer, offset), texture)| {
                        (*buffer == buffer_binding && *offset == field_offset).then_some(texture)
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer texture {buffer_binding}+{field_offset} was not bound"
                        )
                    })?;
            let resource = case
                    .argument_buffer_textures
                    .iter()
                    .find(|resource| {
                        resource.buffer_binding == buffer_binding
                            && resource.field_offset == field_offset
                    })
                    .ok_or_else(|| {
                        format!(
                            "output argument-buffer texture {buffer_binding}+{field_offset} is not declared"
                        )
                    })?;
            read_texture_region(
                texture,
                resource.texture_type,
                resource.format,
                origin,
                dimensions,
            )
        }
        OutputSelection::RenderTarget {
            index,
            origin,
            dimensions,
        } => {
            let texture = render_targets
                .iter()
                .find_map(|(candidate, texture)| (*candidate == index).then_some(texture))
                .ok_or_else(|| format!("output render target {index} was not bound"))?;
            let resource = case
                .render_targets
                .iter()
                .find(|resource| resource.index == index)
                .ok_or_else(|| format!("output render target {index} is not declared"))?;
            read_texture_region(
                texture,
                TextureType::D2,
                resource.format,
                [origin[0], origin[1], 0],
                [dimensions[0], dimensions[1], 1],
            )
        }
        OutputSelection::Depth { origin, dimensions }
        | OutputSelection::Stencil { origin, dimensions } => {
            let attachment = depth_stencil
                .ok_or_else(|| "depth/stencil output attachment was not bound".to_string())?;
            let depth = matches!(case.output, OutputSelection::Depth { .. });
            let texture = if depth {
                attachment.depth.as_ref()
            } else {
                attachment.stencil.as_ref()
            }
            .ok_or_else(|| "selected Metal depth/stencil aspect was not allocated".to_string())?;
            let pixel_size = if depth { 4 } else { 1 };
            let bytes_per_row = dimensions[0] as usize * pixel_size;
            let mut output = vec![0u8; bytes_per_row * dimensions[1] as usize];
            let pointer = NonNull::new(output.as_mut_ptr().cast::<c_void>())
                .ok_or_else(|| "depth/stencil output pointer is null".to_string())?;
            unsafe {
                texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                    pointer,
                    bytes_per_row,
                    bytes_per_row * dimensions[1] as usize,
                    MTLRegion {
                        origin: MTLOrigin {
                            x: origin[0] as usize,
                            y: origin[1] as usize,
                            z: 0,
                        },
                        size: MTLSize {
                            width: dimensions[0] as usize,
                            height: dimensions[1] as usize,
                            depth: 1,
                        },
                    },
                    0,
                    0,
                );
            }
            Ok(output)
        }
        OutputSelection::FragmentImageblock {
            origin, dimensions, ..
        } => {
            let (buffer, source_dimensions, pixel_size) =
                output_resources.fragment_imageblock.ok_or_else(|| {
                    "selected Metal fragment imageblock output was not resolved".to_string()
                })?;
            let bytes = unsafe {
                std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), buffer.length())
            };
            crate::literal::select_tightly_packed_2d(
                bytes,
                source_dimensions,
                origin,
                dimensions,
                pixel_size,
            )
        }
    }
}

fn read_texture_region(
    texture: &ProtocolObject<dyn MTLTexture>,
    texture_type: TextureType,
    format: TextureFormat,
    origin: [u32; 3],
    dimensions: [u32; 3],
) -> Result<Vec<u8>, String> {
    let span = crate::literal::subresource_span(texture_type, origin, dimensions, 1)?;
    let bytes_per_row = span.size[0] as usize * format.bytes_per_pixel();
    let bytes_per_image = bytes_per_row * span.size[1] as usize;
    let bytes_per_slice = bytes_per_image * span.size[2] as usize;
    let mut output = vec![0u8; bytes_per_slice * span.slice_count as usize];
    let region = MTLRegion {
        origin: MTLOrigin {
            x: span.origin[0] as usize,
            y: span.origin[1] as usize,
            z: span.origin[2] as usize,
        },
        size: MTLSize {
            width: span.size[0] as usize,
            height: span.size[1] as usize,
            depth: span.size[2] as usize,
        },
    };
    for slice in 0..span.slice_count as usize {
        let pointer = unsafe {
            NonNull::new_unchecked(
                output
                    .as_mut_ptr()
                    .add(slice * bytes_per_slice)
                    .cast::<c_void>(),
            )
        };
        unsafe {
            texture.getBytes_bytesPerRow_bytesPerImage_fromRegion_mipmapLevel_slice(
                pointer,
                bytes_per_row,
                bytes_per_image,
                region,
                0,
                span.base_slice as usize + slice,
            );
        }
    }
    Ok(output)
}
