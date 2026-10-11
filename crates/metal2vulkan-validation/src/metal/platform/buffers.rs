//! Native buffer uploads and ownership for direct bindings and array elements.

use super::Buffer;
use crate::case::out_of_bounds::OutOfBoundsResource;
use crate::case::AuthoredCase;
use crate::case::RateMapLiteral;
use crate::check::interface::AirInterface;
use crate::executor_contract::buffer_slots::stage_input_slots;
use crate::executor_contract::out_of_bounds::{buffer_reach, reach, GUARD_FILL};
use crate::literal::LiteralResources;
use core::ffi::c_void;
use core::ptr::NonNull;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2::AllocAnyThread;
use objc2_metal::{
    MTLBuffer, MTLDevice, MTLRasterizationRateLayerDescriptor, MTLRasterizationRateMap,
    MTLRasterizationRateMapDescriptor, MTLResourceOptions, MTLSize,
};

pub(super) type DeviceBufferArrayElement = ((u32, u32), Buffer);

pub(super) struct NativeBuffers {
    pub(super) buffers: Vec<(u32, Buffer)>,
    pub(super) device_buffer_array_elements: Vec<DeviceBufferArrayElement>,
}

pub(super) fn upload_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    bytes: &[u8],
    label: &str,
) -> Result<Buffer, String> {
    let pointer = NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>())
        .ok_or_else(|| format!("{label} bytes pointer is null"))?;
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            bytes.len(),
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| format!("create Metal {label}"))
}

pub(super) fn make_buffers(
    device: &ProtocolObject<dyn MTLDevice>,
    resources: &LiteralResources,
    reflection: &crate::air::reflect::ShaderInterface,
    interface: &AirInterface,
) -> Result<NativeBuffers, String> {
    let mut buffers = resources
        .buffers
        .iter()
        .map(|resource| {
            let label = format!("buffer {}", resource.binding);
            if let Some(map) = &resource.rasterization_rate_map {
                return Ok((resource.binding, rate_map_buffer(device, map, &label)?));
            }
            let bytes = crate::executor_contract::buffer_bytes_padded_to_declared_size(
                &resource.bytes,
                crate::executor_contract::reflected_buffer_declared_size(
                    reflection,
                    resource.binding,
                ),
            );
            let buffer = upload_buffer(device, &bytes, &label)?;
            Ok((resource.binding, buffer))
        })
        .collect::<Result<Vec<_>, String>>()?;
    let stage_input_slots = stage_input_slots(interface)?;
    for resource in &resources.kernel_stage_inputs {
        let slot = stage_input_slots.get(&resource.location).ok_or_else(|| {
            format!(
                "kernel stage input {} is no AIR stage input",
                resource.location
            )
        })?;
        let buffer = upload_buffer(
            device,
            &resource.bytes,
            &format!("stage-input buffer {}", resource.location),
        )?;
        buffers.push((*slot, buffer));
    }
    let mut nested = Vec::new();
    for array in &resources.device_buffer_arrays {
        for element in &array.elements {
            let label = format!(
                "device-buffer-array {} element {}",
                array.binding, element.index
            );
            let binding = array.binding.checked_add(element.index).ok_or_else(|| {
                format!(
                    "device-buffer-array {} element {} binding overflows",
                    array.binding, element.index
                )
            })?;
            let buffer = upload_buffer(device, &element.bytes, &label)?;
            buffers.push((binding, buffer.clone()));
            nested.push(((array.binding, element.index), buffer));
        }
    }
    Ok(NativeBuffers {
        buffers,
        device_buffer_array_elements: nested,
    })
}

/// A buffer holding Metal's parameter data of `map`: the device compiles the descriptor
/// (`newRasterizationRateMapWithDescriptor:`) and copies the data it defines
/// (`copyParameterDataToBuffer:offset:`) to offset 0 of a buffer of its stated size. A buffer
/// argument binds it; an argument-buffer member encodes it.
pub(super) fn rate_map_buffer(
    device: &ProtocolObject<dyn MTLDevice>,
    map: &RateMapLiteral,
    label: &str,
) -> Result<Buffer, String> {
    let compiled = compile_rate_map(device, map, label)?;
    let length = compiled.parameterBufferSizeAndAlign().size.max(1);
    let buffer = device
        .newBufferWithLength_options(length, MTLResourceOptions::StorageModeShared)
        .ok_or_else(|| format!("create Metal {label}"))?;
    // SAFETY: a shared buffer of the map's stated size, at offset 0 (aligned for every size).
    unsafe { compiled.copyParameterDataToBuffer_offset(&buffer, 0) };
    Ok(buffer)
}

/// Metal's compile of `map` (`newRasterizationRateMapWithDescriptor:`): the map a buffer
/// literal's parameter data comes from, and a render pass's `rasterizationRateMap`.
pub(super) fn compile_rate_map(
    device: &ProtocolObject<dyn MTLDevice>,
    map: &RateMapLiteral,
    label: &str,
) -> Result<Retained<ProtocolObject<dyn MTLRasterizationRateMap>>, String> {
    let size = |width: usize, height: usize| MTLSize {
        width,
        height,
        depth: 0,
    };
    let layers = map.layers.len();
    // SAFETY: a plain query of the device.
    if !unsafe { device.supportsRasterizationRateMapWithLayerCount(layers) } {
        return Err(format!(
            "{label}: this Metal device supports no rasterization-rate map of {layers} layers"
        ));
    }
    let [width, height] = map.screen_size;
    // SAFETY: the extents are the validated case's, each in 1..=65536.
    let descriptor = unsafe {
        MTLRasterizationRateMapDescriptor::rasterizationRateMapDescriptorWithScreenSize(size(
            width as usize,
            height as usize,
        ))
    };
    for (index, layer) in map.layers.iter().enumerate() {
        let mut horizontal = layer.horizontal.iter().map(|s| s.0).collect::<Vec<f32>>();
        let mut vertical = layer.vertical.iter().map(|s| s.0).collect::<Vec<f32>>();
        if horizontal.is_empty() || vertical.is_empty() {
            return Err(format!(
                "{label}: layer {index} has an axis without samples"
            ));
        }
        let count = size(horizontal.len(), vertical.len());
        // Each pointer is derived from its whole slice, so it may reach every one of the `count`
        // floats the initializer reads.
        let h = NonNull::from(horizontal.as_mut_slice()).cast::<f32>();
        let v = NonNull::from(vertical.as_mut_slice()).cast::<f32>();
        // SAFETY: `h` and `v` point to `count.width` and `count.height` initialized floats, live
        // until the call returns; the initializer copies them.
        let layer = unsafe {
            MTLRasterizationRateLayerDescriptor::initWithSampleCount_horizontal_vertical(
                MTLRasterizationRateLayerDescriptor::alloc(),
                count,
                h,
                v,
            )
        };
        // SAFETY: `index` is below the layer count the descriptor grows to.
        unsafe { descriptor.setLayer_atIndex(Some(&layer), index) };
    }
    device
        .newRasterizationRateMapWithDescriptor(&descriptor)
        .ok_or_else(|| format!("{label}: Metal compiles no rasterization-rate map"))
}

/// Move each out-of-bounds buffer target of `case` into its shadow: one Metal allocation of the
/// target's `reach_before` bytes, its bytes, and its `reach_after` bytes, the shadow filled with
/// [`GUARD_FILL`]. Bound at offset `reach_before` ([`shadow_offset`]), the shader sees the target
/// as before, and every access the case makes past it stays inside this allocation: Metal leaves a
/// buffer access past its range undefined, so the oracle never makes one. A kernel stage input's
/// buffer (at its slot for the entry's AIR-layer `interface`) that the case declares a target is
/// shadowed past its data the same way: Metal's stage-in fetch checks nothing either.
pub(super) fn shadow_out_of_bounds_targets(
    device: &ProtocolObject<dyn MTLDevice>,
    case: &AuthoredCase,
    interface: &AirInterface,
    buffers: &mut [(u32, Buffer)],
) -> Result<(), String> {
    let stage_inputs = if case.kernel_stage_inputs.is_empty() {
        Default::default()
    } else {
        stage_input_slots(interface)?
    };
    for (binding, buffer) in buffers.iter_mut() {
        let stage_input = stage_inputs
            .iter()
            .find(|(_, slot)| **slot == *binding)
            .map(|(&location, _)| location);
        let shadow = match stage_input {
            Some(location) => {
                let reach = reach(case, OutOfBoundsResource::AttributeBuffer { location });
                if reach.is_some_and(|(before, _)| before != 0) {
                    return Err(format!(
                        "stage input {location}'s shadow lies before its data, which its \
                         descriptor fetches from offset 0"
                    ));
                }
                reach
            }
            None => buffer_reach(case, *binding),
        };
        let Some((before, after)) = shadow else {
            continue;
        };
        let (before, after) = (before as usize, after as usize);
        let length = buffer.length();
        let mut bytes = vec![GUARD_FILL; before + length + after];
        // SAFETY: a shared-storage buffer's contents are `length` readable bytes, and no command
        // uses the buffer yet.
        let contents =
            unsafe { std::slice::from_raw_parts(buffer.contents().as_ptr().cast::<u8>(), length) };
        bytes[before..before + length].copy_from_slice(contents);
        *buffer = upload_buffer(device, &bytes, &format!("shadowed buffer {binding}"))?;
    }
    Ok(())
}

/// `bytes` followed by the shadow past an attribute buffer or index buffer target of `reach`:
/// `reach_after` bytes of [`GUARD_FILL`], so every fetch the draw or dispatch makes past the data
/// stays inside the allocation. The case checker keeps such a target's `reach_before` at 0 (it is
/// bound and fetched from offset 0); `bytes` alone without a target.
pub(super) fn with_shadow_past(
    bytes: &[u8],
    reach: Option<(u32, u32)>,
    label: &str,
) -> Result<Vec<u8>, String> {
    match reach {
        None => Ok(bytes.to_vec()),
        Some((0, after)) => {
            let mut shadowed = bytes.to_vec();
            shadowed.resize(bytes.len() + after as usize, GUARD_FILL);
            Ok(shadowed)
        }
        Some(_) => Err(format!(
            "{label}'s shadow lies before its data, which is bound at offset 0"
        )),
    }
}

/// The offset the oracle binds the buffer at `binding` at: its shadow's `reach_before` when
/// `case` declares it an out-of-bounds target, else 0.
pub(super) fn shadow_offset(case: &AuthoredCase, binding: u32) -> usize {
    buffer_reach(case, binding).map_or(0, |(before, _)| before as usize)
}

pub(super) fn new_buffer_from_slice<T>(
    device: &ProtocolObject<dyn MTLDevice>,
    values: &[T],
    label: &str,
) -> Result<Buffer, String> {
    let length = std::mem::size_of_val(values);
    if length == 0 {
        return device
            .newBufferWithLength_options(1, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| format!("create Metal buffer for {label}"));
    }
    let pointer = NonNull::new(values.as_ptr().cast_mut().cast::<c_void>())
        .ok_or_else(|| format!("{label} pointer is null"))?;
    unsafe {
        device.newBufferWithBytes_length_options(
            pointer,
            length,
            MTLResourceOptions::StorageModeShared,
        )
    }
    .ok_or_else(|| format!("create Metal buffer for {label}"))
}
