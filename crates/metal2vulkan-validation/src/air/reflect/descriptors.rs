//! The old engine's default descriptor ABI (set 0; bindings 0 / 32 / 160 / 192 / 200 / 224 / 480 for
//! buffers, sampled textures, samplers, color inputs, implicit imageblocks, fragment imageblocks
//! and storage textures) and the checks its `inspect` made of an interface against it. An interface
//! the old engine refused here is refused here, with the same message.

use super::{DescriptorLocation, ResourceAccess, ResourceKind, ShaderInterface};
use crate::air::meta::{storage_air_type_for_size, AirScalar, AirType, TextureDimension};

/// The descriptor set of every resource.
pub(super) const RESOURCE_DESCRIPTOR_SET: u32 = 0;

/// One half-open band of bindings.
#[derive(Clone, Copy)]
struct Band {
    start: u32,
    end: u32,
}

impl Band {
    const fn binding(self, index: u32) -> Option<u32> {
        if index < self.end - self.start {
            Some(self.start + index)
        } else {
            None
        }
    }
}

const BUFFERS: Band = Band { start: 0, end: 32 };
const SAMPLED_TEXTURES: Band = Band {
    start: 32,
    end: 160,
};
const SAMPLERS: Band = Band {
    start: 160,
    end: 192,
};
const COLOR_INPUTS: Band = Band {
    start: 192,
    end: 200,
};
const IMAGEBLOCKS: Band = Band {
    start: 200,
    end: 224,
};
const FRAGMENT_IMAGEBLOCKS: Band = Band {
    start: 224,
    end: 480,
};
const STORAGE_TEXTURES: Band = Band {
    start: 480,
    end: 608,
};

/// The Metal sampler argument table's entries; the rest of the sampler band holds constexpr
/// samplers.
const SAMPLER_ARGUMENT_COUNT: u32 = 16;
/// Binding planes per implicit-imageblock attachment: data rates 0, 1 and 2.
const IMAGEBLOCK_DATA_RATE_STRIDE: u32 = 3;

/// The binding past each band, where the old builder put a resource whose index its band lacks
/// (refused by [`check_bands`]).
pub(super) const SAMPLED_TEXTURE_BINDING_END: u32 = SAMPLED_TEXTURES.end;
pub(super) const STORAGE_TEXTURE_BINDING_END: u32 = STORAGE_TEXTURES.end;
pub(super) const IMAGEBLOCK_BINDING_END: u32 = IMAGEBLOCKS.end;

pub(super) const fn buffer_binding(index: u32) -> Option<u32> {
    BUFFERS.binding(index)
}

pub(super) const fn sampled_texture_binding(index: u32) -> Option<u32> {
    SAMPLED_TEXTURES.binding(index)
}

pub(super) const fn storage_texture_binding(index: u32) -> Option<u32> {
    STORAGE_TEXTURES.binding(index)
}

pub(super) const fn sampler_binding(index: u32) -> Option<u32> {
    if index < SAMPLER_ARGUMENT_COUNT {
        SAMPLERS.binding(index)
    } else {
        None
    }
}

pub(super) const fn color_input_binding(index: u32) -> Option<u32> {
    COLOR_INPUTS.binding(index)
}

pub(super) const fn imageblock_binding(attachment: u32, data_rate: u32) -> Option<u32> {
    if data_rate >= IMAGEBLOCK_DATA_RATE_STRIDE {
        return None;
    }
    let Some(offset) = attachment.checked_mul(IMAGEBLOCK_DATA_RATE_STRIDE) else {
        return None;
    };
    let Some(offset) = offset.checked_add(data_rate) else {
        return None;
    };
    IMAGEBLOCKS.binding(offset)
}

pub(super) const fn fragment_imageblock_binding(member: u32) -> Option<u32> {
    FRAGMENT_IMAGEBLOCKS.binding(member)
}

/// The sampler band's bindings, in order: where constexpr samplers are placed.
pub(super) fn sampler_band() -> std::ops::Range<u32> {
    SAMPLERS.start..SAMPLERS.end
}

/// The first binding of the sampler band.
pub(super) const SAMPLER_BAND_START: u32 = SAMPLERS.start;

fn storage_texture(kind: ResourceKind, access: Option<ResourceAccess>) -> bool {
    matches!(kind, ResourceKind::StorageImage)
        || matches!(
            kind,
            ResourceKind::TextureArray | ResourceKind::EmbeddedArgBufferTexture
        ) && access == Some(ResourceAccess::Storage)
}

/// The binding the default layout gives a descriptor-backed resource, if its band has the index.
fn band_binding(kind: ResourceKind, metal_index: u32, storage: bool) -> Option<u32> {
    match kind {
        ResourceKind::Buffer
        | ResourceKind::KernelStageInput
        | ResourceKind::AccelerationStructureShadow
        | ResourceKind::PrimitiveAccelerationStructure => buffer_binding(metal_index),
        ResourceKind::Texture
        | ResourceKind::TextureArray
        | ResourceKind::StorageImage
        | ResourceKind::EmbeddedArgBufferTexture => {
            if storage {
                storage_texture_binding(metal_index)
            } else {
                sampled_texture_binding(metal_index)
            }
        }
        ResourceKind::Sampler => sampler_binding(metal_index),
        ResourceKind::ColorInput => color_input_binding(metal_index),
        ResourceKind::StaticSampler
        | ResourceKind::ThreadgroupBuffer
        | ResourceKind::VisibleFunctionTable
        | ResourceKind::IntersectionFunctionTable
        | ResourceKind::EmbeddedArgBufferBuffer => None,
    }
}

/// The old engine's placement of the built interface in the default layout: every descriptor and
/// implicit-imageblock plane must have a binding in its band. (The builder already placed each at
/// that binding, so placing them again moves nothing; what remains is the refusal.)
pub(super) fn check_bands(interface: &ShaderInterface) -> Result<(), String> {
    for resource in &interface.bindings {
        if resource.descriptor.is_none() {
            continue;
        }
        let storage = storage_texture(resource.kind, resource.access);
        band_binding(resource.kind, resource.metal_index, storage).ok_or_else(|| {
            format!(
                "{:?} resource {} exceeds the selected descriptor layout",
                resource.kind, resource.metal_index
            )
        })?;
    }
    for attachment in &interface.implicit_imageblock_attachments {
        imageblock_binding(attachment.attachment, attachment.data_rate).ok_or_else(|| {
            format!(
                "implicit imageblock attachment {} rate {} exceeds the selected descriptor layout",
                attachment.attachment, attachment.data_rate
            )
        })?;
    }
    Ok(())
}

/// The descriptor class a binding takes: two resources may share a binding only in one class.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DescriptorClass {
    StorageBuffer,
    SampledImage,
    UniformTexelBuffer,
    StorageImage,
    StorageTexelBuffer,
    Sampler,
    InputAttachment,
}

/// The old engine's check of the interface's descriptor ABI as one closed, collision-free mapping:
/// a buffer layout no wider than its declared size, no resource listed twice, every descriptor at
/// its band's binding, and no binding shared by two descriptor classes.
pub(super) fn validate_descriptor_abi(interface: &ShaderInterface) -> Result<(), String> {
    // A member layout reaching past the argument's declared size describes bytes the argument does
    // not have.
    for resource in &interface.bindings {
        let (Some(layout), Some(declared)) = (&resource.type_layout, resource.declared_size) else {
            continue;
        };
        let Some(extent) = metadata_extent(layout) else {
            continue;
        };
        if extent > u64::from(declared) {
            return Err(format!(
                "reflection reports a {extent}-byte member layout for the {declared}-byte argument of {:?}({})",
                resource.kind, resource.metal_index
            ));
        }
    }
    // Two entries equal in every field describe one resource twice.
    for (index, resource) in interface.bindings.iter().enumerate() {
        if let Some(first) = interface.bindings[..index]
            .iter()
            .position(|other| other == resource)
        {
            return Err(format!(
                "reflection reports {:?}({}) twice, identically, at bindings[{first}] and bindings[{index}]",
                resource.kind, resource.metal_index
            ));
        }
    }

    let mut occupied =
        std::collections::BTreeMap::<(u32, u32), (String, DescriptorClass, u32)>::new();
    let mut record = |location: DescriptorLocation,
                      owner: String,
                      class: DescriptorClass|
     -> Result<(), String> {
        if location.set != RESOURCE_DESCRIPTOR_SET {
            return Err(format!(
                "{owner} uses descriptor set {}, expected {RESOURCE_DESCRIPTOR_SET}",
                location.set
            ));
        }
        if location.count == 0 {
            return Err(format!("{owner} has zero descriptor count"));
        }
        if let Some((previous, previous_class, previous_count)) = occupied.insert(
            (location.set, location.binding),
            (owner.clone(), class, location.count),
        ) {
            if previous_class != class {
                return Err(format!(
                        "descriptor set {} binding {} is shared incompatibly by {previous} ({previous_class:?}, count {previous_count}) and {owner} ({class:?}, count {})",
                        location.set, location.binding, location.count
                    ));
            }
        }
        Ok(())
    };

    for resource in &interface.bindings {
        let storage = storage_texture(resource.kind, resource.access);
        let texel_buffer = resource
            .texture_shape
            .is_some_and(|shape| shape.dimension == TextureDimension::Buffer);
        let class = match resource.kind {
            ResourceKind::Buffer
            | ResourceKind::KernelStageInput
            | ResourceKind::AccelerationStructureShadow
            | ResourceKind::PrimitiveAccelerationStructure
            | ResourceKind::ThreadgroupBuffer
            | ResourceKind::VisibleFunctionTable
            | ResourceKind::IntersectionFunctionTable
            | ResourceKind::EmbeddedArgBufferBuffer => DescriptorClass::StorageBuffer,
            ResourceKind::Texture
            | ResourceKind::TextureArray
            | ResourceKind::StorageImage
            | ResourceKind::EmbeddedArgBufferTexture => match (storage, texel_buffer) {
                (true, true) => DescriptorClass::StorageTexelBuffer,
                (true, false) => DescriptorClass::StorageImage,
                (false, true) => DescriptorClass::UniformTexelBuffer,
                (false, false) => DescriptorClass::SampledImage,
            },
            ResourceKind::Sampler | ResourceKind::StaticSampler => DescriptorClass::Sampler,
            ResourceKind::ColorInput => DescriptorClass::InputAttachment,
        };
        let expected = match resource.kind {
            ResourceKind::PrimitiveAccelerationStructure if resource.descriptor.is_none() => {
                continue;
            }
            ResourceKind::StaticSampler => SAMPLERS.binding(resource.metal_index),
            ResourceKind::ThreadgroupBuffer
            | ResourceKind::VisibleFunctionTable
            | ResourceKind::IntersectionFunctionTable
            | ResourceKind::EmbeddedArgBufferBuffer => {
                if resource.descriptor.is_some() {
                    return Err(format!(
                        "{:?} resource {} unexpectedly consumes a descriptor",
                        resource.kind, resource.metal_index
                    ));
                }
                continue;
            }
            kind => band_binding(kind, resource.metal_index, storage),
        };
        let owner = format!("{:?}({})", resource.kind, resource.metal_index);
        let expected =
            expected.ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
        let location = resource
            .descriptor
            .ok_or_else(|| format!("{owner} is missing its descriptor"))?;
        if location.binding != expected {
            return Err(format!(
                "{owner} uses binding {}, expected {expected}",
                location.binding
            ));
        }
        record(location, owner, class)?;
    }

    for attachment in &interface.implicit_imageblock_attachments {
        let owner = format!(
            "implicit imageblock attachment {} rate {}",
            attachment.attachment, attachment.data_rate
        );
        let expected = imageblock_binding(attachment.attachment, attachment.data_rate)
            .ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
        if attachment.binding != expected {
            return Err(format!(
                "{owner} uses binding {}, expected {expected}",
                attachment.binding
            ));
        }
        record(
            DescriptorLocation {
                set: RESOURCE_DESCRIPTOR_SET,
                binding: attachment.binding,
                count: 1,
            },
            owner,
            DescriptorClass::StorageImage,
        )?;
    }
    if let Some(imageblock) = &interface.fragment_imageblock {
        for (index, member) in imageblock.members.iter().enumerate() {
            let Some(binding) = member.binding else {
                continue;
            };
            let owner = format!("fragment imageblock member {index}");
            let expected = fragment_imageblock_binding(index as u32)
                .ok_or_else(|| format!("{owner} exceeds its descriptor ABI band"))?;
            if binding != expected {
                return Err(format!(
                    "{owner} uses binding {binding}, expected {expected}"
                ));
            }
            record(
                DescriptorLocation {
                    set: RESOURCE_DESCRIPTOR_SET,
                    binding,
                    count: 1,
                },
                owner,
                DescriptorClass::StorageImage,
            )?;
        }
    }
    Ok(())
}

/// How far an AIR member layout reaches, by the old engine's rule: a struct reaches its furthest
/// member's end, an array its element's reach times its length, and a leaf its LLVM allocation
/// size with no datalayout (a vector rounded up to its power-of-two size, a packed vector an array
/// of its scalars, a matrix an array of column vectors).
pub(super) fn metadata_extent(ty: &AirType) -> Option<u64> {
    match ty {
        AirType::Struct(members) => members
            .iter()
            .map(|member| u64::from(member.offset).checked_add(metadata_extent(&member.ty)?))
            .try_fold(0, |reach, end| Some(reach.max(end?))),
        AirType::Array { elem, len } => metadata_extent(elem)?.checked_mul(u64::from(*len)),
        AirType::Opaque { size } => metadata_extent(&storage_air_type_for_size(*size)),
        AirType::Scalar(scalar) => Some(scalar_allocation(*scalar).0),
        AirType::Vec { scalar, lanes } => vector_allocation(*scalar, *lanes).map(|(size, _)| size),
        AirType::PackedVec { scalar, lanes } => {
            let (size, alignment) = scalar_allocation(*scalar);
            round_up(size, alignment)?.checked_mul(u64::from(*lanes))
        }
        AirType::Matrix { scalar, cols, rows } => {
            // `{ [cols x <rows x scalar>] }`: the struct rounds the array up to the column's
            // alignment, which the array's size already is a multiple of.
            let (size, alignment) = vector_allocation(*scalar, *rows)?;
            let array = round_up(size, alignment)?.checked_mul(u64::from(*cols))?;
            round_up(array, alignment.max(1))
        }
    }
}

/// An AIR scalar's LLVM allocation size and alignment with no datalayout.
fn scalar_allocation(scalar: AirScalar) -> (u64, u64) {
    let bytes = scalar_bits(scalar).div_ceil(8);
    (bytes, bytes.next_power_of_two().min(8))
}

fn scalar_bits(scalar: AirScalar) -> u64 {
    match scalar {
        AirScalar::Float | AirScalar::UInt | AirScalar::SInt => 32,
        AirScalar::Half | AirScalar::UShort | AirScalar::SShort => 16,
        AirScalar::ULong | AirScalar::SLong => 64,
        AirScalar::UChar | AirScalar::Bool => 8,
    }
}

/// An LLVM vector's allocation size and alignment with no datalayout: its store size, aligned to
/// the next power of two.
fn vector_allocation(scalar: AirScalar, lanes: u32) -> Option<(u64, u64)> {
    let store_bits = u32::try_from(scalar_bits(scalar).checked_mul(u64::from(lanes))?).ok()?;
    let store_size = u64::from(store_bits).div_ceil(8);
    let alignment = store_size.checked_next_power_of_two()?;
    Some((round_up(store_size, alignment)?, alignment))
}

fn round_up(value: u64, alignment: u64) -> Option<u64> {
    let remainder = value.checked_rem(alignment)?;
    value.checked_add(if remainder == 0 {
        0
    } else {
        alignment - remainder
    })
}
