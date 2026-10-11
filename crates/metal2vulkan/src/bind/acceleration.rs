//! Acceleration structures as the consumer builds them for Metal's.
//!
//! The binder never issues Vulkan calls: the consumer builds a Vulkan structure for each Metal
//! one it is asked to build, and a plan writes the top-level structure of the Metal structure
//! bound at an argument's slot (`WriteSource::AccelerationStructure`). This module gives what
//! those builds must hold for the translation's ray queries to read Metal's values:
//!
//! - **Geometry.** A primitive structure's geometry keeps Metal's opacity and duplicate-call
//!   settings ([`geometry_flags`]). Metal adds each geometry's `intersectionFunctionTableOffset`
//!   to its instance's to index the intersection function table; a ray query reads only the
//!   instance's shader-binding-table record offset, so the offsets of one structure's geometries
//!   must be equal, and the shared one moves to the instance ([`geometry_table_offset`]).
//! - **Instances.** An instance of a top-level structure is Vulkan's
//!   `VkAccelerationStructureInstanceKHR` ([`instance_record`]): Metal's transform, its mask in
//!   the 8 mask bits, its user id as the 24-bit custom index (the result's `user_instance_id`),
//!   its table offset plus its structure's geometry offset as the 24-bit shader-binding-table
//!   record offset (where the translation reads a candidate's table entry), and its options as the
//!   instance flags, bit for bit: disable triangle culling (1), counterclockwise front facing (2),
//!   opaque (4), and non-opaque (8) are Vulkan's `TRIANGLE_FACING_CULL_DISABLE`,
//!   `TRIANGLE_FLIP_FACING` (`..._FRONT_COUNTERCLOCKWISE`), `FORCE_OPAQUE`, and `FORCE_NO_OPAQUE`.
//!   Vulkan compares 8 mask bits and keeps 24 custom-index and 24 offset bits, so a mask of 256 or
//!   more, a user id of `2^24` or more, and an offset of `2^24` or more are refused
//!   (`NotRepresentable`): Metal compares all 32 mask bits, which no 8-bit instance mask reproduces
//!   for every ray mask.
//! - **The one-instance wrap.** A `primitive_acceleration_structure` argument is bound through a
//!   top-level structure of one instance ([`primitive_wrap`]): the identity transform, mask
//!   `0xFF` (the translation traces a primitive intersector with ray mask `0xFF`), instance index
//!   and custom index 0, no flags (the geometry's own opacity decides), and the geometry's table
//!   offset as its record offset.
//!
//! Motion instances (`MTLAccelerationStructureMotionInstanceDescriptor`) have no representation:
//! a ray query takes no time. A Vulkan instance references a bottom-level structure only, so an
//! instance of an instance structure (Metal's multi-level instancing, a [`MetalInstance`] whose
//! `structure` is an instance structure) is refused (`NotImplemented`): flattening a hierarchy into
//! one top-level structure, with the per-level instance ids a multi-level intersector reads, is not
//! built. A multi-level intersector over a structure the consumer can build traverses its one
//! level, which the translation reads as such.

use crate::error::{Feature, Unsupported};
use crate::pipeline::AccelerationStructureLevel;

/// `VK_GEOMETRY_OPAQUE_BIT_KHR`.
pub const GEOMETRY_OPAQUE: u32 = 0x1;
/// `VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR`.
pub const GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION: u32 = 0x2;

/// Metal's instance options (`MTLAccelerationStructureInstanceOptions`), which are Vulkan's
/// instance flags (`VkGeometryInstanceFlagBitsKHR`) bit for bit: disable triangle culling (1),
/// counterclockwise front facing (2), opaque (4), and non-opaque (8).
const INSTANCE_OPTIONS: u32 = 0xF;
/// One past the largest 24-bit field value.
const FIELD_24: u32 = 1 << 24;
/// One past the largest 8-bit instance mask.
const MASK_8: u32 = 1 << 8;

/// One Metal instance of a top-level structure: an `MTLAccelerationStructureInstanceDescriptor`
/// or its `UserID` form, as the consumer read it from the guest's instance descriptors.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MetalInstance {
    /// `transformationMatrix`, an `MTLPackedFloat4x3`: four columns of three floats.
    pub transform: [[f32; 3]; 4],
    /// `options` (`MTLAccelerationStructureInstanceOptions`).
    pub options: u32,
    /// `mask`.
    pub mask: u32,
    /// `intersectionFunctionTableOffset`.
    pub intersection_function_table_offset: u32,
    /// `userID` of a user-ID descriptor. A default descriptor has none, and what Metal reads as
    /// its instance's `user_instance_id` is unmeasured; a consumer passes 0.
    pub user_id: u32,
    /// The level of the structure the instance references (its `accelerationStructureIndex`'s
    /// entry of `instancedAccelerationStructures`): a primitive structure, or an instance
    /// structure (Metal's multi-level instancing, refused).
    pub structure: AccelerationStructureLevel,
}

impl MetalInstance {
    /// An instance of a primitive structure with the identity transform and no options, mask,
    /// offset, or user id.
    pub fn new() -> MetalInstance {
        MetalInstance {
            transform: IDENTITY,
            options: 0,
            mask: 0,
            intersection_function_table_offset: 0,
            user_id: 0,
            structure: AccelerationStructureLevel::Primitive,
        }
    }
}

impl Default for MetalInstance {
    fn default() -> Self {
        MetalInstance::new()
    }
}

/// The identity transform, as four columns.
const IDENTITY: [[f32; 3]; 4] = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
    [0.0, 0.0, 0.0],
];

/// The 64 bytes of one `VkAccelerationStructureInstanceKHR`, as the consumer writes them into a
/// top-level build's instance buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct InstanceRecord([u8; InstanceRecord::BYTES]);

impl InstanceRecord {
    /// The size of one record.
    pub const BYTES: usize = 64;

    /// The record's bytes.
    pub fn bytes(&self) -> &[u8; InstanceRecord::BYTES] {
        &self.0
    }

    fn new(
        transform: &[[f32; 3]; 4],
        custom_index: u32,
        mask: u32,
        record_offset: u32,
        flags: u32,
        reference: u64,
    ) -> InstanceRecord {
        let mut b = [0u8; InstanceRecord::BYTES];
        // `VkTransformMatrixKHR`: three rows of four floats; row `r`, column `c` is column `c`'s
        // component `r`.
        for r in 0..3 {
            for (c, column) in transform.iter().enumerate() {
                let at = (r * 4 + c) * 4;
                if let (Some(dst), Some(v)) = (b.get_mut(at..at + 4), column.get(r)) {
                    dst.copy_from_slice(&v.to_le_bytes());
                }
            }
        }
        let words = [
            (48, (custom_index & (FIELD_24 - 1)) | (mask << 24)),
            (52, (record_offset & (FIELD_24 - 1)) | (flags << 24)),
        ];
        for (at, w) in words {
            if let Some(dst) = b.get_mut(at..at + 4) {
                dst.copy_from_slice(&w.to_le_bytes());
            }
        }
        if let Some(dst) = b.get_mut(56..64) {
            dst.copy_from_slice(&reference.to_le_bytes());
        }
        InstanceRecord(b)
    }
}

/// The Vulkan geometry flags (`VkGeometryFlagsKHR`) of a Metal geometry descriptor with `opaque`
/// and `allowDuplicateIntersectionFunctionInvocation` (both `YES` by default in Metal).
pub fn geometry_flags(opaque: bool, allow_duplicate_invocation: bool) -> u32 {
    let mut f = 0;
    if opaque {
        f |= GEOMETRY_OPAQUE;
    }
    if !allow_duplicate_invocation {
        f |= GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION;
    }
    f
}

/// The intersection function table offset every geometry of a primitive structure shares, given
/// each geometry's `intersectionFunctionTableOffset` (an `NSUInteger`; 0 for a structure without
/// geometry). An offset of `2^24` or more is refused (`NotRepresentable`: the 24-bit record
/// offset cannot hold it, module docs). A ray query reads no per-geometry offset, so different
/// offsets are not implemented.
pub fn geometry_table_offset(offsets: &[u64]) -> Result<u32, Unsupported> {
    let field = u64::from(FIELD_24);
    if offsets.iter().any(|&o| o >= field) {
        return Err(Unsupported::not_representable(
            Feature::IntersectionFunctionTable,
        ));
    }
    let first = offsets.first().copied().unwrap_or(0);
    if !offsets.iter().all(|&o| o == first) {
        return Err(Unsupported::not_implemented(
            Feature::IntersectionFunctionTable,
        ));
    }
    u32::try_from(first)
        .map_err(|_| Unsupported::not_representable(Feature::IntersectionFunctionTable))
}

/// The record of Metal instance `instance` of the primitive structure whose Vulkan device address
/// is `reference` and whose geometries share the table offset `geometry_table_offset`
/// ([`geometry_table_offset`]), or the refusal of what Vulkan's fields cannot hold, and of an
/// instance of an instance structure (module docs).
pub fn instance_record(
    instance: &MetalInstance,
    reference: u64,
    geometry_table_offset: u32,
) -> Result<InstanceRecord, Unsupported> {
    // A Vulkan instance references a bottom-level structure only.
    if instance.structure != AccelerationStructureLevel::Primitive {
        return Err(Unsupported::not_implemented(Feature::RayTracing));
    }
    let too_wide = |f: Feature| Err(Unsupported::not_representable(f));
    if instance.mask >= MASK_8 || instance.user_id >= FIELD_24 {
        return too_wide(Feature::RayTracing);
    }
    let offset = instance
        .intersection_function_table_offset
        .checked_add(geometry_table_offset)
        .filter(|&o| o < FIELD_24);
    let Some(offset) = offset else {
        return too_wide(Feature::IntersectionFunctionTable);
    };
    if instance.options & !INSTANCE_OPTIONS != 0 {
        return Err(Unsupported::not_implemented(Feature::RayTracing));
    }
    Ok(InstanceRecord::new(
        &instance.transform,
        instance.user_id,
        instance.mask,
        offset,
        instance.options,
        reference,
    ))
}

/// The one instance of the top-level structure a `primitive_acceleration_structure` argument is
/// bound through: the primitive structure at device address `reference` whose geometries share the
/// table offset `geometry_table_offset`, with the identity transform, mask `0xFF`, custom index 0,
/// and no flags.
pub fn primitive_wrap(
    reference: u64,
    geometry_table_offset: u32,
) -> Result<InstanceRecord, Unsupported> {
    let instance = MetalInstance {
        mask: 0xFF,
        ..MetalInstance::new()
    };
    instance_record(&instance, reference, geometry_table_offset)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(r: &InstanceRecord, at: usize) -> u32 {
        let b = r.bytes();
        u32::from_le_bytes([b[at], b[at + 1], b[at + 2], b[at + 3]])
    }

    /// A record is Vulkan's instance layout: the transform's rows, the custom index under the
    /// mask, the record offset under the flags, and the structure's address.
    #[test]
    fn a_record_is_vulkans_instance_layout() {
        let instance = MetalInstance {
            transform: [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
                [10.0, 11.0, 12.0],
            ],
            options: 4 | 1,
            mask: 0x7F,
            intersection_function_table_offset: 3,
            user_id: 0x00AB_CDEF,
            structure: AccelerationStructureLevel::Primitive,
        };
        let r = instance_record(&instance, 0x1122_3344_5566_7788, 2).expect("fits");
        let rows: Vec<f32> = (0..12).map(|i| f32::from_bits(word(&r, i * 4))).collect();
        assert_eq!(
            rows,
            [1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0]
        );
        assert_eq!(word(&r, 48), 0x7FAB_CDEF);
        assert_eq!(word(&r, 52), (5 << 24) | 5);
        assert_eq!(
            u64::from(word(&r, 56)) | (u64::from(word(&r, 60)) << 32),
            0x1122_3344_5566_7788
        );
    }

    /// Metal's instance options are Vulkan's instance flags, bit for bit.
    #[test]
    fn options_are_the_instance_flags() {
        for option in [1, 2, 4, 8] {
            let instance = MetalInstance {
                options: option,
                ..MetalInstance::new()
            };
            let r = instance_record(&instance, 0, 0).expect("fits");
            assert_eq!(word(&r, 52) >> 24, option);
        }
        let unknown = MetalInstance {
            options: 16,
            ..MetalInstance::new()
        };
        assert_eq!(
            instance_record(&unknown, 0, 0),
            Err(Unsupported::not_implemented(Feature::RayTracing))
        );
    }

    /// What Vulkan's instance fields cannot hold is refused as not representable: a mask of 256
    /// or more, a user id of `2^24` or more, and a table offset of `2^24` or more.
    #[test]
    fn wide_fields_are_not_representable() {
        let ray = Err(Unsupported::not_representable(Feature::RayTracing));
        let table = Err(Unsupported::not_representable(
            Feature::IntersectionFunctionTable,
        ));
        let at = |mask, user_id, offset, geometry| {
            let instance = MetalInstance {
                mask,
                user_id,
                intersection_function_table_offset: offset,
                ..MetalInstance::new()
            };
            instance_record(&instance, 0, geometry)
        };
        assert!(at(0xFF, FIELD_24 - 1, FIELD_24 - 1, 0).is_ok());
        assert_eq!(at(0x100, 0, 0, 0), ray);
        assert_eq!(at(u32::MAX, 0, 0, 0), ray);
        assert_eq!(at(0, FIELD_24, 0, 0), ray);
        assert_eq!(at(0, 0, FIELD_24, 0), table);
        assert_eq!(at(0, 0, FIELD_24 - 1, 1), table);
        assert_eq!(at(0, 0, u32::MAX, 1), table);
    }

    /// An instance of an instance structure (Metal's multi-level instancing) has no Vulkan
    /// record: a Vulkan instance references a bottom-level structure only.
    #[test]
    fn an_instance_of_an_instance_structure_is_refused() {
        let nested = MetalInstance {
            structure: AccelerationStructureLevel::Instance,
            mask: 0xFF,
            ..MetalInstance::new()
        };
        assert_eq!(
            instance_record(&nested, 0, 0),
            Err(Unsupported::not_implemented(Feature::RayTracing))
        );
        let leaf = MetalInstance {
            structure: AccelerationStructureLevel::Primitive,
            ..nested
        };
        assert!(instance_record(&leaf, 0, 0).is_ok());
    }

    /// A primitive argument's one instance is the identity, mask `0xFF`, custom index 0, no
    /// flags, and the geometries' table offset.
    #[test]
    fn the_primitive_wrap_is_one_identity_instance() {
        let r = primitive_wrap(0xABCD, 6).expect("fits");
        let rows: Vec<f32> = (0..12).map(|i| f32::from_bits(word(&r, i * 4))).collect();
        assert_eq!(
            rows,
            [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        );
        assert_eq!(word(&r, 48), 0xFF00_0000);
        assert_eq!(word(&r, 52), 6);
        assert_eq!(word(&r, 56), 0xABCD);
    }

    /// Geometries share one table offset or are not implemented, and an offset the 24-bit record
    /// offset cannot hold, however wide (`NSUInteger`), is not representable rather than
    /// truncated; flags follow Metal's settings.
    #[test]
    fn geometries_share_their_table_offset() {
        assert_eq!(geometry_table_offset(&[]), Ok(0));
        assert_eq!(geometry_table_offset(&[4, 4, 4]), Ok(4));
        assert_eq!(
            geometry_table_offset(&[0xFF_FFFF, 0xFF_FFFF]),
            Ok(0xFF_FFFF)
        );
        assert_eq!(
            geometry_table_offset(&[0, 1]),
            Err(Unsupported::not_implemented(
                Feature::IntersectionFunctionTable
            ))
        );
        for wide in [1 << 24, 1 << 32, (1 << 32) + 4, u64::MAX] {
            assert_eq!(
                geometry_table_offset(&[wide, wide]),
                Err(Unsupported::not_representable(
                    Feature::IntersectionFunctionTable
                )),
                "{wide:#x}"
            );
            assert_eq!(
                geometry_table_offset(&[4, wide]),
                Err(Unsupported::not_representable(
                    Feature::IntersectionFunctionTable
                )),
                "{wide:#x}"
            );
        }
        assert_eq!(geometry_flags(true, true), GEOMETRY_OPAQUE);
        assert_eq!(
            geometry_flags(false, false),
            GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION
        );
    }

    /// The raw values are Vulkan's.
    #[cfg(feature = "ash")]
    #[test]
    fn the_flags_are_vulkans() {
        use ash::vk;
        assert_eq!(GEOMETRY_OPAQUE, vk::GeometryFlagsKHR::OPAQUE.as_raw());
        assert_eq!(
            GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION,
            vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION.as_raw()
        );
        let flags = [
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FACING_CULL_DISABLE,
            vk::GeometryInstanceFlagsKHR::TRIANGLE_FLIP_FACING,
            vk::GeometryInstanceFlagsKHR::FORCE_OPAQUE,
            vk::GeometryInstanceFlagsKHR::FORCE_NO_OPAQUE,
        ];
        for (option, flag) in [1, 2, 4, 8].into_iter().zip(flags) {
            assert_eq!(option, flag.as_raw());
        }
        assert_eq!(
            InstanceRecord::BYTES,
            std::mem::size_of::<vk::AccelerationStructureInstanceKHR>()
        );
    }
}
