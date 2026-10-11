//! Ray intersection: the intersector's `intersect` calls, `air.intersect[.<tag>]*`.
//!
//! Metal's `intersector<Tags...>::intersect` lowers to one intrinsic whose name spells the
//! intersector's tags in their ABI bit order (`intersection_function_buffer` 1, `instancing` 2,
//! `max_levels` 4, `triangle_data` 8, `curve_data` 16, `world_space_data` 32, `user_data` 64,
//! `primitive_motion` 128, `instance_motion` 256, `extended_limits` 512). `max_levels<N>` with
//! `N > 2` spells `multi_level_instancing` in place of `instancing`; `max_levels<2>` is plain
//! `instancing`. An intersector without tags calls `air.intersect` itself.
//!
//! The tags decide the operand list and the result struct; every setting of the intersector
//! (winding, the cull modes, forced opacity, the assumed geometry and curve shapes, identity
//! transforms, accept-any) is an ordinary operand, often a runtime value, whose meaning lowering
//! owns. Operand roles ([`Role`]) name each position, in the order the Metal compiler passes
//! them (read from Apple's `metal_raytracing` header and confirmed on compiled probes):
//!
//! `origin <3 x float>, direction <3 x float>, min_distance float, max_distance float,
//! structure ptr addrspace(1)`, then `mask i32` (instancing), `time float` (either motion tag),
//! the intersection function table `ptr addrspace(1)` (or, with `intersection_function_buffer`,
//! the buffer `ptr addrspace(1)` with its size and stride `i64, i64`), `user_data ptr
//! addrspace(1)` (`user_data`), the payload `ptr` and its size `i64`, then `i8 max_levels, ptr
//! instance_ids, ptr user_instance_ids` (multi-level), the nine `i32` settings (front-facing
//! winding, triangle cull mode, geometry cull mode, opacity cull mode, forced opacity, geometry
//! type, curve basis, curve type, curve control-point count), `i1 assume_identity_transforms`
//! (instancing), `i1 accept_any_intersection`, and `i32 base_id, i32 geometry_multiplier`
//! (`intersection_function_buffer`).
//!
//! The result is the literal struct `{ i32 type, float distance, i32 primitive_id, i32
//! geometry_id, ptr addrspace(1) primitive_data }`, then `i32 instance_id, i32
//! user_instance_id` (instancing) or `i8 instance_count` (multi-level; the ids are written
//! through the two pointers), `<2 x float> barycentrics, i1 front_facing` (`triangle_data`),
//! `float curve_parameter` (`curve_data`), and the world-to-object then object-to-world
//! transforms as eight `<3 x float>` columns (`world_space_data`).
//!
//! A name whose tags are out of order, repeated, or combined in a way Apple's header rejects
//! (`user_data` without a function buffer, `world_space_data` or `instance_motion` without
//! instancing) is not decoded and stays an unknown intrinsic.

use super::layout::Layout;
use super::sig::{HandleKind, Operand, Signature};
use super::ty::{Elem, NumTy};
use crate::ir::AddrSpace;

/// How many levels of instances a query traverses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Instancing {
    /// A primitive acceleration structure.
    None,
    /// `instancing` (with at most `max_levels<2>`): instances of primitive structures.
    Single,
    /// `multi_level_instancing` (`max_levels<N>`, `N > 2`): instances of instance structures.
    Multi,
}

/// An intersector's tags, which decide its call's operands and result.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IntersectTags {
    /// `intersection_function_buffer`: intersection functions come from a buffer of function
    /// handles (Metal 4) rather than an intersection function table.
    pub function_buffer: bool,
    /// `instancing` or `multi_level_instancing`: the instance levels the query traverses.
    pub instancing: Instancing,
    /// `triangle_data`: the result has the barycentrics and the front-facing flag.
    pub triangle_data: bool,
    /// `curve_data`: the result has the curve parameter.
    pub curve_data: bool,
    /// `world_space_data`: the result has the world-to-object and object-to-world transforms.
    pub world_space_data: bool,
    /// `user_data`: a user-data buffer passed to the buffer's functions.
    pub user_data: bool,
    /// `primitive_motion`: primitive motion; the call takes a `time` operand.
    pub primitive_motion: bool,
    /// `instance_motion`: instance motion; the call takes a `time` operand.
    pub instance_motion: bool,
    /// `extended_limits`: larger acceleration structures; no operand or result field.
    pub extended_limits: bool,
}

impl IntersectTags {
    /// No tag: `intersector<>`.
    pub const NONE: IntersectTags = IntersectTags {
        function_buffer: false,
        instancing: Instancing::None,
        triangle_data: false,
        curve_data: false,
        world_space_data: false,
        user_data: false,
        primitive_motion: false,
        instance_motion: false,
        extended_limits: false,
    };

    /// Whether either motion tag is present (they share the one `time` operand).
    pub const fn motion(&self) -> bool {
        self.primitive_motion || self.instance_motion
    }

    /// Whether the query traverses instances (`instancing` or `multi_level_instancing`).
    pub const fn instanced(&self) -> bool {
        !matches!(self.instancing, Instancing::None)
    }

    /// The result struct's shape.
    pub const fn result(&self) -> ResultShape {
        ResultShape {
            instancing: self.instancing,
            triangle_data: self.triangle_data,
            curve_data: self.curve_data,
            world_space_data: self.world_space_data,
        }
    }
}

/// The shape of an intersection result struct: the tags that add fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResultShape {
    pub instancing: Instancing,
    pub triangle_data: bool,
    pub curve_data: bool,
    pub world_space_data: bool,
}

/// One field of an intersection result struct.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ResultField {
    /// `i32`: `intersection_type` (none 0, triangle 1, bounding box 2, curve 3).
    Type,
    /// `float`: the distance along the ray to the committed intersection.
    Distance,
    /// `i32`.
    PrimitiveId,
    /// `i32`.
    GeometryId,
    /// `ptr addrspace(1)`: the primitive's data in the acceleration structure.
    PrimitiveData,
    /// `i32` (single-level instancing).
    InstanceId,
    /// `i32` (single-level instancing).
    UserInstanceId,
    /// `i8`: the instance levels traversed (multi-level instancing).
    InstanceCount,
    /// `<2 x float>` (`triangle_data`).
    Barycentrics,
    /// `i1` (`triangle_data`).
    FrontFacing,
    /// `float` (`curve_data`).
    CurveParameter,
    /// `<3 x float>`: column `n` of the world-to-object transform (`world_space_data`).
    WorldToObject(u8),
    /// `<3 x float>`: column `n` of the object-to-world transform (`world_space_data`).
    ObjectToWorld(u8),
}

impl ResultField {
    /// The field's IR shape: a number, or the device pointer of [`ResultField::PrimitiveData`].
    pub const fn operand(self) -> Operand {
        let f32s = Operand::Num(NumTy::scalar(Elem::F32));
        let i32s = Operand::Num(NumTy::scalar(Elem::I32));
        match self {
            ResultField::Type
            | ResultField::PrimitiveId
            | ResultField::GeometryId
            | ResultField::InstanceId
            | ResultField::UserInstanceId => i32s,
            ResultField::Distance | ResultField::CurveParameter => f32s,
            ResultField::PrimitiveData => Operand::Ptr(AddrSpace::Device),
            ResultField::InstanceCount => Operand::Num(NumTy::scalar(Elem::I8)),
            ResultField::Barycentrics => Operand::Num(NumTy {
                elem: Elem::F32,
                lanes: 2,
            }),
            ResultField::FrontFacing => Operand::Num(NumTy::scalar(Elem::I1)),
            ResultField::WorldToObject(_) | ResultField::ObjectToWorld(_) => Operand::Num(NumTy {
                elem: Elem::F32,
                lanes: 3,
            }),
        }
    }
}

impl ResultShape {
    /// The fields, in struct order.
    pub fn fields(&self) -> impl Iterator<Item = ResultField> + '_ {
        let base = [
            ResultField::Type,
            ResultField::Distance,
            ResultField::PrimitiveId,
            ResultField::GeometryId,
            ResultField::PrimitiveData,
        ];
        let instance: &[ResultField] = match self.instancing {
            Instancing::None => &[],
            Instancing::Single => &[ResultField::InstanceId, ResultField::UserInstanceId],
            Instancing::Multi => &[ResultField::InstanceCount],
        };
        let triangle: &[ResultField] = if self.triangle_data {
            &[ResultField::Barycentrics, ResultField::FrontFacing]
        } else {
            &[]
        };
        let curve: &[ResultField] = if self.curve_data {
            &[ResultField::CurveParameter]
        } else {
            &[]
        };
        let world = (0..4)
            .map(ResultField::WorldToObject)
            .chain((0..4).map(ResultField::ObjectToWorld))
            .filter(move |_| self.world_space_data);
        base.into_iter()
            .chain(instance.iter().copied())
            .chain(triangle.iter().copied())
            .chain(curve.iter().copied())
            .chain(world)
    }

    /// The index of `field` in the struct, if the shape has it.
    pub fn index(&self, field: ResultField) -> Option<u32> {
        self.fields()
            .position(|f| f == field)
            .and_then(|i| u32::try_from(i).ok())
    }
}

/// What one operand position of an `intersect` call is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// The ray's origin, `<3 x float>`.
    Origin,
    /// The ray's direction, `<3 x float>`.
    Direction,
    /// The ray's minimum distance, `float`.
    MinDistance,
    /// The ray's maximum distance, `float`.
    MaxDistance,
    /// The acceleration structure.
    Structure,
    /// The instance mask (instancing).
    Mask,
    /// The motion time (either motion tag).
    Time,
    /// The intersection function table (not with a function buffer).
    Table,
    /// The intersection function buffer, its size, and its stride.
    FunctionBuffer,
    /// The intersection function buffer's size, `i64`.
    FunctionBufferSize,
    /// The intersection function buffer's stride, `i64`.
    FunctionStride,
    /// The user-data buffer (`user_data`).
    UserData,
    /// The payload the intersection functions receive, and its size in bytes.
    Payload,
    /// The payload's size in bytes, `i64`.
    PayloadSize,
    /// The instance levels to traverse and the arrays the ids of each level are written to
    /// (multi-level instancing).
    MaxLevels,
    /// The array the instance id of each level is written to (multi-level instancing).
    InstanceIds,
    /// The array the user instance id of each level is written to (multi-level instancing).
    UserInstanceIds,
    /// `winding`: clockwise 0, counterclockwise 1.
    Winding,
    /// `triangle_cull_mode`: none 0, front 1, back 2.
    TriangleCull,
    /// `geometry_cull_mode`, a mask: triangle 1, bounding box 2, curve 4.
    GeometryCull,
    /// `opacity_cull_mode`: none 0, opaque 1, non-opaque 2.
    OpacityCull,
    /// `forced_opacity`: none 0, opaque 1, non-opaque 2.
    ForcedOpacity,
    /// `geometry_type`, a mask: triangle 1, bounding box 2, curve 4.
    GeometryType,
    /// `curve_basis`: the assumed curve basis (`i32`).
    CurveBasis,
    /// `curve_type`: the assumed curve type (`i32`).
    CurveType,
    /// `curve_control_point_count`: the assumed control points per curve (`i32`).
    CurveControlPoints,
    /// `assume_identity_transforms` (instancing).
    IdentityTransforms,
    /// `accept_any_intersection`.
    AcceptAny,
    /// The function buffer's base id and geometry multiplier.
    BaseId,
    /// The intersection function buffer's geometry multiplier (`i32`).
    GeometryMultiplier,
}

/// A decoded `air.intersect` call: the intersector's tags.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Intersect {
    /// The intersector's tags, as the name spells them.
    pub tags: IntersectTags,
}

/// The tag tokens in ABI bit order; `instancing` and `multi_level_instancing` share the second
/// place.
const TOKENS: [&str; 10] = [
    "intersection_function_buffer",
    "instancing",
    "multi_level_instancing",
    "triangle_data",
    "curve_data",
    "world_space_data",
    "user_data",
    "primitive_motion",
    "instance_motion",
    "extended_limits",
];

impl Intersect {
    /// Parse the part of an `air.*` name after `air.`: `intersect` and its tags.
    pub(super) fn parse(rest: &str) -> Option<Intersect> {
        let tags = rest.strip_prefix("intersect")?;
        let mut t = IntersectTags::NONE;
        let mut next = 0;
        if !tags.is_empty() {
            for token in tags.strip_prefix('.')?.split('.') {
                let at = TOKENS.iter().position(|k| *k == token)?;
                // Strictly increasing, with the two instancing spellings exclusive.
                let place = if at >= 2 { at - 1 } else { at };
                if place < next {
                    return None;
                }
                next = place + 1;
                match at {
                    0 => t.function_buffer = true,
                    1 => t.instancing = Instancing::Single,
                    2 => t.instancing = Instancing::Multi,
                    3 => t.triangle_data = true,
                    4 => t.curve_data = true,
                    5 => t.world_space_data = true,
                    6 => t.user_data = true,
                    7 => t.primitive_motion = true,
                    8 => t.instance_motion = true,
                    _ => t.extended_limits = true,
                }
            }
        }
        let consistent = (!t.user_data || t.function_buffer)
            && (!t.world_space_data || t.instanced())
            && (!t.instance_motion || t.instanced());
        consistent.then_some(Intersect { tags: t })
    }

    /// The operand positions with their roles.
    pub fn layout(&self) -> Layout<Role> {
        let t = self.tags;
        let num = |elem: Elem, lanes: u32| Operand::Num(NumTy { elem, lanes });
        let (f32s, i32s, i64s, i1s) = (
            num(Elem::F32, 1),
            num(Elem::I32, 1),
            num(Elem::I64, 1),
            num(Elem::I1, 1),
        );
        let device = Operand::Ptr(AddrSpace::Device);
        let thread = Operand::Ptr(AddrSpace::Thread);
        let mut l = Layout::new(Operand::IntersectionResult(t.result()));
        l.push(Role::Origin, num(Elem::F32, 3));
        l.push(Role::Direction, num(Elem::F32, 3));
        l.push(Role::MinDistance, f32s);
        l.push(Role::MaxDistance, f32s);
        l.push(
            Role::Structure,
            Operand::Handle(HandleKind::AccelerationStructure),
        );
        l.push_if(t.instanced(), Role::Mask, i32s);
        l.push_if(t.motion(), Role::Time, f32s);
        if t.function_buffer {
            l.push(Role::FunctionBuffer, device);
            l.push(Role::FunctionBufferSize, i64s);
            l.push(Role::FunctionStride, i64s);
        } else {
            l.push(
                Role::Table,
                Operand::Handle(HandleKind::IntersectionFunctionTable),
            );
        }
        l.push_if(t.user_data, Role::UserData, device);
        l.push(Role::Payload, thread);
        l.push(Role::PayloadSize, i64s);
        if matches!(t.instancing, Instancing::Multi) {
            l.push(Role::MaxLevels, num(Elem::I8, 1));
            l.push(Role::InstanceIds, thread);
            l.push(Role::UserInstanceIds, thread);
        }
        for role in [
            Role::Winding,
            Role::TriangleCull,
            Role::GeometryCull,
            Role::OpacityCull,
            Role::ForcedOpacity,
            Role::GeometryType,
            Role::CurveBasis,
            Role::CurveType,
            Role::CurveControlPoints,
        ] {
            l.push(role, i32s);
        }
        l.push_if(t.instanced(), Role::IdentityTransforms, i1s);
        l.push(Role::AcceptAny, i1s);
        l.push_if(t.function_buffer, Role::BaseId, i32s);
        l.push_if(t.function_buffer, Role::GeometryMultiplier, i32s);
        l
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        self.layout().signature()
    }

    /// The operand index of `role`, if the call has it.
    pub fn index(&self, role: Role) -> Option<usize> {
        self.layout().index(role)
    }
}
