//! Object and mesh functions' operations: an object function's
//! `air.set_threadgroups_per_grid_mesh_properties`, and a mesh function's `air.set_*_mesh` writes
//! of its output's vertices, indices, primitives and primitive count. Each takes its handle (the
//! `air.mesh_grid_properties` or `air.mesh` argument) first; a slot, vertex, index or primitive
//! operand is an `i32` value.

use super::sig::{HandleKind, Operand, Signature};
use super::ty::{Elem, NumTy};

fn num(elem: Elem, lanes: u32) -> Operand {
    Operand::Num(NumTy { elem, lanes })
}

/// An object or mesh function's operation, named by what it sets (`air.set_<what>_mesh`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Mesh {
    /// `air.set_threadgroups_per_grid_mesh_properties`: `void (grid, <3 x i32>)`.
    ThreadgroupsPerGrid,
    /// `air.set_position_mesh`: `void (mesh, i32 vertex, <4 x float>)`.
    Position,
    /// `air.set_point_size_mesh`: `void (mesh, i32 vertex, float)`.
    PointSize,
    /// `air.set_clip_distance_mesh`: `void (mesh, i32 vertex, i32 index, float)`.
    ClipDistance,
    /// `air.set_vertex_data_mesh.<T>`: `void (mesh, i32 slot, i32 vertex, T)`.
    VertexData(NumTy),
    /// `air.set_index_mesh`: `void (mesh, i32 index, i8)`.
    Index,
    /// `air.set_indices_mesh.v<N>i8`: `void (mesh, i32 index, <N x i8>)`, `N` 2 or 4.
    Indices(u32),
    /// `air.set_primitive_data_mesh.<T>`: `void (mesh, i32 slot, i32 primitive, T)`.
    PrimitiveData(NumTy),
    /// `air.set_primitive_culled_mesh`: `void (mesh, i32 primitive, i1)`.
    PrimitiveCulled,
    /// `air.set_primitive_id_mesh`: `void (mesh, i32 primitive, i32)`.
    PrimitiveId,
    /// `air.set_render_target_array_index_mesh.<T>`: `void (mesh, i32 primitive, T)`, `T` an
    /// integer scalar.
    RenderTargetArrayIndex(NumTy),
    /// `air.set_viewport_array_index_mesh.<T>`: `void (mesh, i32 primitive, T)`, `T` an integer
    /// scalar.
    ViewportArrayIndex(NumTy),
    /// `air.set_primitive_count_mesh`: `void (mesh, i32)`.
    PrimitiveCount,
}

impl Mesh {
    pub(super) fn parse(rest: &str) -> Option<Mesh> {
        Some(match rest {
            "set_threadgroups_per_grid_mesh_properties" => Mesh::ThreadgroupsPerGrid,
            "set_position_mesh" => Mesh::Position,
            "set_point_size_mesh" => Mesh::PointSize,
            "set_clip_distance_mesh" => Mesh::ClipDistance,
            "set_index_mesh" => Mesh::Index,
            "set_primitive_culled_mesh" => Mesh::PrimitiveCulled,
            "set_primitive_id_mesh" => Mesh::PrimitiveId,
            "set_primitive_count_mesh" => Mesh::PrimitiveCount,
            _ => {
                let (stem, ty) = rest.split_once('.')?;
                let ty = NumTy::parse(ty)?;
                let data = matches!(
                    ty.elem,
                    Elem::F16 | Elem::F32 | Elem::I8 | Elem::I16 | Elem::I32
                ) && matches!(ty.lanes, 1..=4);
                let index = ty.lanes == 1 && matches!(ty.elem, Elem::I8 | Elem::I16 | Elem::I32);
                match stem {
                    "set_vertex_data_mesh" if data => Mesh::VertexData(ty),
                    "set_primitive_data_mesh" if data => Mesh::PrimitiveData(ty),
                    "set_indices_mesh" if ty.elem == Elem::I8 && matches!(ty.lanes, 2 | 4) => {
                        Mesh::Indices(ty.lanes)
                    }
                    "set_render_target_array_index_mesh" if index => {
                        Mesh::RenderTargetArrayIndex(ty)
                    }
                    "set_viewport_array_index_mesh" if index => Mesh::ViewportArrayIndex(ty),
                    _ => return None,
                }
            }
        })
    }

    /// The typed signature.
    pub fn signature(&self) -> Signature {
        let mesh = Operand::Handle(HandleKind::Mesh);
        let i32s = num(Elem::I32, 1);
        let f32s = num(Elem::F32, 1);
        match *self {
            Mesh::ThreadgroupsPerGrid => Signature::new(
                Operand::Void,
                &[Operand::Handle(HandleKind::MeshGrid), num(Elem::I32, 3)],
            ),
            Mesh::Position => Signature::new(Operand::Void, &[mesh, i32s, num(Elem::F32, 4)]),
            Mesh::PointSize => Signature::new(Operand::Void, &[mesh, i32s, f32s]),
            Mesh::ClipDistance => Signature::new(Operand::Void, &[mesh, i32s, i32s, f32s]),
            Mesh::VertexData(t) | Mesh::PrimitiveData(t) => {
                Signature::new(Operand::Void, &[mesh, i32s, i32s, Operand::Num(t)])
            }
            Mesh::Index => Signature::new(Operand::Void, &[mesh, i32s, num(Elem::I8, 1)]),
            Mesh::Indices(n) => Signature::new(Operand::Void, &[mesh, i32s, num(Elem::I8, n)]),
            Mesh::PrimitiveCulled => Signature::new(Operand::Void, &[mesh, i32s, num(Elem::I1, 1)]),
            Mesh::PrimitiveId => Signature::new(Operand::Void, &[mesh, i32s, i32s]),
            Mesh::RenderTargetArrayIndex(t) | Mesh::ViewportArrayIndex(t) => {
                Signature::new(Operand::Void, &[mesh, i32s, Operand::Num(t)])
            }
            Mesh::PrimitiveCount => Signature::new(Operand::Void, &[mesh, i32s]),
        }
    }
}
