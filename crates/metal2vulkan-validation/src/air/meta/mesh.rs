//! The roles only an object or a mesh function declares: its payload (`air.payload`), an object
//! function's mesh grid (`air.mesh_grid_properties`), and a mesh function's output type
//! (`air.mesh` with `air.mesh_type_info`). An object or mesh entry's other parameters are a
//! kernel's (resources, threadgroup memory, grid builtins), parsed by the kernel's decode
//! ([`super::kernel`]), which collects these into [`MeshStageMeta`].

use super::*;

/// What an object or mesh function's metadata states beyond a kernel's parameters.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct MeshStageMeta {
    /// The `[[payload]]` parameter: an object function writes it, a mesh function reads it.
    pub payload: Option<PayloadMeta>,
    /// The parameter index of an object function's `mesh_grid_properties`.
    pub mesh_grid_param: Option<u32>,
    /// A mesh function's output type.
    pub outputs: Option<MeshOutputsMeta>,
    /// `air.max_mesh_work_groups` on an object function's root
    /// (`[[max_total_threadgroups_per_mesh_grid(N)]]`).
    pub max_mesh_work_groups: Option<u32>,
}

/// An object or mesh function's `air.payload` parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PayloadMeta {
    pub param_index: u32,
    /// `air.arg_type_size`.
    pub size: Option<u32>,
    /// `air.arg_type_align_size`.
    pub align: Option<u32>,
    /// `air.arg_type_name`.
    pub type_name: Option<String>,
    /// `air.struct_type_info`, reconstructed.
    pub layout: Option<AirType>,
}

/// A mesh function's output type, `mesh<V, P, NV, NP, t>` (`air.mesh_type_info`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshOutputsMeta {
    /// The parameter index of the `air.mesh` handle.
    pub param_index: u32,
    /// The vertex struct's fields, in declaration order.
    pub vertex: Vec<MeshFieldMeta>,
    /// The primitive struct's fields, in declaration order (none for a `void` primitive type).
    pub primitive: Vec<MeshFieldMeta>,
    /// `NV`.
    pub max_vertices: u32,
    /// `NP`.
    pub max_primitives: u32,
    pub topology: MeshTopology,
}

/// The primitive kind a mesh function writes (`air.point`, `air.line`, `air.triangle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshTopology {
    Point,
    Line,
    Triangle,
}

/// One field of a mesh function's vertex or primitive struct.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshFieldMeta {
    pub role: MeshFieldRole,
    /// `air.arg_type_name`.
    pub type_name: Option<String>,
}

/// What a mesh output field is: a builtin, or a user data field with the user key a fragment
/// input matches it by.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum MeshFieldRole {
    /// `air.position`.
    Position,
    /// `air.point_size`.
    PointSize,
    /// `air.clip_distance`.
    ClipDistance,
    /// `air.mesh_vertex_data, <slot>, <user key>`.
    VertexData { slot: u32, key: String },
    /// `air.primitive_culled`.
    PrimitiveCulled,
    /// `air.primitive_id`.
    PrimitiveId,
    /// `air.render_target_array_index`.
    RenderTargetArrayIndex,
    /// `air.viewport_array_index`.
    ViewportArrayIndex,
    /// `air.mesh_primitive_data, <slot>, <user key>`.
    PrimitiveData { slot: u32, key: String },
    /// A field role this reflection does not know, as AIR spells it: unknown stays unknown.
    Unrecognized(String),
}

/// The roles an object or mesh function's own parameters carry, beside a kernel's.
pub(crate) const MESH_STAGE_ROLES: &[&str] = &["mesh", "mesh_grid_properties", "payload"];

impl MeshStageMeta {
    /// Record the mesh-stage role `role` of the parameter `node` at `param_index`, if it is one.
    /// Returns whether it was.
    pub(crate) fn record(
        &mut self,
        role: &str,
        param_index: u32,
        node: &str,
        nodes: &HashMap<u32, String>,
    ) -> bool {
        match role {
            "payload" => {
                self.payload = Some(PayloadMeta {
                    param_index,
                    size: i32_after_marker(node, "air.arg_type_size"),
                    align: i32_after_marker(node, "air.arg_type_align_size"),
                    type_name: arg_type_name(node),
                    layout: struct_info_ref(node)
                        .and_then(|sref| parse_struct_info(nodes, sref, 0)),
                });
                true
            }
            "mesh_grid_properties" => {
                self.mesh_grid_param = Some(param_index);
                true
            }
            "mesh" => {
                self.outputs = ref_after_marker(node, "air.mesh")
                    .and_then(|info| nodes.get(&info))
                    .and_then(|info| mesh_type_info(param_index, info, nodes));
                true
            }
            _ => false,
        }
    }
}

/// `!{!"air.mesh_type_info", !VERTEX_FIELDS, !PRIMITIVE_FIELDS, i32 NV, i32 NP, !"air.<t>"}`.
fn mesh_type_info(
    param_index: u32,
    body: &str,
    nodes: &HashMap<u32, String>,
) -> Option<MeshOutputsMeta> {
    let operands = split_metadata_operands(body);
    if operands.first().map(String::as_str) != Some("!\"air.mesh_type_info\"") {
        return None;
    }
    let fields = |operand: Option<&String>| -> Option<Vec<MeshFieldMeta>> {
        let id = operand?.strip_prefix('!')?.parse::<u32>().ok()?;
        Some(
            refs_in(nodes.get(&id)?)
                .into_iter()
                .filter_map(|field| nodes.get(&field))
                .filter(|field| !field.contains("!\"air.function_constant_disabled\""))
                .map(String::as_str)
                .map(mesh_field)
                .collect(),
        )
    };
    let int = |operand: Option<&String>| operand?.strip_prefix("i32 ")?.trim().parse::<u32>().ok();
    let topology = match operands.get(5).map(String::as_str)? {
        "!\"air.point\"" => MeshTopology::Point,
        "!\"air.line\"" => MeshTopology::Line,
        "!\"air.triangle\"" => MeshTopology::Triangle,
        _ => return None,
    };
    Some(MeshOutputsMeta {
        param_index,
        vertex: fields(operands.get(1))?,
        primitive: fields(operands.get(2))?,
        max_vertices: int(operands.get(3))?,
        max_primitives: int(operands.get(4))?,
        topology,
    })
}

/// One field node of a mesh output struct, read past an `air.function_constant` predicate pair.
fn mesh_field(body: &str) -> MeshFieldMeta {
    let mut operands = split_metadata_operands(body).into_iter().peekable();
    if operands.peek().map(String::as_str) == Some("!\"air.function_constant\"") {
        operands.next();
        operands.next();
    }
    let role = operands
        .next()
        .and_then(|operand| {
            operand
                .strip_prefix("!\"air.")
                .and_then(|rest| rest.strip_suffix('"'))
                .map(str::to_string)
        })
        .unwrap_or_default();
    let mut data = |kind: fn(u32, String) -> MeshFieldRole| {
        let slot = operands
            .next()
            .and_then(|operand| operand.strip_prefix("i32 ")?.trim().parse::<u32>().ok());
        let key = operands.next().and_then(|operand| {
            operand
                .strip_prefix("!\"")
                .and_then(|rest| rest.strip_suffix('"'))
                .map(str::to_string)
        });
        match (slot, key) {
            (Some(slot), Some(key)) => kind(slot, key),
            _ => MeshFieldRole::Unrecognized(role.clone()),
        }
    };
    let role = match role.as_str() {
        "position" => MeshFieldRole::Position,
        "point_size" => MeshFieldRole::PointSize,
        "clip_distance" => MeshFieldRole::ClipDistance,
        "mesh_vertex_data" => data(|slot, key| MeshFieldRole::VertexData { slot, key }),
        "primitive_culled" => MeshFieldRole::PrimitiveCulled,
        "primitive_id" => MeshFieldRole::PrimitiveId,
        "render_target_array_index" => MeshFieldRole::RenderTargetArrayIndex,
        "viewport_array_index" => MeshFieldRole::ViewportArrayIndex,
        "mesh_primitive_data" => data(|slot, key| MeshFieldRole::PrimitiveData { slot, key }),
        _ => MeshFieldRole::Unrecognized(role.clone()),
    };
    MeshFieldMeta {
        role,
        type_name: arg_type_name(body),
    }
}
