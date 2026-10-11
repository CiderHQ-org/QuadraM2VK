use super::*;

/// Role of a single vertex-shader entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertRole {
    /// `[[stage_in]]` / `[[attribute(n)]]` vertex attribute -> Input var at Location N.
    VertexInput(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[texture(n)]]` -> sampled image (vertex texture fetch). WindowServer vertices read data via
    /// textures, so the vertex stage needs the same texture/sampler handling as the fragment stage.
    Texture(u32),
    /// `[[sampler(n)]]` -> sampler.
    Sampler(u32),
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[vertex_id]]` -> Input BuiltIn VertexIndex (32-bit uint).
    VertexId,
    /// `[[instance_id]]` -> Input BuiltIn InstanceIndex (32-bit uint).
    InstanceId,
    /// Opaque Metal patch handle consumed only by the metadata-named control-point accessor.
    PatchControlPoints,
    /// Per-patch user input at the AIR location.
    PatchInput(u32),
    /// `[[position_in_patch]]` -> the leading components of Vulkan `TessCoord`.
    PositionInPatch,
    /// `[[patch_id]]` -> Vulkan `PrimitiveId` in tessellation evaluation.
    PatchId,
    /// Metal vertex-amplification identifiers have no Vulkan tessellation builtin and are exposed
    /// through the translator's per-patch system-input locations.
    AmplificationId,
    AmplificationCount,
    /// A fact about the execution group this vertex's thread runs in. See
    /// [`FragRole::ExecutionGroup`].
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    VariantAbsentTexture,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchDomain {
    Triangle,
    Quad,
    Isoline,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PatchControlPointField {
    pub location: u32,
    pub type_name: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationMeta {
    pub domain: PatchDomain,
    pub control_point_count: u32,
    pub control_point_function: Option<String>,
    pub control_point_fields: Vec<PatchControlPointField>,
}

/// Role of a single vertex return-struct member.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VertOutRole {
    /// `[[position]]` -> Output BuiltIn Position.
    Position,
    /// `[[point_size]]` -> Output BuiltIn PointSize. Does not consume a Location.
    PointSize,
    /// `[[clip_distance]]` -> Output BuiltIn ClipDistance. Does not consume a Location.
    ClipDistance,
    /// `[[viewport_array_index]]` -> Output BuiltIn ViewportIndex. Does not consume a Location.
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Output BuiltIn Layer. Does not consume a Location.
    RenderTargetArrayIndex,
    /// User varying output -> Output var at Location N.
    Varying(u32),
    /// Function-constant-gated output disabled by the translator's default-zero FC model.
    FunctionConstantDisabled,
    Other,
}

/// A vertex shader's decoded parameter roles. The OUTPUT struct (member 0 = `[[position]]`, the rest
/// follows `output_roles`; point-size-style builtins do not consume varying locations.
#[derive(Clone, Debug, Default)]
pub struct VertMeta {
    pub roles: Vec<(u32, VertRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// Entry parameter index -> AIR type name. Tessellation system values use this to expose the
    /// exact cross-stage scalar type instead of forcing executors to infer it from a location.
    pub parameter_type_names: HashMap<u32, String>,
    pub output_roles: Vec<VertOutRole>,
    /// `(member index, AIR role)` for every output member decoded as [`VertOutRole::Other`].
    ///
    /// The vertex output walk gives an unrecognised member the next free user Location, so an
    /// unmodelled role does not vanish — it becomes a varying the pipeline never wired, at a
    /// Location it takes from a real one. Reporting it lets emission reject instead.
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Output member indices AIR marked `air.invariant` — Metal `[[position, invariant]]`.
    ///
    /// The guarantee is bit-exact: the same vertex fed through two pipelines that both declare it
    /// must produce the identical clip position, which is what lets a depth-prepass and the pass
    /// that tests against it agree instead of z-fighting. Vulkan spells it `OpDecorate … Invariant`.
    /// A translation that drops it stays valid and reflects identically, so nothing but this record
    /// carries the request across.
    pub invariant_outputs: Vec<u32>,
    /// User-varying output Location -> AIR type name.
    pub output_varying_types: HashMap<u32, String>,
    /// User-varying output Location -> Metal field name.
    pub output_varying_names: HashMap<u32, String>,
    /// User-varying output Location -> Metal linker semantic.
    pub output_varying_user_semantics: HashMap<u32, String>,
    /// Vertex input Location -> AIR type name (`float2`, `float4`, ...). Used by conformance
    /// oracles that must synthesize a Metal vertex descriptor before pipeline reflection exists.
    pub vertex_input_types: HashMap<u32, String>,
    /// Vertex input Location -> Metal argument name, when AIR metadata carries one.
    pub vertex_input_names: HashMap<u32, String>,
    /// Per-patch tessellation input Location -> AIR type name.
    pub patch_input_types: HashMap<u32, String>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_address_spaces`]).
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` for buffer args, when the AIR carries it (see
    /// [`FragMeta::buffer_type_sizes`]).
    pub buffer_type_sizes: HashMap<u32, u32>,
    pub buffer_object_sizes: HashMap<u32, u32>,
    pub buffer_type_names: HashMap<u32, String>,
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
    pub tessellation: Option<TessellationMeta>,
    /// Why an `air.patch` node the function does carry could not be decoded into
    /// [`VertMeta::tessellation`], if that happened.
    ///
    /// The two are not interchangeable with `tessellation: None`. A vertex function with no patch
    /// node is an ordinary vertex shader; one whose patch node did not decode is a post-tessellation
    /// evaluation shader whose domain, spacing, winding and per-patch inputs would all go missing,
    /// and the module that results is valid, binds, reflects, and draws the wrong geometry. The
    /// stage-input pass refuses on this rather than emitting it.
    pub undecoded_patch_shape: Option<String>,
    /// Every attribute on the `!air.vertex` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
}

impl VertMeta {
    pub fn tessellation_system_input_location(&self, role: &VertRole) -> Option<u32> {
        let base = self
            .roles
            .iter()
            .filter_map(|(_, role)| match role {
                VertRole::PatchInput(location) => Some(*location),
                _ => None,
            })
            .chain(
                self.tessellation
                    .iter()
                    .flat_map(|meta| meta.control_point_fields.iter().map(|field| field.location)),
            )
            .max()
            .map_or(0, |location| location + 1);
        match role {
            VertRole::InstanceId => Some(base),
            VertRole::AmplificationId => Some(base + 1),
            VertRole::AmplificationCount => Some(base + 2),
            _ => None,
        }
    }
}

impl VertMeta {
    pub fn role_of(&self, idx: u32) -> Option<&VertRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn output_role_of(&self, idx: u32) -> Option<&VertOutRole> {
        self.output_roles.get(idx as usize)
    }
    /// Whether AIR marked output member `idx` `air.invariant`.
    pub fn output_is_invariant(&self, idx: u32) -> bool {
        self.invariant_outputs.contains(&idx)
    }
    pub fn output_varying_type(&self, loc: u32) -> Option<&str> {
        self.output_varying_types.get(&loc).map(String::as_str)
    }
    pub fn vertex_input_type(&self, loc: u32) -> Option<&str> {
        self.vertex_input_types.get(&loc).map(String::as_str)
    }
}

/// Parse vertex metadata and its entry name from one AIR metadata-node table.
pub fn parse_air_vertex_meta(ll: &str) -> (Option<VertMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "vertex", &nodes);
    let meta = parse_air_vertex_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

fn parse_air_vertex_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<VertMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "vertex")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let out_ref = *refs.first()?;
    let in_ref = *refs.get(1)?;
    // Read the whole attribute tail, not just the one attribute this stage models. A patch node is
    // recognised by what it says (`air.patch`) rather than by where it sits, and anything else the
    // root carries becomes a named refusal instead of silence.
    let mut patch_node = None;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::Patch(node) => patch_node = Some(node),
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let mut undecoded_patch_shape = None;
    let patch_shape = patch_node.as_deref().and_then(|node| {
        let domain = if node.contains("!\"quad\"") {
            PatchDomain::Quad
        } else if node.contains("!\"triangle\"") {
            PatchDomain::Triangle
        } else if node.contains("!\"isoline\"") {
            PatchDomain::Isoline
        } else {
            // Dropping the shape here would emit an ordinary vertex shader: no domain, no spacing,
            // no winding, and a per-patch input set the pipeline never wires. Carry the failure out
            // so the passes can refuse instead.
            undecoded_patch_shape = Some("air.patch names no tessellation domain".to_string());
            return None;
        };
        match i32_after_marker(node, "air.patch_control_point") {
            Some(count) => Some((domain, count)),
            None => {
                undecoded_patch_shape =
                    Some("air.patch states no air.patch_control_point count".to_string());
                None
            }
        }
    });

    let mut output_roles = vec![];
    let mut invariant_outputs = vec![];
    let mut unmodelled_output_members = vec![];
    let mut output_varying_types = HashMap::new();
    let mut output_varying_names = HashMap::new();
    let mut output_varying_user_semantics = HashMap::new();
    let mut out_loc = 0u32;
    for r in refs_in(nodes.get(&out_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let strs = role_strings(node);
        let Some(first) = gated_role(&strs, |role| vertex_output_kind(role).is_some()) else {
            continue;
        };
        let kind = vertex_output_kind(first).filter(|kind| {
            // A gated builtin the shader cannot have written is absent. Declaring it anyway puts a
            // variable in the entry-point interface -- and, for `Layer` and `ViewportIndex`, a
            // device capability in the module -- for a value nothing produces.
            !kind.is_system_value() || metadata_enabled_by_default(node, nodes, &static_int_globals)
        });
        let role = match kind {
            Some(VertexOutputKind::Position) => VertOutRole::Position,
            Some(VertexOutputKind::PointSize) => VertOutRole::PointSize,
            Some(VertexOutputKind::ClipDistance) => VertOutRole::ClipDistance,
            Some(VertexOutputKind::ViewportArrayIndex) => VertOutRole::ViewportArrayIndex,
            Some(VertexOutputKind::RenderTargetArrayIndex) => VertOutRole::RenderTargetArrayIndex,
            Some(VertexOutputKind::Varying) => {
                let l = location_index_with_static(node, out_loc, &static_int_globals);
                out_loc += 1;
                if let Some(name) = arg_type_name(node) {
                    output_varying_types.insert(l, name);
                }
                if let Some(name) = arg_name(node) {
                    output_varying_names.insert(l, name);
                }
                if let Some(semantic) = string_after_marker(node, "air.vertex_output") {
                    output_varying_user_semantics.insert(l, semantic);
                }
                VertOutRole::Varying(l)
            }
            // The member is in the return struct but nothing writes it: an unresolved wrapper over
            // a role this stage does not lower, an AIR-specialized-off predicate, or a builtin the
            // filter above ruled absent.
            None if matches!(first, "function_constant" | "function_constant_disabled")
                || vertex_output_kind(first).is_some() =>
            {
                VertOutRole::FunctionConstantDisabled
            }
            // A role no vertex output lowering models. Reported, not skipped: AIR states an output
            // because the shader writes it, so a member nothing knows how to write is a value that
            // silently disappears.
            None => VertOutRole::Other,
        };
        if strs.iter().any(|s| s == "invariant") {
            invariant_outputs.push(output_roles.len() as u32);
        }
        if matches!(role, VertOutRole::Other) {
            unmodelled_output_members.push((output_roles.len() as u32, first.to_string()));
        }
        output_roles.push(role);
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut parameter_type_names = HashMap::new();
    let mut vertex_input_types = HashMap::new();
    let mut vertex_input_names = HashMap::new();
    let mut patch_input_types = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut indirect_buffer_struct_refs = Vec::new();
    let mut patch_control_point = None;
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let mut vin_loc = 0u32;
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        if let Some(name) = arg_type_name(node) {
            parameter_type_names.insert(idx, name);
        }
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(mut first) = vertex_input_gated_role(&strs) else {
            continue;
        };
        // The one demotion the promotion set cannot state: a wrapped texture this variant declares
        // no slot for has no binding to promote it to. See [`variant_texture_slot`].
        let texture_slot = variant_texture_slot(node, idx, nodes, &static_int_globals);
        if primary_role(&strs) == Some("texture") && texture_slot.is_none() {
            first = VARIANT_ABSENT_TEXTURE_ROLE;
        }
        let first = present_gated_buffer_role(first, &strs, node, nodes, &static_int_globals);
        let first = present_system_value_role(first, node, nodes, &static_int_globals);
        if let Some(declared) = unmodelled_declared_role(
            |role| air_input_role_is_modelled(Stage::Vertex, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        let role =
            match first {
                "vertex_input" => {
                    let l = location_index_with_static(node, vin_loc, &static_int_globals);
                    vin_loc += 1;
                    if let Some(name) = arg_type_name(node) {
                        vertex_input_types.insert(l, name);
                    }
                    if let Some(name) = arg_name(node) {
                        vertex_input_names.insert(l, name);
                    }
                    VertRole::VertexInput(l)
                }
                "buffer" | "indirect_buffer" => {
                    if first == "indirect_buffer" {
                        if let Some(sref) = struct_info_ref(node) {
                            indirect_buffer_struct_refs.push((
                                idx,
                                location_index_with_static(node, idx, &static_int_globals),
                                sref,
                            ));
                        }
                    }
                    if let Some(space) =
                        address_space(node).or_else(|| param_address_spaces.get(&idx).copied())
                    {
                        buffer_address_spaces.insert(idx, space);
                    }
                    if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                        .or_else(|| i32_after_marker(node, "air.buffer_size"))
                    {
                        buffer_type_sizes.insert(idx, size);
                    }
                    if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                        buffer_object_sizes.insert(idx, size);
                    }
                    if let Some(name) = arg_type_name(node) {
                        buffer_type_names.insert(idx, name);
                    }
                    if let Some(access) = declared_buffer_access(node) {
                        buffer_accesses.insert(idx, access);
                    }
                    VertRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
                }
                "texture" => {
                    if let Some(name) = arg_type_name(node) {
                        texture_type_names.insert(idx, name);
                    }
                    VertRole::Texture(texture_slot.unwrap_or_else(|| {
                        location_index_with_static(node, idx, &static_int_globals)
                    }))
                }
                "sampler" => {
                    VertRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
                }
                "visible_function_table" => VertRole::VisibleFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "intersection_function_table" => VertRole::IntersectionFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "vertex_id" => VertRole::VertexId,
                "instance_id" => VertRole::InstanceId,
                "patch_control_point_input" => {
                    let refs = refs_in(node);
                    let function = refs
                        .first()
                        .and_then(|reference| nodes.get(reference))
                        .and_then(|body| pointer_symbol(body));
                    let fields = refs
                        .iter()
                        .skip(1)
                        .filter_map(|reference| nodes.get(reference))
                        .map(|field| PatchControlPointField {
                            location: location_index_with_static(field, 0, &static_int_globals),
                            type_name: arg_type_name(field),
                        })
                        .collect::<Vec<_>>();
                    if let Some(function) = function {
                        patch_control_point = Some((function, fields));
                    }
                    VertRole::PatchControlPoints
                }
                "patch_input" => {
                    let location = location_index_with_static(node, idx, &static_int_globals);
                    if let Some(name) = arg_type_name(node) {
                        patch_input_types.insert(location, name);
                    }
                    VertRole::PatchInput(location)
                }
                "position_in_patch" => VertRole::PositionInPatch,
                "patch_id" => VertRole::PatchId,
                "amplification_id" => VertRole::AmplificationId,
                "amplification_count" => VertRole::AmplificationCount,
                VARIANT_ABSENT_TEXTURE_ROLE => VertRole::VariantAbsentTexture,
                // The specializer's spelling of the same fact. Once a VALUE is supplied,
                // `specialize_function_constant_metadata` rewrites the resolved-off wrapper, and past
                // that rewrite the wrapped role is unreadable -- so the demotion above never runs and
                // this is the only place the fact survives.
                _ if declares_disabled_texture(&strs) => VertRole::VariantAbsentTexture,
                other => stage_execution_group_role(Stage::Vertex, other).map_or(
                    VertRole::Other,
                    |(fact, lanes)| VertRole::ExecutionGroup { fact, lanes },
                ),
            };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            VertRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    Some(VertMeta {
        roles,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll, entry)?,
        parameter_type_names,
        output_roles,
        invariant_outputs,
        unmodelled_output_members,
        output_varying_types,
        output_varying_names,
        output_varying_user_semantics,
        vertex_input_types,
        vertex_input_names,
        patch_input_types,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &argument_buffers),
        unsurfaced_embedded_resources: unsurfaced_embedded_resources(nodes, &argument_buffers),
        undecoded_patch_shape,
        unmodelled_stage_attributes,
        tessellation: patch_shape.map(|(domain, control_point_count)| {
            let (control_point_function, control_point_fields) = patch_control_point
                .map(|(function, fields)| (Some(function), fields))
                .unwrap_or_default();
            TessellationMeta {
                domain,
                control_point_count,
                control_point_function,
                control_point_fields,
            }
        }),
    })
}
