use super::types::{tokenize, Tok};
use super::*;

/// Role of a single fragment-shader entry parameter, keyed by its parameter index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FragRole {
    /// `[[position]]` -> Input BuiltIn FragCoord (often unused).
    Position,
    /// `[[point_coord]]` -> Input BuiltIn PointCoord.
    PointCoord,
    /// `[[front_facing]]` -> Input BuiltIn FrontFacing.
    FrontFacing,
    /// `[[barycentric_coord]]` -> Input BuiltIn `BaryCoordKHR` / `BaryCoordNoPerspKHR` (float3).
    ///
    /// AIR states the perspective axis on the same node, exactly as it does for a varying, and
    /// SPIR-V spells the two as different builtins rather than as a decoration — so the flag has to
    /// travel with the role.
    BarycentricCoord { no_perspective: bool },
    /// `[[primitive_id]]` -> Input BuiltIn PrimitiveId (32-bit uint).
    PrimitiveId,
    /// `[[sample_id]]` -> Input BuiltIn SampleId (32-bit uint).
    SampleId,
    /// `[[sample_mask]]` on an *argument* -> Input BuiltIn `SampleMask`, the coverage the
    /// rasterizer produced for this fragment. AIR spells the input form `air.sample_mask_in` to
    /// distinguish it from the return member; SPIR-V uses one builtin in two storage classes.
    SampleMaskIn,
    /// `[[viewport_array_index]]` -> Input BuiltIn ViewportIndex (32-bit uint).
    ViewportArrayIndex,
    /// `[[render_target_array_index]]` -> Input BuiltIn Layer (32-bit uint).
    RenderTargetArrayIndex,
    /// `[[amplification_id]]` -> Input BuiltIn ViewIndex (32-bit uint).
    ///
    /// Metal vertex amplification and Vulkan multiview are the same feature: one draw rasterized
    /// into several views, with the shader told which view it is running for. Metal spells that
    /// index `[[amplification_id]]`; Vulkan spells it `gl_ViewIndex`. A pass with a single view
    /// answers zero on both sides, so the mapping holds whether or not multiview is enabled.
    AmplificationId,
    /// `[[amplification_count]]` -> the count the render encoder set, bound as a constant.
    ///
    /// The count is a property of the draw, not of the stage that reads it: the same
    /// `setVertexAmplificationCount:viewMappings:` answers a fragment and a vertex alike. Only the
    /// vertex role was modelled, so a fragment declaring it fell to [`FragRole::Other`] and the
    /// module was refused for a value the stage-input pass already knew how to bind.
    AmplificationCount,
    /// `[[stage_in]]` interpolated input -> Input var at Location N (N = order among fragment_inputs).
    Varying(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// A Metal visible-function table resolved during authored dependency linking.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table resolved during authored dependency linking.
    IntersectionFunctionTable(u32),
    /// `[[buffer(n)]]` -> Uniform/StorageBuffer block.
    Buffer(u32),
    /// `[[color(n)]]` framebuffer-fetch input -> Vulkan input attachment (out of scope this milestone).
    ColorInput(u32),
    /// A custom fragment `[[imageblock_data]]` projection. Its fields are mapped to the master
    /// imageblock layout by AIR user semantic, never by the source struct or argument name.
    ImageblockData,
    /// A fact about the execution group this fragment's thread runs in. Only the facts that need no
    /// threadgroup reach a fragment; see [`ExecutionGroupFact::needs_a_threadgroup`].
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    VariantAbsentTexture,
    /// Anything we don't model.
    Other,
}

/// Where inside a pixel AIR asked a `fragment_input` varying to be sampled.
///
/// Metal spells this as the first half of an interpolation attribute — `[[center_perspective]]`,
/// `[[centroid_no_perspective]]`, `[[sample_perspective]]` — and AIR emits it as its own marker
/// alongside the perspective marker. `air.center` is the default and needs no SPIR-V decoration.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VaryingSampling {
    /// `air.center` — sampled at the pixel center. Vulkan's default; no decoration.
    #[default]
    Center,
    /// `air.centroid` — sampled inside the covered area of the primitive (`Centroid`).
    Centroid,
    /// `air.sample` — sampled per covered sample, which forces per-sample shading (`Sample`,
    /// capability `SampleRateShading`).
    Sample,
}

/// How AIR asked a `fragment_input` varying to be interpolated.
///
/// One record per varying rather than one set per qualifier: AIR states the whole interpolation
/// attribute on the argument node, and reading only the part the emitter happened to support is
/// how `air.no_perspective` and `air.centroid` were silently dropped for as long as only
/// `air.flat` was decoded. `interpolation_markers_are_decoded_or_deliberately_ignored` in
/// `src/meta/tests.rs` pins the full marker inventory against this record.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VaryingInterpolation {
    /// `air.flat` — not interpolated at all. AIR states this instead of, not alongside, the
    /// perspective/sampling pair, so it takes precedence over both fields below.
    pub flat: bool,
    /// `air.no_perspective` — interpolated linearly in screen space rather than
    /// perspective-correct (`NoPerspective`). Its complement `air.perspective` is Vulkan's
    /// default and needs no decoration.
    pub no_perspective: bool,
    /// Where in the pixel the interpolated value is taken from.
    pub sampling: VaryingSampling,
}

impl VaryingInterpolation {
    /// Decode the interpolation attribute from an argument node's `air.*` marker list.
    fn from_role_strings(strs: &[String]) -> Self {
        let has = |marker: &str| strs.iter().any(|s| s == marker);
        Self {
            flat: has("flat"),
            no_perspective: has("no_perspective"),
            sampling: if has("centroid") {
                VaryingSampling::Centroid
            } else if has("sample") {
                VaryingSampling::Sample
            } else {
                VaryingSampling::Center
            },
        }
    }
}

/// A fragment shader's decoded parameter roles + render-target count/indices.
#[derive(Clone, Debug, Default)]
pub struct FragMeta {
    /// `(param_idx, role)` — one per fragment input, in declaration order.
    pub roles: Vec<(u32, FragRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering.
    ///
    /// An unrecognised parameter is bound to a zero value so the body stays well formed. That is
    /// right for a function-constant-disabled resource, which Metal itself defines as absent, and
    /// wrong for anything else: a `[[barycentric_coord]]` the emitter does not model becomes a
    /// silent zero, and everything computed from it is wrong in a module that validates. Emission
    /// rejects on a non-empty list instead.
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Every attribute on the `!air.fragment` root this stage has no model for, as it reads in
    /// AIR. See [`FragMeta::early_fragment_tests`]: the root's attribute tail changes what the
    /// stage does, so an entry carrying an attribute nothing consumes must be refused rather than
    /// emitted as if the root had carried nothing.
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[early_fragment_tests]]`: depth and stencil testing happens before the fragment body.
    ///
    /// This is not an optimisation hint. A fragment the depth test rejects runs no part of the
    /// body, so none of its buffer, texture or imageblock stores happen; under the default late
    /// test the same shader performs every store and only its color output is discarded.
    pub early_fragment_tests: bool,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    /// Detected from the module's intrinsic calls, which is a property of the body rather than of
    /// the stage — the interface pass materializes the plane wherever it lowers one of those calls,
    /// so every stage that can carry them has to be able to report them.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `fragment_input` Location -> AIR type name (`float2`, `float4`, ...). Used by passthrough
    /// vertex synthesis when the pipeline binds a built-in vertex slot.
    pub varying_types: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal field/argument name, when AIR metadata carries one. The
    /// Metal oracle uses this to generate a vertex struct that Apple's pipeline linker matches.
    pub varying_names: HashMap<u32, String>,
    /// `fragment_input` Location -> Metal user semantic, such as `user(texturecoord)`, when AIR
    /// metadata carries one.
    pub varying_user_semantics: HashMap<u32, String>,
    /// `fragment_input` Location -> the interpolation attribute AIR declared for it. Absent means
    /// AIR said nothing, which is Vulkan's default (perspective-correct, pixel center).
    pub varying_interpolation: HashMap<u32, VaryingInterpolation>,
    /// Return-struct member index -> color attachment Location for actual `air.render_target`
    /// outputs. Non-color outputs such as `air.stencil` are deliberately absent.
    pub render_target_members: Vec<(u32, u32)>,
    /// Return-struct member index -> AIR render-target type name (`float4`, `int4`, ...).
    pub render_target_type_names: HashMap<u32, String>,
    /// Return-struct member indices tagged as `air.depth` (`[[depth(...)]]`).
    pub depth_members: Vec<u32>,
    /// Conservative depth-test relation declared by `[[depth(...)]]`.
    pub depth_qualifier: Option<DepthQualifier>,
    /// Return-struct member indices tagged as `air.stencil` (`[[stencil]]`).
    pub stencil_members: Vec<u32>,
    /// Return-struct member indices tagged as `air.sample_mask` (`[[sample_mask]]`).
    ///
    /// The value is a coverage mask: a sample whose bit the shader clears is not written, which is
    /// how alpha-to-coverage and custom MSAA resolves are expressed. Vulkan spells it as the
    /// `SampleMask` builtin, an array of `uint` rather than the scalar Metal returns.
    pub sample_mask_members: Vec<u32>,
    /// `(member index, AIR role)` for every enabled return member whose role the emitter has no
    /// lowering for.
    pub unmodelled_output_members: Vec<(u32, String)>,
    /// Custom per-pixel fragment imageblock master plus the input/output projections that expose
    /// subsets of its fields: of views of two masters, the larger master (the first output view's
    /// on a tie), whose cell every view reaches. `None` when the fragment carries no
    /// `air.imageblock_master` contract.
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// `param_idx -> reconstructed struct layout` for buffer args that carry `air.struct_type_info`.
    /// Used to rebuild the real struct when native emission represents the buffer as a bare pointer.
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> AIR address space` for buffer args, when the AIR node carries it (device=1,
    /// constant=2). Populated only from `air.address_space` / the function param pointer address
    /// space — absent (not guessed) when the IR does not state it. Mirrors
    /// [`KernMeta::buffer_address_spaces`] for the fragment stage.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer byte size` (`air.arg_type_size` / `air.buffer_size`), when
    /// the AIR node carries it. Mirrors [`KernMeta::buffer_type_sizes`] for the fragment stage.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` for reference-like arguments whose metadata bounds the
    /// reachable object to exactly that many bytes. Kept separate from `buffer_type_sizes`, which
    /// may only be an unbounded pointer's element/pointee size.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name` for every stage.
    pub buffer_type_names: HashMap<u32, String>,
    /// `param_idx -> declared AIR read/write qualifier`.
    pub buffer_accesses: HashMap<u32, BufferAccess>,
    /// `param_idx -> AIR texture argument type name`, e.g. `texture2d<uint, read>`.
    pub texture_type_names: HashMap<u32, String>,
    /// `param_idx -> the descriptor count `air.location_index` states, when it states more than one.
    ///
    /// A texture or sampler argument above `1` is a handle ARRAY occupying that many descriptors at
    /// its Metal slot. Absent for the ordinary single-descriptor argument and for a count spelled as
    /// a function-constant global.
    pub declared_descriptor_counts: HashMap<u32, u32>,
    /// Framebuffer-fetch color input Location -> AIR render-target type name, e.g. `float4`.
    pub color_input_type_names: HashMap<u32, String>,
    pub embedded_textures: Vec<EmbeddedTexture>,
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
}

/// One field in AIR's custom fragment-imageblock master layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblockMember {
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub semantic: String,
    pub raster_order_group: u32,
}

/// One field exposed by an entry input or return-value imageblock projection, and one master
/// member whose bytes it reaches: the projection member's bytes are its own master's member of its
/// key, so a view of another master reaches each master member its bytes overlap (one entry per
/// member).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblockProjectionMember {
    pub projection_member: u32,
    pub master_member: u32,
}

/// A partial struct view of a custom fragment imageblock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblockProjection {
    /// Entry parameter index for an input projection, or return-struct member index for an output.
    pub interface_index: u32,
    pub members: Vec<FragmentImageblockProjectionMember>,
}

/// AIR's exact custom fragment-imageblock ABI.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblock {
    pub sample_size: u32,
    pub members: Vec<FragmentImageblockMember>,
    pub inputs: Vec<FragmentImageblockProjection>,
    pub outputs: Vec<FragmentImageblockProjection>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DepthQualifier {
    Any,
    Less,
    Greater,
}

impl FragMeta {
    pub fn role_of(&self, idx: u32) -> Option<&FragRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }
    pub fn varying_type(&self, loc: u32) -> Option<&str> {
        self.varying_types.get(&loc).map(String::as_str)
    }
    /// The interpolation attribute AIR declared for the varying at `loc`, defaulting to Vulkan's
    /// own default when AIR declared none.
    pub fn varying_interpolation(&self, loc: u32) -> VaryingInterpolation {
        self.varying_interpolation
            .get(&loc)
            .copied()
            .unwrap_or_default()
    }
    pub fn varying_is_flat(&self, loc: u32) -> bool {
        self.varying_interpolation(loc).flat
    }
    pub fn render_target_type_name(&self, member_idx: u32) -> Option<&str> {
        self.render_target_type_names
            .get(&member_idx)
            .map(String::as_str)
    }
    pub fn is_depth_member(&self, member_idx: u32) -> bool {
        self.depth_members.contains(&member_idx)
    }
    pub fn is_stencil_member(&self, member_idx: u32) -> bool {
        self.stencil_members.contains(&member_idx)
    }
    pub fn is_sample_mask_member(&self, member_idx: u32) -> bool {
        self.sample_mask_members.contains(&member_idx)
    }
}

fn parse_fragment_imageblock_master(body: &str) -> Option<Vec<FragmentImageblockMember>> {
    let toks = tokenize(body);
    let mut members = Vec::new();
    let mut i = 0;
    while i + 4 < toks.len() {
        let (offset, size, type_name, semantic) = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(offset)),
                Some(Tok::Int(size)),
                Some(Tok::Int(_array_len)),
                Some(Tok::Str(type_name)),
                Some(Tok::Str(semantic)),
            ) => (*offset, *size, type_name.clone(), semantic.clone()),
            _ => return None,
        };
        i += 5;
        let raster_order_group = match (toks.get(i), toks.get(i + 1)) {
            (Some(Tok::Str(marker)), Some(Tok::Int(group)))
                if marker == "air.raster_order_group" =>
            {
                i += 2;
                *group
            }
            _ => return None,
        };
        members.push(FragmentImageblockMember {
            offset,
            size,
            type_name,
            semantic,
            raster_order_group,
        });
    }
    (!members.is_empty() && i == toks.len()).then_some(members)
}

fn parse_fragment_imageblock_projection(
    nodes: &HashMap<u32, String>,
    node: &str,
    interface_index: u32,
    master_members: &[FragmentImageblockMember],
) -> Option<FragmentImageblockProjection> {
    let projection_ref = struct_info_ref(node)?;
    let projection = nodes.get(&projection_ref)?;
    // The view's own master (its layout when it names none): where each of its members lives.
    let own_ref = ref_after_marker(node, "air.imageblock_master").unwrap_or(projection_ref);
    let own_members = parse_fragment_imageblock_master(nodes.get(&own_ref)?)?;
    let toks = tokenize(projection);
    let mut members = Vec::new();
    let mut i = 0;
    let mut projection_member = 0;
    while i + 4 < toks.len() {
        let semantic = match (
            toks.get(i),
            toks.get(i + 1),
            toks.get(i + 2),
            toks.get(i + 3),
            toks.get(i + 4),
        ) {
            (
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Int(_)),
                Some(Tok::Str(_)),
                Some(Tok::Str(semantic)),
            ) => semantic,
            _ => return None,
        };
        let own = own_members
            .iter()
            .find(|member| member.semantic == *semantic)?;
        let (start, end) = (own.offset, own.offset.checked_add(own.size)?);
        let before = members.len();
        for (index, member) in master_members.iter().enumerate() {
            if member.offset < end && start < member.offset.saturating_add(member.size) {
                members.push(FragmentImageblockProjectionMember {
                    projection_member,
                    master_member: index as u32,
                });
            }
        }
        if members.len() == before {
            return None;
        }
        projection_member += 1;
        i += 5;
        if matches!(toks.get(i), Some(Tok::Str(marker)) if marker == "air.raster_order_group")
            && matches!(toks.get(i + 1), Some(Tok::Int(_)))
        {
            i += 2;
        }
    }
    (!members.is_empty() && i == toks.len()).then_some(FragmentImageblockProjection {
        interface_index,
        members,
    })
}

fn parse_fragment_imageblock(
    nodes: &HashMap<u32, String>,
    out_ref: u32,
    in_ref: u32,
) -> Option<FragmentImageblock> {
    let output_nodes = nodes
        .get(&out_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    let input_nodes = nodes
        .get(&in_ref)
        .map(|body| refs_in(body))
        .unwrap_or_default();
    // The views may view two masters: each reaches its own master's bytes of one cell, which spans
    // the larger. The master is the one of the most bytes, the first output view's on a tie, else
    // the first input view's: the cell every view reaches.
    let mut imageblock_node: Option<(&String, u32)> = None;
    for body in output_nodes
        .iter()
        .chain(input_nodes.iter())
        .filter_map(|id| nodes.get(id))
        .filter(|body| primary_role(&role_strings(body)) == Some("imageblock_data"))
    {
        let size = i32_after_marker(body, "air.imageblock_data_size").unwrap_or(0);
        if imageblock_node.is_none_or(|(_, best)| size > best) {
            imageblock_node = Some((body, size));
        }
    }
    let (imageblock_node, _) = imageblock_node?;
    // Some AIR producers attach an explicit `air.imageblock_master` to a narrow projection. Others
    // pass the complete imageblock-data struct directly, in which case its own struct metadata is
    // the master layout. Both contracts carry the same offset/size/type/semantic/ROG tuple stream;
    // use that structure rather than requiring the optional indirection.
    let master_ref = ref_after_marker(imageblock_node, "air.imageblock_master")
        .or_else(|| struct_info_ref(imageblock_node))?;
    let master_members = parse_fragment_imageblock_master(nodes.get(&master_ref)?)?;
    let sample_size = i32_after_marker(imageblock_node, "air.imageblock_data_size")?;

    let outputs = output_nodes
        .iter()
        .enumerate()
        .filter_map(|(index, id)| {
            let node = nodes.get(id)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| {
                    parse_fragment_imageblock_projection(nodes, node, index as u32, &master_members)
                })
                .flatten()
        })
        .collect();
    let inputs = input_nodes
        .iter()
        .filter_map(|id| {
            let node = nodes.get(id)?;
            let index = first_i32(node)?;
            (primary_role(&role_strings(node)) == Some("imageblock_data"))
                .then(|| parse_fragment_imageblock_projection(nodes, node, index, &master_members))
                .flatten()
        })
        .collect();
    Some(FragmentImageblock {
        sample_size,
        members: master_members,
        inputs,
        outputs,
    })
}

/// Parse fragment metadata and its entry name from one AIR metadata-node table.
pub fn parse_air_fragment_meta(ll: &str) -> (Option<FragMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, "fragment", &nodes);
    let meta = parse_air_fragment_meta_with_nodes(ll, &nodes, entry.as_deref());
    (meta, entry)
}

fn parse_air_fragment_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
) -> Option<FragMeta> {
    let static_int_globals = static_init_int_global_values(ll);
    let root = stage_root(ll, "fragment")?;
    let rootc = nodes.get(&root)?;
    let refs = refs_in(rootc);
    let (out_ref, in_ref) = (*refs.first()?, *refs.get(1)?);
    let mut early_fragment_tests = false;
    let mut unmodelled_stage_attributes = vec![];
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::EarlyFragmentTests => early_fragment_tests = true,
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }
    let fragment_imageblock = parse_fragment_imageblock(nodes, out_ref, in_ref);
    let render_target_members: Vec<(u32, u32)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    (is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals))
                    .then(|| {
                        (
                            i as u32,
                            render_target_location(node, i as u32, &static_int_globals),
                        )
                    })
                })
                .collect()
        })
        .unwrap_or_default();
    // Which return members carry a given non-color output role. One reader for every such role, so
    // adding one is a call rather than another copy of this walk — `air.sample_mask` was dropped
    // for as long as it was the role nobody had copied the walk for.
    let output_members_with_role = |role: &str| -> Vec<u32> {
        nodes
            .get(&out_ref)
            .map(|c| {
                refs_in(c)
                    .into_iter()
                    .enumerate()
                    .filter_map(|(i, r)| {
                        let node = nodes.get(&r)?;
                        (primary_role(&role_strings(node)) == Some(role)
                            && metadata_enabled_by_default(node, nodes, &static_int_globals))
                        .then_some(i as u32)
                    })
                    .collect()
            })
            .unwrap_or_default()
    };
    let depth_members = output_members_with_role("depth");
    let depth_qualifier = nodes.get(&out_ref).and_then(|c| {
        refs_in(c).into_iter().find_map(|r| {
            let node = nodes.get(&r)?;
            (primary_role(&role_strings(node)) == Some("depth"))
                .then(|| string_after_marker(node, "air.depth_qualifier"))
                .flatten()
                .and_then(|qualifier| match qualifier.as_str() {
                    "air.any" => Some(DepthQualifier::Any),
                    "air.less" => Some(DepthQualifier::Less),
                    "air.greater" => Some(DepthQualifier::Greater),
                    _ => None,
                })
        })
    });
    let stencil_members = output_members_with_role("stencil");
    let sample_mask_members = output_members_with_role("sample_mask");
    let unmodelled_output_members: Vec<(u32, String)> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let role = unmodelled_declared_role(
                        |role| FRAGMENT_OUTPUT_ROLES.contains(&role),
                        node,
                        nodes,
                        &static_int_globals,
                    )?;
                    Some((i as u32, role))
                })
                .collect()
        })
        .unwrap_or_default();
    let render_target_type_names: HashMap<u32, String> = nodes
        .get(&out_ref)
        .map(|c| {
            refs_in(c)
                .into_iter()
                .enumerate()
                .filter_map(|(i, r)| {
                    let node = nodes.get(&r)?;
                    let roles = role_strings(node);
                    let is_render_target = primary_role(&roles) == Some("render_target");
                    if is_render_target
                        && metadata_enabled_by_default(node, nodes, &static_int_globals)
                    {
                        arg_type_name(node).map(|name| (i as u32, name))
                    } else {
                        None
                    }
                })
                .collect()
        })
        .unwrap_or_default();

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut buffer_layouts = HashMap::new();
    let mut varying_types = HashMap::new();
    let mut varying_names = HashMap::new();
    let mut varying_user_semantics = HashMap::new();
    let mut varying_interpolation = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut color_input_type_names = HashMap::new();
    let mut varying_loc = 0u32;
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = Vec::new();
    // AIR function-param pointer address spaces for the fragment entry, the fallback the kernel
    // parser uses when a buffer arg node omits `air.address_space`.
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        // Reconstruct any buffer struct layout used when native emission produces a bare pointer.
        if let Some(sref) = struct_info_ref(node) {
            if let Some(t) = parse_struct_info(nodes, sref, 0) {
                buffer_layouts.insert(idx, t);
            }
        }
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        let Some(role_str) = primary_role(&strs) else {
            continue;
        };
        let texture_slot = variant_texture_slot(node, idx, nodes, &static_int_globals);
        let role_str = present_system_value_role(role_str, node, nodes, &static_int_globals);
        if let Some(declared) = unmodelled_declared_role(
            |role| air_input_role_is_modelled(Stage::Fragment, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        let role =
            match role_str {
                "position" => FragRole::Position,
                "point_coord" => FragRole::PointCoord,
                "front_facing" => FragRole::FrontFacing,
                "barycentric_coord" => FragRole::BarycentricCoord {
                    no_perspective: VaryingInterpolation::from_role_strings(&strs).no_perspective,
                },
                "primitive_id" => FragRole::PrimitiveId,
                "sample_id" => FragRole::SampleId,
                "sample_mask_in" => FragRole::SampleMaskIn,
                "viewport_array_index" => FragRole::ViewportArrayIndex,
                "render_target_array_index" => FragRole::RenderTargetArrayIndex,
                "amplification_id" => FragRole::AmplificationId,
                "amplification_count" => FragRole::AmplificationCount,
                "fragment_input" => {
                    let l = varying_loc;
                    varying_loc += 1;
                    if let Some(name) = arg_type_name(node) {
                        varying_types.insert(l, name);
                    }
                    if let Some(name) = arg_name(node) {
                        varying_names.insert(l, name);
                    }
                    if let Some(semantic) = string_after_marker(node, "air.fragment_input") {
                        varying_user_semantics.insert(l, semantic);
                    }
                    varying_interpolation.insert(l, VaryingInterpolation::from_role_strings(&strs));
                    FragRole::Varying(l)
                }
                "texture" => match texture_slot {
                    Some(slot) => {
                        if let Some(name) = arg_type_name(node) {
                            texture_type_names.insert(idx, name);
                        }
                        FragRole::Texture(slot)
                    }
                    // A texture this variant declares no slot for has no binding to report. See
                    // [`variant_texture_slot`].
                    None => FragRole::VariantAbsentTexture,
                },
                "sampler" => {
                    FragRole::Sampler(location_index_with_static(node, idx, &static_int_globals))
                }
                "visible_function_table" => FragRole::VisibleFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "intersection_function_table" => FragRole::IntersectionFunctionTable(
                    location_index_with_static(node, idx, &static_int_globals),
                ),
                "buffer" | "indirect_buffer" => {
                    if role_str == "indirect_buffer" {
                        if let Some(sref) = struct_info_ref(node) {
                            indirect_buffer_struct_refs.push((
                                idx,
                                location_index_with_static(node, idx, &static_int_globals),
                                sref,
                            ));
                        }
                    }
                    // Populate address space / declared size ONLY when the IR actually carries them
                    // (no invented default), so reflection never reports a guessed value.
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
                    FragRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
                }
                // an air.render_target in the INPUT list = framebuffer fetch ([[color(n)]]).
                "render_target" => {
                    let location = render_target_location(node, idx, &static_int_globals);
                    if let Some(name) = arg_type_name(node) {
                        color_input_type_names.insert(location, name);
                    }
                    FragRole::ColorInput(location)
                }
                "imageblock_data" => FragRole::ImageblockData,
                // The specializer's spelling of a texture this variant leaves out. The wrapped
                // role is unreadable past it, so this is the only place the fact survives.
                _ if declares_disabled_texture(&strs) => FragRole::VariantAbsentTexture,
                other => stage_execution_group_role(Stage::Fragment, other).map_or(
                    FragRole::Other,
                    |(fact, lanes)| FragRole::ExecutionGroup { fact, lanes },
                ),
            };
        roles.push((idx, role));
    }
    let top_level_texture_locations = roles
        .iter()
        .filter_map(|(_, role)| match role {
            FragRole::Texture(location) => Some(*location),
            _ => None,
        })
        .collect::<Vec<_>>();
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    Some(FragMeta {
        roles,
        unmodelled_stage_attributes,
        early_fragment_tests,
        unmodelled_input_params,
        implicit_imageblock_attachments: detect_implicit_imageblock_attachments(ll, entry)?,
        varying_types,
        varying_names,
        varying_user_semantics,
        varying_interpolation,
        render_target_members,
        render_target_type_names,
        depth_members,
        depth_qualifier,
        stencil_members,
        sample_mask_members,
        unmodelled_output_members,
        fragment_imageblock,
        buffer_layouts,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        color_input_type_names,
        embedded_textures: if body_uses_texture_intrinsic(ll) {
            detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
        } else {
            Vec::new()
        },
        embedded_arguments: detect_embedded_arguments(nodes, &argument_buffers),
        unsurfaced_embedded_resources: unsurfaced_embedded_resources(nodes, &argument_buffers),
    })
}
