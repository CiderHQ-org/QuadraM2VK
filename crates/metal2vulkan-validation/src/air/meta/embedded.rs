use super::types::{member_tuples, parse_struct_info, struct_info_ref, tokenize};
use super::{arg_type_name, role_strings, texture_shape_from_name, BufferAccess, TextureFormat};
use crate::air::meta::{TextureComponent, TextureDimension};
use std::collections::HashMap;

/// The `air.indirect_buffer` arguments of one entry, in the order the synthetic embedded-texture
/// index counts them: `(parameter index, Metal `[[buffer(N)]]` slot, `air.struct_type_info` node)`
/// per argument buffer, ordered by PARAMETER INDEX.
///
/// The order is part of an ABI, not a detail. [`embedded_synthetic_texture_index`] fixes where the
/// synthetic indices start and [`detect_embedded_textures`] hands out `K`, `K+1`, ... by walking
/// this list, so a consumer that binds by synthetic index has to reach the same order the
/// translator did. Walking the AIR metadata list in the order it happens to name its argument
/// nodes made that order a property of the metadata layout: reversing the list of one corpus kernel
/// moves the same embedded texture from `Binding 32` to `Binding 480` and from Sampled to Storage.
///
/// Ordered by PARAMETER index rather than by Metal buffer index, because the latter is not a total
/// order -- mutually exclusive function-constant alternatives share one `[[buffer(N)]]` -- while an
/// entry's parameter index is unique by construction.
///
/// The type exists so the order cannot be skipped: the walkers below take this and nothing else,
/// and [`ArgumentBuffers::new`] is the only way to make one.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(super) struct ArgumentBuffers(Vec<(u32, u32, u32)>);

impl ArgumentBuffers {
    pub(super) fn new(mut declared: Vec<(u32, u32, u32)>) -> Self {
        declared.sort_by_key(|(param_index, _, _)| *param_index);
        Self(declared)
    }

    fn iter(&self) -> impl Iterator<Item = (u32, u32, u32)> + '_ {
        self.0.iter().copied()
    }
}

/// A texture that lives inside an `air.indirect_buffer` argument buffer, marked
/// `air.indirect_argument` → `air.texture` in the buffer's `air.struct_type_info`, and used by the
/// shader body through an AIR texture intrinsic.
///
/// The translator materializes a synthetic UniformConstant sampled image for it and the validation
/// harness binds the SAME deterministically-seeded texture on both the Apple-oracle side (encoded
/// into the arg buffer via `MTLArgumentEncoder`) and the Vulkan-runner side (the reflected synthetic
/// descriptor). The authored identity remains the owning buffer plus field offset; the synthetic
/// texture index `K` is assigned after the highest top-level texture index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedTexture {
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_param_index: u32,
    /// Kernel parameter index of the owning `air.indirect_buffer` argument.
    pub buffer_index: u32,
    /// Byte offset of the texture handle within the argument-buffer struct.
    pub field_offset: u32,
    pub field_ordinal: u32,
    /// Metal argument-encoder index (`[[id(n)]]`) of this texture inside the argument buffer.
    pub argument_index: u32,
    /// Texture dimensionality (2D, with `arrayed` carrying the 2D-array distinction).
    pub dim: TextureDimension,
    pub arrayed: bool,
    /// Sampled/storage component type (Float for `texture2d<float, read/write>`).
    pub comp: TextureComponent,
    /// Storage-image format for write/read_write textures; `None` for sampled/read textures.
    pub storage_format: Option<TextureFormat>,
    /// Fixed number of opaque texture handles in this argument-buffer field. `None` denotes one
    /// handle; embedded runtime `array_ref` fields are not yet representable by Metal's fixed
    /// argument-buffer layout metadata.
    pub array_length: Option<u32>,
    /// The synthetic texture index `K` this embedded texture binds at.
    pub synthetic_texture_index: u32,
}

/// One resource-handle member of an AIR indirect argument buffer. This is the common structural
/// coordinate used by authored embedded resources, Metal argument encoding, and static lowering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EmbeddedArgument {
    pub buffer_param_index: u32,
    pub buffer_index: u32,
    pub field_ordinal: u32,
    pub field_offset: u32,
    pub argument_index: u32,
    pub resource_kind: Option<ArgumentBufferResourceKind>,
    pub resource_address_space: Option<u32>,
    pub resource_declared_size: Option<u32>,
    /// Pointee aggregate layout, distinct from the owning field's eight-byte handle storage.
    pub resource_type_layout: Option<super::AirType>,
    pub resource_access: Option<BufferAccess>,
}

/// The synthetic texture index `K` for embedded argument-buffer textures, derived so the translator
/// (binding decoration) and the validation harness (model + oracle) agree WITHOUT any name key.
///
/// **ABI convention:** `K = 1 + max(top-level air.texture air.location_index)`, or `0` when the
/// kernel has no top-level textures. This guarantees `K` never collides with a real top-level
/// texture's location index, so the synthetic image binds at index `K` in the sampled- or
/// storage-texture band. `texture_locations` is the set of `air.location_index` values of the
/// kernel's top-level texture args.
fn embedded_synthetic_texture_index(texture_locations: &[u32]) -> u32 {
    texture_locations.iter().copied().max().map_or(0, |m| m + 1)
}

/// True iff the shader body calls an AIR texture intrinsic. These are stable AIR intrinsic
/// symbols, not shader identifiers. A declaration line is only emitted when the body actually calls it,
/// so its presence in the `.ll` text is a sound structural signal.
pub(super) fn body_uses_texture_intrinsic(ll: &str) -> bool {
    ll.contains("@air.sample_")
        || ll.contains("@air.gather_")
        || ll.contains("@air.read_texture")
        || ll.contains("@air.write_texture")
        || ll.contains("@air.write_imageblock_slice_to_texture")
}

/// Scan each `air.indirect_buffer`'s `air.struct_type_info` for members marked
/// `air.indirect_argument` whose nested node is an `air.texture`. Each such texture is
/// surfaced as an `EmbeddedTexture` with a synthetic index K, K+1, … (K from
/// [`embedded_synthetic_texture_index`]) so the translator and harness agree on the binding without
/// a name key. Every texture shape [`texture_shape_from_name`] reads shares this contract; see
/// [`expressible_embedded_texture`] for what is left out.
pub(super) fn detect_embedded_textures(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
    top_level_texture_locations: &[u32],
) -> Vec<EmbeddedTexture> {
    let mut out = vec![];
    let mut next_k = embedded_synthetic_texture_index(top_level_texture_locations);
    for (buffer_param_index, buffer_index, sref) in argument_buffers.iter() {
        for member in embedded_argument_members(nodes, sref) {
            let Some(tex_node) = nodes.get(&member.node_ref) else {
                continue;
            };
            let Some(shape) = expressible_embedded_texture(&member, tex_node) else {
                continue;
            };
            out.push(EmbeddedTexture {
                buffer_param_index,
                buffer_index,
                field_offset: member.field_offset,
                field_ordinal: member.field_ordinal,
                argument_index: member.argument_index(tex_node),
                dim: shape.dimension,
                arrayed: shape.arrayed,
                comp: shape.component,
                storage_format: shape.storage_format,
                array_length: shape.array_length,
                synthetic_texture_index: next_k,
            });
            next_k += 1;
        }
    }
    out
}

/// The texture shape of an `air.indirect_argument` member the embedded lowering can materialize a
/// descriptor for, or `None` for any other member.
///
/// One predicate, so [`detect_embedded_textures`] and [`unsurfaced_embedded_resources`] cannot
/// disagree about which handles are surfaced: whatever this refuses stays a refusal reason.
///
/// The shape is derived by [`texture_shape_from_name`], the same reading the top-level texture
/// arguments get, and every dimension it names is materialized -- `register_embedded_textures`
/// builds the image from `dim`/`arrayed`/`comp` without caring which. An earlier 2D-only gate here
/// silently dropped the embedded `texturecube`, `texturecube_array`, `texture1d` and `texture3d`
/// members of 97 corpus sources; 10 of those reached a sample and answered black.
fn expressible_embedded_texture(
    member: &ArgumentMember,
    node: &str,
) -> Option<super::TextureShape> {
    // A wrapper declared as an ARRAY of structs repeats its members once per element, at offsets
    // and argument ids this walk does not enumerate. Surfacing element 0 alone would bind one
    // descriptor and leave the rest reading black.
    if member.in_array_wrapper {
        return None;
    }
    let strs = role_strings(node);
    // Structural gate: the node must be an `air.texture` with an access class the read/write
    // lowerings can express.
    if !strs.iter().any(|s| s == "texture")
        || !strs
            .iter()
            .any(|s| matches!(s.as_str(), "sample" | "read" | "write" | "read_write"))
    {
        return None;
    }
    let mut shape = texture_shape_from_name(&arg_type_name(node).unwrap_or_default());
    // A descriptor-array member has TWO spellings and only one of them is in the type name.
    // `metal::array<texture2d<half, sample>, 2>` names the array, and `texture_shape_from_name`
    // reads the length out of it. A C array -- `texture2d<float, sample> slices[32]` -- names the
    // ELEMENT type, and states the length only in the member tuple's `array_len` slot; the node's
    // own `air.location_index` count operand stays `1` for it, unlike the top-level spelling
    // (see the corpus fact that count and `array<..., N>` always agree -- that holds outside an
    // argument buffer, not inside one). Reading the name alone surfaced one descriptor and let a
    // runtime index sample element 0 for every value of the index.
    if shape.array_length.is_none() && member.array_len != 0 {
        shape.array_ref = true;
        shape.array_length = Some(member.array_len);
    }
    Some(shape)
}

pub(super) fn detect_embedded_arguments(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
) -> Vec<EmbeddedArgument> {
    argument_buffers
        .iter()
        .flat_map(|(buffer_param_index, buffer_index, sref)| {
            embedded_argument_members(nodes, sref)
                .into_iter()
                // Flat members only. A wrapped `air.buffer` is a device address a consumer has to
                // write into the owning argument buffer, and nothing binds one yet, so it stays a
                // refusal reason in `unsurfaced_embedded_resources` rather than a silent descriptor.
                .filter(|member| member.depth == 0)
                .flat_map(move |member| {
                    let node = nodes.get(&member.node_ref);
                    let resource_kind = node.and_then(|node| {
                        role_strings(node)
                            .iter()
                            .find_map(|role| ArgumentBufferResourceKind::from_air_role(role))
                    });
                    let is_buffer = resource_kind == Some(ArgumentBufferResourceKind::Buffer);
                    let resource_type_layout = node.filter(|_| is_buffer).and_then(|node| {
                        struct_info_ref(node).and_then(|sref| parse_struct_info(nodes, sref, 0))
                    });
                    // A C-array member -- `device half4* bufs[2]` -- declares ONE handle per
                    // element, at consecutive argument ids and consecutive 8-byte slots, and states
                    // its length only in the member tuple. A consumer given the first element alone
                    // leaves every later slot of the argument buffer unwritten, so a runtime index
                    // into the member reads a device address nothing ever populated. Only the
                    // handle roles repeat this way: `air.indirect_constant` is stored INLINE, so its
                    // array is one span of data rather than N resources.
                    let elements = match node {
                        // A texture array is already counted ONCE, by
                        // `EmbeddedTexture::array_length`, and every consumer derives its elements
                        // from that. Repeating them here would be a second derivation of one fact
                        // for no reader, and 512 entries wide in the corpus's largest atlas.
                        Some(node) if expressible_embedded_texture(&member, node).is_some() => 1,
                        Some(node) if member.repeats_a_handle(node) => member.array_len,
                        _ => 1,
                    };
                    (0..elements).filter_map(move |element| {
                        let node = node?;
                        let argument_index = member.argument_index(node).checked_add(element)?;
                        Some(EmbeddedArgument {
                            buffer_param_index,
                            buffer_index,
                            field_ordinal: member.field_ordinal,
                            field_offset: member
                                .field_offset
                                .checked_add(element.checked_mul(HANDLE_BYTES)?)?,
                            argument_index,
                            resource_kind,
                            resource_address_space: is_buffer
                                .then(|| super::address_space(node))
                                .flatten(),
                            resource_declared_size: is_buffer
                                .then(|| super::i32_after_marker(node, "air.arg_type_size"))
                                .flatten(),
                            resource_type_layout: resource_type_layout.clone(),
                            resource_access: is_buffer
                                .then(|| super::declared_buffer_access(node))
                                .flatten(),
                        })
                    })
                })
        })
        .collect()
}

/// Resource handles an argument buffer declares that [`detect_embedded_textures`] cannot
/// materialize a descriptor for, each named so a refusal can say what was missed.
///
/// Metal lets an argument-buffer member be a user struct that itself holds a texture, sampler or
/// buffer -- `struct texture2d_wrapper { texture2d<half> t; };` as one member. A wrapped texture is
/// surfaced like a flat one, at the argument id [`ArgumentMember::argument_index`] derives; anything
/// else wrapped -- a buffer, a sampler, a function table, or any member of a wrapper declared as an
/// ARRAY of structs -- has no descriptor. Nor does a flat `air.texture` whose access class the
/// read/write lowerings cannot express.
///
/// Such a member decodes as opaque storage, the handle load yields a Private placeholder, and every
/// texture operation on it takes the "resource is absent" path and folds to zero. That path is
/// right for a resource AIR says is absent and wrong for one it declares, so a module carrying one
/// of these refuses rather than sampling black.
///
/// A flat `air.buffer`, `air.sampler` or function-table handle is NOT named here: those have their
/// own representation as an [`EmbeddedArgument`], and the ones that lack a descriptor lack it for
/// reasons this list is not the diagnosis of.
pub(super) fn unsurfaced_embedded_resources(
    nodes: &HashMap<u32, String>,
    argument_buffers: &ArgumentBuffers,
) -> Vec<String> {
    let mut out = vec![];
    for (buffer_param_index, _, sref) in argument_buffers.iter() {
        for member in embedded_argument_members(nodes, sref) {
            let Some(node) = nodes.get(&member.node_ref) else {
                continue;
            };
            if expressible_embedded_texture(&member, node).is_some() {
                continue;
            }
            let Some(role) = role_strings(node)
                .into_iter()
                .find(|role| ArgumentBufferResourceKind::from_air_role(role).is_some())
            else {
                continue;
            };
            let offset = member.field_offset;
            match (member.depth, role.as_str()) {
                (0, "texture") => out.push(format!(
                    "argument buffer parameter {buffer_param_index} holds an air.texture at byte \
                     offset {offset} the embedded lowering cannot express ({})",
                    arg_type_name(node).unwrap_or_else(|| "unnamed".to_string())
                )),
                (0, _) => {}
                (_, _) => out.push(format!(
                    "argument buffer parameter {buffer_param_index} holds an air.{role} at byte \
                     offset {offset} inside a nested struct member"
                )),
            }
        }
    }
    out
}

/// One `air.indirect_argument` resource member of an argument buffer, flattened out of the nested
/// `air.struct_type_info` wrappers Metal allows around it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ArgumentMember {
    /// Member ordinal of the OUTERMOST struct member that reaches this handle -- the first index of
    /// the LLVM GEP that loads it. Members of one wrapper share it; `field_offset` separates them.
    field_ordinal: u32,
    /// Byte offset of the handle from the start of the argument buffer.
    field_offset: u32,
    /// The `air.<role>` node describing the resource.
    node_ref: u32,
    /// Sum of the `air.indirect_argument` ids of the enclosing wrappers; zero for a flat member. A
    /// wrapped member's own `air.location_index` is RELATIVE to it, so the Metal argument-encoder id
    /// is `argument_base + air.location_index`.
    argument_base: u32,
    /// How many `air.struct_type_info` wrappers enclose this handle. Zero for a flat member.
    depth: u32,
    /// Whether any enclosing wrapper is declared as an ARRAY of structs. Such a wrapper repeats its
    /// members once per element, and this walk names element 0 only.
    in_array_wrapper: bool,
    /// The member's OWN declared C-array length (`texture2d<float> slices[32]`), or zero when it is
    /// a single handle. This is a third, independent statement of a descriptor-array length -- see
    /// [`expressible_embedded_texture`] for why the type name cannot answer it.
    array_len: u32,
}

impl ArgumentMember {
    /// The Metal `[[id(n)]]` this resource occupies in its argument buffer.
    ///
    /// Proved against the Metal front end, which refuses
    /// `struct Inner { texture2d<half> x [[id(3)]]; }` at `[[id(7)]]` beside a sibling at
    /// `[[id(10)]]` with "attribute 'id' set location to 10, but minimum is 11": 7 + 3 is taken, so
    /// the ids ADD down the nesting path.
    fn argument_index(&self, node: &str) -> u32 {
        self.argument_base
            .saturating_add(super::location_index(node, self.field_ordinal))
    }

    /// True when this member is a C array of RESOURCE HANDLES, each of which is its own argument id
    /// and its own [`HANDLE_BYTES`] slot. An `air.indirect_constant` array is stored inline and is
    /// one span of data, not N resources, so it is not one of these.
    fn repeats_a_handle(&self, node: &str) -> bool {
        self.array_len > 1
            && role_strings(node)
                .iter()
                .any(|role| ArgumentBufferResourceKind::from_air_role(role).is_some())
    }
}

/// The size of one opaque resource handle in an argument buffer. Every consumer of an embedded
/// resource -- the Metal argument encoder, the Vulkan descriptor writer and `corpus-case-check` --
/// already derives an element from `field_offset` by this stride, so it is stated once here.
const HANDLE_BYTES: u32 = 8;

/// Metal's own argument-buffer nesting limit is far below this; the cap only stops a malformed node
/// graph whose `air.struct_type_info` reaches itself from recursing forever.
const MAX_ARGUMENT_NESTING: u32 = 8;

/// Yield every `air.indirect_argument` resource member of an `air.struct_type_info` node, descending
/// into the nested wrappers Metal allows around one.
///
/// The member grammar itself is [`member_tuples`]; this walk only decides what each tuple means.
/// A tuple whose `air.indirect_argument` suffix is a bare `i32` beside a nested member list is a
/// wrapper, and that `i32` is the wrapper's own argument id (see [`ArgumentMember::argument_index`]).
fn embedded_argument_members(nodes: &HashMap<u32, String>, sref: u32) -> Vec<ArgumentMember> {
    let mut out = vec![];
    collect_argument_members(nodes, sref, Nesting::default(), &mut out);
    out
}

/// What the enclosing `air.struct_type_info` wrappers contribute to every member inside them.
/// `Nesting::default()` is the argument buffer's own member list.
#[derive(Clone, Copy, Debug, Default)]
struct Nesting {
    field_ordinal: u32,
    field_offset: u32,
    argument_base: u32,
    depth: u32,
    in_array_wrapper: bool,
}

fn collect_argument_members(
    nodes: &HashMap<u32, String>,
    sref: u32,
    enclosing: Nesting,
    out: &mut Vec<ArgumentMember>,
) {
    if enclosing.depth > MAX_ARGUMENT_NESTING {
        return;
    }
    let Some(body) = nodes.get(&sref) else {
        return;
    };
    for (field_ordinal, member) in member_tuples(&tokenize(body)).into_iter().enumerate() {
        // A wrapper's members keep the OUTERMOST ordinal, which is the GEP index that reaches them,
        // so a flat member's ordinal is exactly what it was before nesting was walked at all.
        let outer_ordinal = if enclosing.depth == 0 {
            field_ordinal as u32
        } else {
            enclosing.field_ordinal
        };
        match (member.nested, member.argument_wrapper_id) {
            (Some(wrapper), Some(id)) => collect_argument_members(
                nodes,
                wrapper,
                Nesting {
                    field_ordinal: outer_ordinal,
                    field_offset: enclosing.field_offset + member.offset,
                    argument_base: enclosing.argument_base + id,
                    depth: enclosing.depth + 1,
                    in_array_wrapper: enclosing.in_array_wrapper || member.array_len != 0,
                },
                out,
            ),
            _ => {
                if let Some(node_ref) = member.argument_node {
                    out.push(ArgumentMember {
                        field_ordinal: outer_ordinal,
                        field_offset: enclosing.field_offset + member.offset,
                        node_ref,
                        argument_base: enclosing.argument_base,
                        depth: enclosing.depth,
                        in_array_wrapper: enclosing.in_array_wrapper,
                        array_len: member.array_len,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests;

/// Declared resource kind of an argument-buffer field, independent of lowering support.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ArgumentBufferResourceKind {
    Buffer,
    CommandBuffer,
    ComputePipelineState,
    IndirectBuffer,
    IntersectionFunctionTable,
    PrimitiveAccelerationStructure,
    RenderPipelineState,
    Sampler,
    Texture,
    VisibleFunctionTable,
}

impl ArgumentBufferResourceKind {
    pub fn from_air_role(role: &str) -> Option<Self> {
        Some(match role {
            "buffer" => Self::Buffer,
            "command_buffer" => Self::CommandBuffer,
            "compute_pipeline_state" => Self::ComputePipelineState,
            "indirect_buffer" => Self::IndirectBuffer,
            "intersection_function_table" => Self::IntersectionFunctionTable,
            "primitive_acceleration_structure" => Self::PrimitiveAccelerationStructure,
            "render_pipeline_state" => Self::RenderPipelineState,
            "sampler" => Self::Sampler,
            "texture" => Self::Texture,
            "visible_function_table" => Self::VisibleFunctionTable,
            _ => return None,
        })
    }
}
