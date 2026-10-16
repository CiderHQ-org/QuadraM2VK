//! AIR stage-interface metadata parsed from sanitized `.ll`: each entry-function parameter's role
//! (binding, stage input, stage output). Validation's copy of the product's old-engine parser, as
//! far as classification, case checks and the executors read it.

use crate::case::Stage;
use std::collections::HashMap;

mod embedded;
mod fragment;
mod function_constants;
mod globals;
mod intersections;
mod kernel;
mod mesh;
mod textures;
pub(crate) mod types;
mod vertex;
use embedded::{
    body_uses_texture_intrinsic, detect_embedded_arguments, detect_embedded_textures,
    unsurfaced_embedded_resources, ArgumentBuffers,
};
pub use embedded::{ArgumentBufferResourceKind, EmbeddedArgument, EmbeddedTexture};
pub use fragment::{
    parse_air_fragment_meta, DepthQualifier, FragMeta, FragRole, FragmentImageblock,
    FragmentImageblockMember, FragmentImageblockProjection, FragmentImageblockProjectionMember,
    VaryingInterpolation, VaryingSampling,
};
pub use function_constants::FunctionConstant;
pub(crate) use function_constants::{
    function_constant_definedness, function_constant_symbol, global_initializer_body,
    has_function_constant_section, parse_function_constants,
};
use globals::{location_index_with_static, static_init_int_global_values};
pub use intersections::{
    AirIntersectionFamily, AirIntersectionInstancing, AirIntersectionResultField,
};
pub use kernel::{parse_air_kernel_meta, parse_air_mesh_stage_meta, KernMeta, KernRole};
pub use mesh::{
    MeshFieldMeta, MeshFieldRole, MeshOutputsMeta, MeshStageMeta, MeshTopology, PayloadMeta,
};
pub use textures::{
    texture_shape_from_name, TextureComponent, TextureDimension, TextureFormat, TextureShape,
    TEXTURE_HANDLE_ARRAY_DESCRIPTOR_COUNT,
};
pub(crate) use types::storage_air_type_for_size;
use types::{parse_struct_info, struct_info_ref};
pub use types::{AirMember, AirScalar, AirType};
pub use vertex::{
    parse_air_vertex_meta, PatchControlPointField, PatchDomain, TessellationMeta, VertMeta,
    VertOutRole, VertRole,
};

/// Whether stable AIR argument metadata describes a runtime array of device-buffer addresses.
pub fn is_device_buffer_array_type_name(name: &str) -> bool {
    name.chars()
        .filter(|character| !character.is_whitespace())
        .eq("array_ref<void>".chars())
}

/// Declared AIR access on a buffer argument. This is a conservative contract: it may be broader
/// than the specialized body actually uses, but it never understates reads or writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferAccess {
    ReadOnly,
    WriteOnly,
    ReadWrite,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplicitImageblockAttachment {
    pub attachment: u32,
    /// The plane's `imageblock_data_rate`: 0 for the pixel's texel (`read(coord)`), 1 for a
    /// color's (`color`), 2 for a sample's (`sample`).
    pub data_rate: u32,
    pub format: TextureFormat,
    pub reads: bool,
    pub writes: bool,
}

/// One member of an explicit imageblock that `air.alias_implicit_imageblock` aliases onto the
/// implicit imageblock, resolved to the render-target plane it names.
///
/// The alias says the explicit layout IS the implicit layout, and the implicit layout is the colour
/// attachments in order, so member `i` in declaration order is attachment `i`. Nothing else in AIR
/// states the mapping: `air.struct_type_info` gives each member a source name (`color`, `depth`,
/// `customParams`), and those are names the shader author chose, not attachment indices -- a member
/// typed and named `depth` is still colour attachment 2 when it is declared third.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AliasedImageblockPlane {
    /// Colour attachment index, which is the member's position in declaration order.
    pub attachment: u32,
    /// The member's byte offset inside the imageblock cell.
    pub offset: u32,
    pub format: TextureFormat,
    /// The `air.load/store.implicit_imageblock.<suffix>` family that carries this texel type.
    pub intrinsic_suffix: &'static str,
}

/// Resolve an aliased explicit imageblock's members to the render-target planes they alias, or
/// state why they cannot be resolved.
///
/// `Ok(None)` means the parameter declares no member layout at all. An unsupported member type is
/// an error rather than a silently dropped plane: dropping one would stage that member in
/// threadgroup memory while its siblings reached the attachments, which is the exact silence the
/// alias marker exists to prevent.
pub(crate) fn aliased_imageblock_planes(
    layout: &AirType,
) -> Result<Option<Vec<AliasedImageblockPlane>>, String> {
    let AirType::Struct(members) = layout else {
        return Ok(None);
    };
    if members.is_empty() {
        return Ok(None);
    }
    let mut planes = Vec::with_capacity(members.len());
    for (index, member) in members.iter().enumerate() {
        let Some(&(intrinsic_suffix, format)) =
            implicit_texel_suffix(&member.ty).and_then(|suffix| {
                IMPLICIT_IMAGEBLOCK_TEXELS
                    .iter()
                    .find(|(texel, _)| *texel == suffix)
            })
        else {
            return Err(format!(
                "aliased imageblock member {index} has type {:?}, which no implicit imageblock \
                 plane format represents",
                member.ty
            ));
        };
        planes.push(AliasedImageblockPlane {
            attachment: index as u32,
            offset: member.offset,
            format,
            intrinsic_suffix,
        });
    }
    Ok(Some(planes))
}

/// The texel types of the stable AIR implicit-imageblock family
/// (`air.load.implicit_imageblock.<T>`, `air.store.implicit_imageblock.<T>`), each with its exact
/// storage plane format: the attachment format whose texel holds the type's value unchanged. An
/// aliased explicit imageblock's member reaches its plane through the same table
/// ([`aliased_imageblock_planes`]).
const IMPLICIT_IMAGEBLOCK_TEXELS: [(&str, TextureFormat); 7] = [
    ("f16", TextureFormat::R16f),
    ("v2f16", TextureFormat::Rg16f),
    ("v4f16", TextureFormat::Rgba16f),
    ("f32", TextureFormat::R32f),
    ("v2f32", TextureFormat::Rg32f),
    ("v4f32", TextureFormat::Rgba32f),
    ("i32", TextureFormat::R32ui),
];

/// The implicit-imageblock texel suffix AIR spells `ty` with (`f16`, `v2f32`, ...), when `ty` is
/// a scalar or vector of the scalars the family names: `half`, `float`, and `uint`.
fn implicit_texel_suffix(ty: &AirType) -> Option<String> {
    let (scalar, lanes) = match ty {
        AirType::Scalar(scalar) => (scalar, 1),
        AirType::Vec { scalar, lanes } => (scalar, *lanes),
        _ => return None,
    };
    let scalar = match scalar {
        AirScalar::Half => "f16",
        AirScalar::Float => "f32",
        AirScalar::UInt => "i32",
        _ => return None,
    };
    Some(if lanes == 1 {
        scalar.to_owned()
    } else {
        format!("v{lanes}{scalar}")
    })
}

/// Decode the stable AIR implicit-imageblock intrinsic suffix to its exact storage plane format,
/// from the family's table of texel types; a store to the samples a coverage mask selects
/// (`air.store.implicit_imageblock.mask.<T>`) writes the same planes. `Ok(None)` means the symbol
/// is not in this intrinsic family; an unknown family suffix is an explicit error so reflection and
/// corpus capability audits cannot silently omit a new ABI shape.
pub fn implicit_imageblock_texture_format(name: &str) -> Result<Option<TextureFormat>, String> {
    let suffix = name
        .strip_prefix("air.load.implicit_imageblock.")
        .or_else(|| {
            name.strip_prefix("air.store.implicit_imageblock.")
                .map(|suffix| suffix.strip_prefix("mask.").unwrap_or(suffix))
        });
    let Some(suffix) = suffix else {
        return Ok(None);
    };
    IMPLICIT_IMAGEBLOCK_TEXELS
        .iter()
        .find(|(texel, _)| *texel == suffix)
        .map(|&(_, format)| Some(format))
        .ok_or_else(|| format!("{name} has unsupported implicit imageblock texel type"))
}

/// The implicit imageblock attachments of `entry`'s calls: those of the text the entry reaches
/// (its projection), so another entry of the module adds none, and an unknown suffix in another
/// entry's calls cannot stop this one's reflection. Where the text has no projection, every call of
/// it, as before. `None` for an unknown suffix the entry reaches.
fn detect_implicit_imageblock_attachments(
    ll: &str,
    entry: Option<&str>,
) -> Option<Vec<ImplicitImageblockAttachment>> {
    let reached = entry
        .filter(|_| ll.contains("implicit_imageblock."))
        .and_then(|entry| crate::air::entry_projection(ll, entry).ok());
    let ll = reached.as_deref().unwrap_or(ll);
    let mut attachments =
        std::collections::BTreeMap::<(u32, u32, TextureFormat), ImplicitImageblockAttachment>::new(
        );
    for line in ll.lines() {
        let Some(at) = line.find("@air.") else {
            continue;
        };
        let call = &line[at + 1..];
        let Some(open) = call.find('(') else { continue };
        let name = &call[..open];
        let masked = name.starts_with("air.store.implicit_imageblock.mask.");
        let (reads, writes, value_prefix) = if name.starts_with("air.load.implicit_imageblock.") {
            (true, false, 0usize)
        } else if name.starts_with("air.store.implicit_imageblock.") {
            (false, true, 1usize)
        } else {
            continue;
        };
        let Some(close) = call[open + 1..].find(')') else {
            continue;
        };
        let args = crate::air::text::split_top_level(&call[open + 1..open + 1 + close], ',');
        let Some(attachment) = args
            .get(value_prefix)
            .and_then(|arg| typed_u32_constant(arg))
        else {
            continue;
        };
        let data_rate = if masked {
            // `(value, attachment, coord, coverage mask)`: no further operand. It writes the
            // samples the mask selects; the plane it reaches is the pixel's own texel, the one a
            // plain store of the pixel (`write(value, coord)`, data rate 0) writes.
            0
        } else {
            // `(.., coord, data rate, index)`: the `i32` is the data rate, a constant, and the
            // `i16` the color's or sample's index within the pixel, which may be computed and
            // names no other plane. Probed on the M3 Max (macOS 27.0.1's compiler):
            // `read(coord)` passes `i32 0, i16 0`, `read(coord, i, imageblock_data_rate::color)`
            // `i32 1, i16 i`, and `read(coord, i, imageblock_data_rate::sample)` and
            // `write(value, coord, i, imageblock_data_rate::sample)` `i32 2, i16 i`.
            let Some(data_rate) = args
                .get(value_prefix + 2)
                .and_then(|arg| typed_u32_constant(arg))
            else {
                continue;
            };
            data_rate
        };
        let format = implicit_imageblock_texture_format(name).ok().flatten()?;
        let entry = attachments
            .entry((attachment, data_rate, format))
            .or_insert(ImplicitImageblockAttachment {
                attachment,
                data_rate,
                format,
                reads: false,
                writes: false,
            });
        entry.reads |= reads;
        entry.writes |= writes;
    }
    Some(attachments.into_values().collect())
}

fn typed_u32_constant(value: &str) -> Option<u32> {
    value.split_whitespace().last()?.parse().ok()
}

fn body_uses_acceleration_structure_shadow(ll: &str) -> bool {
    ll.contains("@air.get_instance_count_instance_acceleration_structure")
        || ll.contains("@air.get_primitive_acceleration_structure_instance_acceleration_structure")
        || ll.lines().any(|line| {
            let Some(start) = line.find("@air.intersect.") else {
                return false;
            };
            let Some(end) = line[start + 1..].find('(') else {
                return false;
            };
            let callee = &line[start + 1..start + 1 + end];
            AirIntersectionFamily::parse(callee)
                .ok()
                .flatten()
                .is_some_and(|family| family.instancing != AirIntersectionInstancing::None)
        })
}

/// Collect every `!N = !{...}` metadata node body, keyed by N. Shared by both stage parsers.
fn collect_nodes(ll: &str) -> HashMap<u32, String> {
    let mut nodes = HashMap::new();
    for l in ll.lines() {
        let l = l.trim();
        let Some(rest) = l.strip_prefix('!') else {
            continue;
        };
        // expect "<digits> = !{<body>}" or "<digits> = distinct !{<body>}"
        let Some((eq, prefix_len)) =
            rest.find(" = !{")
                .map(|eq| (eq, " = !{".len()))
                .or_else(|| {
                    rest.find(" = distinct !{")
                        .map(|eq| (eq, " = distinct !{".len()))
                })
        else {
            continue;
        };
        let Ok(id) = rest[..eq].parse::<u32>() else {
            continue;
        };
        let body = &rest[eq + prefix_len..];
        let body = body.strip_suffix('}').unwrap_or(body);
        nodes.insert(id, body.to_string());
    }
    nodes
}

#[cfg(test)]
thread_local! {
    static KERNEL_META_DECODE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn entry_name_from_nodes(ll: &str, stage: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    let root = stage_root(ll, stage)?;
    let body = nodes.get(&root)?;
    pointer_symbol(body)
}

fn pointer_symbol(body: &str) -> Option<String> {
    let at = body.find('@')?;
    let after = &body[at + 1..];
    let name = if let Some(quoted) = after.strip_prefix('"') {
        quoted_symbol_name(quoted)?
    } else {
        after
            .chars()
            .take_while(|c| c.is_alphanumeric() || matches!(*c, '_' | '.' | '$'))
            .collect()
    };
    (!name.is_empty()).then_some(name)
}

fn quoted_symbol_name(s: &str) -> Option<String> {
    let mut out = String::new();
    let mut chars = s.chars().peekable();
    while let Some(ch) = chars.next() {
        match ch {
            '"' => return Some(out),
            '\\' => {
                let hi = chars.peek().copied();
                let mut clone = chars.clone();
                let lo = {
                    clone.next();
                    clone.peek().copied()
                };
                if let (Some(hi), Some(lo)) = (hi, lo) {
                    if hi.is_ascii_hexdigit() && lo.is_ascii_hexdigit() {
                        chars.next();
                        chars.next();
                        let byte = u8::from_str_radix(&format!("{hi}{lo}"), 16).ok()?;
                        out.push(byte as char);
                        continue;
                    }
                }
                out.push(chars.next().unwrap_or('\\'));
            }
            _ => out.push(ch),
        }
    }
    None
}

fn function_param_pointer_address_spaces(ll: &str, name: &str) -> Option<HashMap<u32, u32>> {
    let params = function_param_list(ll, name)?;
    let mut out = HashMap::new();
    for (idx, param) in crate::air::text::split_top_level(&params, ',')
        .into_iter()
        .enumerate()
    {
        let param = param.trim_start();
        if !param.starts_with("ptr") {
            continue;
        }
        if let Some(addrspace) = param.find("addrspace(").and_then(|pos| {
            let after = &param[pos + "addrspace(".len()..];
            let digits: String = after.chars().take_while(|c| c.is_ascii_digit()).collect();
            digits.parse::<u32>().ok()
        }) {
            out.insert(idx as u32, addrspace);
        }
    }
    Some(out)
}

fn function_param_list(ll: &str, name: &str) -> Option<String> {
    let unquoted = format!("@{name}(");
    let quoted = format!("@\"{name}\"(");
    let (start, needle_len) = ll
        .find(&unquoted)
        .map(|start| (start, unquoted.len()))
        .or_else(|| ll.find(&quoted).map(|start| (start, quoted.len())))?;
    let start = start + needle_len;
    let mut depth = 1u32;
    let mut end = start;
    for (off, ch) in ll[start..].char_indices() {
        match ch {
            '(' => depth += 1,
            ')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    end = start + off;
                    break;
                }
            }
            _ => {}
        }
    }
    (depth == 0).then(|| ll[start..end].to_string())
}

fn stage_roots(ll: &str, stage: &str) -> Vec<u32> {
    let name = format!("!air.{stage}");
    ll.lines()
        .find_map(|line| {
            let (key, value) = line.trim().split_once('=')?;
            if key.trim() != name {
                return None;
            }
            let body = value.trim().strip_prefix("!{")?.split_once('}')?.0;
            Some(refs_in(body))
        })
        .unwrap_or_default()
}

fn stage_root(ll: &str, stage: &str) -> Option<u32> {
    stage_roots(ll, stage).into_iter().next()
}

/// Select exactly one metadata-rooted entry, `name` of `stage`, as the old engine's translation
/// request did before linking and interface construction: a module with several entries keeps
/// only the selected one's stage root.
pub(crate) fn select_translation_entry(
    ll: String,
    stage: Stage,
    name: &str,
) -> Result<String, String> {
    let nodes = collect_nodes(&ll);
    let mut entries = Vec::new();
    for candidate in Stage::ALL {
        for root in stage_roots(&ll, crate::air::stage_name(candidate)) {
            let symbol = nodes
                .get(&root)
                .and_then(|body| pointer_symbol(body))
                .ok_or_else(|| {
                    format!(
                        "AIR {} entry !{root} has no function reference",
                        crate::air::stage_name(candidate)
                    )
                })?;
            entries.push((candidate, symbol, root));
        }
    }
    let matched = entries
        .iter()
        .enumerate()
        .filter(|(_, (candidate, symbol, _))| *candidate == stage && symbol == name)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    let [selected] = matched[..] else {
        let available = entries
            .iter()
            .map(|(candidate, symbol, _)| {
                format!("{}:{symbol}", crate::air::stage_name(*candidate))
            })
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "entry selection matched {} entries; select one by name/stage (available: {available})",
            matched.len()
        ));
    };
    let root = entries[selected].2;
    if entries.len() == 1 {
        return Ok(ll);
    }
    let mut selected_source = String::with_capacity(ll.len());
    for line in ll.lines() {
        if line.trim().split_once('=').is_some_and(|(key, _)| {
            Stage::ALL
                .into_iter()
                .any(|stage| key.trim() == format!("!air.{}", crate::air::stage_name(stage)))
        }) {
            continue;
        }
        selected_source.push_str(line);
        selected_source.push('\n');
    }
    selected_source.push_str(&format!(
        "!air.{} = !{{!{root}}}\n",
        crate::air::stage_name(stage)
    ));
    Ok(selected_source)
}

/// Refuse unresolved locations in the selected entry's enabled interface.
pub(crate) fn validate_entry_locations(ll: &str, stage: Stage) -> Result<(), String> {
    let nodes = collect_nodes(ll);
    let values = static_init_int_global_values(ll);
    let mut pending = stage_roots(ll, crate::air::stage_name(stage));
    let mut visited = std::collections::HashSet::new();
    while let Some(id) = pending.pop() {
        if !visited.insert(id) {
            continue;
        }
        let Some(body) = nodes.get(&id) else {
            continue;
        };
        if body.contains("!\"air.function_constant_disabled\"")
            || function_constant_gate_disabled(body, &nodes, &values)
        {
            continue;
        }
        for (marker, count) in [("air.location_index", 2), ("air.render_target", 1)] {
            for operand in marker_operands(body, marker)
                .into_iter()
                .take(count)
                .flatten()
            {
                if let LocationOperand::Global(global) = operand {
                    if !values.contains_key(&global) {
                        return Err(format!("unresolved {marker} global {global} in AIR entry metadata !{id}; supply the required function constants"));
                    }
                }
            }
        }
        pending.extend(refs_in(body));
    }
    Ok(())
}

/// One per-entry attribute carried by a stage root node, past its
/// `(function, outputs, inputs)` operand triple.
///
/// All three roots -- `!air.kernel`, `!air.vertex`, `!air.fragment` -- grow their entry
/// attributes in that same tail, but not in the same form: some are references to a keyed node
/// (`!{!"air.patch", ...}`), some are bare strings (`!"early_fragment_tests"`). A stage that
/// scans the tail for only the one attribute it knows cannot tell "no attribute" from "an
/// attribute I have never seen", so a new one is dropped in silence. Decoding the whole tail in
/// one place is what makes the difference observable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum StageAttribute {
    /// `!{!"air.patch", !"<domain>", !"air.patch_control_point", i32 N}` on a post-tessellation
    /// vertex function. Carries the node body; the vertex decode reads the shape out of it.
    Patch(String),
    /// `!"early_fragment_tests"` -- `[[early_fragment_tests]]`. Depth and stencil tests run
    /// before the fragment body, so a fragment the test rejects performs none of the body's
    /// stores. Vulkan spells it `OpExecutionMode ... EarlyFragmentTests`.
    EarlyFragmentTests,
    /// `!{!"air.max_work_group_size", i32 N}` -- `[[max_total_threads_per_threadgroup(N)]]`. The
    /// largest threadgroup this entry was compiled to run.
    MaxWorkGroupSize(u32),
    /// `!{!"air.max_mesh_work_groups", i32 N}` -- an object function's
    /// `[[max_total_threadgroups_per_mesh_grid(N)]]`: the most mesh threadgroups one of its
    /// threadgroups launches.
    MaxMeshWorkGroups(u32),
    /// A tail operand this translator has no model for. Carried out so a stage can refuse rather
    /// than emit a module that silently lacks whatever the attribute asked for.
    Unrecognized(String),
}

impl StageAttribute {
    /// How the attribute reads back in a refusal message.
    pub(crate) fn describe(&self) -> String {
        match self {
            StageAttribute::Patch(_) => "air.patch".to_string(),
            StageAttribute::EarlyFragmentTests => "early_fragment_tests".to_string(),
            StageAttribute::MaxWorkGroupSize(_) => "air.max_work_group_size".to_string(),
            StageAttribute::MaxMeshWorkGroups(_) => "air.max_mesh_work_groups".to_string(),
            StageAttribute::Unrecognized(text) => text.clone(),
        }
    }
}

/// Decode the attribute tail of a stage root node body.
///
/// The first three operands are the entry function pointer, its output node and its input node;
/// every operand after them is an attribute. Recognition is by the attribute's own ABI marker,
/// never by its position in the tail.
fn stage_root_attributes(rootc: &str, nodes: &HashMap<u32, String>) -> Vec<StageAttribute> {
    let mut operands = crate::air::text::split_top_level(rootc, ',');
    if operands.last() == Some(&"") {
        operands.pop();
    }
    operands
        .into_iter()
        .skip(3)
        .filter_map(|operand| {
            let referenced = operand
                .strip_prefix('!')
                .and_then(|digits| digits.parse::<u32>().ok())
                .and_then(|id| nodes.get(&id));
            Some(match referenced {
                Some(node) if node.contains("!\"air.patch\"") => {
                    StageAttribute::Patch(node.clone())
                }
                Some(node) if node.contains("!\"air.max_work_group_size\"") => {
                    match i32_after_marker(node, "air.max_work_group_size") {
                        Some(size) => StageAttribute::MaxWorkGroupSize(size),
                        None => StageAttribute::Unrecognized(
                            "air.max_work_group_size states no thread count".to_string(),
                        ),
                    }
                }
                Some(node) if node.contains("!\"air.max_mesh_work_groups\"") => {
                    match i32_after_marker(node, "air.max_mesh_work_groups") {
                        Some(count) => StageAttribute::MaxMeshWorkGroups(count),
                        None => StageAttribute::Unrecognized(
                            "air.max_mesh_work_groups states no threadgroup count".to_string(),
                        ),
                    }
                }
                // A node with no operands states nothing, so there is nothing here to model and
                // nothing emission could drop. Refusing it would cost a translation for no safety.
                Some(node) if node.trim().is_empty() => return None,
                Some(node) => StageAttribute::Unrecognized(format!("!{{{node}}}")),
                None if operand == "!\"early_fragment_tests\"" => {
                    StageAttribute::EarlyFragmentTests
                }
                None => StageAttribute::Unrecognized(operand.to_owned()),
            })
        })
        .collect()
}

/// The metadata refs (`!N`) appearing in a node body, in order. Skips `ptr @func` operands.
fn refs_in(body: &str) -> Vec<u32> {
    body.split(',')
        .filter_map(|s| {
            s.trim()
                .strip_prefix('!')
                .and_then(|x| x.parse::<u32>().ok())
        })
        .collect()
}

/// The `i32 N` immediate (parameter index) inside a metadata node body, if present.
fn first_i32(body: &str) -> Option<u32> {
    let mut it = body.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        if tok == "i32" {
            if let Some(n) = it.peek() {
                let n = n.trim_end_matches(',');
                if let Ok(v) = n.parse::<u32>() {
                    return Some(v);
                }
            }
        }
    }
    None
}

fn i32_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    first_i32(&body[pos + marker.len()..])
}

fn location_index(body: &str, fallback: u32) -> u32 {
    match location_operands(body).map(|operands| operands.index) {
        Some(LocationOperand::Literal(index)) => index,
        _ => fallback,
    }
}

/// How many descriptors an argument node states it occupies, when AIR states more than one.
///
/// `None` for the ordinary single-descriptor argument, and for a count spelled as a
/// function-constant global -- the consumer picks that length at pipeline creation, so there is no
/// compile-time array to size.
fn declared_descriptor_count(body: &str) -> Option<u32> {
    match location_operands(body)?.count? {
        LocationOperand::Literal(count) if count > 1 => Some(count),
        _ => None,
    }
}

/// One operand of the `air.location_index` pair: either a compile-time value or a pointer to a
/// function-constant global whose value the consumer chooses at pipeline creation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum LocationOperand {
    Literal(u32),
    Global(String),
}

/// The two operands `air.location_index` always carries: the resource's Metal slot, and how many
/// descriptors the argument occupies at that slot.
///
/// `!"air.location_index", i32 4, i32 2` is `[[texture(4)]]` holding two texture handles. The count
/// is `1` for an ordinary resource; above `1` the argument is a handle ARRAY, and every corpus
/// declaration above `1` agrees with the `array<..., N>` length in the type name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct LocationOperands {
    pub index: LocationOperand,
    pub count: Option<LocationOperand>,
}

/// The operands that follow `marker` in an argument node, POSITIONALLY, each decoded as a literal
/// or as the function-constant global that supplies it at pipeline creation.
///
/// AIR spells a slot either way, and a decoder that scans forward for "the next `i32`" or "the next
/// `@`" reads a DIFFERENT operand whenever the one it wants is spelled the other way. Measured over
/// 14579 corpus sources, `air.location_index`'s pair is 112946 `(i32, i32)`, 3404 `(ptr, i32)`,
/// 1447 `(ptr, ptr)` and 439 `(i32, ptr)`, and `air.render_target`'s first operand is 3213 `i32`
/// and 103 `ptr` -- so scanning finds the descriptor COUNT in place of the texture slot, or the
/// dual-source index in place of the color attachment.
fn marker_operands(body: &str, marker: &str) -> Vec<Option<LocationOperand>> {
    let marker = format!("!\"{marker}\"");
    let Some(position) = body.find(&marker) else {
        return vec![];
    };
    split_metadata_operands(&body[position + marker.len()..])
        .into_iter()
        .map(|operand| {
            if let Some(literal) = operand.strip_prefix("i32 ") {
                return literal.trim().parse().ok().map(LocationOperand::Literal);
            }
            let at = operand.find('@')?;
            let name = operand[at..]
                .chars()
                .take_while(|character| {
                    !character.is_whitespace() && !matches!(*character, ',' | ')' | '(' | '[' | ']')
                })
                .collect::<String>();
            (name.len() > 1).then_some(LocationOperand::Global(name))
        })
        .collect()
}

/// Decode the `air.location_index` operand pair.
pub(crate) fn location_operands(body: &str) -> Option<LocationOperands> {
    let mut operands = marker_operands(body, "air.location_index").into_iter();
    Some(LocationOperands {
        index: operands.next().flatten()?,
        count: operands.next().flatten(),
    })
}

/// The Metal slot a node declares under `marker`: `air.location_index` for a bound resource,
/// `air.render_target` for a color attachment.
///
/// Both are the FIRST operand after their marker, and both may name a function-constant global
/// instead of a literal. `fallback` -- the parameter or member ordinal -- stands in when the global
/// is one this module does not initialize, because the operands AFTER the slot describe something
/// else entirely and answering with one of those is answering the wrong question.
pub(crate) fn declared_slot(
    body: &str,
    marker: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    match marker_operands(body, marker).into_iter().next().flatten() {
        Some(LocationOperand::Literal(slot)) => slot,
        Some(LocationOperand::Global(global)) => {
            static_int_globals.get(&global).copied().unwrap_or(fallback)
        }
        None => fallback,
    }
}

/// Split an AIR metadata operand list on the commas that separate operands. A `!"..."` string
/// operand may itself contain commas (`!"array<texture2d<half, sample>, 2>"`), so a plain
/// `split(',')` would tear one operand into three and shift every position after it.
fn split_metadata_operands(tail: &str) -> Vec<String> {
    let mut operands = vec![];
    let mut current = String::new();
    let mut quoted = false;
    for character in tail.chars() {
        match character {
            '"' => {
                quoted = !quoted;
                current.push(character);
            }
            ',' if !quoted => {
                operands.push(std::mem::take(&mut current).trim().to_string());
            }
            _ => current.push(character),
        }
    }
    operands.push(current.trim().to_string());
    operands.retain(|operand| !operand.is_empty());
    operands
}

fn render_target_location(
    body: &str,
    fallback: u32,
    static_int_globals: &HashMap<String, u32>,
) -> u32 {
    declared_slot(body, "air.render_target", fallback, static_int_globals)
}

fn address_space(body: &str) -> Option<u32> {
    i32_after_marker(body, "air.address_space")
}

/// The first `air.<role>` string literal in a node body (e.g. `air.texture`).
fn role_strings(body: &str) -> Vec<String> {
    let mut out = vec![];
    let mut rest = body;
    while let Some(p) = rest.find("!\"air.") {
        let after = &rest[p + 6..];
        let end = after.find('"').unwrap_or(after.len());
        out.push(after[..end].to_string());
        rest = &after[end..];
    }
    out
}

/// The primary arg-kind role string for an argument node, looking past a leading
/// `air.function_constant` wrapper. A conditionally-present (function-constant-gated) resource
/// emits `!"air.function_constant", !REF` BEFORE its real `!"air.<role>"` marker, so a naive
/// "first role string" sees `function_constant` and mis-classifies the argument as `Other`. The
/// real role is the first marker that isn't the wrapper. `air.function_constant` is a stable AIR
/// metadata-ABI symbol, not a shader identifier.
fn primary_role(strs: &[String]) -> Option<&str> {
    strs.iter()
        .map(String::as_str)
        .find(|s| *s != "function_constant")
}

/// The role marker a declaration carries, read POSITIONALLY past an `air.function_constant`
/// wrapper, or `None` when it carries none.
///
/// The wrapper is the operand PAIR `!"air.function_constant", !PREDICATE`; the role, when there is
/// one, is the operand that follows the predicate. Both other readings of this node are wrong here:
///
/// - Scanning for "the first `air.` marker that is not the wrapper" reads a BARE function-constant
///   parameter's own `air.arg_type_name` as though it were a role. That parameter *is* the function
///   constant -- there is no role behind the wrapper at all -- and the positional read answers
///   `None` for it, because the operand after its predicate is not an `air.` marker.
/// - Asking [`resource_input_role`] answers `None` for every gated role the emitter does not PROMOTE,
///   and that is precisely the set this predicate exists to name. A gated `air.amplification_id` on
///   a fragment entry was classified `Other`, bound to a zero and never reported -- in a module that
///   validated, bound and reflected as though nothing were missing. Whether the gate is on is a
///   DIFFERENT question (see [`unmodelled_declared_role`]), not a reason to stop reading.
///
/// Measured: every one of the 14000-odd function-constant-wrapped argument and return nodes across
/// the 14579 local corpus sources carries a real role marker in that position, and none carries a
/// qualifier -- [`ARGUMENT_IDENTITY_MARKERS`] is what answers the role-less parameter whichever way
/// AIR spells its wrapper.
fn declared_role(body: &str) -> Option<String> {
    let role_name = |operand: &str| {
        operand
            .strip_prefix("!\"air.")
            .and_then(|rest| rest.strip_suffix('"'))
            .map(str::to_string)
    };
    let mut operands = split_metadata_operands(body).into_iter();
    let first = operands.find_map(|operand| role_name(&operand))?;
    if first != "function_constant" {
        return (first != "function_constant_disabled").then_some(first);
    }
    // The predicate the wrapper gates on. A wrapper carrying none gates nothing and wraps nothing.
    operands.next().filter(|operand| {
        operand.strip_prefix('!').is_some_and(|rest| {
            !rest.is_empty() && rest.chars().all(|character| character.is_ascii_digit())
        })
    })?;
    operands
        .next()
        .as_deref()
        .and_then(role_name)
        .filter(|role| !ARGUMENT_IDENTITY_MARKERS.contains(&role.as_str()))
}

/// The two markers that name an argument's C-level identity rather than its role. Every argument
/// node carries both whatever else it declares, so one of them sitting where a role would sit means
/// the declaration has no role: it is the bare function constant itself.
const ARGUMENT_IDENTITY_MARKERS: &[&str] = &["arg_name", "arg_type_name"];

fn function_constant_gate_global(body: &str, nodes: &HashMap<u32, String>) -> Option<String> {
    if role_strings(body).first().map(String::as_str) != Some("function_constant") {
        return None;
    }
    refs_in(body)
        .into_iter()
        .filter_map(|r| nodes.get(&r))
        .find_map(|node| {
            let at = node.find('@')?;
            let name = node[at..]
                .chars()
                .take_while(|c| !c.is_whitespace() && !matches!(*c, ',' | ')' | '(' | '[' | ']'))
                .collect::<String>();
            (name.len() > 1).then_some(name)
        })
}

/// Specialize metadata-only `air.function_constant` wrappers from statically evaluated AIR
/// predicates. A true predicate exposes the wrapped role as unconditional. A false predicate is
/// marked disabled so interface construction cannot retain an unreachable descriptor or output.
/// Unknown predicates keep the wrapper and the ordinary unspecialized projection.
pub(crate) fn specialize_function_constant_metadata(ll: &str) -> String {
    let nodes = collect_nodes(ll);
    let static_int_globals = static_init_int_global_values(ll);
    let marker = "!\"air.function_constant\"";
    let mut output = String::with_capacity(ll.len());

    for line in ll.split_inclusive('\n') {
        let value = function_constant_gate_global(line, &nodes)
            .and_then(|global| static_int_globals.get(&global).copied());
        match value {
            Some(0) => {
                output.push_str(&line.replacen(marker, "!\"air.function_constant_disabled\"", 1))
            }
            Some(_) => {
                let rewritten = line.find(marker).and_then(|start| {
                    let after_marker = &line[start + marker.len()..];
                    let next_role = after_marker.find("!\"air.")?;
                    Some(format!("{}{}", &line[..start], &after_marker[next_role..]))
                });
                if let Some(rewritten) = rewritten {
                    output.push_str(&rewritten);
                } else {
                    output.push_str(line);
                }
            }
            None => output.push_str(line),
        }
    }
    output
}

/// True when the node's `air.function_constant` gate is one this module's own static initializers
/// drive to ZERO -- the argument is absent from the variant this module describes.
///
/// This is deliberately NOT [`metadata_enabled_by_default`] negated. That predicate answers "may
/// this declaration be present", so an UNRESOLVED gate is false for it. Here the question is the
/// opposite one -- "did the variant the initializers describe leave this argument out" -- and an
/// unresolved gate is no evidence of that.
fn function_constant_gate_disabled(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .and_then(|global| static_int_globals.get(&global).copied())
        == Some(0)
}

/// True when the node's `air.function_constant` gate is one this module's own static initializers
/// drive NONZERO -- the argument is PRESENT in the variant this module describes.
///
/// The mirror of [`function_constant_gate_disabled`], and it has to be a separate question from
/// [`metadata_enabled_by_default`] for the same reason that one is: an UNRESOLVED gate reads as
/// "may be present" there, and "may be present" is not evidence of presence any more than it is
/// evidence of absence. Only the resolved-nonzero case is.
fn function_constant_gate_enabled(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .and_then(|global| static_int_globals.get(&global).copied())
        .is_some_and(|value| value != 0)
}

/// The role a gated BUFFER argument carries once the module's own initializers have answered
/// whether it is there.
///
/// [`FC_PROMOTED_RESOURCE_ROLES`] deliberately omits `buffer`: a gated buffer whose gate this
/// module cannot resolve may or may not be bound, and the "possibly-absent -> Private zero
/// placeholder" default is the honest answer for it. But that argument is about an UNRESOLVED gate,
/// and it was being applied to a resolved-ON one too. A gate this module's own static initializers
/// drive nonzero states that the variant this module describes DOES carry the buffer -- the same
/// evidence standard [`function_constant_gate_disabled`] uses to drop a descriptor in the other
/// direction -- and binding a definitely-present buffer to a Private zero sends its every store
/// into per-invocation scratch.
///
/// It also settles a split reading: the fragment decode reaches the wrapped role through
/// [`primary_role`], so a gated buffer on a fragment entry has always been a buffer, while kernel
/// and vertex read it through [`gated_role`] and left it a placeholder. One declaration, two
/// classifications.
///
/// The remaining half of that split -- a gated-OFF buffer, which the fragment still binds and the
/// other two still drop -- stays, and stays deliberately. Making the fragment agree costs 490
/// buffer bindings and 1060 loads across 100 corpus modules and rescues nothing; a gated-off
/// argument declaring a LITERAL slot is Metal's spelling of mutually exclusive alternatives, and
/// no gated fragment buffer in the corpus declares anything else. Measured in
/// `a_fragment_gated_buffer_keeps_its_literal_slot_whichever_way_the_gate_resolves`.
fn present_gated_buffer_role<'a>(
    role: &'a str,
    strs: &'a [String],
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> &'a str {
    if role != "function_constant" || primary_role(strs) != Some("buffer") {
        return role;
    }
    if function_constant_gate_enabled(body, nodes, static_int_globals) {
        "buffer"
    } else {
        role
    }
}

/// The Metal texture slot an argument node declares FOR THE VARIANT THIS MODULE DESCRIBES, or
/// `None` when it declares none and the argument therefore has no descriptor to bind.
///
/// Two spellings mean "no slot", and reading only the first of them decorated two textures with one
/// descriptor in 250 of the 14579 local corpus sources:
///
/// - `air.location_index` resolving to `-1`: AIR states outright that specialization has not
///   assigned a binding yet.
/// - A slot supplied by a function-constant global on an argument whose own gate the module's
///   initializers drive to zero. Metal compiles `[[texture(fc_expr)]]` into a RUNNING SUM over the
///   arguments the pipeline enables, so the global holds this argument's slot only when this
///   argument is one of them. For an argument the variant leaves out, the same global holds
///   wherever the sum stopped -- which is a LIVE argument's slot. One corpus fragment shader put
///   seventeen differently-shaped textures on `Binding 38` that way, and a kernel put a
///   `texture2d<float, write>` and a 128-element `array_ref` on `Binding 480`. Reflection then
///   asks a consumer to bind two different textures to one Metal index, and the emitted module
///   aliases descriptors whose only correct reading is the one the shader is not compiled for.
///
/// An argument gated off but declaring a LITERAL slot keeps it: `[[texture(0), function_constant(a)]]`
/// beside `[[texture(0), function_constant(!a)]]` is Metal's own way of spelling mutually exclusive
/// typed alternatives, the slot is stated rather than summed, and the alternatives are never live
/// together.
fn variant_texture_slot(
    body: &str,
    fallback: u32,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> Option<u32> {
    let slot = location_index_with_static(body, fallback, static_int_globals);
    if slot == u32::MAX {
        return None;
    }
    let summed_by_function_constant = matches!(
        location_operands(body).map(|operands| operands.index),
        Some(LocationOperand::Global(_))
    );
    if summed_by_function_constant
        && function_constant_gate_disabled(body, nodes, static_int_globals)
    {
        return None;
    }
    Some(slot)
}

/// The role marker a texture argument carries once [`variant_texture_slot`] has found the variant
/// declares no slot for it. It is deliberately not `"texture"` and deliberately not
/// `"function_constant"`: the first would ask for a descriptor that does not exist, and the second
/// throws away which argument was demoted, which is the only thing that tells an absent resource
/// apart from a handle the lowering merely lost.
/// True when this declaration is a TEXTURE the variant this module describes leaves out, spelled
/// the way [`specialize_function_constant_metadata`] spells it.
///
/// Two spellings state that one fact. `air.function_constant` beside a gate this module's own
/// static initializers drive to zero is what an unspecialized module carries, and
/// [`variant_texture_slot`] demotes it. Once a VALUE is supplied, the specializer rewrites that
/// same node to `air.function_constant_disabled`, after which the wrapped role is not readable at
/// all -- [`declared_role`] answers `None` for it by design, because a disabled declaration
/// declares nothing. That is right for the interface and wrong for the operand: the argument still
/// reaches the body, still binds to a Private placeholder, and is still the absent resource. Naming
/// only the first spelling is what left every function-constant-specialized translation reaching
/// the recovery in `recovered_image_for_private_operand` that the unspecialized one no longer does.
///
/// The role is read POSITIONALLY: `role_strings` skips the wrapper's `!PREDICATE` operand, so the
/// marker after the wrapper is the role. `texture` is never an [`ARGUMENT_IDENTITY_MARKERS`] entry,
/// so this cannot mistake a bare function-constant parameter for one.
fn declares_disabled_texture(strs: &[String]) -> bool {
    strs.first().map(String::as_str) == Some("function_constant_disabled")
        && strs.get(1).map(String::as_str) == Some("texture")
}

const VARIANT_ABSENT_TEXTURE_ROLE: &str = "variant_absent_texture";

/// The role a declaration carries that this stage's emitter does NOT model, or `None` when the
/// role is one it models -- or when the variant this module describes leaves the declaration out.
///
/// Reporting the role is what turns an unsupported entry parameter or return member into an honest
/// `FALLBACK` instead of a silently dropped input. The four callers (kernel, fragment and vertex
/// entry parameters, and fragment return members) differ only in which role list is theirs, so the
/// rule -- including which gates excuse a refusal -- is stated once here.
///
/// The excusing predicate is [`function_constant_gate_disabled`] and deliberately NOT
/// `!metadata_enabled_by_default`. Those two are not complements. `metadata_enabled_by_default`
/// answers "may this declaration be present", so an UNRESOLVED gate reads as "absent" for it --
/// and an unmodelled role behind a gate we could not evaluate is still unmodelled. Suppressing the
/// refusal there is precisely the case where the emitter goes on to bind a zero for a parameter the
/// pipeline may well have supplied a value for. Only a gate this module's own static initializers
/// drive to zero is evidence that the variant left the declaration out.
fn unmodelled_declared_role(
    is_modelled: impl Fn(&str) -> bool,
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> Option<String> {
    let declared = declared_role(body)?;
    (!is_modelled(&declared) && !function_constant_gate_disabled(body, nodes, static_int_globals))
        .then_some(declared)
}

fn metadata_enabled_by_default(
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> bool {
    function_constant_gate_global(body, nodes)
        .map(|global| {
            static_int_globals
                .get(&global)
                .is_some_and(|value| *value != 0)
        })
        .unwrap_or(true)
}

/// Resource roles whose ABI survives a function-constant wrapper. Samples need both
/// texture and sampler declarations. Buffer presence instead requires predicate evidence
/// through `present_gated_buffer_role`, rather than role-only promotion.
const FC_PROMOTED_RESOURCE_ROLES: &[&str] = &[
    "imageblock",
    "intersection_function_table",
    "sampler",
    "stage_in",
    "texture",
    "visible_function_table",
];

/// Resolve gated resource and system-value roles. Gated buffer presence is resolved
/// separately from its predicate by `present_gated_buffer_role`.
fn resource_input_role(strs: &[String]) -> Option<&str> {
    gated_role(strs, |role| {
        FC_PROMOTED_RESOURCE_ROLES.contains(&role) || air_role_is_system_value(role)
    })
}

/// The role marker an interface node declares, resolving an `air.function_constant` wrapper against
/// the roles the reading stage models.
///
/// Every stage decode asks this one question — "a gated declaration: which role is it?" — and the
/// answer has to be the SAME for the same declaration, because a stage interface is matched across
/// stages by position and by `Location`, not by name. Each decode used to answer it its own way:
/// the fragment lists read the role past the wrapper unconditionally, the vertex output list read
/// it through a resource allowlist that named no output role at all. So one gated member of one
/// Metal varying struct was a varying on the fragment side and nothing on the vertex side, and
/// every later varying in that struct landed on a different `Location` in the two modules — a
/// silent read of the wrong interpolant, in two modules that each validate.
///
/// `modelled` answers "does the caller lower this role", asked of the caller's own lowering table,
/// so a role the stage handles cannot go missing from a second hand-maintained copy of that list.
/// A wrapper with anything else behind it
/// — an unmodelled role, or a BARE function constant whose next marker is its own
/// `air.arg_type_name` — stays the wrapper. `air.function_constant_disabled` (an
/// AIR-specialized-off predicate) is never a role: it is not the wrapper this looks past.
fn gated_role(strs: &[String], modelled: impl Fn(&str) -> bool) -> Option<&str> {
    let first = strs.first().map(String::as_str)?;
    if first != "function_constant" {
        return Some(first);
    }
    Some(match primary_role(strs) {
        Some(role) if modelled(role) => role,
        _ => first,
    })
}

fn string_after_marker(body: &str, marker: &str) -> Option<String> {
    let marker = format!("!\"{marker}\"");
    let pos = body.find(&marker)?;
    let after = &body[pos + marker.len()..];
    let pos = after.find("!\"")?;
    let value = &after[pos + 2..];
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

fn arg_type_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_type_name")
}

fn declared_buffer_access(body: &str) -> Option<BufferAccess> {
    if body.contains("air.read_write") {
        Some(BufferAccess::ReadWrite)
    } else if body.contains("air.write") {
        Some(BufferAccess::WriteOnly)
    } else if body.contains("air.read") {
        Some(BufferAccess::ReadOnly)
    } else {
        None
    }
}

fn arg_name(body: &str) -> Option<String> {
    string_after_marker(body, "air.arg_name")
}

fn ref_after_marker(body: &str, marker: &str) -> Option<u32> {
    let marker = format!("!\"{marker}\"");
    let after = body.get(body.find(&marker)? + marker.len()..)?;
    let bang = after.find('!')?;
    let digits = after[bang + 1..]
        .chars()
        .take_while(char::is_ascii_digit)
        .collect::<String>();
    digits.parse().ok()
}

/// The fragment entry-parameter roles the emitter models.
///
/// Any other enabled role lands in [`FragMeta::unmodelled_input_params`] and is rejected at
/// emission. `render_target` appears here because an `air.render_target` in the *input* list is
/// framebuffer fetch (`[[color(n)]]`), not an output.
/// The AIR entry-parameter roles that name a SPIR-V builtin rather than a bound resource, across
/// every stage. The execution-group family answers for itself -- see [`air_role_is_system_value`].
///
/// A resource keeps its descriptor whether or not its function constant is on: the pipeline layout
/// has to match what the application binds either way. A system value does not. When its constant
/// is off the parameter is absent, and declaring the builtin anyway puts a variable in the entry
/// point interface — and, for `viewport_array_index` and `render_target_array_index`, a device
/// capability in the module — for a value the shader cannot read.
///
/// The converse is the half that went missing for two stages. The fragment decode reads the role
/// past its wrapper unconditionally and consults this list only to drop the gated-OFF ones; the
/// kernel and vertex decodes asked `resource_input_role`, which collapses every role it does not
/// PROMOTE back to the wrapper -- gated-ON builtins included. Those bound a zero for a value the
/// pipeline had, which is the same silent read the list exists to prevent, from the other side.
pub(crate) const AIR_SYSTEM_VALUE_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "barycentric_coord",
    "dispatch_threads_per_threadgroup",
    "front_facing",
    "instance_id",
    "patch_id",
    "point_coord",
    "position",
    "position_in_patch",
    "primitive_id",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "vertex_id",
    "viewport_array_index",
];

/// Whether an AIR entry-parameter role names a system value. See [`AIR_SYSTEM_VALUE_ROLES`]; the
/// execution-group family is decoded from the marker's shape rather than listed, so it is asked
/// rather than enumerated and a group added to [`AIR_EXECUTION_GROUPS`] needs no second entry here.
pub(crate) fn air_role_is_system_value(role: &str) -> bool {
    AIR_SYSTEM_VALUE_ROLES.contains(&role) || execution_group_role(role).is_some()
}

/// The role a stage should classify a declaration as, once a gated-OFF system value is accounted
/// for: that parameter is ABSENT, and binding a zero for it is what Metal defines. Blanked rather
/// than dropped so the caller's own `match` falls through to its unmodelled arm.
///
/// One statement of the rule for all three stages. It is only about system values: a RESOURCE keeps
/// its descriptor whichever way its gate goes, because the pipeline layout has to match what the
/// application binds either way.
fn present_system_value_role<'a>(
    role: &'a str,
    body: &str,
    nodes: &HashMap<u32, String>,
    static_int_globals: &HashMap<String, u32>,
) -> &'a str {
    if air_role_is_system_value(role)
        && !metadata_enabled_by_default(body, nodes, static_int_globals)
    {
        return "";
    }
    role
}

pub(crate) const FRAGMENT_INPUT_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "barycentric_coord",
    "buffer",
    "fragment_input",
    "front_facing",
    "imageblock_data",
    "indirect_buffer",
    "intersection_function_table",
    "point_coord",
    "position",
    "primitive_id",
    "render_target",
    "render_target_array_index",
    "sample_id",
    "sample_mask_in",
    "sampler",
    "texture",
    "viewport_array_index",
    "visible_function_table",
];

/// The vertex entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
pub(crate) const VERTEX_INPUT_ROLES: &[&str] = &[
    "amplification_count",
    "amplification_id",
    "buffer",
    "indirect_buffer",
    "instance_id",
    "intersection_function_table",
    "patch_control_point_input",
    "patch_id",
    "patch_input",
    "position_in_patch",
    "sampler",
    "texture",
    "vertex_id",
    "vertex_input",
    "visible_function_table",
];

/// The kernel entry-parameter roles the emitter models. See [`FRAGMENT_INPUT_ROLES`].
///
/// The execution-group family is NOT listed here. Its members are decoded from the marker's shape by
/// [`execution_group_role`], and [`air_input_role_is_modelled`] asks that decoder rather than a
/// second spelling of the same family -- a member could otherwise have a lowering and still be
/// refused for want of a line in a list.
pub(crate) const KERNEL_INPUT_ROLES: &[&str] = &[
    "buffer",
    "dispatch_threads_per_threadgroup",
    "imageblock",
    "indirect_buffer",
    "instance_acceleration_structure",
    "intersection_function_table",
    "primitive_acceleration_structure",
    "sampler",
    "stage_in",
    "texture",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "visible_function_table",
];

/// An object or mesh function's parameter roles beside its own stage's ([`mesh::MESH_STAGE_ROLES`]):
/// a kernel's, less the stage input, the imageblock and the requested threadgroup size, which only
/// a dispatch states.
pub(crate) const MESH_PIPELINE_INPUT_ROLES: &[&str] = &[
    "buffer",
    "indirect_buffer",
    "instance_acceleration_structure",
    "intersection_function_table",
    "primitive_acceleration_structure",
    "sampler",
    "texture",
    "thread_index_in_threadgroup",
    "thread_position_in_grid",
    "thread_position_in_threadgroup",
    "threadgroup_position_in_grid",
    "threadgroups_per_grid",
    "threads_per_grid",
    "threads_per_threadgroup",
    "visible_function_table",
];

/// Whether the emitter models `role` on an entry parameter of `stage`.
///
/// The stage's own inventory, plus the execution-group facts that stage can answer. Those are read
/// off the marker's shape, so the family needs no entry in any inventory and cannot end up listed
/// for one stage and forgotten for another -- which is how it started, listed for the kernel alone
/// while a fragment declaring `[[thread_index_in_simdgroup]]` was refused outright.
pub(crate) fn air_input_role_is_modelled(stage: Stage, role: &str) -> bool {
    let inventory = match stage {
        Stage::Fragment => FRAGMENT_INPUT_ROLES,
        Stage::Vertex => VERTEX_INPUT_ROLES,
        Stage::Kernel => KERNEL_INPUT_ROLES,
        Stage::Object | Stage::Mesh => {
            return (MESH_PIPELINE_INPUT_ROLES.contains(&role)
                || mesh::MESH_STAGE_ROLES.contains(&role))
                || stage_execution_group_role(stage, role).is_some()
        }
    };
    inventory.contains(&role) || stage_execution_group_role(stage, role).is_some()
}

/// Metal's execution groups inside a threadgroup, and how many threads each holds.
///
/// A quadgroup is four threads and a simdgroup thirty-two, and EVERY role in the family --
/// a lane's index inside its group, its group's index inside the threadgroup, how many groups the
/// threadgroup holds, and the group's own width -- is one derivation over that single number.
/// Spelling the family out per group is what left `air.quadgroups_per_threadgroup` with no arm at
/// all beside `air.simdgroups_per_threadgroup`, so a kernel declaring it read a zero where the
/// hardware had a count.
pub(crate) const AIR_EXECUTION_GROUPS: &[(&str, u32)] = &[("quadgroup", 4), ("simdgroup", 32)];

/// One fact an entry parameter can ask about the execution group its thread runs in.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutionGroupFact {
    /// `[[thread_index_in_simdgroup]]`, `[[thread_index_in_quadgroup]]`: this lane's index inside
    /// its own group.
    ThreadIndexInGroup,
    /// `[[simdgroup_index_in_threadgroup]]`, `[[quadgroup_index_in_threadgroup]]`: which group
    /// inside the threadgroup this lane belongs to.
    GroupIndexInThreadgroup,
    /// `[[simdgroups_per_threadgroup]]`, `[[quadgroups_per_threadgroup]]`: how many whole groups the
    /// threadgroup's local size divides into.
    GroupsPerThreadgroup,
    /// `[[threads_per_simdgroup]]`: the group's width.
    ThreadsPerGroup,
}

impl ExecutionGroupFact {
    /// Whether a shader stage has to be dispatched as a threadgroup grid for this fact to mean
    /// anything.
    ///
    /// A lane's index inside its group and the group's own width are facts about the GROUP: every
    /// stage runs its threads in execution groups, and this translator states both off
    /// `SubgroupLocalInvocationId` and the group's width, neither of which mentions a workgroup.
    /// The other two are facts about the THREADGROUP the groups divide -- which one this thread's
    /// group is, and how many the local size holds -- and only a compute grid has one, so a
    /// fragment or vertex entry declaring them is refused rather than answered with a zero.
    pub fn needs_a_threadgroup(self) -> bool {
        match self {
            Self::ThreadIndexInGroup | Self::ThreadsPerGroup => false,
            Self::GroupIndexInThreadgroup | Self::GroupsPerThreadgroup => true,
        }
    }
}

/// The execution-group fact an AIR role marker names and the group's width, or `None` when the
/// marker names no execution group.
///
/// Read from the marker's SHAPE against [`AIR_EXECUTION_GROUPS`] rather than from an arm per
/// (group, fact) pair, so a member of the family cannot be modelled for one group and missing for
/// the other. The four shapes are exhaustive over the family and unambiguous: no marker outside it
/// matches one and then names a group in the table.
///
/// The answer carries no stage in it -- a Metal simdgroup is the same thirty-two threads whichever
/// entry declares it. [`ExecutionGroupFact::needs_a_threadgroup`] is where a stage enters.
///
/// `dispatch_` is Metal's "what the dispatch asked for" spelling, and it is stripped here only so a
/// member of THIS family cannot be modelled under one spelling and missing under the other. It does
/// not follow that the two spellings denote one value: for the threadgroup they do not, which is why
/// `air.dispatch_threads_per_threadgroup` is its own [`KernRole::DispatchThreadsPerThreadgroup`].
/// No group in [`AIR_EXECUTION_GROUPS`] is the threadgroup, so that role never reaches this decode.
pub(crate) fn execution_group_role(marker: &str) -> Option<(ExecutionGroupFact, u32)> {
    let marker = marker.strip_prefix("dispatch_").unwrap_or(marker);
    let (fact, group) = if let Some(group) = marker.strip_prefix("thread_index_in_") {
        (ExecutionGroupFact::ThreadIndexInGroup, group)
    } else if let Some(group) = marker.strip_suffix("_index_in_threadgroup") {
        (ExecutionGroupFact::GroupIndexInThreadgroup, group)
    } else if let Some(group) = marker.strip_suffix("s_per_threadgroup") {
        (ExecutionGroupFact::GroupsPerThreadgroup, group)
    } else {
        let group = marker.strip_prefix("threads_per_")?;
        (ExecutionGroupFact::ThreadsPerGroup, group)
    };
    let lanes = AIR_EXECUTION_GROUPS
        .iter()
        .find_map(|&(name, lanes)| (name == group).then_some(lanes))?;
    Some((fact, lanes))
}

/// The execution-group fact a stage models, if this marker names one it can answer.
///
/// One statement of "does this stage lower that marker", shared by the inventory that decides what
/// to refuse and by all three decodes. See [`ExecutionGroupFact::needs_a_threadgroup`].
fn stage_execution_group_role(stage: Stage, marker: &str) -> Option<(ExecutionGroupFact, u32)> {
    execution_group_role(marker).filter(|(fact, _)| {
        matches!(stage, Stage::Kernel | Stage::Object | Stage::Mesh) || !fact.needs_a_threadgroup()
    })
}

/// The fragment return-member roles the emitter lowers.
///
/// A member carrying any other role is reported through `FragMeta::unmodelled_output_members` and
/// rejected at emission rather than skipped. AIR states an output because the shader writes it, so
/// a member nothing knows how to write is a value that silently disappears — the shape that let
/// `[[sample_mask]]` be dropped without a diagnostic for as long as it was unrecognised.
pub(crate) const FRAGMENT_OUTPUT_ROLES: &[&str] = &[
    "render_target",
    "depth",
    "stencil",
    "sample_mask",
    "imageblock_data",
];

/// What a vertex return member is once its role marker is read. The `Varying` case still needs its
/// location, which the decode assigns positionally, so this names the arm rather than the value.
#[derive(Clone, Copy, PartialEq, Eq)]
enum VertexOutputKind {
    Position,
    PointSize,
    ClipDistance,
    ViewportArrayIndex,
    RenderTargetArrayIndex,
    Varying,
}

impl VertexOutputKind {
    /// Whether this member is a SPIR-V builtin rather than a user varying.
    ///
    /// The two answer the gating question differently, exactly as they do on an entry parameter
    /// list (see [`AIR_SYSTEM_VALUE_ROLES`]): a varying keeps its slot whatever its predicate
    /// says, because the stage consuming the same struct keeps it too, while a builtin the shader
    /// cannot have written is absent. `Position` is not a gated case — a vertex shader without a
    /// clip position is not a vertex shader, and AIR does not gate it.
    fn is_system_value(self) -> bool {
        !matches!(self, Self::Varying | Self::Position)
    }
}

/// The vertex return-member roles the emitter lowers, and what each one lowers to. See
/// [`FRAGMENT_OUTPUT_ROLES`] for the fragment side of the same contract.
///
/// One table, not a role list beside a `match`: this is BOTH the set an `air.function_constant`
/// wrapper is resolved against and the mapping the decode applies, so a role cannot be lowered
/// without being promoted past the wrapper, or promoted without being lowered. Those were two
/// spellings of one fact, and they disagreed — the wrapper list named no output role at all, so a
/// gated `air.vertex_output` was a varying to the fragment decode reading the same declaration and
/// nothing to this one, which renumbered every later varying in the struct on one side only.
const VERTEX_OUTPUT_ROLES: &[(&str, VertexOutputKind)] = &[
    ("clip_distance", VertexOutputKind::ClipDistance),
    ("point_size", VertexOutputKind::PointSize),
    ("position", VertexOutputKind::Position),
    (
        "render_target_array_index",
        VertexOutputKind::RenderTargetArrayIndex,
    ),
    ("vertex_output", VertexOutputKind::Varying),
    ("viewport_array_index", VertexOutputKind::ViewportArrayIndex),
];

/// The roles the VERTEX entry-parameter decode reads past an `air.function_constant` wrapper: the
/// shared resource band, plus the two stage-input roles this stage owns.
///
/// A stage input is not a resource, but promoting it is the same argument the resource band already
/// makes. SPIR-V specialization may enable the arm that reads it, and an argument that never became
/// an input variable reads `OpUndef` instead of the attribute the application bound and reflection
/// asked for — silently, in a module that validates. It is also what the fragment decode answers for
/// the mirror-image `air.fragment_input`, and one declaration must not be classified two ways.
/// All 573 gated `air.vertex_input` arguments in 89 of 14579 local corpus sources carry a real
/// `air.location_index`, and none collides with another attribute in its own module.
///
/// Stated as one promotion set rather than as patches applied to the resource classifier's answer
/// afterwards, which is how `patch_input` came to be corrected at the call site while `vertex_input`
/// was not corrected at all.
fn vertex_input_gated_role(strs: &[String]) -> Option<&str> {
    gated_role(strs, |role| {
        FC_PROMOTED_RESOURCE_ROLES.contains(&role)
            || air_role_is_system_value(role)
            || matches!(role, "patch_input" | "vertex_input")
    })
}

/// The [`VertexOutputKind`] a vertex return-member role marker names, if the emitter lowers it.
fn vertex_output_kind(role: &str) -> Option<VertexOutputKind> {
    VERTEX_OUTPUT_ROLES
        .iter()
        .find(|(name, _)| *name == role)
        .map(|(_, kind)| *kind)
}

#[cfg(test)]
mod tests;
