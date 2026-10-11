//! Whether a manifest describes the Metal ABI of its AIR.
//!
//! Every rule here compares an authored resource against an [`AirInterface`] fact: which
//! argument-table slots the entry binds, the argument-buffer members at each byte offset, the
//! declared access, the stage interface, and the texture shapes the AIR's operations spell. A
//! slot the pipeline requires (a live argument AIR does not mark unused) must be authored; an
//! authored slot must be one some declaration of the entry binds, a function-constant alternative
//! included.

use super::interface::{
    declared_writable, AirInterface, Member, NumShape, Param, Root, TextureUse, VertexIdUse,
};
use crate::case::{
    AccelerationStructureKind, AuthoredCase, OutputSelection, ResourceRole, Stage, TextureFormat,
    TextureType,
};
use metal2vulkan_air::abi::{
    AccelerationStructureKind as AbiAccelerationStructureKind, BufferArg, OutputRole, ParamRole,
    TextureArg,
};
use std::collections::{BTreeMap, BTreeSet, HashSet};

/// Bytes a texture handle occupies in a table or an argument buffer.
const HANDLE_BYTES: u64 = 8;

pub(super) fn validate(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    validate_buffers(case, interface, errors);
    validate_argument_buffers(case, interface, errors);
    validate_threadgroup_memory(case, interface, errors);
    validate_stage_inputs(case, interface, errors);
    validate_objects(case, interface, errors);
    validate_textures(case, interface, errors);
    validate_samplers(case, interface, errors);
    validate_vertex_inputs(case, interface, errors);
    validate_out_of_bounds(case, interface, errors);
    super::grid_reach::validate(case, interface, errors);
    if case.stage == Stage::Fragment {
        validate_fragment_outputs(case, interface, errors);
    }
}

/// Each out-of-bounds target agrees with its AIR argument:
///
/// - a buffer target's form with its buffer: a `uniform_buffer` is a `constant` buffer, every
///   other buffer form a `device` one, and neither is an argument buffer (whose owner bytes the
///   executors rewrite); a `buffer_atomic` target is one an atomic operation accesses, and a
///   target of any other form one none does (its reads are loads, judged by the load rule, and
///   the executors take an atomic's medium from the form);
/// - an argument-buffer buffer target is one the entry loads and stores through: a pointer it
///   loads from the argument buffer's member;
/// - an argument buffer's texture-array target is the first element of a texture member of at
///   least two elements, whose handles the entry loads at a dynamic offset (an index into the
///   array);
/// - a thread-memory target is the entry's one `alloca`, an object holding an array;
/// - an attribute buffer target's attribute is one the AIR types (the authored format has the lanes
///   the shader reads, so no fetch fills a component and an out-of-bounds read is the format's
///   zero), and a control point's reach covers `patches × N` elements;
/// - an index buffer target's vertex function reads its `[[vertex_id]]` only clamped to a
///   constant ([`VertexIdUse`]). The checker does not compare the constant with the buffer the
///   clamped id indexes (no fact ties the value to an access's bound): that the bound stays inside
///   it is the authoring contract, as a texture array's reach is.
fn validate_out_of_bounds(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    use crate::case::out_of_bounds::OutOfBoundsResource;
    let Some(declaration) = &case.out_of_bounds else {
        return;
    };
    for target in &declaration.targets {
        match target.resource {
            OutOfBoundsResource::Buffer { binding } => {
                validate_out_of_bounds_buffer(binding, target.form, interface, errors)
            }
            OutOfBoundsResource::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
            } => {
                let label = format!(
                    "out_of_bounds target argument-buffer buffer {buffer_binding}+{field_offset}"
                );
                let Some((param, _)) = interface.member_at(buffer_binding, field_offset) else {
                    errors.push(format!("{label}: the AIR entry binds no member there"));
                    continue;
                };
                let root = Root::Loaded {
                    param: param.binding.param,
                    offset: Some(u64::from(field_offset)),
                };
                if !interface.accesses(root) {
                    errors.push(format!(
                        "{label}: the AIR entry loads and stores nothing through the pointer it \
                         holds"
                    ));
                }
            }
            OutOfBoundsResource::ArgumentBufferTextureArray {
                buffer_binding,
                field_offset,
            } => {
                let label = format!(
                    "out_of_bounds target argument-buffer texture array \
                     {buffer_binding}+{field_offset}"
                );
                let Some((param, member)) = interface.member_at(buffer_binding, field_offset)
                else {
                    errors.push(format!("{label}: the AIR entry binds no member there"));
                    continue;
                };
                if !matches!(member.role, ParamRole::Texture(_))
                    || member.element != 0
                    || member.elements < 2
                {
                    errors.push(format!(
                        "{label}: the AIR member there is not the first element of a texture \
                         array of at least two elements"
                    ));
                }
                let dynamic = Root::Loaded {
                    param: param.binding.param,
                    offset: None,
                };
                if interface.texture_uses(dynamic).next().is_none() {
                    errors.push(format!(
                        "{label}: no AIR texture operation reaches a handle the entry loads at a \
                         dynamic offset of its argument buffer"
                    ));
                }
            }
            OutOfBoundsResource::ThreadMemory => {
                if !matches!(
                    interface.thread_objects.as_slice(),
                    [object] if object.holds_array
                ) {
                    errors.push(format!(
                        "out_of_bounds target thread memory: the AIR entry must allocate exactly \
                         one thread-local object, holding an array, but it allocates {:?}",
                        interface.thread_objects
                    ));
                }
            }
            OutOfBoundsResource::AttributeBuffer { location } => {
                validate_out_of_bounds_attribute_type(case, interface, location, errors);
                validate_out_of_bounds_control_point(case, interface, location, target, errors)
            }
            // An index read past the buffer is the oracle's shadow, a vertex id far past every
            // buffer the case binds (`0xE7E7E7E7`, or `0xE7E7`): the entry may use the id only
            // clamped to a constant, so no access of Metal's carries it.
            OutOfBoundsResource::IndexBuffer => {
                if interface.vertex_id == VertexIdUse::Unclamped {
                    errors.push(
                        "out_of_bounds target index buffer: the AIR entry reads its \
                         [[vertex_id]] other than as an operand of an unsigned min with a \
                         constant, so an index read past the buffer (the shadow's vertex id) \
                         could reach an access"
                            .to_string(),
                    );
                }
            }
            OutOfBoundsResource::ThreadgroupMemory { .. }
            | OutOfBoundsResource::Texture { .. }
            | OutOfBoundsResource::TextureArray { .. } => {}
        }
    }
}

/// A vertex input or kernel stage input target's attribute is one the AIR types: an out-of-bounds
/// read of it is judged by the format's zero ([`crate::executor_contract::out_of_bounds`]), which
/// holds because the authored format has the lanes the shader reads (`attribute_shape`), so no
/// fetch fills a component with Vulkan's `(0, 0, 1)`. A tessellation input's type is the reflected
/// one (`check::validate_tessellation_attributes`).
fn validate_out_of_bounds_attribute_type(
    case: &AuthoredCase,
    interface: &AirInterface,
    location: u32,
    errors: &mut Vec<String>,
) {
    let vertex = case
        .vertex_inputs
        .iter()
        .any(|input| input.location == location);
    let stage = case
        .kernel_stage_inputs
        .iter()
        .any(|input| input.location == location);
    if !vertex && !stage {
        return;
    }
    let typed = interface.params.iter().any(|param| {
        let role = match param.binding.role {
            ParamRole::VertexInput { .. } => vertex,
            ParamRole::StageIn { .. } => stage,
            _ => false,
        };
        role && param.slots().map(|(slot, _)| slot) == Some(location) && param.value.is_some()
    });
    if !typed {
        errors.push(format!(
            "out_of_bounds target attribute buffer {location}: the AIR types no value of the \
             attribute, so its format's lanes cannot be checked against the shader's"
        ));
    }
}

/// A control-point buffer target: a tessellation draw fetches `patches × N` elements of it, `N`
/// the AIR's control points per patch (`[[patch(_, N)]]`; control point `k` of patch `p` is
/// element `p · N + k`, the executors' shared draw), and its reach past the buffer covers them. The
/// literal check covers every other attribute buffer target's reach.
fn validate_out_of_bounds_control_point(
    case: &AuthoredCase,
    interface: &AirInterface,
    location: u32,
    target: &crate::case::out_of_bounds::OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some(draw) = &case.tessellation else {
        return;
    };
    let Some(input) = draw
        .control_points
        .iter()
        .find(|input| input.location == location)
    else {
        return;
    };
    let label = format!("out_of_bounds target attribute buffer {location}");
    let control_points = match crate::executor_contract::tessellation::control_points(interface) {
        Ok(control_points) => control_points,
        Err(error) => return errors.push(format!("{label}: {error}")),
    };
    let Ok(bytes) =
        base64::Engine::decode(&base64::engine::general_purpose::STANDARD, &input.bytes_b64)
    else {
        return;
    };
    let elements = u32::try_from(draw.factors.len())
        .ok()
        .and_then(|patches| patches.checked_mul(control_points))
        .unwrap_or(u32::MAX);
    crate::case::out_of_bounds::validate_fetched_reach(
        &label,
        bytes.len() as u64,
        crate::case::out_of_bounds::fetch_end(elements, input),
        target,
        errors,
    );
}

/// The out-of-bounds buffer target at `binding` agrees with its AIR buffer argument
/// ([`validate_out_of_bounds`]).
fn validate_out_of_bounds_buffer(
    binding: u32,
    form: crate::case::out_of_bounds::ContainmentForm,
    interface: &AirInterface,
    errors: &mut Vec<String>,
) {
    use crate::case::out_of_bounds::ContainmentForm;
    use metal2vulkan_air::ir::types::AddrSpace;
    let spaces = interface
        .params
        .iter()
        .filter(|param| param.slots().map(|(slot, _)| slot) == Some(binding))
        .filter_map(|param| match &param.binding.role {
            ParamRole::Buffer(buffer) => Some(Some(buffer.space)),
            ParamRole::ArgumentBuffer(_) => Some(None),
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let wanted = match form {
        ContainmentForm::UniformBuffer => AddrSpace::Constant,
        _ => AddrSpace::Device,
    };
    if spaces != BTreeSet::from([Some(wanted)]) {
        errors.push(format!(
            "out_of_bounds target buffer {binding}: form {form} needs a {wanted:?} buffer \
             argument, but the AIR entry binds {spaces:?} there"
        ));
        return;
    }
    let atomic = interface
        .params
        .iter()
        .filter(|param| param.slots().map(|(slot, _)| slot) == Some(binding))
        .any(|param| interface.atomic_through(param.binding.param));
    match (form, atomic) {
        (ContainmentForm::BufferAtomic, false) => errors.push(format!(
            "out_of_bounds target buffer {binding}: form buffer_atomic needs an atomic \
             operation on the buffer, but the AIR entry performs none"
        )),
        (ContainmentForm::BufferAtomic, true) | (_, false) => {}
        (form, true) => errors.push(format!(
            "out_of_bounds target buffer {binding}: an atomic operation accesses it, so its \
             form is buffer_atomic, not {form}"
        )),
    }
}

/// The slots (or argument-buffer coordinates) one family of AIR arguments occupies.
struct Occupied<T> {
    /// Those the pipeline requires: the argument is live and not optional.
    required: BTreeSet<T>,
    /// Those some declaration binds, live or not.
    declared: BTreeSet<T>,
}

impl<T: Copy + Ord + std::fmt::Debug> Occupied<T> {
    fn new() -> Self {
        Self {
            required: BTreeSet::new(),
            declared: BTreeSet::new(),
        }
    }

    fn insert(&mut self, key: T, required: bool) {
        self.declared.insert(key);
        if required {
            self.required.insert(key);
        }
    }

    /// Report every required key the manifest leaves out and every authored key nothing
    /// declares.
    fn report(&self, kind: &str, authored: &BTreeSet<T>, errors: &mut Vec<String>) {
        for missing in self.required.difference(authored) {
            errors.push(format!(
                "AIR requires {kind} {missing:?}, but the manifest does not declare it"
            ));
        }
        for extra in authored.difference(&self.declared) {
            errors.push(format!(
                "manifest declares {kind} {extra:?}, but the AIR entry binds none there"
            ));
        }
    }
}

/// The first slots of the parameters `select` picks.
fn slot_sets<'a>(interface: &'a AirInterface, select: impl Fn(&'a Param) -> bool) -> Occupied<u32> {
    let mut occupied = Occupied::new();
    for param in interface.params.iter().filter(|param| select(param)) {
        if let Some((first, _)) = param.slots() {
            occupied.insert(first, param.required());
        }
    }
    occupied
}

fn buffer_arg(role: &ParamRole) -> Option<&BufferArg> {
    match role {
        ParamRole::Buffer(buffer) | ParamRole::ArgumentBuffer(buffer) => Some(buffer),
        _ => None,
    }
}

fn validate_buffers(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    let occupied = slot_sets(interface, |param| buffer_arg(&param.binding.role).is_some());
    let authored = case
        .buffers
        .iter()
        .map(|buffer| buffer.binding)
        .chain(case.device_buffer_arrays.iter().map(|array| array.binding))
        .collect();
    occupied.report("buffer", &authored, errors);
    for (param, _) in interface.params_of(buffer_arg) {
        let Some((slot, count)) = param.slots() else {
            continue;
        };
        let array = case
            .device_buffer_arrays
            .iter()
            .find(|array| array.binding == slot);
        let authored = case.buffers.iter().any(|buffer| buffer.binding == slot);
        match (param.table, array) {
            // A count of 0 is a table whose length the pipeline chooses.
            (true, Some(array)) if count != 0 && array.length != count => errors.push(format!(
                "device-buffer array {slot} authors {} elements, but AIR declares {count}",
                array.length
            )),
            (true, None) if authored => errors.push(format!(
                "buffer {slot} is a device-buffer array in AIR, but the manifest does not author it as one"
            )),
            (false, Some(_)) => errors.push(format!(
                "buffer {slot} is not a device-buffer array in AIR, but the manifest authors it as one"
            )),
            _ => {}
        }
    }
    for resource in &case.device_buffer_arrays {
        if let Some(error) =
            crate::executor_contract::device_buffer_array_completeness_error(resource, "manifest")
        {
            errors.push(error);
        }
    }
}

/// Exactly the resources a coordinate-mapping call reads a rasterization-rate map from are
/// authored as that map (`rasterization_rate_map`): an input buffer, or the buffer an
/// argument-buffer member holds, at the places
/// [`RateMapPlacement::places`](crate::executor_contract::rate_map::RateMapPlacement) names. The
/// parameter data's layout is private to each implementation, so bytes captured from one are not
/// the other's map. A call the placement does not place keeps its map as authored bytes, and the
/// source carries `rasterization_rate_map_placement`, which the engine's candidate refuses.
pub(super) fn validate_rate_maps(
    case: &AuthoredCase,
    interface: &AirInterface,
    placement: &crate::executor_contract::rate_map::RateMapPlacement,
    errors: &mut Vec<String>,
) {
    use crate::executor_contract::rate_map::MapPlace;
    for buffer in &case.buffers {
        let maps = interface
            .params_of(buffer_arg)
            .filter(|(param, _)| {
                !param.table && param.slots().map(|(slot, _)| slot) == Some(buffer.binding)
            })
            .any(|(param, _)| {
                placement.places.contains(&MapPlace::Buffer {
                    param: param.binding.param,
                })
            });
        match (maps, buffer.rasterization_rate_map.is_some()) {
            (true, false) => errors.push(format!(
                "buffer {} holds the rasterization-rate map AIR maps coordinates through, but the \
                 manifest authors bytes, not rasterization_rate_map",
                buffer.binding
            )),
            (false, true) => errors.push(format!(
                "buffer {} is authored as a rasterization-rate map, but no coordinate-mapping \
                 call reads a map from offset 0 of it",
                buffer.binding
            )),
            _ => {}
        }
    }
    for member in &case.argument_buffer_buffers {
        let maps = interface
            .bound_argument_buffers()
            .into_iter()
            .filter(|(slot, _)| *slot == member.buffer_binding)
            .any(|(_, param)| {
                placement.places.contains(&MapPlace::ArgumentBufferMember {
                    param: param.binding.param,
                    offset: member.field_offset,
                })
            });
        match (maps, member.rasterization_rate_map.is_some()) {
            (true, false) => errors.push(format!(
                "argument-buffer buffer {}+{} holds the rasterization-rate map AIR maps \
                 coordinates through, but the manifest authors bytes, not rasterization_rate_map",
                member.buffer_binding, member.field_offset
            )),
            (false, true) => errors.push(format!(
                "argument-buffer buffer {}+{} is authored as a rasterization-rate map, but no \
                 coordinate-mapping call reads a map from offset 0 of it",
                member.buffer_binding, member.field_offset
            )),
            _ => {}
        }
    }
}

/// Every member of the argument buffers that bind ([`AirInterface::bound_argument_buffers`]),
/// keyed by `(buffer slot, byte offset)`, with its parameter: the one
/// [`AirInterface::member_at`] answers with.
fn argument_buffer_members(interface: &AirInterface) -> BTreeMap<(u32, u32), (&Param, Member)> {
    let mut members = BTreeMap::new();
    for (slot, param) in interface.bound_argument_buffers() {
        for member in interface.members(param) {
            members
                .entry((slot, member.offset))
                .or_insert((param, member));
        }
    }
    members
}

fn member_sets(
    members: &BTreeMap<(u32, u32), (&Param, Member)>,
    select: impl Fn(&ParamRole) -> bool,
) -> Occupied<(u32, u32)> {
    let mut occupied = Occupied::new();
    for (key, (param, member)) in members {
        if select(&member.role) {
            occupied.insert(*key, param.required());
        }
    }
    occupied
}

fn validate_argument_buffers(
    case: &AuthoredCase,
    interface: &AirInterface,
    errors: &mut Vec<String>,
) {
    let members = argument_buffer_members(interface);
    let occupied = member_sets(&members, |role| matches!(role, ParamRole::Buffer(_)));
    let authored = case
        .argument_buffer_buffers
        .iter()
        .map(|buffer| (buffer.buffer_binding, buffer.field_offset))
        .collect();
    occupied.report("argument-buffer buffer", &authored, errors);
    for resource in &case.argument_buffer_buffers {
        let key = (resource.buffer_binding, resource.field_offset);
        let Some((
            param,
            Member {
                role: ParamRole::Buffer(buffer),
                ..
            },
        )) = members.get(&key)
        else {
            continue;
        };
        if resource.role != ResourceRole::Input
            && !declared_writable(buffer.access)
            && !interface.writes(Root::Loaded {
                param: param.binding.param,
                offset: Some(u64::from(resource.field_offset)),
            })
        {
            errors.push(format!(
                "argument-buffer buffer {}+{} is authored as {:?}, but AIR neither declares nor performs a write through it",
                resource.buffer_binding, resource.field_offset, resource.role
            ));
        }
    }

    let occupied = member_sets(&members, |role| matches!(role, ParamRole::Texture(_)));
    let authored = case
        .argument_buffer_textures
        .iter()
        .map(|texture| (texture.buffer_binding, texture.field_offset))
        .collect();
    occupied.report("argument-buffer texture", &authored, errors);
    for resource in &case.argument_buffer_textures {
        let key = (resource.buffer_binding, resource.field_offset);
        let Some((
            param,
            Member {
                role: ParamRole::Texture(texture),
                ..
            },
        )) = members.get(&key)
        else {
            continue;
        };
        let label = format!(
            "argument-buffer texture {}+{}",
            resource.buffer_binding, resource.field_offset
        );
        if resource.role != ResourceRole::Input && !declared_writable(texture.access) {
            errors.push(format!(
                "{label} is authored as {:?}, but AIR declares it read-only",
                resource.role
            ));
        }
        // As for a table's elements, a use at a dynamic offset into the buffer (an indexed
        // `array<texture…, N>` member) may reach this member, so it is checked here too.
        let uses = interface
            .loaded_uses(param.binding.param, u64::from(resource.field_offset))
            .collect::<Vec<_>>();
        validate_texture_uses(
            &label,
            resource.texture_type,
            resource.format,
            &uses,
            errors,
        );
    }

    for (kind, select, authored) in [
        (
            "pipeline-state field",
            (|role: &ParamRole| {
                matches!(
                    role,
                    ParamRole::ComputePipelineState(_) | ParamRole::RenderPipelineState(_)
                )
            }) as fn(&ParamRole) -> bool,
            case.pipeline_states
                .iter()
                .map(|resource| (resource.buffer_binding, resource.field_offset))
                .collect::<BTreeSet<_>>(),
        ),
        (
            "indirect command-buffer field",
            |role: &ParamRole| matches!(role, ParamRole::CommandBuffer(_)),
            case.command_buffers
                .iter()
                .map(|resource| (resource.buffer_binding, resource.field_offset))
                .collect(),
        ),
        // A sampler state the argument buffer holds: Metal reads whatever its field holds, so every
        // member a live argument buffer declares is authored.
        (
            "argument-buffer sampler",
            |role: &ParamRole| matches!(role, ParamRole::Sampler { .. }),
            case.argument_buffer_samplers
                .iter()
                .map(|resource| (resource.buffer_binding, resource.field_offset))
                .collect(),
        ),
    ] {
        let occupied = member_sets(&members, select);
        occupied.report(kind, &authored, errors);
    }
    for table in &case.argument_buffer_intersection_function_tables {
        let key = (table.buffer_binding, table.field_offset);
        if !matches!(
            members.get(&key),
            Some((
                _,
                Member {
                    role: ParamRole::IntersectionTable(_),
                    ..
                }
            ))
        ) {
            errors.push(format!(
                "manifest declares argument-buffer intersection-function table at buffer {} offset {}, but AIR has no such member",
                table.buffer_binding, table.field_offset
            ));
        }
    }
}

fn validate_threadgroup_memory(
    case: &AuthoredCase,
    interface: &AirInterface,
    errors: &mut Vec<String>,
) {
    let occupied = slot_sets(interface, |param| {
        matches!(param.binding.role, ParamRole::Threadgroup(_))
    });
    let authored = case
        .threadgroup_memory
        .iter()
        .map(|resource| resource.binding)
        .collect();
    occupied.report("threadgroup memory", &authored, errors);
    for resource in &case.threadgroup_memory {
        if let Some(error) =
            crate::executor_contract::threadgroup_memory_error(resource, "manifest")
        {
            errors.push(error);
        }
        let minimum = interface
            .params_of(|role| match role {
                ParamRole::Threadgroup(buffer) => Some(buffer),
                _ => None,
            })
            .filter(|(param, _)| param.slots().map(|(slot, _)| slot) == Some(resource.binding))
            .filter_map(|(param, buffer)| threadgroup_size(param, buffer))
            .max();
        if let Some(minimum) = minimum.filter(|minimum| resource.length < *minimum) {
            errors.push(format!(
                "threadgroup-memory binding {} length {} is smaller than the AIR element size {minimum}",
                resource.binding, resource.length
            ));
        }
    }
}

/// The byte size a threadgroup argument's pointee declares: `air.arg_type_size`, else
/// `air.buffer_size`.
pub(super) fn threadgroup_size(param: &Param, buffer: &BufferArg) -> Option<u32> {
    param.binding.ty.size.or(buffer.buffer_size)
}

fn validate_stage_inputs(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    let occupied = slot_sets(interface, |param| {
        matches!(param.binding.role, ParamRole::StageIn { .. })
    });
    let authored = case
        .kernel_stage_inputs
        .iter()
        .map(|resource| resource.location)
        .collect();
    occupied.report("kernel stage input", &authored, errors);
    for resource in &case.kernel_stage_inputs {
        let (scalar, lanes) = resource.format.scalar_lanes();
        attribute_shape(
            &format!("kernel stage input {}", resource.location),
            interface,
            |role| matches!(role, ParamRole::StageIn { .. }),
            resource.location,
            NumShape::of_scalar(scalar, lanes),
            resource.format,
            errors,
        );
    }
}

fn attribute_shape(
    label: &str,
    interface: &AirInterface,
    select: impl Fn(&ParamRole) -> bool,
    location: u32,
    authored: NumShape,
    format: crate::case::AttributeFormat,
    errors: &mut Vec<String>,
) {
    let shapes = interface
        .params
        .iter()
        .filter(|param| select(&param.binding.role))
        .filter(|param| param.slots().map(|(slot, _)| slot) == Some(location))
        .filter_map(|param| param.value)
        .collect::<Vec<_>>();
    if !shapes.is_empty() && !shapes.iter().any(|shape| shape.agrees(authored)) {
        errors.push(format!(
            "{label} format {format:?} does not match the AIR value type {shapes:?}"
        ));
    }
}

fn validate_vertex_inputs(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    let occupied = slot_sets(interface, |param| {
        matches!(param.binding.role, ParamRole::VertexInput { .. })
    });
    let authored = case
        .vertex_inputs
        .iter()
        .map(|input| input.location)
        .collect();
    occupied.report("vertex input", &authored, errors);
    for input in &case.vertex_inputs {
        let (scalar, lanes) = input.format.scalar_lanes();
        attribute_shape(
            &format!("vertex input {}", input.location),
            interface,
            |role| matches!(role, ParamRole::VertexInput { .. }),
            input.location,
            NumShape::of_scalar(scalar, lanes),
            input.format,
            errors,
        );
    }
}

/// Acceleration structures and function tables: a slot and, for an acceleration structure, its
/// kind.
fn validate_objects(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    let occupied = slot_sets(interface, |param| {
        matches!(param.binding.role, ParamRole::AccelerationStructure { .. })
    });
    let authored = case
        .acceleration_structures
        .iter()
        .map(|resource| resource.binding)
        .collect();
    occupied.report("acceleration structure", &authored, errors);
    for resource in &case.acceleration_structures {
        let expected = match resource.kind {
            AccelerationStructureKind::Instance => AbiAccelerationStructureKind::Instance,
            AccelerationStructureKind::Primitive => AbiAccelerationStructureKind::Primitive,
        };
        let kinds = interface
            .params
            .iter()
            .filter_map(|param| match &param.binding.role {
                ParamRole::AccelerationStructure { kind, .. }
                    if param.slots().map(|(slot, _)| slot) == Some(resource.binding) =>
                {
                    Some(*kind)
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if !kinds.is_empty() && !kinds.contains(&expected) {
            errors.push(format!(
                "acceleration-structure binding {} kind {:?} does not match AIR {kinds:?}",
                resource.binding, resource.kind
            ));
        }
    }
    for (label, select, authored) in [
        (
            "visible-function table",
            (|role: &ParamRole| matches!(role, ParamRole::FunctionTable(_)))
                as fn(&ParamRole) -> bool,
            case.visible_function_tables
                .iter()
                .map(|table| table.binding)
                .collect::<BTreeSet<_>>(),
        ),
        (
            "intersection-function table",
            |role: &ParamRole| matches!(role, ParamRole::IntersectionTable(_)),
            case.intersection_function_tables
                .iter()
                .map(|table| table.binding)
                .collect(),
        ),
    ] {
        let occupied = slot_sets(interface, |param| select(&param.binding.role));
        occupied.report(label, &authored, errors);
    }
}

fn texture_arg(role: &ParamRole) -> Option<&TextureArg> {
    match role {
        ParamRole::Texture(texture) => Some(texture),
        _ => None,
    }
}

/// A texture argument is an array when it spans several slots or is passed as a table.
fn is_texture_array(param: &Param) -> bool {
    param.table || param.slots().is_some_and(|(_, count)| count > 1)
}

pub(super) fn validate_textures(
    case: &AuthoredCase,
    interface: &AirInterface,
    errors: &mut Vec<String>,
) {
    let singles = slot_sets(interface, |param| {
        texture_arg(&param.binding.role).is_some() && !is_texture_array(param)
    });
    let arrays = slot_sets(interface, |param| {
        texture_arg(&param.binding.role).is_some() && is_texture_array(param)
    });
    // A texture array that overrides the texture at its base slot is the function-constant
    // alternative of that texture: either one satisfies the slot.
    let overridden = case
        .texture_arrays
        .iter()
        .filter(|array| array.overrides_texture_at_base)
        .map(|array| array.binding)
        .collect::<BTreeSet<_>>();
    let authored = case
        .textures
        .iter()
        .map(|texture| texture.binding)
        .collect::<BTreeSet<_>>();
    let authored_arrays = case
        .texture_arrays
        .iter()
        .map(|array| array.binding)
        .collect::<BTreeSet<_>>();
    let singles = Occupied {
        required: &singles.required - &authored_arrays,
        declared: &singles.declared | &overridden,
    };
    let arrays = Occupied {
        required: &arrays.required - &(&authored - &overridden),
        declared: arrays.declared,
    };
    singles.report("texture", &authored, errors);
    arrays.report("texture array", &authored_arrays, errors);

    for resource in &case.textures {
        let alternatives = interface
            .params_of(texture_arg)
            .filter(|(param, _)| {
                !is_texture_array(param)
                    && param.slots().map(|(slot, _)| slot) == Some(resource.binding)
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            continue;
        }
        let label = format!("texture binding {}", resource.binding);
        if resource.role != ResourceRole::Input
            && !alternatives
                .iter()
                .any(|(_, texture)| declared_writable(texture.access))
        {
            errors.push(format!(
                "{label} is authored as {:?}, but AIR declares it read-only",
                resource.role
            ));
        }
        // A sparse texture the function writes is admitted as an input: the executors make it
        // writable (the oracle's shader-write usage, the candidate's storage image) and read
        // nothing of it back, so a write is observed only by the function's own later reads, which
        // Metal answers (`case::sparse`).
        validate_alternatives(
            &label,
            resource.texture_type,
            resource.format,
            alternatives
                .iter()
                .map(|(param, _)| {
                    interface
                        .texture_uses(Root::Param(param.binding.param))
                        .collect()
                })
                .collect(),
            errors,
        );
    }

    for resource in &case.texture_arrays {
        let alternatives = interface
            .params_of(texture_arg)
            .filter(|(param, _)| {
                is_texture_array(param)
                    && param.slots().map(|(slot, _)| slot) == Some(resource.binding)
            })
            .collect::<Vec<_>>();
        if alternatives.is_empty() {
            continue;
        }
        let label = format!("texture-array binding {}", resource.binding);
        // A count of 0 is a table whose length the pipeline chooses.
        let lengths = alternatives
            .iter()
            .filter_map(|(param, _)| param.slots().map(|(_, count)| count))
            .collect::<Vec<_>>();
        if !lengths
            .iter()
            .any(|length| *length == 0 || *length as usize == resource.elements.len())
        {
            errors.push(format!(
                "{label} has {} authored elements, but AIR alternatives declare {lengths:?}",
                resource.elements.len()
            ));
        }
        if resource.role != ResourceRole::Input
            && !alternatives
                .iter()
                .any(|(_, texture)| declared_writable(texture.access))
        {
            errors.push(format!(
                "{label} is authored as {:?}, but AIR declares it read-only",
                resource.role
            ));
        }
        validate_alternatives(
            &label,
            resource.texture_type,
            resource.format,
            alternatives
                .iter()
                .map(|(param, _)| {
                    let index = param.binding.param;
                    if param.table {
                        (0..resource.elements.len() as u32)
                            .flat_map(|element| {
                                interface.loaded_uses(index, u64::from(element) * HANDLE_BYTES)
                            })
                            .collect()
                    } else {
                        interface.texture_uses(Root::Param(index)).collect()
                    }
                })
                .collect(),
            errors,
        );
    }
}

/// Check an authored texture type and format against every use of each alternative; the
/// manifest is valid when some alternative's uses all accept it.
pub(super) fn validate_alternatives(
    label: &str,
    texture_type: TextureType,
    format: TextureFormat,
    alternatives: Vec<Vec<TextureUse>>,
    errors: &mut Vec<String>,
) {
    let attempts = alternatives
        .iter()
        .map(|uses| {
            let mut attempt = Vec::new();
            validate_texture_uses(label, texture_type, format, uses, &mut attempt);
            attempt
        })
        .collect::<Vec<_>>();
    if attempts.iter().any(Vec::is_empty) {
        return;
    }
    if let Some(first) = attempts.into_iter().next() {
        errors.extend(first);
    }
}

pub(super) fn validate_texture_uses(
    label: &str,
    texture_type: TextureType,
    format: TextureFormat,
    uses: &[TextureUse],
    errors: &mut Vec<String>,
) {
    let mut types = Vec::new();
    for usage in uses {
        let expected = usage.texture_type();
        if !types.contains(&expected) {
            types.push(expected);
        }
    }
    for expected in &types {
        match expected {
            Some(expected) if *expected != texture_type => errors.push(format!(
                "{label} requires texture type {expected:?}, got {texture_type:?}"
            )),
            None => errors.push(format!(
                "{label} is used with an image shape the manifest has no texture type for"
            )),
            _ => {}
        }
    }
    if let Some(usage) = uses.iter().find(|usage| !usage.accepts_format(format)) {
        errors.push(format!(
            "{label} format {format:?} does not match the {:?} texels AIR reads or writes",
            usage.texel
        ));
    }
}

fn validate_samplers(case: &AuthoredCase, interface: &AirInterface, errors: &mut Vec<String>) {
    let occupied = slot_sets(interface, |param| {
        matches!(param.binding.role, ParamRole::Sampler { .. })
    });
    let authored = case
        .samplers
        .iter()
        .map(|sampler| sampler.binding)
        .collect();
    occupied.report("sampler", &authored, errors);
}

/// A fragment function's color outputs and framebuffer-fetch inputs, and its depth and stencil
/// outputs.
fn validate_fragment_outputs(
    case: &AuthoredCase,
    interface: &AirInterface,
    errors: &mut Vec<String>,
) {
    let mut occupied = Occupied::new();
    let mut shapes = BTreeMap::<u32, Vec<NumShape>>::new();
    for output in &interface.outputs {
        let OutputRole::RenderTarget { attachment, .. } = output.binding.role else {
            continue;
        };
        let Some(attachment) = super::interface::constant_int(attachment) else {
            continue;
        };
        occupied.insert(attachment, output.live);
        shapes.entry(attachment).or_default().extend(output.value);
    }
    for param in &interface.params {
        let ParamRole::ColorInput { index, .. } = param.binding.role else {
            continue;
        };
        let Some(index) = super::interface::constant_int(index) else {
            continue;
        };
        occupied.insert(index, param.required());
        shapes.entry(index).or_default().extend(param.value);
    }
    let authored = case
        .render_targets
        .iter()
        .map(|target| target.index)
        .collect();
    occupied.report("render target", &authored, errors);
    for target in &case.render_targets {
        for shape in shapes.get(&target.index).into_iter().flatten() {
            let class_matches = match target.format.component_class() {
                crate::air::reflect::TextureComponent::Float => shape.float,
                _ => !shape.float,
            };
            if target.format.is_depth() || !class_matches {
                errors.push(format!(
                    "render target {} format {:?} is incompatible with the AIR value type {shape:?}",
                    target.index, target.format
                ));
            }
        }
    }
    let live = |select: fn(&OutputRole) -> bool| {
        interface
            .outputs
            .iter()
            .any(|output| output.live && select(&output.binding.role))
    };
    let depth = live(|role| matches!(role, OutputRole::Depth(_)));
    let stencil = live(|role| matches!(role, OutputRole::Stencil));
    let authored_depth = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_depth_b64.is_some());
    let authored_stencil = case
        .depth_stencil
        .as_ref()
        .is_some_and(|attachment| attachment.initial_stencil_b64.is_some());
    match (depth, authored_depth) {
        (false, true) => {
            errors.push("manifest has a depth aspect the fragment does not write".into())
        }
        (true, false) => {
            errors.push("the fragment writes depth, which requires an authored depth aspect".into())
        }
        _ => {}
    }
    match (stencil, authored_stencil) {
        (false, true) => {
            errors.push("manifest has a stencil aspect the fragment does not write".into())
        }
        (true, false) => errors
            .push("the fragment writes stencil, which requires an authored stencil aspect".into()),
        _ => {}
    }
}

/// Whether the case's selected output is something the shader can write, by the AIR's declared
/// access and the writes its IR performs. `imageblock` answers the render-target and
/// fragment-imageblock selections the tile and imageblock families still take from product
/// reflection.
pub(super) fn selected_output_is_writable(
    case: &AuthoredCase,
    interface: &AirInterface,
    imageblock: &ImageblockOutputs,
) -> bool {
    let buffer_writable = |binding: u32| {
        interface.params_of(buffer_arg).any(|(param, buffer)| {
            param.slots().map(|(slot, _)| slot) == Some(binding)
                && (declared_writable(buffer.access)
                    || interface.writes_through(param.binding.param))
        })
    };
    let texture_writable = |binding: u32| {
        interface.params_of(texture_arg).any(|(param, texture)| {
            param.slots().is_some_and(|(slot, count)| {
                slot <= binding && (count == 0 || binding - slot < count.max(1))
            }) && declared_writable(texture.access)
        })
    };
    let members = argument_buffer_members(interface);
    let live_output = |select: fn(&OutputRole) -> bool| {
        interface
            .outputs
            .iter()
            .any(|output| output.live && select(&output.binding.role))
    };
    match &case.output {
        OutputSelection::None => {
            let writes = interface.params.iter().any(|param| {
                param.live
                    && match &param.binding.role {
                        ParamRole::Buffer(buffer) | ParamRole::ArgumentBuffer(buffer) => {
                            declared_writable(buffer.access)
                                || interface.writes_through(param.binding.param)
                        }
                        ParamRole::Texture(texture) => declared_writable(texture.access),
                        _ => false,
                    }
            });
            !writes && !interface.outputs.iter().any(|output| output.live) && !imageblock.writes
        }
        OutputSelection::Buffer { binding, .. }
        | OutputSelection::DeviceBufferArrayElement { binding, .. } => buffer_writable(*binding),
        OutputSelection::ArgumentBufferBuffer {
            buffer_binding,
            field_offset,
            ..
        } => match members.get(&(*buffer_binding, *field_offset)) {
            Some((
                param,
                Member {
                    role: ParamRole::Buffer(buffer),
                    ..
                },
            )) => {
                declared_writable(buffer.access)
                    || interface.writes(Root::Loaded {
                        param: param.binding.param,
                        offset: Some(u64::from(*field_offset)),
                    })
            }
            _ => false,
        },
        OutputSelection::Texture { binding, .. }
        | OutputSelection::TextureArrayElement { binding, .. } => texture_writable(*binding),
        OutputSelection::ArgumentBufferTexture {
            buffer_binding,
            field_offset,
            ..
        } => matches!(
            members.get(&(*buffer_binding, *field_offset)),
            Some((_, Member {
                role: ParamRole::Texture(texture),
                ..
            })) if declared_writable(texture.access)
        ),
        OutputSelection::RenderTarget { index, .. } => {
            interface.outputs.iter().any(|output| {
                output.live
                    && matches!(output.binding.role, OutputRole::RenderTarget { attachment, .. }
                        if super::interface::constant_int(attachment) == Some(*index))
            }) || imageblock.attachments.contains(index)
                || (case.stage == Stage::Vertex && *index == 0 && case.vertex_observation.is_some())
        }
        OutputSelection::Depth { .. } => live_output(|role| matches!(role, OutputRole::Depth(_))),
        OutputSelection::Stencil { .. } => live_output(|role| matches!(role, OutputRole::Stencil)),
        OutputSelection::FragmentImageblock { semantic, .. } => {
            imageblock.fragment_members.contains(semantic)
        }
    }
}

/// The writable outputs of the imageblock families, which the checker still reads from product
/// reflection: implicit imageblock attachments, fragment imageblock members.
#[derive(Default)]
pub(super) struct ImageblockOutputs {
    pub writes: bool,
    pub attachments: HashSet<u32>,
    pub fragment_members: HashSet<String>,
}
