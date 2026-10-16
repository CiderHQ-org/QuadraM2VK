//! Out-of-bounds cases: a case that accesses resources past the range its binding gives them, and
//! declares what a run must show.
//!
//! A case declares each resource it accesses out of bounds (a [`OutOfBoundsTarget`]) with the
//! containment form it exercises ([`ContainmentForm`]), and the output bytes that hold results of
//! out-of-bounds reads ([`OutOfBoundsRead`]).
//!
//! - **Where Metal defines the result** (a texture or a texture buffer: reads return zero with the
//!   format's default components, writes are ignored; MSL 4.1 §2.9.1 and §6.12), the Metal oracle
//!   runs every access, and the candidate compares with Metal as any case does. A texture buffer
//!   is host-visible buffer memory in the candidate, which allocates it inside a guard of its
//!   `reach_after` and requires every guard byte unchanged; an image's memory is opaque and has
//!   no guard.
//! - **Where Metal does not** (a buffer argument, a buffer reached through a pointer loaded from
//!   an argument buffer, threadgroup memory), the Metal oracle never accesses memory outside an
//!   allocation: it allocates each such target with a *shadow* of `reach_before` bytes before it
//!   and `reach_after` bytes past it, so every access the case makes stays inside the Metal buffer
//!   (or threadgroup allocation), and it records zeros in the out-of-bounds read bytes, which Metal
//!   leaves undefined. The candidate allocates a buffer target inside a *guard* of the same reach,
//!   and checks the containment property of the target's compiled encoding: the read bytes against
//!   the values that encoding allows, every other output byte against Metal, and the guard bytes
//!   unchanged (`executor_contract::out_of_bounds`).
//! - **Where a draw or a stage-in dispatch fetches** (a vertex input's, a tessellation control
//!   point's or patch input's, or a kernel stage input's buffer, and an indexed draw's index
//!   buffer), Metal's fetch checks nothing either: the oracle binds the buffer with a shadow of
//!   `reach_after` bytes past its data (fetches step forward from its start, so nothing lies before
//!   it), the candidate allocates it inside a guard of that reach, and the output bytes the fetches
//!   past it produce are declared reads. An index read past the index buffer is a vertex id Metal
//!   leaves undefined (the shadow's), so such a draw fetches no vertex attribute, and its vertex
//!   function reads the id only clamped to a constant (`check::conformance`).
//! - **Where no allocation can be shadowed** (a sized thread-local array, a handle array in an
//!   argument buffer; a top-level texture array), the case's own layout holds what its accesses
//!   reach: the array sits in a thread-local object with words before and past it, the handle array
//!   in an argument buffer with further handles past it, or the texture array before further bound
//!   texture slots (its `reach_after` counts the slots its indices reach, each one bound), so
//!   Metal's accesses stay inside the object, the argument buffer, or the bound slots. The read
//!   bytes (and the bytes a case derives from what its stores did past the array) are Metal's
//!   undefined results, recorded as zeros, and the candidate checks them against the values its
//!   encoding allows.
//!
//! The reach is part of the case's derivation: every access the case makes lies inside it.

use super::{AttributeInput, AuthoredCase, OutputSelection, Stage, TextureType};
use base64::Engine as _;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// The largest shadow or guard reach a target may declare on either side, in bytes.
pub const MAX_REACH: u32 = 64 * 1024;

/// The granularity of a buffer target's length and reach past it: a buffer whose length is a
/// multiple of 16 is its own padded allocation, so "past the buffer" is past the range the binder
/// can bind.
pub const BUFFER_GRANULARITY: u32 = 16;

/// The granularity of a buffer target's reach before it: the offset the Metal oracle binds the
/// target at inside its shadow, which a `constant` buffer needs aligned to 256 bytes on macOS.
pub const REACH_BEFORE_GRANULARITY: u32 = 256;

/// Metal's threadgroup memory limit (`maxThreadgroupMemoryLength` on every Apple GPU family the
/// oracle runs on): a threadgroup target and its shadow fit it.
pub const THREADGROUP_MEMORY_LIMIT: u32 = 32 * 1024;

/// An out-of-bounds case's declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutOfBounds {
    /// Each resource the case accesses past its range: at least one, each resource once.
    pub targets: Vec<OutOfBoundsTarget>,
    /// The output bytes holding results of out-of-bounds reads (loads, or the values atomics
    /// return) of a target whose result Metal leaves undefined.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub reads: Vec<OutOfBoundsRead>,
}

/// One resource a case accesses out of bounds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutOfBoundsTarget {
    pub resource: OutOfBoundsResource,
    /// The containment form the case exercises on it.
    pub form: ContainmentForm,
    /// Bytes before the resource's start that the case's accesses may reach (a buffer target's
    /// negative offsets): the Metal shadow and the candidate guard before it.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub reach_before: u32,
    /// Bytes past the resource's end that the case's accesses may reach: the Metal shadow and the
    /// candidate guard past it (a texture buffer's: the candidate guard only, since Metal defines
    /// its out-of-bounds accesses).
    #[serde(default, skip_serializing_if = "is_zero")]
    pub reach_after: u32,
}

/// The resource of an [`OutOfBoundsTarget`], by its Metal binding (or its argument buffer's and
/// field's).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum OutOfBoundsResource {
    /// The case's buffer at `[[buffer(binding)]]`.
    Buffer { binding: u32 },
    /// The case's threadgroup memory at `[[threadgroup(binding)]]`.
    ThreadgroupMemory { binding: u32 },
    /// The case's texture at `[[texture(binding)]]` (a texture buffer included).
    Texture { binding: u32 },
    /// The case's argument-buffer buffer at `(buffer_binding, field_offset)`: the buffer whose
    /// address the argument buffer at `[[buffer(buffer_binding)]]` holds at byte
    /// `field_offset`, which the kernel reaches through the pointer it loads from there.
    ArgumentBufferBuffer {
        buffer_binding: u32,
        field_offset: u32,
    },
    /// The texture handle array the argument buffer at `[[buffer(buffer_binding)]]` holds from
    /// byte `field_offset`: an AIR array member of at least two textures, each element one of
    /// the case's argument-buffer textures.
    ArgumentBufferTextureArray {
        buffer_binding: u32,
        field_offset: u32,
    },
    /// The entry's thread-local memory: its one `alloca`, an object holding a sized array the
    /// kernel indexes at a dynamic index.
    ThreadMemory,
    /// The buffer feeding attribute `location`: a vertex case's vertex input, a tessellation draw's
    /// control point or patch input, or a kernel's stage input. Each attribute's data is its own
    /// buffer, at the slot the executors' shared slot rule gives it.
    AttributeBuffer { location: u32 },
    /// The index buffer of the case's indexed draw (`Draw::index_buffer`).
    IndexBuffer,
    /// The case's top-level texture array at `[[texture(binding)]]` (`texture_arrays`): an index
    /// past it is clamped onto its last element in the shader.
    TextureArray { binding: u32 },
}

/// Output bytes `[offset, offset + length)` of the selected output (a buffer range) hold results
/// of out-of-bounds reads of `targets[target]`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OutOfBoundsRead {
    pub offset: u64,
    pub length: u64,
    pub target: u32,
}

/// A containment form: a kind of access the product keeps inside the resources the binder supplied,
/// each with its own out-of-bounds rule. The vocabulary is every form of the inventory, each
/// authorable in the cases of the stage [`ContainmentForm::stage`] names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContainmentForm {
    /// Loads and stores through a `device` buffer argument.
    StorageBuffer,
    /// Loads through a `constant` buffer argument.
    UniformBuffer,
    /// Atomics on a `device` buffer argument.
    BufferAtomic,
    /// Accesses to a `device` buffer argument the product reaches by its device address (a
    /// `BufferForm::Psb` slot: a typed atomic or a wide pointer reaches it).
    DeviceAddress,
    /// Accesses to a `[[threadgroup(n)]]` buffer, in the threadgroup arena.
    Threadgroup,
    /// Reads and writes of a `texture_buffer`.
    TexelBuffer,
    /// Reads and writes of a texture.
    Texture,
    /// Accesses through a pointer loaded from memory (a dynamic root), contained by the resident
    /// set: the pointer resolves to the resident allocation that holds it, and each access is
    /// checked against that allocation's size (the device-address encoding). A case reaches past
    /// the buffer an argument buffer points to.
    LoadedPointer,
    /// A dynamic index into a sized thread-local, `constant`, or stage-IO array (the sized arrays
    /// and the constant blob): each array level's index is checked against its length, a load past
    /// it yielding zero and a store past it skipped. A case reaches past a thread-local array
    /// inside a larger thread-local object.
    SizedArray,
    /// A vertex attribute fetch past its vertex buffer: fixed-function input under buffer
    /// robustness, else the pulled fetch with the check. A case reaches past a vertex input's
    /// buffer.
    VertexInput,
    /// An index fetch past the index buffer: bounded by `robustBufferAccess2`, or the binder's
    /// clamp of a direct indexed draw. A case's indexed draw reads past its index buffer.
    IndexBuffer,
    /// A kernel stage-input fetch past its buffer: the pulled fetch through the input's buffer
    /// slot, with the check or the device's buffer robustness. A case reaches past a stage input's
    /// buffer.
    StageInput,
    /// A tessellation control-point or patch-input read past its buffer: the post-tessellation
    /// vertex function's pulled fetch, contained as a stage input's is. A case reaches past a
    /// control point's or a patch input's buffer.
    ControlPoint,
    /// An index past a texture or sampler handle array: a clamped handle-array load, or, in an
    /// argument buffer, a resource id the handle table resolves to a resident texture or the null
    /// handle. A case reaches past an argument buffer's texture array and past a top-level texture
    /// array.
    HandleArrayIndex,
}

impl ContainmentForm {
    pub const ALL: [Self; 14] = [
        Self::StorageBuffer,
        Self::UniformBuffer,
        Self::BufferAtomic,
        Self::DeviceAddress,
        Self::Threadgroup,
        Self::TexelBuffer,
        Self::Texture,
        Self::LoadedPointer,
        Self::SizedArray,
        Self::VertexInput,
        Self::IndexBuffer,
        Self::StageInput,
        Self::ControlPoint,
        Self::HandleArrayIndex,
    ];

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::StorageBuffer => "storage_buffer",
            Self::UniformBuffer => "uniform_buffer",
            Self::BufferAtomic => "buffer_atomic",
            Self::DeviceAddress => "device_address",
            Self::Threadgroup => "threadgroup",
            Self::TexelBuffer => "texel_buffer",
            Self::Texture => "texture",
            Self::LoadedPointer => "loaded_pointer",
            Self::SizedArray => "sized_array",
            Self::VertexInput => "vertex_input",
            Self::IndexBuffer => "index_buffer",
            Self::StageInput => "stage_input",
            Self::ControlPoint => "control_point",
            Self::HandleArrayIndex => "handle_array_index",
        }
    }

    /// Whether Metal defines an out-of-bounds access of this form (MSL 4.1: a texture's or a
    /// texture buffer's read returns zero with the format's default components, and a write is
    /// ignored), so a case compares it with Metal instead of shadowing it.
    pub const fn metal_defined(self) -> bool {
        matches!(self, Self::TexelBuffer | Self::Texture)
    }

    /// Whether a resource of `kind` is a resource of this form.
    fn admits(self, kind: Kind) -> bool {
        match (self, kind) {
            (
                Self::StorageBuffer
                | Self::UniformBuffer
                | Self::BufferAtomic
                | Self::DeviceAddress,
                Kind::Buffer,
            )
            | (Self::Threadgroup, Kind::Threadgroup)
            | (Self::LoadedPointer, Kind::ArgumentBufferBuffer)
            | (Self::HandleArrayIndex, Kind::ArgumentBufferTextureArray | Kind::TextureArray)
            | (Self::SizedArray, Kind::ThreadMemory)
            | (Self::VertexInput, Kind::Attribute(Attribute::Vertex))
            | (
                Self::ControlPoint,
                Kind::Attribute(Attribute::ControlPoint | Attribute::PatchInput),
            )
            | (Self::StageInput, Kind::Attribute(Attribute::Stage))
            | (Self::IndexBuffer, Kind::IndexBuffer) => true,
            (Self::TexelBuffer, Kind::Texture(ty)) => ty == TextureType::Buffer,
            (Self::Texture, Kind::Texture(ty)) => ty != TextureType::Buffer,
            _ => false,
        }
    }

    /// The stage of the cases that can declare a target of this form: a vertex case for the forms
    /// only a draw reaches (vertex input, control point, index buffer), a kernel case otherwise.
    pub const fn stage(self) -> Stage {
        match self {
            Self::VertexInput | Self::ControlPoint | Self::IndexBuffer => Stage::Vertex,
            _ => Stage::Kernel,
        }
    }
}

/// What a target's resource is in its case ([`ContainmentForm::admits`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    Buffer,
    Threadgroup,
    Texture(TextureType),
    ArgumentBufferBuffer,
    ArgumentBufferTextureArray,
    ThreadMemory,
    Attribute(Attribute),
    IndexBuffer,
    TextureArray,
}

/// Which input an attribute buffer feeds, and so how its elements are stepped (the executors'
/// shared derivations: a vertex input per vertex, a control point per patch control point, a patch
/// input per patch, and a kernel stage input per thread position in the grid on x).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Attribute {
    Vertex,
    ControlPoint,
    PatchInput,
    Stage,
}

/// The attribute input at `location` of `case`, with which input it is.
fn attribute(case: &AuthoredCase, location: u32) -> Option<(Attribute, &AttributeInput)> {
    let tessellation = case.tessellation.iter().flat_map(|draw| {
        draw.control_points
            .iter()
            .map(|input| (Attribute::ControlPoint, input))
            .chain(
                draw.patch_inputs
                    .iter()
                    .map(|input| (Attribute::PatchInput, input)),
            )
    });
    case.vertex_inputs
        .iter()
        .map(|input| (Attribute::Vertex, input))
        .chain(tessellation)
        .chain(
            case.kernel_stage_inputs
                .iter()
                .map(|input| (Attribute::Stage, input)),
        )
        .find(|(_, input)| input.location == location)
}

impl std::fmt::Display for ContainmentForm {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl OutOfBoundsResource {
    pub fn label(self) -> String {
        match self {
            Self::Buffer { binding } => format!("buffer {binding}"),
            Self::ThreadgroupMemory { binding } => format!("threadgroup memory {binding}"),
            Self::Texture { binding } => format!("texture {binding}"),
            Self::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
            } => format!("argument-buffer buffer {buffer_binding}+{field_offset}"),
            Self::ArgumentBufferTextureArray {
                buffer_binding,
                field_offset,
            } => format!("argument-buffer texture array {buffer_binding}+{field_offset}"),
            Self::ThreadMemory => "thread memory".into(),
            Self::AttributeBuffer { location } => format!("attribute buffer {location}"),
            Self::IndexBuffer => "index buffer".into(),
            Self::TextureArray { binding } => format!("texture array {binding}"),
        }
    }
}

const fn is_zero(value: &u32) -> bool {
    *value == 0
}

/// The literal rules of `case`'s out-of-bounds declaration, appended to `errors`.
pub(super) fn validate(case: &AuthoredCase, errors: &mut Vec<String>) {
    let Some(declaration) = &case.out_of_bounds else {
        return;
    };
    if declaration.targets.is_empty() {
        errors.push("out_of_bounds declares no target".into());
    }
    if !case.command_buffers.is_empty() || !case.pipeline_states.is_empty() {
        errors.push(
            "out_of_bounds: an out-of-bounds case encodes no indirect commands (the Metal oracle \
             binds a target inside its shadow in the dispatch's own bindings only)"
                .into(),
        );
    }
    let mut resources = HashSet::new();
    for (index, target) in declaration.targets.iter().enumerate() {
        let label = format!("out_of_bounds target {index} ({})", target.resource.label());
        if !resources.insert(target.resource) {
            errors.push(format!("{label} is declared twice"));
        }
        if case.stage != target.form.stage() {
            errors.push(format!(
                "{label}: form {} is authorable in {:?} cases only",
                target.form,
                target.form.stage()
            ));
            continue;
        }
        let Some(kind) = kind(case, target.resource) else {
            errors.push(format!("{label}: the case declares no such resource"));
            continue;
        };
        if !target.form.admits(kind) {
            errors.push(format!(
                "{label}: form {} is not a form of this resource",
                target.form
            ));
            continue;
        }
        for (side, reach) in [
            ("reach_before", target.reach_before),
            ("reach_after", target.reach_after),
        ] {
            if reach > MAX_REACH {
                errors.push(format!("{label}: {side} {reach} exceeds {MAX_REACH} bytes"));
            }
        }
        match target.resource {
            OutOfBoundsResource::Buffer { binding } => {
                validate_buffer_target(case, &label, binding, target, errors)
            }
            OutOfBoundsResource::ThreadgroupMemory { binding } => {
                validate_threadgroup_target(case, &label, binding, target, errors)
            }
            OutOfBoundsResource::Texture { .. } => validate_texture_target(&label, target, errors),
            OutOfBoundsResource::ArgumentBufferBuffer {
                buffer_binding,
                field_offset,
            } => validate_loaded_pointer_target(
                case,
                &label,
                (buffer_binding, field_offset),
                target,
                errors,
            ),
            OutOfBoundsResource::ArgumentBufferTextureArray {
                buffer_binding,
                field_offset,
            } => validate_handle_array_target(
                case,
                &label,
                (buffer_binding, field_offset),
                target,
                errors,
            ),
            OutOfBoundsResource::ThreadMemory => validate_in_layout_reach(&label, target, errors),
            OutOfBoundsResource::TextureArray { binding } => {
                validate_texture_array_target(case, &label, binding, target, errors)
            }
            OutOfBoundsResource::AttributeBuffer { location } => {
                validate_attribute_target(case, &label, location, target, errors)
            }
            OutOfBoundsResource::IndexBuffer => validate_index_target(case, &label, target, errors),
        }
    }
    validate_reads(case, declaration, errors);
}

/// What `resource` is in `case`, or `None` when the case declares no such resource.
fn kind(case: &AuthoredCase, resource: OutOfBoundsResource) -> Option<Kind> {
    let declared = |present: bool, kind: Kind| present.then_some(kind);
    match resource {
        OutOfBoundsResource::Buffer { binding } => declared(
            case.buffers.iter().any(|buffer| buffer.binding == binding),
            Kind::Buffer,
        ),
        OutOfBoundsResource::ThreadgroupMemory { binding } => declared(
            case.threadgroup_memory
                .iter()
                .any(|memory| memory.binding == binding),
            Kind::Threadgroup,
        ),
        OutOfBoundsResource::Texture { binding } => case
            .textures
            .iter()
            .find(|texture| texture.binding == binding)
            .map(|texture| Kind::Texture(texture.texture_type)),
        // Checked against the argument-buffer resources by the target's own rules.
        OutOfBoundsResource::ArgumentBufferBuffer { .. } => Some(Kind::ArgumentBufferBuffer),
        OutOfBoundsResource::ArgumentBufferTextureArray { .. } => {
            Some(Kind::ArgumentBufferTextureArray)
        }
        OutOfBoundsResource::ThreadMemory => Some(Kind::ThreadMemory),
        OutOfBoundsResource::AttributeBuffer { location } => {
            attribute(case, location).map(|(attribute, _)| Kind::Attribute(attribute))
        }
        OutOfBoundsResource::IndexBuffer => declared(
            case.draw
                .as_ref()
                .is_some_and(|draw| draw.index_buffer.is_some()),
            Kind::IndexBuffer,
        ),
        OutOfBoundsResource::TextureArray { binding } => declared(
            case.texture_arrays
                .iter()
                .any(|array| array.binding == binding),
            Kind::TextureArray,
        ),
    }
}

/// A target the Metal oracle binds inside a shadow of its reach past it and the candidate
/// allocates inside a guard: an attribute buffer or an index buffer, `length` bytes, whose fetches
/// step forward from its start (an element index or an index position is unsigned), so nothing
/// lies before it. The buffer must be its own padded allocation.
fn validate_fetched_layout(
    label: &str,
    length: u64,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    if !length.is_multiple_of(u64::from(BUFFER_GRANULARITY)) {
        errors.push(format!(
            "{label}: its length {length} must be a multiple of {BUFFER_GRANULARITY} bytes, so the \
             buffer is its own padded allocation and past it is past every range the \
             binder binds"
        ));
    }
    if target.reach_before != 0 {
        errors.push(format!(
            "{label}: its fetches step forward from its start, so nothing lies before it: \
             reach_before must be 0"
        ));
    }
    if target.reach_after == 0 || !target.reach_after.is_multiple_of(BUFFER_GRANULARITY) {
        errors.push(format!(
            "{label}: reach_after must be a nonzero multiple of {BUFFER_GRANULARITY} bytes"
        ));
    }
}

/// The reach rule of a fetched target ([`validate_fetched_layout`]) of `length` bytes whose
/// fetches read through byte `end`: they reach past it (else nothing is out of bounds), and its
/// reach past it covers them, so every access Metal makes stays inside the shadow.
pub(crate) fn validate_fetched_reach(
    label: &str,
    length: u64,
    end: u64,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    if end <= length {
        errors.push(format!(
            "{label}: the case fetches through byte {end} of its {length} bytes, never past it"
        ));
    } else if end - length > u64::from(target.reach_after) {
        errors.push(format!(
            "{label}: the case fetches through byte {end}, {} bytes past it, beyond its \
             reach_after {}",
            end - length,
            target.reach_after
        ));
    }
}

/// An attribute buffer target: the buffer feeding a vertex input, a patch input, or a kernel stage
/// input, whose elements the draw or dispatch fetches one per vertex, patch, or thread on x. Its
/// length is a multiple of its stride, so no element straddles its end, and the fetches past it are
/// whole elements. A control point's elements are counted against the AIR's control points per
/// patch, which this literal check does not read (`check::conformance`).
fn validate_attribute_target(
    case: &AuthoredCase,
    label: &str,
    location: u32,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some((kind, input)) = attribute(case, location) else {
        return;
    };
    let Ok(bytes) = base64::engine::general_purpose::STANDARD.decode(&input.bytes_b64) else {
        return;
    };
    let length = bytes.len() as u64;
    if input.stride == 0 || !length.is_multiple_of(u64::from(input.stride)) {
        errors.push(format!(
            "{label}: its length {length} must be a multiple of its stride {}, so no element \
             straddles its end",
            input.stride
        ));
    }
    let elements = match kind {
        Attribute::Vertex => {
            if case
                .draw
                .as_ref()
                .is_some_and(|draw| draw.index_buffer.is_some())
            {
                errors.push(format!(
                    "{label}: an indexed draw's vertex elements are its indices, so a vertex \
                     input target is a draw without an index buffer's"
                ));
                return;
            }
            case.vertex_elements()
        }
        Attribute::PatchInput => case
            .tessellation
            .as_ref()
            .and_then(|draw| u32::try_from(draw.factors.len()).ok()),
        // One past the last element the dispatch's stage-in region lets a thread fetch.
        Attribute::Stage => case.dispatch.as_ref().and_then(|d| d.stage_in_elements()),
        // `patches × control points` elements: the AIR states the control points
        // (`check::conformance`).
        Attribute::ControlPoint => return validate_fetched_layout(label, length, target, errors),
    };
    let Some(elements) = elements.filter(|elements| *elements > 0) else {
        errors.push(format!("{label}: the case fetches no element of it"));
        return;
    };
    validate_fetched_layout(label, length, target, errors);
    validate_fetched_reach(label, length, fetch_end(elements, input), target, errors);
}

/// One past the last byte the fetches of `elements` elements of `input` read: the last element's
/// start plus its format's size.
pub fn fetch_end(elements: u32, input: &AttributeInput) -> u64 {
    u64::from(elements.saturating_sub(1)) * u64::from(input.stride)
        + u64::from(input.format.byte_size())
}

/// The index buffer target of an indexed draw. The draw fetches no vertex attribute: an index read
/// past the buffer is a vertex id Metal leaves undefined (the oracle's shadow gives it a large
/// one), and Metal's vertex-descriptor fetch at such an id reads unchecked memory. The indices the
/// draw reads inside the buffer include index 0, the contained result of a read past it under
/// `robustBufferAccess2`, so that result is a vertex id the draw shades anyway, and an index past
/// the buffer can change no byte that the in-range ones do not (the binder's clamp draws none of
/// them).
fn validate_index_target(
    case: &AuthoredCase,
    label: &str,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some(draw) = &case.draw else {
        return;
    };
    let Some(index_buffer) = &draw.index_buffer else {
        return;
    };
    if !case.vertex_inputs.is_empty() || case.tessellation.is_some() {
        errors.push(format!(
            "{label}: an index read past the buffer is a vertex id Metal leaves undefined, so the \
             draw fetches no vertex attribute (Metal's fetch at such an id is unchecked)"
        ));
    }
    if !index_buffer.indices_inside(draw.vertex_count).contains(&0) {
        errors.push(format!(
            "{label}: the indices the draw reads inside the buffer must include index 0, the \
             contained result of a read past it under robustBufferAccess2"
        ));
    }
    let Some(bytes) = index_buffer.bytes() else {
        return;
    };
    let end = u64::from(index_buffer.offset)
        + u64::from(draw.vertex_count) * u64::from(index_buffer.index_type.size());
    validate_fetched_layout(label, bytes.len() as u64, target, errors);
    validate_fetched_reach(label, bytes.len() as u64, end, target, errors);
}

/// A top-level texture array target: an index past it names the argument-table slots after its last
/// element, which Metal reads as they are bound. The target's `reach_after` is the number of those
/// slots the case's indices reach (at least 1; `reach_before` is 0, an index is unsigned), and the
/// case binds a texture at every one of them, so no index of the case names a slot Metal holds
/// nothing in. The index is dynamic data the AIR loads, whose range no structural fact bounds: that
/// the case's indices stay inside the array and those slots is the authoring contract this reach
/// states. Its elements are at least two, so the shader's index is dynamic over the array (the
/// clamp).
fn validate_texture_array_target(
    case: &AuthoredCase,
    label: &str,
    binding: u32,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    if target.reach_before != 0 || target.reach_after == 0 {
        errors.push(format!(
            "{label}: reach_after is the number of argument-table slots past the array that the \
             case's indices reach (at least 1), and reach_before must be 0"
        ));
    }
    let Some(array) = case
        .texture_arrays
        .iter()
        .find(|array| array.binding == binding)
    else {
        return;
    };
    if array.elements.len() < 2 {
        errors.push(format!(
            "{label}: the array must hold at least two elements"
        ));
    }
    let bound = |slot: u32| {
        case.textures.iter().any(|texture| texture.binding == slot)
            || case.texture_arrays.iter().any(|other| {
                u32::try_from(other.elements.len())
                    .ok()
                    .and_then(|elements| other.binding.checked_add(elements))
                    .is_some_and(|end| (other.binding..end).contains(&slot))
            })
    };
    let past = u32::try_from(array.elements.len())
        .ok()
        .and_then(|elements| binding.checked_add(elements));
    let unbound = (0..target.reach_after)
        .map(|step| past.and_then(|past| past.checked_add(step)))
        .filter(|slot| !slot.is_some_and(bound))
        .collect::<Vec<_>>();
    if !unbound.is_empty() {
        errors.push(format!(
            "{label}: the case binds no texture at {} of the {} slots past the array's last \
             element that its indices reach",
            unbound.len(),
            target.reach_after
        ));
    }
}

/// A texture or texture buffer target: Metal defines its out-of-bounds accesses, so the oracle
/// needs no shadow. A texture buffer is host-visible buffer memory in the candidate, which
/// allocates it inside a guard: its `reach_after` is that guard's reach past it, a nonzero multiple
/// of 16 covering the furthest texel the case accesses, and texel coordinates are unsigned, so
/// nothing lies before it. An image's memory is opaque: it has no guard and declares no reach.
fn validate_texture_target(label: &str, target: &OutOfBoundsTarget, errors: &mut Vec<String>) {
    if target.reach_before != 0 {
        errors.push(format!(
            "{label}: a texture's coordinates are unsigned, so nothing lies before it: \
             reach_before must be 0"
        ));
    }
    match target.form {
        ContainmentForm::TexelBuffer => {
            if target.reach_after == 0 || !target.reach_after.is_multiple_of(BUFFER_GRANULARITY) {
                errors.push(format!(
                    "{label}: reach_after must be a nonzero multiple of {BUFFER_GRANULARITY} \
                     bytes covering the furthest texel the case accesses (the candidate's guard \
                     past the texture buffer)"
                ));
            }
        }
        _ => {
            if target.reach_after != 0 {
                errors.push(format!(
                    "{label}: an image's memory is opaque, so it has no guard: reach_after must \
                     be 0"
                ));
            }
        }
    }
}

fn validate_buffer_target(
    case: &AuthoredCase,
    label: &str,
    binding: u32,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some(buffer) = case.buffers.iter().find(|buffer| buffer.binding == binding) else {
        errors.push(format!("{label}: the case declares no such buffer"));
        return;
    };
    validate_no_rate_map(label, buffer.rasterization_rate_map.is_some(), errors);
    validate_shadowed_buffer(
        label,
        super::initial_bytes(
            buffer.bytes_b64.as_deref(),
            buffer.initial_bytes_b64.as_deref(),
        )
        .map(|bytes| bytes.len()),
        target,
        errors,
    );
    if case
        .argument_buffer_buffers
        .iter()
        .any(|embedded| embedded.buffer_binding == binding)
        || case
            .argument_buffer_textures
            .iter()
            .any(|embedded| embedded.buffer_binding == binding)
    {
        errors.push(format!(
            "{label}: an argument buffer's owner is not an out-of-bounds target"
        ));
    }
}

/// A rasterization-rate map's buffer holds each executor's own data of the map at offset 0 of its
/// own allocation, so it has no shadow and is no out-of-bounds target.
fn validate_no_rate_map(label: &str, rate_map: bool, errors: &mut Vec<String>) {
    if rate_map {
        errors.push(format!(
            "{label}: a rasterization-rate map's buffer is no out-of-bounds target"
        ));
    }
}

/// The rules of a target the Metal oracle binds inside its shadow and the candidate allocates
/// inside its guard (a buffer, or a buffer an argument buffer points to), whose bytes are
/// `length` long.
fn validate_shadowed_buffer(
    label: &str,
    length: Option<usize>,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    if length.is_some_and(|length| length % BUFFER_GRANULARITY as usize != 0) {
        errors.push(format!(
            "{label}: its length must be a multiple of {BUFFER_GRANULARITY} bytes, so the buffer \
             is its own padded allocation and past it is past every range the binder \
             binds"
        ));
    }
    if target.reach_after == 0 || !target.reach_after.is_multiple_of(BUFFER_GRANULARITY) {
        errors.push(format!(
            "{label}: reach_after must be a nonzero multiple of {BUFFER_GRANULARITY} bytes"
        ));
    }
    if !target.reach_before.is_multiple_of(REACH_BEFORE_GRANULARITY) {
        errors.push(format!(
            "{label}: reach_before must be a multiple of {REACH_BEFORE_GRANULARITY} bytes (the \
             offset the Metal oracle binds the buffer at inside its shadow)"
        ));
    }
}

/// A buffer an argument buffer points to: one of the case's argument-buffer buffers, shadowed and
/// guarded as a buffer target is. It is not a command buffer's record storage (an out-of-bounds
/// case encodes no indirect commands).
fn validate_loaded_pointer_target(
    case: &AuthoredCase,
    label: &str,
    (owner, field): (u32, u32),
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some(buffer) = case
        .argument_buffer_buffers
        .iter()
        .find(|buffer| buffer.buffer_binding == owner && buffer.field_offset == field)
    else {
        errors.push(format!(
            "{label}: the case declares no such argument-buffer buffer"
        ));
        return;
    };
    validate_no_rate_map(label, buffer.rasterization_rate_map.is_some(), errors);
    validate_shadowed_buffer(
        label,
        super::initial_bytes(
            buffer.bytes_b64.as_deref(),
            buffer.initial_bytes_b64.as_deref(),
        )
        .map(|bytes| bytes.len()),
        target,
        errors,
    );
}

/// A handle array in an argument buffer: its first element is one of the case's argument-buffer
/// textures, and the handles an index past it reaches lie in the argument buffer itself (the case's
/// own layout), so the target has no shadow and no guard.
fn validate_handle_array_target(
    case: &AuthoredCase,
    label: &str,
    (owner, field): (u32, u32),
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    if !case
        .argument_buffer_textures
        .iter()
        .any(|texture| texture.buffer_binding == owner && texture.field_offset == field)
    {
        errors.push(format!(
            "{label}: the case declares no argument-buffer texture at the array's first element"
        ));
    }
    validate_in_layout_reach(label, target, errors);
}

/// A target whose out-of-bounds accesses stay inside the case's own layout (a thread-local
/// object's words around its array, an argument buffer's handles past its array): no allocation
/// surrounds it, so it declares no reach.
fn validate_in_layout_reach(label: &str, target: &OutOfBoundsTarget, errors: &mut Vec<String>) {
    if target.reach_before != 0 || target.reach_after != 0 {
        errors.push(format!(
            "{label}: what its accesses reach lies in the case's own layout, not in a shadow or \
             a guard: reach_before and reach_after must be 0"
        ));
    }
}

fn validate_threadgroup_target(
    case: &AuthoredCase,
    label: &str,
    binding: u32,
    target: &OutOfBoundsTarget,
    errors: &mut Vec<String>,
) {
    let Some(memory) = case
        .threadgroup_memory
        .iter()
        .find(|memory| memory.binding == binding)
    else {
        errors.push(format!(
            "{label}: the case declares no such threadgroup memory"
        ));
        return;
    };
    if target.reach_before != 0 {
        errors.push(format!(
            "{label}: threadgroup memory has no shadow before it: reach_before must be 0"
        ));
    }
    if target.reach_after == 0 || !target.reach_after.is_multiple_of(BUFFER_GRANULARITY) {
        errors.push(format!(
            "{label}: reach_after must be a nonzero multiple of {BUFFER_GRANULARITY} bytes"
        ));
    }
    let total = case
        .threadgroup_memory
        .iter()
        .map(|memory| u64::from(memory.length))
        .sum::<u64>()
        + u64::from(target.reach_after);
    if total > u64::from(THREADGROUP_MEMORY_LIMIT) {
        errors.push(format!(
            "{label}: the threadgroup memory with its shadow ({total} bytes) exceeds Metal's \
             {THREADGROUP_MEMORY_LIMIT}"
        ));
    }
    if case.threadgroup_memory.len() != 1 || memory.length == 0 {
        errors.push(format!(
            "{label}: a threadgroup target is the case's one threadgroup allocation, so past it is \
             past the threadgroup arena (the arena is every allocation's)"
        ));
    }
}

fn validate_reads(case: &AuthoredCase, declaration: &OutOfBounds, errors: &mut Vec<String>) {
    if declaration.reads.is_empty() {
        return;
    }
    let OutputSelection::Buffer { length, .. } = case.output else {
        errors.push("out_of_bounds reads select bytes of a buffer output only".into());
        return;
    };
    let mut ranges = Vec::<(u64, u64)>::new();
    for (index, read) in declaration.reads.iter().enumerate() {
        let label = format!("out_of_bounds read {index}");
        let Some(target) = usize::try_from(read.target)
            .ok()
            .and_then(|target| declaration.targets.get(target))
        else {
            errors.push(format!("{label} names no target {}", read.target));
            continue;
        };
        if target.form.metal_defined() {
            errors.push(format!(
                "{label}: Metal defines a {}'s out-of-bounds read, so the case compares it with \
                 Metal, not as a read",
                target.form
            ));
        }
        if read.length == 0 || read.offset % 4 != 0 || read.length % 4 != 0 {
            errors.push(format!(
                "{label}: its offset and a nonzero length must be multiples of 4 bytes"
            ));
        }
        let Some(end) = read
            .offset
            .checked_add(read.length)
            .filter(|end| *end <= length)
        else {
            errors.push(format!(
                "{label} lies past the selected output ({length} bytes)"
            ));
            continue;
        };
        if ranges
            .iter()
            .any(|&(start, stop)| read.offset < stop && start < end)
        {
            errors.push(format!("{label} overlaps another read"));
        }
        ranges.push((read.offset, end));
    }
}
