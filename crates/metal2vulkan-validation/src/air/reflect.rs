//! The declared interface of a selected AIR entry: its resources and their descriptor locations,
//! its stage inputs and outputs, its imageblocks, and its function constants, read from the AIR
//! metadata ([`crate::air::meta`]) without translating anything.
//!
//! Validation's copy of the old engine's `inspect`, which the checker, the Metal oracle and
//! classification share: the old engine's interface builder, its descriptor bands and checks, its
//! buffer-access refinement and its constexpr-sampler decode, over the entry as the old engine's
//! request selected it, with the case's function constants applied. What the old engine did beyond
//! that before building the interface (linking the case's visible functions, lowering asynchronous
//! copies, dynamic copies and command buffers) is not repeated: none of it moves a fact this
//! interface reports.

use crate::air::meta;
/// The metadata types the interface's fields carry.
pub use crate::air::meta::{
    AirType, ArgumentBufferResourceKind, DepthQualifier, FragmentImageblockProjection,
    FunctionConstant, MeshFieldMeta as MeshField, MeshFieldRole, MeshOutputsMeta as MeshOutputs,
    MeshTopology, PatchDomain, PayloadMeta as MeshPayload, TextureComponent, TextureDimension,
    TextureFormat, TextureShape,
};
use crate::case::Stage;

mod access;
mod build;
mod descriptors;
mod samplers;
#[cfg(test)]
mod tests;

#[cfg(test)]
pub(crate) use build::{interface_from_kernel, interface_from_vertex};
pub use samplers::{
    SamplerAddressMode, SamplerBorderColor, SamplerCompareFunction, SamplerCoordinates,
    SamplerFilter, SamplerMipFilter, SamplerReduction, SamplerState,
};

/// Entries of the Metal threadgroup-buffer argument table (`air.max_threadgroup_buffers` is 31, so
/// indices run 0..=30). A threadgroup buffer takes no descriptor.
pub const THREADGROUP_BUFFER_ARGUMENT_COUNT: u32 = 31;
pub const THREADGROUP_BUFFER_ARGUMENT_COUNT_USIZE: usize =
    THREADGROUP_BUFFER_ARGUMENT_COUNT as usize;

/// AIR address space 3, threadgroup memory: a `[[buffer(n)]]` in it consumes no descriptor.
const ADDRESS_SPACE_THREADGROUP: u32 = 3;
/// AIR address space 2, constant memory: a read-only buffer.
const ADDRESS_SPACE_CONSTANT: u32 = 2;

/// The shader stage of an interface.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderStage {
    Vertex,
    /// A post-tessellation vertex function.
    TessellationEvaluation,
    Fragment,
    Kernel,
    /// An object function: a mesh render pipeline's first stage.
    Object,
    /// A mesh function: a mesh render pipeline's geometry stage.
    Mesh,
}

/// The kind of a bound resource.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceKind {
    /// A `[[buffer(n)]]` device or constant buffer.
    Buffer,
    /// A `[[buffer(n)]]` in threadgroup memory: no descriptor.
    ThreadgroupBuffer,
    /// A kernel `[[stage_in]]` attribute array: `metal_index` is its synthetic buffer slot.
    KernelStageInput,
    /// A `[[texture(n)]]` sampled texture.
    Texture,
    /// A texture-handle array; [`ResourceBinding::access`] tells sampled from storage.
    TextureArray,
    /// A `write` or `read_write` texture.
    StorageImage,
    /// A `[[sampler(n)]]`.
    Sampler,
    /// An AIR constexpr sampler, at the first sampler-band binding no Metal sampler claims; its
    /// state is [`ResourceBinding::static_sampler`].
    StaticSampler,
    /// A `[[color(n)]]` framebuffer-fetch input.
    ColorInput,
    /// An acceleration structure the old engine shadowed with a buffer.
    AccelerationStructureShadow,
    /// A primitive acceleration structure.
    PrimitiveAccelerationStructure,
    /// A visible-function table: no descriptor.
    VisibleFunctionTable,
    /// An intersection-function table: no descriptor.
    IntersectionFunctionTable,
    /// A texture an `air.indirect_buffer` argument buffer holds, at a synthetic texture index.
    EmbeddedArgBufferTexture,
    /// A device buffer an `air.indirect_buffer` argument buffer holds: no descriptor.
    EmbeddedArgBufferBuffer,
}

/// A descriptor location in the old engine's descriptor ABI (set 0).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorLocation {
    pub set: u32,
    pub binding: u32,
    /// Descriptors at this binding: a texture-handle array's length, otherwise 1.
    pub count: u32,
}

/// A resource's declared access, tightened for buffers by the entry's own parameter attributes and
/// uses (`reflect::access`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceAccess {
    /// The entry never names this buffer.
    Unused,
    ReadOnly,
    WriteOnly,
    ReadWrite,
    /// A sampled texture.
    Sampled,
    /// A storage texture.
    Storage,
}

/// Whether AIR bounds a buffer to one object or leaves its extent open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferExtent {
    /// One object of `bytes` bytes.
    Object { bytes: u32 },
    /// An element size and no array length.
    Unbounded,
    /// The metadata does not say.
    Unknown,
}

/// The argument buffer a resource sits in, and where.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EmbeddedArgBuffer {
    /// Entry parameter index of the owning `air.indirect_buffer`.
    pub buffer_param_index: u32,
    /// Metal buffer index of the owning argument buffer.
    pub buffer_index: u32,
    /// Byte offset of the field in the argument buffer.
    pub field_offset: u32,
    /// The field's member ordinal.
    pub field_ordinal: u32,
    /// The field's `[[id(n)]]`.
    pub argument_index: u32,
    /// The handle kind the field declares, if AIR names one.
    pub resource_kind: Option<ArgumentBufferResourceKind>,
}

/// One bound resource.
#[derive(Clone, Debug, PartialEq)]
pub struct ResourceBinding {
    pub kind: ResourceKind,
    /// The Metal index, or the synthetic index of an embedded texture or constexpr sampler.
    pub metal_index: u32,
    /// Its descriptor location, or `None` for a resource that takes none.
    pub descriptor: Option<DescriptorLocation>,
    /// The entry parameter it comes from; `None` for an embedded or constexpr resource.
    pub param_index: Option<u32>,
    /// A kernel stage input's AIR attribute location.
    pub stage_input_location: Option<u32>,
    /// A buffer's AIR address space.
    pub address_space: Option<u32>,
    /// A buffer's `air.arg_type_size`.
    pub declared_size: Option<u32>,
    pub extent: Option<BufferExtent>,
    /// A buffer's `air.struct_type_info` layout.
    pub type_layout: Option<AirType>,
    /// The AIR argument type name.
    pub type_name: Option<String>,
    pub texture_shape: Option<TextureShape>,
    /// For an embedded resource, the argument-buffer field it sits in.
    pub embedded_source: Option<EmbeddedArgBuffer>,
    pub access: Option<ResourceAccess>,
    /// A constexpr sampler's state.
    pub static_sampler: Option<SamplerState>,
}

/// A vertex input attribute.
#[derive(Clone, Debug, PartialEq)]
pub struct VertexAttribute {
    pub location: u32,
    pub type_name: Option<String>,
}

/// A fragment `[[stage_in]]` input, or a vertex function's user varying.
#[derive(Clone, Debug, PartialEq)]
pub struct Varying {
    pub location: u32,
    pub type_name: Option<String>,
}

/// The vertex builtins a vertex function writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VertexBuiltins {
    pub writes_position: bool,
}

/// A fragment render-target output.
#[derive(Clone, Debug, PartialEq)]
pub struct RenderTarget {
    pub location: u32,
    pub type_name: Option<String>,
}

/// A kernel `[[imageblock]]` parameter.
#[derive(Clone, Debug, PartialEq)]
pub struct ImageblockLayout {
    pub param_index: u32,
}

/// One implicit-imageblock attachment plane the module loads or stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImplicitImageblockAttachment {
    pub attachment: u32,
    pub data_rate: u32,
    pub binding: u32,
    pub format: TextureFormat,
    pub access: ResourceAccess,
}

/// One master member of a fragment function's explicit imageblock.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblockMember {
    pub offset: u32,
    pub size: u32,
    pub type_name: String,
    pub semantic: String,
    pub raster_order_group: u32,
    /// Its binding when an entry projection reads or writes it.
    pub binding: Option<u32>,
    pub access: ResourceAccess,
}

/// A fragment function's explicit imageblock: the master members and the projections the entry
/// reads and writes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FragmentImageblock {
    pub sample_size: u32,
    pub members: Vec<FragmentImageblockMember>,
    pub inputs: Vec<FragmentImageblockProjection>,
    pub outputs: Vec<FragmentImageblockProjection>,
}

/// A post-tessellation vertex function's patch interface.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationInterface {
    pub domain: PatchDomain,
    pub control_point_count: u32,
    pub control_point_attributes: Vec<TessellationAttribute>,
    pub patch_attributes: Vec<TessellationAttribute>,
    pub instance_id: Option<TessellationAttribute>,
    pub amplification_id: Option<TessellationAttribute>,
    pub amplification_count: Option<TessellationAttribute>,
}

/// One patch or control-point attribute, or a system input of a post-tessellation function.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TessellationAttribute {
    pub location: u32,
    pub type_name: Option<String>,
}

/// What an object or mesh function declares beyond a kernel's interface: the payload it writes
/// (object) or reads (mesh), an object function's mesh grid, and a mesh function's outputs.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MeshStageInterface {
    /// The `[[payload]]` parameter.
    pub payload: Option<MeshPayload>,
    /// An object function's `mesh_grid_properties` parameter is declared.
    pub mesh_grid: bool,
    /// The entry sets the grid of mesh threadgroups it launches
    /// (`air.set_threadgroups_per_grid_mesh_properties`); an object threadgroup that sets none
    /// launches nothing.
    pub sets_mesh_grid: bool,
    /// A mesh function's output type.
    pub outputs: Option<MeshOutputs>,
    /// `[[max_total_threads_per_threadgroup(N)]]`.
    pub max_total_threads: Option<u32>,
    /// An object function's `[[max_total_threadgroups_per_mesh_grid(N)]]`.
    pub max_mesh_threadgroups: Option<u32>,
}

impl MeshStageInterface {
    /// Whether the mesh function's vertices state a position, which a rasterizing pipeline's
    /// fragment stage reads its coverage from.
    pub fn writes_position(&self) -> bool {
        self.outputs.as_ref().is_some_and(|outputs| {
            outputs
                .vertex
                .iter()
                .any(|field| field.role == MeshFieldRole::Position)
        })
    }
}

/// The declared interface of one entry.
#[derive(Clone, Debug, PartialEq)]
pub struct ShaderInterface {
    pub stage: ShaderStage,
    /// The entry function's name.
    pub entry_point: Option<String>,
    /// Every bound resource: the entry's parameters, then embedded resources, then constexpr
    /// samplers.
    pub bindings: Vec<ResourceBinding>,
    /// Every resource-handle field the entry's argument buffers declare.
    pub argument_buffer_fields: Vec<EmbeddedArgBuffer>,
    pub vertex_attributes: Vec<VertexAttribute>,
    /// Fragment `[[stage_in]]` inputs, or vertex user varyings.
    pub varyings: Vec<Varying>,
    pub render_targets: Vec<RenderTarget>,
    /// Return-struct members tagged `[[depth]]`.
    pub depth_members: Vec<u32>,
    pub depth_qualifier: Option<DepthQualifier>,
    /// Return-struct members tagged `[[stencil]]`.
    pub stencil_members: Vec<u32>,
    pub vertex_builtins: Option<VertexBuiltins>,
    pub tessellation: Option<TessellationInterface>,
    /// Kernel `[[imageblock]]` parameters, by parameter index.
    pub imageblock_layouts: Vec<ImageblockLayout>,
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    pub fragment_imageblock: Option<FragmentImageblock>,
    /// An object or mesh function's own interface.
    pub mesh_stage: Option<MeshStageInterface>,
    /// The module's `[[function_constant(N)]]` inventory.
    pub function_constants: Vec<FunctionConstant>,
}

impl ShaderInterface {
    /// The binding of `kind` at `metal_index`, if there is one.
    pub fn binding_at(&self, kind: ResourceKind, metal_index: u32) -> Option<&ResourceBinding> {
        self.bindings
            .iter()
            .find(|binding| binding.kind == kind && binding.metal_index == metal_index)
    }
}

/// The interface of the entry `entry` of `stage` in the module text `ll`, as the old engine's
/// `inspect` reported it for that request with these function constants:
///
/// 1. the text sanitized as the old engine sanitized its input;
/// 2. the entry selected (a module of several entries keeps only its stage root);
/// 3. the function constants baked into their initializers, and the metadata they gate
///    specialized;
/// 4. the entry's interface locations resolved;
/// 5. the interface built from the stage's metadata, its descriptors checked against their bands;
/// 6. the function constants listed, buffer access tightened from the entry body, constexpr
///    samplers decoded and placed;
/// 7. the descriptor ABI checked as one collision-free mapping.
///
/// An object or mesh entry's interface is a kernel's (its resources, threadgroup memory and grid
/// builtins, in the kernel's descriptor bands) with its own stage's facts
/// ([`ShaderInterface::mesh_stage`]).
pub fn inspect(
    ll: &str,
    stage: Stage,
    entry: &str,
    function_constants: &[(u32, Vec<u8>)],
) -> Result<ShaderInterface, String> {
    let source = crate::air::sanitize::sanitize_ll_text(ll);
    let source = meta::select_translation_entry(source, stage, entry)?;
    let source =
        crate::air::fc_specialize::specialize_air_function_constants(&source, function_constants)
            .map_err(|error| error.to_string())?;
    meta::validate_entry_locations(&source, stage)?;
    let mut interface = match stage {
        Stage::Fragment => {
            let (frag, entry_name) = meta::parse_air_fragment_meta(&source);
            build::interface_from_fragment(&frag.unwrap_or_default(), entry_name.as_deref())
        }
        Stage::Vertex => {
            let (vert, entry_name) = meta::parse_air_vertex_meta(&source);
            build::interface_from_vertex(&vert.unwrap_or_default(), entry_name.as_deref())
        }
        Stage::Kernel => {
            let (kern, entry_name) = meta::parse_air_kernel_meta(&source);
            build::interface_from_kernel(&kern.unwrap_or_default(), entry_name.as_deref())
        }
        Stage::Object | Stage::Mesh => {
            let (kern, entry_name) = meta::parse_air_mesh_stage_meta(&source, stage);
            // What the entry's body does is read from what it reaches alone: another entry of the
            // module may set a mesh grid this one does not.
            let reached = crate::air::entry_text(ll, entry)?;
            build::interface_from_mesh_stage(
                stage,
                &kern.unwrap_or_default(),
                entry_name.as_deref(),
                &reached,
            )
        }
    };
    descriptors::check_bands(&interface)?;
    interface.function_constants = meta::parse_function_constants(&source);
    access::refine_buffer_access_from_entry(&mut interface, &source);
    samplers::add_static_samplers(&mut interface, &source)?;
    descriptors::validate_descriptor_abi(&interface)?;
    Ok(interface)
}

/// [`inspect`] of the one entry of `stage` the module declares, as a request naming only the
/// stage selected it (a module with several is refused), with no function constants.
#[cfg(test)]
pub(crate) fn inspect_stage(ll: &str, stage: Stage) -> Result<ShaderInterface, String> {
    let label = stage.metadata_label();
    let entries = crate::air::module_entries(ll)?
        .into_iter()
        .filter(|entry| entry.stage == label)
        .collect::<Vec<_>>();
    let [entry] = &entries[..] else {
        return Err(format!(
            "entry selection matched {} {label} entries; select one by name",
            entries.len()
        ));
    };
    inspect(ll, stage, &entry.name, &[])
}
