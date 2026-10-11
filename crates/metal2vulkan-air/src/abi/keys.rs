//! The closed vocabulary of AIR argument, output, member, visible-function, and Core Image kernel
//! keys.

/// Every key an argument, output, or struct-member node may carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(super) enum Key {
    // ---- Roles: resources ----
    Buffer,
    IndirectBuffer,
    Texture,
    Sampler,
    InstanceAccelerationStructure,
    PrimitiveAccelerationStructure,
    VisibleFunctionTable,
    IntersectionFunctionTable,
    IndirectConstant,
    CommandBuffer,
    ComputePipelineState,
    RenderPipelineState,
    /// `air.depth_stencil_state` (an argument buffer's `depth_stencil_state`).
    DepthStencilState,
    Imageblock,
    ImageblockData,
    /// `air.payload` (object and mesh functions).
    Payload,
    /// `air.mesh_grid_properties` (object functions).
    MeshGridProperties,
    /// `air.mesh` (mesh functions).
    Mesh,
    // ---- Roles: stage inputs ----
    StageIn,
    VertexInput,
    FragmentInput,
    PatchInput,
    PatchControlPointInput,
    // ---- Roles: builtins ----
    ThreadPositionInGrid,
    ThreadPositionInThreadgroup,
    ThreadgroupPositionInGrid,
    ThreadsPerThreadgroup,
    ThreadsPerGrid,
    ThreadgroupsPerGrid,
    ThreadIndexInThreadgroup,
    ThreadIndexInSimdgroup,
    ThreadIndexInQuadgroup,
    SimdgroupIndexInThreadgroup,
    QuadgroupIndexInThreadgroup,
    ThreadsPerSimdgroup,
    SimdgroupsPerThreadgroup,
    QuadgroupsPerThreadgroup,
    DispatchThreadsPerThreadgroup,
    DispatchSimdgroupsPerThreadgroup,
    StageInGridSize,
    StageInGridOrigin,
    VertexId,
    InstanceId,
    BaseVertex,
    BaseInstance,
    AmplificationId,
    AmplificationCount,
    PositionInPatch,
    PatchId,
    FrontFacing,
    PointCoord,
    PrimitiveId,
    SampleId,
    SampleMaskIn,
    BarycentricCoord,
    // ---- Roles: inputs and outputs ----
    Position,
    RenderTarget,
    RenderTargetArrayIndex,
    ViewportArrayIndex,
    // ---- Roles: outputs ----
    VertexOutput,
    PointSize,
    ClipDistance,
    Depth,
    SampleMask,
    Stencil,
    // ---- Prefix ----
    FunctionConstant,
    // ---- Attributes ----
    LocationIndex,
    BufferSize,
    Read,
    Write,
    ReadWrite,
    Sample,
    AddressSpace,
    StructTypeInfo,
    ArgTypeSize,
    ArgTypeAlignSize,
    ArgTypeName,
    ArgName,
    ArgUnused,
    RasterOrderGroup,
    Center,
    Centroid,
    Perspective,
    NoPerspective,
    Flat,
    ImageblockDataSize,
    ImageblockMaster,
    AliasImplicitImageblock,
    Shared,
    Invariant,
    ClipDistanceArraySize,
    DepthQualifier,
    // ---- Depth qualifier values ----
    Any,
    Less,
    Greater,
    // ---- Mesh output types (`air.mesh_type_info`) ----
    MeshTypeInfo,
    MeshVertexData,
    MeshPrimitiveData,
    PrimitiveCulled,
    Point,
    Line,
    Triangle,
    // ---- Struct-member suffixes and control-point nodes ----
    IndirectArgument,
    PatchControlPointFunction,
    // ---- Visible-function declarations (`!air.visible`) ----
    VisibleInput,
    VisiblePadding,
    VisibleOutput,
    // ---- Core Image kernel declarations (`!air.ci`) ----
    CiBuiltin,
    CiMatrix,
    CiTexture,
    CiSampler,
    CiPointer,
    LocationCount,
}

/// The access keys a texture role takes, one of which it requires: the error location of a
/// texture without one.
pub(super) const TEXTURE_ACCESS: &str = "air.sample|air.read|air.write|air.read_write";

/// The access keys a buffer (or Core Image pointer) role takes, one of which it requires: the
/// error location of one without it.
pub(super) const BUFFER_ACCESS: &str = "air.read|air.write|air.read_write";

/// The spelling of every key.
const KEYS: &[(&str, Key)] = &[
    ("air.buffer", Key::Buffer),
    ("air.payload", Key::Payload),
    ("air.mesh_grid_properties", Key::MeshGridProperties),
    ("air.mesh", Key::Mesh),
    ("air.mesh_type_info", Key::MeshTypeInfo),
    ("air.mesh_vertex_data", Key::MeshVertexData),
    ("air.mesh_primitive_data", Key::MeshPrimitiveData),
    ("air.primitive_culled", Key::PrimitiveCulled),
    ("air.point", Key::Point),
    ("air.line", Key::Line),
    ("air.triangle", Key::Triangle),
    ("air.indirect_buffer", Key::IndirectBuffer),
    ("air.texture", Key::Texture),
    ("air.sampler", Key::Sampler),
    (
        "air.instance_acceleration_structure",
        Key::InstanceAccelerationStructure,
    ),
    (
        "air.primitive_acceleration_structure",
        Key::PrimitiveAccelerationStructure,
    ),
    ("air.visible_function_table", Key::VisibleFunctionTable),
    (
        "air.intersection_function_table",
        Key::IntersectionFunctionTable,
    ),
    ("air.indirect_constant", Key::IndirectConstant),
    ("air.command_buffer", Key::CommandBuffer),
    ("air.compute_pipeline_state", Key::ComputePipelineState),
    ("air.render_pipeline_state", Key::RenderPipelineState),
    ("air.depth_stencil_state", Key::DepthStencilState),
    ("air.imageblock", Key::Imageblock),
    ("air.imageblock_data", Key::ImageblockData),
    ("air.stage_in", Key::StageIn),
    ("air.vertex_input", Key::VertexInput),
    ("air.fragment_input", Key::FragmentInput),
    ("air.patch_input", Key::PatchInput),
    ("air.patch_control_point_input", Key::PatchControlPointInput),
    ("air.thread_position_in_grid", Key::ThreadPositionInGrid),
    (
        "air.thread_position_in_threadgroup",
        Key::ThreadPositionInThreadgroup,
    ),
    (
        "air.threadgroup_position_in_grid",
        Key::ThreadgroupPositionInGrid,
    ),
    ("air.threads_per_threadgroup", Key::ThreadsPerThreadgroup),
    ("air.threads_per_grid", Key::ThreadsPerGrid),
    ("air.threadgroups_per_grid", Key::ThreadgroupsPerGrid),
    (
        "air.thread_index_in_threadgroup",
        Key::ThreadIndexInThreadgroup,
    ),
    ("air.thread_index_in_simdgroup", Key::ThreadIndexInSimdgroup),
    ("air.thread_index_in_quadgroup", Key::ThreadIndexInQuadgroup),
    (
        "air.simdgroup_index_in_threadgroup",
        Key::SimdgroupIndexInThreadgroup,
    ),
    (
        "air.quadgroup_index_in_threadgroup",
        Key::QuadgroupIndexInThreadgroup,
    ),
    ("air.threads_per_simdgroup", Key::ThreadsPerSimdgroup),
    (
        "air.simdgroups_per_threadgroup",
        Key::SimdgroupsPerThreadgroup,
    ),
    (
        "air.quadgroups_per_threadgroup",
        Key::QuadgroupsPerThreadgroup,
    ),
    (
        "air.dispatch_threads_per_threadgroup",
        Key::DispatchThreadsPerThreadgroup,
    ),
    (
        "air.dispatch_simdgroups_per_threadgroup",
        Key::DispatchSimdgroupsPerThreadgroup,
    ),
    ("air.stage_in_grid_size", Key::StageInGridSize),
    ("air.stage_in_grid_origin", Key::StageInGridOrigin),
    ("air.vertex_id", Key::VertexId),
    ("air.instance_id", Key::InstanceId),
    ("air.base_vertex", Key::BaseVertex),
    ("air.base_instance", Key::BaseInstance),
    ("air.amplification_id", Key::AmplificationId),
    ("air.amplification_count", Key::AmplificationCount),
    ("air.position_in_patch", Key::PositionInPatch),
    ("air.patch_id", Key::PatchId),
    ("air.front_facing", Key::FrontFacing),
    ("air.point_coord", Key::PointCoord),
    ("air.primitive_id", Key::PrimitiveId),
    ("air.sample_id", Key::SampleId),
    ("air.sample_mask_in", Key::SampleMaskIn),
    ("air.barycentric_coord", Key::BarycentricCoord),
    ("air.position", Key::Position),
    ("air.render_target", Key::RenderTarget),
    ("air.render_target_array_index", Key::RenderTargetArrayIndex),
    ("air.viewport_array_index", Key::ViewportArrayIndex),
    ("air.vertex_output", Key::VertexOutput),
    ("air.point_size", Key::PointSize),
    ("air.clip_distance", Key::ClipDistance),
    ("air.depth", Key::Depth),
    ("air.sample_mask", Key::SampleMask),
    ("air.stencil", Key::Stencil),
    ("air.function_constant", Key::FunctionConstant),
    ("air.location_index", Key::LocationIndex),
    ("air.buffer_size", Key::BufferSize),
    ("air.read", Key::Read),
    ("air.write", Key::Write),
    ("air.read_write", Key::ReadWrite),
    ("air.sample", Key::Sample),
    ("air.address_space", Key::AddressSpace),
    ("air.struct_type_info", Key::StructTypeInfo),
    ("air.arg_type_size", Key::ArgTypeSize),
    ("air.arg_type_align_size", Key::ArgTypeAlignSize),
    ("air.arg_type_name", Key::ArgTypeName),
    ("air.arg_name", Key::ArgName),
    ("air.arg_unused", Key::ArgUnused),
    ("air.raster_order_group", Key::RasterOrderGroup),
    ("air.center", Key::Center),
    ("air.centroid", Key::Centroid),
    ("air.perspective", Key::Perspective),
    ("air.no_perspective", Key::NoPerspective),
    ("air.flat", Key::Flat),
    ("air.imageblock_data_size", Key::ImageblockDataSize),
    ("air.imageblock_master", Key::ImageblockMaster),
    (
        "air.alias_implicit_imageblock",
        Key::AliasImplicitImageblock,
    ),
    ("air.shared", Key::Shared),
    ("air.invariant", Key::Invariant),
    ("air.clip_distance_array_size", Key::ClipDistanceArraySize),
    ("air.depth_qualifier", Key::DepthQualifier),
    ("air.any", Key::Any),
    ("air.less", Key::Less),
    ("air.greater", Key::Greater),
    ("air.indirect_argument", Key::IndirectArgument),
    (
        "air.patch_control_point_function",
        Key::PatchControlPointFunction,
    ),
    ("air.visible_input", Key::VisibleInput),
    ("air.visible_padding", Key::VisiblePadding),
    ("air.visible_output", Key::VisibleOutput),
    ("air.ci_builtin", Key::CiBuiltin),
    ("air.ci_matrix", Key::CiMatrix),
    ("air.ci_texture", Key::CiTexture),
    ("air.ci_sampler", Key::CiSampler),
    ("air.ci_pointer", Key::CiPointer),
    ("air.location_count", Key::LocationCount),
];

impl Key {
    /// The key spelled `bytes`, if it is one.
    pub(super) fn parse(bytes: &[u8]) -> Option<Key> {
        KEYS.iter()
            .find(|(s, _)| s.as_bytes() == bytes)
            .map(|&(_, k)| k)
    }

    /// The key's spelling, for error locations.
    pub(super) fn name(self) -> &'static str {
        KEYS.iter()
            .find(|&&(_, k)| k == self)
            .map_or("air.?", |&(s, _)| s)
    }

    /// Every key, for tests.
    #[cfg(test)]
    pub(super) fn all() -> impl Iterator<Item = (&'static str, Key)> {
        KEYS.iter().copied()
    }
}
