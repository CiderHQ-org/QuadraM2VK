//! The binder: from what the guest bound to what the consumer records, as plain data.
//!
//! The binder never calls Vulkan. The consumer describes what is bound for one draw or dispatch,
//! keyed by Metal slot, in a [`BindState`]: per [`Stage`](crate::pipeline::Stage) a [`StageTable`]
//! of [`BoundBuffer`]s, [`BoundTexture`]s, [`BoundSampler`]s, and threadgroup memory lengths, the
//! resources made resident with `useResource` or `useHeap` ([`UsedResource`]), the live sampler
//! states an argument buffer can name ([`ArgumentSampler`]), and encoder state
//! such as the compute encoder's stage-in region ([`StageInRegion`]) and the render encoder's
//! tessellation factor buffer ([`TessellationFactors`]). Binding then runs in two levels:
//!
//! 1. **Plan, per shape.** [`BindState::shape`] keeps what can change a plan ([`BindShape`]:
//!    which slots are bound with what kind of resource, formats, sampler modes, alignment
//!    classes, and the like). `ComputePipeline::plan` and `RenderPipeline::plan` turn a pipeline
//!    and a shape into a [`BindingPlan`]: one [`DescriptorWrite`] per layout binding (the resource
//!    bound at a slot, a buffer the binder synthesizes, or a [`NeutralResource`] where nothing
//!    compatible is bound), the vertex buffers, the [`ResourceUse`]s that drive staging,
//!    writeback, and barriers, and where push fields go. A plan is a pure function of the
//!    pipeline, the device, and the shape: memoize it on `(PipelineKey, BindShape)`.
//! 2. **Instantiate, per command.** [`BindingPlan::instantiate_dispatch`] of a [`Dispatch`] gives
//!    [`DispatchCommands`], and [`BindingPlan::instantiate_draw`] of a [`DrawCall`] gives
//!    [`DrawCommands`]: the range to bind each buffer descriptor over ([`DescriptorRange`]:
//!    exactly that range, never more), the bytes to upload for argument and handle tables
//!    ([`DescriptorBytes`]) and for the resident root table ([`RootTable`]), the push-constant
//!    bytes with the device addresses the consumer fills in ([`PushHole`]), and the Vulkan
//!    dispatches or draw to record. A Metal dispatch can become several Vulkan dispatches
//!    ([`DispatchRegion`]), each with its complete specialization constants, so the consumer
//!    creates one `VkPipeline` per distinct specialization.
//!
//! A plan whose [`variant`](BindingPlan::variant) is not empty asks for another variant of the
//! pipeline: compile that variant (see [`pipeline`](crate::pipeline)) and plan it for the same
//! shape. Refusals are [`BindError`](crate::error::BindError)s.
//!
//! Per device, create every [`NeutralResource`] of
//! [`Device::neutral_kinds`](crate::device::Device::neutral_kinds) once. Per command buffer, bind
//! one [`ExecutionStatus`] buffer: initialize it with [`ExecutionStatus::initial`], and read it
//! back with [`ExecutionStatus::decode`], which reports clamped indirect counts and resident-set
//! misses. A command reaches it by its device address where the device addresses buffers, and
//! through the binding [`WriteSource::ExecutionStatus`] names where it does not. An indirect
//! dispatch can carry an [`IndirectClamp`] prepass of [`indirect_clamp_module`] to record first.
//! An indexed indirect draw whose index fetch
//! nothing bounds (`DrawCommands::index_bounds_pass`) can bound it on the GPU instead of reading
//! its arguments: an [`IndexBoundsPass`] of [`index_bounds_plan_module`] and
//! [`index_bounds_fill_module`] before the render pass instance, and two indirect draws. An
//! indexed draw whose primitives Vulkan's input assembly does not form as Metal does (a strip
//! restarting at another index than all-ones) can draw them as a list instead: an
//! [`IndexRewritePass`] of three [`index_rewrite_module`] dispatches ([`RewritePhase`]) before the
//! render pass instance, and one indirect draw. A pass's `Boolean` visibility results are query
//! counts copied to their offsets and made Metal's 1 or 0 in place by a [`VisibilityResolve`]
//! dispatch of [`visibility_resolve_module`]; where a split pass continues a result's query in a
//! later render pass instance, a [`VisibilityCombine`] dispatch of [`visibility_combine_module`]
//! first ORs the continuations' counts into it. A pass whose fetching draws read a multisampled
//! attachment's sample planes (`StorageAttachmentView::SamplePlanes`) splits the attachment into
//! them with [`planes_split_module`] and merges them back with [`planes_merge_modules`], through
//! the [`PlanesCopy`] view of its texel size. A multisampled depth or stencil aspect moves
//! between images without colour copies through the draws of [`depth_stencil_copy_modules`] and
//! the kernel of [`stencil_split_module`], and resolves where the device's resolve modes do not
//! state Metal's filter through the draws of [`depth_stencil_resolve_modules`]. The mipmaps of the
//! colour formats Metal's filter computes are its answer through the [`MipmapFilterChain`]
//! dispatches of [`mipmap_filter_module`]. Compute commands
//! a kernel encodes into an indirect command buffer replay on the device where it has device
//! addresses: the [`replay_preprocess`] of an execution ([`ReplayExecution`]) over registered
//! [`ReplayVariant`]s, recorded with [`replay_preprocess_module`], writes each command's arguments
//! and slot table for the variants' dispatches ([`BindingPlan::instantiate_replay`]);
//! [`ReplayFaults`] reports the commands it refused. Where the device states
//! `VK_EXT_device_generated_commands` with compute pipeline binding,
//! [`replay_preprocess_generated`] over each variant's [`GeneratedVariant`] writes a token stream
//! for one execution per segment instead ([`GeneratedLayout`]). An indirect dispatch of a kernel
//! that does not observe its threadgroup partition runs on a flat variant through
//! [`flat_dispatch_module`], which reads its arguments on the device ([`FlatIndirect`]). Where one
//! command writes a buffer's bytes through a texture copied in and out of them and also another
//! way, [`shadow_merge_module`] merges the bytes the texture changed into the buffer's in place of
//! the copy back ([`shadow_merge_image_module`] reading the shadow's texels through its
//! unsigned-integer view, where it has one, and [`shadow_merge_kept_image_module`] reading the
//! kept texels from an image copy of the shadow too), and where a device copies a shadow back faster by a
//! dispatch, [`shadow_copy_back_module`] stores its texels' words at their rows, and
//! [`shadow_copy_in_module`] copies a merged texture's texels in and keeps them with one dispatch.
//! A copy of many
//! equal rows between two buffers is one dispatch of [`rows_copy_module`] where a copy's region
//! per row costs more. A render pass that loads a
//! 32-bit float depth holding a value Vulkan's depth test compares otherwise than Metal's (a
//! negative value or a NaN, whose magnitude Metal compares) renders over its stand-in, written
//! before the pass and merged back after it by [`depth_magnitude_module`]. A blit between a buffer
//! and a two-plane extended-range texture moves its color words through [`plane_words_module`],
//! and a blit between two such textures moves both planes as Metal's blit does through
//! [`plane_copy_module`]. A rotated texture's texels move between its
//! pass's frame and its store through [`rotate_texels_module`], and a heap's two-plane
//! extended-range texture's texels between its image and its two planes in the heap's bytes through
//! [`two_plane_texels_module`]; a blit into such a texture of a heap writes the heap's bytes at
//! Metal's rows through [`plane_bytes_copy_module`]. A heap texture over host pages keeps its
//! elements in the heap's bytes in Metal's tiled layout, moved between its image and those bytes
//! through [`tiled_texels_module`]. A render pass that loads an extended-range, padded 10-bit or
//! bfloat color attachment and draws first rewrites the bits its codes leave unused through
//! [`unused_bits_module`] of its format, in the 32 x 32 tiles a mask names, as Metal's tile store
//! does; under the exact tile store it also raises a signed normalized attachment's minimum codes
//! and normalizes an `RGB9E5Float` attachment's words there. The tiles a pass's draws reach, which Metal's tile store writes back, are the mask one
//! dispatch of [`tile_reach_module`] bins its captured primitives into: [`capture_assembly_module`]
//! assembles them from a pass's capture draws' records, and [`capture_settle_module`] widens the
//! mask to every tile where the binning counted a primitive it cannot answer. A GPU blit that reads
//! or writes a texture Metal keeps losslessly compressed writes the texels Metal's compression
//! leaves of the formats whose bytes it changes ([`CompressedBlit`]), which
//! [`compressed_blit_module`] of the rule rewrites after the copy. A BC1 to BC5 texture a device
//! decodes otherwise than Metal is read through a twin of its texels as Metal decodes them
//! ([`BlockDecode`]), which [`block_decode_module`] writes from the blocks each write left; an
//! ASTC LDR texture on a device without its block through a twin of Metal's decode of its blocks
//! ([`AstcDecode`], in the texels [`AstcTexels`] names), which [`astc_decode_module`] writes; a
//! PVRTC texture on a device without its format through a twin of its texels' codes
//! ([`PvrtcDecode`]), which [`pvrtc_decode_module`] writes; an ETC2 or EAC texture on a device
//! without the formats through a twin of its codes or an 11-bit format's values ([`EtcDecode`], in
//! the texels [`EtcTexels`] names), which [`etc_decode_module`] writes; a 4:2:2 texture through a
//! twin of its texels' codes ([`ChromaDecode`]), which [`chroma_decode_module`] writes.
//! [`sampler_desc`] gives the Vulkan sampler a plan's sampler write needs, [`heap_classes`] the
//! classes of the handle heap's arrays on a profile, [`FunctionTableWords`] the words of a
//! visible-function table's buffer, and [`FunctionTableSnapshot`] the one buffer of a command's
//! tables' words on a device without the function-table array's features. A plan writes an
//! acceleration structure as the top-level structure the consumer builds for it:
//! [`instance_record`] gives each Metal instance's record, [`primitive_wrap`] the one instance over
//! a bound primitive structure, and [`geometry_flags`] and [`geometry_table_offset`] a primitive
//! structure's geometries. A [`RasterizationRateMap`] is the consumer's `MTLRasterizationRateMap`:
//! it compiles a [`RateMapDescriptor`], answers the map's queries, and gives the parameter data a
//! `rasterization_rate_map_decoder` reads, and plans a pass rendered through one region by region
//! ([`RateMapRegions`]). A [`SparseTileMap`] mirrors a sparse texture's tile mappings where the
//! device has no shader residency: a plan binds its bytes where a function reads the texture's
//! residency. An [`ArgumentBufferLayout`] says where an argument encoder (`MTLArgumentEncoder`)
//! writes each argument of an argument buffer: a function's (`Function::argument_buffer_layout`),
//! or one of [`ArgumentDescriptor`]s ([`ArgumentBufferLayout::from_descriptors`]).
//!
//! The frozen surface is the items named at this root; the submodules are crate-private.

pub(crate) mod acceleration;
pub(crate) mod argument_buffer;
pub(crate) mod astc_decode;
pub(crate) mod block_decode;
pub(crate) mod capture_assembly;
pub(crate) mod chroma_decode;
pub(crate) mod color_resolve;
pub(crate) mod compressed_blit;
pub(crate) mod depth_magnitude;
pub(crate) mod depth_stencil_copy;
pub(crate) mod depth_stencil_resolve;
pub(crate) mod dispatch;
pub(crate) mod etc_decode;
pub(crate) mod extended_shadow;
pub(crate) mod graphics;
pub(crate) mod heap;
pub(crate) mod index_bounds;
pub(crate) mod index_rewrite;
pub(crate) mod indirect_clamp;
pub mod indirect_commands;
pub(crate) mod instantiate;
pub(crate) mod mesh_scan;
pub(crate) mod mipmap_filter;
pub(crate) mod neutral;
pub(crate) mod plan;
pub(crate) mod plane_bytes_copy;
pub(crate) mod plane_copy;
pub(crate) mod plane_words;
pub(crate) mod pvrtc_decode;
pub(crate) mod rate_map;
pub(crate) mod replay;
pub(crate) mod resident;
pub(crate) mod rotate_texels;
pub(crate) mod rows_copy;
pub(crate) mod sample_interleave;
pub(crate) mod sample_planes;
pub(crate) mod shadow_copy_back;
pub(crate) mod shadow_copy_in;
pub(crate) mod shadow_merge;
pub(crate) mod shadow_merge_image;
pub(crate) mod state;
pub(crate) mod status;
pub(crate) mod table;
pub(crate) mod tessellation_scan;
pub(crate) mod tile_map;
pub(crate) mod tile_reach;
pub(crate) mod tiled_texels;
pub(crate) mod two_plane_texels;
pub(crate) mod unused_bits;
pub(crate) mod visibility_combine;
pub(crate) mod visibility_resolve;
pub(crate) mod ycbcr_convert;
pub(crate) mod ycbcr_write;

pub use acceleration::{
    geometry_flags, geometry_table_offset, instance_record, primitive_wrap, InstanceRecord,
    MetalInstance, GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION, GEOMETRY_OPAQUE,
};
pub use argument_buffer::{
    ArgumentBufferLayout, ArgumentDescriptor, ArgumentLayoutError, ArgumentMember, ArgumentPlace,
    ArgumentType, ARGUMENT_HANDLE_BYTES,
};
pub use astc_decode::{astc_decode_module, AstcDecode, AstcTexels};
pub use block_decode::{block_decode_module, BlockDecode, BlockDecodeModule};
pub use capture_assembly::{
    capture_assembly_module, capture_settle_module, CaptureAssemblyModule, CaptureSettleModule,
};
pub use chroma_decode::{chroma_decode_module, ChromaDecode};
pub use color_resolve::{
    color_resolve_modules, ColorResolveFormat, ColorResolveModules, ResolveChannel, ResolveRule,
    ResolveSource, SRGB_RESOLVE_LINEAR, SRGB_RESOLVE_THRESHOLDS,
};
pub use compressed_blit::{compressed_blit_module, CompressedBlit, CompressedBlitModule};
pub use etc_decode::{etc_decode_module, EtcDecode, EtcTexels};
// The bytes of `WriteSource::ColorWriteTable`'s buffer.
pub use crate::lower::arith::srgb::color_write_table;
pub use depth_magnitude::{depth_magnitude_module, DepthMagnitudeModule};
pub use depth_stencil_copy::{
    depth_split_module, depth_stencil_copy_modules, stencil_split_module, CopySource,
    DepthSplitModule, DepthStandInForm, DepthStencilCopy, DepthStencilCopyModules, StencilBit,
    StencilSplitModule,
};
pub use depth_stencil_resolve::{
    depth_stencil_resolve_modules, DepthPick, DepthStencilResolveDraw, DepthStencilResolveModules,
    ResolveWrite,
};
pub use dispatch::{Dispatch, DispatchCommands, DispatchRegion, IndirectArgs};
pub use extended_shadow::{extended_shadow_module, ExtendedShadowModule};
pub use graphics::{
    DrawCall, DrawCommands, IndexOverrun, MeshDispatch, MeshGroups, MeshRecordsRegion,
    TessellationDispatch, TessellationRecords,
};
pub use heap::{
    handle_entry, handle_entry_flags, heap_classes, heap_element, heap_raw_form,
    heap_write_nans_converted, HeapElement, HeapElements, HeapSource, HANDLE_ENTRY_BYTES,
    HANDLE_TABLE_CAPACITY, NEUTRAL_HEAP_ELEMENT,
};
pub use index_bounds::{
    index_bounds_fill_module, index_bounds_plan_module, IndexBoundsFillModule, IndexBoundsPass,
    IndexBoundsPlanModule,
};
pub use index_rewrite::{
    index_rewrite_module, IndexRewriteModule, IndexRewritePass, RewriteOutput, RewritePhase,
    RewriteProgram, RewriteSource,
};
pub use indirect_clamp::{indirect_clamp_module, IndirectClamp, IndirectClampModule};
pub use instantiate::{
    BufferExtent, BufferRange, DescriptorBytes, DescriptorRange, IndexBufferRef, IndirectRef,
    PushConstantWrite, PushHole, ResidentRange, ResidentSet, RootTable,
};
pub use mesh_scan::{mesh_scan_module, MeshScanModule};
pub use mipmap_filter::{
    mipmap_filter_module, MipmapFilterChain, MipmapFilterKernel, MipmapFilterLevel,
    MipmapFilterModule, MipmapRule,
};
pub use neutral::NeutralResource;
pub use plan::resources::sampler_desc;
pub use plan::{
    AccessClass, BindingPlan, ByteRange, ComputeTemplate, DescriptorWrite, FieldLocation,
    GraphicsTemplate, ImageSource, ImageViewDesc, PlanDiagnostic, PushFieldPlacement, PushTemplate,
    ResourceUse, SamplerSource, UseTarget, VertexBufferBind, ViewSetup, WriteSource,
};
pub use rate_map::{
    RasterizationRateMap, RateMapAxis, RateMapDescriptor, RateMapError, RateMapLayerDescriptor,
    RateMapRegion, RateMapRegions, RegionError, RegionLimits, UnpinnedCompile, MAX_SCREEN_EXTENT,
};
pub use replay::{
    flat_dispatch_module, replay_preprocess, replay_preprocess_generated, replay_preprocess_module,
    FlatArguments, FlatDispatchModule, FlatIndirect, GeneratedLayout, GeneratedRefusal,
    GeneratedReplay, GeneratedVariant, ReplayExecution, ReplayLayout, ReplayPreprocess,
    ReplayPreprocessModule, ReplayRefusal, ReplayVariant, MAX_REPLAY_DISPATCHES,
    MAX_REPLAY_REGIONS, MAX_REPLAY_SLOTS, MAX_REPLAY_VARIANTS,
};
pub use replay::{
    predication_step_module, PredicationCompare, PredicationOp, PredicationStep,
    PredicationStepModule,
};
pub use state::{
    ArgumentFunctionTable, ArgumentSampler, ArgumentSamplersShape, BindShape, BindState,
    BoundAccelerationStructure, BoundBuffer, BoundSampler, BoundSamplerShape, BoundTexture,
    BufferShape, CompensatedViewport, FragmentTests, FunctionTablesShape, HeapPlacement,
    ImageblockSize, ResourceClass, SamplerDescriptor, SamplerShape, StageInRegion, StageShape,
    StageTable, TessellationFactorBuffer, TessellationFactors, TextureFacts, TextureShape,
    TileThreadgroupArgument, TileThreadgroupMemory, UsedClass, UsedResource, UsedShape,
    VertexAmplification, ViewMapping,
};
// How `BindState::shape` classes a buffer: the crate reads it from `state`, and a consumer takes
// shapes from `BindState::shape`; only the crate's tests and validation reach it here.
pub use plane_bytes_copy::{
    plane_bytes_copy_module, PlaneBytes, PlaneBytesCopyModule, PlaneSource,
};
pub use plane_copy::{plane_copy_module, PlaneCopyModule, PlaneRows};
pub use plane_words::{plane_words_module, PlaneWordsModule};
pub use pvrtc_decode::{pvrtc_decode_module, PvrtcDecode};
pub use rotate_texels::{rotate_texels_module, RotateTexelsModule};
pub use rows_copy::{rows_copy_module, RowsCopyModule};
pub use sample_interleave::{
    interleave_merge_modules, interleave_split_module, InterleaveMergeModules,
    InterleaveSplitModule,
};
pub use sample_planes::{
    planes_merge_modules, planes_split_module, PlanesCopy, PlanesMergeModules, PlanesSplitModule,
};
pub use shadow_copy_back::{shadow_copy_back_module, ShadowCopyBackModule, ShadowTexelView};
pub use shadow_copy_in::{shadow_copy_in_module, ShadowCopyInModule};
pub use shadow_merge::{shadow_merge_module, ShadowMergeModule};
pub use shadow_merge_image::{
    shadow_merge_image_module, shadow_merge_kept_image_module, ShadowMergeImageModule,
};
pub use status::{
    ExecutionStatus, MeshRecordsFault, ReplayFaults, StatusBits, StatusReport, StatusWord,
    TessellationRecordsFault,
};
pub use table::{FunctionTableSnapshot, FunctionTableWords};
pub use tessellation_scan::{
    tessellation_scan_module, TessellationScanKernel, TessellationScanModule,
};
pub use tile_map::{SparseTileMap, TileMapError, TileMapping};
pub use tile_reach::{tile_reach_module, TileReachModule};
pub use tiled_texels::{tiled_texels_module, TiledLevelMove, TiledTexelsModule};
pub use two_plane_texels::{two_plane_texels_module, PlaneLayout, TwoPlaneTexelsModule};
pub use unused_bits::{unused_bits_module, TexelWords, UnusedBitsModule};
pub use visibility_combine::{
    visibility_combine_module, ContinuedResult, VisibilityCombine, VisibilityCombineModule,
};
pub use visibility_resolve::{
    visibility_resolve_module, VisibilityResolve, VisibilityResolveModule,
};
pub use ycbcr_convert::{
    ycbcr_convert_module, ycbcr_read, ycbcr_texel, YCbCrCodes, YCbCrConvertModule, YCbCrMatrix,
    YCbCrOutput, YCbCrPlanes, YCbCrRange,
};
pub use ycbcr_write::{ycbcr_store, ycbcr_write, ycbcr_write_module, YCbCrWriteModule};
