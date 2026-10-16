use super::*;

/// Role of a single compute-kernel entry parameter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KernRole {
    /// AIR `buffer` argument. `KernMeta::buffer_address_space` distinguishes device/constant
    /// resource buffers from threadgroup scratch buffers.
    Buffer(u32),
    /// `[[texture(n)]]` -> UniformConstant sampled image.
    Texture(u32),
    /// `[[sampler(n)]]` -> UniformConstant sampler.
    Sampler(u32),
    /// Host-populated StorageBuffer shadow for an opaque Metal acceleration structure used by AIR
    /// introspection intrinsics. Bound at the resource's `air.location_index`.
    AccelerationStructureShadow(u32),
    /// An AIR primitive acceleration structure that is not consumed by an AIR intersection
    /// intrinsic. Metal still binds the native object; Vulkan needs no descriptor.
    PrimitiveAccelerationStructure(u32),
    /// An AIR primitive acceleration structure consumed by AIR intersection lowering. Metal binds
    /// the native object and Vulkan exposes authored triangle geometry through a StorageBuffer.
    PrimitiveAccelerationStructureShadow(u32),
    /// A Metal visible-function table. Logical SPIR-V has no descriptor for the opaque table;
    /// authored linking resolves its entries before ordinary interface lowering.
    VisibleFunctionTable(u32),
    /// A Metal intersection-function table. Like visible tables, this is a link-time authored
    /// resource rather than a Vulkan descriptor.
    IntersectionFunctionTable(u32),
    /// `[[threads_per_threadgroup]]` (`uint` or `uint3`) -> the size of the threadgroup this thread
    /// actually runs in. Scalar params receive component .x.
    ///
    /// This is the EXECUTING size, which a decomposed exact-thread dispatch makes smaller in a tail
    /// region: `KernelDispatchPlan` gives each region its own `LocalSize`, so the fact follows the
    /// region's specialized local size rather than the one the dispatch asked for.
    ThreadsPerThreadgroup,
    /// `[[dispatch_threads_per_threadgroup]]` (`uint` or `uint3`) -> the threadgroup size the
    /// dispatch ASKED FOR, which a short final threadgroup does not change.
    ///
    /// It is a separate role from [`KernRole::ThreadsPerThreadgroup`] because the two differ exactly
    /// where this translator emulates `dispatchThreads:`. A ten-thread grid in groups of four gives
    /// Metal `threads_per_threadgroup == 2` and `dispatch_threads_per_threadgroup == 4` in the last
    /// group, and `KernelDispatchPlan` reproduces that split by specializing the tail region's local
    /// size to two -- so aliasing the two roles reports the tail size for both. The requested size is
    /// `KernelOptions::local_size`, which is the same value the plan decomposes.
    DispatchThreadsPerThreadgroup,
    /// `[[thread_position_in_threadgroup]]` (`uint` or `uint3`) -> LocalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInThreadgroup,
    /// `[[threadgroups_per_grid]]` (`uint` or `uint3`) -> NumWorkgroups.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupsPerGrid,
    /// `[[threads_per_grid]]` (`uint` or `uint3`) -> the selected kernel dispatch grid. Whole-
    /// workgroup dispatches derive it as NumWorkgroups * LocalSize; exact-thread dispatches read
    /// the complete logical grid from the region payload.
    ThreadsPerGrid,
    /// `[[threadgroup_position_in_grid]]` (`uint` or `uint3`) -> WorkgroupId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadgroupPositionInGrid,
    /// `[[thread_index_in_threadgroup]]` (`uint`) -> LocalInvocationIndex.
    ThreadIndexInThreadgroup,
    /// One fact about the AIR execution group this thread runs in, over a group `lanes` threads
    /// wide. See [`ExecutionGroupFact`] for the requested quantity.
    ExecutionGroup {
        fact: ExecutionGroupFact,
        lanes: u32,
    },
    /// `[[thread_position_in_grid]]` (`uint` or `uint3`) -> GlobalInvocationId.
    /// Scalar params receive component .x; vector params receive the full v3uint.
    ThreadPositionInGrid,
    /// Kernel `[[stage_in]]` attribute data. Metal feeds this through a stage-input descriptor keyed
    /// by `air.location_index`; Vulkan lowering needs an explicit per-invocation data ABI and must
    /// not silently bind it to zero.
    StageInput(u32),
    /// A texture argument the variant this module describes declares no slot for, so no descriptor
    /// exists for it. Metal reads zero through such a resource and stores nowhere; naming the role
    /// is what lets the lowering tell that operand apart from a handle it merely lost track of.
    VariantAbsentTexture,
    Other,
}

/// A compute kernel's decoded parameter roles. The body has no return value (a `void` kernel that
/// writes through its buffer pointers), so no output handling is needed.
#[derive(Clone, Debug, Default)]
pub struct KernMeta {
    pub roles: Vec<(u32, KernRole)>,
    /// `(parameter index, AIR role)` for every enabled entry parameter whose role has no
    /// lowering. See [`FragMeta::unmodelled_input_params`].
    pub unmodelled_input_params: Vec<(u32, String)>,
    /// Parameter indices of explicit imageblocks whose AIR node carries
    /// `air.alias_implicit_imageblock`.
    ///
    /// An explicit imageblock is ordinarily tile-local scratch, and the emitter gives it Private
    /// storage. This marker says the opposite: the storage *is* the implicit imageblock -- the
    /// render targets the rasterizer already wrote -- so the kernel's first read is of framebuffer
    /// content, and its writes have to land back there. Private scratch is neither, so the marker
    /// cannot be dropped.
    pub aliased_implicit_imageblock_params: Vec<u32>,
    /// The render-target planes an aliased explicit imageblock's members resolve to, keyed by the
    /// same parameter index. One derivation, read by reflection (which must declare a descriptor
    /// for every plane) and by the emitter (which must fill the cell from them and write it back).
    pub aliased_implicit_imageblock_planes: HashMap<u32, Vec<AliasedImageblockPlane>>,
    /// Every attribute on the `!air.kernel` root this stage has no model for, as it reads in AIR.
    /// Mirrors [`FragMeta::unmodelled_stage_attributes`].
    pub unmodelled_stage_attributes: Vec<String>,
    /// `[[max_total_threads_per_threadgroup(N)]]`: the largest threadgroup the entry was compiled
    /// to run. A dispatch wider than this is outside what the AIR body was built for, so the
    /// requested `LocalSize` is checked against it rather than emitted unread.
    pub max_work_group_size: Option<u32>,
    /// Function-constant-wrapped buffer parameter index -> Metal buffer location. Multiple mutually
    /// exclusive typed alternatives may intentionally share one location.
    pub function_constant_buffer_locations: HashMap<u32, u32>,
    /// `param_idx -> reconstructed struct layout` for buffer args (see `FragMeta::buffer_layouts`).
    pub buffer_layouts: HashMap<u32, AirType>,
    /// `param_idx -> reconstructed air.imageblock_data layout` for imageblock args.
    pub imageblock_layouts: HashMap<u32, AirType>,
    /// Descriptor-backed render-target planes used by implicit imageblock load/store intrinsics.
    pub implicit_imageblock_attachments: Vec<ImplicitImageblockAttachment>,
    /// `param_idx -> AIR address space` for buffer args. Address space 3 is threadgroup memory.
    pub buffer_address_spaces: HashMap<u32, u32>,
    /// `param_idx -> declared AIR buffer argument byte size`, from `air.arg_type_size` or
    /// `air.buffer_size` when present.
    pub buffer_type_sizes: HashMap<u32, u32>,
    /// `param_idx -> air.buffer_size` when AIR declares one exact reference-object extent.
    pub buffer_object_sizes: HashMap<u32, u32>,
    /// `param_idx -> AIR buffer argument type name`, e.g. `char`, `void`, or a struct name.
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
    /// `param_idx -> AIR kernel stage-input scalar/vector type name`.
    pub stage_input_type_names: HashMap<u32, String>,
    /// Textures EMBEDDED inside an `air.indirect_buffer` argument buffer (via `air.indirect_argument`
    /// → nested `air.texture`) that the kernel body reads/writes with AIR texture intrinsics. Each is
    /// surfaced as a standalone image resource so the read/write lowers to a real descriptor instead of a
    /// private placeholder. See [`EmbeddedTexture`].
    pub embedded_textures: Vec<EmbeddedTexture>,
    /// Every resource-handle member carried by an `air.indirect_buffer`, including handles whose
    /// concrete kind is supplied by an authored manifest and verified from its AIR use.
    pub embedded_arguments: Vec<EmbeddedArgument>,
    /// What an object or mesh entry declares beyond a kernel's parameters (empty for a kernel).
    pub mesh_stage: MeshStageMeta,
    /// Resource handles an argument buffer declares inside a nested `air.struct_type_info` member,
    /// which the embedded-argument walk does not surface. Non-empty means every texture operation on
    /// an unrecovered handle has to refuse rather than take the "resource is absent" path -- see
    /// `embedded::unsurfaced_embedded_resources`.
    pub unsurfaced_embedded_resources: Vec<String>,
}

impl KernRole {
    /// The Metal BUFFER-TABLE slot this role occupies, or `None` for a role that occupies none.
    ///
    /// This is the set [`KernMeta::stage_input_bindings`] must not allocate a synthetic slot on top
    /// of. Written as an exhaustive match rather than a list with a `_ => None` tail: a role added
    /// later that names a buffer-table index has to answer here, and the cost of forgetting is a
    /// synthetic stage-input descriptor decorated with a real buffer's binding.
    ///
    /// Deliberately conservative at both ends. The function-table and acceleration-structure roles
    /// reserve their slot even where they reflect with no descriptor of their own, because they
    /// still name a Metal buffer-table index the pipeline binds; reserving one nothing reads costs a
    /// synthetic slot, sharing one costs a wrong read. `StageInput` answers `None` on purpose --
    /// its own `u32` is the AIR attribute location, which is what the synthetic slot exists to
    /// stand in for.
    pub fn buffer_table_slot(&self) -> Option<u32> {
        match self {
            Self::Buffer(slot)
            | Self::AccelerationStructureShadow(slot)
            | Self::PrimitiveAccelerationStructure(slot)
            | Self::PrimitiveAccelerationStructureShadow(slot)
            | Self::VisibleFunctionTable(slot)
            | Self::IntersectionFunctionTable(slot) => Some(*slot),
            Self::Texture(_)
            | Self::Sampler(_)
            | Self::StageInput(_)
            | Self::ThreadsPerThreadgroup
            | Self::DispatchThreadsPerThreadgroup
            | Self::ThreadPositionInThreadgroup
            | Self::ThreadgroupsPerGrid
            | Self::ThreadsPerGrid
            | Self::ThreadgroupPositionInGrid
            | Self::ThreadIndexInThreadgroup
            | Self::ExecutionGroup { .. }
            | Self::ThreadPositionInGrid
            | Self::VariantAbsentTexture
            | Self::Other => None,
        }
    }
}

impl KernMeta {
    pub fn role_of(&self, idx: u32) -> Option<&KernRole> {
        self.roles.iter().find(|(i, _)| *i == idx).map(|(_, r)| r)
    }
    pub fn layout_of(&self, idx: u32) -> Option<&AirType> {
        self.buffer_layouts.get(&idx)
    }
    pub fn buffer_address_space(&self, idx: u32) -> Option<u32> {
        self.buffer_address_spaces.get(&idx).copied()
    }
    pub fn buffer_type_size(&self, idx: u32) -> Option<u32> {
        self.buffer_type_sizes.get(&idx).copied()
    }
    pub fn buffer_type_name(&self, idx: u32) -> Option<&str> {
        self.buffer_type_names.get(&idx).map(String::as_str)
    }
    pub fn texture_type_name(&self, idx: u32) -> Option<&str> {
        self.texture_type_names.get(&idx).map(String::as_str)
    }

    /// Synthetic buffer slots used for kernel `[[stage_in]]` attributes.
    ///
    /// Metal supplies stage-input data through ordinary buffer-table slots selected by the pipeline
    /// descriptor. Vulkan exposes the same arrays as read-only storage buffers. Keeping allocation
    /// here makes reflection and lowering consume one ABI decision instead of duplicating it.
    ///
    /// Allocation walks the stage inputs in PARAMETER-INDEX order, not in the order the AIR argument
    /// list happens to name them. The two coincide in all 14553 kernel argument lists of the local
    /// corpus, so ordering them costs nothing today; what it buys is that this ABI is a function of
    /// the entry's signature rather than of a metadata layout no other decision depends on. A
    /// producer that emitted the same kernel with its argument nodes reordered would otherwise hand
    /// a consumer different slots for the same shader.
    pub fn stage_input_bindings(&self) -> HashMap<u32, u32> {
        let mut occupied = self
            .roles
            .iter()
            .filter_map(|(_, role)| role.buffer_table_slot())
            .collect::<std::collections::HashSet<_>>();
        let mut stage_inputs = self
            .roles
            .iter()
            .filter(|(_, role)| matches!(role, KernRole::StageInput(_)))
            .map(|(param_index, _)| *param_index)
            .collect::<Vec<_>>();
        stage_inputs.sort_unstable();
        let mut next = 0u32;
        let mut bindings = HashMap::new();
        for param_index in stage_inputs {
            while occupied.contains(&next) {
                next = next.saturating_add(1);
            }
            occupied.insert(next);
            bindings.insert(param_index, next);
        }
        bindings
    }
}

/// Parse kernel metadata and its entry name from one AIR metadata-node table.
pub fn parse_air_kernel_meta(ll: &str) -> (Option<KernMeta>, Option<String>) {
    parse_air_compute_like_meta(ll, Stage::Kernel)
}

/// Parse an object or mesh entry's metadata and its entry name: a kernel's parameters, with the
/// roles of its own stage in [`KernMeta::mesh_stage`].
pub fn parse_air_mesh_stage_meta(ll: &str, stage: Stage) -> (Option<KernMeta>, Option<String>) {
    debug_assert!(stage.is_mesh_pipeline_stage());
    parse_air_compute_like_meta(ll, stage)
}

fn parse_air_compute_like_meta(ll: &str, stage: Stage) -> (Option<KernMeta>, Option<String>) {
    let nodes = collect_nodes(ll);
    let entry = entry_name_from_nodes(ll, stage.entry_list(), &nodes);
    let meta = parse_air_kernel_meta_with_nodes(ll, &nodes, entry.as_deref(), stage);
    (meta, entry)
}

fn parse_air_kernel_meta_with_nodes(
    ll: &str,
    nodes: &HashMap<u32, String>,
    entry: Option<&str>,
    stage: Stage,
) -> Option<KernMeta> {
    #[cfg(test)]
    KERNEL_META_DECODE_COUNT.with(|count| count.set(count.get() + 1));
    let root = stage_root(ll, stage.entry_list())?;
    let rootc = nodes.get(&root)?;
    let static_int_globals = static_init_int_global_values(ll);
    let resource_location =
        |node: &str, fallback: u32| location_index_with_static(node, fallback, &static_int_globals);
    let param_address_spaces = entry
        .and_then(|name| function_param_pointer_address_spaces(ll, name))
        .unwrap_or_default();
    let refs = refs_in(rootc);
    // The argument-info list is the SECOND ref (`!EMPTY` is the first — an empty placeholder node).
    let in_ref = *refs.get(1)?;
    let mut max_work_group_size = None;
    let mut unmodelled_stage_attributes = vec![];
    let mut mesh_stage = MeshStageMeta::default();
    for attribute in stage_root_attributes(rootc, nodes) {
        match attribute {
            StageAttribute::MaxWorkGroupSize(size) => max_work_group_size = Some(size),
            StageAttribute::MaxMeshWorkGroups(count) if stage == Stage::Object => {
                mesh_stage.max_mesh_work_groups = Some(count)
            }
            other => unmodelled_stage_attributes.push(other.describe()),
        }
    }

    let mut roles = vec![];
    let mut unmodelled_input_params: Vec<(u32, String)> = vec![];
    let mut aliased_implicit_imageblock_params: Vec<u32> = vec![];
    let mut function_constant_buffer_locations = HashMap::new();
    let mut buffer_layouts = HashMap::new();
    let mut imageblock_layouts = HashMap::new();
    let mut buffer_address_spaces = HashMap::new();
    let mut buffer_type_sizes = HashMap::new();
    let mut buffer_object_sizes = HashMap::new();
    let mut buffer_type_names = HashMap::new();
    let mut buffer_accesses = HashMap::new();
    let mut texture_type_names = HashMap::new();
    let mut declared_descriptor_counts = HashMap::new();
    let mut stage_input_type_names = HashMap::new();
    // `air.location_index` of every top-level `air.texture` arg — the basis for the synthetic
    // embedded-texture index K (see `embedded_synthetic_texture_index`).
    let mut top_level_texture_locations: Vec<u32> = vec![];
    // `(buffer_param_index, struct_type_info_node_ref)` for each `air.indirect_buffer` arg, so
    // embedded-texture detection can run once K is known (after the whole arg list is scanned).
    let mut indirect_buffer_struct_refs: Vec<(u32, u32, u32)> = vec![];
    for r in refs_in(nodes.get(&in_ref)?) {
        let Some(node) = nodes.get(&r) else { continue };
        let Some(idx) = first_i32(node) else { continue };
        let layout = struct_info_ref(node).and_then(|sref| parse_struct_info(nodes, sref, 0));
        let strs = role_strings(node);
        if let Some(count) = declared_descriptor_count(node) {
            declared_descriptor_counts.insert(idx, count);
        }
        if strs.first().map(String::as_str) == Some("function_constant")
            && primary_role(&strs) == Some("buffer")
        {
            function_constant_buffer_locations.insert(idx, resource_location(node, idx));
        }
        let Some(mut first) = resource_input_role(&strs) else {
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
            |role| air_input_role_is_modelled(stage, role),
            node,
            nodes,
            &static_int_globals,
        ) {
            unmodelled_input_params.push((idx, declared));
        }
        if stage.is_mesh_pipeline_stage() && mesh_stage.record(first, idx, node, nodes) {
            roles.push((idx, KernRole::Other));
            continue;
        }
        let role = match first {
            "buffer" | "indirect_buffer" => {
                if first == "indirect_buffer" {
                    if let Some(sref) = struct_info_ref(node) {
                        // Key by the buffer's `air.location_index` (the Metal `[[buffer(N)]]` slot the
                        // harness binds), NOT the AIR argument position — they differ (e.g. arg 2 but
                        // buffer(0)). The oracle/runner both index buffers by location.
                        indirect_buffer_struct_refs.push((idx, resource_location(node, idx), sref));
                    }
                }
                if let Some(t) = layout.clone() {
                    buffer_layouts.insert(idx, t);
                }
                buffer_address_spaces.insert(
                    idx,
                    address_space(node)
                        .or_else(|| param_address_spaces.get(&idx).copied())
                        .unwrap_or(1),
                );
                if let Some(name) = arg_type_name(node) {
                    buffer_type_names.insert(idx, name);
                }
                if let Some(size) = i32_after_marker(node, "air.arg_type_size")
                    .or_else(|| i32_after_marker(node, "air.buffer_size"))
                {
                    buffer_type_sizes.insert(idx, size);
                }
                if let Some(size) = i32_after_marker(node, "air.buffer_size") {
                    buffer_object_sizes.insert(idx, size);
                }
                if let Some(access) = declared_buffer_access(node) {
                    buffer_accesses.insert(idx, access);
                }
                KernRole::Buffer(location_index_with_static(node, idx, &static_int_globals))
            }
            "texture" => {
                if let Some(name) = arg_type_name(node) {
                    texture_type_names.insert(idx, name);
                }
                let loc = texture_slot.unwrap_or_else(|| resource_location(node, idx));
                top_level_texture_locations.push(loc);
                KernRole::Texture(loc)
            }
            "instance_acceleration_structure" if body_uses_acceleration_structure_shadow(ll) => {
                KernRole::AccelerationStructureShadow(resource_location(node, idx))
            }
            "primitive_acceleration_structure" => {
                let binding = resource_location(node, idx);
                if ll.contains("@air.intersect.") {
                    KernRole::PrimitiveAccelerationStructureShadow(binding)
                } else {
                    KernRole::PrimitiveAccelerationStructure(binding)
                }
            }
            "visible_function_table" => {
                KernRole::VisibleFunctionTable(resource_location(node, idx))
            }
            "intersection_function_table" => {
                KernRole::IntersectionFunctionTable(resource_location(node, idx))
            }
            "imageblock" => {
                if let Some(t) = layout {
                    imageblock_layouts.insert(idx, t);
                }
                if strs.iter().any(|s| s == "alias_implicit_imageblock") {
                    aliased_implicit_imageblock_params.push(idx);
                }
                KernRole::Other
            }
            "sampler" => KernRole::Sampler(resource_location(node, idx)),
            "threads_per_threadgroup" => KernRole::ThreadsPerThreadgroup,
            "dispatch_threads_per_threadgroup" => KernRole::DispatchThreadsPerThreadgroup,
            "thread_position_in_threadgroup" => KernRole::ThreadPositionInThreadgroup,
            "threadgroups_per_grid" => KernRole::ThreadgroupsPerGrid,
            "threads_per_grid" => KernRole::ThreadsPerGrid,
            "threadgroup_position_in_grid" => KernRole::ThreadgroupPositionInGrid,
            "thread_index_in_threadgroup" => KernRole::ThreadIndexInThreadgroup,
            "thread_position_in_grid" => KernRole::ThreadPositionInGrid,
            "stage_in" => {
                if let Some(name) = arg_type_name(node) {
                    stage_input_type_names.insert(idx, name);
                }
                KernRole::StageInput(resource_location(node, idx))
            }
            VARIANT_ABSENT_TEXTURE_ROLE => KernRole::VariantAbsentTexture,
            // The specializer's spelling of the same fact. Once a VALUE is supplied,
            // `specialize_function_constant_metadata` rewrites the resolved-off wrapper, and past
            // that rewrite the wrapped role is unreadable -- so the demotion above never runs and
            // this is the only place the fact survives.
            _ if declares_disabled_texture(&strs) => KernRole::VariantAbsentTexture,
            other => {
                stage_execution_group_role(stage, other).map_or(KernRole::Other, |(fact, lanes)| {
                    KernRole::ExecutionGroup { fact, lanes }
                })
            }
        };
        roles.push((idx, role));
    }
    // Detect argument-buffer-embedded textures that the body uses through AIR texture
    // intrinsics. Gated purely on AIR structure/semantics — the `air.indirect_argument` →
    // `air.texture` marker chain plus stable AIR intrinsic families — so it cannot key on any shader
    // name. The body must actually use a texture intrinsic for us to surface it.
    let argument_buffers = ArgumentBuffers::new(indirect_buffer_struct_refs);
    let embedded_textures = if body_uses_texture_intrinsic(ll) {
        detect_embedded_textures(nodes, &argument_buffers, &top_level_texture_locations)
    } else {
        vec![]
    };
    let embedded_arguments = detect_embedded_arguments(nodes, &argument_buffers);
    let unsurfaced = unsurfaced_embedded_resources(nodes, &argument_buffers);
    let mut implicit_imageblock_attachments = detect_implicit_imageblock_attachments(ll, entry)?;
    // An aliased explicit imageblock reaches the same planes the implicit intrinsics do, and the
    // module the emitter builds for it loads and stores every one. Reflection has to declare those
    // descriptors from here or a consumer builds a set layout missing the bindings the module it
    // was handed reads.
    let mut aliased_implicit_imageblock_planes = HashMap::new();
    for param in &aliased_implicit_imageblock_params {
        let Some(layout) = imageblock_layouts.get(param) else {
            continue;
        };
        let Some(planes) = aliased_imageblock_planes(layout).ok().flatten() else {
            continue;
        };
        for plane in &planes {
            implicit_imageblock_attachments.push(ImplicitImageblockAttachment {
                attachment: plane.attachment,
                data_rate: 0,
                format: plane.format,
                reads: true,
                writes: true,
            });
        }
        aliased_implicit_imageblock_planes.insert(*param, planes);
    }
    Some(KernMeta {
        roles,
        unmodelled_input_params,
        aliased_implicit_imageblock_params,
        aliased_implicit_imageblock_planes,
        unmodelled_stage_attributes,
        max_work_group_size,
        function_constant_buffer_locations,
        buffer_layouts,
        imageblock_layouts,
        implicit_imageblock_attachments,
        buffer_address_spaces,
        buffer_type_sizes,
        buffer_object_sizes,
        buffer_type_names,
        buffer_accesses,
        texture_type_names,
        declared_descriptor_counts,
        stage_input_type_names,
        embedded_textures,
        embedded_arguments,
        unsurfaced_embedded_resources: unsurfaced,
        mesh_stage,
    })
}
