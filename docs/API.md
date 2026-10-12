# The API

`metal2vulkan` compiles Metal AIR into Vulkan pipelines through an API shaped like Metal's object
lifecycle. You hand it Metal-shaped inputs (a device description, a library, pipeline
descriptors, what a draw or dispatch binds) and get back plain data to create and record Vulkan
objects with: SPIR-V per stage, one pipeline layout, fixed-function state, and per-draw descriptor
writes and push-constant bytes. You never parse AIR, never patch SPIR-V, never derive binding
numbers from Metal indices, and never classify errors by message text.

This guide walks the flow on the repository's own fixtures. Every `rust` block is quoted from
[`examples/api_preview.rs`](../crates/metal2vulkan/examples/api_preview.rs), which the test suite
runs (`tests/api_preview_example.rs` fails when a block here stops matching it). Sections 9 and 10
also cover the runtime crate (`metal2vulkan-runtime`): its degrade notices and its deliberate
differences from Metal. The runtime's own guide is [its README](../crates/metal2vulkan-runtime/README.md).

## Stability

- **No feature needed.** Feature `ash` adds conversions to `ash::vk` and
  `DeviceDescription::query`; `serde` serializes the plain-data values; `cli` builds the command.
- **Frozen names, additive changes.** The surface is the items at the roots of `error`, `device`,
  `library`, `reflection`, `pipeline`, `bind`, `cache`, `metal` (with `metal::{format, sampler,
  state, vertex}`), `vk` and `ash_interop`, plus `Budget` and `CancelToken` at the crate root.
  `tests/api_surface.rs` pins every name. Structs and enums are `#[non_exhaustive]` or opaque, so
  build values with `new`, `Default` or constants, assign fields, and match with a wildcard arm.
  The exceptions are exhaustive on purpose: keys and Vulkan structs (`SlotRef`, `ResourceSlot`,
  `ApiVersion`, ...), and the enums a consumer must interpret completely to bind and draw
  correctly (`StageSource`, `PushHole`, `HeapSource`, `ImageSource`, `SamplerSource`,
  `UseTarget`, `SlotClass`, `StorageAttachmentView`, `VertexStride`, `TileMapping`, `CopySource`,
  `DrawCommand`, `PrecisionNote`, and Metal's `PrimitiveType`, `IndexType`, `CullMode`, `Winding`
  and `TextureType`), where a new variant breaks your build on purpose.
- **Semver** covers this surface from 0.2.0 (still `0.x`: a breaking change bumps the minor
  version).
- **Not API**: anything the reference does not show. Hidden items, every submodule of `library`,
  `pipeline`, `bind` and `device` (write `pipeline::SlotRef`, never `pipeline::sym::SlotRef`), and
  the translator core are private.
- **The AIR front end** (container, bitcode reader, typed IR, metadata ABI, intrinsics,
  canonicalization) is the `metal2vulkan-air` crate, pinned exactly. The input and budget errors
  this crate re-exports are that crate's types.

`scripts/api-docs` renders the reference of exactly this surface (add `--open` to open it); it
fails when a page links to a hidden or internal item.

## Run the tour

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH cargo run -p metal2vulkan --example api_preview
```

No GPU is involved. The fixtures are committed as LLVM text, so the example assembles them with
`llvm-as` first; the API itself reads bitcode only.

## 1. A device

A `DeviceDescription` states the Vulkan physical device as you create it: API version, enabled
features and extensions, limits, subgroup and float-controls properties, format support and the
driver's identity. Three ways to get one:

- `DeviceDescription::from_khronos_json(json, profile_name)` reads a Khronos profiles JSON. The
  tour uses the committed profiles under `crates/metal2vulkan-validation/profiles/`, from a bare
  Vulkan 1.2 device to desktop GPUs and MoltenVK.
- `unsafe DeviceDescription::query(&instance, physical_device, instance_api_version)` (feature
  `ash`) reads a physical device. That is what it supports: enable at least
  `description.recommended_enables(&policy)` at `vkCreateDevice`, then pass
  `with_enabled(&enabled)` so the description states what is enabled. `with_enabled` refuses a set
  that breaks one of Vulkan's co-enable rules, naming its VUID. `VkFeature::exposure()` and
  `VkExtension::requires(version)` say where each feature lives at the device's version, to chain
  the create info.
- Build one field by field.

```rust
let description = DeviceDescription::from_khronos_json(&json, None)?;
let device = Device::new(description, DevicePolicy::default())?;
```

`Device::new` refuses a device below Vulkan 1.2 and an inconsistent description or policy. A
`Device` is cheap to clone; every compile and plan takes it.

### The policy

`DevicePolicy` holds your product choices; together with the description they decide everything a
compile does. `DevicePolicy::default()` is `DevicePolicy::UNTRUSTED`:

- **Containment and isolation on.** Every access stays inside the resources the binder supplied,
  and guest shaders cannot reach each other's resources. Where the device enables
  `robustBufferAccess`, buffer accesses and vertex input are left to it. (MoltenVK advertises the
  feature without implementing it; there such accesses are not contained, as on native Metal.)
- **Ordered framebuffer fetch required** (`FramebufferFetchPolicy::RequireOrdered`).
- **Fast native forms off Apple's GPUs.** The device's own sampler, maths library, divisions,
  fused multiply-adds, texel conversions, `half` rounding, NaN bits, min/max, rounding order of
  `dot`/`mix`/`fast::sqrt`, and blend of a `half` color. Each records a precision note with its
  bound, and each has an exact switch. `DevicePolicy::exact_answers()` turns on every exact form
  but sampling's; `sampling = SamplingPolicy::Exact` emulates Metal's sampler too.
  [FAST-PATHS.md](FAST-PATHS.md) lists every choice, its bound and the measured cost of its exact
  form. On MoltenVK on an Apple GPU, whose sampler and `float` functions are Metal's own, most of
  these choices change nothing.

Loops run as written: no execution bound is translated in.

### What the device answers

```rust
let device = committed_device("moltenvk-m3")?;
let enables = device.required_enables();
// ...
hex(&device.profile().fingerprint()),
// ...
println!("  Metal properties: {:?}", device.metal_properties());
```

- `required_enables()`: everything an artifact compiled for this device can need enabled.
- `metal_properties()`: the answers to the guest's device queries (thread execution width,
  threadgroup limits, argument-buffer and read-write texture tiers, ...).
- `profile().fingerprint()`: a hash of every profile field, the policy included. Every cache key
  includes it, so an artifact never serves another device. Facts that follow the machine's memory
  (`DeviceDescription::allocation`: `maxMemoryAllocationSize`, `maxBufferSize`) are not profile
  fields, so two machines that differ only there share artifacts.
- `profile().spirv_validation()`: the `spirv-val` environments and relaxations every module for
  the device is valid under. The command line runs exactly these.
- The profile's facts: `enabled()`, `limits()`, `images()`, `fragment()`, `descriptors()`, the
  driver workarounds and measured deviations it takes (`workarounds()`, `driver_deviations()`),
  its `subgroups()` model and `simd_tier(...)`, and how it contains an out-of-bounds access
  (`containment()`, `checks_buffers_in_shader()`, `checks_atomics_in_shader()`).
- `neutral_kinds()`: the neutral resources plans bind where nothing compatible is bound. Create
  each once, up front.

### Measured driver behavior

Where Vulkan leaves an answer to the driver and a driver's answer is Metal's, the profile relies on
it only where that driver was measured to give it, keyed on the description's driver identity
(`driver_id`, `vendor_id`, `driver_version`, and `device_id` where one GPU differs from another),
never on a claim of the description itself. These are `DriverDeviation`s (simdgroups that run in
lockstep, Metal's own `float` functions, sampler and canonical NaNs, color writes, ...), measured
on MoltenVK 1.4.1 on Apple's M3 Max and M4, with a few per-GPU facts measured elsewhere. A driver,
GPU or release that was not measured takes the conservative, exact route. A newer MoltenVK release
takes the measured facts and records `PrecisionNote::BeyondMeasuredDeviation`.

Metal semantics the translation keeps on every device, beyond Vulkan's: Metal's 32-lane
simdgroups (including lockstep races on threadgroup memory within a barrier interval), and the
bit-exact `fneg`, `fabs` and `half`-to-`float` widening of NaNs and denormals. Under the exact
switches it also keeps Metal's canonical NaNs and Metal's operand choice in `fmin`, `fmax`,
`min3`, `max3` and `clamp`.

What the runtime enables for itself outside the vocabulary enters no enable set, compile profile
or fingerprint: among it the extensions of its dma-buf fallback for no-copy memory whose pointer
the device refuses (`HostMemory::with_dma_buf`), `VK_EXT_external_memory_dma_buf` with
`VK_KHR_external_memory_fd`, and `VK_EXT_image_drm_format_modifier` with
`VK_EXT_queue_family_foreign` beside them, wherever a device lists them.

## 2. A library

`Library::parse` takes a metallib, a bitcode wrapper or a raw AIR module; `Library::from_air`
takes the last two. LLVM text is not an input: assemble `.ll` with `llvm-as` first. Listing
functions decodes nothing.

```rust
let sampling = Library::from_air(&fixture_bitcode("sampling")?)?;
for info in sampling.functions() {
    println!("library: {} ({:?})", info.name, info.stage);
}
```

- `library.function(name)` finds a function like `newFunctionWithName:`; a missing name is
  `Error::InvalidDescriptor(EntryNotFound)`.
- `function.reflection(&budget)` answers as `reflectionForFunctionWithName:` does: the function's
  arguments in declaration order with their types, access, indices and use, in Metal's shape
  (module `reflection`). It is `None` for a metallib function whose library carries no reflection,
  where Metal answers nil. An argument whose Metal answer was not measured is
  `Unsupported(Feature::Reflection)`.
- A pipeline descriptor's `reflection(&budget)` answers what the `reflection:` out-parameter of
  Metal's creation call does.
- `function.argument_buffer_layout(index, &constants, &budget)` gives the layout of the argument
  buffer the function takes at a buffer index, as `newArgumentEncoderWithBufferIndex:` encodes it:
  its length, alignment, and each argument's offset and type by `[[id(n)]]`.
  `ArgumentBufferLayout::from_descriptors` and `from_buffer_binding` give the layouts of Metal's
  other two encoder constructors. Each argument is 8 bytes: a buffer's address plus offset, or a
  resource ID.
- `function.properties(&budget)` answers what `MTLFunction` states about the function.

### One entry of a bare module

A tool that holds bitcode and an entry name, not a library, compiles through the profile:

```rust
let bitcode = fixture_bitcode("sampling")?;
let entry = EntryDescriptor::new(&bitcode, EntryRef::Name("sample_level"));
let profile = device.profile();
let (compiled, report) = profile.compile_entry(&entry, &LoweringInputs::new(), &budget);
let alone = profile.link_standalone(&compiled?)?;
```

`link_standalone` links the function with no partner stage. The report says what the compile cost
against its budget, whether it succeeded or not.

## 3. A compute pipeline

Descriptors mirror Metal's. Each stage is a `StageSource`: a `FunctionDescriptor` to compile (a
function, its constant values, the functions it links) or a `CompiledFunction` to link as it is.
Every compile takes a `Budget`:

```rust
let cancel = CancelToken::new();
let mut budget = Budget::standard();
budget.cancel = Some(cancel.clone());
let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
let descriptor = ComputePipelineDescriptor::new(StageSource::Function(kernel));
let pipeline = device.compile_compute_pipeline(&descriptor, &budget)?;
```

`Budget::standard()` is a 20-second deadline. `cancel.cancel()` from any thread ends the attempt
at its next poll point with `Error::Cancelled`; a passed deadline is `Error::Budget`. Both are
transient. Deterministic caps (work units, a memory ledger, structural limits) apply whatever the
budget, and exceeding one is `Error::LimitExceeded`, permanent for that input.

A `ComputePipeline` is immutable plain data:

- `stage`: a `StageModule` (SPIR-V 1.5 words valid for the device, the entry point, specialization
  constants, a required subgroup size and stage flags, precision notes);
- `layout`: the pipeline layout, dense and complete; create it as it is;
- `interface`, `dependencies`: what it binds and what bind-time state it depends on (section 5);
- `metal_info`: answers to Metal's per-pipeline queries (`maxTotalThreadsPerThreadgroup`, ...);
- `requirements`: what the device must have enabled.

Do not create the compute `VkPipeline` from `stage` alone: its specialization defaults are
placeholders (a kernel's local size is 1). Create each pipeline from a planned dispatch's
`DispatchRegion` (section 6), with the region's `specialization`, `stage_flags` and
`required_subgroup_size`. A required subgroup size needs `subgroupSizeControl` and the stage in
`requiredSubgroupSizeStages`; full subgroups need `computeFullSubgroups`. A pipeline compiled for
one device pins only what that device can take; `DeviceProfile::check` tells whether another
device can.

### A kernel with `[[stage_in]]`

A kernel's `[[stage_in]]` attributes are fetched in the shader, as Metal does. The compile asks for
the stage-input descriptor it needs, and you supply it:

```rust
match device.compile_compute_pipeline(&descriptor, budget) {
    Err(Error::NeedsInput(RuntimeInput::StageInputDescriptor { attribute })) => {
        let mut stage_input = StageInputDescriptor::new();
        let fetch = StageInputAttributeDescriptor::new(AttributeFormat::Float, 0, 1);
        stage_input.attributes.insert(attribute, fetch);
        let mut layout = StageInputLayoutDescriptor::new(Some(4));
        layout.step_function = StepFunction::ThreadPositionInGridX;
        stage_input.layouts.insert(1, layout);
        descriptor.stage_input = Some(stage_input);
    }
// ...
let mut inputs = LoweringInputs::new();
inputs.stage_input = descriptor.stage_input.clone();
let compiled = device.compile_function(&kernel, &inputs, budget)?;
```

A fresh `StageInputLayoutDescriptor` steps per vertex at rate 1, as a fresh Metal layout does. A
layout stride of `None` is `MTLBufferLayoutStrideDynamic`: the stride comes from the encoder.

### Other compute features

- **Visible functions and recursion.** Linked visible functions, function tables and
  intersection function tables compile; recursion through visible functions runs on a stack of
  frames sized by `max_call_stack_depth`. A call that finds the stack full returns zero.
- **Binary functions, required threadgroup sizes, buffer mutability.** These descriptor fields
  change no translation and are in no compile key; the runtime checks them as Metal does.
- **Dynamic libraries.** `DynamicLibrary::new(library, &budget)` checks a dynamic metallib as
  Metal's `newDynamicLibrary:` does; pipelines link its exported functions.

## 4. A render pipeline

```rust
let mut vertex_descriptor = VertexDescriptor::new();
let layout = VertexBufferLayoutDescriptor::new(Some(24));
vertex_descriptor.layouts.insert(0, layout);
// ...
let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
descriptor.fragment = Some(StageSource::Function(fragment));
descriptor.vertex_descriptor = Some(vertex_descriptor);
let color = ColorAttachmentDescriptor::new(PixelFormat::RGBA8Unorm);
descriptor.color_attachments[0] = Some(color);
// ...
let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
```

A `RenderPipeline` holds:

- `stages`: vertex, then fragment when rasterization is enabled;
- `layout`: one pipeline layout for every stage;
- `vertex_input`: Vulkan bindings and attributes. An attribute Vulkan cannot fetch as described is
  pulled in the shader instead, and so is every attribute on a device without
  `robustBufferAccess` (unless containment is off);
- `graphics`: the fixed-function state (topology, attachment formats and blend states, sample
  count, dynamic states, input attachments, ...). Build
  `VkPipelineColorBlendStateCreateInfo::pAttachments` from `graphics.color_blend_attachments()`.

Where Metal's rules differ from Vulkan's, the pipeline gives Metal's answer:

- A color attachment no fragment output names keeps its texels (an empty write mask).
- What Metal refuses, the link refuses as `InvalidDescriptor(ColorAttachment)`: an output with
  fewer components than its attachment's channels, a blend that reads the second output without
  the first, alpha to coverage over an integer attachment 0. Function-constant values that put two
  outputs at one attachment are `InvalidDescriptor(DuplicateColorOutput)`.
- On a device without `independentBlend`, attachments the function does not write copy the blend
  state of those it does; where those differ the pipeline is refused, naming the feature.
- A fragment function that reads a color attachment (framebuffer fetch) reads it in an ordered
  form wherever the device has one: rasterization-order attachment access, or the interlocked
  storage form. A `[[stencil]]` output uses `VK_EXT_shader_stencil_export`.
- `graphics` also states what the runtime needs to record draws correctly: whether the fragment
  function reads flat inputs, barycentrics or its position's depth, whether the vertex function
  writes memory or a layer or viewport index, and which colors the function may write.
- On a device with `VK_KHR_portability_subset` (MoltenVK), each of the subset's missing features
  is checked against what the pipeline uses.

`metal::state` maps the encoder-side Metal state a pipeline does not own (cull mode, winding,
depth/stencil state, ...) to Vulkan values.

### Tessellation

A post-tessellation vertex function makes a tessellation pipeline:

```rust
let vertex = FunctionDescriptor::new(library.function("tess_quad_buffer")?);
let mut descriptor = RenderPipelineDescriptor::new(StageSource::Function(vertex));
descriptor.tessellation = Some(TessellationDescriptor::default());
let pipeline = device.compile_render_pipeline(&descriptor, &Budget::standard())?;
```

Its `stages` are a synthesized vertex and tessellation control stage, the function as the
evaluation stage, and the fragment stage. It draws patches only (`DrawCall::Patches` or
`PatchesIndirect`), with the factors from a tessellation factor buffer.

### Mesh pipelines

`Device::compile_mesh_pipeline` compiles a `MeshPipelineDescriptor` (an optional object function,
a mesh function and a fragment function), and `compile_mesh_variant` its variants. Its plan draws
mesh draws only (`DrawCall::MeshThreadgroups`, `MeshThreads` and their indirect forms). With
`VK_EXT_mesh_shader` the stages are task and mesh shaders. Without it (MoltenVK, older ANV) the
pipeline is an exact emulation: the object and mesh functions run as compute stages that write the
mesh's vertices and primitives as records into buffers, and a synthesized vertex stage draws those
records; the plan says which dispatches and draw to record.

### Tile pipelines

`Device::compile_tile_pipeline` compiles a `TileRenderPipelineDescriptor` (the tile function and the
render pass's color attachment formats) into a `ComputePipeline` that stands in for
`dispatchThreadsPerTile:`. Dispatch it as `Dispatch::Threadgroups` over the render area, one
threadgroup per tile, with `BindState::imageblock_size` and `BindState::render_area` set, and bind
each attachment it reaches as a storage image where the plan writes `WriteSource::TileAttachment`.
Its threads are numbered as Metal's tile dispatch numbers them, for the threadgroup layouts that
were measured; others are refused.

## 5. The interface and the plan

The pipeline states what it binds and how:

```rust
for (slot, interface) in &pipeline.interface.slots {
    println!("  slot {slot:?}: {interface:?}");
}
for set in &pipeline.layout.sets {
    for binding in &set.bindings {
// ...
for dependency in &pipeline.dependencies {
    let (fact, mechanism) = (dependency.fact, dependency.mechanism);
    println!("  depends on {fact:?} via {mechanism:?}");
}
```

- `interface.slots` holds every statically used Metal slot (`SlotRef`: stage, table, index): how
  the shader accesses it, and for a buffer the byte window it can touch. A buffer slot's `kind`
  says whether Metal binds a buffer, a visible-function table or an intersection function table
  there.
- The layout numbers set 0 densely, in `ResourceSlot` order. Set 1 holds the heap arrays (the
  resources reached through argument buffers and resource IDs). Push constants carry the values the
  binder writes per command.
- `dependencies` lists the bind-time state the pipeline depends on (`RuntimeFact`: threadgroup
  sizes, sampler coordinate modes, buffer ranges, ...) and how this profile honors each
  (`Mechanism`: a specialization constant, a push field, a plan-time check, a variant). The plan
  acts on them; you do not have to.
- The format class of a bound texture is read by the shader as push data, not specialized, so binds
  that differ only in their textures' formats share one `VkPipeline`. Other bound values (a slot's
  presence, a sampler's word) are specialization constants.
- `interface.reaches_command_buffers` and `interface.texture_writing_stages` say, provably, which
  pipelines may touch an indirect command buffer and which stages may write a texture.

### Plans

Per draw or dispatch, describe what is bound, keyed by Metal slot, as a `BindState`. The binder
never sees your resources, only facts about them:

```rust
let mut table = StageTable::default();
table.buffers.insert(0, buffer(4096, 0));
let facts = TextureFacts::new(TextureType::Type2D, PixelFormat::RGBA8Unorm);
let texture = BoundTexture::Texture {
    facts,
    resource_id: None,
};
table.textures.insert(0, texture);
let sampler = BoundSampler::new(SamplerDescriptor::new());
table.samplers.insert(0, sampler);
let mut state = BindState::default();
state.stages.insert(Stage::Kernel, table);

let plan = pipeline.plan(&device, &state.shape())?;
assert!(plan.variant.is_empty(), "this shape needs no other variant");
```

`state.shape()` keeps only what can change a plan (which slots are bound with what kind of
resource, formats, level counts, sampler modes, alignment classes). A plan is a pure function of the
pipeline, the device and the shape: memoize it on `(PipelineKey, BindShape)`.

The plan holds one `DescriptorWrite` per layout binding element (the resource at a slot, a buffer
the binder synthesizes, or a neutral resource where nothing compatible is bound), the vertex
buffers, and the `ResourceUse`s that drive staging, writeback and barriers. List in the state
every live sampler state and visible-function table the command can reach through argument buffers
(`argument_samplers`, function tables), and the resources it uses (`used_resources`).

A plan whose `variant` is not empty asks for another variant of the pipeline (a storage texture's
format, a buffer range past the descriptor limit, a vertex buffer that fixed-function input cannot
fetch, ...). Compile it and plan it for the same shape: `device.compile_render_variant(&descriptor,
&plan.variant, &budget)` for a render pipeline, or compile the kernel with
`plan.variant.lowering_inputs(Stage::Kernel)` and link it as `StageSource::Compiled`. A dispatch can
ask too (`plan.dispatch_variant(&dispatch)`, or `BindError::NeedsVariant` from instantiation).

Textures follow Metal's texel rules: sparse textures report residency where the device can,
`Depth32Float` texels keep any `float`, `ulong` textures and their atomics use 64-bit images where
the device has them, and `half` and sRGB reads give Metal's values or record their bound. The
rustdoc of `BindState`, `TextureFacts` and `BindingPlan` states each rule.

## 6. At bind time

Instantiate the plan per dispatch or draw:

```rust
let dispatch = Dispatch::Threads {
    threads: [100, 1, 1],
    threads_per_group: [32, 1, 1],
};
let commands = plan.instantiate_dispatch(&state, &dispatch)?;
for region in &commands.regions {
// ...
println!("  descriptor ranges: {:?}", commands.descriptor_ranges);
println!("  push holes: {:?}", commands.push_holes);
```

```rust
let plan = pipeline.plan(device, &state.shape())?;
let draw = DrawCall::Primitives {
    vertex_start: 0,
    vertex_count: 6,
    instance_count: 1,
    base_instance: 0,
};
let commands = plan.instantiate_draw(&state, &draw)?;
```

A dispatch is one or more regions (a grid that is not whole threadgroups adds edge regions), each
with its group count, specialization and push constants: create one `VkPipeline` per distinct
specialization. A kernel that steps a stage-in attribute by thread position, or reads
`[[grid_origin]]` or `[[grid_size]]`, needs the encoder's stage-in region:

```rust
match plan.instantiate_dispatch(&state, &dispatch) {
    Err(BindError::MissingStageInRegion) => {}
// ...
state.stage_in_region = Some(StageInRegion::new([0, 0, 0], [64, 1, 1]));
let commands = plan.instantiate_dispatch(&state, &dispatch)?;
```

A tessellation pipeline draws patches, with the factors' buffer in the state:

```rust
let factors = TessellationFactorBuffer::new(4096, 0, 0);
state.tessellation_factors = TessellationFactors::new(Some(factors), 1.0);
let plan = pipeline.plan(device, &state.shape())?;
let draw = DrawCall::Patches {
    control_points: 4,
    patch_start: 0,
    patch_count: 16,
    patch_index: None,
    instance_count: 1,
    base_instance: 0,
    control_point_index: None,
};
let commands = plan.instantiate_draw(&state, &draw)?;
```

What you record from `DispatchCommands` or `DrawCommands`:

- **Descriptors.** Write each of the plan's `descriptors`, binding each buffer descriptor over
  exactly its `descriptor_ranges` entry. Upload `descriptor_bytes` (argument and handle tables) and
  the resident root tables where present. A handle table maps each resident resource's
  `gpuResourceID` to its heap element; a consumer that keeps one heap for every command buffer can
  bind one table of every texture ID instead (`bind::handle_entry`, `bind::heap_element`,
  `bind::handle_entry_flags`), as the runtime does.
- **Push constants.** Write `push_constants`, and fill each of the `push_holes` with the device
  address it names: the uploaded root table, the command buffer's `ExecutionStatus` buffer, an
  indirect command buffer's records.
- **The dispatch or draw.** Record `regions` or `draw` as given: the binder has already applied
  Metal's offsets, bases and clamps.
- **Status.** Per command buffer, keep one `ExecutionStatus` buffer for what translated shaders
  report (a resident-set miss, for one), and read it when the command buffer completes.

### Indirect command buffers

An indirect command buffer's commands are records in ordinary buffer storage
(`bind::indirect_commands`): one format whatever the descriptor, written by the CPU or by a kernel.
The binder replays records into dispatches and draws: on the host, or on the device through a
preprocessing kernel (`replay_preprocess`), and through `VK_EXT_device_generated_commands` where
the device has it (`replay_preprocess_generated`). A kernel that encodes commands writes the
records at `bind::indirect_commands::records_address` of the buffer's ID, outside every Metal
address.

## 7. Caching

The crate keeps no cache; keys and encodings let you keep one across processes. Keys are SHA-256
hashes of the inputs, `cache::SEMANTICS_VERSION` and the profile fingerprint, known before
compiling (`device.compute_pipeline_key(&descriptor)`, `render_pipeline_key(&descriptor,
&budget)`, `tile_pipeline_key`, `function_compile_key`):

```rust
let bytes = pipeline.to_bytes();
let cached = ComputePipeline::from_bytes_for_key(&bytes, &pipeline.key, &device)?;
assert_eq!(cached, pipeline);
```

A decode checks the format and semantics versions, the key, the profile fingerprint and a
checksum. Every `ArtifactError` is a cache miss: compile again and replace the entry.

Functions cache on their own, so pipelines that share a function share its compile, and a
pipeline links a cached function without decoding it again:

```rust
let kernel = FunctionDescriptor::new(sampling.function("sample_level")?);
let inputs = LoweringInputs::new();
let key = device.function_compile_key(&kernel, &inputs);
let compiled = device.compile_function(&kernel, &inputs, &budget)?;
let bytes = compiled.to_bytes();
let function = CompiledFunction::from_bytes_for_key(&bytes, &key, &device)?;
let descriptor = ComputePipelineDescriptor::new(StageSource::Compiled(function));
let linked = device.compile_compute_pipeline(&descriptor, &budget)?;
assert_eq!(linked, pipeline);
```

- A compiled function states before linking what its stage will: `stage()`, `requirements()`
  (check it against another profile with `DeviceProfile::check`), `precision_notes()`,
  `reconvergence()`, `required_subgroup_size()`, `requires_full_subgroups()`.
  `compile_function_with_report` also returns a `CompileReport` of what the compile cost.
- A render pipeline's stages compile under inputs that a decode of its functions learns.
  `render_pipeline_key` returns `RenderPipelineKeys`: each stage's function key, and the inputs to
  compile a missing stage under.
- A key names an artifact by its inputs and versions; a content digest names it by what it says.
  `pipeline.content_digest()` hashes everything but the key and profile, and `plan.content_digest()`
  (with `serde`) everything a plan states. A cache of driver objects keyed on the digest survives a
  release that bumped `SEMANTICS_VERSION` without changing the output. A digest identifies
  content; it authenticates nothing.
- Versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output),
  `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION` (what the runtime does with an
  unchanged artifact). `SOURCE_DIGEST` names the sources a build was compiled from, for caches kept
  across builds.

## 8. Refusals, requested inputs, and precision

Every failure is typed; match variants, never messages. A refusal says what would change the
answer:

```rust
let floor = committed_device("vk12-floor")?;
let min_lod = FunctionDescriptor::new(sampling.function("sample_min_lod")?);
match floor.compile_function(&min_lod, &LoweringInputs::new(), &Budget::standard()) {
    Err(Error::Unsupported(refusal)) => match (&refusal.feature, &refusal.needs) {
        (feature, Needs::AnyOf(alternatives)) => {
            println!("refused {feature:?}: needs any of {alternatives:?}");
        }
        (feature, needs) => println!("refused {feature:?}: {needs:?}"),
    },
```

`Error` has these variants:

| Variant | Means | Retry |
|---|---|---|
| `Unsupported` | The device cannot run this exactly. `feature` names the Metal feature, `needs` what would change it | No |
| `NeedsInput` | The compile needs a pipeline input you did not supply | After supplying it |
| `InvalidInput` | The AIR or container is malformed | No |
| `InvalidDescriptor` | The descriptor is invalid as Metal would refuse it, or names a missing entry | No |
| `LimitExceeded` | A deterministic cap (work units, ledger bytes, output size, structural limits) | No |
| `Budget` | The deadline passed | Later |
| `Cancelled` | The `CancelToken` was triggered | Later |
| `Internal` | A translator bug (a panic, a verifier failure) | No; report it |

`Needs` says what a refusal needs:

- `AnyOf`: alternative sets of device capabilities; a device that enables every capability of one
  set compiles it.
- `NotImplemented`: a gap of this version. `NotRepresentable`: no Vulkan mechanism exists.
- `DriverDefect`: a defect the device's driver is measured to have. `RuntimeDriverDefect`: one in a
  command the runtime records. `UnmeasuredDeviation`: driver behavior the input needs that no
  measurement covers for this driver release.
- `MetalGpuFamily`: a Metal GPU family the device does not claim, where Metal itself refuses the
  input without it.

`location` and `symbol` are diagnostics only. A render pipeline's refusal of one attachment's
state names it (`location.attachment()`). [REFUSALS.md](REFUSALS.md) lists every refusal in the
code, with its device condition.

A compile that needs an input asks for it (`Error::NeedsInput`) instead of guessing: a
function-constant value, a storage texture's format, a linked function, a kernel's stage-input
descriptor. Supply it and compile again:

```rust
match device.compile_compute_pipeline(&descriptor, &budget) {
    Err(Error::NeedsInput(RuntimeInput::FunctionConstant { index })) => {
        let words = [1u32, 2, 3, 4].map(u32::to_le_bytes).concat();
        let value = FunctionConstantValue::new(DataType::UInt4, words);
        kernel.constants = FunctionConstantValues::defaults().with_index(index, value);
    }
```

A storage texture's format (`RuntimeInput::StorageTextureFormat { slot }`) comes from what is
bound: `metal::format::storage_image_choice(pixel_format, device.profile())`, placed in a
`VariantKey`'s `storage_images` and compiled as a variant. The plan asks for it itself where the
bound format needs a declared format.

**What to cache.** `error.retry()` says whether the same request can succeed later, and
`error.is_permanent_for()` which key may memoize the failure (the input, the input on this device,
or never). A `BindError` from `plan` depends only on the pipeline, the device and the shape, so it
memoizes beside the plan; one from instantiation does not.

**Precision.** Where an output gives a near-equivalent of Metal's answer, its stage lists it in
`precision_notes` (a `PrecisionNote`, each documenting its bound), and `reconvergence` says
whether subgroup reconvergence after divergence matches Metal's
(`Reconvergence::ImplementationDefined` where the profile lacks maximal reconvergence). A note
that holds only under some bound values is in `specialized_notes`;
`stage.precision_notes_bound(&plan.specialization, &|key| plan.specialized_value(key))` gives the
notes a plan reaches. Section 9 lists every note.

**The runtime's errors** are its own type, `metal2vulkan_runtime::Error`, which tells apart what
Metal's validation stops (`Invalid`), what plain Metal answers nil for (`Nil`, `MetalError`), what
the device lacks (`Unsupported`), what the runtime has not built (`NotImplemented`) and the
translator's errors (`Compile`). Its [README](../crates/metal2vulkan-runtime/README.md#errors)
lists them.

## 9. Degrade notices

Where the runtime gives Metal's answer only at a cost Metal does not pay, or gives a recorded
near-equivalent, it still answers, and says so. Each kind of degradation is a
`metal2vulkan_runtime::Degradation`:

- **performance**: Metal's answer, at an extra cost (a copy, a staging or scratch resource, an
  extra pass, a slower variant);
- **precision**: a recorded near-equivalent of Metal's answer, including every `PrecisionNote` a
  pipeline stage records;
- **validity**: Metal's answer exactly, through a use of the device its driver does not advertise
  but was measured to run as Metal does, so Vulkan's validation layer reports the use.

How a notice reaches you:

- Each kind is announced **once per device**, the first time the device takes it. A kind listed in
  `DeviceOptions::announce_each` (`RawDevice::announce_each` for an adopted device) is announced at
  every occurrence instead. The notes a pipeline's making records are announced at the first
  dispatch or draw that binds it.
- Through the device's message sink (`DeviceOptions::messages`) as a `DebugMessage` of `Warning`
  severity whose `id_name` is the kind's code and whose `degradation` holds a `DegradeNotice`: the
  `Degradation`, the object degraded, and the reason. Without a sink, one line on stderr:
  `metal2vulkan: <object>: <reason>; <cost> [<code>]`.
- A notice is never fatal: the work runs.
- Decode a notice by its `Degradation` or its code, never its text. The codes are a contract: none
  is renamed or reused, and a retired code stays listed below.
- `Degradation::exactness()` says whether the program still gets Metal's answer: `Exact` (most
  performance and validity kinds), `ExactWithinPool` (Metal's answer while a command buffer's pool
  holds it, else the command buffer fails), `OneOfMetals` (Metal's own answer is not one value, and
  the runtime gives one of them), `FailsInstead(error)` (the command buffer fails rather than
  differ), or `NearEquivalent` (the precision notes and other near-equivalents).
- `Device::degraded_objects(kind)` counts the textures a device met as a kind that is announced
  once per object.

With `METAL2VULKAN_LOG_PIPELINES` set, a device also logs one line per render `VkPipeline` it
makes (id name `metal2vulkan-pipeline-creation`, `Info` severity). That text is for people, not a
contract.

Each entry below gives the code, its category, the `Degradation` kind (or the `PrecisionNote`) and
what triggers it; a runtime kind's cost follows. The kind's rustdoc states the trigger, cost and
how to avoid it in full. [FAST-PATHS.md](FAST-PATHS.md) lists which notes the default policy's
fast forms record, and each `PrecisionNote`'s rustdoc states its bound.

- `metal2vulkan-degrade-shadowed-texture` (performance). `ShadowedTexture`: A 2D texture over a
  buffer is a shadow image: no linear image of the device lays its texels out as Metal does in the
  buffer's memory (over a no-copy buffer's imported host memory, no linear image made for host
  allocations does), or the buffer is over the caller's `VkBuffer`, whose memory backs no image of
  the runtime's, or over host pages imported as dma-bufs (`HostMemory::with_dma_buf`) where the
  device makes no image over a dma-buf of the texture's own pages (the notice names why;
  `WorkCounts::dma_buf_image_fallbacks` counts it). Cost: its texels are copied into an image before
  each use and back after each write.
- `metal2vulkan-degrade-staged-texture-buffer` (performance). `StagedTextureBuffer`: A texture
  buffer is viewed through a staging buffer: its texels start where the device views none
  (`minTexelBufferOffsetAlignment`). Cost: its texels are copied into a staging buffer before each
  use and back after each write.
- `metal2vulkan-degrade-shadow-merge` (performance). `ShadowMerge`: One command writes a buffer's
  bytes through a shadowed texture or a staged texture buffer and also another way. Cost: the
  shadow's texels are kept in a scratch buffer of three times the texture's bytes before the
  command, and after it a dispatch merges the bytes it changed through the texture into the
  buffer's: three copies and a dispatch in place of one copy back.
- `metal2vulkan-degrade-copied-draw-bytes` (performance). `CopiedDrawBytes`: A draw's index buffer
  or indirect arguments start where Vulkan reads neither. Cost: they are copied into a buffer of the
  command buffer before the pass, and the draw reads the copy.
- `metal2vulkan-degrade-isolated-index-bytes` (performance). `IsolatedIndexBytes`: An indexed draw
  reads index positions past where Metal's fetch stops reading its index buffer (its length rounded
  up to 4 bytes), for which Vulkan states no index: a direct draw whose positions run past there, on
  every device; or an indirect draw that stays one draw (a driver measured to read zero past the
  bound range, `DriverFact::RobustIndexFetchReadsZero`) on a device that enables
  `robustBufferAccess2` but not `maintenance5`, whose bound range runs to the end of the index
  buffer's `VkBuffer`, on past the bytes into its padding or into a heap's bytes after a placed
  buffer. Cost: its whole indices inside the buffer are copied into a buffer of the command buffer
  before the pass, between two barriers, and the draw reads the copy: a direct draw's at its start,
  every position past them zero.
- `metal2vulkan-degrade-unordered-draw-bytes` (performance). `UnorderedDrawBytes`: An indexed
  indirect draw reads arguments or indices a draw of its encoder wrote, with no memory barrier in
  the encoder before it, a race Metal does not order; or a draw's vertex stage reads a buffer of its
  vertex table a fragment stage of its render pass wrote. Cost: they are copied into a buffer of the
  command buffer before the pass, between two barriers, and the draw reads the bytes from before the
  write, which Metal's draw read in every run measured.
- `metal2vulkan-degrade-unorm-depth-bias` (precision). `UnormDepthBias`: A draw biases a
  `Depth16Unorm` depth by a constant on a device whose driver is not measured to store Metal's codes
  so. Cost: a biased depth can store a code next to Metal's: the device rounds the biased sum its
  own way, and without depthBiasExact counts the bias in its own unit.
- `metal2vulkan-degrade-unorm-depth-rounding` (precision). `UnormDepthRounding`: A draw tests or
  writes a `Depth16Unorm` depth on a device whose driver is not measured to convert a depth to
  Metal's code. Cost: a depth near a code's midpoint can store the code below Metal's: the device
  converts the depth to the 16-bit code its own way, where Metal rounds to nearest, a tie up.
- `metal2vulkan-degrade-point-guard-band` (precision). `PointGuardBand`: A draw of points keeps
  Vulkan's clipping of points, where Metal's guard band cannot be emulated: in a rate-mapped pass,
  on a device without `shaderClipDistance` or with too few clip distances beside the vertex
  function's own, or through a viewport whose widened Vulkan viewport would leave the device's
  limits or not be a `float` exactly. Cost: a point whose center lies past its viewport, within
  Metal's guard band of 1.5 times the viewport's half-size, is culled where Metal draws it, and a
  wide point can lose pixels at a fractional viewport edge by the device's own clipping.
- `metal2vulkan-degrade-nan-depth` (precision). `NanDepth`: A draw whose fragment function writes
  its depth, in depth clamp mode through a range above 0, keeps the device's clamp of that depth,
  where the function cannot clamp it itself under a Vulkan clamp that holds 0: on a device without
  `VK_EXT_depth_clamp_control` whose vertex stage cannot compensate the depth exactly, a mesh
  pipeline there, through viewports of different clamps or depth ranges, under a depth bias that
  saturates or offsets the range, or past the device's clip distances. Cost: a NaN depth the
  fragment function writes: a negative one in depth clamp mode stores the depth clamp range's low
  end, where Metal stores 0; a positive one runs no stencil operation, where Metal runs the
  depth-failure one.
- `metal2vulkan-degrade-biased-depth-past-unit-range` (precision). `BiasedDepthPastUnitRange`: A
  draw of triangles in depth clip mode under a depth bias of a constant or slope factor other than
  0, on a device that enabled `VK_EXT_depth_range_unrestricted` but lacks `depthClamp`,
  `VK_EXT_depth_clip_enable` or `VK_EXT_depth_clamp_control`: such a device stores a depth the bias
  takes past `[0, 1]` unclamped, where Metal clamps it. Cost: a depth a bias takes past [0, 1] in
  depth clip mode is stored unclamped, where Metal stores 0 or 1.
- `metal2vulkan-degrade-clipped-depth-pieces` (precision). `ClippedDepthPieces`: A draw of lines or
  triangles whose vertex stage clamps each vertex's depth as Metal does, through a viewport depth
  range with a negative depth in depth clip mode or past `[0, 1]` in depth clamp mode: Metal clips a
  primitive at its near and far planes and at its guard band first, and clamps the depths of the
  pieces' vertices. Cost: a primitive that crosses Metal's near or far plane, or its guard band of
  1.5 times the viewport's half-size, stores the interpolation of its own vertices' clamped depths,
  where Metal clamps the depths of the pieces it clips the primitive into.
- `metal2vulkan-degrade-sample-depth-clamp` (precision). `SampleDepthClamp`: A draw of lines or
  triangles through such a range whose vertex stage cannot clamp each vertex's depth: its fragment
  function reads its position's depth, a mesh pipeline, on a device without `shaderClipDistance` or
  past its clip distances, or through viewports whose depths need different compensations on a
  device that takes the ranges natively; and a draw of triangles in depth clamp mode under a depth
  bias that moves the depth through a range inside `[0, 1]`, where Metal adds the bias to each
  vertex's depth clamped to `[0, 1]`, which a primitive past its near or far plane shows. Cost: a
  sloped primitive's depth is clamped at each sample, where Metal clamps each vertex's depth before
  the interpolation, at 0 in depth clip mode and to [0, 1] in depth clamp mode.
- `metal2vulkan-degrade-depth-bias-past-one` (precision). `DepthBiasPastOne`: A draw of triangles in
  depth clamp mode through a viewport depth range whose larger depth lies past 1, under a depth bias
  of factors, whose Vulkan clamp holds at most 1. Cost: a depth the bias moves past 1 stores 1,
  where Metal clamps it to the range's larger depth.
- `metal2vulkan-degrade-in-place-vertex-reads` (precision). `InPlaceVertexReads`: A draw's vertex
  stage reads in place bytes a fragment stage of its render pass wrote, through an argument buffer
  or a resident resource, of a buffer another stage of the pass or the vertex function writes, or in
  an execution of render commands. Cost: the device may order the fragments' write before the read,
  where Metal's vertex stages read the bytes from before the pass's fragments in every run measured.
- `metal2vulkan-degrade-fragments-before-geometry` (precision). `FragmentsBeforeGeometry`: A draw's
  object or mesh stage reads in place bytes a fragment stage of its render pass wrote (through an
  argument buffer or a resident resource, or of a buffer another stage of the pass or the stage's
  function writes; an emulated stage's where its work records after the pass's earlier render pass
  instances), or a draw's vertex, object or mesh stage writes bytes a fragment stage of its render
  pass read or wrote: Metal's GPUs run every draw's vertex, object and mesh functions before any
  draw's fragments of the pass. Cost: the device may run the pass's earlier fragments before the
  draw's stage, and does across a split of the pass (a command's barrier, a memory barrier): the
  read sees their write, their read misses the write, or the write outlasts theirs, where Metal's
  fragments run after every draw's vertex, object and mesh functions in every run measured.
- `metal2vulkan-degrade-indirect-index-bounds-pass` (performance). `IndirectIndexBoundsPass`: An
  indexed indirect draw whose index fetch nothing bounds (the device lacks `robustBufferAccess2`,
  its driver is not measured to read zero past the bound range,
  `DriverFact::RobustIndexFetchReadsZero`, or the index offset lies at or past the buffer's length)
  is bounded on the GPU before the pass. Cost: two dispatches before the pass write two draws'
  arguments and the indices past the buffer into the command buffer's index pool (a sixty-fourth of
  the device's largest device-local heap), between barriers, and the draw is two indirect draws; a
  draw the pool cannot hold fails the command buffer.
- `metal2vulkan-degrade-emulated-mesh-draws` (performance). `EmulatedMeshDraws`: A mesh render
  pipeline's draws on a device without `VK_EXT_mesh_shader`. Cost: the object and mesh functions run
  as compute dispatches before the pass, between barriers, writing records a synthesized vertex
  stage draws; a draw whose mesh threadgroups the GPU decides is scanned first and reserves its
  records in the command buffer's records pool (a sixty-fourth of the device's largest device-local
  heap), and a draw the pool cannot hold fails the command buffer.
- `metal2vulkan-degrade-landed-visibility-results` (performance). `LandedVisibilityResults`:
  Visibility results start where Vulkan writes no query result. Cost: they land in an upload region
  and are copied on into the buffer after the pass.
- `metal2vulkan-degrade-overlapping-buffer-copy` (performance). `OverlappingBufferCopy`: A blit
  copies between overlapping ranges of one buffer, which Vulkan's copy forbids. Cost: the range goes
  through a scratch buffer of its size, which its creation clears by a fill it waits for: two copies
  instead of one.
- `metal2vulkan-degrade-split-buffer-fill` (performance). `SplitBufferFill`: A blit fills bytes that
  are not whole 4-byte words at multiples of 4. Cost: the words are filled and the up to 3 bytes at
  each end copied from a 1-KiB pattern buffer the encoder makes once.
- `metal2vulkan-degrade-staged-texture-copy` (performance). `StagedTextureCopy`: A blit copies
  between a buffer and a texture in a layout Vulkan cannot state. Cost: the rows go through a
  tightly packed scratch buffer, which its creation clears by a fill it waits for: two copies
  instead of one.
- `metal2vulkan-degrade-overlapping-texel-copy` (performance). `OverlappingTexelCopy`: A blit copies
  texel rows of a texture over a buffer to or from bytes of the same buffer that they share, which
  Vulkan's copy forbids. Cost: the rows go through a tightly packed scratch buffer, which its
  creation clears by a fill it waits for: two copies instead of one.
- `metal2vulkan-degrade-split-dispatch` (performance). `SplitDispatch`: A dispatch is more than one
  Vulkan dispatch: its grid is not a whole number of threadgroups, or a group count is past the
  device's `maxComputeWorkGroupCount`. Cost: one Vulkan dispatch per region (the interior and each
  partial edge, per chunk of group counts), each on its own specialized pipeline, a barrier between
  each two.
- `metal2vulkan-degrade-emulated-simdgroups` (performance). `EmulatedSimdgroups`: A dispatch whose
  threadgroups are not whole subgroups of the kernel's pinned subgroup size runs on the kernel's
  emulated-simdgroups variant. Cost: a second compile of the kernel whose simdgroup operations go
  through threadgroup memory between barriers.
- `metal2vulkan-degrade-separate-depth-stencil` (performance). `SeparateDepthStencil`: A pass of
  separate depth and stencil textures, or of a stencil-only texture alone at custom sample
  positions, renders into a scratch image of a combined format. Cost: each loaded aspect is copied
  into the scratch image before the pass and each stored one back after it, through a staging
  buffer, or a multisampled colour image for multisampled textures, or sample-rate draws where the
  device's copies of those are missing or wrong.
- `metal2vulkan-degrade-one-aspect-resolve` (performance). `OneAspectResolve`: A pass resolves a
  combined depth/stencil attachment one aspect at a time, on a device that resolves the two aspects
  only together or only in one mode, or where its aspects resolve into different textures. Cost: a
  scratch image per aspect, the second aspect resolved after the pass in a render pass of its own
  from the stored samples, and each resolved aspect copied into the resolve texture after the pass
  through a staging buffer.
- `metal2vulkan-degrade-resolve-scratch` (performance). `ResolveScratch`: A pass resolves into a
  texture that cannot be an attachment (its usage lacks `RenderTarget`), or a layered pass into a
  texture whose slices from the resolve slice hold fewer layers than the pass's. Cost: the pass
  resolves into a scratch image, whose render area is copied into the texture after the pass.
- `metal2vulkan-degrade-shader-resolve` (performance). `ShaderResolve`: A multisample color resolve
  drawn after the pass by Metal's exact rules, where the device's `AVERAGE` resolve is not measured
  to be Metal's; or a stencil resolved by `DepthResolvedSample` after `Min` or `Max`, which no
  Vulkan resolve mode states, or a `Min` or `Max` depth on a device without the mode, drawn after
  the pass. Cost: one more render pass instance after the pass reads the stored samples (or the
  sample planes) and writes the resolve texture, where the device would resolve in the pass.
- `metal2vulkan-degrade-shader-color-conversion` (performance). `ShaderColorConversion`: A fragment
  function writes a float to a normalized, sRGB or half-float color attachment on a device whose own
  conversion is not Metal's (`StageModule::rounded_colors`). Cost: the fragment function rounds each
  such channel to the code Metal stores before it writes it: about 35 more instructions per
  normalized or half-float channel, and about 70 and one read of a 4 KiB table per sRGB channel.
- `metal2vulkan-degrade-shader-half-color-bits` (performance). `ShaderHalfColorBits`: A fragment
  function writes a `half` to a half-float color attachment as the bits of the half Metal stores,
  through a 16-bit output, on a device that does not store a 32-bit output's NaN as Metal does
  (`StageModule::half_bit_colors`). Cost: the fragment function writes each such output as its
  half's bits through a 16-bit output: about 20 more instructions per output, and the attachment
  takes no Vulkan blend of one and zero.
- `metal2vulkan-degrade-render-pass-split` (performance). `RenderPassSplit`: A Metal render pass
  runs as several Vulkan render pass instances: a framebuffer fetch through the pixel interlock
  keeps its attachment out of its draws' subpass, and the pass's draws leave different attachments
  out, or its last draws leave out an attachment that resolves, which resolves in one more instance;
  or a fetch through input attachments reads an attachment whose image takes no input attachment
  usage (a caller's) after other draws recorded, and the stand-in that has it makes it an input
  attachment of the later draws' subpass alone; or a tile dispatch runs between two instances; or
  some of its draws render some of its layers, each run of them in an instance over those layers; or
  a memory barrier over buffers or textures orders its draws after it behind those before it, which
  a barrier between instances does. Cost: each instance after the first loads the attachments and
  each before the last stores them, with a barrier from everything before to everything after
  between them: on a tiler, a tile flush per split.
- `metal2vulkan-degrade-amplification-draws` (performance). `AmplificationDraws`: A draw under
  vertex amplification that Vulkan multiview cannot express (view mappings that offset the viewport
  or name layers out of order, more amplifications than the device's views, or a pipeline that
  chooses or reads its render target slice) draws one amplification at a time
  (`bind::DrawCommands::amplifications`). Cost: one Vulkan draw per amplification, each after the
  amplification's push data, on the pipeline state's variant that reads it, where multiview would
  draw them all at once.
- `metal2vulkan-degrade-memoryless-backed` (performance). `MemorylessBacked`: A memoryless
  attachment renders into a backing image of the device's own memory, because a framebuffer fetch
  through the pixel interlock reads it or the pass splits; not for a split `render-pass-split`
  leaves unannounced. Cost: a scratch image per pass: memory, and the bandwidth to write and read
  it, that Metal does not spend on a memoryless texture.
- `metal2vulkan-degrade-fetch-through-scratch` (performance). `FetchThroughScratch`: A color render
  target that a framebuffer fetch reads lacks the usage the fetch's form needs, so its pass renders
  into a scratch image that has it: `STORAGE` in the pixel interlock, or for its sample planes
  sampled usage with a view of its texel size's unsigned-integer format, either of which the runtime
  gives a texture only where it leaves the image's layout (its compression) alone; or input
  attachment usage, which a caller's image may lack. Cost: a scratch image per pass, the texture
  copied into it before the pass on a `Load` and back after it on a `Store`.
- `metal2vulkan-degrade-fetched-target-storage` (performance). `FetchedTargetStorage`: A runtime
  texture that a pass rendered into a scratch image for want of `STORAGE` (`fetch-through-scratch`
  through the pixel interlock) took its storage form: its image made again with `STORAGE` and its
  texels copied in once, at the commit of the device's only uncommitted command buffer; with a
  submission check, once that command buffer's checks ran at its submission and every encoder of it
  ran. Cost: the texture's image laid out as one with `STORAGE` usage from then on (on MoltenVK
  `ShaderWrite`, without lossless compression), and one copy of its texels, where Metal's texture
  keeps its usage.
- `metal2vulkan-degrade-positions-through-scratch` (performance). `PositionsThroughScratch`: A
  depth/stencil render target whose image was not made for the pass's custom sample positions (a
  caller's image without `VK_IMAGE_CREATE_SAMPLE_LOCATIONS_COMPATIBLE_DEPTH_BIT_EXT`, which Vulkan
  needs to rasterize a depth/stencil attachment at them), or was on a driver that keeps a depth
  rendered at them per sample only through a copy
  (`metal2vulkan::device::DriverFact::PositionedDepthReadAtStandardPositions`: NVIDIA's, measured),
  renders into a scratch image made for them. Cost: a scratch image per pass, each aspect copied
  into it before the pass on a `Load` and back after it on a `Store`.
- `metal2vulkan-degrade-sample-planes` (performance). `SamplePlanes`: A framebuffer fetch through
  the pixel interlock in a multisampled pass, on a device without multisampled storage images, reads
  and writes the attachment as its sample planes: a single-sampled array image of one layer per
  sample. Cost: a planes image per fetched attachment and pass; a compute dispatch copies the
  samples into it before the draws that fetch (on a `Load`, or after other draws rendered it) and a
  sample-rate pass copies them back after those draws (before other draws render it, and where the
  pass keeps or resolves it).
- `metal2vulkan-degrade-shader-side-stencil` (performance). `ShaderSideStencil`: A render pass's
  draws test and write the stencil in the fragment shader, on a device without
  `VK_EXT_shader_stencil_export`: the function's `[[stencil]]` output is Metal's stencil reference,
  which only that extension hands the fixed-function test. Cost: an 8-bit shadow image of the
  stencil per pass, which the stencil is copied into before each run of such draws and back from
  after it (through a staging buffer at one sample), each run in a render pass instance of its own,
  its fragment shader testing and writing the shadow in the pixel interlock.
- `metal2vulkan-degrade-tile-dispatch-scratch` (performance). `TileDispatchScratch`: A tile dispatch
  stages its pass's attachments through storage views in their staging formats, which it takes of a
  scratch image of each attachment's format: each is copied into its scratch before the dispatch and
  back after it. Cost: a scratch image per staged attachment and pass, over the pixels the dispatch
  covers, which each dispatch copies the attachment into before it and back from after it, between
  barriers from everything before to everything after: memory and bandwidth that Metal's tile memory
  does not spend.
- `metal2vulkan-degrade-fragment-imageblock-storage` (performance). `FragmentImageblockStorage`: A
  render pass's draws' fragment functions declare an explicit imageblock, which they read and write
  in a storage image of the render area. Cost: an R32Uint array image per pass over the render area,
  a layer per 4 bytes of a pixel's imageblock, cleared before the pass, which the draws read and
  write in the pixel interlock: memory and bandwidth that Metal's tile memory does not spend.
- `metal2vulkan-degrade-wrapped-image-layout-transitions` (performance).
  `WrappedImageLayoutTransitions`: A command buffer uses a texture over the caller's image whose
  home layout is not `GENERAL`: every image the runtime uses rests in `GENERAL`. Cost: two
  whole-image barriers per batch that uses it: into GENERAL before its first encoder, and back to
  its home layout at its end.
- `metal2vulkan-degrade-host-replayed-indirect-commands` (performance).
  `HostReplayedIndirectCommands`: An execution of an indirect command buffer that an encoder made
  resident for writing: the GPU may write the records it runs, so they are known only once the work
  before it completed. Cost: one GPU-to-CPU round trip per execution: its command buffer's work
  before it, and every batch submitted before that on any queue, completes before the CPU records
  its commands from their records and submits the rest; the device's scheduler submits nothing else
  while it records them, and the first command of a threadgroup size or threadgroup lengths its
  pipeline state has not dispatched makes that VkPipeline then. The commands run in the large-buffer
  form (their range-limited buffer slots at device addresses, with in-shader bounds checks), whose
  variants the commit compiles for every pipeline state a record may name (or, for an execution
  committed before the buffer was made resident for writing, the use_resource call that made it so);
  without device addresses, a command's first buffer shape compiles its variant then.
- `metal2vulkan-degrade-uncalibrated-gpu-times` (precision). `UncalibratedGpuTimes`: A command
  buffer's GPU times on a device that calibrates its timestamps against no host clock the runtime
  reads (no `VK_KHR_calibrated_timestamps` or `VK_EXT_calibrated_timestamps` with the device's time
  domain and a monotonic host one), or whose calibration failed at the commit. Cost: the end is the
  host time the runtime saw the command buffer complete, late by that latency, and the start the
  GPU's exact duration before it.
- `metal2vulkan-degrade-untimed-command-buffer` (precision). `UntimedCommandBuffer`: A command
  buffer's GPU times on a device whose timestamp pools had no slot free at the command buffer's
  commit, and which could make no pool more: every slot was held by a command buffer in flight.
  Cost: its GPU start and end times are refused (`Unsupported`, `GpuTimes`), where Metal gives them;
  its kernel times stay.
- `metal2vulkan-degrade-gpu-times-from-host` (precision). `GpuTimesFromHost`: A command buffer's GPU
  times on a device told to answer them from the host (`DeviceOptions::gpu_times_from_host`,
  `RawDevice::gpu_times_from_host`), or whose driver's timestamps wait for every queue's
  (`DriverFact::TimestampsWaitForEveryQueue`: MoltenVK 1.4.1 on Apple's GPUs), which writes no GPU
  timestamps. Cost: its GPU start is its kernel end time (or its queue's latest GPU end before it)
  and its GPU end the host time the runtime saw it complete: a window its work ran in, which holds
  the submission's and the completion's latencies, not the GPU's own start and end.
- `metal2vulkan-degrade-device-rounded-resolve` (precision). `DeviceRoundedResolve`: A multisample
  color resolve averages through Vulkan's `AVERAGE` mode, which leaves the rounding of the average
  (and the sRGB encoding) to the device, on a device not measured to round as Metal does: a format
  without Metal's closed rule at its count (half floats past 2 samples, packed 16-bit,
  `RG11B10Float`, `RGB9E5Float`), or an attachment whose image the exact resolve cannot read. Cost:
  a resolved texel can differ from Metal's by one unit where the average lies at a rounding tie, and
  wholly where a sample is a NaN or an infinity: Metal's resolve counts a half or small-float NaN as
  0 and a half infinity as 65536 of its sign, and makes any 32-bit float NaN 0x7fc00000, where the
  device's average may keep them.
- `metal2vulkan-degrade-unmeasured-resolve` (precision). `UnmeasuredResolve`: A multisample color
  resolve of more than 4 samples, of which Apple's GPUs make no texture: no Metal answer is measured
  there; likewise a stencil resolved by `DepthResolvedSample` after `Min` or `Max`, whose pick among
  equal depths is measured at 2 and 4 samples. Cost: the resolve applies the rules Metal's resolve
  follows at 2 and 4 samples (or the device's average, for a format without them), which no Metal
  answer confirms at this count.
- `metal2vulkan-degrade-dont-care-store` (precision). `DontCareStore`: An attachment stored with
  `DontCare` (or the multisampled attachment of `MultisampleResolve`) is Vulkan's `DONT_CARE`, where
  Apple's GPU keeps the earlier texels unless its tiler overflows. Cost: the attachment's texels
  after the pass are each its earlier texel or the one the pass rendered, both Metal's answers, but
  not always the one Metal's tiler keeps.
- `metal2vulkan-degrade-dont-care-load-tiles` (precision). `DontCareLoadTiles`: An attachment loaded
  with `DontCare` and stored, in a pass that draws, is cleared before the draws to zero (a color) or
  its clear value (depth, stencil) over the render area and the tiles past it a clear reaches, where
  Metal clears only the tiles a draw's primitives touch and keeps the texels of the others. Cost:
  the tiles no draw touches hold zero or the clear value where Metal keeps their texels.
- `metal2vulkan-degrade-fresh-pattern-zeroed` (precision). `FreshPatternZeroed`: A new `Private`
  texture Metal keeps losslessly compressed reads a fixed pattern in Metal, which the device cannot
  write exactly into its image: a NaN depth where a copy into a depth image may store it clamped or
  the image is multisampled, a NaN texel of a multisampled image (whose bits a clear leaves to the
  device), or an image holding other bytes than Metal's texel. Cost: the new texture reads zero
  where Metal reads its fixed pattern until a write.
- `metal2vulkan-degrade-zeroed-index-fetch` (precision). `ZeroedIndexFetch`: An indexed draw from an
  index offset at or past its index buffer's length draws index 0 plus the base vertex at each
  position, a direct draw from a zero-padded copy and an indirect one through the index bounds pass,
  where Metal reads the memory past the buffer. Cost: each index position draws index 0 plus the
  base vertex, where Metal reads whatever memory follows the buffer.
- `metal2vulkan-degrade-implementation-defined-reconvergence` (precision).
  `ImplementationDefinedReconvergence`: A pipeline stage's reconvergence after divergence is
  implementation-defined on the device, where Metal's is maximal
  (`Reconvergence::ImplementationDefined`). Cost: simdgroup operations after divergent control flow
  can see other lanes active than Metal's.
- `metal2vulkan-degrade-resource-id-takeover` (precision). `ResourceIdTakeover`: A function can read
  a resource ID whose holder the runtime cannot give it as Metal's command buffer reads it: a
  sampler state was made while a committed command buffer not yet completed lists the argument
  samplers as it committed. Cost: a function reading the ID can see an earlier holder of it, or the
  neutral resource, where Metal's sees the resource holding it as the command buffer runs.
- `metal2vulkan-degrade-unlisted-argument-sampler` (precision). `UnlistedArgumentSampler`: A sampler
  state whose element no set-1 sampler array can hold, as more states live than it holds: the
  device's, as the state is made, or a pipeline's, as a command lists the argument samplers or its
  command buffer lists them again as it commits. Cost: a command buffer whose commands list the
  argument samplers while the state lives fails (UnlistedArgumentSampler), where Metal's function
  reads the state through its ID.
- `metal2vulkan-degrade-depth-twin` (performance). `DepthTwin`: A `Depth32Float` texture the program
  only reads or samples, on a device whose driver is not measured to keep every `float` in a depth
  image, is held in an `R32_SFLOAT` image with a `D32_SFLOAT` twin for the comparisons the device
  makes. Cost: a second image of its size, which every copy into it writes too; a copy into it from
  a depth texture without a twin goes through a scratch buffer.
- `metal2vulkan-degrade-restricted-depth-upload` (precision). `RestrictedDepthUpload`: An upload
  (`replace_region`, a blit from a buffer) into the 32-bit float depth of a texture's depth image,
  on a device without `VK_EXT_depth_range_unrestricted` enabled, where Vulkan leaves a value outside
  `[0, 1]` undefined, whose driver is not measured to keep one
  (`DriverFact::DepthImageKeepsValuesOutsideUnitRange`): a texture without a twin. Cost: a value
  outside [0, 1] may be stored clamped to [0, 1], where Metal's keeps it, so reads, samples,
  comparisons and depth tests of it can differ.
- `metal2vulkan-degrade-host-read-dispatch-arguments` (performance). `HostReadDispatchArguments`: An
  indirect dispatch, or a dispatch after `setStageInRegionWithIndirectBuffer:indirectBufferOffset:`:
  its arguments or its stage-in region are GPU memory the work before it may write, so they are
  known only once that work completed. Cost: one GPU-to-CPU round trip per dispatch: a copy of its
  arguments where it is encoded, then its command buffer's work before it, and every batch submitted
  before that on any queue, completes before the CPU reads them and records the dispatch they state;
  the device's scheduler submits nothing else meanwhile, and a threadgroup size its pipeline state
  has not dispatched makes that VkPipeline then.
- `metal2vulkan-degrade-non-strict-residency` (precision). `NonStrictResidency`: A sparse texture on
  a device whose sparse residency is not `residencyNonResidentStrict`. Cost: a read of a texel no
  mapped tile holds, by a function or a copy, answers the device's value where Metal's reads zero,
  and a write to one may be kept where Metal's drops it.
- `metal2vulkan-degrade-host-read-mapping-arguments` (performance). `HostReadMappingArguments`: An
  indirect mapping update of a sparse texture, whose regions the host reads. Cost: one GPU-to-CPU
  round trip per batch of such updates: the command buffer's work before it, and every batch
  submitted before that on any queue, completes before the CPU reads the regions (through a copy for
  a Private buffer) and binds them; the device's scheduler submits nothing else meanwhile.
- `metal2vulkan-degrade-emulated-sparse-texture` (performance). `EmulatedSparseTexture`: A sparse
  texture the device makes no sparse image of (MoltenVK binds no sparse memory), where no function
  reads an image's own residency (the device does not enable `shaderResourceResidency`): an ordinary
  image whose unmapped tiles the runtime keeps at zero, which gives Metal's answers. Cost: the
  texture holds its whole image's memory from when it is made, where Metal's backs only its mapped
  tiles; each batch that changes its mappings zeroes the tiles it unmaps and rewrites its tile map
  before its work, and each blit that writes it zeroes the unmapped tiles it wrote.
- `metal2vulkan-degrade-shared-in-system-memory` (performance). `SharedInSystemMemory`: A `Shared`
  buffer or heap the program makes with the default CPU cache mode, on a device without unified
  memory (`DeviceProperties::has_unified_memory`), lies in host-cached system memory, which the host
  reads quickly, as a shared resource's on a Mac with a discrete GPU. Cost: the GPU reads and writes
  it across the bus, as a discrete Mac's GPU does a shared resource: coalesced accesses at the bus's
  speed, and scattered writes far slower (4-byte stores 12 bytes apart at 0.08 GB/s on an RTX 3090,
  where device-local memory takes them at 850 GB/s).
- `metal2vulkan-degrade-mipmaps-within-spread` (precision). `MipmapsWithinSpread`:
  `generate_mipmaps` through the device's blit chain, which the runtime's check accepted for the
  texture's pixel format within the spread Metal's own GPUs show: Metal has no one answer, an Apple
  M3 Max's and an Apple M4's differing by up to 1 unit in the last place. Cost: each level is the
  device's linear blit of the one above, which on the check's inputs lay within 1 unit in the last
  place (2 for 32-bit float) of an Apple M3 Max's or an Apple M4's answer, not at either's exactly;
  other inputs are not checked.
- `metal2vulkan-degrade-heap-depth-stencil-apart` (precision). `HeapDepthStencilApart`: A depth or
  stencil texture made in a heap on a device whose driver lays depth and stencil images apart from
  the memory they are bound to that no shadow of the heap's bytes serves: a multisampled texture,
  which Vulkan copies to and from no buffer, one whose texels do not fit its bytes, or one with a
  depth twin. Cost: the texture's texels lie apart from the heap's bytes: a write through a buffer
  or texture over its bytes changes none of them, and its writes change none of those resources'
  bytes, where Metal's texture aliases them.
- `metal2vulkan-degrade-heap-texture-host-transfer` (performance). `HeapTextureHostTransfer`: A
  colour texture made in a heap on a device whose driver's compressed images hide writes through an
  alias, whose image takes host-transfer usage, which the driver keeps uncompressed in the heap's
  bytes (`metal2vulkan::device::DriverFact::HostTransferImagesKeepTheirBytes`). Cost: the texture's
  image is kept uncompressed, so writes through the heap's buffers and textures over its bytes show
  in it as Metal's do: a clear of it loses the driver's fast clear (334.5 us where a compressed
  2048x2048 RGBA8Unorm image's took 4.1 us on an RPL-S), while its copies, sampling and blending
  measured no slower.
- `metal2vulkan-degrade-heap-texture-linear` (performance). `HeapTextureLinear`: A colour texture
  made in a heap on such a device whose image takes no host-transfer usage (the device did not
  enable `hostImageCopy`, or an adopted device's consumer did not say so, or the format or shape
  takes none there): a linear image, which the driver does not compress, where one does all the
  optimal image does for the texture. Cost: the texture's image is linear, so writes through the
  heap's buffers and textures over its bytes show in it as Metal's do: its reads and writes go
  without the optimal tiling's locality and compression (3.9 to 9.6 times the host-transfer image's
  time for copies, clears and draws of a 2048x2048 RGBA8Unorm texture on an RPL-S).
- `metal2vulkan-degrade-heap-texture-hides-aliased-writes` (precision).
  `HeapTextureHidesAliasedWrites`: A colour texture made in a heap on such a device whose image
  takes neither host-transfer usage nor linear tiling: an optimal image the driver may compress, a
  near-equivalent recorded open. Cost: the driver may compress the texture's image: a write through
  a buffer or texture over its bytes may leave texels as they were, and its writes may show in those
  resources' bytes compressed, where Metal's texture aliases them byte for byte.
- `metal2vulkan-degrade-combined-archive-stages` (precision). `CombinedArchiveStages`: A render
  creation that fails on a binary-archive miss, made from one add's vertex stage and another add's
  fragment stage: Metal's vertex key holds the vertex outputs its compiler finds the fragment
  reading, which the runtime's lookup does not decode (`binary_archive::metal_key`). Cost: the
  pipeline is made where Metal may refuse it with a binary-archive miss: Metal misses when the two
  adds' fragment functions read different vertex outputs.
- `metal2vulkan-degrade-shadowed-texture-view` (performance). `ShadowedTextureView`: A texture view
  is an image of its own, a shadow of the viewed image's texels, where that image takes no Vulkan
  view of its format: a view of another component layout on a device whose Vulkan views reinterpret
  no image's components (a portability subset without `imageViewFormatReinterpretation`, or an
  adopted device whose consumer did not say it enabled it), or a view of another format than the
  texture's own and its sRGB twin's of a texture made without `PixelFormatView` usage. Cost: its
  texels are copied from its texture's image before each use and back after each write.
- `metal2vulkan-degrade-mipmaps-as-apple-m4` (precision). `MipmapsAsAppleM4`: `generate_mipmaps`
  computed as Metal's filter where an Apple M3 Max's answer may differ from the chain's
  (`MipmapFilterChain::unlike_an_m3`): a 2D texture, array or cube of three levels or more whose M3
  Max passes are not known, an `A1BGR5Unorm` 2D texture, an `ABGR4Unorm` one of three levels or
  more, or a 3D texture of a format summed as half floats. Cost: each level is the Apple M3 Max's
  but where its answer is not known: below the first level of a 2D chain whose M3 Max passes are not
  known an Apple M4's filter, and a few measured values of the 4-bit and 1-bit formats and of 3D
  half-float sums one unit off.
- `metal2vulkan-degrade-repeated-execution-start` (precision). `RepeatedExecutionStart`: A render
  encoder executes an indirect command buffer's commands from a command it executed them from before
  with another length, each range given (`executeCommandsInBuffer:withRange:`) or read from a
  buffer: Metal's executions then do not each run their own range, and not the same way from run to
  run, on an Apple M3 Max and an Apple M4 alike, so no stable answer exists. Cost: each execution
  runs its own range, as Metal documents, where Metal's GPU may run one execution with the other's
  length, so commands run more or fewer times than the ranges name, differently from run to run.
- `metal2vulkan-degrade-private-resource-option` (precision). `PrivateResourceOption`: A resource
  made with `MTLResourceOptions` bit 24, a private option Metal takes
  (`ResourceOptions::PRIVATE_BIT`), is made as without it, and reports its options without it, as
  Metal's does. Cost: the private option has no effect here: on Metal a no-copy buffer made with it
  kept the GPU's writes from the CPU (RT4 probe nc_bit24), where the runtime's reach it.
- `metal2vulkan-degrade-optimized-depth-stencil-texture` (precision).
  `OptimizedDepthStencilTexture`: A depth or stencil texture with `allowGPUOptimizedContents` off
  sized or made, outside memoryless storage, on a device whose driver keeps depth and stencil images
  GPU-optimized whatever their create info: Metal lays the texture out without its compression
  metadata, and the runtime's image keeps it. Cost: the texture keeps its GPU-optimized layout:
  heap_texture_size_and_align answers Metal's size for the texture with the property on, larger than
  Metal's with it off (16512 bytes for a 64x64 Depth32Float texture, where Metal answers 16384), so
  a heap sized by Metal's answers holds fewer such textures.
- `metal2vulkan-degrade-narrow-surface-rows` (precision). `NarrowSurfaceRows`: A texture of an
  IOSurface plane of more than one row under 16 bytes a row: Metal's GPU writes its rows 16 bytes
  apart and reads every row past the first as memory-dependent bytes (zeros in a fresh process;
  after a write, bytes neither the surface's nor written). Cost: a GPU read of a row past the first
  reads the bytes the GPU's writes put there, 16 bytes a row apart, where Metal's GPU reads
  memory-dependent bytes; CPU access and the GPU's writes and first row are Metal's.
- `metal2vulkan-degrade-absent-argument-read` (precision). `AbsentArgumentRead`: A pipeline state's
  function may read an argument its function constants removed (a texture's read, sample, gather,
  query or atomic, or a sample through a removed sampler; `RuntimeFact::AbsentArgumentRead`), which
  Metal gives no value: its compiler drops or folds such a read, and with its debug layer every such
  read is zero. Cost: such a read takes zero, Metal's answer under its debug layer, and the command
  buffer completes, its execution report stating the read (absent_argument_read); without the layer
  Metal's compiler drops or folds the read instead.
- `metal2vulkan-degrade-mixed-depth-stencil-resolve` (precision). `MixedDepthStencilResolve`: A
  render pass whose one depth/stencil texture stores and resolves its depth
  (`StoreAndMultisampleResolve`) while it resolves its stencil alone (`MultisampleResolve`), which
  Metal's validation stops (`resolve::layout_patterned`). Cost: both aspects resolve and the depth's
  samples are kept, as the store actions document, and the stencil's samples are kept too, where
  Metal's GPU overwrites the depth's samples and its resolve texture with a layout pattern of the
  resolved depth and leaves the stencil's texels (measured on an Apple M4).
- `metal2vulkan-degrade-capped-anisotropy` (precision). `CappedAnisotropy`: A sampler state whose
  `maxAnisotropy`, which Metal's validation stops past 16, plain Metal samples with an anisotropy
  past the device's largest (`maxSamplerAnisotropy`): 17 to 31 in its low five bits, which Metal
  samples with 32. Cost: it filters with the device's largest anisotropy (16 on every venue
  measured), where Metal filters with 32: a footprint more than 16 times as long as it is wide reads
  a coarser level than Metal's (one level coarser at 32 to 1 and beyond).
- `metal2vulkan-degrade-unreadable-handle-texture` (precision). `UnreadableHandleTexture`: A command
  whose function reads texture handles from memory (an argument buffer, or any memory holding a
  texture's resource ID) has a texture the runtime holds as raw blocks for copies made resident, or
  bound in a stage that reads handles: the handle's element is the neutral texture, as the runtime
  does not know which of the textures it reaches the function reads. Cost: a read or sample of the
  texture through its handle reads the neutral texture where Metal's decodes its blocks; a function
  that reads it at a slot is refused (Unsupported(Format)).
- `metal2vulkan-degrade-unnamed-texture-unviewable` (precision). `UnnamedTextureUnviewable`: A
  texture that a heap array's class fits, but whose format the array cannot view (one the device
  cannot sample, or store without a format, or one whose raw texels need its function's decode): the
  device's heap set holds the class's neutral at its ID, which a command that names the texture
  refuses instead. Cost: a function reading the texture through its ID, in a command that does not
  name it (bind it, or make it resident), reads the neutral texture where Metal's reads the texture.
- `metal2vulkan-degrade-unnamed-shadowed-texture` (precision). `UnnamedShadowedTexture`: A shadowed
  texture in the device's heap set. Cost: a function reading the texture through its ID, in a
  command that does not name it, reads the texels an encoder that named it last copied in (undefined
  before any did), where Metal's reads the texels as they stand.
- `metal2vulkan-degrade-unnamed-textures-past-capacity` (precision). `UnnamedTexturesPastCapacity`:
  A live texture whose resource ID is past the elements of a pipeline's set-1 heap arrays (their
  count, the device's update-after-bind limits shared among the pipeline's arrays), so the texture
  table a command binds misses it. Cost: a function of such a pipeline reading the texture through
  its ID reads the neutral texture where Metal's reads the texture; a command that names it is
  refused (Bind(HeapTooLarge)).
- `metal2vulkan-degrade-heap-set-copied` (precision). `HeapSetCopied`: A heap set on a device
  without `descriptorBindingUpdateUnusedWhilePending`, which writes no set a pending command buffer
  binds: each command buffer binds a copy of its own, written as it first binds the set and until it
  commits. Cost: each command buffer's first command that binds the set writes a copy of it (every
  element a texture's ID may name, per array); a function reading a texture through its ID reads the
  texture that held the ID as its command buffer committed (the neutral texture where none did, or
  where it was released before the command buffer's first submission), where Metal's reads the one
  holding it as the function runs.
- `metal2vulkan-degrade-uncovered-color-kept` (precision). `UncoveredColorKept`: A draw of a render
  pipeline state in a pass with a color at no place of the state's per-sample record, under a state
  color that blends through a destination factor of the fragment's output (`SourceAlpha`,
  `OneMinusSourceAlpha`, `SourceColor`,...) and reads the destination no other way. Cost: the pass's
  color keeps its contents where the draw covers it, as Metal keeps them for an output it computes;
  where Metal's compiler folds the output to a constant that zeroes the factor (an alpha of the
  literal 1 under `OneMinusSourceAlpha`), Metal stores zero there (measured on an Apple M4).
- `metal2vulkan-degrade-drawn-depth-copy` (precision). `DrawnDepthCopy`: A blit copy into a
  multisampled 32-bit float depth texture from a colour one, or a render pass's separate
  multisampled 32-bit float depth texture copied to and from its scratch image, by the sample-rate
  `FragDepth` draw, on a device whose copies go by draws and whose driver's fragment depth is
  measured to change some values (`depth_copy::drawn`: MoltenVK, NVIDIA's). Cost: each sample's
  depth is written by a fragment-depth draw, where Metal copies its bytes: the values the driver's
  fragment depth changes, which the reason names, land changed; every other value lands as Metal's
  copy leaves it.
- `metal2vulkan-degrade-undeclared-usage-shadow` (performance). `UndeclaredUsageShadow`: A texture
  bound by a slot as a storage image without `ShaderWrite` usage, or as a sampled one without
  `ShaderRead`, whose image lacks that usage because adding it would cost the image its size or the
  driver's compression: the binding takes an image of its own with both shader usages, a shadow of
  the texture's texels. Cost: its texels are copied into an image of its own before each command
  that binds it beyond its declared usage, and back after each write through it.
- `metal2vulkan-degrade-heap-two-plane-texture` (precision). `HeapTwoPlaneTexture`: A texture of
  `RGB10A8_2P_XR10` or its sRGB twin made in a heap other than a 2D texture of one level, whose
  planes the heap's bytes keep: its one image holds each texel's colour word and alpha byte side by
  side, more than Metal's two planes take in the heap, so it lies in memory of its own. Cost: the
  texture's size and alignment answers are Metal's and its reads, writes and copies as outside a
  heap, but its texels lie in memory of its own, apart from the heap's bytes: a buffer or texture
  over its range of the heap neither sees nor changes them, where Metal's texture keeps its two
  planes there.
- `metal2vulkan-degrade-moltenvk-argument-buffers-unread` (precision).
  `MoltenVkArgumentBuffersUnread`: A MoltenVK device at a release whose `pipelineCacheUUID` layout
  the runtime has not read: whether MoltenVK binds through Metal argument buffers is unknown, so a
  device without them is not refused (as at a read release). Cost: whether MoltenVK binds through
  Metal argument buffers is not read at this release: with MVK_CONFIG_USE_METAL_ARGUMENT_BUFFERS
  off, Metal's compiler can crash on large fragment functions' pipelines, and argument buffers hold
  fewer resources than Metal's.
- `metal2vulkan-degrade-two-plane-texture-copy` (precision). `TwoPlaneTextureCopy`: A blit between
  two textures of `RGB10A8_2P_XR10` or its sRGB twin whose answer is not Metal's measured one: one
  the runtime copies texel for texel (textures of more than one level or slice, a copy within one
  texture, a level larger than a storage buffer), or one whose rows overlap in the destination, or
  of a texture narrower than 4 texels, where Metal's blit writes in an order that varies or reads
  bytes from outside the planes, or one reading bytes outside the source's texels that an earlier
  blit wrote there (or a heap's texture's), which Metal's blit keeps and reads back. Cost: where
  Metal's answer was measured (2D textures of one level and one slice) the copy moves both planes'
  bytes as Metal's blit does, at its own rows; here it copies both words of each texel, where
  Metal's blit moves bytes at those rows and so loses or moves texels, or it keeps the later row
  where the copy's rows overlap (Metal's blit writes them in an order that varies), or it reads as 0
  the bytes Metal's blit reads from outside the texels: from outside the planes below a width of 4,
  and those an earlier blit left in a row's padding or past a plane, which Metal keeps.
- `metal2vulkan-degrade-extended-range-bits-kept` (precision). `ExtendedRangeBitsKept`: A draw in a
  render pass that loads a color attachment of an extended-range or padded 10-bit format
  (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12` and their sRGB twins)
  and does not discard it at the store, where the pass's load pass does not take it: a multisampled
  attachment, a texture held as a shadow of another image or of a heap's bytes, a sparse one, or one
  whose image the device does not copy both ways (a padded format has none of these). Cost: the
  texels the pass's draws do not write keep their bytes; Metal's pass keeps their codes but rewrites
  the bits they leave unused in each tile a draw reaches (BGR10_XR's bits 30 and 31 to ones,
  BGRA10_XR's low six bits of each field to zeros, the two-plane color word's top two bits to zeros)
  and clamps BGRA10_XR's alpha code to [384, 894]: a texel whose bytes no Metal write left (uploaded
  bytes, or an alpha code past 894) keeps bytes Metal's pass would change.
- `metal2vulkan-degrade-two-plane-buffer-blit` (precision). `TwoPlaneBufferBlit`: A blit between a
  buffer and a level of `RGB10A8_2P_XR10` or its sRGB twin whose width is not a multiple of 4: it
  moves the texels at the rows the texture holds them in, where Metal's blit addresses the level's
  rows `⌊4 w / 16⌋ · 16` bytes apart, short of its rows, and so moves texels into and out of other
  rows (into the texture overlapping rows, in an order that varies). Cost: the texels move between
  the buffer's rows and the texture's as getBytes and replaceRegion move them; Metal's blit
  addresses the texture's rows floor(4w/16)*16 bytes apart where they lie align(4w,16) bytes apart,
  so out of the texture it reads texels of earlier rows and padding (0) into later rows of the
  buffer, and into it writes rows over each other in an order that varies.
- `metal2vulkan-degrade-unadvertised-vertex-simdgroups` (validity). `UnadvertisedVertexSimdgroups`:
  A render pipeline state whose vertex function runs simdgroup functions on a driver measured to run
  a vertex function's subgroup operations as Metal does though its `subgroupSupportedStages` leaves
  the vertex stage out (`metal2vulkan::error::DriverDeviation::MetalVertexSimdgroups`): its pipeline
  requires `VkProperty::MetalVertexSimdgroups` in place of the vertex stage's subgroups. Cost: the
  device runs them as Metal does, but its driver leaves the vertex stage out of
  subgroupSupportedStages, so Vulkan's validation layer reports each shader module of the function
  (VUID-RuntimeSpirv-None-06343).
- `metal2vulkan-degrade-released-command-state` (precision). `ReleasedCommandState`: An indirect
  command naming the ID of a pipeline state the program released is executed where the runtime
  cannot run what Metal runs. Cost: a command naming another ID than the state set on it runs
  nothing, where Metal runs the released state's program; one naming that state's ID runs its
  program, as Metal runs a command the CPU encoded, where for a record the GPU wrote Metal runs the
  ID's holder since (nothing, for a state of the other kind).
- `metal2vulkan-degrade-ycbcr-filter-not-exact` (precision). `YCbCrFilterNotExact`: A pipeline
  stage's filtered sample of a texture slot bound to a two-plane YCbCr texture does not take Metal's
  exact filter of the format: the stage records `PrecisionNote::HalfSampleFilter` or
  `FloatSampleFilter` under the slot's bound format (a format whose filter is not measured, a
  swizzle that moves its channels, a linear mip filter, or a sample the emulation does not reach).
  Cost: the texels hold Metal's reads exactly, but the sample filters them as the device does, off
  the grid Metal sums the format's steps on, so a filtered result can miss Metal's by a step of that
  grid or more.
- `metal2vulkan-degrade-rotated-load-tiles` (precision). `RotatedLoadTiles`: A render pass that
  draws loads and stores a rotated texture of rotation 4, whose load is the plane transposed and
  whose store is rotation 1's: every tile of the frame lands mirrored, where Metal loads and stores
  only the 32x32 tiles a draw's primitives reach and keeps the texels of the others in place. Cost:
  the tiles no draw reaches land mirrored as the tiles it reaches do, where Metal keeps their texels
  in place.
- `metal2vulkan-degrade-encoded-use-rebound` (precision). `EncodedUseRebound`: A command buffer's
  command named a buffer over the caller's host pages (a binding, a blit's operand, an attachment, a
  texture over the pages, an indirect command's resolved buffer) that was rebound
  (`rebind_bytes_no_copy`) after the command was encoded and before the command buffer was first
  submitted, in an encoder the rebind could not encode again. Cost: each of Metal's submissions of
  the command buffer holding such a command runs nothing and the command buffer fails
  (EncodedUseRebound), where Metal resolves the use as it submits the command buffer and reaches the
  pages the buffer lies in then; the rebind encodes compute, blit and render encoders again, but not
  this one.
- `metal2vulkan-degrade-stale-detached-reach` (precision). `StaleDetachedReach`: A command buffer
  that bound a heap set, whose commands may reach any texture through its resource ID without naming
  it, was submitted while a texture's bytes (a no-copy buffer's or heap's, the texture over them or
  in it) were detached (`detach_backing`) after a submitted command buffer made them resident. Cost:
  a read of the texture that does not name it reads zero bytes (the stand-in its detached bytes
  present), where Metal's reads the pages the program gave back, which a correct program never
  relies on.
- `metal2vulkan-degrade-detached-storage-keeps-writes` (precision). `DetachedStorageKeepsWrites`: A
  command buffer that bound a heap set, whose commands may reach any texture through its resource ID
  without naming it, was committed while a texture of storage usage was over detached bytes
  (`detach_backing`; made before the detach or while detached): its stand-in has a sink its
  storage-class elements name. Cost: a read of the texture through a storage-class element (read or
  read_write access) that does not name it reads what an earlier such write left in the stand-in's
  sink, where Metal drops those writes and reads zero; a sampled read, and a size query, are
  Metal's.
- `metal2vulkan-degrade-metal-level-below-host` (precision). `MetalLevelBelowHost`: A device made,
  with no level named in its options, on a macOS host whose release has no measured `MetalLevel`: it
  answers as the newest measured release below the host's. Cost: where Metal's answers differ by
  release (which pixel formats a texture descriptor takes and Metal describes), the device gives the
  older release's: it can stop or refuse a texture the host's Metal makes, and never makes one the
  older release stops.
- `metal2vulkan-degrade-unreached-tiles-rewritten` (precision). `UnreachedTilesRewritten`: A render
  pass that loads a color attachment of an extended-range or padded 10-bit format (`BGR10_XR`,
  `BGRA10_XR`, `RGB10A8_2P_XR10`, `R10Unorm_X6`, `RG10Unorm_X12` and their sRGB twins), stores it
  and draws, whose load pass rewrites every 32 x 32 tile of its render area: Metal's pass rewrites
  only the tiles a draw's primitives reach, and every tile only where a color attachment of the pass
  is cleared and stored (announced where none is, or where the pass renders more than one layer).
  Cost: in tiles no draw reaches the pass rewrites what Metal keeps: the unused bits made canonical
  (BGR10_XR's bits 30 and 31 ones, BGRA10_XR's low six bits of each field and the two-plane color
  word's top two bits zeros) and BGRA10_XR alpha codes clamped to [384, 894], so an out-of-range raw
  alpha reads back changed, and a padded word's low six bits zeroed (an sRGB twin's code moved
  through Metal's table); the tiles a draw reaches are Metal's.
- `metal2vulkan-degrade-zero-layer-map-after-release` (precision). `ZeroLayerMapAfterRelease`: A
  draw in a render pass over a rasterization-rate map of no layers that was made after the device
  released a map of one or more layers. Cost: the draws write nothing, where Metal draws rows of the
  pass (a 64-pixel map's rows 0 to 15, at the positions of a pass with no map).
- `metal2vulkan-degrade-heap-texture-lossless-layout` (precision). `HeapTextureLosslessLayout`: A
  texture made in a heap over host pages (`new_heap_with_bytes_no_copy`) that Metal keeps losslessly
  compressed: its bytes on Metal are no layout of its texels, so the runtime lays them out as its
  own. Cost: the texture's reads, writes and copies are Metal's, but its bytes in the heap are not
  Metal's compressed bytes (a single-sample texture's texels in the runtime's layout; a multisample
  texture's image lies in memory of its own, the heap's bytes keeping what they held): a buffer or
  texture over them reads and writes other bytes than Metal's would.
- `metal2vulkan-degrade-heap-texture-block-compressed-layout` (precision).
  `HeapTextureBlockCompressedLayout`: A texture of a block-compressed format made in a heap over
  host pages whose placement the device's Metal level does not measure: Metal places a format
  family's blocks in each level by a rule that differs by family and by machine, measured for BC,
  ETC2, EAC and ASTC on an Apple M4 under macOS 27.0.1 and an Apple M3 Max under 26.7 (whether the
  GPU or the release makes their difference is not known), where those lie as Metal lays them out,
  and not for PVRTC at any level, whose blocks lie in the runtime's layout. Cost: the texture's
  reads, writes and copies are Metal's, but its blocks lie in the heap's bytes in the runtime's
  layout, where Metal tiles them: a buffer or texture over them meets the blocks at other bytes than
  on Metal.
- `metal2vulkan-degrade-heap-texture-untiled-layout` (precision). `HeapTextureUntiledLayout`: A
  texture made in a heap over host pages that Metal lays out tiled where the runtime does not: a
  layout not measured (a two-aspect depth and stencil format, a PVRTC format), an image that holds
  other bytes than Metal's elements, or moves past the device's limits. Cost: the texture's reads,
  writes and copies are Metal's, but its texels lie in the heap's bytes in the runtime's layout,
  where Metal tiles them: a buffer or texture over them meets the texels at other bytes than on
  Metal.
- `metal2vulkan-degrade-shader-side-depth-test` (performance). `ShaderSideDepthTest`: A render
  pass's draws test the depth in the fragment shader: a fragment function that stores a color
  through the pixel interlock before Vulkan's tests, in a draw whose depth test may drop the
  fragment, tests the fragment's depth itself against a stand-in of the depth, so it stores only
  what passes, as Metal stores only after the tests. Cost: a 32-bit float stand-in image of the
  depth per pass, which the depth is copied into before each run of such draws and back from after
  it (through a staging buffer at one sample, image to image or by a compute split and a
  fragment-depth draw at more), each run in a render pass instance of its own under depth `ALWAYS`
  without writes, its fragment shader testing and writing the stand-in in the pixel interlock; where
  the stencil test acts too, an 8-bit stand-in of the stencil per pass, copied in and back the same
  way (image to image at more than one sample) under the stencil test off.
- `metal2vulkan-degrade-overlapping-texture-copy-as-memmove` (precision).
  `OverlappingTextureCopyAsMemmove`: A blit copy between overlapping regions of one texture's
  texels, other than a region onto itself: Metal's copy reads as `memmove`'s for regions up to a few
  hundred texels across, but past that some texels read what another part of the copy already wrote,
  differently from run to run. Cost: the destination holds the source's texels as they were before
  the copy, memmove's answer, which is Metal's for small regions, where Metal's copy of a larger
  region races and differs from run to run; the source goes through a scratch buffer, two copies
  instead of one.
- `metal2vulkan-degrade-mipmap-tie-rounding` (precision). `MipmapTieRounding`: `generate_mipmaps`
  computed as Metal's filter whose levels take a rounding the chain does not fit at sums a hair from
  a rounding boundary (`MipmapFilterChain::near_ties`): a texture of a format with an 8-bit sRGB
  colour channel (`R8Unorm_sRGB`, `RG8Unorm_sRGB`, `RGBA8Unorm_sRGB`, `BGRA8Unorm_sRGB`), whose
  filtered levels take it; of an extended-range sRGB format (`BGR10_XR_sRGB`, `BGRA10_XR_sRGB`),
  whose filtered levels and levels read from kept values take it; or of a 16-bit float format
  (`R16Float`, `RG16Float`, `RGBA16Float`) with a level read from the level above's kept values (a
  2D chain's even level below a square of a power of two, a pass's second level). Cost: each level
  is Metal's but at a value whose sum lies a hair from a rounding boundary, which may be one unit
  off: an 8-bit sRGB colour code of a filtered level (1 of 5,331,270 measured values), an
  extended-range sRGB code, a 16-bit float read from kept values one unit in the last place (8 of
  538,088).
- `metal2vulkan-degrade-extended-range-blends-stored-codes` (precision).
  `ExtendedRangeBlendsStoredCodes`: A draw whose render pipeline state blends into, or reads (a
  framebuffer fetch, or the interlocked storage form an extended-range blend takes), a color
  attachment of an extended-range format (`BGR10_XR`, `BGRA10_XR`, `RGB10A8_2P_XR10` and their sRGB
  twins) that an earlier draw or execution of render commands of the same render pass may have
  written; an execution of render commands in a pass with such an attachment, whose commands may
  blend onto each other. Cost: a blend into, or a read of, an extended-range color an earlier draw
  of the pass wrote sees the codes the runtime stored at that draw (each channel rounded to its
  10-bit code, the two-plane format's alpha to its byte), where Metal's tile memory holds the color
  as half through the pass and rounds it once at the store: a blended or read value can differ from
  Metal's by a code, and the difference can grow over many draws that blend onto each other.
- `metal2vulkan-degrade-unmeasured-heap-texture-size` (precision). `UnmeasuredHeapTextureSize`: A
  heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`) whose size Metal's
  layout the runtime does not model: a format or a sample count whose layout in Metal's heaps is not
  measured. Cost: heap_texture_size_and_align answers the device's own image's size and alignment,
  which may be larger or smaller than Metal's, so a heap sized by Metal's answers may hold more or
  fewer such textures and they lie at other offsets.
- `metal2vulkan-degrade-plain-state-unique-identifier` (precision). `PlainStateUniqueIdentifier`: A
  program's call answered the `uniqueIdentifier` of a pipeline state made without
  `supportIndirectCommandBuffers` (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, a tile state's too). Cost: the identifier is one value
  of the runtime's count, unique and fixed for the state's life, where Metal's is an address that
  differs from run to run.
- `metal2vulkan-degrade-unmeasured-unique-identifier` (precision). `UnmeasuredUniqueIdentifier`: A
  program's call answered a `uniqueIdentifier` Metal answers alike in every run of one program
  started alike, which the runtime's is not measured to be: a pipeline state's made with
  `supportIndirectCommandBuffers` (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, an indirect command's state identifier other than 0), or
  an indirect command buffer's made after its device opened a command encoder
  (`IndirectCommandBuffer::unique_identifier`). Cost: the identifier is unique and fixed for its
  holder's life, and a released one is reused as Metal reuses it, but it is not the number Metal
  answers there, which follows how the process was started, what it made and Metal's own encoders'
  allocations.
- `metal2vulkan-degrade-indirect-value-past-record` (precision). `IndirectValuePastRecord`: A
  program set a buffer at a slot past 31, or a threadgroup memory length at an index past 30, of a
  command of an indirect command buffer whose descriptor counts more: Metal's validation stops such
  a descriptor, and plain Metal keeps the value for its private getters alone, as no function binds
  past index 30. Cost: the command's read-back (Metal's private getters) answers a fresh command's
  value for a buffer slot past 31 or a threadgroup length past 30, where Metal answers the value
  set; no command runs otherwise.
- `metal2vulkan-degrade-cross-class-texture-neutral` (precision). `CrossClassTextureNeutral`: A draw
  or dispatch that binds, at a texture slot, a texture whose format's numeric class (float, signed
  or unsigned integer) is not the class the function declares there and that its plan has no exact
  answer for (`PlanDiagnostic::CrossClassNeutral`): a depth, stencil, extended-range or YCbCr
  texture, a depth slot, an atomic or `ulong` access, a written storage slot over a swizzled
  texture, or an integer write to an `RGB9E5Float` texture. Cost: the function reads the slot's
  neutral texture (zero texels; zero size), where plain Metal reads the texture in its format's own
  class and hands back its words as the declared type.
- `metal2vulkan-degrade-indirect-nil-buffer-offset` (precision). `IndirectNilBufferOffset`: An
  execution of an indirect command buffer replayed on the host ran a command that holds a nil buffer
  set at an offset whose low 32 bits are not 0. Cost: the function sees no buffer, a null address,
  where Metal's sees the nil buffer's offset as its address; neither holds a buffer, so only the
  address's value differs.
- `metal2vulkan-degrade-compressed-texel-decode` (precision). `CompressedTexelDecode`: A texture of
  a BC1 to BC5 format (`BC1_RGBA` to `BC5_RGSnorm`, the sRGB forms too) made on a device whose
  driver is not measured to decode their texels as Metal does (BC1 to BC3
  `DriverFact::ColorBlockDecodeIsMetals`: MoltenVK 1.4.1 on Apple's GPUs, RADV from Mesa 26.2.4 on
  the 780M; BC4 and BC5 `DriverFact::ChannelBlockDecodeIsMetals`: MoltenVK 1.4.1 on Apple's GPUs).
  Cost: a read, sample or gather of its texels answers the device's decode of its blocks (an sRGB
  form's code through Metal's table), where Metal interpolates with 6-bit weights: NVIDIA's
  (615.71.09, the RTX 3090 and RTX 5080 Laptop GPU) answers a BC1 to BC3 channel up to 3/255 off
  Metal's (BC3's alpha 6/255), BC4 and BC5 up to 7.3/255 (signed 0.064); ANV's up to 1.3/255 (signed
  0.013), lavapipe's 2.3/255 (signed 0.021), RADV's (Mesa 26.2.4, the Radeon 780M) a BC4 or BC5
  signed texel with an endpoint of -128 up to 1.7/255; its blocks' bytes, copies and sizes are
  Metal's.
- `metal2vulkan-degrade-decoded-block-twin` (performance). `DecodedBlockTwin`: A texture of a BC1 to
  BC5 format made on a device whose driver is not measured to decode its texels as Metal does
  (`compressed-texel-decode`'s facts), whose image is its own for good: it keeps a decoded twin,
  Metal's texels, which its samples read, and announces no `compressed-texel-decode`. Cost: a second
  image of its decoded texels beside its blocks, 4 to 8 times their bytes (R8G8B8A8_UNORM for BC1 to
  BC3, R32_SFLOAT and R32G32_SFLOAT for BC4 and BC5), and a decode after each write into it: a pass
  on the GPU for a blit, a copy from a texture or a CPU copy through a submission; on the CPU, on
  the device's decode thread, where its CPU copies go on the host or the staging queue, and then
  once as it is made, the first command buffer that may read it waiting for that at its submission.
- `metal2vulkan-degrade-cpu-block-decode` (performance). `CpuBlockDecode`: A CPU copy into a texture
  with a decoded twin (`replace_region`) whose CPU copies go on the host or the staging queue: its
  region is decoded on the CPU, on the device's decode thread, and copied into the twin; the first
  command buffer that may read the texture waits at its submission for that, and the call itself
  where a command buffer that may read it is submitted and not yet complete (which then reads the
  copied texels, as Metal's does) or the texture is untracked. Cost: the written blocks decoded on
  the CPU, on the device's decode thread split over up to 8 threads, and their texels copied into
  the decoded twin on the host or the staging queue: the first command buffer that may read the
  texture waits at its submission for that, and the call waits for it where a command buffer that
  may read the texture is submitted and not yet complete, or the texture is untracked.
- `metal2vulkan-degrade-shared-present-queue` (performance). `SharedPresentQueue`: A command queue
  made for presenting (`Device::new_command_queue_for_presenting`) on a device that reserved no
  `VkQueue` for presenting: its queue family has no queue past the ones the round robin takes
  (MoltenVK's four families of one queue, RADV's and lavapipe's single graphics queue). Cost: it
  submits to a `VkQueue` other command queues share, so a present that blocks under its lock (FIFO
  on a window the compositor does not show, a wait for the queue's idle at a resize) holds their
  submissions until it returns, where Metal's present holds no other queue.
- `metal2vulkan-degrade-heap-texture-larger-than-metal` (precision). `HeapTextureLargerThanMetal`: A
  heap texture sized or made (`heap_texture_size_and_align`, `Heap::new_texture`) whose image on the
  device takes more of a heap than Metal's answer, which the runtime models: more bytes, or a
  coarser alignment (the image's own, at least `bufferImageGranularity`, its size in whole
  granularity pages). Cost: heap_texture_size_and_align answers more bytes or a coarser alignment
  than Metal's, so a heap sized by Metal's answers holds fewer such textures, and they lie at other
  offsets; each texture keeps its own bytes.
- `metal2vulkan-degrade-linear-attachment-copies` (performance). `LinearAttachmentCopies`: A pass
  renders a color attachment whose texture's image is linear (a texture over a buffer's memory, a
  no-copy buffer's host pages included) through an image its texture keeps, where the device renders
  such an image only under `VK_NV_linear_color_attachment`'s rule, that every color attachment of
  the render pass instance be linear, and the pass breaks it: another of its color attachments is
  not linear, or it resolves a multisampled attachment into the texture. Cost: its texels are copied
  into an image the pass renders, and what the pass wrote is merged or copied back after it.
- `metal2vulkan-degrade-linear-attachment-clear-copies` (performance).
  `LinearAttachmentClearCopies`: A pass clears a color attachment whose texture's image is linear
  and which the device renders under `VK_NV_linear_color_attachment`, where the rule allows it in
  place: it clears an image its texture keeps and copies it back after, as clearing a linear image
  of host memory in place measured slower than that copy. Cost: the pass clears an image of the
  device's and copies it back after, as clearing the texture's linear image in place costs more.
- `metal2vulkan-degrade-host-deallocator-delayed` (precision). `HostDeallocatorDelayed`: A no-copy
  buffer or heap released with a deallocator and no keep-alive (`KeepAlive`) while the device's
  import of its pages is still held: a command buffer submitted before a texture over them was
  released may still read that texture through its resource ID, and its image holds the import until
  the command buffer completes. Cost: the deallocator runs once every command buffer submitted
  before the texture's release completed, where Metal's runs at the release; the pages stay mapped
  until then.
- `metal2vulkan-degrade-misplaced-notice` (performance). `MisplacedNotice`: In a build without debug
  assertions, a notice announced where the runtime's rule announces none (`DegradeNotice::misplaced`
  names its kind and `NoticePlace`): inside a pipeline pool's piece, or, for a kind announced at
  every occurrence, outside the program's call and outside a host step. Cost: the notice reached the
  message sink from the runtime's notice thread after the call that met it, not in the program's
  order on the thread that met it; Metal's answers are unchanged.
- `metal2vulkan-degrade-depth-denormals` (precision). `DepthDenormals`: A write that may store a
  denormal in the 32-bit float depth of a depth image (a pass's clear to one, a blit's copy into a
  `Depth32Float` or `Depth32Float_Stencil8` texture's depth, a `replace_region` whose bytes hold
  one), on a device whose driver stores one a write on the GPU gives it as a zero (no
  `DriverFact::DepthImageKeepsDenormals`: RADV, ANV; a host copy keeps it) or whose depth test
  compares one otherwise than Metal's (no `DriverFact::DepthTestKeepsDenormals`: every driver
  measured but MoltenVK on an Apple M4). Cost: a denormal depth may read, sample and copy out as a
  zero, or the depth test may compare it as zero (a positive one) or below zero (a negative one),
  where Metal keeps it and compares a positive one above zero and a negative one not below it.
- `metal2vulkan-degrade-first-sparse-resource-wait` (performance). `FirstSparseResourceWait`: The
  device's first sparse texture made (`Heap::new_texture` in a sparse heap) on a device whose
  driver's first sparse resource waits for the work submitted before it on every queue
  (`DriverFact::FirstSparseResourceWaitsForQueues`: ANV through TR-TT, Mesa 26.2.4 on Intel's GPUs):
  `vkCreateImage` sets the driver's sparse translation tables up with a batch on each queue and
  waits for it. Cost: the call returns only once every queue finished the work submitted before it,
  once per device, where Metal's returns at once; what the texture holds and reads is Metal's.
- `metal2vulkan-degrade-driver-pipeline-cache-kept` (performance). `DriverPipelineCacheKept`: A
  device the runtime makes whose driver it cannot make with the driver's internal pipeline cache off
  (`VkDevicePipelineBinaryInternalCacheControlKHR`): its driver lists no `VK_KHR_pipeline_binary`,
  reports no `pipelineBinaryInternalCacheControl`, or the device enables no `VK_KHR_maintenance5`,
  which the extension requires. Cost: the driver keeps its own cache of the device's pipelines
  beside the shader cache's records, with whatever it writes to disk and holds in memory for it; the
  pipelines made are the same.
- `metal2vulkan-degrade-metal-level-not-macos` (precision). `MetalLevelNotMacOS`: A device made off
  macOS with no Metal level named: it answers as the newest measured release (`MetalLevel::NEWEST`),
  no release of the host's. Cost: where Metal's answers differ by release (which pixel formats a
  texture descriptor takes and Metal describes, how a heap texture's blocks lie), the device gives
  the newest measured release's, whatever release the program expects.
- `metal2vulkan-degrade-private-properties-unknown` (precision). `PrivatePropertiesUnknown`: Apple's
  private device properties the device has no answer for: no document describes them and nothing the
  device reports maps to them. Cost: the device gives no answer for them (`PrivateAnswer::value` is
  `None`); Metal's measured answers on Apple's GPUs ride beside, never as this device's.
- `metal2vulkan-degrade-unpacked-fill-color` (precision). `UnpackedFillColor`: A color fill
  (`BlitCommandEncoder::fill_texture_color`) whose color is packed in a format whose fills do not
  follow the color: compressed, 4:2:2, depth, stencil, two-plane or padded 10-bit. Cost: every unit
  of the region holds zero bytes, whatever the color, where Metal's constant is its own process's.
- `metal2vulkan-degrade-unsized-points` (precision). `UnsizedPoints`: A draw of points that
  rasterizes, of some vertices and instances, under a render pipeline state whose vertex function
  writes no `[[point_size]]` and whose input primitive topology class is not the point class: a
  direct draw, an indirect one, or an indirect command's in an execution. Cost: the points rasterize
  nothing, the vertex function run for each, where what Metal draws is undefined: most of its draws
  draw nothing, and some a few of the points or a box.
- `metal2vulkan-degrade-f16-fma-double-rounding` (precision). `PrecisionNote::F16FmaDoubleRounding`:
  Widened half fma rounds twice.
- `metal2vulkan-degrade-f16-int-conv-double-rounding` (precision).
  `PrecisionNote::F16IntConvDoubleRounding`: Widened half-integer conversions round twice.
- `metal2vulkan-degrade-grad-from-bias` (precision). `PrecisionNote::GradFromBias`: Bias sampling
  emulated with gradients may pick another level and another anisotropic footprint than Metal's:
  derivatives hoisted out of a per-fragment texture choice, which scale by `2^B` through every
  sampler, and, on a device that does not sample through Metal's own sampler, a bias whose sum with
  the sampler's own may lie beyond the device's `maxSamplerLodBias`, which Vulkan clamps, or that an
  anisotropic sampler applies to its footprint's minor axis alone.
- `metal2vulkan-degrade-unordered-framebuffer-fetch` (precision).
  `PrecisionNote::UnorderedFramebufferFetch`: Framebuffer fetch without rasterization-order
  attachment access.
- `metal2vulkan-degrade-device-transcendentals` (precision). `PrecisionNote::DeviceTranscendentals`:
  Transcendentals answer within Apple's stated bound, not Metal's bits: the device's own functions,
  or the translation's routines where a device's instruction may leave the bound (and, for the
  functions no routine covers yet, the device's precision).
- `metal2vulkan-degrade-reduction-order` (precision). `PrecisionNote::ReductionOrder`: Float
  reductions use the device's association order.
- `metal2vulkan-degrade-device-denormals` (precision). `PrecisionNote::DeviceDenormals`: `float`
  denormals follow the device: Metal reads an `f32` denormal operand of its float arithmetic,
  comparisons, and functions as the zero of its sign and flushes a denormal result to one, where a
  Vulkan device keeps or flushes them by its own float controls (this translation declares no
  `DenormFlushToZero`).
- `metal2vulkan-degrade-pixel-sampler-scaling` (precision). `PrecisionNote::PixelSamplerScaling`:
  Pixel-coordinate samplers scale coordinates in the shader, rounding differently.
- `metal2vulkan-degrade-device-arithmetic` (precision). `PrecisionNote::DeviceArithmetic`: Float
  division, remainder, square root, fused multiply-add, and dot products use Vulkan's precision
  bounds (e.g. 2.5 ULP division, a possibly unfused `Fma`) rather than correct rounding.
- `metal2vulkan-degrade-device-signed-zero-inf-nan` (precision).
  `PrecisionNote::DeviceSignedZeroInfNan`: Float operations LLVM requires to keep signed zeros,
  infinities, and NaNs run without `SignedZeroInfNanPreserve`, which the device does not support at
  their width.
- `metal2vulkan-degrade-device-ray-intersection` (precision).
  `PrecisionNote::DeviceRayIntersection`: A ray intersection traversed by the device's ray query.
- `metal2vulkan-degrade-fast-math-rewrites` (precision). `PrecisionNote::FastMathRewrites`: Float
  arithmetic whose AIR lets a compiler rewrite it with a neighbouring operation: an addition or
  subtraction and the multiply feeding it both carry LLVM's `contract`, two chained additions,
  subtractions, or multiplies both carry `reassoc`, or an `llvm.fmuladd` (fused or not, as the
  implementation chooses).
- `metal2vulkan-degrade-tessellation-winding` (precision). `PrecisionNote::TessellationWinding`: A
  tessellation pipeline's vertex order: Metal's clockwise output winding is declared
  `VertexOrderCcw` and its counter-clockwise one `VertexOrderCw`, as MoltenVK maps them under
  Vulkan's default upper-left domain origin.
- `metal2vulkan-degrade-tessellation-factor-edge-cases` (precision).
  `PrecisionNote::TessellationFactorEdgeCases`: A tessellation pipeline's factors at the edges of
  Metal's stated rules: a subnormal edge factor is positive and does not cull (its level is the
  range's lower end), where a flush to zero would cull the patch; and a NaN inside factor's level is
  the range's lower end.
- `metal2vulkan-degrade-tessellation-factor-scale` (precision).
  `PrecisionNote::TessellationFactorScale`: A tessellation pipeline with
  `tessellationFactorScaleEnabled` multiplies each factor by the encoder's scale rounded to `half`,
  exactly in `float`, after the cull.
- `metal2vulkan-degrade-tessellation-factor-instance` (precision).
  `PrecisionNote::TessellationFactorInstance`: A tessellation pipeline whose factor step function is
  per instance reads instance `i`'s factors at `i · instanceStride`, `i` counting the draw's base
  instance as `[[instance_id]]` does.
- `metal2vulkan-degrade-fp8-value` (precision). `PrecisionNote::Fp8Value`: An 8-bit float converts
  to an integer from the format's documented value with Metal's saturating float-to-integer rule,
  where Metal's device here answers otherwise: Metal on Apple M3 compiles these reads but extends
  the carrier byte as an integer, the same for every format, so the result is unverified on Metal.
- `metal2vulkan-degrade-multisample-position` (precision). `PrecisionNote::MultisamplePosition`: A
  fragment function of a render pipeline that rasterizes more than one sample reads, at pixel rate,
  a value Metal reads at the pixel center in every invocation and Vulkan evaluates at an
  implementation-defined position within the fragment: `[[position]]`'s z or w (the interpolated
  depth and `1/w`, whose position Vulkan does not state), a center-sampled varying on a device
  without its pixel form (interpolation functions whose `InterpolateAtOffset` reads from the pixel
  center: "an implementation-defined position within the fragment area being shaded"), a centroid
  varying (anywhere in the fragment's covered area, where Metal, measured on Apple M3 Max, moves it
  to a covered sample's location only under partial coverage), `[[point_coord]]` ("the fragment
  center, or the location of a sample"), or `[[barycentric_coord]]`.
- `metal2vulkan-degrade-rasterization-rate-map` (precision). `PrecisionNote::RasterizationRateMap`:
  A render pass drawn by regions over a rasterization-rate map that is not the identity: its
  regions' lines are this crate's tiles' exact lines, where Apple's are its fixed-point ones, so a
  fragment's evaluated point lies within a bounded distance of Metal's.
- `metal2vulkan-degrade-bicubic-filter-arithmetic` (precision).
  `PrecisionNote::BicubicFilterArithmetic`: Metal's bicubic filter, emulated by fetches with Metal's
  weights, the 8- and 16-bit normalized formats' exact sum rounded half up onto the format's grid,
  and the 32-bit float and half-float formats' sums through Metal's accumulators, differs from an
  Apple M3 Max in what is not yet pinned: a 2D normalized sum within about `2^-21` of the format's
  range (`1` unsigned, `2` signed) of a rounding tie: Metal may round it to the other side, one grid
  step away (`1/65535` or `1/32767` for 16-bit, `1/4080` or `1/2032` for 8-bit).
- `metal2vulkan-degrade-sparse-residency` (precision). `PrecisionNote::SparseResidency`: A stage
  samples or gathers a texture whose residency it observes (`sparse_color::resident()`) through the
  slot's sparse form, which the module has where the profile enables `shaderResourceResidency`.
- `metal2vulkan-degrade-device-nan-encoding` (precision). `PrecisionNote::DeviceNanEncoding`: A
  `float` NaN result's bit pattern follows the device: Metal answers the canonical `0x7fc00000` for
  the NaN results of its float arithmetic and functions, and Vulkan states no encoding, so a device
  may answer another sign or payload (llvmpipe: `0xffc00000`).
- `metal2vulkan-degrade-restricted-depth-range` (precision). `PrecisionNote::RestrictedDepthRange`:
  The device compares a depth texture's texels on a device without `VK_EXT_depth_range_unrestricted`
  enabled.
- `metal2vulkan-degrade-cube-gradient-footprint` (precision).
  `PrecisionNote::CubeGradientFootprint`: An explicit-gradient sample of a cube texture takes
  Metal's footprint in a form a Vulkan device reads alike, but for what is not reproduced.
- `metal2vulkan-degrade-half-texel-conversion` (precision). `PrecisionNote::HalfTexelConversion`:
  The module reads, gathers, or samples a texture with a `half` result.
- `metal2vulkan-degrade-half-sample-filter` (precision). `PrecisionNote::HalfSampleFilter`: The
  module samples a texture with a `half` result through a filter (linear minification or
  magnification, or a runtime sampler) on a device that does not sample through Metal's own sampler.
- `metal2vulkan-degrade-srgb-texel-decode` (precision). `PrecisionNote::SrgbTexelDecode`: The module
  reads, samples, or gathers a texture with a `half` or `float` result through the device's sRGB
  decode where the texture may be bound in an 8-bit sRGB format.
- `metal2vulkan-degrade-float-sample-filter` (precision). `PrecisionNote::FloatSampleFilter`: The
  module samples a texture with a `float` result through a filter on a device that does not sample
  through Metal's own sampler.
- `metal2vulkan-degrade-device-gather-footprint` (precision).
  `PrecisionNote::DeviceGatherFootprint`: The module leaves a gather to the device's footprint on a
  device that does not sample through Metal's own sampler.
- `metal2vulkan-degrade-cube-face-reciprocal` (precision). `PrecisionNote::CubeFaceReciprocal`: The
  module hands a nearest cube sample the centre of the texel Metal reads, on a device that does not
  sample through Metal's own sampler.
- `metal2vulkan-degrade-function-storage-assumed` (precision).
  `PrecisionNote::FunctionStorageAssumed`: The stage keeps private memory (its `alloca`s, a call
  stack among them, or constants held as `Private` variables) on a device whose Function storage was
  not measured, so it assumes the device holds the floor, 16 KiB a thread
  (`device::function_storage`).
- `metal2vulkan-degrade-quads-counted-before-discard` (precision).
  `PrecisionNote::QuadsCountedBeforeDiscard`: A fragment function runs a non-quad simdgroup
  operation (a reduction, scan, ballot, broadcast, shuffle, vote, or simdgroup matrix) before a
  discard on a path to it, in a shape where Metal's placement of it is not known.
- `metal2vulkan-degrade-device-depth-comparison` (precision).
  `PrecisionNote::DeviceDepthComparison`: The module leaves a depth comparison (a sample or gather
  with comparison) to the device, on a device that does not sample through Metal's own sampler (one
  that does, `VkProperty::MetalSampler`, compares as Metal does: the module requires it instead).
- `metal2vulkan-degrade-device-texel-denormals` (precision). `PrecisionNote::DeviceTexelDenormals`:
  A `float` sample or gather, a comparison's included, whose texels the device reads, on a device
  that does not sample through Metal's own sampler.
- `metal2vulkan-degrade-beyond-measured-deviation` (precision).
  `PrecisionNote::BeyondMeasuredDeviation`: The module is compiled for a device whose driver release
  is newer than the last one measured with a deviation the profile takes
  (`ProfileFields::deviations_beyond`, which names each with that release): where the specifications
  leave the answer to the driver, the module relies on its answer being Metal's (a simdgroup's
  lockstep, Metal's own `float` functions, its canonical NaN, its sampler, its cube-gradient route),
  as measured on the earlier release.
- `metal2vulkan-degrade-view-base-texel-offset` (precision). `PrecisionNote::ViewBaseTexelOffset`:
  The module places a texel offset on an axis that may repeat or mirror as Metal does in an access
  of an argument buffer's texture: on a device that does not sample through Metal's own sampler (one
  that does, `VkProperty::MetalSampler`, is Metal's texture unit itself), in a snapped gather or
  nearest sample, fetched texels, or an emulated linear filter, and on every device in an emulated
  raw sample (`heap_raw`).
- `metal2vulkan-degrade-device-color-conversion` (precision).
  `PrecisionNote::DeviceColorConversion`: A fragment function writes a float to a color channel
  whose conversion the device makes, not Metal's rule (`ChannelWrite::Device`): an attachment that
  blends with factors other than one and zero, the alpha of attachment 0 under alpha-to-coverage, a
  format whose conversion Metal was not measured with (the packed 16-bit unorm formats, `A8Unorm`,
  the packed floats).
- `metal2vulkan-degrade-device-interpolation` (precision). `PrecisionNote::DeviceInterpolation`: A
  fragment function reads a varying the device interpolates (`center_perspective`, `centroid`,
  `sample` or `no_perspective`, not `flat`) on a device whose interpolation is measured to differ
  from Metal's (`DriverWorkaround::InterpolationNotMetals`): NVIDIA's driver rounds an interpolated
  value that is exactly representable one `f32` ulp toward zero where Metal gives it exactly.
- `metal2vulkan-degrade-quads-split-by-clipping` (precision). `PrecisionNote::QuadsSplitByClipping`:
  A fragment function runs a quad or simdgroup operation on a device that shades a triangle it clips
  as several triangles (`DriverWorkaround::ClippedTrianglesSplitPixelFragments`): a quad a seam
  between the pieces crosses runs once per piece, each run with the other piece's pixels as helper
  invocations, so the operation counts or combines fewer active lanes than Metal's, whose quad holds
  all four.
- `metal2vulkan-degrade-vertex-simdgroup-grouping` (precision).
  `PrecisionNote::VertexSimdgroupGrouping`: A vertex function runs a simdgroup or quad operation on
  a device whose vertex stage takes no required subgroup size and whose driver is measured to group
  vertices otherwise than Apple's GPUs (`DriverWorkaround::VertexGroupingNotApples`: ANV, 8 vertices
  a subgroup, interleaved): the operation runs over the device's group of vertices, so a sum, a
  ballot or a shuffle combines other vertices than on an Apple GPU, whose simdgroup holds 32
  consecutive vertices.
- `metal2vulkan-degrade-fragment-simdgroup-grouping` (precision).
  `PrecisionNote::FragmentSimdgroupGrouping`: A fragment function runs a simdgroup operation (a
  reduction, scan, vote, ballot, shuffle, broadcast or AGX multiply-add) or reads its
  `[[thread_index_in_simdgroup]]` on a device whose fragment subgroups are not Metal's simdgroups
  (the profile does not take `DriverDeviation::MetalFragmentSimdgroups`): the operation runs over
  the pixels the device's rasterizer packs into a subgroup, so a sum, a ballot or a shuffle combines
  other pixels than on an Apple GPU, whose packing is measured, and the lane is the pixel's place in
  the device's subgroup.
- `metal2vulkan-degrade-blended-write-rounding` (precision). `PrecisionNote::BlendedWriteRounding`:
  A fragment function writes a float to a half-float color channel of a render pipeline under
  `TextureWriteRoundingMode::NearestEven` that blends with factors other than one and zero
  (`ChannelWrite::BlendedNearestEven`): Metal rounds the blend's result to nearest even, and the
  device's blend converts it by its own rounding (MoltenVK's toward zero), so a blended value off
  the half grid may store the neighbouring half; under a constant factor Metal's result is neither
  rule's on 2,376 of 580,508 `RGBA16Float` channels near the half denormals.
- `metal2vulkan-degrade-rotated-blend-rounding` (precision). `PrecisionNote::RotatedBlendRounding`:
  A fragment function writes an attachment held with its channels rotated (`ARGB8Unorm`) whose blend
  is none of the measured forms (`RotatedBlendForm::Unmeasured`), or the premultiplied form through
  a first output that is not 32-bit float: Apple's blend computes a factor pair apart in its color
  and alpha arithmetic, and the rotation moves Metal's blue into the image's alpha and Metal's alpha
  into its red, so a channel may store one code apart from Metal's.
- `metal2vulkan-degrade-linked-sampler-numbering` (precision).
  `PrecisionNote::LinkedSamplerNumbering`: The sampler resource IDs a pipeline state takes for the
  constexpr samplers of the visible functions its stage links may be other than Metal's: two or more
  linked functions bring constexpr samplers, which Metal numbers in an order the pipeline's inputs
  do not decide; or a function a table can hold that linking did not import could not be planned, so
  its samplers are not known; or a dynamic library's symbol, a binary function or a function-pointer
  group's function brings them, whose numbering is not measured.
- `metal2vulkan-degrade-computed-sample-depth` (precision). `PrecisionNote::ComputedSampleDepth`: A
  fragment function runs its draw's depth test itself at more than one sample on a device whose
  `FragCoord.z` is the pixel centre's there (`VkProperty::CenterFragCoordDepth`), and computes each
  sample's depth from its vertex function's window depth read at the sample, plus `FragCoord.z` less
  that depth read at the centre, which carries the depth bias.
- `metal2vulkan-degrade-argument-pixel-sampler-scaling` (precision).
  `PrecisionNote::ArgumentPixelSamplerScaling`: An access through an argument buffer's sampler (not
  a 2D gather, which snaps behind the sampler's word) whose word the specialized module cannot read
  may hand the device a pixel coordinate scaled by the texture's size: where the sampler states
  pixel coordinates and a linear filter or a comparison, on a device without
  `VkProperty::MetalSampler` (a filtered sample of a class the emulation does not reach), or a
  comparison on one with it.
- `metal2vulkan-degrade-half-output-precision` (precision). `PrecisionNote::HalfOutputPrecision`: A
  fragment function returns a `half` color output (a scalar or vector of AIR's `half`) to an
  attachment the pipeline has, and either its module is compiled without fast math
  (`air.compile.fast_math_disable`) and that attachment blends, factors one and zero included, with
  some channel written (`StageIo::half_blend_colors`, which the pipeline's stage reads), or its
  module names no math mode (under fast math the function records
  `PrecisionNote::HalfOutputFastMathBlend` instead).
- `metal2vulkan-degrade-extended-range-blend-source` (precision).
  `PrecisionNote::ExtendedRangeBlendSource`: A fragment function's `half` color output (a scalar or
  vector of AIR's `half`) is blended into an extended-range attachment (`BGR10_XR`, `BGRA10_XR`,
  `RGB10A8_2P_XR10`, or an `_sRGB` twin) where Metal may round the blend's source otherwise than the
  function's computed blend: under fast math a lane of the output is not a load; or the module names
  no math mode.
- `metal2vulkan-degrade-discard-drops-simdgroup-stores` (precision).
  `PrecisionNote::DiscardDropsSimdgroupStores`: A fragment function meets the store-dropping defect
  of `DriverWorkaround::DiscardDropsSimdgroupStores` (MoltenVK on an Apple GPU: Apple's compiler
  drops the texture writes of every live thread of a simdgroup holding a discarded thread, where the
  function discards, then writes storage, samples, and calls SPIRV-Cross's `optnone` arithmetic
  helpers), and a discard site that would discard at once cannot wait until after the writes: a
  store, atomic, image write or non-quad subgroup operation follows it, which a predicated discard
  does not yet guard (`lower::fragment::discard`).
- `metal2vulkan-degrade-helper-branch-drops-simdgroup-stores` (precision).
  `PrecisionNote::HelperBranchDropsSimdgroupStores`: A fragment function meets the store-dropping
  defect of `DriverWorkaround::DiscardDropsSimdgroupStores` without a discard: SPIRV-Cross writes an
  operation of the module as an `optnone` call (a sum, difference or product decorated
  `NoContraction`, among others) where only helper invocations (the quad helpers past a primitive's
  edges, or demoted invocations) may run it before a storage write, and keeping them out would not
  be exact (`spirv::helper_branches`): a derivative or subgroup operation reads what they compute
  there or after, an arm leaves its selection otherwise than at its merge, or the call is in a loop
  whose helper invocations may iterate alone.
- `metal2vulkan-degrade-extended-range-float-blend-source` (precision).
  `PrecisionNote::ExtendedRangeFloatBlendSource`: `PrecisionNote::ExtendedRangeBlendSource` for a
  `float` color output (a scalar or vector of AIR's `float`): blended into an extended-range
  attachment where Metal may round the blend's source otherwise than the function's computed blend,
  under fast math a lane of the output that is not a load, or in a module that names no math mode.
- `metal2vulkan-degrade-anisotropic-filter` (precision). `PrecisionNote::AnisotropicFilter`: The
  module samples a texture through an anisotropic sampler with Metal's anisotropic filter emulated
  by the device's samples at explicit levels, on a device that does not sample through Metal's own
  sampler (`lower::sampler::anisotropic`): a sample without comparison of a 2D texture or 2D array
  at the fragment's implicit level or with explicit gradients, through a constexpr sampler of
  `maxAnisotropy` above 1, or through a runtime sampler slot whose word states anisotropy (under
  that word alone).
- `metal2vulkan-degrade-device-anisotropic-filter` (precision).
  `PrecisionNote::DeviceAnisotropicFilter`: The module samples a texture through a sampler whose
  anisotropy is known where Metal's anisotropic filter is not emulated
  (`PrecisionNote::AnisotropicFilter`'s reach), on a device that does not sample through Metal's own
  sampler: a cube, a 3D texture, a comparison, a sample whose residency the function observes, or a
  texture of another class than its declared one, at an implicit level or with explicit gradients,
  through a constexpr sampler of `maxAnisotropy` above 1 or a runtime sampler slot whose word states
  anisotropy (under that word alone).
- `metal2vulkan-degrade-device-point-size` (precision). `PrecisionNote::DevicePointSize`: A vertex
  function writes `[[point_size]]`, or a mesh function `set_point_size`, on a device whose
  `pointSizeRange` ends below Metal's largest point, 511 pixels (`Limits::max_point_size`: 255 on
  ANV, 256 on lavapipe).
- `metal2vulkan-degrade-clipped-coverage-split` (precision). `PrecisionNote::ClippedCoverageSplit`:
  A fragment function compiled for a multisampled pipeline reads `[[sample_mask]]` on a device whose
  rasterizer is not Apple's (`DriverWorkaround::ClippedCoverageNotMetals`): Apple's GPU draws a
  triangle that extends past the viewport as pieces whose shared edges cross pixels, and such a
  pixel's fragments can run once per piece, each reading that piece's coverage, where the device
  hands the pixel's fragment its coverage by the whole primitive (lavapipe by its own pieces).
- `metal2vulkan-degrade-device-unorm8-linear-filter` (precision).
  `PrecisionNote::DeviceUnorm8LinearFilter`: The module hands an 8-bit unsigned normalized texture's
  linear sample to the device's filter, at the coordinate of Metal's footprint step
  (`lower::sampler::native`), on a device measured to filter that class within one code of Metal
  (`DriverDeviation::Unorm8LinearFilter`, which the module requires as
  `VkProperty::Unorm8LinearFilter`): a sample with a `float` or `half` result of a 2D texture or 2D
  array slot bound in `R8Unorm`, `RG8Unorm`, `RGBA8Unorm` or `BGRA8Unorm`, without comparison,
  offset or pixel coordinates, through a constexpr sampler Metal's linear rule answers or a runtime
  sampler slot whose word states one (under that class and word alone).
- `metal2vulkan-degrade-half-output-fast-math-blend` (precision).
  `PrecisionNote::HalfOutputFastMathBlend`: A fragment function of a module compiled with fast math
  (`air.compile.fast_math_enable`) returns a `half` color output to an attachment the pipeline has,
  where `PrecisionNote::HalfOutputPrecision`'s conditions hold.
- `metal2vulkan-degrade-device-sampling` (precision). `PrecisionNote::DeviceSampling`: Native device
  sampling: the module hands a sample, gather or comparison to the device's own sampler where a
  device whose sampler is not Metal's would emulate Metal's (`SamplingPolicy::NativeDevice`, the
  default; the module requires `VkProperty::DeviceSampling`).
- `metal2vulkan-degrade-native-transcendentals` (precision). `PrecisionNote::NativeTranscendentals`:
  Native device maths: the module computes a `float` or `half` math-library function by the device's
  own `GLSL.std.450` instruction (or Metal's composition of the device's `Exp2` and `Log2`), not by
  the translation's routine of Metal's answer, on a device whose functions are not Metal's own,
  under `FloatMathsPolicy::Native`; and `fast::sqrt` the device's `Sqrt`, under
  `RoundingOrderPolicy::Native`.
- `metal2vulkan-degrade-native-division` (precision). `PrecisionNote::NativeDivision`: Native device
  maths: a precise `float` or `half` division, or Metal's fast division or reciprocal, is the
  device's `OpFDiv`, not the translation's correctly rounded routine or its `RN(1 / y)`, on a device
  whose division is not Metal's own, under `FloatMathsPolicy::Native` (the default).
- `metal2vulkan-degrade-native-fma` (precision). `PrecisionNote::NativeFma`: Native device maths: a
  `float` fused multiply-add on a device without `OpFmaKHR`, or a `half` one where the device's
  16-bit `Fma` is not measured to be Metal's, is the device's `GLSL.std.450 Fma` (on the `half`
  operands held in `float`, its result rounded to `half`), not the translation's correctly rounded
  routine, under `FloatMathsPolicy::Native` (the default).
- `metal2vulkan-degrade-device-texel-conversion` (precision).
  `PrecisionNote::DeviceTexelConversion`: Native device sampling (a texel conversion): the module
  hands a read, sample or gather with a `float` result the device's own conversion of the texel's
  codes, where Metal's is emulated exactly: a read's under `TexelConversionPolicy::Exact`, a
  sample's or gather's under it with `SamplingPolicy::Exact`.
- `metal2vulkan-degrade-native-half-conversion` (precision). `PrecisionNote::NativeHalfConversion`:
  Native half rounding (a conversion): the module rounds a `half` result it computes in `float` (a
  `half` operation's, a function's, a conversion from `float`'s, a texel's) by the device's
  conversion of the `float` to 16 bits and its widening, not by the translation's integer rounding,
  on a device with `shaderFloat16` whose conversion is not measured to be Metal's `half(float)`
  (every driver but MoltenVK on an Apple GPU), under `HalfRoundingPolicy::Native`.
- `metal2vulkan-degrade-native-half-quantize` (precision). `PrecisionNote::NativeHalfQuantize`:
  Native half rounding (a quantize): the module rounds a `half` result it computes in `float` by the
  device's `OpQuantizeToF16`, a result the quantize flushes to zero rounded by the translation to
  the half denormal grid, not by its integer rounding, on a device without `shaderFloat16` whose
  quantize is not measured to round as Metal's conversion does, under `HalfRoundingPolicy::Native`.
- `metal2vulkan-degrade-native-nan-encoding` (precision). `PrecisionNote::NativeNanEncoding`: Native
  device maths: a `float` or `half` operation's NaN result is the device's, its sign and payload
  unselected, on a device whose arithmetic is not measured to answer Metal's canonical NaN, under
  `NanMinMaxPolicy::Native`.
- `metal2vulkan-degrade-native-min-max` (precision). `PrecisionNote::NativeMinMax`: Native device
  maths: Metal's `fmin`, `fmax`, `min3`, `max3`, `clamp` or `saturate` (its `clamp(x, +0, 1)`) on
  `float` or `half` is the device's `GLSL.std.450` `NMin`, `NMax` or `NClamp`, not the translation's
  selects of Metal's operand, on a device not measured to answer Metal's, under
  `NanMinMaxPolicy::Native`.
- `metal2vulkan-degrade-native-dot` (precision). `PrecisionNote::NativeDot`: Native device maths:
  Metal's `dot` on `float`, or on `half` where its fused multiply-add is not the measured 16-bit
  one, is the device's `OpDot` (a `half` one summed in `float` and rounded once to `half`), not
  Metal's fused chain of its products, under `RoundingOrderPolicy::Native`.
- `metal2vulkan-degrade-native-mix` (precision). `PrecisionNote::NativeMix`: Native device maths:
  Metal's `mix(x, y, t)` on `float`, or on `half` where its fused multiply-add is not the measured
  16-bit one, is the device's `FMix` (a `half` one computed in `float` and rounded once to `half`),
  not Metal's `fma(t, y, fma(-t, x, x))`, under `RoundingOrderPolicy::Native`.

- `metal2vulkan-degrade-depth-twin-comparison` (retired). Replaced by
  metal2vulkan-degrade-device-depth-comparison and, on a device without
  VK_EXT_depth_range_unrestricted, metal2vulkan-degrade-restricted-depth-range, which the pipeline
  whose stage makes the device's depth comparison announces.
- `metal2vulkan-degrade-device-half-rounding` (retired). No replacement: every half rounding is
  Metal's conversion to half, on every device.
- `metal2vulkan-degrade-rate-map-parameter-data` (retired). No replacement: a rasterization-rate
  map's parameter data is Metal's bytes, which a translated decoder reads with Apple's arithmetic.
- `metal2vulkan-degrade-ycbcr-linear-filter` (retired). Replaced by
  metal2vulkan-degrade-ycbcr-filter-not-exact, which the pipeline whose stage filters a two-plane
  YCbCr texture otherwise than Metal's exact filter of the format announces.
- `metal2vulkan-degrade-reissued-resource-id` (retired). No replacement: plain allocation hands out
  an ID a live texture holds where Metal does.
- `metal2vulkan-degrade-padded-bits-kept` (retired). Replaced by
  metal2vulkan-degrade-unreached-tiles-rewritten, which a pass whose load pass rewrites a padded
  attachment's every tile announces where that is not Metal's, and
  metal2vulkan-degrade-extended-range-bits-kept, which names an attachment the load pass does not
  take.
- `metal2vulkan-degrade-input-attachment-srgb-decode` (retired). No replacement: an 8-bit
  four-channel sRGB color read through an input attachment is Metal's decode of its code.
- `metal2vulkan-degrade-beyond-measured-driver-release` (retired). No replacement: a device whose
  driver, GPU and release were not measured takes the exact path in place of an answer measured
  elsewhere, which degrades nothing; it says so in the informational message
  metal2vulkan-unmeasured-driver.
- `metal2vulkan-degrade-unmeasured-driver` (retired). No replacement: the paths an unmeasured
  identity takes are exact; the device says so in the informational message
  metal2vulkan-unmeasured-driver.
- `metal2vulkan-degrade-deferred-half-rounding` (retired). No replacement: the opt-in deferred half
  rounding is removed; every half result rounds where it is computed, under native half rounding
  (metal2vulkan-degrade-native-half-conversion or metal2vulkan-degrade-native-half-quantize) or
  Metal's.

## 10. Accepted differences

Where the runtime answers otherwise than Metal on purpose, and nothing is slower or approximate
for it (so no degrade notice says so), the difference is listed here. Most are answers Metal leaves
undefined or ends the process for, and answers the runtime adds beyond Metal's API. The rustdoc of
each item named states the rule in full.

**Draws and index buffers**

- **An indexed draw past its index buffer draws.** Metal's API validation stops it; the runtime
  draws as Metal does without validation: a partial last index reads up to the length rounded up
  to 4 bytes, and every later position draws index 0 plus the base vertex.
- **An indexed draw from an index offset at or past the length draws index 0** plus the base vertex
  at each position, where Metal reads whatever memory follows (`zeroed-index-fetch`).
- **A draw's `VkPipeline` is made while the draw goes on.** A draw whose pipeline is not made yet
  returns once encoded; the pipeline is made on the device's pool of at most four threads, and the
  command buffer's work waits for it in queue order.
- **A dispatch's `VkPipeline` is made while the dispatch goes on**, the same way: one pipeline per
  threadgroup size and edge region, made on the pool.

**Binary archives and dynamic libraries**

- **A binary archive serializes after its file is deleted.** `Device::new_binary_archive` reads the
  file whole, so `BinaryArchive::serialize_to_url` writes what was loaded; Metal's fails there.
- **An archive's file is a path.** `BinaryArchiveDescriptor::url` and `serialize_to_url` take a file
  path, not a URL.
- **A dynamic library's file is its metallib.** `DynamicLibrary::serialize_to_url` writes the bytes
  the library was made from, without a slice of GPU code; the runtime loads either form.

**Resource IDs and identifiers**

- **A gone holder's texture ID reads the neutral texture.** A function reading a texture ID from an
  argument buffer reaches the texture holding the ID as the command buffer runs; once the holder
  is gone, the ID reads the neutral texture, where Metal's read is undefined.
- **Two live textures with one ID: a function reads the holder**, the texture placed there last.
- **An indirect command buffer's ID is Metal's small integer**, from 1, the lowest free, in the
  indirect command buffers' own space; it stays held while committed work that runs or makes the
  buffer resident has not completed.
- **Resource IDs are numbered as Metal numbers them**: textures', sampler states', pipeline states'
  and indirect command buffers' IDs follow Metal's allocator, around reservations
  (`Device::reserve_resource_ids`) too. IDs that end Metal's process are refused.
- **The device counts the sampler IDs it holds** (`Device::sampler_ids_live`,
  `Device::sampler_id_pool_size`), runtime extensions for a consumer that models Metal's sampler
  heap.
- **A texture can take no resource ID** (`IdPlacement::Outside`), for a consumer's helper textures.
- **An indirect command buffer placed at an ID** (`IndirectCommandBufferDescriptor::placement`,
  Apple's private `setResourceIndex:`) takes it as Metal does; an index of 16384 or more is refused.
- **A pipeline state placed before any pipeline ID is made** (`PipelineOptions::placement`) is made,
  where Metal ended the process; an index of 16384 or more is refused.
- **An indirect command names its pipeline state without holding it.** As in Metal, releasing a
  state gives its ID back while a command names it; a CPU-encoded command still runs what the state
  ran, and a GPU-written record runs the ID's holder, or nothing (`released-command-state`).
- **Unique identifiers** (`ComputePipelineState::unique_identifier`,
  `RenderPipelineState::unique_identifier`, `IndirectCommandBuffer::unique_identifier`) are
  nonzero and unique among live objects, laid out by Metal's rule where it was measured
  (`plain-state-unique-identifier`, `unmeasured-unique-identifier` where it differs).

**Indirect command buffers**

- **A command's read-back answers Metal's private getters**: `command_type`, the buffer addresses
  and attribute strides, as Metal's `get*` getters answer them.
- **Whether the GPU may have written the records** (`IndirectCommandBuffer::gpu_written`, no Metal
  counterpart) turns true at the first call that lets GPU work write them; from then on an
  execution runs the records as they stand once the work before it completed.
- **What an execution runs, where its records are read**
  (`CommandBuffer::add_executed_commands_handler`, no Metal counterpart): a handler is called each
  time the runtime reads an execution's records, at `commit`, in a host step, or not at all for an
  execution the GPU alone runs.
- **Command predication** (`encode_start_if`, `encode_start_else`, `encode_end_if` and the
  while and do-while loops) runs Metal's blocks and loops. A loop is laid down as 64 copies (16
  inside another loop); one that runs past them fails its command buffer
  (`PredicatedLoopPastCopies`), where Metal runs on. Streams Metal answers by no measured rule are
  refused (`RuntimeGap::Predication*`).
- **A GPU-written indirect command naming a placed ID runs that ID's live holder**, the state made
  last; once it is released, nothing.
- **An indirect render command keeps a draw's values to 32 bits**, as Metal's records keep most of
  them; a value that cannot fit is refused.
- **An indexed patch command reads its control-point indices as Metal's did**, with its pipeline's
  control points and index type; an index past its buffer reads 0.
- **An indirect render command's index and factor buffers must be resident**: an execution resolves
  them against the resources the encoder made resident, as its kernel buffers.
- **A kernel's patch command scales its factors** by the scale its `draw_patches` passes, as Metal
  does.
- **A render execution records its pass at commit**, from the records as they stand then, as
  Metal's does.
- **A command's refusal is one typed error wherever it is recorded**: the CPU's records are checked
  as `commit` records them, and `commit` returns what a command refuses.
- **A render execution's range read from a buffer is read before its pass**
  (`execute_commands_in_buffer_indirect`).
- **An indirect command's buffer outside the resident resources misses**: its Metal address
  resolves against the buffers and heaps the encoder made resident; outside them the slot reads
  zero, drops writes, and the execution report counts a resident-set miss. Metal leaves the access
  undefined.
- **Indirect commands the GPU may write run in the large-buffer form**: buffer slots at device
  addresses with in-shader bounds checks (`VariantKey::large_buffer_form`), so no buffer a command
  binds needs another compile. Inside a slot's window every access is the CPU form's; outside it,
  two contained answers differ, neither defined by Metal.
- **A blit copy between indirect command buffers of different descriptors copies the commands**;
  a record the destination's descriptor does not allow runs nothing.
- **Render states an indirect command buffer leaves to its commands are the encoder's** on a device
  that does not claim Apple10.
- **A function that sets a render state on a render command is not made** (a kernel's
  `render_command::set_cull_mode` and its kin): the pipeline state is refused, as Metal refuses it
  without Apple10.
- **An indirect command buffer's `allocatedSize` is Metal's** (its layout in 16-KiB pages), not the
  runtime's records' footprint.

**Blits**

- **Multisampled blit copies between depth or stencil and colour formats** copy each sample's
  bytes as plain Metal does (its validation stops them), through `VK_KHR_maintenance8` or
  sample-rate draws. The two pairs the draws cannot copy exactly are refused: a 16-bit depth, and a
  copy into a 32-bit float depth on a driver not measured to store fragment depths unchanged.
- **A blit copy of 24-bit depth between textures of different formats is refused**: Apple's
  reference GPUs make no `Depth24Unorm_Stencil8` texture, so there is no Metal answer to match.

**Heaps and memory**

- **A heap's size and alignment answers are the device's own**
  (`Device::heap_texture_size_and_align`): the requirement of the image the runtime makes.
- **Metal's own heap answer, for bookkeeping** (`Device::metal_heap_texture_size_and_align`, no Metal
  counterpart): what plain Metal answers, for a consumer that lays out a guest's heap.
- **Where Metal's answer is not its image's, every device answers Metal's**, for the few formats
  the runtime holds in an image of another layout.
- **A multisample texture answers at least Metal's size and alignment.**
- **Metal's answer by usage**: Metal's size depends on whether it compresses the texture, which
  follows its usage bits and shape; the bookkeeping answer follows the measured rule.
- **Apple's private texture usage bits are kept** in `TextureUsage` (`NO_CONCURRENT_ACCESS`,
  `VIDEO`, `BLOCK_WRITES_ONLY`).
- **A heap larger than the device's `maxBufferSize` is refused**: a heap's buffers live in one
  `VkBuffer` over the heap's allocation.
- **A `Shared` heap's textures are linear images** where the device's optimal images take no
  host-visible memory.
- **A sparse texture on a device without shader residency holds all its texels' memory.**
- **A heap's buffer and texture over the same bytes alias them, in the device's layout**, not
  Apple's tiled or compressed one.

**Kernels**

- **An imageblock's length rounds its cell as an M4 does**
  (`ComputePipelineState::imageblock_memory_length_for_dimensions`), never less than what a
  dispatch counts against the device's threadgroup memory.
- **A recursive call past the call stack yields zero**: the stack holds what Metal's byte model
  holds for the pipeline's `max_call_stack_depth`; a call that finds it full returns zero, where
  Metal's answer is corrupt.

**Rasterization-rate maps**

- **A map of no layers answers its coordinate maps**, where Metal ends the process.
- **A screen past 16384 pixels a side is refused**, where Apple's compile crashes or returns a
  corrupt size.
- **A parameter-data copy Metal makes unsafely is refused**: into a `Private` buffer, or past the
  buffer's length.
- **A physical size past the map's layers has width 0.**
- **A rate-mapped pass draws region by region, and refuses a cutting scissor.**
