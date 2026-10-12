# Integrating metal2vulkan

The checklist around the library: the tools each piece needs, caching, failures, isolating a
compile, host memory a driver refuses, and checking an integration. The API itself is
[API.md](API.md); the runtime's is [its README](../crates/metal2vulkan-runtime/README.md).

## 1. Tools

The libraries read AIR bitcode (a metallib, a bitcode wrapper or a raw module) and run no external
tool. The command line (feature `cli`) runs two, both optional:

- `llvm-as` (LLVM 21 or newer) to assemble an LLVM text input (`.ll`). Not needed for a metallib
  or bitcode.
- `spirv-val` (SPIRV-Tools) to check each module it writes. When it is missing the tool says so and
  still writes the modules, which the library has verified itself.

Put them on `PATH`, or name each one:

```sh
export METAL2VULKAN_LLVM_AS=/path/to/llvm-as
export METAL2VULKAN_SPIRV_VAL=/path/to/spirv-val
```

The runtime opens the Vulkan loader when it makes a device (or takes the consumer's
`vkGetInstanceProcAddr` when it adopts one). Nothing links against Vulkan at build time.

## 2. Compile

From the command line, see the [README](../README.md#command-line). From Rust, [API.md](API.md)
walks the flow in the order a consumer meets it:

1. [A device](API.md#1-a-device): describe the Vulkan device as created and choose a policy.
2. [A library](API.md#2-a-library): parse a metallib or AIR bitcode.
3. [A compute pipeline](API.md#3-a-compute-pipeline) or
   [a render pipeline](API.md#4-a-render-pipeline), under a `Budget`.
4. [The interface and the plan](API.md#5-the-interface-and-the-plan): the layout, what each slot
   binds, what a dispatch or draw needs.
5. [At bind time](API.md#6-at-bind-time): descriptor writes and push-constant bytes from what is
   bound.

Every `air.simd_*` operation keeps Metal's 32-lane simdgroup; there is no option to use the
driver's subgroup width instead.

To read AIR without translating it (list a metallib's functions, decode a module into typed IR,
read its entry points' bindings, canonicalize an entry under function-constant values), depend on
[`metal2vulkan-air`](../crates/metal2vulkan-air). Use the same version as `metal2vulkan`, which
pins it exactly; the translator's input errors are that crate's types.

## 3. Cache

The translator keeps no cache; it gives you keys and encodings to keep one
([API.md section 7](API.md#7-caching)):

- A key is a SHA-256 of the inputs, `cache::SEMANTICS_VERSION` and the device's profile
  fingerprint, known before compiling.
- A decode checks the format and semantics versions, the key, the profile fingerprint and a
  checksum. Every `ArtifactError` is a cache miss: compile again and replace the entry.
- The versions move independently: the crate's (the API), `SEMANTICS_VERSION` (the output; a bump
  misses every cached artifact), `ARTIFACT_FORMAT_VERSION` (the encoding) and `RUNTIME_VERSION`
  (what the runtime does with an unchanged artifact).

### The runtime's shader cache

The runtime keeps every shader it translates on disk, as Metal programs get from the OS, so a
second process does not translate it again. It cannot be turned off.

- **Where.** By default `~/Library/Caches/metal2vulkan` on macOS, `$XDG_CACHE_HOME/metal2vulkan`
  elsewhere, or `~/.cache/metal2vulkan` when `XDG_CACHE_HOME` is unset. `ShaderCacheOptions`
  (in `DeviceOptions`, or `RawDevice` for an adopted device) names another directory. Several
  processes and builds may share it; each build reads only its own translations.
- **How much.** At most `capacity` bytes on disk (25 GiB by default), least recently used first;
  another build's entries go after 30 days unused. In memory, at most `memory_capacity` bytes
  (256 MiB by default): a pipeline state made again takes its translation from there without
  reading the disk.
- **The driver's pipeline cache.** Each pipeline is made with a `VkPipelineCache` of its own,
  seeded from a record named by what the making reads (its SPIR-V, layout and state, and the
  driver's identity) and written back only when the making added to it. After an update of
  metal2vulkan, a pipeline whose SPIR-V and state did not change is made from its record; a driver
  update starts empty. On drivers that report `pipelineBinaryInternalCacheControl` (NVIDIA's), a
  device the runtime makes turns the driver's own internal cache off; an adopted device keeps what
  its creator chose.
- **When it fills or a write fails.** Each device says so once, through its `MessageSink` (or on
  stderr) as a `Warning` whose `id_name` is `metal2vulkan-shader-cache-evicted` or
  `metal2vulkan-shader-cache-write-failed`. Match on `id_name`, never on the text. Later events are
  counted in `Device::shader_cache_statistics`.
- **Failures.** Making or adopting a device fails with `Error::ShaderCache` when the directory
  cannot be made or written. After that, a failed read or write only costs a translation.
- **Cold runs.** To measure with nothing cached, give the device an empty directory, and turn the
  driver's own shader cache off (`__GL_SHADER_DISK_CACHE=0` for NVIDIA,
  `MESA_SHADER_CACHE_DISABLE=true` for Mesa). Give each test its own directory.
- **It is a cache.** Deleting the directory, or any file in it, is always safe.

## 4. Handle failures

Every failure is a typed value; match variants, never messages
([API.md section 8](API.md#8-refusals-requested-inputs-and-precision)).

- A refusal names what would change the answer: the feature the device lacks, the pipeline input
  the compile needs, or a limit.
- A deadline or a cancellation (`Error::Budget`, `Error::Cancelled`) is transient: retry later.
  Every other error is the answer for that input and device and may be cached as one
  (`error.retry()`, `error.is_permanent_for()`).
- Never use a module from a compile that returned an error.
- The command line prints `metal2vulkan: FALLBACK (<why>)`, exits 2, and writes a repro bundle
  (the input, the profile, `repro.sh`, `failure.txt`) under `$TMPDIR/metal2vulkan-repros`, or
  `METAL2VULKAN_REPRO_DIR`.

## 5. Bound a compile

Every compile runs in-process under the caller's `Budget`: a wall deadline (`Budget::standard()`
is 20 seconds), a `CancelToken` any thread may trigger, a deterministic work-unit cap and a byte
ledger charged before each input-sized allocation. A panic inside a compile becomes
`Error::Internal`. That is enough for a consumer that trusts its process.

When an abort, a stack overflow or a failed allocation in the compiler must not take the consumer
down, or a hard memory ceiling must hold, compile in a child process:

1. Spawn one worker per compile in its own process group, with the input in a file and the result
   written to a file. Send the library bytes, the entry and descriptor, and the device description
   and policy (with feature `serde` they serialize; equal description and policy give the same
   profile fingerprint and keys).
2. In the child, install a counting global allocator that refuses allocations above a budget, build
   the `Device`, and compile with the same `Budget`.
3. In the parent, poll the deadline, cancellation and the process group's resident memory. On a
   breach, kill the whole group and remove its scratch files. A result that arrives late is a
   timeout.
4. Load the result with `PipelineArtifact::from_bytes` against the parent's own `Device`, and check
   that its key is the one the parent computed (`Device::compute_pipeline_key`,
   `render_pipeline_key`).
5. Classify as in-process: a typed error is the answer; a deadline, cancellation or killed child is
   transient.

The validation crate's worker (`crates/metal2vulkan-validation/src/worker/`) implements this,
under 20 seconds and 500 MiB per attempt.

## 6. Host memory a driver refuses

`Device::new_buffer_with_bytes_no_copy`, `new_heap_with_bytes_no_copy` and the
`rebind_bytes_no_copy` calls import the caller's pages by their pointer
(`VK_EXT_external_memory_host`). Some drivers refuse the pointer of shared memory: RADV (AMD on
Linux) imports a pointer through amdgpu's userptr, which takes anonymous pages only, so it refuses
`MAP_SHARED` mappings of a `memfd`, a shared memory object or any file (QEMU's
`memory-backend-memfd,share=on` guest RAM among them) with `Unsupported`, `HostMemory`. NVIDIA and
lavapipe import the same memory.

A consumer whose memory is such a mapping passes `HostMemory::with_dma_buf(pointer, source)` instead
of `HostMemory::new(pointer)`. The runtime still imports the pointer first, exactly as without a
source, and calls the source only where the device refuses it: for dma-bufs of the pages the range
touches, and again once for each texture's image over its own pages in them. The rustdoc of
`HostMemory::with_dma_buf`, `DmaBufSource` and `DmaBufRequest` is the contract: when the source is
called, the chunks it answers, and what is refused. In short, answer each `DmaBufRequest` with
dma-bufs of exactly the pages asked for, in order, split only at multiples of its `granule`. Where
the granule is less than the length (64 KiB on RADV of Mesa 26.2.4, 4 KiB on 26.0.8), several
chunks are bound to one sparse buffer and must total the length rounded up to the granule: pad the
last with a scratch page of your own, repeated. A texture's request takes one chunk, of the same
file pieces the buffer's chunks hold there.

On Linux, `udmabuf` makes such a dma-buf of the file pieces behind the pages: open `/dev/udmabuf`
(the process needs read and write access to it) and issue `UDMABUF_CREATE_LIST` with one item per
run of contiguous file pages; the `memfd` must be sealed against shrinking (`F_SEAL_SHRINK`, as QEMU
seals it) and not against writing. The consumer knows the pieces behind every byte; the runtime runs
no udmabuf code. Each udmabuf takes at most `list_limit` items and `size_limit_mb` MiB (the module's
parameters under `/sys/module/udmabuf/parameters`, 1024 and 64 by default). Keep each chunk within
the host's limits as they are: read them, never assume them raised. Scattered memory needs many
items: a macOS desktop guest of 14 GiB was measured importing up to 63.3 MiB in one object, in up to
16,201 separate 4 KiB runs, with 19 to 29 of its imports past 1024 runs; at a 64 KiB granule such an
object takes 16 chunks.

A texture over such memory is an image over a dma-buf of its own pages where the device also lists
`VK_EXT_image_drm_format_modifier` and `VK_EXT_queue_family_foreign` (RADV does). RADV (Mesa 26.2.4
and 26.0.8) lays such an image's plane only at an offset in its page that is a multiple of 256
bytes, with rows a multiple of 256 bytes apart: keep a texture's offset and `bytes_per_row` so (a
4 KiB-aligned surface with 256-byte-aligned rows always is). Elsewhere the texture takes a shadow of
the bytes, announced (`shadowed-texture`, naming why), and a rebind that cannot make the image again
is refused (`NotImplemented(RebindTextures)`). `WorkCounts::dma_buf_imports`,
`dma_buf_sparse_buffers`, `dma_buf_images` and `dma_buf_image_fallbacks` count what the route did.

## 7. Check the integration

`spirv-val` checks structural validity, not agreement with Metal:

```sh
spirv-val --target-env vulkan1.2 output.spv
```

`device.profile().spirv_validation()` names the environments and relaxations a module compiled
for the device is valid under; the command line runs each.

For a claim about behavior, follow [VALIDATION.md](VALIDATION.md): a synthetic regression test,
the golden output hashes where output must not move, and an authored case run on Metal and Vulkan
where it should.
