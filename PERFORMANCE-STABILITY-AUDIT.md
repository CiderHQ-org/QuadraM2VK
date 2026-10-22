**QuadraM2VK — deeper performance and stability audit**

2026-10-20 audit snapshot, prepared before implementation. This extends [PLAN.md](PLAN.md). Sources: Quadra's existing Metal bridge; its upstream pin `720ef1ce9ae2a2068999a8675717418b9416283e`; upstream snapshot `a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b`. Both source revisions were compared locally. The audit itself did not include builds or execution. See [RESULTS.md](RESULTS.md) for subsequent implementation and validation. Numeric texture sizes below are arithmetic, not driver allocation measurements.

“Confirmed” means the implementation contains the described behavior. A failure scenario derived from that behavior still needs an executable regression. A performance candidate needs measurements of this workload before it becomes a claimed improvement.

The priorities are warm gameplay throughput, frame-time tails, and stability over a session. Native legacy AIR remains useful, principally for startup and dependency reliability. The deeper audit moves recurring draw costs, texture expansion, and drawable ownership ahead of that parser work.

**Work ranked by likely impact**

| Order | Work | Expected benefit | Evidence and scope |
| --- | --- | --- | --- |
| 1 | Compare and qualify the newer upstream draw path | Less CPU work in draw-heavy scenes | Several mechanisms are absent from Quadra's pin and present in the newer source; target speedup unmeasured |
| 2 | Repair drawable ownership, swapchain failures, and presentation locking | Avoid exhausted image pools, recovery failures, and long frame stalls | Concrete missing cleanup and shared blocking locks in the bridge; failure injection required |
| 3 | Correct texture footprints and forward explicit compute/synchronization semantics | Prevent invalid memory spans, incorrect dispatches, and unsupported synchronization being silently ignored | Confirmed bridge behavior; actual game use of each call needs counters |
| 4 | Add a native BC decode policy while retaining exact twins | Less GPU memory and texture update work | Exact decoded twins are created for eligible BC1–BC5 images on devices without the relevant measured decode fact |
| 5 | Preserve unchanged binding state and batch host calls | Lower CPU encoding cost | Latest setters still invalidate a binding-table stamp even for repeated equal bindings; bridge bulk setters loop over scalar calls |
| 6 | Bound pipeline work by memory headroom and prewarm measured variants | Fewer compile-related hitches and memory spikes | Existing CPU-based concurrency and cache behavior confirmed; target workload size unknown |
| 7 | Optimize sampled uploads, layout transitions, and generated shader helpers | Lower bandwidth or compile cost where profiling identifies them | Experiments; several analogous upstream optimizations already exist |

**1. The current dependency pin misses relevant upstream CPU optimizations**

Quadra's pinned render path allocates and updates a set-0 descriptor set for each draw that needs it. The newer source has a [push-descriptor path for eligible layouts](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/pipeline/objects.rs#L1070), [shared pipeline layouts](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/pipeline/shared_layouts.rs), and [binding-state reuse](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/live_bind.rs). The last includes table-generation stamps, per-pipeline saved states, and reusable vectors for command construction, descriptor writes, push constants, uses, and vertex bindings. Source checks found all four mechanisms absent from the pin and present in the newer snapshot.

The first experiment should compare these revisions through the same Quadra bridge. Count how often actual Roblox layouts qualify for pushed descriptors, how many descriptors are written per draw, and how often a table or bind shape is rebuilt. A feature present in source provides no benefit to a workload that cannot take it. The existing [draw-cost workload](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-validation/src/runtime_lane/draw_costs.rs) supplies unchanged, inline-bytes, texture-switch, and pipeline-switch streams. Its measurement configuration uses the system allocator and release builds; its counting allocator changes measured cost. Extend the streams with instancing and offset changes representative of this bridge.

The newer revision is a candidate to qualify. Preserve the tested pin as the baseline and rollback point. Changes between the revisions are substantial, so an upgrade needs correctness and memory checks as well as timings. Khronos also identifies per-draw descriptor updates as a potential CPU bottleneck, but its sample's measured speedup does not transfer to Quadra. [Descriptor management sample](https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html)

**2. Compressed textures have a substantial avoidable cost to investigate**

In [the twin implementation](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/texture/decoded_twin.rs#L464), eligible single-sample BC1–BC5 textures gain an additional decoded image when the device lacks a measured fact that its block decode matches Metal. Both images stay live. Writes update the compressed image and decode into the other image; some CPU upload paths use a decode worker. This applies to supported native BC formats, separately from emulation of formats the GPU lacks.

Illustrative 4096×4096 base-level storage, excluding mipmaps, alignment, and image metadata:

| Format | Compressed image | Additional decoded image | Combined texel/block storage |
| --- | ---: | ---: | ---: |
| BC1 | 8 MiB | 64 MiB | 72 MiB |
| BC3 | 16 MiB | 64 MiB | 80 MiB |
| BC5 | 16 MiB | 128 MiB | 144 MiB |

These are individual texture examples, not an inventory of Roblox's textures. Measure format distribution, twin bytes, CPU decodes, decode dispatches, upload duration, and memory-pressure stalls first.

Proposed fork change: expose a policy that uses the hardware's supported BC decode and filtering, with recorded accuracy differences, while retaining the exact twin path as an explicit choice. The current compact-versus-exact decoded-texel policy concerns carried ASTC/EAC textures; it does not disable BC twins. A native BC policy must update the appropriate runtime/cache semantics and notices, and validate signed channels, sRGB views, alpha edges, and partial writes. Preserve emulation for unsupported formats. This could reduce both memory pressure and upload work; an FPS gain remains a hypothesis.

**3. Repeated setters can defeat the new binding-table fast path**

The newer [buffer setter](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/argument_table.rs#L460) takes a fresh stamp before checking whether the bound object, offset, and stride changed. Repeated equal resource bindings can therefore force the next draw to revisit the table despite unchanged binding values.

Candidate: after required validation, preserve the stamp for an equal binding of the same kind. Include offset, stride, sampler clamps, and competing slot representations in equality. Keep the separate checks for no-copy rebinds, detaches, and changed view-alignment state. Never treat an unchanged pointer as proof that `setBytes` has unchanged data or that a CPU write can be skipped.

The bridge's [bulk render setters](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:777) expand arrays into individual Objective-C methods and host calls. Forward a validated bulk operation through one C ABI entry and the runtime's bulk API. Besides reducing calls, this preserves the runtime's validation-before-binding behavior instead of applying earlier scalar updates before a later failure. Measure host calls, table refreshes, allocation counts, and CPU encode time with equal and changing bindings.

**4. Drawable cancellation and partial swapchain construction need ownership fixes**

The host [Drawable](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/present.rs:269) owns an acquired image index but has no drop/cancellation path that returns an unused acquisition. Dropping a guest drawable releases its host wrapper, not the Vulkan acquisition. Repeating acquire/release without presenting can consume the swapchain's available images until acquisition times out. This is a derived failure scenario; it was not executed here. Metal's drawable reuse depends on releasing strong references, and Vulkan offers unused-image release through swapchain-maintenance support. [Apple drawable lifetime](https://developer.apple.com/documentation/quartzcore/cametallayer), [Vulkan image release](https://docs.vulkan.org/refpages/latest/refpages/source/vkReleaseSwapchainImagesKHR.html)

Add an acquisition token with explicit states for acquired, referenced by encoded work, submitted, presented, and abandoned. Reclaim an image only once its outstanding GPU uses are safe. Use negotiated swapchain-maintenance functionality where available and define a valid fallback otherwise. Simply returning the index to a Rust free list would leave Vulkan ownership wrong.

The [rebuild path](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/present.rs:431) removes the old chain from `inner` before fallible creation and only adds it to the retired list after success. `Chain` has no Vulkan-handle destructor. A failed creation loses that tracking; failure while retrieving/wrapping new images or creating semaphores also lacks complete rollback. An old swapchain passed to Vulkan is retired even when replacement creation fails, so restoring the previous active state is insufficient. [Swapchain retirement rules](https://docs.vulkan.org/refpages/latest/refpages/source/VkSwapchainCreateInfoKHR.html)

Use scoped owners for the new handle, wrappers, and semaphores, and retain every retired generation until safe destruction. Inject failure at every construction step. Repeated recoverable resize/allocation failures must neither leak handles nor make subsequent recreation impossible.

**5. Presentation waits can affect the rest of rendering**

The acquisition path holds the host surface mutex over blocking acquisition/fence waits. Presentation takes the same mutex. The guest [layer acquisition](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMLayer.m:219) additionally keeps its layer mutex while calling the host. This creates a lock chain spanning guest state, host state, and WSI waits. Reproduce all-images-held, delayed presentation, resize, and minimized-window scenarios.

Separate short state snapshots from blocking work, protect generations explicitly, and attach acquire synchronization to GPU work where appropriate. Per-image presentation semaphores already exist and should be preserved. Reuse/destruction must follow presentation completion; render completion alone is insufficient. [Khronos semaphore guidance](https://docs.vulkan.org/guide/latest/swapchain_semaphore_reuse.html)

The runtime [reserves presenting queues where hardware permits](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/queue.rs#L199). Devices without spare queues share one with other work. Thus a blocking present/idle operation can have different consequences across vendors. The guest always sets visibility to true; propagate actual visibility and avoid unnecessary work while hidden. Test the shared-queue path explicitly.

Timed presentation methods also ignore their requested time/minimum duration. Presented handlers currently use command completion or immediate host time, not display feedback. These are pacing and measurement gaps. Avoid reporting their timestamps as measured display latency.

**6. Compute and synchronization semantics are lost at the bridge**

Confirmed in [compute dispatch](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:1146): `dispatchThreads` rounds the requested grid up to whole groups. A requested 65-thread grid with groups of 64 becomes 128 dispatched threads. Kernels that need the exact grid can produce incorrect results or reach extra indices. Upstream already implements `dispatch_threads`, including partial groups; add the C ABI entry and forward the original dimensions. Check edge-group builtins, barriers, zero extents, and dimensions around a group boundary. [Apple grid semantics](https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes)

[Dynamic threadgroup-memory sizes](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:1086) are logged and ignored. Explicit compute/render barriers, encoder fences, resource residency, and heap residency calls are empty. Automatic tracking can order ordinary tracked resources, so this does not prove every current frame races. It leaves applications' explicit untracked/indirect dependencies unsupported. Forward the existing runtime operations and give missing resources/features explicit errors. Test write-to-read dependencies with untracked resources, explicit barriers/fences, and resources reached through addresses or argument buffers.

[Function-constant values](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:132) are discarded when creating a specialized function. Preserve them through the ABI and cache identity. Test two constant values that must produce different outputs. Compute texture bindings are also limited by a 31-entry array despite the runtime supporting 128 texture slots; reconcile implemented limits with capability answers. Instrument use before claiming these cases explain an existing game failure.

**7. Texture-copy byte spans are calculated incorrectly before unsafe borrows**

The host [packed-length helper](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/util.rs:104) estimates bytes per texel from broad numeric format ranges. For example, it gives format 125 (`RGBA32Float`) 8 bytes per texel instead of 16, and format 252 (`Depth32Float`) 16 instead of 4. Compressed formats require block geometry. The [CPU upload wrapper](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:537) uses texel height instead of block rows for pitched copies and forms its Rust slice before runtime layout validation.

An overlarge slice is invalid even if later code touches only a smaller prefix; Rust requires the whole slice to lie in one valid allocation. `catch_unwind` does not repair undefined behavior. [Rust slice requirements](https://doc.rust-lang.org/std/slice/fn.from_raw_parts.html)

Add a public checked footprint query shared with the runtime's copy planner, then form the borrow over the exact required host span. For ordinary layouts, the span is `(images - 1) × image_pitch + (block_rows - 1) × row_pitch + row_bytes`; retain format-specific rules for packed/carried formats. Check arithmetic, size conversion, empty regions, and null handling before constructing a slice. Apply the same contract to readback. Test tightly sized buffers, padding, compressed edge blocks, mip levels, 1D/3D copies, and malformed pitches.

**8. There is an additional encoder-handle leak**

Switching from ordinary compute to a [builtin texture upload](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:1122) ends the host encoder and sets its handle to zero without calling host release. [Host end_encoding](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:1416) consumes the encoder inside the boxed handle, not the box itself. Repeated switches therefore leak host wrapper allocations. This finding does not by itself establish leaked GPU memory.

Pair end/release on every switch and failure path. Keep the previously identified queue-worker retirement/failure fixes in the same lifecycle workstream. Also forward the [requested maximum command-buffer count](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:486), which currently gets discarded. Test repeated switches, queue creation/release, callback failures, and bounded pending work.

**9. Upload batching is useful, but its semantics must remain explicit**

Compute `setBytes` currently allocates and copies into guest storage; when a host encoder exists, the runtime copies the same input into its upload arena too. Reopening after a builtin upload replays the bindings. Reuse guest storage capacity, track bindings needed for replay, and batch consecutive compatible builtin blits. Preserve the required copy at `setBytes` time and the ordering of ordinary compute around uploads.

The runtime already uses growable upload chunks, a Vulkan suballocator, spare pools, and dedicated texture staging where supported. Add counters for allocation/zero-fill/pack/copy/decode work before changing those mechanisms. A synchronous Metal CPU texture-write call must make its caller's bytes safe to reuse before returning. Moving its source pointer to an asynchronous job without owning the bytes would introduce a lifetime bug.

**10. The measurement and release plan must distinguish bottlenecks**

Add a bounded diagnostics entry to the versioned C ABI. Reuse `Device::work_counts`, shader-cache statistics, and pipeline-creation diagnostics. Useful deltas include draw count, pushed writes, descriptor allocations, table/shape rebuilds, variant creation, replay commands, submits, upload bytes, twin bytes/decodes, and pool high-water marks. Sample the full statistics snapshot periodically or after a hitch; it takes internal locks and should not run per draw. Use cheap counters/timers for frame boundaries, keep trace buffers bounded, and measure collection overhead. Preserve ABI size/version checks when adding calls.

Measure CPU encode, translation/driver compile, submission delay, actual GPU execution, acquire wait, and presentation separately. The bridge sets `gpu_times_from_host`, and presented callbacks currently do not establish display time. Use a sampled diagnostic GPU timestamp path and supported presentation feedback, measure instrumentation overhead, and use production settings for the final comparison. CPU timers cannot measure GPU execution. [Khronos profiling guide](https://docs.vulkan.org/guide/latest/profiling.html)

Start with the documented RTX 3060 Ti/native Wayland configuration. Compare pin/new upstream/fork using the same client, driver, scene, camera, settings, resolution, and present mode, with isolated cold and warm caches. Separate uncapped CPU-heavy tests from the frame-cap/refresh-rate experience. Report frame-time median/p95/p99, hitch counts, CPU time per draw, first-use pipeline stalls, peak/post-soak memory, and thread/FD/handle counts. Keep validation-enabled correctness runs separate from release performance runs.

Use authored stress streams for deterministic regression, then repeat representative gameplay runs. Include a long session and repeated join/leave, resize/minimize, dropped frames, and teardown. Fault-inject cache, allocation, thread, pipeline, and WSI failures. Real AMD/Intel/hybrid and X11 checks follow; a CPU ICD can check semantics but cannot certify hardware performance. A finite wait expiring must preserve resources still used by GPU work and lead to a controlled failure path.

**Experiments to keep behind measured evidence**

- Read-only/attachment-specific image layouts and narrower usage flags may help bandwidth or compression. The current runtime rests images in `GENERAL`, with existing Intel depth optimizations and aliasing rules. Scope any new layout tracking by subresource and queue; include caller-owned images, storage access, feedback, and views. Khronos's layout sample is Mali-specific evidence, not proof of a desktop gain. [Layout sample](https://docs.vulkan.org/samples/latest/samples/performance/layout_transitions/README.html)
- Additional graphics-pipeline-library or dynamic-state work should first verify which existing paths the device enables. Prewarm actual observed descriptor/specialization combinations within resource limits. Do not delay a required draw indefinitely behind speculative compilation.
- No-copy import tuning, mesh/tile support, and a complete bindless/descriptor-buffer rewrite have lower priority until this bridge and workload exercise them. Keep cache keys, containment, synchronization, and exact-mode regressions intact during performance work.
- Command pools deliberately remain at their high-water count during device lifetime because upstream documents an NVIDIA destruction hang. Monitor that memory before proposing trimming; preserve that workaround until a replacement is qualified.

The revised first milestone is a measured upstream comparison plus a bridge that correctly owns drawables, survives partial construction failures, computes valid copy spans, and forwards the required GPU semantics. The next fork-specific performance milestone is native BC handling with explicit accuracy policy, followed by demonstrated reductions in CPU encoding and compilation pressure.
