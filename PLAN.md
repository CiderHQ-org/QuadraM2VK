**QuadraM2VK — proposed fork and integration plan**

Prepared 2026-10-20 as a source audit and implementation proposal, before implementation began. The findings and priorities below describe that audit snapshot. See [README.md](README.md) and [RESULTS.md](RESULTS.md) for the implemented fork, measurements, checks and remaining work.

The [deeper performance and stability audit](/home/aubree/Documents/QuadraM2VK/PERFORMANCE-STABILITY-AUDIT.md) supersedes the priority ordering below. Its main additions are a source comparison of upstream draw optimizations, a native BC decode policy, drawable cancellation and partial-construction cleanup, correct texture-copy footprints, and missing compute/synchronization forwarding. Recurring gameplay costs and these stability defects now precede native legacy AIR implementation.

The intended result is a maintained metal2vulkan fork for Quadra's Metal renderer, with companion fixes in its existing guest/host bridge. Quadra, Cider, and Mac O' Blox are successive names of the same local project; its internal compatibility names remain in use. CiderDarling supplies the macOS compatibility environment.

**What was inspected**

- Upstream `steelbrain/metal2vulkan`, `master` snapshot [`a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b`](https://github.com/steelbrain/metal2vulkan/tree/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b), dated 2026-10-19: AIR parsing, pipeline scheduling, caches, allocation, runtime interop, validation documentation, and CI.
- Quadra's current dependency pin, `720ef1ce9ae2a2068999a8675717418b9416283e`, dated 2026-10-14, read from the host manifest and existing Cargo checkout. The newer snapshot has substantial changes; an update needs its own compatibility and performance comparison.
- The local [Metal bridge and its documented status](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/README.md), including its Objective-C framework, Rust host, AIR upgrader, and presenter.
- Quadra's [project README](/home/aubree/Documents/MacOBlox/README.md), hardware/bug notes, and CiderDarling's [performance guide](/home/aubree/Documents/CiderHQ-DarlingFork/docs/PERFORMANCE.md).

The bridge README records gameplay on Roblox 0.743.0.7431134, CiderDarling b0fcb070, an RTX 3060 Ti, and native Wayland. It also records decoding 2,266 shader libraries and compiling 40 compute kernels. These are prior documented results, not runs performed during this audit. Neither count establishes coverage of all render pipeline combinations or prolonged gameplay.

**Existing foundations to preserve**

Upstream already provides translated-function and pipeline caching, persistent driver pipeline-cache records, bounded in-memory memos, asynchronous pipeline workers, prioritized compilation, hazard tracking, and a Vulkan memory suballocator. See its [integration guide](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/docs/HOWTO.md), [pipeline pool](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/pipeline/pool.rs), and [allocator](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/memory.rs).

Quadra already catches panics around many host operations, gives concurrent AIR-cache writers separate temporary files before rename, presents from a dedicated host thread, and selects upstream's host-clock GPU timing option. Those mechanisms should be extended where evidence requires it. The host-clock option means its reported GPU duration cannot serve as a hardware GPU benchmark.

**Prioritized findings and proposed changes**

1. **Repair memory accounting and capability reporting. Priority: first batch. Owner: Quadra bridge; optional budget-query support in the fork.**

   Confirmed source behavior: [`count_allocation`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/util.rs:74) adds buffer/texture estimates to a process-wide counter and never subtracts releases. [`device_current_allocated_size`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:384) returns that cumulative value. Thus allocate/release cycles can report continually growing use even when resources have been freed. Whether this currently changes Roblox's streaming or quality decisions needs a runtime check.

   The bridge also [defaults its recommended working set to 4 GiB](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:348), unless overridden, and [hardcodes family/feature-set answers](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:310), despite exposing a host capability query. These answers need to reflect the selected device and implemented bridge surface.

   Proposed change: define per-device live allocation accounting, including correct ownership of views and resources still retained by submitted work. Keep live resource bytes, committed Vulkan pool bytes, and available heap budget distinct. Use upstream allocation counters as a cross-check, and add `VK_EXT_memory_budget` querying where supported with a documented fallback. Route capability answers through the host and account for APIs the bridge actually implements.

   Acceptance: allocate/release and aliasing fixtures return to an explained steady state after GPU completion; multiple devices do not share totals; forced capability profiles never advertise an unavailable operation. Validate UMA and discrete-GPU budget behavior separately.

2. **Fix command queue lifetime, scheduling, and error propagation. Priority: first batch. Owner: Quadra bridge.**

   Confirmed source behavior: [`track:`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:543) retains the queue when it starts its detached worker. [`queue_worker`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:579) loops forever with no stop condition or matching release of that worker retain. Thread creation failure also leaves its running state set. [`waitUntilScheduled`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMObjects.m:699) is empty, while scheduled handlers run when the guest worker dequeues a buffer rather than from an observed host scheduled transition.

   Separately, [`report`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:1037) only logs runtime encoder errors. The framework [fabricates an object-returning method signature for unknown selectors](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/framework/QMHost.m:97) and returns zero. This can obscure unsupported calls and does not represent every Objective-C return ABI.

   Proposed change: give guest completion workers a defined retirement/restart lifecycle and failure path; implement scheduling waits through the runtime's existing state transitions while keeping guest blocks on guest threads. Preserve the first real encoder failure in the owning command buffer and expose it through Metal error/status behavior. Implement observed selectors with correct signatures; keep harmless optional probes explicitly defined and treat unsupported substantive operations according to their actual API contract.

   Acceptance: repeated queue creation/commit/release retires threads and host handles; thread-creation failure does not strand handlers; scheduled/completed blocks run once in the required order; an injected encoder error cannot become a successful command buffer. Test handlers that call back into command-buffer methods.

3. **Reduce presentation stalls and validate swapchain lifetime. Priority: first batch. Owner: Quadra presenter and layer bridge.**

   Confirmed source behavior: [`next_drawable`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/present.rs:531) holds `inner` across image acquisition and a CPU fence wait. [`queue_present`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/present.rs:575) needs that same mutex. When free images depend on pending presentation, this creates a potential acquire/present stall. This is a source-derived risk, not a reproduced gameplay diagnosis. Retired-chain cleanup also calls [`queue_wait_idle`](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/present.rs:503) from the drawable-acquisition path.

   Proposed change: first reproduce acquisition while all drawables are held and presentation is delayed. Then separate short state updates from blocking acquire/present/retirement operations, with explicit swapchain-generation ownership. Evaluate acquire semaphores attached to the first GPU use of a drawable to avoid a host fence wait. Retire swapchains only after both rendering and presentation have released their resources; use supported presentation-completion facilities and a correct conservative fallback.

   Acceptance: delayed presentation cannot block the presenter behind acquisition's state mutex; repeated resize/minimize/restore and surface recreation do not leak resources or stall the UI. Cover FIFO, mailbox, and immediate only where supported. Measure acquire/present tails, queue depth, and frame times; preserve semaphore and image-layout correctness.

4. **Support legacy metallib/AIR directly. Priority: core fork work after baseline and bridge fixes. Owner: `metal2vulkan-air`, with translator metadata changes as needed.**

   Confirmed gap: upstream's [bitcode reader](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-air/src/bitcode/module.rs#L292) accepts module version 2. Roblox's older input uses version 1. Its metallib layout also predates the extended-header layout upstream parses. Quadra currently [loads libLLVM and rewrites textual IR](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/air.rs:1), then [inserts an extended header](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/lib.rs:661).

   Proposed change: implement the older container layout and version-1 bitcode records structurally, normalize legacy intrinsic signatures into the existing typed IR, and derive texture access and buffer extent from the legacy ABI. Preserve the runtime-sized StructuredBuffer correction without removing genuine fixed-size bounds. Keep modern-format behavior covered throughout. This is a parser/ABI feature, not a version-number relaxation or an app/shader-name special case.

   Acceptance: synthetic old/new inputs produce equivalent typed meaning and GPU results for samples, gathers, reads/writes, atomics, render outputs, and runtime-sized buffers. Include multiple entry points, malformed records, and genuine fixed-size buffers. Privately rerun the full installed pack and recorded render descriptors; public regressions use authored inputs. Retire the libLLVM workaround only after differential checks pass. Expected benefits are fewer runtime dependencies and fewer conversion/cache failure paths; warm-cache FPS gains are unproven.

5. **Make pipeline concurrency sensitive to memory pressure. Priority: core fork performance/stability work. Owner: runtime pipeline budget.**

   Confirmed policy: [`turns_for`](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/pipeline/budget.rs#L90) chooses at least four concurrent pipeline creations, rising to twelve on larger Linux hosts. It is CPU-based. Its comments report roughly 0.4–0.5 GB of driver memory per concurrent compile for measured large shaders. Those measurements concern other workloads; Roblox's peak remains unknown.

   Proposed change: expose a supported process-level compile policy and combine effective CPU availability with cgroup/host memory headroom and measured per-job cost. Allow smaller concurrency under constrained memory. Preserve urgency ordering, aging, and progress for work a submitted command buffer needs. Bound queued work and avoid holding compilation turns while waiting on another job.

   Acceptance: cold pipeline storms under one/two effective CPUs and constrained memory complete without deadlock or sustained memory growth. Compare time to first useful frame, p99 frame time, and process-tree peak memory against the current policy. Driver allocation estimates guide concurrency; they are not a hard in-process memory guarantee. AIR translation deadlines do not cancel a blocked `vkCreate*Pipelines` call.

6. **Make caches resilient and prewarming workload-aware. Priority: after correctness baselines. Owners: runtime cache and Quadra loader.**

   Confirmed behavior: upstream [requires a writable disk cache at device creation](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/crates/metal2vulkan-runtime/src/shader_cache/mod.rs#L256). A cache-directory failure can prevent opening the renderer. Later I/O failures already degrade to misses. Quadra's separate [AIR-upgrade cache read](/home/aubree/Documents/MacOBlox/prototype/metal-vulkan/host/src/air.rs:574) accepts cached bytes without validating the upgraded artifact there; malformed bytes can fail later without retrying the original upgrade.

   Proposed change: offer an explicit bounded memory-only cache mode for unavailable storage, retain the default persistent mode, and report the changed startup cost. Validate/rebuild corrupt legacy-upgrade entries while that cache exists; bound its size and deduplicate concurrent upgrades of the same input. Record actual render descriptor combinations and prewarm useful variants during loading or idle periods within the same resource budget. Keep version, device, policy, and driver identity in the appropriate cache keys.

   Acceptance: read-only/full/unavailable storage and corrupt entries do not produce false success or endless retries; simultaneous writers remain safe. Compare empty-cache, warm-cache, and driver-update runs. Parsing all shaders or warming only compute pipelines does not count as warming the render workload.

7. **Reduce generated shader size where profiling supports it. Priority: later experiment. Owner: translator lowering.**

   Upstream's [known gaps](https://github.com/steelbrain/metal2vulkan/blob/a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b/docs/FAST-PATHS.md#known-gaps) identify duplicated sampling helpers as a cold NVIDIA compilation cost. Investigate sharing compatible generated helpers after identifying the largest Roblox render modules. Measure translation work, emitted words, driver compile time, and GPU time. Retain an optimization only if semantic comparisons and the target drivers pass; a smaller module can still compile or execute more slowly.

   The same profiling pass should rank descriptor updates, barriers, upload traffic, and cache-lock contention. Change only demonstrated hot paths. Preserve Metal's memory, sampling, and synchronization contracts and the existing explicit accuracy policies.

**Implementation order and review gates**

1. Establish the baseline. Preserve the current Quadra pin, record the client/runtime/driver versions, and compare it with a separately pinned latest-upstream candidate, including pushed descriptors, shared layouts, table stamps, and reusable draw storage. Add bounded numeric timing and counters for draw encoding, shader load, translation, pipeline creation, drawable acquire, submit, and presentation. Record actual render descriptor combinations and compressed-texture formats. Confirm which upstream changes improve this workload before selecting the fork's base revision.
2. Deliver the bridge repair batch: valid texture-copy footprints; drawable cancellation, swapchain construction cleanup, and presentation locking; compute dispatch/barrier/fence/residency forwarding; function constants; allocation/capability answers; encoder/queue lifetime; scheduling; and sticky command errors. Keep each change independently reviewable with a regression that fails before the fix.
3. Add and measure a native BC decode policy with the exact twin path retained. Evaluate repeated-binding suppression and bulk host setters, preserving validation and resource lifetime semantics. Measure warm gameplay throughput, frame-time tails, upload work, and memory pressure.
4. Add resource-aware pipeline limits, cache fallback/repair, and descriptor-driven prewarming. Deliver native legacy AIR support in small parser, intrinsic, and metadata increments; remove the upgrader only after differential checks pass. Rebenchmark each change independently.
5. Investigate shader-size, upload, and layout hot paths supported by profiling, then qualify GPU/display combinations and package the selected revision. Keep the previous renderer/runtime selection available for rollback.

Keep upstream crate names initially, pin all three product crates to the same revision, retain their notices, and maintain a small set of focused fork changes. Keep reusable parser/runtime fixes in the fork and guest Objective-C/window integration in Quadra. Extend upstream's existing validation/CI with bridge tests and real hardware runs.

**Evidence required before calling a release improved**

| Area | Required comparison or check |
| --- | --- |
| Startup | Empty and warm caches; time to first frame and usable scene; separate translation and driver compile costs |
| Gameplay | Same client, place, resolution, quality, frame cap, and present mode; repeatable capture or fixed camera sequence; median/p95/p99 frame times and 1% lows |
| Correctness | Actual render pipelines and draws, textures, instancing, depth/stencil, MSAA, visibility queries, and exact readback for authored cases; distinguish shader validity from correct output |
| Resources | CPU use, process-tree RSS, live/committed GPU allocations, thread/FD counts, cache hit/miss counts, and a long-session memory trend |
| Lifecycle | Repeated queue creation, join/leave, resize/fullscreen, minimize/restore, surface destruction, shutdown, and interrupted work |
| Failure handling | Allocation/compile/encoder/thread failures, corrupt/full/read-only cache, lost surface/device error propagation, and useful retained error details |
| Portability | Documented RTX 3060 Ti/Wayland first, then AMD RADV, Intel ANV/Arc, other NVIDIA hardware, hybrid-device selection, and X11 separately |
| Automated checks | Upstream format/lint/docs/unit/doctest gates; authored lavapipe execution and Vulkan validation; bridge C ABI/guest lifecycle regressions; real Metal comparisons for changed shader semantics where available |

Run shader execution checks on authored inputs before the real workload. Preserve upstream's translation work/memory limits. Track gameplay results per device and display backend; software Vulkan and one NVIDIA system cannot establish cross-vendor gameplay support. VR requires its own renderer integration because the current bridge targets OpenGL.

No fixed FPS improvement is promised. Each optimization must improve its intended measured workload without correctness regressions or unexplained tail-latency/memory regressions. A device-loss path should report and terminate/restart cleanly through the launcher; transparent recovery is a separate feature requiring its own resource-recreation design.

The first implementation milestone is a reproducible upstream comparison plus the expanded bridge repair batch. The first fork-specific gameplay performance candidates are native BC handling and reduced draw-encoding work. Native legacy AIR support and resource-aware pipeline compilation remain substantial compatibility and startup/stutter improvements.
