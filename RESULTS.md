# QuadraM2VK implementation and qualification

2026-10-22. The current local fork, Quadra launcher and matching ABI 2 bridge are built and
committed. Shader preloading, live-memory accounting, cache recovery and diagnostics have
authored software Vulkan coverage. The native inputs of the current `4430bd2` bridge match `cb5b342`, which passed agent-run RTX
3060 Ti checks, including native Wayland and the first-launch preparation stages. Quadra 0.34 integrates this renderer as **Quadra M2VK**
and pins the public runtime Git revision. Its release source `df78a59` has the same tree
as companion `4430bd2`. The user reports persistent home/minimum-graphics slowdowns gone and smaller first-visit drops
with `57af2fb`. Gameplay verification of the additional launcher stages remains open.
The older benchmark and full RTX regression results below describe `c50214a` / `5cbc9fb`.

## 2026-10-23 follow-up: hardware compatibility

Measured on this computer's RTX 3060 Ti (NVIDIA 615.78.08, Linux 7.3.0, Wayland) unless noted. Each
item says what was run; none is a gameplay frame-time measurement.

- **Base.** Quadra 0.35.p2, the version installed on this computer, is ahead of this workspace's 0.34-based
  bridge tree and has its own graphics memory budget (`memory.rs`: a one-second poll, the largest
  device-local heap for a discrete card, a low cap for APUs after a real report). The working-set change
  first made here was reverted in its favor; everything else was replayed onto 0.35.p2 (branch
  `quadram2vk-followup`, the installed one) with two real conflicts, both in the warm-state limit.
- **Installed.** On this computer's Quadra 0.35.p2 checkout (`~/.local/share/QuadraHQ`) as local branch
  `quadram2vk-followup` (11 commits over `main` at `3c4f3cb`), with the host library and framework
  built in place by the launcher's own routine against fork `f44f1b2`. Its probe returns the RTX 3060 Ti with
  a passing test copy, and Quadra 0.35's C ABI host test (including its budget checks) passes on the GPU
  against the installed library. The renderer setting was left as it was (Vulkan/Zink). Return with `git checkout main`; the
  launcher then rebuilds Metal from `main`, or restore `build/metal` from the backup under
  `artifacts/install-backups-followup/`.
- **Memory pressure.** On the 8 GiB card, 256 MiB `Private` buffers were made until the device-local
  heap plus 1 GiB. With the runtime's memory priority and pageable device-local memory (now default):
  9,216 MiB made, no refusal, 0.31 s. With them withheld: the allocation past 7,168 MiB failed with the
  typed out-of-device-memory error. A bare Vulkan program gave the same two numbers. The runtime test is
  `memory_pressure` (ignored; it fills the device on purpose).
- **Recorded sessions.** Four real sessions on this computer (30 s to 1,177 s; 2,715 to 107,195
  drawable acquisitions) had `cancelled` 0 in all four, so releasing unused drawables through swapchain
  maintenance would fix a case that did not occur and was not built. Each had acquire timeouts of one
  second (1, 1, 1 and 5 in the order 1,177 s, 446 s, 39 s, 30 s): one at 2.1 s, one at 14 s and one at 2.1 s
  in the first three, and in the 30 s session five after the swapchain was remade for the 954x1032 to
  1912x1032 resize (5.1 s of the 30.4 s inside acquire; from about 160 acquisitions a second to about 9 a
  second for the next 6 s). The cause is not known. The presenter now logs,
  per distinct state, the images, those held by the game, those queued, the swapchain's age and the time
  since a present returned.
- **Same sessions, memory.** The 30 s session's live allocator bytes rose from 540 MiB to 7,672 MiB
  (8,432 MiB reserved, host and device memory together) within about two seconds, then fell to 886 MiB;
  what the game was doing, and how much of it lay in device-local memory, is not recorded (the diagnostics now split live bytes that way).
- **Shader upgrader and LLVM.** The authored legacy-AIR tests pass on LLVM 18.1.8, 19.1.7 (Debian
  packages unpacked in a scratch directory) and 23.1, and the installed client's shader pack (2,275
  libraries) upgrades and loads with no failure on each. A Debian-style `libLLVM.so.1` in
  `llvm-N/lib` had been read as version 1 and discarded; it now takes N from its directory. LLVM 15 to
  17 were not run.
- **Not changed, with reasons.** Automatic GPU choice stays in loader order (a user's `DRI_PRIME` or
  Mesa device-select order is that order). The RTX 3060 Ti is not added to the runtime's measured
  identities: that table holds GPUs and driver releases its author captured and ran the lanes on, as an
  exact match, and the 3060 Ti at 615.78.08 is not among them. The cost of being unmeasured is that the
  runtime withholds eight driver answers measured elsewhere (for example `RowCopiesDispatchFaster` and
  `SemaphoreOperationsStallSubmits`) and assumes `MultisampledStencilColourCopies`; its effect on frame
  time was not measured.
- **Checks.** The fork's full gate at `8624a3b`: 1,328 validation tests passed (4 skipped), 1,006 runtime
  and devtools tests passed (2 skipped), and format, clippy in every configuration, rustdoc, doctests and
  the API reference passed; the memory split after it has a unit test, a clean clippy and the runtime
  source record. On the RTX under the Khronos validation layer: 17 runtime GPU tests with no validation
  errors, the C ABI, X11 presentation, preload and host suite (30 host tests, 2 of which need a device),
  and the native Wayland test (1,192 frames presented, no acquire timeout, 3 mailbox swapchains). Launcher
  tests 261 with one failure (`session_housekeeping_test` trace file) that also fails on the previous
  commit here because this computer's runtime is CiderDarling; two settings UI regressions under Xvfb.
  Not run: the benchmark comparison, gameplay, any GPU but this one, LLVM 15 to 17.

## Revisions and outputs

| Component | Revision |
| --- | --- |
| Fork `development` | `8406bd26f51ff267775280024296c707deff1686` |
| Upstream base | `a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b` |
| Original Quadra runtime pin | `720ef1ce9ae2a2068999a8675717418b9416283e` |
| Companion Quadra `development` | `4430bd2ca9db2beb1d9f7059fc31a78c83b86fbc` |
| Quadra 0.34 release source | `df78a59ff76333e6d244ba827abd5a071700eed7` |
| Quadra source base | `94de4c6dec5e466d98ae9121cf7442f835dbf09e` |
| Runtime cache version | `1499` |

The repositories are `metal2vulkan/` and `integration/quadra/`. The original MacOBlox checkout
is unchanged. Bridge `4430bd2` (Quadra 0.34) is installed in the user's Quadra installation, with a
verified backup and isolated upgrade/rollback check. Matching
binaries are in `build/metal/`; the guest
framework was cross-built against CiderDarling `b0fcb070`. Both the framework and host require
ABI 2. The release host has debug sections removed while retaining symbols.

The [Quadra bridge patch](artifacts/quadra-bridge.patch) passed `git apply --check` against the
original checkout. The [fork patch](artifacts/metal2vulkan.patch) passed the same check against
the unmodified upstream base. Source archives and the commit-preserving mbox are in
`artifacts/`. The source archives can be used offline without Git history. Component commits
are published in CiderHQ-org/QuadraM2VK; the workspace pins both with submodules.

## Shader preload and stability qualification

The launcher now prepares libraries from the installed shader pack before learned pipeline
replay, then warms bounded local textures, fonts and models in the OS file cache. All stages
share one helper and expose progress and Skip. Pack ranges are validated, identical Metal
containers are deduplicated, and each library is released after preparation. These calls use
the existing library loader; they do not precompile all unseen pipeline/state combinations.

On this machine the previous `146ad14` installed-client preparation completed 2,275 library loads with zero
failures, 359 learned pipeline descriptions, and 2,756 local resource files/87,901,418 bytes.
The first observed pass using the existing cache took 19.905 seconds; the repeat took 1.034
seconds. The maximum child peak RSS across the two passes was 495,833,088 bytes. These are
prelaunch call/file-cache observations, not a shader-semantic census or game FPS measurements.
Only numeric summaries are retained in [the local preparation review](artifacts/launcher-preload-local-review.json).
That installed launcher completed the same preparation in 1.211 seconds with zero library
failures and the expected library → pipeline → resource progress order. Its child peak RSS
was 448,425,984 bytes. This repeat used the existing local caches.

Library preparation is limited to 4,096 unique pack ranges and 128 MiB of input. Local resource
warming has a 128 MiB/three-second ceiling with a smaller memory-based budget. The launcher
checks helper RSS and host/cgroup headroom every 250 ms, stops stalled progress after 20 seconds,
and keeps the 120-second overall timeout. A skipped or memory-limited helper ends before play.
These are sampled operational bounds; host RSS does not account for all driver/VRAM memory.

The current [RTX qualification](artifacts/hardware-20261022T050006Z-25608/review.json) passed
bridge rendering/compute/transfer/error checks, 98-state same-device reuse with zero additional
creation pipelines, the new first-launch preparation stages, all 56 focused launcher tests, four runtime
GPU regressions, both exact BC regressions, and native Wayland visible/covered/restore/resize.
Khronos validation was active with no VUID/validation errors in the reviewed correctness logs.
All 21 benchmark processes also completed and checked output; this report does not interpret
those small workloads as gameplay gains. The actual GTK stage dialogs were previewed in an
isolated Xvfb session and their Skip/callback cleanup was exercised. The Quadra 0.34
renderer UI check and the full suite of 249 Python tests also passed, alongside native
CPU/shim regressions. The rebuilt host uses the pinned public Git dependency.

The current presentation candidate also changes automatic vsync presentation from FIFO to
mailbox where the surface supports it, requests the driver's minimum image count plus one
within its maximum, and releases the guest layer lock before acquiring an image. FIFO is
retained explicitly and as the no-mailbox fallback. A bounded presenter queue and chain
retirement after failed presents protect against blocked/completed acquisition lifetimes.
Surface alpha selection follows advertised modes.

The user-run native Wayland test at bridge `0b73dd0` passed visible, covered, restored and
resized phases: 1,191 acquired/presented frames, no acquisition timeouts, no acquisitions
over 100 ms, and no present errors or queue rejections. Two mailbox swapchains used three
images each. Khronos validation was active with no VUID/validation errors in the saved log.
The old FIFO baseline also passed this authored test, so it is qualification rather than a
reproducer proving the old behavior. See [the scoped review](artifacts/presentation/review.json).
Separately, the user confirmed both persistent Roblox slowdowns were gone after installing
`0b73dd0`. This is a user gameplay report, not a controlled frame-time benchmark.

Two user-run numeric traces showed repeated one-second acquisitions, including steady
intervals with no new pipelines and very little GPU work. In the later 122.7-second trace,
64.1 seconds accumulated in acquisition calls while render-pipeline creation accumulated
0.292 seconds. Counters are sums across threads/phases, not a partition of elapsed time.
The new counters distinguish acquisition timeouts, queue delay and actual present-call
outcomes. This supports investigating presentation backpressure; it does not prove the
candidate fixes every reported frame-time drop.

The subsequent gameplay trace on `0b73dd0` contains a one-second interval with 15 render
pipeline calls accumulating 448.1 ms, plus other bursts of 156–319 ms. Acquisition time in
those intervals is below one millisecond. These are summed calls across threads, not single
frame times. The user reports mostly first-visit area slowdowns. The session recorded 189
pipeline recipes while retaining only 64 warm states. These observations support improving
warm retention; they do not establish that every area drop is shader compilation. New buffer
and texture creation/upload timings in this candidate help separate measured upload work.

The current release's [preload results](artifacts/preload/results.json) verify a cold authored
render/compute session, helper replay, and retention of both pipeline states on the next
device. Creating the application's matching states consumes the warm holds with **zero
additional creation pipelines**. Exact magenta render pixels and all 65 compute invocations
are checked. New state variants and unseen shaders may still require gameplay compilation;
no frame-time improvement from the retention increase has been measured in a game.

A new 98-state authored scene exceeds the previous 64-state cap. On the prior host it retained
64 states and required 33 additional creation pipelines. On this candidate it retained all 98
and required zero additional creation pipelines. Both checked the authored render/compute
output; the new check also verifies exact texture upload/readback bytes. These are software
Vulkan reuse checks, not hardware timing or game FPS measurements.

The same check verifies allocation release over repeated cycles, device isolation, shared
texture views, resource retention through GPU work, memory-only fallback for a blocked cache
path, corrupt-recipe rejection and isolation between client versions. Upgrade-cache tests
verify concurrent identical requests upgrade once and a corrupted entry is rebuilt.
The 56 launcher tests include staged progress, pack bounds/deduplication, local resource limits,
memory pressure, skip/timeout descendant cleanup, settings and host/guest environment forwarding.

Preloading is enabled by default for Metal. The helper is limited to 4096 recipes and 120
seconds; the game retains up to 256 warm states with a 128 MiB source-weight limit. The count
uses the smaller of one quarter of available host/cgroup memory and the bytes left after
768 MiB of headroom, with an estimated 8 MiB per state. Unknown availability keeps the prior
64-state limit. Driver allocations are not directly measured, so this is not an exact byte cap. It
checks five seconds and low host memory between creations; a driver call can take longer.
Warmup does not introduce background compilation workers during play. Learned shaders stay
local and are excluded from packages and diagnostic exports. Diagnostics are off by default;
timing percentiles use the last 512 samples. Live allocator bytes exclude Vulkan driver
allocations and externally wrapped images.

## Previous build: software Vulkan release comparison

Measured on a Ryzen 7 5800X, Linux `7.2.9-2-cachyos`, lavapipe/llvmpipe with LLVM 23.1.1,
`LP_NUM_THREADS=1`. All three host libraries used Rust 1.98.1 release builds, optimization
level 3 and the same authored C harness compiled with clang 23.1.1 `-O3`. Vulkan validation
was disabled for timing. Builds and test suites had finished before these measurements.

The two reference builds use the original Quadra host source with only dependency paths
changed: the old pin and updated upstream. The fork column includes the runtime changes
and modified ABI 2 bridge. These measurements do not isolate each individual patch.

Each variant had one discarded process warmup, then seven measured processes in alternating
variant order. Each process discarded five draw frames and two uploads per compressed format.
Reported timings are medians of process medians; all processes checked rendered pixels and
compressed readback bytes. The complete [JSON observations](artifacts/benchmark/results.json)
and [run log](artifacts/benchmark.log) retain the individual results.

| Authored workload | Original pin | Updated upstream | Fork + bridge |
| --- | ---: | ---: | ---: |
| Encode 500 draws with the same buffer rebound, ms | 4.910 | 1.296 | 1.202 |
| Those draws through submit and readback, ms | 5.966 | 2.177 | 2.098 |
| 1024² BC1 create/upload/readback/release, ms | 3.594 | 3.353 | 0.116 |
| 1024² BC5 create/upload/readback/release, ms | 4.783 | 4.177 | 0.172 |
| Whole benchmark process wall time, seconds | 0.326 | 0.213 | 0.103 |
| Maximum process peak RSS across seven runs, MiB | 129.3 | 127.4 | 106.5 |

Updating the upstream base accounts for most of the draw improvement. Against that updated
base, fork + bridge reduced median draw encoding time by **7.25%**, and the median through
readback by **3.63%**. Encoding process medians ranged from 1.277–1.554 ms upstream and
1.191–1.338 ms for the fork, so the ranges overlap. This is a modest measured CPU improvement.

The BC1 and BC5 upload workloads took **28.95×** and **24.33×** less time respectively than
updated upstream. Their default native policy skips decoded twins and update work; tests
assert that those twins are absent and decode counters do not increase. The exact policy
remains tested. Native BC results can differ from Metal; native filtering/texel accuracy
was not compared with a Metal GPU here.

Maximum process peak RSS fell 16.41% against updated upstream in this bounded workload.
RSS is host process memory, including loaded libraries and runtime/driver allocations;
it is not a GPU memory budget or a gameplay soak result. Wall/CPU time and peak RSS use a
small C `wait4` supervisor (`scripts/measure_process.c`). GPU timestamps are not used.

These are software Vulkan microbenchmarks: a 16² procedural draw target and constant-pattern
compressed texture transfers. They establish neither game FPS nor hardware GPU speedups,
display latency, first useful game frame, or tail latency during real shader compilation.

## RTX 3060 Ti hardware run

The user ran `scripts/test-hardware.sh --wsi` from the normal desktop terminal. The completed
run is [hardware-20261021T224406Z-199088](artifacts/hardware-20261021T224406Z-199088/hardware-tests.status),
which reports **PASS**. The reviewed logs identify an NVIDIA GeForce RTX 3060 Ti using
`libGLX_nvidia.so.615.78.08`. The hardware runner selected the NVIDIA ICD explicitly.

The following passed with the Khronos validation layer recorded as active and **zero VUID
or validation errors** in the saved correctness logs:

- 1× and 4× MSAA clear/resolve with every readback pixel checked.
- Guard-page texture transfers, exactly 65 compute invocations, dynamic threadgroup memory,
  fences/barriers, bulk binding ranges and persistent command-buffer failures.
- X11/XWayland presentation, 24 present/cancel cycles, two resizes, and surface/image lifetime.
- All four runtime GPU regressions and both exact BC decode/write-update regressions.
- All three host Rust tests, including legacy AIR scatter readback and the handle-size guard.

The exact BC test reports zero differing channels in each of ten BC1–BC5 linear/sRGB/signed
format cases, against its existing Metal reference, and zero twin-update cases wrong. This
qualifies the tested exact-policy cases; the native default's texel accuracy remains a
separate question. No new execution on a Metal GPU was performed.

The hardware [benchmark JSON](artifacts/hardware-20261021T224406Z-199088/benchmark/results.json)
contains 21 measured processes: seven per variant, alternating order after a discarded
warmup. All checked output bytes. Validation was disabled for timing. The libraries and
authored workload are the same as the software comparison; these are host-observed timings
on the real GPU, not GPU timestamp measurements or game FPS.

| Authored workload | Original pin | Updated upstream | Fork + bridge |
| --- | ---: | ---: | ---: |
| Encode 500 draws with the same buffer rebound, ms | 1.840 | 1.265 | 1.231 |
| Those draws through submit and readback, ms | 1.951 | 1.360 | 1.317 |
| 1024² BC1 create/upload/readback/release, ms | 3.621 | 3.394 | 0.239 |
| 1024² BC5 create/upload/readback/release, ms | 4.912 | 4.866 | 0.367 |
| Whole benchmark process wall time, seconds | 0.761 | 0.765 | 0.643 |
| Maximum process peak RSS across seven runs, MiB | 425.7 | 427.0 | 416.6 |

Against updated upstream, draw encoding took **2.67% less host time** and the draw/submit/
readback workload took **3.16% less wall time**. Encoding process medians ranged from
1.253–1.299 ms upstream and 1.215–1.250 ms for the fork. Against Quadra's original runtime
pin, median encoding time fell **33.08%**; most of that change comes from updating upstream.

The BC1 and BC5 transfer workloads were **14.18×** and **13.24×** faster than updated upstream.
The native policy's removal of decoded twins and their updates remains the substantial
gain in this workload. Maximum process RSS fell **2.43%** (10.38 MiB); this measures host
process memory, not VRAM. The benchmark process finished **15.94%** sooner overall.

The run covers small authored workloads and a short X11 lifecycle test. It does not qualify
actual Roblox gameplay, native Wayland, guest completion-worker execution, long sessions,
other GPUs, or end-user frame-time improvements. The agent's sandbox still lacks direct
GPU/display access; these results were obtained from the user's host run and reviewed here.

## Implemented behavior

- Supported native BC1–BC5 decoding is the default; `BlockDecodePolicy::Exact` and
  `DevicePolicy::exact_answers()` retain the fitted twin path. Unsupported compressed
  formats keep their prior emulation. The fast-path ledger and API docs describe the choice.
  The bridge's reserved C device options do not yet expose the exact switch.
- Equal buffer, texture and sampler bindings preserve the table stamp after validation.
  Changes to offsets, dynamic strides, sampler clamp bits or inline byte contents still
  invalidate it. A signed-zero sampler clamp change is retained as a change.
- `Texture::cpu_copy_byte_length` shares the runtime copy planner and rejects overflow before
  a host slice is created. The bridge uses it for upload and readback, including padding,
  compressed blocks and the exact last row; zero-length operations accept null pointers.
- Wrapped images can retain an external owner through views and GPU work. Swapchain owners
  hold their Vulkan objects and surface, including cleanup after partial construction.
  Acquire and present do not wait while holding the same surface state mutex. Unused
  drawables trigger conservative chain retirement on the next acquisition.
- Partial threadgroups use `dispatch_threads`; dynamic threadgroup memory, fences,
  buffer/texture residency and explicit barriers are forwarded. Render bulk buffer,
  texture and sampler ranges cross the boundary once per range. Guest compute inline
  allocations are reused and still copy the caller's bytes immediately.
- The first runtime encoder error persists on the command buffer. A refused commit also
  makes status and both waits return an error, including a commit with an open encoder.
  Guest scheduling observes the host transition and idle guest completion workers retire.
- Encoder storage is boxed separately. The release handle-size check reports **96 bytes**
  per ordinary host handle; a large encoder no longer pads every resource allocation.
- Validation child waits use bounded polling without a process-wide signal handler. Worker
  descendant cleanup uses bounded stdout pipes in its tests, preserving the ready/EOF proof.

## Checks performed

Counts in these rows overlap; they must not be added as unique test coverage.

| Check | Observed result |
| --- | --- |
| Four core crate suites after the initial runtime changes | [5,002 passed, 22 skipped](artifacts/core-tests.log) |
| Current runtime and devtools suites | [1,004 passed, 2 skipped](artifacts/runtime-final-tests.log) |
| Full validation suite at the final fork revision | [1,328 passed, 4 skipped](artifacts/validation-tests.log) |
| New runtime device regressions | [4 passed](artifacts/cpu-regressions.log): binding stamps, checked footprints, native/exact BC storage and retained image ownership |
| Existing exact BC execution regressions | [2 passed](artifacts/exact-bc-tests.log): decoded texels and twin updates after writes |
| Bridge C ABI | [Passed](artifacts/abi-regressions.log): guarded transfer spans, exact 65-thread output, dynamic memory, synchronization, bulk ranges and persistent failures |
| Clear/MSAA/readback through C ABI | [1× and 4× passed](artifacts/host-smoke.log), every RGBA pixel checked |
| Current release host Rust tests | [11 passed](artifacts/host-rust-tests.log), including presentation policies/limits, AIR cache recovery, memory/residency limits, scatter readback and handle-size guard |
| Shader preload, live-memory and cache regressions | [Passed](artifacts/preload/results.json), exact authored GPU output and same-device pipeline reuse |
| Launcher preload/settings/housekeeping | [56 passed](artifacts/hardware-20261022T034304Z-568993/bridge/preload/launcher-tests.log), including memory pressure and exited-leader descendant cleanup |
| Format and clippy | Passed for workspace/all targets/all features, translator with no features and each individual feature, CPU runtime and bridge |
| Rustdoc and API reference | Passed with warnings denied |
| Doctest command | Passed; no doctest examples were present |
| Darwin framework and Linux host | Both built successfully; current retention candidate guest execution not tested |
| Current RTX/native Wayland qualification | [Passed](artifacts/hardware-20261022T050006Z-25608/review.json), validation enabled; gameplay/long sessions remain separate |
| Numeric trace analysis | [4 passed](artifacts/trace-analysis-tests.log), including cumulative deltas, reset boundaries and a partially written final line |

CPU execution and bridge logs record the loader inserting `VK_LAYER_KHRONOS_validation`;
successful-test output was retained and checked for VUID/validation errors. No such errors
were found in these runs. Tests use authored inputs; no private shader census or new Metal
hardware comparison was run. Device tests outside the listed filters were not run here.

The release child-wait regressions passed in 9.06 ms wall time with 3.17 MiB peak RSS.
The [bounded tool batch](artifacts/wait-benchmark.log) also exercises the actual wait helper
with `llvm-as`/`llvm-dis` across eight size-spaced authored public IR fixtures. The largest
and slowest row was `hfb_filter_bank.air.ll`, 485,416 input bytes: assemble plus disassemble
completed in **22.45 ms**. The whole batch took **110.09 ms** wall time and reported
**33.61 MiB** peak RSS via `wait4` (largest child high-water mark, not summed simultaneous
process RSS). These are tool wait measurements, not an end-to-end translation census.
Translation ceilings remain 20 seconds and 500 MiB; they were not raised.

## Remaining work from the audit

1. Qualify gameplay on the documented RTX/Wayland setup and then AMD and Intel. Extend X11
   testing beyond the completed authored lifecycle check. Exercise
   guest handlers calling waits, queue retirement/restart, thread-creation failure, resize,
   minimize/restore, unused drawables and teardown over longer sessions.
2. Audit capability answers and function-constant behavior across the bridge. Device family
   queries reach the host; heap residency and unknown-selector return ABI behavior remain open.
3. Measure cold/warm frame times, memory and cache behavior on hardware and during long
   sessions. Then qualify AMD, Intel and additional NVIDIA models as requested.
4. Implement legacy AIR/container normalization in the native front end. The existing libLLVM
   text upgrader remains; removing it requires broader old/new differential qualification.
5. Extend WSI qualification to native Wayland and longer runs. The conservative fallback still uses
   queue-idle waits during retirement and CPU acquire fences. Native Wayland targets retain
   their existing caller-lifetime contract. VR remains outside this Metal bridge.

Published sources are at https://github.com/CiderHQ-org/QuadraM2VK. The workspace's distro
workflow exercises Debian, Fedora, Arch and Ubuntu with software Vulkan; physical AMD/Intel/
additional NVIDIA testing remains separate. The matching release also includes offline
source archives, patches and binaries.

Quadra 0.34 qualification: the 249-test Python suite and renderer-settings UI passed.
Its native source inputs match the preceding rebuilt candidate. Distribution CI at
`c814bb0` passed all five containers with the same native and preload implementation;
0.34 changes the release metadata and launcher version.
