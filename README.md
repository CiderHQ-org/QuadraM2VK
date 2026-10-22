# QuadraM2VK

[![Linux distributions](https://github.com/CiderHQ-org/QuadraM2VK/actions/workflows/distros.yml/badge.svg?branch=quadram2vk)](https://github.com/CiderHQ-org/QuadraM2VK/actions/workflows/distros.yml)

A local metal2vulkan fork and companion Quadra Metal bridge for performance and stability
work. The fork is on `development`, based on upstream `a2c440b`; the bridge was imported
from Quadra `94de4c6`. The original MacOBlox checkout is unchanged.

The published workspace is [CiderHQ-org/QuadraM2VK](https://github.com/CiderHQ-org/QuadraM2VK),
on branch `quadram2vk`. It pins `runtime-development` and `quadra-development` as component
submodules in the same repository. Clone their sources together:

```sh
git clone --recurse-submodules https://github.com/CiderHQ-org/QuadraM2VK.git
```

[Quadra 0.33](https://github.com/CiderHQ-org/Cider/releases/tag/v0.33) integrates this
fork as **Settings → Environment → Game → Renderer → Quadra M2VK (experimental)**.
Native packages include the matching bridge and preloading controls. Existing Metal
selections use this entry; OpenGL remains the default for new installations.

This experimental build includes shader preloading in the Quadra launcher, live memory
accounting, cache recovery and optional performance diagnostics. The current revision passed
authored RTX 3060 Ti and software Vulkan checks, including native Wayland lifecycle, learned
pipeline reuse and first-launch library/resource preparation. The user reports that persistent
home/minimum-graphics stalls are gone and first-visit drops decreased after increasing warm
retention. Gameplay verification of the additional prelaunch stages and long-session stability
remain open.
See [RESULTS.md](RESULTS.md) for measurements, test evidence and remaining work. The
original [plan](PLAN.md) and [deeper audit](PERFORMANCE-STABILITY-AUDIT.md) remain available.

The [experimental build package](https://github.com/CiderHQ-org/QuadraM2VK/releases/tag/experimental-20261022-preload)
contains matching binaries, source archives, patches, licenses and the qualification report.
The release also provides `SHA256SUMS` and `build-manifest.json`.

For the authorized real-GPU testing, see [HARDWARE-TESTING.md](HARDWARE-TESTING.md).
The current agent-run RTX checks used the now-accessible GPU and desktop. The
[distribution workflow](CI.md) covers Debian, Fedora, Arch and Ubuntu using software Vulkan.

## Included changes

- Client-scoped shader preloading learns successful pipelines locally and replays them on
  later launches, with progress and **Skip and play**. New shaders can still compile during
  play. Enable it under **Settings → Environment → Game → Shader preloading** (default on).
- First-launch library preparation normalizes installed client libraries through the existing
  loader. A separate bounded warmup reads bundled textures, fonts and models into the OS cache.
  Both have settings and stage progress; world assets still load in game.
- The helper stops on memory pressure, stalled progress, timeout or Skip. It releases each
  prepared library and kills its process group before starting the game, including descendants
  left by an exited helper. Extra preparation adds no gameplay compilation workers.
- Bounded same-device warm pipeline retention, CPU/cgroup-aware compile limits, checksummed
  AIR/preload caches, corrupt-entry recovery and memory-only fallback when storage fails.
  The 98-state authored regression requires 33 additional creation pipelines on the preceding
  build and zero on this candidate. This measures reuse, not game FPS.
- Per-device live allocation accounting, device-local heap working-set recommendations and
  optional bounded numeric performance traces. Diagnostics are off by default.
- Mailbox presentation with vsync when supported, driver-aware swapchain sizing and alpha
  selection, a bounded presenter queue, and acquisition without holding the guest layer lock.
  The user confirmed that the home-menu/minimum-graphics stalls were gone on `0b73dd0`.
  See [PRESENTATION-TESTING.md](PRESENTATION-TESTING.md) for the remaining area-hitch checks.
- Native BC1–BC5 decoding by default on supported devices, with the exact decoded-twin
  policy retained for Rust consumers. This removes the extra decoded image and its update
  work from the default path; native texel results may differ from Metal.
- Repeated identical buffer, texture and sampler bindings preserve their table stamp.
  Offset, stride, sampler clamp and inline-byte changes still invalidate the binding state.
- Checked texture-copy footprints shared with the runtime planner, including compressed
  blocks, padding, mip levels and overflow. The bridge validates before borrowing host bytes.
- External Vulkan image owners retained through texture views and GPU use. The presenter
  uses this ownership for swapchains, cleans up partial construction, and separates state
  locking from blocking acquire/present calls.
- Exact partial-threadgroup dispatch, dynamic threadgroup memory, fence/barrier/residency
  forwarding, render bulk bindings, and reused guest compute inline storage.
- Separate encoder storage so every ordinary host handle avoids the largest encoder's padding.
- Persistent command-buffer errors and rejected-commit status, actual scheduled waits,
  and guest completion workers that retire when idle.
- Validation child waits without a global signal handler and descendant-cleanup tests
  that use bounded pipes in restricted runners.

## Layout

| Path | Purpose |
| --- | --- |
| `metal2vulkan/` | Fork repository; all three product crates stay together |
| `integration/quadra/` | Isolated Quadra source, launcher preload controls, bridge and authored tests |
| `build/metal/` | Matching ABI 2 `Metal.framework` and Linux host library |
| `artifacts/` | Checks, benchmark observations, bridge patch and source archives |
| `scripts/` | Build, qualification, baseline preparation and benchmark helpers |

## Build and verify

Requires Rust 1.98+, clang/lld, LLVM tools, SPIR-V Tools, Vulkan, libX11, and a CiderDarling
sysroot for the Darwin framework. The framework is x86_64 Darwin; the host is Linux x86_64.
The companion pins the reviewed runtime Git revision in its Cargo manifest and lock file;
the separate runtime checkout in this workspace is used for core qualification.

```sh
export DARLING_SYSROOT=/path/to/ciderdarling/libexec/darling
bash scripts/build.sh all
bash scripts/test-bridge.sh --offscreen
bash scripts/check-core.sh core
bash scripts/check-core.sh cpu
```

Bridge qualification includes the authored preload, live-memory, cache recovery and launcher
regressions. To run just those checks, use `bash scripts/test-preload.sh`. See the
[preloading and diagnostics guide](integration/quadra/prototype/metal-vulkan/README.md#shader-preloading-and-diagnostics)
for limits and timing interpretation.

To inspect a numeric gameplay trace, run `python3 scripts/analyze-trace.py /path/to/metal-performance.jsonl`.
It reports intervals containing compilation or measured upload bursts. It handles a final
partially written line and counter resets; present-call rate is explicitly separate from FPS.

These checks use the prepared workspace Cargo cache offline. On a new machine, fetch the
dependencies first and install `cargo-nextest` and `VK_LAYER_KHRONOS_validation`. Bridge
and CPU checks default to lavapipe with one worker thread; set `VK_DRIVER_FILES` explicitly
to select another driver. For live X11 presentation, run `test-bridge.sh --wsi` with a usable
`DISPLAY`. Wayland and guest handler/queue lifetime need their own live qualification.

To repeat the three-way release comparison, check out the original runtime at `720ef1c`
and unmodified upstream at `a2c440b` into separate directories, then:

```sh
python3 scripts/prepare-baselines.py --old-runtime /path/to/720ef1c --upstream-runtime /path/to/a2c440b
python3 scripts/benchmark.py --runs 7
```

The harness checks output bytes, alternates variant order, warms each variant, disables
validation layers, records release wall time and peak process RSS, and writes JSON results.
It measures authored draw encoding and texture uploads on the selected ICD. It does not
measure game FPS, display latency or hardware GPU execution time.

## Integrate with Quadra

For an existing user installation, close Quadra and run `python3 scripts/install.py` from
this workspace. It checks the patch and binary hashes, backs up affected files and settings,
and enables Metal. Existing QuadraM2VK installations receive an incremental patch; unrelated
local changes are preserved. A rollback command is printed after installation. Use
`python3 scripts/install.py --check` to check compatibility first.

`artifacts/quadra-bridge.patch` applies to Quadra `94de4c6`. Review it before applying in your
own checkout. The matching fork must be at `vendor/metal2vulkan`; the offline source archive
is provided for this. Deploy the ABI 2 framework and host library together.

```sh
git -C /path/to/Quadra apply --check /path/to/QuadraM2VK/artifacts/quadra-bridge.patch
git -C /path/to/Quadra apply /path/to/QuadraM2VK/artifacts/quadra-bridge.patch
mkdir -p /path/to/Quadra/vendor/metal2vulkan
tar -xzf /path/to/QuadraM2VK/artifacts/metal2vulkan-source.tar.gz --strip-components=1 -C /path/to/Quadra/vendor/metal2vulkan
```

Matching build packages and source snapshots are published under the repository's
[releases](https://github.com/CiderHQ-org/QuadraM2VK/releases). Local archives, patches and
the commit-preserving `metal2vulkan-patches.mbox` also work offline; the mbox applies on top
of upstream `a2c440b` using `git am`.

## Licenses

The fork retains upstream's `GPL-2.0-or-later OR LGPL-3.0-or-later` choice and all notices.
The Quadra bridge and workspace helper scripts are GPL-3.0-only. The bridge patch carries
authored test source; no client shader packs, extracted client shaders or shader caches are
included in the source or build package.
