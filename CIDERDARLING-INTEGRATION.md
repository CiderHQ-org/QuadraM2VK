# Making Quadra M2VK part of CiderDarling

Prepared 2026-10-23 from a read of CiderDarling `b0fcb07` (`CiderHQ-DarlingFork`, `main`) and this
workspace. Statements marked **checked** were read in the code by the author of this note; the rest
come from a longer read by a helper and are marked **read**. Nothing here was built or run inside
CiderDarling.

## Where CiderDarling stands

- **Checked.** Metal is off. `CMakeLists.txt` sets `ENABLE_METAL` to `OFF`, and `tools/cider/package.py`
  and `tools/cider/isolate.py` pass `-DENABLE_METAL=OFF` whatever the default. Without it
  `MTLCreateSystemDefaultDevice()` returns `nil` and `MTLCopyAllDevices()` an empty array
  (`src/external/metal/src/Metal/MTLDevice.mm`). A guest app therefore sees no Metal device and falls back
  to OpenGL.
- **Read.** The optional Indium backend is the only Metal there. `MTKView` and the other classes are
  `MTL_UNSUPPORTED_CLASS` stubs that abort when called, `CAMetalLayer` raises on every property, and
  `MetalPerformanceShaders` is empty. Patch 17 only fixes a retain in that backend's command queue.

## How Quadra supplies Metal today

- **Checked.** Quadra's shim (`gpu_info.c`) interposes `MTLCreateSystemDefaultDevice` and
  `MTLCopyAllDevices` to return nothing unless `MACOBLOX_METAL=1`. Its comment records why: Roblox starts its
  Metal renderer whenever a device exists, and a Darling that offers one crashes it. So any integration must
  keep "no device" as the default and offer M2VK only on request.
- **Read.** The launcher builds `Metal.framework` against the Darling sysroot (`prototype/metal-vulkan/build.sh`),
  sets `DYLD_FRAMEWORK_PATH` and `MACOBLOX_METAL_HOST_LIBRARY` for the guest, and the framework opens the Linux
  host library through `_elfcalls`. The host library runs inside the guest process on host glibc.
  Quadra also hooks `CARenderer` for layers without contents, supplies the Wayland subsurface provider and
  raises worker-thread stacks. Almost every `MACOBLOX_*` name is Quadra's.

## What to improve before it can live in CiderDarling

1. **Contract.** The guest/host ABI (`quadra_metal.h`, version 2) is sound: the host returns null for another
   version and the guest rejects a smaller table. What is missing is a published way to find the host library
   without a Quadra environment variable, and a version for the pair that Darling's runtime feed can check.
   The feed carries only `minimum_cider`; `runtime.json` has no graphics field.
2. **Names.** `MACOBLOX_*` variables, the `[Quadra Metal]` log prefix, the `quadra_metal_host_*` exports and the
   `QMMetalLayer` class are Quadra's. Rename to `CIDER_M2VK_*` with the old names accepted for one release.
3. **Roblox behavior in the framework.** `builtins.shaders` emulation by source text, feature-family answers
   tuned to Roblox and the host-clock GPU timing are Roblox-shaped. Move them behind a compatibility profile
   so another Metal app gets Metal's answers.
4. **Selectors.** The framework's fallback for a selector it lacks logs once and returns zeros through the
   signature `@@:` (**checked**), so a method returning a float, a double or a large struct reads the wrong
   register. Generate the selector table from `framework-include/Metal`, and add a strict mode that stops on an
   unknown selector for developers.
5. **Layer hosting.** The framework finds the view for a layer through its swizzled `setLayer:` or a layer
   delegate (**checked**, `QMLayer.m`). A layer made by `makeBackingLayer` has neither (**read**: `NSView.m`
   assigns `_layer` directly), so such a view renders offscreen. `MTKView` needs this and its `DARLING_METAL_ENABLED`
   guard decoupled from Indium before it can work.
6. **Missing API.** `newDefaultLibrary`, URL and bundle libraries, function constants, the compute and blit
   pass descriptors, async creation, `fillBuffer`, heaps and events (the runtime has the last two) are absent
   or stubbed. Only precompiled metallibs work; no Xcode-built `default.metallib` has been tried.
7. **Presentation.** X11 is in the host library; native Wayland needs Quadra's provider, which CiderDarling
   lacks. Define a small presentation-target protocol to replace the `dlsym("macoblox_wayland_*")` lookups.
   Darling's own GL subwindow beside the Vulkan child window on X11 is unverified.
8. **Licenses.** CiderDarling is GPL-3.0, so the fork's `GPL-2.0-or-later OR LGPL-3.0-or-later` can be taken
   under its GPL option and the bridge is GPL-3.0-only. Do not merge the bridge into Darling's MPL-2.0 Metal
   files unless it is dual-licensed. The framework sources carry no SPDX headers.
9. **Build.** The host library needs Rust 1.98 and a git-pinned runtime (28 crates). Keep Rust optional in CMake
   with a prebuilt host library as Quadra does, and vendor the crates for Debian and RPM builds.
10. **Fallback.** A failed probe now falls back to OpenGL once in the launcher. A CiderDarling doctor should
    offer the same check (`ciderdarling-doctor --graphics --metal`), reusing the probe and self-test.

## Order

Each step can be reviewed and tested alone.

1. SPDX headers and a licence note for the framework and host.
2. Move the `CARenderer` nil-contents fix into `patches/cider` and test it with a CiderDarling fixture.
3. Settle whether worker threads need Quadra's 8 MiB stack floor, with a fixture.
4. Import the framework and host into CiderDarling; rename the variables with aliases; verify the exported
   symbols are unchanged.
5. Add a CMake option `CIDER_M2VK` (default off) and install the framework and host beside the stock one; add
   `graphics.m2vk {abi, files}` to `runtime.json`.
6. A `CIDER_METAL_BACKEND=off|m2vk` gate in the framework (off returns no device, replacing Quadra's
   interposer) and discovery of the host library by install prefix. Fixtures: off, lavapipe, wrong ABI (nil and
   a log line, no crash).
7. The presentation-target protocol; the X11 provider in tree; Quadra keeps the Wayland provider.
8. Layer hosting for `makeBackingLayer` and an `MTKView` that does not need Indium.
9. The API gaps above, with a selector table and a strict mode.
10. The Roblox compatibility profile, with the bridge tests passing with it on and off.
11. CI: a lavapipe job; `metal` in the smoke coverage; gate promotion on it.
12. `ciderdarling-doctor --graphics --metal`.
13. The launcher selects by `CIDER_METAL_BACKEND` when `runtime.json` advertises `graphics.m2vk`, drops its own
    framework build and `DYLD_FRAMEWORK_PATH`, and keeps the interposer for older runtimes.
14. An optional `darling-m2vk` distro package carrying a prebuilt host library.

## Not known

Whether Darling's worker stacks are large enough without Quadra's wrapper; the cold Rust build time in
CiderDarling's CI; how the X11 child window behaves beside Darling's GL subwindow; behavior with Xcode-built
metallibs; which CiderDarling revision Quadra 0.34 pins; the licences of the Rust dependencies.
