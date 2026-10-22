# Hardware qualification

The current `146ad14` bridge passed agent-run RTX 3060 Ti checks with NVIDIA 615.78.08:
[review and numeric evidence](artifacts/hardware-20261022T034304Z-568993/review.json).
This covers the authored host/launcher paths, native Wayland lifecycle, runtime GPU and
exact BC regressions. Actual game frame times and long sessions remain separate checks.
The older run below describes `c50214a` / `5cbc9fb`.
The runner now also checks shader replay, exact authored output, live memory release,
cache recovery and launcher cancellation. AMD, Intel and additional NVIDIA qualification
is the next phase after this release.

The intermediate `0b73dd0` bridge separately passed the user-run RTX 3060 Ti native Wayland
visible/covered/restored/resize test, and the user reported that the home and persistent
minimum-graphics slowdowns were gone. See [its scoped review](artifacts/presentation/review.json).
The current candidate retains up to 256 warm states and adds client-library preparation and
local resource warming in the launcher. The 98-state reuse check and new first-launch stages
passed on the RTX, alongside the existing preload and memory checks.

The user-run [RTX 3060 Ti qualification](artifacts/hardware-20261021T224406Z-199088/hardware-tests.status)
finished with **PASS**, using NVIDIA 615.78.08. The reviewed correctness logs show an active
Khronos validation layer and no VUID/validation errors. Bridge rendering, compute, guarded
transfers, X11 presentation/cancellation/resize, four runtime GPU regressions and both exact
BC regressions passed. All 21 measured benchmark processes completed and checked output.

The hardware results are separate from the prior software Vulkan results. See
[RESULTS.md](RESULTS.md) for the comparison, individual measurements and scope. Native
Wayland on the current retention candidate, guest completion-worker edge cases, area hitches
and long-session stability remain open.

Earlier sandbox access diagnostics remain in `artifacts/hardware-access/`; the current run
used direct authorized GPU/display access. The preflight was also checked
against lavapipe: it exits with status 2, so a CPU renderer cannot become a hardware pass.

## Run from the normal desktop terminal

The runner uses the existing build and writes a fresh result directory under this workspace.
It does not install the fork, change Quadra's renderer setting, or launch the game.

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  bash /home/aubree/Documents/QuadraM2VK/scripts/test-hardware.sh --wsi
```

`--wsi` uses the desktop's X11/XWayland `DISPLAY` for a small authored test window.
For GPU execution without a window, substitute `--offscreen`. Use `--wayland` for the native
Wayland lifecycle test. Guest framework and actual gameplay remain separate checks.

For a quick check of just this update's authored preload/retention and memory paths:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  bash /home/aubree/Documents/QuadraM2VK/scripts/test-preload.sh
```

Full retention of the 98-state scene requires enough available host/cgroup memory. The
result reports the selected limit and whether full retention was actually exercised; low
memory must preserve the smaller cap. Select the installed ICD for each actual AMD, Intel
or NVIDIA device. A device absent from the machine remains untested.

The run performs:

1. Vulkan device enumeration, requiring a discrete or integrated GPU from the explicitly
   selected ICD.
2. Validation-enabled host ABI clear/MSAA/readback, guarded texture transfers, exact partial
   dispatch, threadgroup memory, bulk bindings, synchronization, and error-status regressions.
   With `--wsi`, it also tests X11 presentation, unused drawable cancellation, resize and
   surface/image ownership.
3. Runtime GPU regressions under `m2v_gpu`, and exact BC decoded-texel/write-update tests.
   These use `target/hardware` independently of the CPU qualification build.
4. Seven alternating release comparison runs against the original pin, updated upstream
   and fork, with validation disabled for timing and output bytes checked.

Validation logs must show the Khronos layer active and contain no VUID/validation errors.
The runner stops on failure. A success is recorded only after all selected steps finish.
The existing software measurements remain at `artifacts/benchmark/results.json`; new GPU
measurements go to `artifacts/hardware-<timestamp>-<pid>/benchmark/results.json`.

The shell/benchmark helpers passed syntax checks. The new benchmark output option was
executed on software Vulkan in a separate directory and the original results were verified
unchanged. The RTX 3060 Ti host run described above completed successfully.
