# Presentation and gameplay verification

The current candidate addresses repeated one-second drawable acquisitions observed in the
RTX 3060 Ti traces. Mailbox presentation, driver-aware swapchain sizing and releasing the
guest layer lock are implemented. The preceding `0b73dd0` build passed the user-run RTX 3060 Ti
native Wayland test; the user confirmed that the home and persistent minimum-graphics stalls
were gone. The user reports smaller first-visit drops after the 256-state retention update.
The current candidate also prepares client shader libraries and warms bounded local assets
in the launcher. It passed RTX authored preload and native Wayland checks; a game comparison
of the added stages remains open.

Close Quadra before updating the installed build:

```sh
python3 /home/aubree/Documents/QuadraM2VK/scripts/install.py
```

The installer prints a backup path and rollback command. It preserves existing preload and
diagnostics preferences, and selects the Metal renderer. On a repeated installation of the
same matching release it leaves settings alone.

The quick native Wayland test uses authored clear frames and an opaque covering subsurface.
It runs visible, covered, restored and resized phases, checking acquisition stalls, presentation
errors and active Vulkan validation. It does not open Roblox or modify its settings.

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  QM_PRESENT_BASELINE=/home/aubree/Documents/QuadraM2VK/artifacts/presentation-baseline/libquadra_metal_host.so \
  bash /home/aubree/Documents/QuadraM2VK/scripts/test-presentation.sh
```

The optional baseline records the previous build with FIFO separately. Its `SLOW` result is
not a current-build pass. Current results are `artifacts/presentation/current.json` and
`current.log`. All phases must keep producing frames without acquisition timeouts or waits
over 100 ms. Reported frame counts are submitted/completed authored frames, not proof of
display scanout timing. CPU/software drivers cannot qualify hardware performance.

For gameplay use the same window size and frame cap, with Metal selected:

1. Open Roblox home/game browser for 30 seconds, scroll and navigate normally.
2. Join [Write a Letter](https://www.roblox.com/games/8425637426/Write-a-Letter), wait for loading,
   and compare the same view for 30 seconds at maximum graphics and 30 seconds at minimum.
3. Raise graphics again, then repeat the minimum setting. Note whether the slowdown persists.
4. Exercise window resize, minimize/restore, leave/rejoin and quit/restart. Use a longer session
   to check memory, threads and stability after the short reproduction is resolved.
5. Repeat a warm launch. The first session can still learn new pipeline/state combinations.

For the remaining area hitches, keep maximum graphics and follow the same route through
Write a Letter twice. Compare the first visit with a return to already visited areas, then
quit and relaunch with shader preloading allowed to finish. Repeat that route. A learned
pipeline can be warmed next launch; genuinely unseen pipeline descriptions and streamed
game assets still require work when first encountered. Do not clear the local shader cache
between these comparisons.

Performance diagnostics add bounded numeric traces under Quadra's `logs/` directory, named
`metal-performance-*.jsonl`. They include acquisition/queue/present timings, compilation,
memory and cache counters. They contain no shader text. Encoder lifetime, GPU command spans,
and async present-call rate are different measurements; none alone is a gameplay FPS trace.
Repeat final frame-time comparisons with diagnostics disabled to measure production overhead.

The candidate additionally times buffer creation, creation with initial bytes, texture creation
and CPU region uploads. It does not time every asset-loading or encoded GPU-copy path. Analyze
intervals without exporting raw game logs:

```sh
python3 /home/aubree/Documents/QuadraM2VK/scripts/analyze-trace.py /path/to/metal-performance-YYYYMMDD-HHMMSS-PID.jsonl
```

The full hardware runner accepts `--wayland`, `--wsi` (X11/XWayland), or `--offscreen`:

```sh
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/nvidia_icd.json \
  bash /home/aubree/Documents/QuadraM2VK/scripts/test-hardware.sh --wayland
```

For AMD, Intel and other NVIDIA devices, select the installed ICD for that actual hardware.
Each run must record a physical device and its driver, correctness/validation results, native
presentation and gameplay separately. A missing device or unrun stage remains unqualified.
