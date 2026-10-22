#!/usr/bin/env bash
# GPL-3.0-only. Run in a host terminal with GPU/display access; never falls back to CPU.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---offscreen}
if [[ $mode != --offscreen && $mode != --wsi && $mode != --wayland ]]; then
  printf 'usage: %s [--offscreen|--wsi|--wayland]\n' "$0" >&2
  exit 2
fi
stamp=$(date -u +%Y%m%dT%H%M%SZ)-$$
out=${QM_HARDWARE_OUTPUT:-$root/artifacts/hardware-$stamp}
mkdir -p "$out"
out=$(cd -- "$out" && pwd -P)
printf 'RUNNING\n' >"$out/hardware-tests.status"
trap 'result=$?; if (( result != 0 )); then printf "FAIL (%s)\n" "$result" >"$out/hardware-tests.status"; printf "Stopped; diagnostics: %s\n" "$out" >&2; fi' EXIT
if [[ -z ${VK_DRIVER_FILES:-} ]]; then
  printf '%s\n' 'Set VK_DRIVER_FILES to your hardware Vulkan ICD JSON (for this host: /usr/share/vulkan/icd.d/nvidia_icd.json).' >&2
  exit 2
fi
clang -O2 -Wall -Wextra -Werror "$root/scripts/probe_gpu.c" -ldl -o "$out/probe-gpu"
if ! timeout 15s "$out/probe-gpu" >"$out/gpu.json" 2>"$out/gpu.stderr.log"; then
  cat "$out/gpu.stderr.log" >&2
  printf '%s\n' 'Hardware testing requires visible GPU device nodes and a hardware Vulkan driver.' >&2
  exit 2
fi
cat "$out/gpu.json"
if [[ $mode == --wsi && -z ${DISPLAY:-} ]]; then
  printf '%s\n' 'The current authored WSI test needs an X11/XWayland DISPLAY.' >&2
  exit 2
fi
if [[ $mode == --wayland && -z ${WAYLAND_DISPLAY:-} ]]; then
  printf '%s\n' 'The native Wayland presentation test needs WAYLAND_DISPLAY.' >&2
  exit 2
fi
export QM_TEST_OUTPUT="$out/bridge"
export XDG_CACHE_HOME="$out/cache"
export MACOBLOX_METAL_SHADER_CACHE="$out/cache/air"
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target/integration}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
export PATH="$root/.tools:$PATH"
if [[ -d $root/.cargo-home/registry ]]; then
  export CARGO_HOME=${CARGO_HOME:-$root/.cargo-home}
  export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
fi
printf 'Running bridge and presentation checks; logs: %s/bridge\n' "$out"
bridge_mode=$mode
if [[ $mode == --wayland ]]; then bridge_mode=--offscreen; fi
bash "$root/scripts/test-bridge.sh" "$bridge_mode" >"$out/bridge.log" 2>&1
cat "$out/bridge.log"
if [[ $mode == --wayland ]]; then
  printf 'Running native Wayland covered/uncovered/resize checks; logs: %s/presentation\n' "$out"
  QM_PRESENT_OUTPUT="$out/presentation" bash "$root/scripts/test-presentation.sh" >"$out/presentation.log" 2>&1
  cat "$out/presentation.log"
fi
export VK_LOADER_LAYERS_DISABLE='~implicit~'
if [[ -d $root/.tools/validation/usr/share/vulkan/explicit_layer.d ]]; then
  export VK_LAYER_PATH="$root/.tools/validation/usr/share/vulkan/explicit_layer.d"
fi
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=layer
cd "$root/metal2vulkan"
# Keep the runtime's GPU safety cfg separate from ordinary host builds and CPU qualification.
printf 'Building/running runtime GPU regressions; log: %s/runtime-gpu.log\n' "$out"
env RUSTFLAGS='--cfg m2v_gpu' CARGO_TARGET_DIR="$root/target/hardware" cargo +stable nextest run --offline -p metal2vulkan-runtime --all-features --success-output=final \
  -E 'test(binding_repeats) | test(cpu_copy_size) | test(native_block_decode) | test(wrapped_owner)' >"$out/runtime-gpu.log" 2>&1
printf 'Building/running exact BC tests; first compilation can take several minutes. Log: %s/exact-bc-gpu.log\n' "$out"
env RUSTFLAGS='--cfg m2v_gpu' CARGO_TARGET_DIR="$root/target/hardware" cargo +stable nextest run --offline -p metal2vulkan-validation --lib --all-features --success-output=final \
  -E 'test(runtime_lane::compressed_decode::bc1_to_bc5_texels_read_as_metal_s) | test(runtime_lane::compressed_decode::decoded_twins_follow_every_write)' >"$out/exact-bc-gpu.log" 2>&1
for name in runtime-gpu exact-bc-gpu; do
  if ! rg -q 'Insert instance layer.*VK_LAYER_KHRONOS_validation' "$out/$name.log" || rg -n 'Validation Error|VUID-' "$out/$name.log"; then
    printf 'Validation evidence failed in %s.log\n' "$name" >&2
    exit 1
  fi
done
unset VK_INSTANCE_LAYERS VK_LAYER_PATH VK_LOADER_DEBUG
cd "$root"
printf 'Running seven release benchmark rounds per variant; log: %s/benchmark.log\n' "$out"
python3 "$root/scripts/benchmark.py" --runs 7 --output "$out/benchmark" >"$out/benchmark.log" 2>&1
cat "$out/benchmark.log"
printf 'PASS\n' >"$out/hardware-tests.status"
printf 'Hardware results: %s\n' "$out"
