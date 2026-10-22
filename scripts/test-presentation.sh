#!/usr/bin/env bash
# GPL-3.0-only. Native Wayland WSI checks with an authored opaque covering surface.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
out=${QM_PRESENT_OUTPUT:-$root/artifacts/presentation}
library=${QM_HOST_LIBRARY:-$root/build/metal/libquadra_metal_host.so}
mkdir -p "$out"
out=$(cd -- "$out" && pwd -P)
protocol=${WAYLAND_PROTOCOLS_DIR:-/usr/share/wayland-protocols}/stable/xdg-shell/xdg-shell.xml
wayland-scanner client-header "$protocol" "$out/xdg-shell-client-protocol.h"
wayland-scanner private-code "$protocol" "$out/xdg-shell-protocol.c"
read -r -a flags <<< "$(pkg-config --cflags --libs wayland-client)"
clang -O2 -Wall -Wextra -Werror -I"$out" "$root/integration/quadra/tests/metal_present_test.c" \
  "$out/xdg-shell-protocol.c" "${flags[@]}" -ldl -lpthread -o "$out/metal-present-test"
if [[ ${1:-} == --compile-only ]]; then
  printf 'PASS: native Wayland test compiled (%s)\n' "$out/metal-present-test"
  exit 0
fi
if [[ -z ${WAYLAND_DISPLAY:-} ]]; then
  printf '%s\n' 'Run from your Wayland desktop terminal with WAYLAND_DISPLAY set.' >&2
  exit 2
fi
printf 'RUNNING\n' >"$out/presentation-tests.status"
trap 'result=$?; if (( result != 0 )); then printf "FAIL (%s)\n" "$result" >"$out/presentation-tests.status"; fi' EXIT
export VK_LOADER_LAYERS_DISABLE='~implicit~'
if [[ -d $root/.tools/validation/usr/share/vulkan/explicit_layer.d ]]; then
  export VK_LAYER_PATH="$root/.tools/validation/usr/share/vulkan/explicit_layer.d"
fi
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation VK_LOADER_DEBUG=layer
export MACOBLOX_METAL_DIAGNOSTICS=1 MACOBLOX_METAL_GPU_TIMESTAMPS=0
export MACOBLOX_METAL_SHADER_CACHE="$out/cache"
export MACOBLOX_METAL_PRELOAD=0 MACOBLOX_METAL_PRELOAD_IN_PROCESS=0
# Empty selects the bridge's automatic tear-free mode. The optional FIFO baseline keeps its
# observed stalls in a separate result; its failure must not become a current-build pass.
if [[ -n ${QM_PRESENT_BASELINE:-} ]]; then
  MACOBLOX_METAL_PRESENT_MODE=fifo timeout 40s "$out/metal-present-test" "$QM_PRESENT_BASELINE" \
    >"$out/baseline.json" 2>"$out/baseline.log" || baseline_status=$?
  printf 'Baseline exit: %s\n' "${baseline_status:-0}"
fi
MACOBLOX_METAL_PRESENT_MODE= timeout 40s "$out/metal-present-test" "$library" \
  >"$out/current.json" 2>"$out/current.log"
if ! rg -q 'Insert instance layer.*VK_LAYER_KHRONOS_validation' "$out/current.log" || \
   rg -n 'VUID-|Validation Error' "$out/current.log"; then
  printf '%s\n' 'Native Wayland validation evidence failed.' >&2
  exit 1
fi
python3 - "$out/current.json" <<'PY'
import json,sys
from pathlib import Path
result=json.loads(Path(sys.argv[1]).read_text())
assert result['status']=='PASS',result
stats=result['diagnostics']['presentation']
assert stats['rebuilds']>=2 and stats['last_image_count']>0, stats
assert not stats['present_errors'] and not stats['queue_rejections'], stats
assert stats['presented']>=sum(row['frames'] for row in result['phases'].values())-16, stats
print('PASS: native Wayland visible/covered/restored/resize, no acquire timeout or >100 ms acquire')
print('Swapchain modes: mailbox=%d, FIFO=%d, immediate=%d' % (stats['mailbox_chains'],stats['fifo_chains'],stats['immediate_chains']))
PY
printf 'PASS\n' >"$out/presentation-tests.status"
