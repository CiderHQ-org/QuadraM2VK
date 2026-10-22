#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
mode=${1:---offscreen}
if [[ $mode != --offscreen && $mode != --wsi ]]; then
  printf 'usage: %s [--offscreen|--wsi]\n' "$0" >&2
  exit 2
fi
tests="$root/integration/quadra/tests"
out=${QM_TEST_OUTPUT:-$root/artifacts}
library=${QM_HOST_LIBRARY:-$root/build/metal/libquadra_metal_host.so}
mkdir -p "$out"
printf 'RUNNING\n' >"$out/bridge-tests.status"
clang -O2 -Wall -Wextra -Werror "$tests/metal_host_test.c" -ldl -o "$out/metal-host-test"
clang -O2 -Wall -Wextra -Werror "$tests/metal_fork_test.c" -ldl -lpthread -lX11 -o "$out/metal-fork-test"
python3 "$tests/assemble_metal_test.py" "$tests/metal_dispatch.ll" "$out/metal_dispatch.metallib"
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}
export LP_NUM_THREADS=${LP_NUM_THREADS:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$root/.cache/qualification}
export MACOBLOX_METAL_SHADER_CACHE=${MACOBLOX_METAL_SHADER_CACHE:-$root/.cache/qualification/air}
export VK_LOADER_LAYERS_DISABLE='~implicit~'
if [[ -d $root/.tools/validation/usr/share/vulkan/explicit_layer.d ]]; then
  export VK_LAYER_PATH="$root/.tools/validation/usr/share/vulkan/explicit_layer.d"
fi
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=layer
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target/integration}
if [[ -d $root/.cargo-home/registry ]]; then
  export CARGO_HOME=${CARGO_HOME:-$root/.cargo-home}
  export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
fi
timeout 120s "$out/metal-host-test" "$library" >"$out/host-smoke.log" 2>&1
args=()
if [[ $mode == --offscreen ]]; then args+=(--offscreen); fi
timeout 120s "$out/metal-fork-test" "$library" "$out/metal_dispatch.metallib" "${args[@]}" >"$out/abi-regressions.log" 2>&1
manifest="$root/integration/quadra/prototype/metal-vulkan/host/Cargo.toml"
rust_toolchain=${RUSTUP_TOOLCHAIN:-stable}
cargo "+$rust_toolchain" test --locked --release --manifest-path "$manifest" --lib -- --include-ignored --nocapture >"$out/host-rust-tests.log" 2>&1
cargo "+$rust_toolchain" clippy --locked --manifest-path "$manifest" --all-targets -- -D warnings >"$out/clippy-bridge.log" 2>&1
if rg -n 'Validation Error|VUID-|ERROR.*VK_LAYER|Failed to find.*VK_LAYER' "$out/host-smoke.log" "$out/abi-regressions.log" "$out/host-rust-tests.log"; then
  printf '%s\n' 'Validation reported errors; inspect the logs.' >&2
  exit 1
fi
for log in host-smoke abi-regressions host-rust-tests; do
  if ! rg -q 'Insert instance layer.*VK_LAYER_KHRONOS_validation' "$out/$log.log"; then
    printf 'No active validation layer recorded in %s.log\n' "$log" >&2
    exit 1
  fi
done
rg 'PASS:|SKIP:|Device:|host_handle_bytes|test result:' "$out/host-smoke.log" "$out/abi-regressions.log" "$out/host-rust-tests.log"
QM_PRELOAD_OUTPUT="$out/preload" bash "$root/scripts/test-preload.sh"
printf 'PASS\n' >"$out/bridge-tests.status"
