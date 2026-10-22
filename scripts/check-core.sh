#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
if [[ -d $root/.cargo-home/registry ]]; then
  export CARGO_HOME=${CARGO_HOME:-$root/.cargo-home}
fi
export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-4}
export PATH="$root/.tools:$PATH"
mkdir -p "$root/artifacts"
cd "$root/metal2vulkan"

run() {
  local name=$1
  shift
  printf 'Running %s\n' "$name"
  printf 'RUNNING\n' >"$root/artifacts/$name.status"
  if "$@" >"$root/artifacts/$name.log" 2>&1; then
    printf 'PASS\n' >"$root/artifacts/$name.status"
  else
    local result=$?
    printf 'FAIL (%s)\n' "$result" >"$root/artifacts/$name.status"
    tail -n 60 "$root/artifacts/$name.log"
    return "$result"
  fi
}

check_core() {
  export CARGO_TARGET_DIR="$root/target/fork"
  run validation-tests cargo +stable nextest run --offline -p metal2vulkan-validation --all-features --no-fail-fast
  run runtime-final-tests cargo +stable nextest run --offline -p metal2vulkan-runtime -p metal2vulkan-devtools --all-features
  run format cargo +stable fmt --all --check
  run clippy cargo +stable clippy --offline --workspace --all-targets --all-features -- -D warnings
  run clippy-none cargo +stable clippy --offline -p metal2vulkan --all-targets --no-default-features -- -D warnings
  for feature in serde ash cli; do
    run "clippy-$feature" cargo +stable clippy --offline -p metal2vulkan --all-targets --no-default-features --features "$feature" -- -D warnings
  done
  run rustdoc env RUSTDOCFLAGS=-Dwarnings cargo +stable doc --offline --workspace --all-features --no-deps
  run doctests cargo +stable test --offline --workspace --all-features --doc
  run api-docs env RUSTUP_TOOLCHAIN=stable scripts/api-docs
}

check_cpu() {
  export CARGO_TARGET_DIR="$root/target/cpu"
  export RUSTFLAGS="${RUSTFLAGS:+$RUSTFLAGS }--cfg m2v_cpu_icd"
  export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}
  export LP_NUM_THREADS=${LP_NUM_THREADS:-1}
  export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$root/.cache}
  export VK_LOADER_LAYERS_DISABLE='~implicit~'
  if [[ -d $root/.tools/validation/usr/share/vulkan/explicit_layer.d ]]; then
    export VK_LAYER_PATH="$root/.tools/validation/usr/share/vulkan/explicit_layer.d"
  fi
  export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
  export VK_LOADER_DEBUG=layer
  run cpu-regressions cargo +stable nextest run --offline -p metal2vulkan-runtime --all-features --success-output=final -E 'test(binding_repeats) | test(cpu_copy_size) | test(native_block_decode) | test(wrapped_owner)'
  run clippy-cpu cargo +stable clippy --offline -p metal2vulkan-runtime --all-features --all-targets -- -D warnings
  run exact-bc-tests cargo +stable nextest run --offline -p metal2vulkan-validation --lib --all-features --success-output=final -E 'test(runtime_lane::compressed_decode::bc1_to_bc5_texels_read_as_metal_s) | test(runtime_lane::compressed_decode::decoded_twins_follow_every_write)'
  for name in cpu-regressions exact-bc-tests; do
    if ! rg -q 'Insert instance layer.*VK_LAYER_KHRONOS_validation' "$root/artifacts/$name.log" || rg -n 'Validation Error|VUID-' "$root/artifacts/$name.log"; then
      printf 'FAIL (validation evidence)\n' >"$root/artifacts/$name.status"
      return 1
    fi
  done
}

case ${1:-core} in
  core) check_core ;;
  cpu) check_cpu ;;
  all) check_core; check_cpu ;;
  *) printf 'usage: %s [core|cpu|all]\n' "$0" >&2; exit 2 ;;
esac
