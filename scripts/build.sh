#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
what=${1:-all}
if [[ ! -f $root/metal2vulkan/Cargo.toml || ! -f $root/integration/quadra/prototype/metal-vulkan/host/Cargo.toml ]]; then
  printf '%s\n' 'Initialize the source components with: git submodule update --init' >&2
  exit 2
fi
vendor="$root/integration/quadra/vendor/metal2vulkan"
if [[ ! -e $vendor ]]; then
  if [[ -L $vendor ]]; then
    printf '%s\n' 'The companion runtime link is broken; initialize the metal2vulkan submodule first.' >&2
    exit 2
  fi
  mkdir -p "$(dirname -- "$vendor")"
  ln -s ../../../metal2vulkan "$vendor"
fi
export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target/integration}
export MACOBLOX_BUILD_DIR=${MACOBLOX_BUILD_DIR:-$root/build}
if [[ -d $root/.cargo-home/registry ]]; then
  export CARGO_HOME=${CARGO_HOME:-$root/.cargo-home}
  export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
fi
if [[ $what != host && -z ${DARLING_SYSROOT:-} ]]; then
  shopt -s nullglob
  roots=(/opt/ciderdarling/*/libexec/darling)
  if [[ ${#roots[@]} != 1 ]]; then
    printf '%s\n' 'Set DARLING_SYSROOT to the CiderDarling libexec/darling directory.' >&2
    exit 2
  fi
  export DARLING_SYSROOT=${roots[0]}
fi
exec "$root/integration/quadra/prototype/metal-vulkan/build.sh" "$what"
