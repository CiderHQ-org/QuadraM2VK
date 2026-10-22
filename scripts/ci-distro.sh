#!/usr/bin/env bash
# GPL-3.0-only. Container dependencies and the same authored host/launcher checks on each distro.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)

dependencies() {
  if (( EUID != 0 )) || [[ ! -f /.dockerenv && ! -f /run/.containerenv && ${CI:-} != true ]]; then
    printf '%s\n' 'Dependency installation is intended for a disposable root container.' >&2
    exit 2
  fi
  case ${QM_CI_FAMILY:-} in
    debian)
      export DEBIAN_FRONTEND=noninteractive
      apt-get update
      apt-get install -y --no-install-recommends build-essential clang llvm lld pkg-config python3 ripgrep \
        libx11-dev libvulkan1 mesa-vulkan-drivers vulkan-tools vulkan-validationlayers spirv-tools \
        ca-certificates curl gnupg libwayland-dev wayland-protocols
      # Stable Debian/older Ubuntu ship LLVM older than the authored IR's minimum. Use
      # upstream's signed LLVM 23 packages while keeping the distro's Vulkan stack.
      . /etc/os-release
      curl --fail --silent --show-error https://apt.llvm.org/llvm-snapshot.gpg.key -o /tmp/quadra-llvm-key.asc
      gpg --batch --yes --dearmor -o /usr/share/keyrings/quadra-llvm.gpg /tmp/quadra-llvm-key.asc
      printf 'deb [signed-by=/usr/share/keyrings/quadra-llvm.gpg] https://apt.llvm.org/%s/ llvm-toolchain-%s-23 main\n' \
        "$VERSION_CODENAME" "$VERSION_CODENAME" > /etc/apt/sources.list.d/quadra-llvm.list
      apt-get update
      apt-get install -y --no-install-recommends llvm-23 clang-23 lld-23
      ;;
    fedora)
      dnf install -y gcc gcc-c++ make clang llvm llvm-libs lld pkgconf-pkg-config python3 ripgrep \
        libX11-devel vulkan-loader mesa-vulkan-drivers vulkan-tools vulkan-validation-layers spirv-tools \
        wayland-devel wayland-protocols-devel
      ;;
    arch)
      pacman -Syu --noconfirm --needed base-devel clang llvm lld pkgconf python ripgrep libx11 \
        vulkan-icd-loader vulkan-swrast vulkan-tools vulkan-validation-layers spirv-tools wayland wayland-protocols
      ;;
    *) printf 'Unknown container family: %s\n' "${QM_CI_FAMILY:-unset}" >&2; exit 2 ;;
  esac
}

check() {
  cd "$root"
  for version in 23 22 21; do
    if [[ -x /usr/lib/llvm-$version/bin/llvm-as ]]; then
      export PATH="/usr/lib/llvm-$version/bin:$PATH"
      break
    fi
  done
  export CARGO_BUILD_JOBS=${CARGO_BUILD_JOBS:-2}
  export LP_NUM_THREADS=${LP_NUM_THREADS:-1}
  local out="$root/artifacts/distro/${QM_CI_DISTRO:-local}"
  local rust_toolchain=${RUSTUP_TOOLCHAIN:-stable}
  mkdir -p "$out"
  printf 'RUNNING\n' > "$out/distro-tests.status"
  export QM_TEST_OUTPUT="$out"
  export XDG_CONFIG_HOME="$out/config"
  export XDG_DATA_HOME="$out/data"
  export XDG_CACHE_HOME="$out/cache"
  export XDG_RUNTIME_DIR="$out/runtime"
  mkdir -p "$XDG_RUNTIME_DIR"
  chmod 700 "$XDG_RUNTIME_DIR"
  export VK_DRIVER_FILES
  VK_DRIVER_FILES=$(python3 - <<'PY'
from pathlib import Path
paths=sorted(Path('/usr/share/vulkan/icd.d').glob('*lvp*.json'))
if not paths:
    raise SystemExit('No lavapipe ICD installed; software Vulkan execution is required.')
print(paths[0])
PY
)
  # The repository stores both component sources; no Darling SDK or Roblox download is
  # needed for this host-only lane. Point the companion's path dependency at this checkout.
  mkdir -p "$root/integration/quadra/vendor"
  python3 - "$root" <<'PY'
from pathlib import Path
import sys
root=Path(sys.argv[1]);target=root/'integration/quadra/vendor/metal2vulkan'
if target.is_symlink():
    target.unlink()
elif target.exists():
    target.rmdir()  # Only an empty checkout placeholder may be replaced.
target.symlink_to(root/'metal2vulkan',target_is_directory=True)
PY
  rustc "+$rust_toolchain" --version > "$out/versions.log"
  clang --version >> "$out/versions.log"
  llvm-as --version >> "$out/versions.log"
  vulkaninfo --summary >> "$out/versions.log" 2>&1
  cargo "+$rust_toolchain" fmt --manifest-path integration/quadra/prototype/metal-vulkan/host/Cargo.toml --check
  cargo "+$rust_toolchain" fetch --locked --manifest-path integration/quadra/prototype/metal-vulkan/host/Cargo.toml
  bash scripts/build.sh host > "$out/build.log" 2>&1
  bash scripts/test-bridge.sh --offscreen > "$out/bridge.log" 2>&1
  python3 -m unittest discover -s scripts/tests -v > "$out/trace-analysis-tests.log" 2>&1
  bash scripts/test-presentation.sh --compile-only > "$out/presentation-compile.log" 2>&1
  python3 - "$out" <<'PY'
import hashlib,json,os,platform,sys
from pathlib import Path
out=Path(sys.argv[1]);library=Path('build/metal/libquadra_metal_host.so')
summary={'status':'PASS','distro':os.environ.get('QM_CI_DISTRO','local'),
         'platform':platform.platform(),'icd':os.environ['VK_DRIVER_FILES'],
         'scope':'Linux host build, authored offscreen lavapipe correctness, launcher/preload controls and Wayland test compilation',
         'not_executed':['physical GPU tests','native presentation','Darling guest framework','actual Roblox gameplay'],
         'host_sha256':hashlib.file_digest(library.open('rb'),'sha256').hexdigest(),
         'preload':json.loads((out/'preload/results.json').read_text())}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
PY
  printf 'PASS\n' > "$out/distro-tests.status"
}

case ${1:-check} in
  deps) dependencies ;;
  check) check ;;
  *) printf 'usage: %s [deps|check]\n' "$0" >&2; exit 2 ;;
esac
