#!/usr/bin/env bash
# GPL-3.0-only. The legacy-AIR upgrader loads whichever libLLVM (15 or newer) a user's computer
# has, so run its authored tests against every one installed here, not only the newest.
#   scripts/test-llvm-matrix.sh            # needs VK_DRIVER_FILES for the one test that executes
# Writes llvm-matrix.json to $QM_TEST_OUTPUT (default artifacts/); fails if none is installed or any fails.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
out=${QM_TEST_OUTPUT:-$root/artifacts}
mkdir -p "$out"
export RUSTUP_TOOLCHAIN=${RUSTUP_TOOLCHAIN:-stable}
export CARGO_TARGET_DIR=${CARGO_TARGET_DIR:-$root/target/integration}
if [[ -d $root/.cargo-home/registry ]]; then
  export CARGO_HOME=${CARGO_HOME:-$root/.cargo-home}
  export CARGO_NET_OFFLINE=${CARGO_NET_OFFLINE:-true}
fi
export LP_NUM_THREADS=${LP_NUM_THREADS:-1}
manifest="$root/integration/quadra/prototype/metal-vulkan/host/Cargo.toml"
libraries=$(python3 - <<'PY'
import os, re
from pathlib import Path
directories = ["/usr/lib", "/usr/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/local/lib", "/app/lib"]
for root in ("/usr/lib", "/usr/lib64"):
    try:
        for entry in os.scandir(root):
            if entry.name.startswith("llvm") and entry.is_dir():
                directories += [entry.path + "/lib", entry.path + "/lib64"]
    except OSError:
        pass
found = {}
for directory in directories:
    try:
        names = os.listdir(directory)
    except OSError:
        continue
    for name in names:
        if not (name.startswith("libLLVM") and ".so" in name):
            continue
        real = Path(directory, name).resolve()
        if not real.is_file():
            continue
        match = re.search(r"libLLVM(?:-(\d+)|\.so\.(\d+))", real.name)
        version = int(next(group for group in match.groups() if group)) if match and any(match.groups()) else 0
        if version < 3:  # a soname such as libLLVM.so.1 names no version
            directory_version = re.search(r"llvm-?(\d+)", str(real.parent.parent))
            version = int(directory_version.group(1)) if directory_version else 0
        if version == 0 or version >= 15:
            found[str(real)] = version
for path, version in sorted(found.items(), key=lambda item: (-item[1], item[0])):
    print(version, path)
PY
)
if [[ -z $libraries ]]; then
  printf '%s\n' 'No libLLVM 15 or newer is installed; the legacy-AIR upgrader needs one.' >&2
  exit 1
fi
summary='[]'
status=0
while read -r version library; do
  printf 'libLLVM %s: %s\n' "$version" "$library"
  if MACOBLOX_METAL_LIBLLVM=$library cargo "+$RUSTUP_TOOLCHAIN" test --locked --offline --manifest-path "$manifest" \
      --lib air:: -- --include-ignored > "$out/llvm-$version.log" 2>&1; then
    result=PASS
  else
    result=FAIL
    status=1
    tail -n 30 "$out/llvm-$version.log" >&2
  fi
  summary=$(python3 -c 'import json,sys; rows=json.loads(sys.argv[1]); rows.append({"version":int(sys.argv[2]),"library":sys.argv[3],"result":sys.argv[4]}); print(json.dumps(rows))' "$summary" "$version" "$library" "$result")
  printf '  %s\n' "$result"
done <<< "$libraries"
python3 -c 'import json,sys; print(json.dumps(json.loads(sys.argv[1]), indent=2))' "$summary" > "$out/llvm-matrix.json"
exit "$status"
