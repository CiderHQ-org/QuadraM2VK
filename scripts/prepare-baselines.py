#!/usr/bin/env python3
"""Build the original Quadra bridge with separately checked out upstream revisions."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parent.parent
QUADRA = ROOT / "integration/quadra"
PREFIX = "prototype/metal-vulkan/host/"
REVISIONS = {"baseline": "720ef1ce9ae2a2068999a8675717418b9416283e", "upstream": "a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b"}
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--old-runtime", type=Path, required=True)
parser.add_argument("--upstream-runtime", type=Path, required=True)
args = parser.parse_args()
env = os.environ.copy()
if (ROOT / ".cargo-home/registry").is_dir():
    env.setdefault("CARGO_HOME", str(ROOT / ".cargo-home"))
    env.setdefault("CARGO_NET_OFFLINE", "true")
records = {}
files = subprocess.check_output(["git", "-C", str(QUADRA), "ls-tree", "-r", "--name-only", "quadra-integration-base", "--", PREFIX], text=True).splitlines()
for label, runtime in [("baseline", args.old_runtime.resolve()), ("upstream", args.upstream_runtime.resolve())]:
    revision = subprocess.check_output(["git", "-C", str(runtime), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISIONS[label]:
        raise RuntimeError(f"{label}: expected {REVISIONS[label]}, found {revision}")
    dirty = subprocess.check_output(["git", "-C", str(runtime), "status", "--porcelain", "--untracked-files=no"], text=True)
    if dirty:
        raise RuntimeError(f"{runtime} has modified tracked files")
    host = ROOT / ".validation" / f"{label}-host"
    host.mkdir(parents=True, exist_ok=True)
    for name in files:
        path = host / name.removeprefix(PREFIX)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(subprocess.check_output(["git", "-C", str(QUADRA), "show", f"quadra-integration-base:{name}"]))
    manifest = host / "Cargo.toml"
    source = manifest.read_text()
    for crate in ["metal2vulkan", "metal2vulkan-air", "metal2vulkan-runtime"]:
        path = json.dumps(str(runtime / "crates" / crate))
        source, count = re.subn(rf'(?m)^{crate} = \{{ git = "[^"]+", rev = "[^"]+"', lambda _: f'{crate} = {{ path = {path}', source)
        if count != 1:
            raise RuntimeError(f"dependency replacement failed: {crate}")
    manifest.write_text(source)
    # Git sources become path sources; regenerate the lockfile with the same cached versions.
    lockfile = host / "Cargo.lock"
    lockfile.write_text("\n".join(line for line in lockfile.read_text().splitlines() if not line.startswith('source = "git+')) + "\n")
    local = env.copy()
    local["CARGO_TARGET_DIR"] = str(ROOT / "target" / label)
    subprocess.run(["cargo", "+stable", "build", "--locked", "--release", "--manifest-path", str(manifest)], env=local, check=True)
    records[label] = {"runtime_revision": revision, "host_base": "quadra-integration-base", "manifest": str(manifest)}
(ROOT / "artifacts/baselines.json").write_text(json.dumps(records, indent=2) + "\n")
