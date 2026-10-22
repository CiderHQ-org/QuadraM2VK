#!/usr/bin/env python3
"""Package reviewed sources and the matching experimental ABI 2 binaries."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "artifacts"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--darling-sysroot-revision", default="unspecified")
parser.add_argument("--hardware-run", type=Path, help="Include a completed, reviewed hardware run.")
args = parser.parse_args()


def command(*args):
    return subprocess.check_output(args, cwd=ROOT, text=True).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


for repo in ["metal2vulkan", "integration/quadra"]:
    if command("git", "-C", str(ROOT / repo), "status", "--porcelain"):
        raise RuntimeError(f"commit or account for changes in {repo} before packaging")
required = ["validation-tests", "runtime-final-tests", "format", "clippy", "clippy-none", "clippy-serde", "clippy-ash", "clippy-cli", "rustdoc", "doctests", "api-docs", "cpu-regressions", "clippy-cpu", "exact-bc-tests", "bridge-tests"]
for name in required:
    if (OUT / f"{name}.status").read_text().strip() != "PASS":
        raise RuntimeError(f"{name} has not passed")
if (OUT / "preload/preload-tests.status").read_text().strip() != "PASS":
    raise RuntimeError("shader preload qualification has not passed")
source_files = ["metal2vulkan-source.tar.gz", "quadra-bridge-source.tar.gz", "metal2vulkan.patch", "metal2vulkan-patches.mbox", "quadra-bridge.patch"]
# Keep upgrades usable from an extracted package, which has no Git object database.
quadra_repo = ROOT / "integration/quadra"
upgrade_patches = {}
preload_base = "62593606b4dbc10502091f8b36313563a77365bf"
head = command("git", "-C", str(quadra_repo), "rev-parse", "HEAD")
for revision in command("git", "-C", str(quadra_repo), "rev-list", f"{preload_base}^..HEAD").splitlines():
    if revision == head:
        continue
    filename = f"quadra-update-{revision[:12]}.patch"
    patch = subprocess.check_output(["git", "-C", str(quadra_repo), "diff", "--binary", revision, head])
    (OUT / filename).write_bytes(patch)
    source_files.append(filename)
    upgrade_patches[revision] = {"path": f"artifacts/{filename}", "sha256": digest(OUT / filename)}
binaries = [ROOT / "build/metal/libquadra_metal_host.so", ROOT / "build/metal/Metal.framework/Versions/A/Metal"]
for path in [*(OUT / name for name in source_files), *binaries, ROOT / "RESULTS.md", OUT / "benchmark/results.json"]:
    if not path.is_file():
        raise FileNotFoundError(path)
for filename, repo in [("metal2vulkan-source.tar.gz", "metal2vulkan"), ("quadra-bridge-source.tar.gz", "integration/quadra")]:
    with tarfile.open(OUT / filename, "r:gz") as snapshot:
        revision = snapshot.pax_headers.get("comment")
    if revision != command("git", "-C", str(ROOT / repo), "rev-parse", "HEAD"):
        raise RuntimeError(f"{filename} does not match its repository revision")
hardware = None
if args.hardware_run:
    hardware_dir = args.hardware_run.resolve()
    hardware_relative = hardware_dir.relative_to(ROOT)
    hardware = json.loads((hardware_dir / "review.json").read_text())
    if (hardware_dir / "hardware-tests.status").read_text().strip() != "PASS" or hardware["status"] != "PASS":
        raise RuntimeError("hardware run has not passed")
    if hardware["gpu"]["hardware_devices"] < 1 or hardware["benchmark_processes"] != 21:
        raise RuntimeError("hardware device or benchmark evidence is incomplete")
    if hardware["current_binary_sha256"]["fork"] != digest(binaries[0]):
        raise RuntimeError("packaged host differs from the reviewed hardware candidate")
manifest = {
    "name": "QuadraM2VK experimental ABI 2",
    "fork_revision": command("git", "-C", str(ROOT / "metal2vulkan"), "rev-parse", "HEAD"),
    "upstream_base": "a2c440b8bcdcf20d34aa1694b61ae22c2a7ea68b",
    "quadra_bridge_revision": command("git", "-C", str(ROOT / "integration/quadra"), "rev-parse", "HEAD"),
    "quadra_base": "94de4c6dec5e466d98ae9121cf7442f835dbf09e",
    "runtime_version": int(re.search(r"RUNTIME_VERSION\s*:\s*\w+\s*=\s*(\d+)", (ROOT / "metal2vulkan/crates/metal2vulkan/src/cache/mod.rs").read_text())[1]),
    "rust": command("rustc", "+stable", "--version"),
    "clang": command("clang", "--version").splitlines()[0],
    "build_target": "x86_64 Linux host + x86_64 Darwin guest framework",
    "darling_sysroot_used": args.darling_sysroot_revision,
    "qualification": "authored lavapipe offscreen execution with Vulkan validation, shader preload/reuse, live memory, cache recovery and launcher cancellation; guest cross-build only" if hardware is None else "lavapipe plus reviewed RTX 3060 Ti correctness, native Wayland lifecycle, staged launcher preparation and authored release benchmarks; guest cross-build only",
    "preload_qualification": json.loads((OUT / "preload/results.json").read_text()),
    "upgrade_patches": upgrade_patches,
    "historical_benchmarks": {"fork_revision": "c50214a8e1adb29059e62d55fdfff8abdc464153", "bridge_revision": "5cbc9fb5ecb5d0e37f2386b78f323bfecd9ba545", "path": "artifacts/benchmark/results.json"},
    "unqualified": ["hardware GPU performance", "gameplay", "X11/Wayland WSI", "guest handler and queue execution"] if hardware is None else ["gameplay frame times", "guest handler and queue edge cases", "long-session stability", "other hardware GPUs"],
    "sha256": {str(path.relative_to(ROOT)): digest(path) for path in [*(OUT / name for name in source_files), *binaries]},
}
if hardware is not None:
    manifest["hardware_review"] = {"path": str(hardware_relative / "review.json"), "device": hardware["gpu"], "execution": hardware["execution"], "summary": hardware["summary"]}
prior_presentation = OUT / "presentation/review.json"
if prior_presentation.is_file():
    review = json.loads(prior_presentation.read_text())
    if review["status"] != "PASS":
        raise RuntimeError("prior presentation review has not passed")
    for name, expected in review["evidence_sha256"].items():
        if digest(prior_presentation.parent / name) != expected:
            raise RuntimeError(f"prior presentation evidence changed: {name}")
    manifest["prior_presentation_review"] = {"path": "artifacts/presentation/review.json",
        "bridge_revision": review["bridge_revision"], "host_sha256": review["host_sha256"],
        "scope": review["scope"]}
(OUT / "build-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
with tempfile.TemporaryDirectory(prefix="quadram2vk-package-", dir=ROOT) as temporary:
    stage = Path(temporary) / "QuadraM2VK"
    stage.mkdir()
    for name in ["README.md", "RESULTS.md", "HARDWARE-TESTING.md", "PRESENTATION-TESTING.md", "CI.md", "PLAN.md", "PERFORMANCE-STABILITY-AUDIT.md", "LICENSE"]:
        shutil.copy2(ROOT / name, stage / name)
    shutil.copytree(ROOT / "scripts", stage / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / ".github/workflows", stage / ".github/workflows")
    shutil.copytree(ROOT / "build/metal", stage / "build/metal", symlinks=True)
    artifacts = stage / "artifacts"
    artifacts.mkdir()
    for name in source_files + ["build-manifest.json"]:
        shutil.copy2(OUT / name, artifacts / name)
    (artifacts / "SHA256SUMS").write_text("".join(
        f"{digest(artifacts / name)}  {name}\n" for name in source_files + ["build-manifest.json"]
    ))
    (artifacts / "benchmark").mkdir()
    shutil.copy2(OUT / "benchmark/results.json", artifacts / "benchmark/results.json")
    evidence_logs = required + ["core-tests", "benchmark", "wait-benchmark", "process-release", "host-smoke", "abi-regressions", "host-rust-tests", "clippy-bridge", "preload-build", "preload-bridge-checks", "preload-cpu-checks", "presentation-build", "presentation-bridge-checks", "area-preload-build", "area-preload-bridge-checks", "trace-analysis-tests", "launcher-preload-build", "launcher-preload-bridge-checks", "launcher-preload-final-software"]
    for name in evidence_logs:
        path = OUT / f"{name}.log"
        if path.is_file():
            shutil.copy2(path, artifacts / path.name)
    for name in required:
        shutil.copy2(OUT / f"{name}.status", artifacts / f"{name}.status")
    (artifacts / "preload").mkdir()
    for path in (OUT / "preload").iterdir():
        if path.is_file() and path.suffix in {".log", ".json", ".status"}:
            shutil.copy2(path, artifacts / "preload" / path.name)
    for name in ["process-release.time.json", "wait-benchmark.time.json", "wait-benchmark-sources.txt", "baselines.json"]:
        shutil.copy2(OUT / name, artifacts / name)
    # Explicit numeric/authored evidence only. Never include client shader caches or raw game logs.
    for relative in ["presentation/review.json", "presentation/current.json", "presentation/current.log",
                     "presentation/baseline.json", "presentation/baseline.log",
                     "area-preload-baseline/results.json", "gameplay-review-0b73dd0.json",
                     "gameplay-numeric-analysis-0b73dd0.json", "area-upgrade-qualification.json",
                     "launcher-preload-local-review.json", "launcher-preload-ui/review.json"]:
        path = OUT / relative
        if path.is_file():
            destination = artifacts / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    if hardware is not None:
        # Keep the reviewed evidence and timings, without copying driver/shader caches.
        for folder in [hardware_dir, hardware_dir / "bridge", hardware_dir / "bridge/preload",
                       hardware_dir / "presentation", hardware_dir / "benchmark"]:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in {".log", ".json", ".status"}:
                    destination = stage / path.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    # Historical evidence remains explicitly tied to its original binary hashes in review.json.
    historical = OUT / "hardware-20261021T224406Z-199088"
    if historical.is_dir():
        for folder in [historical, historical / "bridge", historical / "benchmark"]:
            for path in folder.iterdir():
                if path.is_file() and path.suffix in {".log", ".json", ".status"}:
                    destination = stage / path.relative_to(ROOT)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, destination)
    for source in (ROOT / "metal2vulkan").glob("LICENSE*"):
        (stage / "licenses/metal2vulkan").mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, stage / "licenses/metal2vulkan" / source.name)
    shutil.copy2(ROOT / "integration/quadra/LICENSE", stage / "licenses/Quadra-GPL-3.0.txt")
    instructions = """Experimental binaries and source archives. Read RESULTS.md before deployment.

To expand the source layout described by README.md, from this directory:
  tar -xzf artifacts/metal2vulkan-source.tar.gz
  mkdir -p integration/quadra
  tar -xzf artifacts/quadra-bridge-source.tar.gz --strip-components=1 -C integration/quadra
  mkdir -p integration/quadra/vendor
  ln -s ../../../metal2vulkan integration/quadra/vendor/metal2vulkan

The archive is a source snapshot. The mbox preserves the fork commits for git am
on top of upstream a2c440b. Guest and host binaries require ABI 2 together.
"""
    (stage / "START-HERE.txt").write_text(instructions)

    def normalized(member):
        member.uid = member.gid = 0
        member.uname = member.gname = ""
        member.mtime = 0
        return member

    archive = OUT / "QuadraM2VK-experimental-linux-x86_64.tar.gz"
    with tarfile.open(archive, "w:gz", compresslevel=6) as tar:
        tar.add(stage, arcname=stage.name, filter=normalized)
hashes = [archive, OUT / "build-manifest.json", *(OUT / name for name in source_files)]
(OUT / "SHA256SUMS").write_text("".join(f"{digest(path)}  {path.name}\n" for path in hashes))
print(archive)
