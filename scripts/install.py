#!/usr/bin/env python3
"""Install the reviewed build into an existing user Quadra checkout, with rollback."""
# GPL-3.0-only
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def command(*args, cwd=None, **kwargs):
    return subprocess.check_output(args, cwd=cwd, text=True, **kwargs).strip()


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def remove(path):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def atomic_write(path, content, mode=0o600):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".quadram2vk-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.fchmod(stream.fileno(), mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def running():
    found = []
    for entry in Path("/proc").glob("[0-9]*/comm"):
        try:
            name = entry.read_text().strip()
            if name == "RobloxPlayer" or name.startswith("quadrahq"):
                found.append(int(entry.parent.name))
            elif name.startswith("python"):
                arguments = (entry.parent / "cmdline").read_bytes().split(b"\0")
                if any(Path(os.fsdecode(arg)).name in ("quadrahq-launcher", "macoblox-launcher")
                       for arg in arguments if arg):
                    found.append(int(entry.parent.name))
        except (OSError, ValueError):
            pass
    if found:
        raise RuntimeError(f"Close Quadra and Roblox before installing (running PIDs: {found}).")


def backup_object(path, backup, index):
    row = {"path": str(path), "saved": f"objects/{index}"}
    saved = backup / row["saved"]
    if path.is_symlink():
        row.update(kind="symlink", link=os.readlink(path))
    elif path.is_file():
        row["kind"] = "file"
        shutil.copy2(path, saved)
    elif path.is_dir():
        row["kind"] = "directory"
        shutil.copytree(path, saved, symlinks=True)
    elif not path.exists():
        row["kind"] = "missing"
    else:
        raise RuntimeError(f"Unsupported installation object: {path}")
    return row


def restore(backup):
    running()
    metadata = json.loads((backup / "restore.json").read_text())
    for row in reversed(metadata["objects"]):
        path = Path(row["path"])
        remove(path)
        if row["kind"] != "missing":
            path.parent.mkdir(parents=True, exist_ok=True)
        if row["kind"] == "file":
            shutil.copy2(backup / row["saved"], path)
        elif row["kind"] == "directory":
            shutil.copytree(backup / row["saved"], path, symlinks=True)
        elif row["kind"] == "symlink":
            path.symlink_to(row["link"])
    print(f"Restored the saved Quadra files and settings from {backup}")


def source_stamp(target, sysroot):
    """Evaluate only the launcher's pure hash function, without importing the backend."""
    module = ast.parse((target / "launcher/macoblox/core.py").read_text())
    function = next(node for node in module.body
                    if isinstance(node, ast.FunctionDef) and node.name == "_metal_sources_hash")
    version_module = ast.parse((target / "launcher/macoblox/__init__.py").read_text())
    version = next(ast.literal_eval(node.value) for node in version_module.body
                   if isinstance(node, ast.Assign)
                   and any(isinstance(name, ast.Name) and name.id == "__version__" for name in node.targets))
    namespace = {"hashlib": hashlib, "__version__": version, "PROJECT": target,
                 "DARLING_SYSROOT": sysroot,
                 "METAL_BUILD_SCRIPT": target / "prototype/metal-vulkan/build.sh"}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<metal-source-stamp>", "exec"), namespace)
    return namespace["_metal_sources_hash"]()


def install(args):
    target, settings = args.target.resolve(), args.settings.absolute()
    patch = ROOT / "artifacts/quadra-bridge.patch"
    manifest = json.loads((ROOT / "artifacts/build-manifest.json").read_text())
    for filename in ("artifacts/quadra-bridge.patch", "build/metal/libquadra_metal_host.so",
                     "build/metal/Metal.framework/Versions/A/Metal"):
        if digest(ROOT / filename) != manifest["sha256"][filename]:
            raise RuntimeError(f"Release checksum mismatch: {filename}")
    fork = ROOT / "metal2vulkan"
    if not (fork / "crates/metal2vulkan-runtime/src/lib.rs").is_file():
        raise RuntimeError("Expand metal2vulkan-source.tar.gz into metal2vulkan/ first.")
    if (fork / ".git").exists():
        if command("git", "rev-parse", "HEAD", cwd=fork) != manifest["fork_revision"]:
            raise RuntimeError("The local fork revision differs from the release.")
        if command("git", "status", "--porcelain", cwd=fork):
            raise RuntimeError("The local fork has uncommitted changes; rebuild before installing.")
    if not (target / "launcher/quadrahq-launcher").is_file():
        raise RuntimeError(f"Quadra launcher not found in {target}")
    marker = target / "build/metal/quadram2vk-install.json"
    previous = json.loads(marker.read_text()).get("quadra_revision") if marker.is_file() else None
    if previous == manifest["quadra_bridge_revision"]:
        if all(digest(target / "build/metal" / name) == manifest["sha256"][f"build/metal/{name}"]
               for name in ("libquadra_metal_host.so", "Metal.framework/Versions/A/Metal")):
            print("This QuadraM2VK release is already installed; renderer/settings unchanged.")
            return
        raise RuntimeError("This revision's installed binaries differ; restore its backup before reinstalling.")
    if previous:
        upgrade = manifest.get("upgrade_patches", {}).get(previous)
        if not upgrade:
            raise RuntimeError(f"No verified upgrade patch from {previous}; retain this installation and its backup.")
        patch = ROOT / upgrade["path"]
        if digest(patch) != upgrade["sha256"]:
            raise RuntimeError("Upgrade patch checksum mismatch.")
        print(f"Updating QuadraM2VK {previous[:12]} -> {manifest['quadra_bridge_revision'][:12]}")
    patch_text = patch.read_text()
    command("git", "apply", "--check", cwd=target, input=patch_text)
    changed = [line.split("\t", 2)[2] for line in
               command("git", "apply", "--numstat", cwd=target, input=patch_text).splitlines()]
    for name in changed:
        if not (target / name).resolve().is_relative_to(target):
            raise RuntimeError(f"Patch path escapes Quadra: {name}")
    configuration = json.loads(settings.read_text()) if settings.exists() else {}
    if not isinstance(configuration, dict):
        raise RuntimeError("Quadra settings must be a JSON object.")
    environment = dict(os.environ, PYTHONPATH=str(target / "launcher"), PYTHONDONTWRITEBYTECODE="1")
    sysroot = command(sys.executable, "-B", "-c",
                      "from macoblox.runtime import resolve_runtime; print(resolve_runtime()['sysroot'])",
                      env=environment)
    if not Path(sysroot).is_dir():
        raise RuntimeError(f"Selected CiderDarling sysroot is missing: {sysroot}")
    print(f"Quadra: {target}\nSettings: {settings}\nRuntime: {sysroot}")
    print(f"Patch: {len(changed)} files; matching ABI 2 binaries; Metal and shader preloading enabled.")
    if args.check:
        print("Installation preflight: PASS (no changes made)")
        return
    running()
    # Catch read-only mounts before changing any installed files.
    for directory in (target, target / "build", settings.parent):
        probe = directory
        while not probe.exists():
            probe = probe.parent
        descriptor, name = tempfile.mkstemp(prefix=".quadram2vk-check-", dir=probe)
        os.close(descriptor)
        Path(name).unlink()
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = ROOT / "artifacts/install-backups" / f"{stamp}-{os.getpid()}"
    backup.mkdir(parents=True, mode=0o700)
    (backup / "objects").mkdir(mode=0o700)
    paths = [*(target / name for name in changed), target / "build/metal",
             target / "vendor/metal2vulkan", settings]
    objects = [backup_object(path, backup, index) for index, path in enumerate(paths)]
    atomic_write(backup / "restore.json", json.dumps({"target": str(target), "objects": objects}, indent=2).encode())
    try:
        command("git", "apply", cwd=target, input=patch_text)
        vendor = target / "vendor/metal2vulkan"
        vendor.parent.mkdir(parents=True, exist_ok=True)
        remove(vendor)
        vendor.symlink_to(fork, target_is_directory=True)
        metal = target / "build/metal"
        remove(metal)
        shutil.copytree(ROOT / "build/metal", metal, symlinks=True)
        atomic_write(metal / "sources.sha256", (source_stamp(target, sysroot) + "\n").encode(), 0o644)
        for name in ("libquadra_metal_host.so", "Metal.framework/Versions/A/Metal"):
            if digest(metal / name) != manifest["sha256"][f"build/metal/{name}"]:
                raise RuntimeError(f"Installed binary checksum mismatch: {name}")
        for name in changed:
            if name.endswith(".py"):
                ast.parse((target / name).read_text(), filename=name)
        configuration["renderer"] = "metal"
        configuration.setdefault("metal_shader_preload", True)
        configuration.setdefault("metal_diagnostics", False)
        atomic_write(settings, (json.dumps(configuration, indent=2) + "\n").encode())
        atomic_write(metal / "quadram2vk-install.json", json.dumps({
            "fork_revision": manifest["fork_revision"], "quadra_revision": manifest["quadra_bridge_revision"],
            "backup": str(backup), "runtime_version": manifest["runtime_version"]}, indent=2).encode())
    except BaseException:
        restore(backup)
        raise
    print(f"Installed and verified. Backup: {backup}")
    print(f"Rollback: python3 {Path(__file__).resolve()} --restore {backup}")
    print("Open Quadra normally. The first session learns pipelines; later launches preload them.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path.home() / ".local/share/QuadraHQ")
    parser.add_argument("--settings", type=Path,
                        default=Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "quadrahq/settings.json")
    parser.add_argument("--check", action="store_true", help="Verify compatibility without installing.")
    parser.add_argument("--restore", type=Path, help="Restore a previous installation backup; close Quadra first.")
    args = parser.parse_args()
    try:
        if args.restore:
            restore(args.restore.resolve())
        else:
            install(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"Installation stopped: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
