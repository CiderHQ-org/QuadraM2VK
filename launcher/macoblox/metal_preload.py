"""Replay locally learned Metal pipeline creations in a bounded, cancellable host process."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

MAX_PIPELINES = 4096
TIMEOUT = 120


def recorded(environment):
    """Count this client's descriptions without reading shader contents."""
    if environment.get("MACOBLOX_METAL_PRELOAD") == "0":
        return 0
    root = environment.get("MACOBLOX_METAL_SHADER_CACHE")
    if not root:
        return 0
    scope = hashlib.sha256(environment.get("MACOBLOX_METAL_PRELOAD_KEY", "unscoped").encode()).hexdigest()
    try:
        count = 0
        for path in (Path(root) / "quadra-preload-v1").glob(f"pipeline-{scope}-*.qmc"):
            if path.is_file():
                count += 1
                if count == MAX_PIPELINES:
                    break
        return count
    except OSError:
        return 0


def _stop(process):
    """Stop the helper and its children before the caller launches the game."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=2)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired as error:
            raise RuntimeError("The shader preload helper did not stop; close it before launching Roblox.") from error


def run(library, environment, progress=None, cancel=None, timeout=TIMEOUT, maximum=MAX_PIPELINES):
    """Return counts/status. Cache/driver failures are nonfatal; new shaders can still compile
    in the game. Skip and timeout end the whole helper before returning. progress(done,total).
    """
    if environment.get("MACOBLOX_METAL_PRELOAD") == "0":
        return {"status": "disabled", "completed": 0, "total": 0, "failed": 0}
    total = recorded(environment)
    if not total:
        return {"status": "empty", "completed": 0, "total": 0, "failed": 0}
    if cancel is not None and cancel.is_set():
        return {"status": "skipped", "completed": 0, "total": total, "failed": 0}
    result = {"status": "failed", "completed": 0, "total": total, "failed": 0}
    child_environment = {key: value for key, value in environment.items() if key != "LD_PRELOAD"}
    # Diagnostics describe the game. The preload helper has its own progress/counts and must
    # not interleave its device's snapshots with the gameplay trace.
    child_environment["MACOBLOX_METAL_DIAGNOSTICS"] = "0"
    child_environment["MACOBLOX_METAL_PRELOAD_IN_PROCESS"] = "0"
    if progress:
        progress(0, total)
    try:
        process = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--worker", str(library),
                                    str(max(1, min(int(maximum), MAX_PIPELINES)))],
                                   env=child_environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    except OSError as error:
        return {**result, "reason": str(error)}
    deadline = time.monotonic() + max(0, timeout)
    buffers = {}
    last_progress = 0.0
    try:
        with selectors.DefaultSelector() as selector:
            for stream in (process.stdout, process.stderr):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ)
                buffers[stream] = bytearray()
            while selector.get_map() or process.poll() is None:
                if cancel is not None and cancel.is_set():
                    result["status"] = "skipped"
                    break
                if time.monotonic() >= deadline:
                    result["status"] = "timeout"
                    break
                for key, _events in selector.select(timeout=0.1):
                    chunk = os.read(key.fileobj.fileno(), 4096)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    pending = buffers[key.fileobj]
                    pending.extend(chunk)
                    while b"\n" in pending:
                        line, _, rest = pending.partition(b"\n")
                        pending[:] = rest
                        if key.fileobj is not process.stdout or len(line) > 4096:
                            continue
                        try:
                            data = json.loads(line)
                            done, count, failed = data["completed"], data["total"], data["failed"]
                            if not all(type(value) is int for value in (done, count, failed)):
                                continue
                            if not 0 <= failed <= done <= count <= MAX_PIPELINES:
                                continue
                        except (ValueError, KeyError, TypeError):
                            continue
                        result.update(completed=done, total=count, failed=failed)
                        if data.get("status") in ("complete", "partial"):
                            result["status"] = data["status"]
                        now = time.monotonic()
                        if progress and (done == count or now - last_progress >= 0.1):
                            progress(done, count)
                            last_progress = now
                    if len(pending) > 8192:
                        del pending[:-4096]
            if result["status"] not in ("skipped", "timeout"):
                process.wait(timeout=5)
                if process.returncode:
                    result["status"] = "failed"
    finally:
        _stop(process)
        process.stdout.close()
        process.stderr.close()
    return result


def _worker(library, maximum):
    host = ctypes.CDLL(library)
    entry = host.quadra_metal_host_api
    entry.argtypes, entry.restype = (ctypes.c_uint32,), ctypes.c_void_p
    api = entry(2)
    if not api:
        raise RuntimeError("the shader preloader requires the matching ABI 2 Metal host")
    words = ctypes.cast(api + 8, ctypes.POINTER(ctypes.c_void_p))
    last_error = ctypes.CFUNCTYPE(ctypes.c_char_p)(words[0])
    callback_type = ctypes.CFUNCTYPE(None, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p)
    latest = {"completed": 0, "total": 0, "failed": 0}

    @callback_type
    def progress(done, total, failed, _context):
        latest.update(completed=done, total=total, failed=failed)
        print(json.dumps(latest), flush=True)

    preload = host.quadra_metal_host_preload
    preload.argtypes, preload.restype = (callback_type, ctypes.c_void_p, ctypes.c_uint32), ctypes.c_int
    count = preload(progress, None, maximum)
    if count < 0:
        raise RuntimeError((last_error() or b"shader preload failed").decode(errors="replace"))
    print(json.dumps({**latest, "status": "partial" if latest["failed"] else "complete"}), flush=True)


if __name__ == "__main__":
    try:
        if len(sys.argv) != 4 or sys.argv[1] != "--worker":
            raise ValueError("This helper is started by Quadra's launcher.")
        _worker(sys.argv[2], int(sys.argv[3]))
    except Exception as error:
        print(f"Shader preloading unavailable: {error}", file=sys.stderr)
        sys.exit(2)
