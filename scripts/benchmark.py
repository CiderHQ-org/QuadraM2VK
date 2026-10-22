#!/usr/bin/env python3
"""Compare identical authored workloads on release bridges, with no validation layer."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess

ROOT = Path(__file__).resolve().parent.parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--runs", type=int, default=7)
parser.add_argument("--baseline", type=Path, default=ROOT / "target/baseline/release/libquadra_metal_host.so")
parser.add_argument("--upstream", type=Path, default=ROOT / "target/upstream/release/libquadra_metal_host.so")
parser.add_argument("--fork", type=Path, default=ROOT / "build/metal/libquadra_metal_host.so")
parser.add_argument("--output", type=Path, default=ROOT / "artifacts/benchmark")
args = parser.parse_args()
if args.runs < 3:
    parser.error("use at least three measured processes per variant")
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=True)
tests = ROOT / "integration/quadra/tests"


def run(command, **kwargs):
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, _ = process.communicate(timeout=120)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"command exited {process.returncode}: {command}")
        return stdout


run(["clang", "-O3", "-Wall", "-Wextra", "-Werror", str(tests / "metal_perf.c"), "-ldl", "-o", str(out / "metal-perf")])
run(["clang", "-O2", "-Wall", "-Wextra", "-Werror", str(ROOT / "scripts/measure_process.c"), "-o", str(out / "measure-process")])
run(["python3", str(tests / "assemble_metal_test.py"), str(tests / "metal_draw.ll"), str(out / "draw.metallib"), "bench_vertex:0", "bench_fragment:1"])
variants = {label: str(getattr(args, label).resolve()) for label in ["baseline", "upstream", "fork"]}
for path in variants.values():
    if not Path(path).is_file():
        raise FileNotFoundError(path)
env = os.environ.copy()
env.pop("VK_INSTANCE_LAYERS", None)
env.pop("VK_LAYER_PATH", None)
env["VK_LOADER_LAYERS_DISABLE"] = "~all~"
env.setdefault("VK_DRIVER_FILES", "/usr/share/vulkan/icd.d/lvp_icd.x86_64.json")
env.setdefault("LP_NUM_THREADS", "1")
env.pop("DISPLAY", None)
env.pop("WAYLAND_DISPLAY", None)
observations = []


def sample(label, index):
    local = env.copy()
    local["XDG_CACHE_HOME"] = str(out / "cache" / label)
    local["MACOBLOX_METAL_SHADER_CACHE"] = str(out / "cache" / label / "air")
    timing = out / f"{label}-{index}.time.json"
    with (out / f"{label}-{index}.stderr.log").open("w") as stderr:
        stdout = run([str(out / "measure-process"), str(timing), str(out / "metal-perf"), variants[label], str(out / "draw.metallib")], env=local, stdout=subprocess.PIPE, stderr=stderr, text=True)
    (out / f"{label}-{index}.stdout.log").write_text(stdout)
    metrics = {}
    for line in stdout.splitlines():
        match = re.fullmatch(r"(\w+) median_ms=([\d.]+) min_ms=([\d.]+) max_ms=([\d.]+) samples=(\d+)", line)
        if match:
            metrics[match[1]] = {"median_ms": float(match[2]), "min_ms": float(match[3]), "max_ms": float(match[4]), "samples": int(match[5])}
    if len(metrics) != 4:
        raise RuntimeError(f"missing benchmark measurements: {stdout}")
    print(f"{label} {index}: {metrics['draw_500_encode']['median_ms']:.3f} ms encode", flush=True)
    if index != "warmup":
        observations.append({"variant": label, "round": index, "measurements": metrics, **json.loads(timing.read_text())})


for label in variants:
    sample(label, "warmup")
for index in range(args.runs):
    order = list(variants) if index % 2 == 0 else list(reversed(variants))
    for label in order:
        sample(label, index + 1)
summary = {}
for label in variants:
    rows = [row for row in observations if row["variant"] == label]
    summary[label] = {name: statistics.median(row["measurements"][name]["median_ms"] for row in rows) for name in rows[0]["measurements"]}
    summary[label]["peak_rss_kib_max"] = max(row["peak_rss_kib"] for row in rows)
    summary[label]["wall_seconds_median"] = statistics.median(row["wall_seconds"] for row in rows)
result = {"variants": variants, "runs_per_variant": args.runs, "icd": env["VK_DRIVER_FILES"], "lp_num_threads": env["LP_NUM_THREADS"], "validation": False, "summary": summary, "observations": observations}
(out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(summary, indent=2))
