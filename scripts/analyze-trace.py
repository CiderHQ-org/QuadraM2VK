#!/usr/bin/env python3
"""Summarize numeric Metal traces without exporting shader data or treating present rate as FPS."""
# GPL-3.0-only
import argparse
import json
from pathlib import Path

MAX_TRACE = 16 << 20
PIPELINE_PHASES = ("render_pipeline", "compute_pipeline")
UPLOAD_PHASES = ("buffer_upload", "texture_upload")


def load(path):
    with path.open("rb") as stream:
        data = stream.read(MAX_TRACE + 1)
    if len(data) > MAX_TRACE:
        raise ValueError("trace exceeds the renderer's 16 MiB limit")
    rows = []
    lines = data.splitlines(keepends=True)
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                break  # The running renderer may be in the middle of its last write.
            raise ValueError(f"invalid JSON on line {index + 1}") from None
        if row.get("schema") != 1 or not isinstance(row.get("elapsed_ns"), int):
            raise ValueError(f"unsupported snapshot on line {index + 1}")
        rows.append(row)
    if not rows:
        raise ValueError("trace contains no complete snapshots")
    return rows


def summarize(rows, threshold_ms=50.0):
    windows = []
    segments = 1
    for before, after in zip(rows, rows[1:]):
        elapsed = after["elapsed_ns"] - before["elapsed_ns"]
        phases = {}
        reset = elapsed <= 0 or before.get("device") != after.get("device")
        for name, value in after.get("phases", {}).items():
            previous = before.get("phases", {}).get(name, {})
            count = value.get("count", 0) - previous.get("count", 0)
            nanos = value.get("total_ns", 0) - previous.get("total_ns", 0)
            reset |= count < 0 or nanos < 0
            phases[name] = {"calls": count, "summed_ms": round(nanos / 1e6, 3)}
        if reset:
            segments += 1
            continue
        seconds = elapsed / 1e9

        def difference(section, key):
            return after.get(section, {}).get(key, 0) - before.get(section, {}).get(key, 0)

        windows.append({
            "segment": segments,
            "end_seconds": round(after["elapsed_ns"] / 1e9, 3),
            "window_seconds": round(seconds, 3),
            "pipeline_calls": sum(phases.get(p, {}).get("calls", 0) for p in PIPELINE_PHASES),
            "pipeline_summed_ms": round(sum(phases.get(p, {}).get("summed_ms", 0) for p in PIPELINE_PHASES), 3),
            "upload_summed_ms": round(sum(phases.get(p, {}).get("summed_ms", 0) for p in UPLOAD_PHASES), 3),
            "present_calls_per_second": round(phases.get("present_call", {}).get("calls", 0) / seconds, 3),
            "cache_misses": difference("shader_cache", "misses"),
            "new_recipes": difference("preload_process", "recorded"),
            "acquire_timeouts": difference("presentation", "acquire_timeouts"),
            "phases": phases,
        })
    last = rows[-1]
    return {
        "scope": "cumulative-call deltas; intervals can overlap across threads and phases; present-call rate is not displayed FPS",
        "snapshots": len(rows),
        "inferred_segments": segments,
        "last_elapsed_seconds": round(last["elapsed_ns"] / 1e9, 3),
        "last_preload_counts": {key: last.get("preload_process", {}).get(key, 0) for key in ("warmed", "recorded", "failed")},
        "upload_timing_available": any(p in last.get("phases", {}) for p in UPLOAD_PHASES),
        "pipeline_bursts": [w for w in windows if w["pipeline_summed_ms"] >= threshold_ms],
        "upload_bursts": [w for w in windows if w["upload_summed_ms"] >= threshold_ms],
        "acquire_timeout_windows": [w for w in windows if w["acquire_timeouts"] > 0],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--threshold-ms", type=float, default=50.0)
    args = parser.parse_args()
    if not 0 <= args.threshold_ms <= 1e9:
        parser.error("threshold must be a nonnegative finite number")
    try:
        print(json.dumps(summarize(load(args.trace), args.threshold_ms), indent=2))
    except (OSError, ValueError, TypeError) as error:
        parser.exit(1, f"Trace analysis stopped: {error}\n")


if __name__ == "__main__":
    main()
