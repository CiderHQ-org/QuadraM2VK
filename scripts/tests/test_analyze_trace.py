"""Regression checks for cumulative timing deltas and traces still being written."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("trace_analysis", Path(__file__).resolve().parents[1] / "analyze-trace.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(seconds, calls, nanos):
    return {"schema": 1, "elapsed_ns": seconds * 10**9, "device": "authored",
            "phases": {"render_pipeline": {"count": calls, "total_ns": nanos}}}


class TraceTests(unittest.TestCase):
    def test_burst_uses_interval_delta_rather_than_cumulative_time(self):
        result = analysis.summarize([row(1, 1, 100_000_000), row(2, 16, 548_000_000), row(3, 16, 548_000_000)])
        self.assertEqual(len(result["pipeline_bursts"]), 1)
        self.assertEqual(result["pipeline_bursts"][0]["pipeline_calls"], 15)
        self.assertEqual(result["pipeline_bursts"][0]["pipeline_summed_ms"], 448)
        self.assertFalse(result["upload_timing_available"])

    def test_device_time_and_counter_resets_do_not_create_negative_windows(self):
        result = analysis.summarize([row(4, 9, 900), row(1, 0, 0), row(2, 1, 500_000_000)])
        self.assertEqual(result["inferred_segments"], 2)
        self.assertEqual(len(result["pipeline_bursts"]), 1)
        result = analysis.summarize([row(1, 9, 900), row(2, 0, 0)])
        self.assertEqual(result["inferred_segments"], 2)

    def test_partial_final_write_is_ignored_but_corrupt_middle_line_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.jsonl"
            path.write_bytes(json.dumps(row(1, 0, 0)).encode() + b'\n{"schema":')
            self.assertEqual(len(analysis.load(path)), 1)
            path.write_bytes(b'bad\n' + json.dumps(row(1, 0, 0)).encode())
            with self.assertRaises(ValueError):
                analysis.load(path)

    def test_present_calls_are_normalized_by_interval_and_uploads_are_separate(self):
        first, second = row(1, 0, 0), row(3, 0, 0)
        for item, count, nanos in [(first, 10, 1000), (second, 130, 100_001_000)]:
            item["phases"]["present_call"] = {"count": count, "total_ns": 0}
            item["phases"]["texture_upload"] = {"count": count, "total_ns": nanos}
        result = analysis.summarize([first, second])
        self.assertEqual(result["upload_bursts"][0]["present_calls_per_second"], 60)
        self.assertEqual(result["upload_bursts"][0]["upload_summed_ms"], 100)
        self.assertTrue(result["upload_timing_available"])


if __name__ == "__main__":
    unittest.main()
