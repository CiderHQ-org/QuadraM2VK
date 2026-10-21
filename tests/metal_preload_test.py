"""Launcher helper bounds and skip behavior; all inputs and helper processes are authored."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

from macoblox import core, graphics, metal_preload
from session_fixture import bare_session

FAKE = r'''
import json, os, signal, subprocess, sys, time
mode=os.environ.get('PRELOAD_TEST_MODE','good')
if mode=='hang':
    child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'])
    with open(os.environ['PRELOAD_TEST_PIDS'],'w') as file: json.dump([os.getpid(),child.pid],file)
    def stop(_signal,_frame):
        child.wait(timeout=1)
        sys.exit(0)
    signal.signal(signal.SIGTERM,stop)
    print(json.dumps({'completed':0,'total':2,'failed':0}),flush=True)
    while True: time.sleep(.1)
if mode=='fail': sys.exit(2)
print('noise from a loader',flush=True)
print(json.dumps({'completed':0,'total':2,'failed':0}),flush=True)
print(json.dumps({'completed':2,'total':2,'failed':0,'status':'complete'}),flush=True)
'''


class MetalPreloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = dict(os.environ, MACOBLOX_METAL_SHADER_CACHE=str(self.root),
                                MACOBLOX_METAL_PRELOAD="1", MACOBLOX_METAL_PRELOAD_KEY="client-one")
        scope = hashlib.sha256(b"client-one").hexdigest()
        directory = self.root / "quadra-preload-v1"
        directory.mkdir()
        for index in range(2):
            (directory / f"pipeline-{scope}-{index:064x}.qmc").write_bytes(b"authored stub")
        self.helper = self.root / "helper.py"
        self.helper.write_text(FAKE)

    def test_empty_disabled_and_other_client_records_do_not_start_a_helper(self):
        for changes, expected in (({"MACOBLOX_METAL_PRELOAD": "0"}, "disabled"),
                                  ({"MACOBLOX_METAL_PRELOAD_KEY": "client-two"}, "empty")):
            with patch.object(metal_preload.subprocess, "Popen") as start:
                self.assertEqual(metal_preload.run("unused", {**self.environment, **changes})["status"], expected)
                start.assert_not_called()

    def test_progress_and_helper_failure_have_explicit_results(self):
        observations = []
        with patch.object(metal_preload, "__file__", str(self.helper)):
            result = metal_preload.run("unused", self.environment, progress=lambda *row: observations.append(row))
            self.assertEqual(result, {"status": "complete", "completed": 2, "total": 2, "failed": 0})
            self.assertEqual(observations[-1], (2, 2))
            result = metal_preload.run("unused", {**self.environment, "PRELOAD_TEST_MODE": "fail"})
            self.assertEqual(result["status"], "failed")

    def test_skip_ends_helper_and_its_descendant_before_returning(self):
        pids = self.root / "pids.json"
        cancel = threading.Event()
        def progress(_done, _total):
            if pids.exists():
                cancel.set()
        with patch.object(metal_preload, "__file__", str(self.helper)):
            result = metal_preload.run("unused", {**self.environment, "PRELOAD_TEST_MODE": "hang", "PRELOAD_TEST_PIDS": str(pids)}, progress, cancel, timeout=5)
        self.assertEqual(result["status"], "skipped")
        for pid in json.loads(pids.read_text()):
            self.assertFalse(Path(f"/proc/{pid}").exists(), pid)

    def test_timeout_returns_after_reaping_helper(self):
        with patch.object(metal_preload, "__file__", str(self.helper)):
            result = metal_preload.run("unused", self.environment, timeout=0)
        self.assertEqual(result["status"], "timeout")

    def test_metal_settings_and_skip_reach_host_and_guest(self):
        session = bare_session(dict(core.DEFAULT_SETTINGS, renderer="metal", metal_diagnostics=True))
        session._metal_preload_skipped = True
        with patch.object(core, "host_vram_bytes", return_value=0), patch.object(core, "icon_argb_file", side_effect=OSError):
            host = session._graphics_environment()
            guest = dict(item.split("=",1) for item in session.shim_variables())
        for environment in (host, guest):
            self.assertEqual(environment["MACOBLOX_METAL_PRELOAD"], "1")
            self.assertEqual(environment["MACOBLOX_METAL_PRELOAD_IN_PROCESS"], "0")
            self.assertEqual(environment["MACOBLOX_METAL_DIAGNOSTICS"], "1")
        disabled = graphics.metal_environment("/a path/metal", "/a path/cache", preload=False)
        self.assertEqual(disabled["MACOBLOX_METAL_PRELOAD"], "0")
        self.assertEqual(disabled["MACOBLOX_METAL_PRELOAD_IN_PROCESS"], "0")
