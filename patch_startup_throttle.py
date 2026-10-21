#!/usr/bin/env python3
"""Toggle Roblox's startup render throttle.

Under Darling the menu renders at ~3 FPS for the first 10 seconds: the
startup throttle is applied when the RenderJob is created and its normal
release never fires because the preRenderJob does not exist here, so only
the 10-second safety timeout restores rendering. (On real Macs the Metal
path compiles shaders on two threads, which wins that race; the GL path we
use compiles for ~850 ms on one thread and loses it.)

The patch makes the throttle argument read as false inside
RenderJob::setStartupThrottle, so the slow frequency is never applied:

    mov ebx, esi   ->   xor ebx, ebx

Usage: patch_startup_throttle.py [--check | --undo] [path/to/RobloxPlayer]

--check prints exactly one word: "patched", "original" or "not found".
--undo restores the original bytes; without flags the patch is applied.

The site is located by pattern on every run, so a client update just needs
the same code shape; when the shape is not found the patch is skipped. Two
log calls of that function must both match around the site (the throttle
value at Warning level and the null preRenderJob at Error level), which
keeps look-alike compiler idioms elsewhere in the binary from matching.
"""

import re
import sys
from pathlib import Path

_ARGS = sys.argv[1:]
UNDO = "--undo" in _ARGS
CHECK = "--check" in _ARGS
_path = next((a for a in _ARGS if not a.startswith("-")), None)
BUNDLE = Path(_path) if _path else (
    Path(__file__).parent / "RobloxPlayer.app" / "Contents" / "MacOS" / "RobloxPlayer")
if BUNDLE.is_dir():  # the .app bundle was given instead of its binary
    BUNDLE = BUNDLE / "Contents" / "MacOS" / "RobloxPlayer"

# mov ebx, esi; mov r14, rdi — the throttle argument and this-pointer move
# right after RenderJob::setStartupThrottle's prologue.
ARG_MOVES = re.compile(rb"\x89\xf3\x49\x89\xfe")
PATCHED = re.compile(rb"\x31\xdb\x49\x89\xfe")
PROLOGUE = re.compile(rb"\x55\x48\x89\xe5")
# Inside the same function: the "RenderJob::setStartupThrottle: {}" log call
# (r8d = 0x4e characters, edx = 3, Warning) ...
VALUE_LOG = re.compile(
    rb"\x41\xb8\x4e\x00\x00\x00\xba\x03\x00\x00\x00\x41\xb9\x01\x00\x00\x00")
# ... and the "preRenderJob is null" one (r8d = 0x55, edx = 2, Error).
NULL_LOG = re.compile(
    rb"\x41\xb8\x55\x00\x00\x00\xba\x02\x00\x00\x00\x41\xb9\x01\x00\x00\x00")
SEARCH_BACK = 0x600


def find_site(data: bytes, moves: re.Pattern):
    """Offset of the argument-moves pair inside RenderJob::setStartupThrottle.

    A candidate pair only counts when a function prologue precedes it and
    both log calls of setStartupThrottle follow within the function."""
    for site in NULL_LOG.finditer(data):
        start = site.start()
        window_start = max(0, start - SEARCH_BACK)
        window = data[window_start:start]
        found = [m.start() for m in moves.finditer(window)
                 if PROLOGUE.search(window[max(0, m.start() - 0x20):m.start()])]
        for move in reversed(found):
            body = data[window_start + move:start + 64]
            if VALUE_LOG.search(body) and NULL_LOG.search(body):
                return window_start + move
    return None


def main() -> int:
    if not BUNDLE.exists():
        print("no client")
        return 0
    data = bytes(BUNDLE.read_bytes())
    if CHECK:
        if find_site(data, PATCHED) is not None:
            print("patched")
        elif find_site(data, ARG_MOVES) is not None:
            print("original")
        else:
            print("not found")
        return 0
    if UNDO:
        offset = find_site(data, PATCHED)
        if offset is None:
            print("throttle patch not found, the client is already original")
            return 0
        BUNDLE.write_bytes(data[:offset] + b"\x89\xf3" + data[offset + 2:])
        print(f"startup render throttle restored (original bytes at 0x{offset:x})")
        return 0
    offset = find_site(data, ARG_MOVES)
    if offset is None:
        if find_site(data, PATCHED) is not None:
            print("startup render throttle already disabled")
            return 0
        print("startup throttle pattern not found in this client, skipping")
        return 1
    BUNDLE.write_bytes(data[:offset] + b"\x31\xdb" + data[offset + 2:])
    print(f"startup render throttle disabled (patched at 0x{offset:x})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
