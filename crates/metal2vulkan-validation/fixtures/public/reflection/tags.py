#!/usr/bin/env python3
# Dumps the metallib function tags Metal answers MTLFunction's properties from: per function, in
# function-list order, its TYPE and TESS function-list tags and its CNST, VATT and VATY
# public-metadata tags, as hex, and the names of its other tags. Owned code, written for this
# repository. Usage: tags.py <fixture>.metallib... ; metallib-tags.txt is its output over the
# reflection fixtures' metallibs (built as probe.swift says).
import os
import struct
import sys

STRUCTURAL = {b"NAME", b"HASH", b"MDSZ", b"OFFT", b"VERS"}
KEPT = [b"TYPE", b"TESS", b"CNST", b"VATT", b"VATY"]


def tag_list(b, at, end):
    """The tags at `at` up to ENDT: (fourcc, bytes) pairs."""
    out = []
    while at < end:
        fourcc = b[at:at + 4]
        if fourcc == b"ENDT":
            return out
        (size,) = struct.unpack_from("<H", b, at + 4)
        out.append((fourcc, b[at + 6:at + 6 + size]))
        at += 6 + size
    raise ValueError("tag list without ENDT")


def dump(path):
    b = open(path, "rb").read()
    if b[:4] != b"MTLB":
        raise ValueError(f"{path}: not a metallib")
    sections = struct.unpack_from("<8Q", b, 0x18)
    flist, public = sections[0], sections[2]
    (count,) = struct.unpack_from("<I", b, flist)
    at = flist + 4
    print(f"== {os.path.basename(path)[:-len('.metallib')]}")
    for _ in range(count):
        (size,) = struct.unpack_from("<I", b, at)
        tags = tag_list(b, at + 4, at + size)
        at += size
        own = dict(tags)
        name = own[b"NAME"].rstrip(b"\0").decode()
        pub_off = struct.unpack_from("<3Q", own[b"OFFT"], 0)[0]
        rec = public + pub_off
        (rsize,) = struct.unpack_from("<I", b, rec)
        pub = tag_list(b, rec + 4, rec + 4 + rsize)
        both = tags + pub
        kept = dict(both)
        fields = [f"{t.decode()}={kept[t].hex()}" for t in KEPT if t in kept]
        other = sorted({t.decode() for t, _ in both if t not in STRUCTURAL and t not in KEPT})
        if other:
            fields.append("other=" + ",".join(other))
        print(f"function {name} " + " ".join(fields))


for p in sys.argv[1:]:
    dump(p)
