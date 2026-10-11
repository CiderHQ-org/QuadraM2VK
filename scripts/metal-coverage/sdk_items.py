#!/usr/bin/env python3
"""List the public items of the macOS SDK's Metal headers, by name only (Mac only: needs Xcode).

Prints one tab-separated line per item: header, container (the protocol or class; empty for a C
function), kind (protocol, class, category, property, method, class method, function), name (a
property's name, a method's full selector), unavailable on macOS (True/False), deprecated on
macOS (True/False). `coverage.py --sdk` compares these names with `coverage.tsv`. Nothing of the
headers but these names is read out; written for this repository.
"""
import re
import subprocess
import sys
from pathlib import Path

SDK = subprocess.check_output(["xcrun", "--sdk", "macosx", "--show-sdk-path"], text=True).strip()
HDR = Path(SDK) / "System/Library/Frameworks/Metal.framework/Headers"


def strip_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


def mac_unavailable(attrs):
    for m in re.finditer(r"API_UNAVAILABLE\(([^)]*)\)", attrs):
        if re.search(r"\bmacos\b", m.group(1)):
            return True
    return False


def deprecated(attrs):
    m = re.search(r"API_DEPRECATED(?:_WITH_REPLACEMENT)?\((.*)", attrs)
    if not m:
        return False
    return "macos(" in m.group(1)


def selector(decl):
    # decl: '- (ret)name:(T)a part:(T)b ...' without trailing attrs
    body = decl[1:].strip()
    body = re.sub(r"^\((?:[^()]|\([^()]*\))*\)", "", body).strip()
    # remove parenthesized types
    parts = []
    depth = 0
    out = ""
    for ch in body:
        if ch == "(":
            depth += 1
            continue
        if ch == ")":
            depth -= 1
            continue
        if depth == 0:
            out += ch
    toks = re.findall(r"([A-Za-z_][A-Za-z0-9_]*)\s*:", out)
    if not toks:
        name = re.match(r"\s*([A-Za-z_][A-Za-z0-9_]*)", out).group(1)
        return name
    return "".join(t + ":" for t in toks)


def prop_name(decl):
    d = re.sub(r"^@property\s*(\([^)]*\))?", "", decl).strip()
    d = re.sub(r"\b(API_[A-Z_]+|NS_[A-Z_]+|MTL_[A-Z_]+)\s*(\((?:[^()]|\([^()]*\))*\))?", " ", d)
    d = d.strip()
    # block-typed property: (^name)
    m = re.search(r"\(\s*\^\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)", d)
    if m:
        return m.group(1)
    m = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(\[[^\]]*\])?\s*$", d)
    return m.group(1)


def split_attrs(decl):
    m = re.search(r"\b(API_AVAILABLE|API_UNAVAILABLE|API_DEPRECATED|API_DEPRECATED_WITH_REPLACEMENT|NS_SWIFT_|NS_REFINED|MTL_API|NS_RETURNS|__attribute__|NS_AVAILABLE|NS_DEPRECATED|API_OBSOLETED)", decl)
    if m:
        return decl[: m.start()], decl[m.start():]
    return decl, ""


def parse(path):
    raw = strip_comments(path.read_text())
    raw = re.sub(r"(?m)^\s*#.*$", " ", raw)
    items = []
    # containers
    pat = re.compile(r"@(protocol|interface)\s+([A-Za-z_][A-Za-z0-9_]*)\s*(\([^)]*\))?([^;@{]*?)(?=\n|\{|$)", re.S)
    pos = 0
    while True:
        m = re.search(r"@(protocol|interface)\s+([A-Za-z0-9_]+)", raw[pos:])
        if not m:
            break
        start = pos + m.start()
        # forward decl?
        semi = raw.find(";", start)
        nl = raw.find("\n", start)
        line = raw[start:nl]
        if re.match(r"@protocol\s+[A-Za-z0-9_]+\s*(,\s*[A-Za-z0-9_]+\s*)*;", line) or re.match(r"@class", line):
            pos = start + len(m.group(0))
            continue
        end = raw.find("@end", start)
        kind = m.group(1)
        name = m.group(2)
        header_line = line
        category = re.match(r"@interface\s+\w+\s*\(([^)]*)\)", header_line)
        # container attrs: text since previous ';' or '@end' or '}' before start
        prev = max(raw.rfind(";", 0, start), raw.rfind("@end", 0, start), raw.rfind("}", 0, start))
        cattrs = raw[prev + 1:start]
        cont_unavail = mac_unavailable(cattrs)
        cont_dep = deprecated(cattrs)
        body = raw[start + len(header_line):end]
        # remove ivar block
        body = re.sub(r"^\s*\{[^}]*\}", "", body)
        ckind = "protocol" if kind == "protocol" else ("category" if category else "class")
        items.append((path.name, name, ckind, name, cont_unavail, cont_dep))
        for decl in body.split(";"):
            d = " ".join(decl.split())
            d = re.sub(r"@(optional|required)\s*", "", d).strip()
            if not d:
                continue
            core, attrs = split_attrs(d)
            unavail = cont_unavail or mac_unavailable(attrs)
            dep = deprecated(attrs) or cont_dep
            if d.startswith("@property"):
                items.append((path.name, name, "property", prop_name(core), unavail, dep))
            elif d.startswith("-") or d.startswith("+"):
                items.append((path.name, name, "method" if d[0] == "-" else "class method", selector(core), unavail, dep))
            elif d.startswith("typedef"):
                continue
            else:
                print("SKIP", path.name, name, d[:80], file=sys.stderr)
        pos = end + 4
    # C functions
    for m in re.finditer(r"MTL_(EXTERN|INLINE)\s+([^;{]*?)\b([A-Z][A-Za-z0-9_]*)\s*\(([^)]*)\)\s*([^;{]*)", raw):
        attrs = m.group(5) + m.group(2)
        nm = m.group(3)
        if nm.startswith("API_") or nm.startswith("NS_"):
            continue
        items.append((path.name, "", "function", nm, mac_unavailable(attrs), deprecated(attrs)))
    return items


def items():
    """Every item, once by (container, kind, name), in header order."""
    seen = set()
    for p in sorted(HDR.glob("*.h")):
        for it in parse(p):
            key = (it[1], it[2], it[3])
            if key in seen:
                continue
            seen.add(key)
            yield it


if __name__ == "__main__":
    for it in items():
        print("\t".join(str(x) for x in it))
