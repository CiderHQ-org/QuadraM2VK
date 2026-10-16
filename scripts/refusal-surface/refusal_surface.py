#!/usr/bin/env python3
"""Generate docs/REFUSALS.md: every refusal of something Metal accepts, from the code.

The refusal surface is read from the construction sites of the typed refusals, outside tests:

- the translator's and the AIR front end's `Unsupported` (a `Feature` and its `Needs`): the
  constructors `Unsupported::not_implemented`, `not_representable`, `needs`, `needs_any_of`,
  `driver_defect`, `runtime_driver_defect`, `metal_gpu_family`, and `Unsupported { .. }`
  literals, in the `src` of crates/metal2vulkan-air, crates/metal2vulkan and
  crates/metal2vulkan-runtime;
- the runtime's `RuntimeGap` (`Error::NotImplemented`) and `AppleOnly` (`Error::AppleOnly`);
- the binder's refusals of valid bind states at a device limit (`BindError`);
- the translation caps (`Limit`, `Error::LimitExceeded`);
- the rasterization-rate maps the compile layer does not compile (`UnpinnedCompile`, the
  runtime's `Error::RateMap`);
- the translation deadline (`BudgetExceeded`, `Error::Budget`; the runtime's
  `Error::Compile(Budget)`).

Each refusal is one row per variant, with what it blocks (the variant's doc comment), the device
condition (from the constructor: every device, or the capabilities, driver defects or families
its sites name), its sites, and a reach mark, which is the one hand-kept column (`reach.tsv` beside this script).

    scripts/refusal-surface/refusal_surface.py           # write docs/REFUSALS.md
    scripts/refusal-surface/refusal_surface.py --check   # fail when a variant, its condition
                                                          # or reach mark is stale (sites and
                                                          # line numbers may drift)
    scripts/refusal-surface/refusal_surface.py --table   # print the plain table only

It reads source text only (no build), so it runs anywhere.
"""
import argparse
import re
import subprocess
import sys
from collections import OrderedDict, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "REFUSALS.md"
REACH = Path(__file__).resolve().parent / "reach.tsv"

CRATES = ROOT / "crates"
TREES = [
    ("air", CRATES / "metal2vulkan-air" / "src"),
    ("translator", CRATES / "metal2vulkan" / "src"),
    ("runtime", CRATES / "metal2vulkan-runtime" / "src"),
]

CONSTRUCTORS = {
    "not_implemented": "NotImplemented",
    "not_representable": "NotRepresentable",
    "needs": "AnyOf",
    "needs_any_of": "AnyOf",
    "driver_defect": "DriverDefect",
    "runtime_driver_defect": "RuntimeDriverDefect",
    "metal_gpu_family": "MetalGpuFamily",
}

# Binder refusals of a valid bind state: a device limit or an unestablished reading. The other
# `BindError` variants are Metal's own validation (misuse), or carry an `Unsupported` already
# counted under its feature.
BIND_REFUSALS = [
    "DescriptorRangeExceeded",
    "TexelBufferTooLarge",
    "HeapTooLarge",
    "TessellationFactorRange",
    "PatchIndexRange",
    "ControlPointIndexRange",
    "MissingStageInRegion",
    "UniformViewAliased",
    "MissingDeviceAddress",
]

REACH_MARKS = OrderedDict(
    [
        ("system", "a guest's CoreAnimation, CoreImage or VideoToolbox plausibly reaches it"),
        ("app", "a typical Metal app plausibly reaches it from public API"),
        ("rare", "public API, but specialized: few apps reach it"),
        ("none", "no public API path: malformed or non-Metal input, Apple-private calls, or a cap no valid program of ordinary size meets"),
    ]
)


def lex(text):
    """Blank comments, strings and char literals (keeping newlines and offsets)."""
    out = list(text)
    i, n = 0, len(text)

    def blank(a, b):
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
            continue
        if c == "r" and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] == "_")):
            m = re.match(r'r(#*)"', text[i : i + 10])
            if m:
                close = '"' + m.group(1)
                j = text.find(close, i + len(m.group(0)))
                j = n if j < 0 else j + len(close)
                blank(i + 1, j - 1)
                i = j
                continue
        if c == '"':
            j = i + 1
            while j < n and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            blank(i + 1, j)
            i = j + 1
            continue
        if c == "'":
            m = re.match(r"'(\\.[^']*|[^'\\])'", text[i : i + 12])
            if m:
                blank(i + 1, i + len(m.group(0)) - 1)
                i += len(m.group(0))
                continue
        i += 1
    return "".join(out)


def close_of(text, start, open_ch, close_ch):
    depth = 0
    for k in range(start, len(text)):
        if text[k] == open_ch:
            depth += 1
        elif text[k] == close_ch:
            depth -= 1
            if depth == 0:
                return k
    return len(text) - 1


def blank_test_items(text):
    """Blank `#[cfg(test)]` items (modules, functions, impls) so their sites do not count."""
    out = list(text)
    for m in re.finditer(r"#\[cfg\(test\)\]", text):
        k = text.find("{", m.end())
        semi = text.find(";", m.end())
        if k < 0 or (0 <= semi < k):
            continue
        e = close_of(text, k, "{", "}")
        for p in range(m.start(), e + 1):
            if out[p] != "\n":
                out[p] = " "
    return "".join(out)


def source_files():
    for tree, base in TREES:
        for p in sorted(base.rglob("*.rs")):
            rel = p.relative_to(ROOT).as_posix()
            if "/tests/" in rel or rel.endswith("/tests.rs") or rel.endswith("_tests.rs"):
                continue
            yield tree, rel, blank_test_items(lex(p.read_text()))


def fn_spans(text):
    """(start, end, name) of every function body in text."""
    spans = []
    for m in re.finditer(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^{;]*?>)?\s*\(", text):
        k = close_of(text, m.end() - 1, "(", ")")
        b = text.find("{", k)
        semi = text.find(";", k)
        if b < 0 or (0 <= semi < b):
            continue
        spans.append((b, close_of(text, b, "{", "}"), m.group(1)))
    return spans


def enclosing(spans, pos):
    best = None
    for b, e, name in spans:
        if b <= pos <= e and (best is None or b > best[0]):
            best = (b, e, name)
    return best


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1


def split_args(s):
    args, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([{<":
            depth += 1
        elif ch in ")]}>":
            depth -= 1
        if ch == "," and depth == 0:
            args.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        args.append(cur.strip())
    return args


CAP_TOKENS = r"(VkFeature|VkExtension|VkProperty|VkLimit|ApiVersion|DriverWorkaround|RuntimeDriverDefect)::([A-Za-z0-9_]+)"


def conditions(kind, needs_text):
    if kind == "NotImplemented":
        return {"every device (not implemented)"}
    if kind == "NotRepresentable":
        return {"every device (no Vulkan mechanism)"}
    toks = re.findall(CAP_TOKENS, needs_text)
    if kind == "AnyOf":
        caps = sorted({f"{a}::{b}" for a, b in toks if a.startswith("Vk") or a == "ApiVersion"})
        if "Capability::Format" in needs_text or "FormatUsage::" in needs_text:
            caps.append("a format's usage")
        if not caps:
            expr = " ".join(needs_text.split())
            expr = re.sub(r"^vec!\[\s*(.*?)\s*\]$", r"\1", expr)
            if "addressed_needs" in expr:
                return {"devices without buffer device addresses (`addressed_needs`)"}
            for cap, what in (("Limit", "a device limit"), ("Feature", "a Vulkan feature"), ("Property", "a Vulkan property"), ("Extension", "a Vulkan extension"), ("Format", "a format's usage")):
                if expr.startswith(f"Capability::{cap}"):
                    return {f"devices short of {what} the site names"}
            if expr and len(expr) <= 70:
                return {f"devices without a capability the site computes (`{expr}`)"}
            return {"devices without a capability the site computes"}
        return {"devices without " + c for c in caps}
    if kind == "DriverDefect":
        ws = sorted({b for a, b in toks if a == "DriverWorkaround"})
        return {"drivers with workaround " + w for w in ws} or {"drivers with a defect"}
    if kind == "RuntimeDriverDefect":
        ws = sorted({b for a, b in toks if a == "RuntimeDriverDefect"})
        return {"drivers with defect " + w for w in ws} or {"drivers with a defect"}
    if kind == "MetalGpuFamily":
        return {"devices that do not claim the Metal GPU family"}
    return {"varies (" + kind + ")"}


def is_pattern(text, start, end):
    """Whether the variant path at [start, end) is a match pattern rather than a value."""
    k = end
    while k < len(text) and text[k] in " \t":
        k += 1
    if k < len(text) and text[k] in "({":
        k = close_of(text, k, text[k], ")" if text[k] == "(" else "}") + 1
    rest = text[k : k + 3].lstrip()
    if rest.startswith("=>") or (rest.startswith("|") and not rest.startswith("||")):
        return True
    before = text[max(0, start - 30) : start].rstrip()
    # A closure's parameters (`let refuse = |a, b| Variant { .. }`) end in `|` too: the value
    # after them is built, not matched.
    closure = re.search(r"(?:=|\(|,|\bmove)\s*\|[^|]*\|$", text[max(0, start - 80) : start].rstrip())
    if before.endswith("|") and not before.endswith("||") and not closure:
        return True
    if re.search(r"\b(let|matches!\([^,]*,)\s*$", text[max(0, start - 60) : start]):
        return True
    if before.endswith("let") or before.endswith("Some(") and "let" in text[max(0, start - 40) : start]:
        return True
    return False


def scan():
    """Rows keyed by (section, variant): conditions, sites."""
    rows = defaultdict(lambda: {"conds": set(), "sites": [], "indirect": []})
    feature_refs = defaultdict(list)
    for tree, rel, text in source_files():
        spans = fn_spans(text)

        def site(pos):
            enc = enclosing(spans, pos)
            return f"{rel}:{line_of(text, pos)}" + (f" ({enc[2]})" if enc else "")

        # Unsupported constructors.
        for m in re.finditer(r"\bUnsupported::(" + "|".join(CONSTRUCTORS) + r")\s*\(", text):
            k = m.end() - 1
            e = close_of(text, k, "(", ")")
            args = split_args(text[k + 1 : e])
            kind = CONSTRUCTORS[m.group(1)]
            feat = args[0] if args else ""
            fm = re.match(r"(?:\w+::)*Feature::([A-Za-z0-9_]+)", feat)
            name = fm.group(1) if fm else None
            needs_text = " ".join(args[1:])
            if name is None:
                key = ("feature", "(computed)")
            else:
                key = ("feature", name)
            rows[key]["conds"] |= conditions(kind, needs_text)
            rows[key]["sites"].append(site(m.start()))
        # Unsupported { feature: .., needs: .. } literals.
        for m in re.finditer(r"\bUnsupported\s*\{", text):
            pre = text[max(0, m.start() - 12) : m.start()]
            if "struct" in pre or "enum" in pre:
                continue
            e = close_of(text, m.end() - 1, "{", "}")
            body = text[m.end() : e]
            fm = re.search(r"\bfeature\s*:\s*(?:\w+::)*Feature::([A-Za-z0-9_]+)", body)
            nm = re.search(r"\bneeds\s*:\s*(?:\w+::)*Needs::([A-Za-z0-9_]+)", body)
            if not re.search(r"\bfeature\s*:", body):
                continue  # a pattern (`Unsupported { feature, .. }`)
            if is_pattern(text, m.start(), e + 1):
                continue
            name = fm.group(1) if fm else "(computed)"
            kind = nm.group(1) if nm else "computed"
            needs_text = body[nm.start() :] if nm else ""
            rows[("feature", name)]["conds"] |= conditions(kind, needs_text)
            rows[("feature", name)]["sites"].append(site(m.start()))
        # Indirect feature references (a feature handed to a helper that builds the refusal).
        for m in re.finditer(r"\bFeature::([A-Za-z0-9_]+)", text):
            if is_pattern(text, m.start(), m.end()):
                continue
            feature_refs[m.group(1)].append(site(m.start()))
        if tree == "runtime":
            gap_alias = ["RuntimeGap"]
            am = re.search(r"\bRuntimeGap\s+as\s+(\w+)", text)
            if am:
                gap_alias.append(am.group(1))
            for alias in gap_alias:
                for m in re.finditer(r"\b" + alias + r"::([A-Z][A-Za-z0-9_]*)", text):
                    if rel.endswith("metal2vulkan-runtime/src/error.rs"):
                        continue
                    if is_pattern(text, m.start(), m.end()):
                        continue
                    section = "gap_family" if "/families/" in rel else "gap"
                    rows[(section, m.group(1))]["sites"].append(site(m.start()))
            for m in re.finditer(r"\bAppleOnly::([A-Z][A-Za-z0-9_]*)", text):
                if rel.endswith("apple_only.rs") and re.match(r"\s*[,{(]", text[m.end() : m.end() + 2]) is None:
                    pass
                if is_pattern(text, m.start(), m.end()):
                    continue
                rows[("apple", m.group(1))]["sites"].append(site(m.start()))
        for m in re.finditer(r"\bBindError::([A-Z][A-Za-z0-9_]*)", text):
            if m.group(1) not in BIND_REFUSALS or is_pattern(text, m.start(), m.end()):
                continue
            rows[("bind", m.group(1))]["sites"].append(site(m.start()))
        for m in re.finditer(r"\bLimit::([A-Z][A-Za-z0-9_]*)", text):
            if is_pattern(text, m.start(), m.end()):
                continue
            if rel.endswith("metal2vulkan-air/src/error.rs"):
                continue
            rows[("limit", m.group(1))]["sites"].append(site(m.start()))
        for enum, section in (("UnpinnedCompile", "ratemap"), ("BudgetExceeded", "budget")):
            for m in re.finditer(r"\b" + enum + r"::([A-Z][A-Za-z0-9_]*)", text):
                if is_pattern(text, m.start(), m.end()):
                    continue
                rows[(section, m.group(1))]["sites"].append(site(m.start()))
    return rows, feature_refs


def enum_variants(path, enum):
    """(variant, doc) of `pub enum <enum>` in path, in order."""
    lines = path.read_text().split("\n")
    out, doc, inside = [], [], False
    for line in lines:
        if not inside:
            if re.match(r"pub enum " + enum + r"\b", line):
                inside = True
            continue
        if line.startswith("}"):
            break
        s = line.strip()
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        m = re.match(r"^    ([A-Z][A-Za-z0-9_]*)\b", line)
        if m:
            out.append((m.group(1), " ".join(doc)))
        if s and not s.startswith("//") and not s.startswith("#"):
            doc = []
    return out


# A citation of the design record, a work package or an open item, which a reader of this
# file cannot follow: dropped from the summaries.
REF = r"(?:refining |DESIGN )?D\d+(?:\.\d+)?(?: (?:items?|part) \d+(?:(?:, | and )\d+)*)?|WP-[A-Z0-9]+[a-z]?|U\d+"


def strip_refs(text):
    text = re.sub(rf"\s*\((?:{REF})(?:[,;] (?:{REF}))*\)", "", text)
    text = re.sub(rf"[,;] (?:{REF})(?=[,;)])", "", text)
    return re.sub(rf"\((?:{REF})[,;] ", "(", text)


def summary(doc):
    doc = strip_refs(doc.strip())
    if not doc:
        return ""
    m = re.match(r"(.+?[.:])(\s|$)", doc)
    first = m.group(1) if m else doc
    first = first.rstrip(":").rstrip(".")
    if len(first) > 150:
        first = first[:147].rsplit(" ", 1)[0] + "..."
    return first.replace("|", "\\|")


def load_reach():
    reach = {}
    if REACH.exists():
        for line in REACH.read_text().split("\n"):
            if not line.strip() or line.startswith("#"):
                continue
            f = line.split("\t")
            while len(f) < 4:
                f.append("")
            reach[(f[0], f[1])] = (f[2], f[3])
    return reach


def sites_cell(sites):
    if not sites:
        return "0"
    first = sorted(set(sites))
    shown = ", ".join(f"`{s.split(' ')[0]}`" for s in first[:2])
    more = f" (+{len(first) - 2})" if len(first) > 2 else ""
    return f"{len(first)}: {shown}{more}"


def head_sha():
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "--short=9", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def build():
    rows, feature_refs = scan()
    reach = load_reach()
    fdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "error" / "mod.rs", "Feature"))
    for n, d in enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "Feature"):
        fdocs.setdefault(n, d)
    gdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-runtime" / "src" / "error.rs", "RuntimeGap"))
    adocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-runtime" / "src" / "apple_only.rs", "AppleOnly"))
    apple_gaps = apple_only_gaps(CRATES / "metal2vulkan-runtime" / "src" / "apple_only.rs")
    bdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "error" / "mod.rs", "BindError"))
    ldocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "Limit"))
    rdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan" / "src" / "bind" / "rate_map.rs", "UnpinnedCompile"))
    tdocs = OrderedDict(enum_variants(CRATES / "metal2vulkan-air" / "src" / "error.rs", "BudgetExceeded"))
    for n, refs in feature_refs.items():
        if n in fdocs and not rows[("feature", n)]["sites"]:
            rows[("feature", n)]["indirect"] = refs
    tables = OrderedDict()
    missing_reach = []
    sections = [
        ("feature", fdocs, "Translator and front-end refusals (`Unsupported`: a `Feature` and its `Needs`)"),
        ("gap", gdocs, "Runtime gaps (`Error::NotImplemented(RuntimeGap)`)"),
        ("apple", adocs, "Apple-only calls (`Error::AppleOnly`)"),
        ("bind", OrderedDict((n, bdocs.get(n, "")) for n in BIND_REFUSALS), "Binder refusals at a device limit (`BindError`)"),
        ("limit", ldocs, "Translation caps (`Error::LimitExceeded(Limit)`)"),
        (
            "ratemap",
            rdocs,
            "Rate maps the compile layer does not compile (`Error::RateMap(UnpinnedCompile)`)",
        ),
        ("budget", tdocs, "The translation deadline (`Error::Budget`; the runtime's `Error::Compile(Budget)`)"),
    ]
    for section, docs, title in sections:
        out = []
        for name, doc in docs.items():
            r = rows.get((section, name))
            fam = rows.get(("gap_family", name)) if section == "gap" else None
            sites = (r["sites"] if r else [])
            indirect = (r["indirect"] if r else [])
            conds = sorted(r["conds"]) if r else []
            if section == "gap":
                if name in ("FeatureEnable", "ExtensionEnable"):
                    conds = ["devices without the feature or extension it names"]
                else:
                    conds = ["every device (runtime piece not built)"]
            elif section == "apple":
                gap = apple_gaps.get(name, "?")
                conds = [f"every device (Apple-private, {APPLE_GAP_WORDS.get(gap, gap)})"]
            elif section in ("bind", "limit"):
                conds = ["at the device's limit" if section == "bind" else "every device (a deterministic cap)"]
            elif section == "ratemap":
                conds = ["every device (this crate's bound: Apple's compile unmeasured or crashing there)"]
            elif section == "budget":
                conds = ["every device (the attempt's deadline, 20 seconds by default)"]
            if not sites and not indirect and not (fam and fam["sites"]):
                state = "no site"
            elif not sites and indirect:
                state = "indirect"
            else:
                state = "direct"
            notsched = " (not scheduled)" if "not scheduled" in doc.lower() else ""
            rm = reach.get((section, name))
            if rm is None:
                missing_reach.append((section, name))
                rm = ("?", "")
            if sites:
                sc = sites_cell(sites)
            elif indirect:
                sc = "via a helper: " + sites_cell(indirect)
            elif fam and fam["sites"]:
                sc = "family audit only: " + sites_cell(fam["sites"])
            else:
                sc = "0"
            out.append(
                {
                    "name": name,
                    "what": summary(doc) + notsched,
                    "cond": "; ".join(conds) if conds else "-",
                    "sites": sc,
                    "state": state,
                    "reach": rm[0],
                    "why": rm[1].replace("|", "\\|"),
                    "size": "",
                    "all": sorted(set(sites or indirect or (fam["sites"] if fam else []))),
                }
            )
        tables[section] = (title, out)
    return tables, missing_reach


def render(tables):
    sha = head_sha()
    lines = []
    lines.append("# The refusal surface")
    lines.append("")
    lines.append(
        "Every refusal of something Metal accepts, read from the code: the construction sites of the "
        "typed refusals outside tests (`scripts/refusal-surface/refusal_surface.py`, which writes this "
        "file; edit its `reach.tsv`, never the tables). Metal's own validation errors (`Invalid`, "
        "`InvalidDescriptor`, malformed input) are misuse, not refusals, and are not listed. "
        "[`METAL-COVERAGE.md`](METAL-COVERAGE.md) lists the Metal API the runtime has no counterpart "
        "for at all."
    )
    lines.append("")
    lines.append("Columns:")
    lines.append("")
    lines.append(
        "- **What it blocks**: the variant's doc comment, first sentence, and \"not scheduled\" "
        "where the doc says so."
    )
    lines.append(
        "- **Where**: the device condition, from the constructor at its sites: `NotImplemented` and "
        "`NotRepresentable` refuse on every device; `needs` names the Vulkan feature, extension, "
        "property or limit a device lacks; a driver defect or deviation names the driver. A site that "
        "computes its capability says so."
    )
    lines.append(
        "- **Sites**: construction sites (file:line, the first two). \"via a helper\" counts the places "
        "that hand the feature to a function that builds the refusal; \"family audit only\" is a gap "
        "no call refuses, only `supportsFamily:`'s audit (the API is missing: see the coverage map); "
        "0 is a variant nothing builds today."
    )
    lines.append("- **Reach** (hand-kept, `reach.tsv`):")
    for k, v in REACH_MARKS.items():
        lines.append(f"  - `{k}`: {v};")
    lines.append("  - `?`: not assessed yet.")
    lines.append("")
    lines.append(f"Generated at `{sha}`.")
    lines.append("")
    # totals
    lines.append("## Totals")
    lines.append("")
    lines.append("| Kind | Variants | With a site | Reach system | Reach app | Reach rare | Reach none | Unassessed |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for section, (title, rows) in tables.items():
        c = defaultdict(int)
        for r in rows:
            c[r["reach"]] += 1
        live = sum(1 for r in rows if r["state"] != "no site")
        lines.append(
            f"| {title.split(' (')[0]} | {len(rows)} | {live} | {c['system']} | {c['app']} | {c['rare']} | {c['none']} | {c['?']} |"
        )
    lines.append("")
    for i, (section, (title, rows)) in enumerate(tables.items(), 1):
        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append("| Variant | What it blocks | Where | Sites | Reach |")
        lines.append("|---|---|---|---|---|")
        for r in rows:
            reach = f"`{r['reach']}`" + (f": {r['why']}" if r["why"] else "")
            lines.append(
                f"| `{r['name']}` | {r['what']} | {r['cond']} | {r['sites']} | {reach} |"
            )
        lines.append("")
    lines.append("## Appendix: every site")
    lines.append("")
    lines.append("Each refusal's sites, with the function that builds it.")
    lines.append("")
    for section, (title, rows) in tables.items():
        for r in rows:
            if not r["all"]:
                continue
            lines.append(f"- `{section}::{r['name']}`: " + "; ".join(f"`{x}`" for x in r["all"]))
    lines.append("")
    return "\n".join(lines) + "\n"


def plain_table(tables):
    """One Markdown table of every refusal, most reachable first, with short device conditions."""
    order = {k: i for i, k in enumerate(REACH_MARKS)}

    def short(c):
        parts = []
        for p in c.split("; "):
            p = p.replace("devices without one of ", "lacking one of ").replace("devices without ", "lacking ")
            p = p.replace("every device (not implemented)", "all devices: not built")
            p = p.replace("every device (no Vulkan mechanism)", "all devices: no Vulkan form")
            p = p.replace("every device (runtime piece not built)", "all devices: not built")
            p = re.sub(r"every device \(Apple-private, ([^)]*)\)", r"all devices (Apple-private): \1", p)
            p = re.sub(r"Vk(Feature|Extension|Limit|Property)::", "", p)
            parts.append(p)
        # Every device's refusal (not built, no Vulkan form) first, so a long list of device
        # conditions never cuts it off.
        parts.sort(key=lambda p: not p.startswith("all devices"))
        s = "; ".join(parts)
        return s if len(s) < 160 else s[:157] + "..."

    rows = [(order.get(x["reach"], len(order)), sec, x) for sec, (_, rs) in tables.items() for x in rs]
    rows.sort(key=lambda t: (t[0], t[1], t[2]["name"]))
    out = [
        f"# The refusal surface at `{head_sha()}`",
        "",
        "Reach: " + "; ".join(f"{k} = {v}" for k, v in REACH_MARKS.items()) + ".",
        "",
        "| Reach | Kind | Refusal | Blocks | Where | Sites |",
        "|---|---|---|---|---|---|",
    ]
    for _, sec, x in rows:
        n = x["sites"].split(" ")[0].rstrip(":")
        if x["sites"].startswith("via"):
            n = "via a helper"
        if x["sites"].startswith("family"):
            n = "family audit only"
        what = x["what"] if len(x["what"]) < 110 else x["what"][:107] + "..."
        out.append(f"| {x['reach']} | {sec} | {x['name']} | {what} | {short(x['cond'])} | {n} |")
    return "\n".join(out) + "\n"


# Why each Apple-only refusal is refused, from `AppleOnly::gap`.
APPLE_GAP_WORDS = {
    "Unmeasured": "Metal's answer not measured",
    "NotBuilt": "not built",
}


def apple_only_gaps(path):
    """Each `AppleOnly` variant's `AppleOnlyGap`, read from `AppleOnly::gap`'s match arms."""
    text = path.read_text()
    start = text.index("pub const fn gap(self) -> AppleOnlyGap")
    body = text[start:text.index("\n    }\n", start)]
    gaps = {}
    for arm in re.finditer(r"((?:\|?\s*Self::\w+\s*)+)=>\s*AppleOnlyGap::(\w+)", body):
        for name in re.findall(r"Self::(\w+)", arm.group(1)):
            gaps[name] = arm.group(2)
    return gaps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="fail when docs/REFUSALS.md is stale")
    ap.add_argument("--table", action="store_true", help="print one plain Markdown table, most reachable first")
    ap.add_argument("--missing-reach", action="store_true", help="list variants without a reach mark")
    a = ap.parse_args()
    tables, missing = build()
    if a.missing_reach:
        for s, n in missing:
            print(f"{s}\t{n}")
        return 0
    if a.table:
        sys.stdout.write(plain_table(tables))
        return 0
    text = render(tables)
    if a.check:
        # Structural: the variants, what each blocks, where, and its reach. Sites and line
        # numbers drift with every edit and are not checked.
        def rows(t):
            out = []
            for line in t.split("\n"):
                f = line.split(" | ")
                if line.startswith("| `") and len(f) == 5:
                    out.append((f[0], f[1], f[2], f[4]))
            return out

        old = OUT.read_text() if OUT.exists() else ""
        if rows(old) != rows(text):
            stale = sorted(set(rows(old)) ^ set(rows(text)))
            for r in stale[:10]:
                print("changed:", " | ".join(r)[:200], file=sys.stderr)
            print("docs/REFUSALS.md is stale: rerun scripts/refusal-surface/refusal_surface.py", file=sys.stderr)
            return 1
        if missing:
            print(f"{len(missing)} variants without a reach mark (--missing-reach)", file=sys.stderr)
            return 1
        return 0
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(ROOT)}; {len(missing)} variants without a reach mark", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
