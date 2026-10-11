#!/usr/bin/env python3
"""Write the translator's SPIR-V grammar module and its test parser's table from Khronos's
machine-readable SPIR-V grammar (SPIRV-Headers), and nothing else.

usage: generate.py --grammar DIR --tag TAG --items FILE --spv OUT --table OUT

DIR holds SPIRV-Headers' `spirv.core.grammar.json`, `extinst.glsl.std.450.grammar.json` and
`LICENSE` at TAG. FILE (items.txt) names the items the translator uses; see its header. The two
outputs are Rust sources that `rustfmt` then formats (regen-spirv-grammar.sh does both).
"""

import argparse
import json
import re
import sys
from pathlib import Path

CORE = "spirv.core.grammar.json"
GLSL = "extinst.glsl.std.450.grammar.json"
HEADERS = "https://github.com/KhronosGroup/SPIRV-Headers"
HEADERS_DIR = "include/spirv/unified1"

# The operand kinds of the grammar's `Id`, `Literal` and `Composite` categories, as the test
# parser's table names them (`spirv_binary/grammar.rs`'s `Kind`). A kind a newer grammar adds stops
# the generator until the parser learns it.
PARSER_KINDS = {
    "IdResultType": "ResultType",
    "IdResult": "ResultId",
    "IdRef": "Id",
    "IdScope": "Scope",
    "IdMemorySemantics": "Semantics",
    "LiteralInteger": "Word",
    "LiteralFloat": "Word",
    "LiteralString": "Text",
    "LiteralContextDependentNumber": "Number",
    "LiteralExtInstInteger": "ExtInstNumber",
    "LiteralSpecConstantOpInteger": "SpecOp",
    "PairLiteralIntegerIdRef": "SwitchTarget",
    "PairIdRefLiteralInteger": "IdWord",
    "PairIdRefIdRef": "IdId",
}
QUANTIFIERS = {None: "One", "?": "Optional", "*": "Any"}


def fail(message):
    sys.exit(f"generate.py: {message}")


def number(value):
    return int(value, 0) if isinstance(value, str) else value


def variant_name(kind, name):
    """The Rust name of the enumerant `name` of a value-enum operand kind `kind`.

    The grammar's name, except that leading digits move to the end (`Dim`'s `2D` is `D2`), a name
    of four or more capitals only is written with one (`GLSL` is `Glsl`), and an opcode's `Op` at
    the end, which would repeat the enum's name, is spelled out (`SpecConstantOp` is
    `SpecConstantOperation`): the forms Rust's and Clippy's naming rules accept."""
    moved = re.fullmatch(r"(\d+)(\w+)", name)
    if moved:
        name = moved.group(2) + moved.group(1)
    if kind == "Op" and name.endswith("Op"):
        name = name[: -len("Op")] + "Operation"
    if re.fullmatch(r"[A-Z]{4,}", name):
        name = name[0] + name[1:].lower()
    return name


def const_name(name):
    """The Rust name of the bit `name` of a bit-enum operand kind: its words in capitals, joined
    by underscores (`AllowRecip` is `ALLOW_RECIP`, `NotNaN` is `NOT_NAN`, `NSZ` stays)."""
    words = re.findall(r"NaN|[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+", name)
    return "_".join(word.upper() for word in words)


class Item:
    """One type of the grammar module: its enumerants (Rust name, grammar name, value, parameter
    kinds), whether it is a mask, and what items.txt asks of it."""

    def __init__(self, name, mask, enumerants, aliases):
        self.name = name
        self.mask = mask
        self.enumerants = enumerants
        self.aliases = aliases
        self.test = False
        self.from_u32 = None
        self.every = {}
        self.test_only = {}

    def rust(self, grammar_name):
        return const_name(grammar_name) if self.mask else variant_name(self.name, grammar_name)


def load_grammar(directory):
    core = json.loads((directory / CORE).read_text())
    glsl = json.loads((directory / GLSL).read_text())
    return core, glsl


def grammar_items(core, glsl):
    """Every type the grammar module can hold, by Rust name."""
    items = {}

    def enum_of(kind):
        aliases = {}
        enumerants = []
        for e in kind["enumerants"]:
            params = [p["kind"] for p in e.get("parameters", [])]
            enumerants.append((e["enumerant"], number(e["value"]), params))
            for alias in e.get("aliases", []):
                aliases[alias] = e["enumerant"]
        return enumerants, aliases

    for kind in core["operand_kinds"]:
        if kind["category"] in ("ValueEnum", "BitEnum"):
            enumerants, aliases = enum_of(kind)
            items[kind["kind"]] = Item(
                kind["kind"], kind["category"] == "BitEnum", enumerants, aliases
            )

    def instructions(name, insts, strip):
        aliases = {}
        enumerants = []
        for inst in insts:
            enumerants.append((inst["opname"][strip:], inst["opcode"], []))
            for alias in inst.get("aliases", []):
                aliases[alias[strip:]] = inst["opname"][strip:]
        items[name] = Item(name, False, enumerants, aliases)

    instructions("Op", core["instructions"], len("Op"))
    instructions("GlslStd450Op", glsl["instructions"], 0)
    return items


def read_items(path, items):
    """Apply items.txt to `items`; returns the selected items in file order and the scalar items
    (`Word`, `MAGIC_NUMBER`) it names."""
    chosen = []
    scalars = {}
    current = None
    for number_, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].split()
        if not line:
            continue
        where = f"{path.name}:{number_}"
        key, args = line[0], line[1:]
        if key in ("type", "const"):
            if [key] + args[:1] not in (["type", "Word"], ["const", "MAGIC_NUMBER"]):
                fail(f"{where}: no scalar item {' '.join(line)}")
            if args[1:] not in ([], ["test"]):
                fail(f"{where}: `{key} {args[0]} [test]`")
            scalars[args[0]] = args[1:] == ["test"]
            continue
        if key == "enum":
            if not args or args[0] not in items or args[1:] not in ([], ["test"]):
                fail(f"{where}: `enum <operand kind> [test]`, not {' '.join(args)}")
            current = items[args[0]]
            if current in chosen:
                fail(f"{where}: {current.name} twice")
            current.test = args[1:] == ["test"]
            chosen.append(current)
            continue
        if current is None:
            fail(f"{where}: {key} before any enum")
        if key == "from_u32":
            if args not in ([], ["test"]) or current.mask:
                fail(f"{where}: `from_u32 [test]` on a value enum")
            current.from_u32 = "test" if args else "every"
            continue
        if key not in ("values", "test-values"):
            fail(f"{where}: unknown line {key}")
        by_rust = {current.rust(e[0]): e for e in current.enumerants}
        target = current.every if key == "values" else current.test_only
        for name in args:
            if name == "*":
                for rust, e in by_rust.items():
                    if rust not in current.every and rust not in current.test_only:
                        target[rust] = e
                continue
            if name not in by_rust:
                primary = current.aliases.get(name)
                hint = f" (the grammar's alias of {current.rust(primary)})" if primary else ""
                fail(f"{where}: {current.name} has no {name}{hint}")
            if name in current.every or name in current.test_only:
                fail(f"{where}: {current.name}::{name} twice")
            target[name] = by_rust[name]
    for item in chosen:
        if not item.every and not item.test_only:
            fail(f"{item.name} names no value")
        if item.from_u32:
            # The conversion sees every value of the grammar in each build it exists in.
            missing = len(item.enumerants) - len(item.every)
            if item.from_u32 == "test":
                missing -= len(item.test_only)
            if missing:
                fail(f"{item.name}'s from_u32 needs every enumerant in its builds ({missing} left)")
        values = {}
        for rust, (_, value, _) in {**item.every, **item.test_only}.items():
            if value in values:
                fail(f"{item.name}::{rust} and {values[value]} share the value {value}")
            values[value] = rust
    return chosen, scalars


def notice(core, glsl, licence, tag):
    """The generated files' header: the source, its revision, and Khronos's MIT notice."""
    text = licence.read_text()
    match = re.search(r"(Permission is hereby granted.*?MATERIALS\.)\n", text, re.S)
    if not match:
        fail("LICENSE carries no MIT permission notice")
    lines = [
        "Generated by scripts/regen-spirv-grammar from Khronos's machine-readable SPIR-V grammar.",
        "Do not edit: change items.txt or the generator and regenerate.",
        "",
        f"Source: {HEADERS}, tag {tag}:",
        f"  {HEADERS_DIR}/{CORE}"
        f" (SPIR-V {core['major_version']}.{core['minor_version']}, revision {core['revision']})",
        f"  {HEADERS_DIR}/{GLSL}"
        f" (version {glsl['version']}, revision {glsl['revision']})",
        "",
        "The grammar is licensed under the MIT license; its notice:",
        "",
    ]
    copyright_ = [line.rstrip() for line in core["copyright"]]
    while copyright_ and not copyright_[-1]:
        copyright_.pop()
    lines += copyright_ + [""] + match.group(1).splitlines()
    return "".join(f"// {line}".rstrip() + "\n" for line in lines)


def cfg(test, indent=""):
    """The attribute that keeps an item to test builds, when `test`."""
    return f"{indent}#[cfg(test)]\n" if test else ""


def ordered(item):
    """The item's chosen enumerants by value: (value, Rust name, grammar name, test-only)."""
    return sorted(
        [(e[1], rust, e[0], False) for rust, e in item.every.items()]
        + [(e[1], rust, e[0], True) for rust, e in item.test_only.items()]
    )


def emit_value_enum(item, out):
    rows = ordered(item)
    renamed = any(rust != grammar for _, rust, grammar, _ in rows)
    derive = "Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord"
    if not renamed:
        derive = "Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord"
    out.append(f"/// SPIR-V's `{item.name}`, by its grammar values.\n")
    out.append(cfg(item.test))
    out.append(f"#[derive({derive})]\n#[repr(u32)]\npub enum {item.name} {{\n")
    for value, rust, _, test in rows:
        out.append(cfg(test, "    ") + f"    {rust} = {value},\n")
    out.append("}\n\n")
    if renamed:
        # Debug prints the grammar's name, as a derived Debug does where the two names agree.
        out.append(cfg(item.test))
        out.append(f"impl std::fmt::Debug for {item.name} {{\n")
        out.append("    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {\n")
        out.append("        f.write_str(match self {\n")
        for _, rust, grammar, test in rows:
            out.append(cfg(test, "            ") + f"            Self::{rust} => {json.dumps(grammar)},\n")
        out.append("        })\n    }\n}\n\n")
    if item.from_u32:
        out.append(cfg(item.test))
        out.append(f"impl {item.name} {{\n")
        out.append("    /// The value whose word is `word`, if the grammar defines one.\n")
        out.append(cfg(item.from_u32 == "test" and not item.test, "    "))
        out.append("    pub fn from_u32(word: u32) -> Option<Self> {\n")
        out.append("        Some(match word {\n")
        for value, rust, _, test in rows:
            out.append(cfg(test, "            ") + f"            {value} => Self::{rust},\n")
        out.append("            _ => return None,\n        })\n    }\n}\n\n")


def emit_mask(item, out):
    if item.from_u32:
        fail(f"{item.name} is a mask")
    out.append(f"/// SPIR-V's `{item.name}` mask, with the bits the translator names.\n")
    out.append(cfg(item.test))
    out.append("#[derive(Clone, Copy, Debug, PartialEq, Eq)]\n")
    out.append(f"pub struct {item.name}(u32);\n\n")
    out.append(cfg(item.test))
    out.append(f"impl {item.name} {{\n")
    for value, rust, _, test in ordered(item):
        out.append(cfg(test, "    ") + f"    pub const {rust}: Self = Self({value:#06x});\n")
    out.append("\n    /// The mask's word.\n")
    out.append("    pub const fn bits(self) -> u32 {\n        self.0\n    }\n}\n\n")
    out.append(cfg(item.test))
    out.append(f"impl std::ops::BitOr for {item.name} {{\n    type Output = Self;\n\n")
    out.append("    fn bitor(self, other: Self) -> Self {\n        Self(self.0 | other.0)\n")
    out.append("    }\n}\n\n")


def emit_spv(header, core, chosen, scalars):
    out = [header, "\n"]
    out.append(
        "//! The SPIR-V grammar's items the translator names (opcodes, enumerants, masks), with their\n"
        "//! grammar values. `scripts/regen-spirv-grammar` generates this module from Khronos's grammar,\n"
        "//! with only what its `items.txt` lists; what only tests name is `cfg(test)`.\n\n"
    )
    if "Word" in scalars:
        out.append("/// One 32-bit word of a SPIR-V module.\n" + cfg(scalars["Word"]))
        out.append("pub type Word = u32;\n\n")
    if "MAGIC_NUMBER" in scalars:
        magic = number(core["magic_number"])
        out.append(
            "/// The first word of every SPIR-V module.\n"
            + cfg(scalars["MAGIC_NUMBER"])
            + f"pub const MAGIC_NUMBER: u32 = 0x{magic >> 16:04x}_{magic & 0xffff:04x};\n\n"
        )
    for item in chosen:
        (emit_mask if item.mask else emit_value_enum)(item, out)
    return "".join(out).rstrip("\n") + "\n"


def emit_table(header, core, items):
    """The test parser's table: every core instruction's operands, and every enumerant (with its
    parameters) of each operand kind they reach."""
    kinds = {k["kind"]: k for k in core["operand_kinds"]}
    reached = []
    named = set()

    def kind_expr(name):
        kind = kinds.get(name)
        if kind is None:
            fail(f"unknown operand kind {name}")
        if kind["category"] in ("ValueEnum", "BitEnum"):
            if name not in reached:
                reached.append(name)
            wrap = "Mask" if kind["category"] == "BitEnum" else "Value"
            named.add(wrap)
            return f"{wrap}(&{static_name(name)})"
        if name not in PARSER_KINDS:
            fail(f"operand kind {name} ({kind['category']}) is new to the test parser")
        named.add(PARSER_KINDS[name])
        return PARSER_KINDS[name]

    forms = []
    counts = set()
    for inst in sorted(core["instructions"], key=lambda i: i["opcode"]):
        operands = ", ".join(
            f"({kind_expr(o['kind'])}, {QUANTIFIERS[o.get('quantifier')]})"
            for o in inst.get("operands", [])
        )
        counts.update(QUANTIFIERS[o.get("quantifier")] for o in inst.get("operands", []))
        forms.append(
            f"    Form {{\n        opcode: Op::{variant_name('Op', inst['opname'][2:])},\n"
            f"        operands: &[{operands}],\n    }},\n"
        )
    statics = []
    # `reached` grows as the enumerants' parameters name further kinds.
    at = 0
    while at < len(reached):
        name = reached[at]
        at += 1
        kind = kinds[name]
        values = [number(e["value"]) for e in kind["enumerants"]]
        if len(set(values)) != len(values):
            fail(f"{name} repeats a value")
        rows = []
        for e in sorted(kind["enumerants"], key=lambda e: number(e["value"])):
            value = number(e["value"])
            if kind["category"] == "BitEnum" and value & (value - 1):
                fail(f"{name}'s {e['enumerant']} is not one bit")
            params = ", ".join(kind_expr(p["kind"]) for p in e.get("parameters", []))
            rows.append(
                f"    Enumerant {{\n        value: {value:#x},\n"
                f"        parameters: &[{params}],\n    }},\n"
            )
        statics.append(
            f"static {static_name(name)}: [Enumerant; {len(rows)}] = [\n{''.join(rows)}];\n\n"
        )
    return (
        header
        + "\n//! The core grammar's instructions and enumerants, as the test parser reads them.\n\n"
        + f"use super::grammar::Count::{{{', '.join(sorted(counts))}}};\n"
        + f"use super::grammar::Kind::{{{', '.join(sorted(named))}}};\n"
        + "use super::grammar::{Enumerant, Form};\nuse crate::spv::Op;\n\n"
        + "/// Every core instruction's form, by opcode.\n"
        + f"pub(super) static FORMS: [Form; {len(forms)}] = [\n{''.join(forms)}];\n\n"
        + "".join(statics).rstrip("\n")
        + "\n"
    )


def static_name(kind):
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", kind).upper()


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grammar", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--spv", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    core, glsl = load_grammar(args.grammar)
    header = notice(core, glsl, args.grammar / "LICENSE", args.tag)
    items = grammar_items(core, glsl)
    chosen, scalars = read_items(args.items, items)
    args.spv.write_text(emit_spv(header, core, chosen, scalars))
    args.table.write_text(emit_table(header, core, items))
    print(f"wrote {args.spv} and {args.table}")


if __name__ == "__main__":
    main()
