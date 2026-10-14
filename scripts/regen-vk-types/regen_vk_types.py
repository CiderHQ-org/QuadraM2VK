#!/usr/bin/env python3
"""Generate `crates/metal2vulkan/src/vk/generated.rs`: plain-data Vulkan enums, flags, and per-format texel blocks from
a pinned vk.xml.

Every value comes from the registry: core `<enums>` blocks plus the values that core versions and
non-provisional Vulkan extensions add. Values reserved for Vulkan SC only, beta (`protect`) values,
and disabled or provisional extensions are excluded, so the output matches the non-beta C header.
Unknown registry forms fail generation instead of being guessed.
"""
import argparse
import hashlib
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REGISTRY_TAG = "vulkan-sdk-1.4.341.0"
REGISTRY_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/"
    f"{REGISTRY_TAG}/registry/vk.xml"
)
REGISTRY_SHA256 = "006807a47518e9ab23ecf7f4c95d936aab477a20d27a9a7ac23cedefb7210178"
OUTPUT = Path(__file__).resolve().parents[2] / "crates" / "metal2vulkan" / "src" / "vk" / "generated.rs"

# Registry `enum` types, emitted as raw-value newtypes (`vk_enum!`).
ENUMS = [
    "VkBlendFactor",
    "VkBlendOp",
    "VkBorderColor",
    "VkCompareOp",
    "VkComponentSwizzle",
    "VkDescriptorType",
    "VkDriverId",
    "VkDynamicState",
    "VkFilter",
    "VkFormat",
    "VkFrontFace",
    "VkImageLayout",
    "VkImageTiling",
    "VkImageType",
    "VkImageViewType",
    "VkIndexType",
    "VkLogicOp",
    "VkPhysicalDeviceType",
    "VkPolygonMode",
    "VkPrimitiveTopology",
    "VkSamplerAddressMode",
    "VkSamplerMipmapMode",
    "VkSamplerReductionMode",
    "VkShaderFloatControlsIndependence",
    "VkStencilOp",
    "VkVertexInputRate",
]

# Registry bitmask types, emitted as raw-value flag newtypes (`vk_flags!`).
FLAGS = [
    "VkAccessFlags",
    "VkAccessFlags2",
    "VkBufferUsageFlags",
    "VkColorComponentFlags",
    "VkCullModeFlags",
    "VkDescriptorBindingFlags",
    "VkDescriptorSetLayoutCreateFlags",
    "VkFormatFeatureFlags",
    "VkFormatFeatureFlags2",
    "VkImageAspectFlags",
    "VkImageCreateFlags",
    "VkImageUsageFlags",
    "VkPipelineShaderStageCreateFlags",
    "VkPipelineStageFlags",
    "VkPipelineStageFlags2",
    "VkResolveModeFlags",
    "VkSampleCountFlags",
    "VkShaderStageFlags",
    "VkSubgroupFeatureFlags",
]

# Closed single-bit types: registry `FlagBits` whose value is exactly one named bit (`vk_bit!`),
# as (bits enums name, Rust type name, bitmask type the bit belongs to). Named combinations such
# as `VK_SHADER_STAGE_ALL_GRAPHICS` are not values of these types.
SINGLE_BITS = [
    ("VkShaderStageFlagBits", "ShaderStage", "VkShaderStageFlags"),
]

ENUM_ATTRS = {
    frozenset(["name", "value"]),
    frozenset(["name", "value", "comment"]),
    frozenset(["name", "bitpos"]),
    frozenset(["name", "bitpos", "comment"]),
    frozenset(["name", "alias"]),
    frozenset(["name", "alias", "comment"]),
    frozenset(["name", "alias", "deprecated"]),
    frozenset(["name", "alias", "deprecated", "comment"]),
    frozenset(["name", "value", "deprecated"]),
    frozenset(["name", "bitpos", "deprecated"]),
}
EXTENDS_ATTRS = {"name", "extends", "value", "bitpos", "offset", "extnumber", "dir", "alias",
                 "comment", "deprecated", "api", "protect"}

# The `<formats>` forms read for `vk_format_blocks!` (texel block facts per `VkFormat`).
FORMAT_ATTRS = {"name", "class", "blockSize", "texelsPerBlock", "blockExtent", "packed",
                "compressed", "chroma"}
FORMAT_CHILDREN = {"component", "plane", "spirvimageformat"}
COMPONENT_ATTRS = {"name", "bits", "numericFormat", "planeIndex"}

# The registry's `numericFormat` values (the Vulkan spec's "Interpretation of Numeric Format"
# table) and their `vk::NumericFormat` variants. An unknown value fails generation.
NUMERIC_FORMATS = {
    "UNORM": "Unorm",
    "SNORM": "Snorm",
    "USCALED": "Uscaled",
    "SSCALED": "Sscaled",
    "UINT": "Uint",
    "SINT": "Sint",
    "UFLOAT": "Ufloat",
    "SFLOAT": "Sfloat",
    "SFIXED5": "Sfixed5",
    "SRGB": "Srgb",
    "BOOL": "Bool",
}
# Component names by aspect: color (`R`, `G`, `B`, `A`), depth (`D`), and stencil (`S`).
COLOR_COMPONENTS = {"R", "G", "B", "A"}


def parse_version(number):
    """`"1.3"` -> `(1, 3)`."""
    match = re.fullmatch(r"([0-9]+)\.([0-9]+)", number)
    if not match:
        raise ValueError(f"unknown version number {number}")
    return (int(match.group(1)), int(match.group(2)))


def applies_to_vulkan(api):
    return api is None or "vulkan" in api.split(",")


def camel_words(name):
    return re.findall(r"[A-Z]+(?=[A-Z][a-z]|[0-9]|$)|[A-Z]?[a-z]+|[A-Z]+|[0-9]+", name)


def value_prefix(enums_name):
    """`VkPipelineStageFlagBits2` -> `VK_PIPELINE_STAGE_2_`, `VkFormat` -> `VK_FORMAT_`."""
    stem = enums_name[2:]
    match = re.fullmatch(r"(.*)FlagBits([0-9]*)", stem)
    if match:
        stem = match.group(1) + match.group(2)
    return "VK_" + "_".join(word.upper() for word in camel_words(stem)) + "_"


def const_name(value_name, prefix, is_bits):
    if not value_name.startswith(prefix):
        raise ValueError(f"{value_name} does not start with {prefix}")
    rest = value_name[len(prefix):]
    if is_bits:
        rest = re.sub(r"_BIT(?=_[A-Z]+$|$)", "", rest)
    if not rest:
        raise ValueError(f"empty constant name for {value_name}")
    if rest[0].isdigit():
        rest = prefix.rstrip("_").split("_")[-1] + "_" + rest
    rest = rest.upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9_]*", rest):
        raise ValueError(f"unexpected constant name {rest} from {value_name}")
    return rest


def rust_type_name(vk_name):
    if not vk_name.startswith("Vk"):
        raise ValueError(f"not a Vk type: {vk_name}")
    return vk_name[2:]


class Registry:
    def __init__(self, data):
        self.root = ET.fromstring(data)
        self.bitmasks = self._bitmasks()
        self.values = {}   # enums name -> {value name: int}
        self.order = {}    # enums name -> [value names] in registry order
        self.aliases = {}  # enums name -> {alias name: target name}
        self.kinds = {}    # enums name -> ("enum" | "bitmask", bitwidth)
        # enums name -> {value or alias name: (core versions, extension names)} that define it.
        self.origins = {}
        self._core_enums()
        self._extensions()

    def _bitmasks(self):
        result = {}
        for t in self.root.find("types"):
            if t.tag != "type" or t.attrib.get("category") != "bitmask":
                continue
            if not applies_to_vulkan(t.attrib.get("api")):
                continue
            if "alias" in t.attrib:
                continue
            name = t.find("name").text
            base = t.find("type").text
            bits = t.attrib.get("bitvalues") or t.attrib.get("requires")
            if base not in ("VkFlags", "VkFlags64"):
                raise ValueError(f"unknown bitmask base {base} for {name}")
            result[name] = (bits, 64 if base == "VkFlags64" else 32)
        return result

    def _add(self, enums, name, value):
        values = self.values.setdefault(enums, {})
        if name in values:
            if values[name] != value:
                raise ValueError(f"{name} redefined: {values[name]} vs {value}")
            return
        values[name] = value
        self.order.setdefault(enums, []).append(name)

    def _add_alias(self, enums, name, target):
        aliases = self.aliases.setdefault(enums, {})
        if aliases.get(name, target) != target:
            raise ValueError(f"alias {name} redefined")
        aliases[name] = target

    def _core_enums(self):
        for block in self.root.findall("enums"):
            name = block.attrib["name"]
            kind = block.attrib.get("type")
            if kind not in ("enum", "bitmask"):
                continue
            width = int(block.attrib.get("bitwidth", "32"))
            self.kinds[name] = (kind, width)
            self.values.setdefault(name, {})
            self.order.setdefault(name, [])
            for e in block:
                if e.tag in ("comment", "unused"):
                    continue
                if e.tag != "enum":
                    raise ValueError(f"unknown element {e.tag} in {name}")
                attrs = dict(e.attrib)
                if not applies_to_vulkan(attrs.pop("api", None)):
                    continue
                if frozenset(attrs) not in ENUM_ATTRS:
                    raise ValueError(f"unknown enum form in {name}: {e.attrib}")
                self._define(name, attrs, extnumber=None, origin=("core", (1, 0)))

    def _define(self, enums, attrs, extnumber, origin):
        name = attrs["name"]
        cores, extensions = self.origins.setdefault(enums, {}).setdefault(name, (set(), set()))
        kind, what = origin
        (cores if kind == "core" else extensions).add(what)
        if "alias" in attrs:
            self._add_alias(enums, name, attrs["alias"])
        elif "value" in attrs:
            self._add(enums, name, int(attrs["value"], 0))
        elif "bitpos" in attrs:
            self._add(enums, name, 1 << int(attrs["bitpos"]))
        elif "offset" in attrs:
            number = int(attrs.get("extnumber", extnumber or 0))
            if number <= 0:
                raise ValueError(f"offset without extension number: {name}")
            value = 1000000000 + (number - 1) * 1000 + int(attrs["offset"])
            if attrs.get("dir", "+") == "-":
                value = -value
            elif attrs.get("dir", "+") != "+":
                raise ValueError(f"unknown dir for {name}")
            self._add(enums, name, value)
        else:
            raise ValueError(f"enum without value: {attrs}")

    def _require_blocks(self, owner):
        for req in owner:
            if req.tag in ("deprecate", "remove"):
                continue
            if req.tag != "require":
                raise ValueError(f"unknown block {req.tag}")
            if not applies_to_vulkan(req.attrib.get("api")):
                continue
            yield req

    def _extend(self, req, extnumber, origin):
        for e in req:
            if e.tag != "enum" or "extends" not in e.attrib:
                continue
            attrs = dict(e.attrib)
            if set(attrs) - EXTENDS_ATTRS:
                raise ValueError(f"unknown extending enum form: {attrs}")
            if not applies_to_vulkan(attrs.pop("api", None)) or "protect" in attrs:
                continue
            enums = attrs.pop("extends")
            for key in ("comment", "deprecated"):
                attrs.pop(key, None)
            self._define(enums, attrs, extnumber, origin)

    def _extensions(self):
        for feature in self.root.findall("feature"):
            if not applies_to_vulkan(feature.attrib.get("api")):
                continue
            version = parse_version(feature.attrib["number"])
            for req in self._require_blocks(feature):
                self._extend(req, None, ("core", version))
        for ext in self.root.find("extensions"):
            supported = ext.attrib.get("supported", "")
            if "vulkan" not in supported.split(","):
                continue
            if ext.attrib.get("platform") == "provisional" or ext.attrib.get("provisional"):
                continue
            number = int(ext.attrib["number"])
            for req in self._require_blocks(ext):
                self._extend(req, number, ("extension", ext.attrib["name"]))

    def exposure(self, enums, name):
        """(lowest core version or None, sorted extension names) that define the canonical value
        `name` of `enums`, directly or through an alias of it."""
        cores, extensions = set(), set()
        for defined, (c, x) in self.origins.get(enums, {}).items():
            target = defined
            while target in self.aliases.get(enums, {}):
                target = self.aliases[enums][target]
            if target == name:
                cores |= c
                extensions |= x
        if not cores and not extensions:
            raise ValueError(f"{name} is defined by no core version or extension")
        return (min(cores) if cores else None, sorted(extensions))

    def entries(self, enums, is_bits):
        """(const, registry name, value) for canonical values, and (alias name, const) pairs."""
        if enums not in self.values:
            raise ValueError(f"unknown enums block {enums}")
        prefix = value_prefix(enums)
        names = {}
        rows = []
        seen = {}
        for name in self.order[enums]:
            value = self.values[enums][name]
            const = const_name(name, prefix, is_bits)
            if const in names:
                raise ValueError(f"colliding constant {const} in {enums}")
            names[const] = name
            if not is_bits and value in seen:
                raise ValueError(f"{name} duplicates {seen[value]} without alias in {enums}")
            seen[value] = name
            rows.append((const, name, value))
        by_name = {name: const for const, name, _ in rows}
        aliases = []
        for alias, target in sorted(self.aliases.get(enums, {}).items()):
            while target in self.aliases.get(enums, {}):
                target = self.aliases[enums][target]
            if target not in by_name:
                # Aliases of excluded (e.g. beta) values stay excluded.
                continue
            aliases.append((alias, by_name[target]))
        return rows, aliases


def render_value(value, width, is_bits):
    if is_bits:
        return f"0x{value:0{width // 4}x}"
    return str(value)


def render(registry):
    out = [
        "// Generated from Vulkan-Headers " + REGISTRY_TAG + "/registry/vk.xml.",
        "// Registry SHA-256: " + REGISTRY_SHA256,
        "// Regenerate with scripts/regen-vk-types/regen_vk_types.py; do not edit.",
        "",
    ]
    for vk_name in ENUMS:
        kind, _ = registry.kinds.get(vk_name, (None, None))
        if kind != "enum":
            raise ValueError(f"{vk_name} is not a registry enum")
        rows, aliases = registry.entries(vk_name, is_bits=False)
        out.append("vk_enum! {")
        out.append(f"    /// `{vk_name}`.")
        out.append(f'    {rust_type_name(vk_name)}(i32) = "{vk_name}" {{')
        for const, name, value in rows:
            out.append(f'        {const} = {value}, "{name}";')
        out.append("    }")
        out.append("    aliases {")
        for alias, const in aliases:
            out.append(f'        "{alias}" => {const};')
        out.append("    }")
        out.append("}")
        out.append("")
    for vk_name in FLAGS:
        if vk_name not in registry.bitmasks:
            raise ValueError(f"{vk_name} is not a registry bitmask")
        bits, width = registry.bitmasks[vk_name]
        if bits is None:
            raise ValueError(f"{vk_name} has no bit values")
        kind, bits_width = registry.kinds.get(bits, (None, None))
        if kind != "bitmask" or bits_width != width:
            raise ValueError(f"{bits} is not a {width}-bit bitmask")
        rows, aliases = registry.entries(bits, is_bits=True)
        raw = "u64" if width == 64 else "u32"
        out.append("vk_flags! {")
        out.append(f"    /// `{vk_name}` (bits: `{bits}`).")
        out.append(f'    {rust_type_name(vk_name)}({raw}) = "{vk_name}" {{')
        for const, name, value in rows:
            if value < 0 or value >= 1 << width:
                raise ValueError(f"{name} out of range")
            out.append(f'        {const} = {render_value(value, width, True)}, "{name}";')
        out.append("    }")
        out.append("    aliases {")
        for alias, const in aliases:
            out.append(f'        "{alias}" => {const};')
        out.append("    }")
        out.append("}")
        out.append("")
    for bits, rust_name, flags in SINGLE_BITS:
        if flags not in FLAGS:
            raise ValueError(f"{bits} belongs to {flags}, which is not generated")
        out.extend(render_single_bits(registry, bits, rust_name, flags))
    out.extend(render_format_blocks(registry))
    return "\n".join(out).rstrip("\n") + "\n"


def aspect_numeric_formats(name, components):
    """(color, depth, stencil) numeric formats of one `<format>` from its `(component name,
    numericFormat)` pairs, each a `vk::NumericFormat` variant name or None.

    An aspect's components share one numeric format, with one exception the Vulkan spec's
    numeric-format table defines: an `SRGB` format's `A` component is plain `UNORM` (the registry
    records it so), and the color aspect is `SRGB`. A format with both color and depth/stencil
    components, an unknown component name or numeric format, or any other mix fails generation.
    """
    by_aspect = {"color": [], "depth": [], "stencil": []}
    for component, numeric in components:
        if numeric not in NUMERIC_FORMATS:
            raise ValueError(f"{name}: unknown numericFormat {numeric}")
        if component in COLOR_COMPONENTS:
            by_aspect["color"].append((component, numeric))
        elif component == "D":
            by_aspect["depth"].append((component, numeric))
        elif component == "S":
            by_aspect["stencil"].append((component, numeric))
        else:
            raise ValueError(f"{name}: unknown component {component}")
    if by_aspect["color"] and (by_aspect["depth"] or by_aspect["stencil"]):
        raise ValueError(f"{name} mixes color and depth/stencil components")
    result = []
    for aspect in ("color", "depth", "stencil"):
        numerics = {numeric for _, numeric in by_aspect[aspect]}
        if not numerics:
            result.append(None)
            continue
        if numerics == {"SRGB", "UNORM"} and all(
            component == "A" for component, numeric in by_aspect[aspect] if numeric == "UNORM"
        ):
            numerics = {"SRGB"}
        if len(numerics) != 1:
            raise ValueError(f"{name}: {aspect} components mix {sorted(numerics)}")
        (numeric,) = numerics
        result.append(NUMERIC_FORMATS[numeric])
    return tuple(result)


def format_blocks(registry):
    """(const, blockSize, texelsPerBlock, class, packed bits or None, shared component bits or
    None, color components, color, depth, stencil numeric format or None, core version or None,
    extension names) for every canonical `VkFormat` value the registry's `<formats>` describes,
    in enum order.

    The class is the format's compatibility class (the spec's "Compatible Formats" table, the
    registry's `class`): a view's format must share its image's, so a format of the same block
    size may still be incompatible (`A8_UNORM`'s "8-bit alpha" against "8-bit").

    The exposure is where the enum value comes from: the lowest core version that defines it and
    every extension that defines it or an alias of it (`Registry.exposure`).

    The shared component width is the one bit count every `<component>` has, when there is one
    and it is a number (compressed formats and mixed widths such as `D24_UNORM_S8_UINT` have
    none). The color components are the names of its `R`, `G`, `B` and `A` `<component>`s, in
    that order (`""` for a depth or stencil format): those a color write mask's bits name. The
    numeric formats are each aspect's `numericFormat` ([`aspect_numeric_formats`]).
    A `<format>` of a value the enum excludes (beta, disabled) is skipped; one naming an alias, a
    duplicate, or an unknown attribute or child fails generation.
    """
    formats = registry.root.find("formats")
    if formats is None:
        raise ValueError("the registry has no <formats>")
    described = {}
    for f in formats:
        if f.tag != "format":
            raise ValueError(f"unknown element {f.tag} in <formats>")
        if set(f.attrib) - FORMAT_ATTRS:
            raise ValueError(f"unknown format form: {f.attrib}")
        name = f.attrib["name"]
        size = int(f.attrib["blockSize"])
        texels = int(f.attrib["texelsPerBlock"])
        if size <= 0 or texels <= 0:
            raise ValueError(f"{name}: empty texel block")
        cls = f.attrib.get("class")
        if not cls or '"' in cls or "\\" in cls:
            raise ValueError(f"{name}: no compatibility class, or one no string literal holds")
        packed = int(f.attrib["packed"]) if "packed" in f.attrib else None
        widths = set()
        numerics = []
        for c in f:
            if c.tag not in FORMAT_CHILDREN:
                raise ValueError(f"unknown element {c.tag} in {name}")
            if c.tag != "component":
                continue
            if set(c.attrib) - COMPONENT_ATTRS:
                raise ValueError(f"unknown component form in {name}: {c.attrib}")
            if "numericFormat" not in c.attrib:
                raise ValueError(f"{name}: a component without a numericFormat")
            widths.add(c.attrib["bits"])
            numerics.append((c.attrib["name"], c.attrib["numericFormat"]))
        if not widths:
            raise ValueError(f"{name} has no components")
        shared = None
        if len(widths) == 1:
            (width,) = widths
            if width != "compressed":
                shared = int(width)
        if name in described:
            raise ValueError(f"{name} described twice")
        named = {component for component, _ in numerics}
        components = "".join(c for c in "RGBA" if c in named)
        described[name] = (size, texels, cls, packed, shared, components,
                           *aspect_numeric_formats(name, numerics))
    rows, aliases = registry.entries("VkFormat", is_bits=False)
    for alias, _ in aliases:
        if alias in described:
            raise ValueError(f"<format> names the alias {alias}")
    return [
        (const, *described[name], *registry.exposure("VkFormat", name))
        for const, name, _ in rows
        if name in described
    ]


def render_format_blocks(registry):
    """One `vk_format_blocks!` invocation: `Format::block` and `Format::exposure` for every
    described format: its texel block, its color components, each aspect's numeric format, and
    where it is defined."""
    out = ["vk_format_blocks! {"]
    for (const, size, texels, cls, packed, shared, components, color, depth, stencil, core,
         extensions) in format_blocks(registry):
        row = f'    {const} = {size}, {texels}, class "{cls}"'
        if packed is not None:
            row += f", packed {packed}"
        if shared is not None:
            row += f", bits {shared}"
        if components:
            row += f", components [{' '.join(components)}]"
        for aspect, numeric in (("color", color), ("depth", depth), ("stencil", stencil)):
            if numeric is not None:
                row += f", {aspect} {numeric}"
        if core is not None:
            row += f", core {core[0]} {core[1]}"
        for extension in extensions:
            row += f', ext "{extension}"'
        out.append(row + ";")
    out.append("}")
    out.append("")
    return out


def render_single_bits(registry, bits, rust_name, flags):
    """One `vk_bit!` invocation: the single-bit values of `bits`, typed as one bit of `flags`."""
    if flags not in registry.bitmasks:
        raise ValueError(f"{flags} is not a registry bitmask")
    bits_of, width = registry.bitmasks[flags]
    if bits_of != bits:
        raise ValueError(f"{flags} has bits {bits_of}, not {bits}")
    rows, aliases = registry.entries(bits, is_bits=True)
    singles = {const for const, _, value in rows if value > 0 and value & (value - 1) == 0}
    if not singles:
        raise ValueError(f"{bits} has no single-bit values")
    if width != 32:
        # `vk_bit!` serializes through `u32` (its serde attribute names the raw type literally).
        raise ValueError(f"{bits} is not a 32-bit bitmask")
    raw = "u32"
    out = ["vk_bit! {",
           f"    /// Exactly one bit of `{flags}` (`{bits}`; named combinations excluded).",
           f'    {rust_name}({raw}) = "{bits}" in {rust_type_name(flags)} {{']
    for const, name, value in rows:
        if const in singles:
            out.append(f'        {const} = {render_value(value, width, True)}, "{name}";')
    out.append("    }")
    out.append("    aliases {")
    for alias, const in aliases:
        if const in singles:
            out.append(f'        "{alias}" => {const};')
    out.append("    }")
    out.append("}")
    out.append("")
    return out


def load(path):
    if path:
        data = Path(path).read_bytes()
    else:
        with urllib.request.urlopen(REGISTRY_URL, timeout=30) as response:
            data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != REGISTRY_SHA256:
        raise ValueError(f"registry SHA-256 mismatch: {actual}")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, help="offline copy of the pinned vk.xml")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--check", action="store_true",
                        help="fail if the committed output differs from a fresh generation")
    args = parser.parse_args()
    text = render(Registry(load(args.registry)))
    if args.check:
        if args.output.read_text() != text:
            print(f"{args.output} is stale; rerun without --check", file=sys.stderr)
            sys.exit(1)
        return
    args.output.write_text(text)


if __name__ == "__main__":
    main()
