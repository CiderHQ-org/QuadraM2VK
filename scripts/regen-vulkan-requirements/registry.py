#!/usr/bin/env python3
"""Extract Vulkan SPIR-V enablement rules from a verified, pinned registry."""
import argparse
import hashlib
import json
import subprocess
import textwrap
from pathlib import Path
import urllib.request
import xml.etree.ElementTree as ET

REGISTRY_URL = (
    "https://raw.githubusercontent.com/KhronosGroup/Vulkan-Headers/"
    "vulkan-sdk-1.4.341.0/registry/vk.xml"
)
REGISTRY_SHA256 = "006807a47518e9ab23ecf7f4c95d936aab477a20d27a9a7ac23cedefb7210178"
FORMS = {
    frozenset(["version"]),
    frozenset(["extension"]),
    frozenset(["struct", "feature", "requires"]),
    frozenset(["struct", "feature", "requires", "alias"]),
    frozenset(["property", "member", "value", "requires"]),
}


def extract(data):
    root = ET.fromstring(data)
    result = {}
    for group, child in [("spirvcapabilities", "spirvcapability"),
                         ("spirvextensions", "spirvextension")]:
        entries = root.find(group)
        if entries is None:
            raise ValueError(f"missing {group}")
        rules = {}
        for entry in entries:
            if entry.tag != child or set(entry.attrib) != {"name"}:
                raise ValueError(f"unknown {group} entry: {ET.tostring(entry)}")
            name = entry.attrib["name"]
            if not name or name in rules:
                raise ValueError(f"empty or duplicate {group} identity: {name}")
            alternatives = []
            for enable in entry:
                if enable.tag != "enable" or frozenset(enable.attrib) not in FORMS:
                    raise ValueError(f"unknown enablement rule for {name}: {enable.attrib}")
                if list(enable) or any(not value for value in enable.attrib.values()):
                    raise ValueError(f"invalid enablement rule for {name}")
                alternatives.append(dict(sorted(enable.attrib.items())))
            if not alternatives:
                raise ValueError(f"no enablement routes for {name}")
            rules[name] = alternatives
        result[group] = dict(sorted(rules.items()))
    return result


# The MIT license's permission notice, as Vulkan-Headers' LICENSES/MIT.txt words it: vk.xml is
# licensed under Apache-2.0 OR MIT, and the generated table uses it under the MIT license.
MIT_PERMISSION = [
    "Permission is hereby granted, free of charge, to any person obtaining a copy of this software "
    "and associated documentation files (the \"Software\"), to deal in the Software without "
    "restriction, including without limitation the rights to use, copy, modify, merge, publish, "
    "distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom "
    "the Software is furnished to do so, subject to the following conditions:",
    "The above copyright notice and this permission notice shall be included in all copies or "
    "substantial portions of the Software.",
    "THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, "
    "INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR "
    "PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE "
    "FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR "
    "OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER "
    "DEALINGS IN THE SOFTWARE.",
]


def header(data):
    """The generated table's header: its source, and the registry's copyright and MIT notice.

    The registry's first comment carries its copyright line and SPDX license expression; a
    registry whose expression no longer offers MIT fails, so the notice cannot go stale.
    """
    comment = ET.fromstring(data).find("comment")
    text = [line.strip() for line in (comment.text if comment is not None else "").splitlines()]
    copyright_ = [line for line in text if line.startswith("Copyright ")]
    licence = [line for line in text if line.startswith("SPDX-License-Identifier:")]
    if len(copyright_) != 1 or licence != ["SPDX-License-Identifier: Apache-2.0 OR MIT"]:
        raise ValueError(f"the registry's copyright or license changed: {copyright_} {licence}")
    lines = ["Generated from Vulkan-Headers vulkan-sdk-1.4.341.0/registry/vk.xml.",
             f"Registry SHA-256: {REGISTRY_SHA256}",
             "Regenerate with scripts/regen-vulkan-requirements/registry.py; do not edit.",
             "",
             "vk.xml is licensed under Apache-2.0 OR MIT; this table uses it under the MIT license:",
             "",
             copyright_[0]]
    for paragraph in MIT_PERMISSION:
        lines += [""] + textwrap.wrap(paragraph, width=97)
    return "".join(f"// {line}".rstrip() + "\n" for line in lines)

def rust_name(value):
    if value.startswith("VK_"):
        return "".join(word.title() for word in value[3:].split("_"))
    return value[0].upper() + value[1:]


def feature_routes_of(rules):
    enables = [enable for group in rules.values() for routes in group.values()
               for enable in routes]
    feature_routes = {"fragmentStoresAndAtomics": {"VK_VERSION_1_0"},
                      "vertexPipelineStoresAndAtomics": {"VK_VERSION_1_0"}}
    for e in enables:
        if "feature" in e:
            feature_routes.setdefault(e.get("alias", e["feature"]), set()).update(e["requires"].split(","))
    for storage in ["Buffer", "Shared", "Image"]:
        for width in ([32] if storage == "Image" else [16, 32, 64]):
            feature_routes[f"shader{storage}Float{width}Atomics"] = {
                "VK_EXT_shader_atomic_float2" if width == 16 else "VK_EXT_shader_atomic_float"
            }
    return enables, feature_routes


def property_name(member, value):
    return rust_name(member) + ("" if value == "VK_TRUE" else rust_name(value))


def render_by_name(rules):
    """Render the enablement rules keyed and valued by name, for the core's requirement tests.

    Names use the Rust spelling of the core's vocabulary (`shaderInt64` -> `ShaderInt64`,
    `VK_KHR_x` -> `KhrX`), so they compare equal to the `Debug` names of its `VkFeature` /
    `VkExtension` / `VkProperty`. A feature alternative carries only the feature: its exposure
    routes are the union in `feature_exposure`.
    """
    _, feature_routes = feature_routes_of(rules)

    def exposure(route):
        if route.startswith("VK_VERSION_"):
            major, minor = route.removeprefix("VK_VERSION_").split("_")
            return f"Exposure::Core({int(major)}, {int(minor)})"
        if not route.startswith("VK_") or not all(c.isalnum() or c == "_" for c in route):
            raise ValueError(f"unknown prerequisite expression: {route}")
        return f"Exposure::Extension({json.dumps(rust_name(route))})"

    lines = ["#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
             "pub(crate) enum Exposure { Core(u8, u8), Extension(&'static str) }",
             "#[derive(Clone, Copy, Debug, PartialEq, Eq)]",
             "pub(crate) enum Enablement {",
             "    Exposed(Exposure),",
             "    Feature(&'static str),",
             "    Property(&'static str),",
             "}",
             "pub(crate) fn feature_exposure(feature: &str) -> Option<&'static [Exposure]> {",
             "    Some(match feature {"]
    for name, routes in sorted(feature_routes.items()):
        values = ", ".join(exposure(route) for route in sorted(routes))
        lines.append(f"        {json.dumps(rust_name(name))} => &[{values}],")
    lines += ["        _ => return None,", "    })", "}"]
    for group, function in [("spirvcapabilities", "capability_enablement"),
                            ("spirvextensions", "extension_enablement")]:
        lines += [f"pub(crate) fn {function}(name: &str) -> Option<&'static [Enablement]> {{",
                  "    Some(match name {"]
        for name, routes in rules[group].items():
            lines.append(f"        {json.dumps(name)} => &[")
            for enable in routes:
                if "version" in enable or "extension" in enable:
                    value = exposure(enable.get("version", enable.get("extension")))
                    expression = f"Enablement::Exposed({value})"
                elif "feature" in enable:
                    feature = rust_name(enable.get("alias", enable["feature"]))
                    expression = f"Enablement::Feature({json.dumps(feature)})"
                else:
                    prop = property_name(enable["member"], enable["value"])
                    expression = f"Enablement::Property({json.dumps(prop)})"
                lines.append(f"            {expression},")
            lines.append("        ],")
        lines += ["        _ => return None,", "    })", "}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path,
                        help="the name-keyed table (crates/metal2vulkan/src/spirv/requirements/registry_generated.rs)")
    parser.add_argument("--registry", type=Path, help="offline copy of the pinned vk.xml")
    args = parser.parse_args()
    if args.registry:
        data = args.registry.read_bytes()
    else:
        with urllib.request.urlopen(REGISTRY_URL, timeout=30) as response:
            data = response.read()
    actual = hashlib.sha256(data).hexdigest()
    if actual != REGISTRY_SHA256:
        raise ValueError(f"registry SHA-256 mismatch: {actual}")
    rules = extract(data)
    config = str(Path(__file__).resolve().parents[2] / "rustfmt.toml")
    args.output.write_text(header(data) + render_by_name(rules))
    subprocess.run(["rustfmt", "--config-path", config, str(args.output)], check=True)


if __name__ == "__main__":
    main()
