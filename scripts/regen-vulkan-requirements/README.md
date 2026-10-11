# Vulkan requirement extraction

`registry.py` extracts SPIR-V capability and extension enablement alternatives
from Vulkan-Headers `vulkan-sdk-1.4.341.0/registry/vk.xml`. It verifies the complete
registry SHA-256 before processing it. Use `--registry PATH` with the same pinned
file for offline operation.

```sh
python3 scripts/regen-vulkan-requirements/registry.py crates/metal2vulkan/src/spirv/requirements/registry_generated.rs
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-vulkan-requirements
```

The generated table retains feature prerequisites, aliases, property predicates, and
alternative routes without interpreting them as unconditional permission. Unknown
XML rule forms, empty routes, and duplicate identities fail extraction.

The table is keyed and valued by name, and it is test-only: the core's requirement tests
(`crates/metal2vulkan/src/spirv/requirements/tests.rs`) compare its own vocabulary against the registry. It
spells features, extensions, and properties as the core's `VkFeature`, `VkExtension`, and
`VkProperty` do (`shaderInt64` -> `ShaderInt64`, `VK_KHR_x` -> `KhrX`). A feature
alternative carries only the feature; its exposure routes are the union listed by
`feature_exposure`. Declaration rules alone do not cover operation-specific storage
classes, stage restrictions, resource limits, or host protocols; those need structural
analysis.

The generated file's header carries the registry's copyright line and the MIT notice the
table is used under (vk.xml is licensed Apache-2.0 OR MIT); a registry whose license no
longer offers MIT fails generation.

Generation requires Python 3 and rustfmt.
