# regen-vk-types

Generates `crates/metal2vulkan/src/vk/generated.rs`, the plain-data Vulkan enums and flags of the translator's `vk`
module, from Vulkan-Headers
`vulkan-sdk-1.4.341.0/registry/vk.xml`, the same pinned registry as
`scripts/regen-vulkan-requirements/`. The script verifies the registry SHA-256 before use.

Each listed registry `enum` becomes a `vk_enum!` invocation, each listed bitmask a `vk_flags!`
invocation, and each `SINGLE_BITS` entry a `vk_bit!` invocation; the macros (in `crates/metal2vulkan/src/vk/mod.rs`)
define transparent raw-value newtypes with named constants, `KNOWN` tables, registry-name lookup
(aliases included), and `Debug`. A `vk_bit!` type (e.g. `vk::ShaderStage`) is closed: its values
are exactly the single named bits of one bitmask, and raw construction is checked. Values come from
the core `<enums>` blocks and from what core versions and extensions add. Vulkan SC-only values,
beta (`protect`) values, and disabled or provisional extensions are excluded, so the output matches
the non-beta C header. Unknown registry forms and conflicting redefinitions fail generation.

The registry's `<formats>` element becomes one `vk_format_blocks!` invocation, which defines
`vk::Format::block`: per named format its `blockSize`, `texelsPerBlock`, compatibility `class`
(which a view's format must share), `packed` width, the bit width every component
shares (when one does), its color components (the `R`, `G`, `B` and `A` it has, which a color
write mask names), and the `numericFormat` of each aspect (color, depth, stencil; an sRGB
format's `UNORM` alpha belongs to its `SRGB` color numeric format, as the spec's numeric-format
table defines). The binder counts texel-buffer elements and aligns vertex
attributes with it, and `vk::Format::numeric_type` reads the numeric formats. The same invocation defines `vk::Format::exposure`: the lowest core version whose
`<feature>` defines the value and every extension that defines it or an alias of it, which says
where a device may use the format. Unknown numeric formats, component names, or aspect mixes fail
generation.

To add a type, append it to `ENUMS`, `FLAGS`, or `SINGLE_BITS` and regenerate. Never hand-edit
the output.

```sh
# regenerate (downloads the pinned registry, or use --registry PATH for an offline copy)
python3 scripts/regen-vk-types/regen_vk_types.py [--registry PATH]
# idempotence: fail if the committed file differs from a fresh generation
python3 scripts/regen-vk-types/regen_vk_types.py --check [--registry PATH]
# generator unit tests
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-vk-types
```

Requires Python 3 only (the output is already `rustfmt`-stable: macro bodies are left untouched).
