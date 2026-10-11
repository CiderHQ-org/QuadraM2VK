# regen-metal-enums

Generates `crates/metal2vulkan/src/metal/generated.rs`, the Metal enums and option sets of the translator's `metal`
module, and `crates/metal2vulkan-air/src/metal/generated.rs`, the ones the AIR front
end reads (`AIR_ENUMS`: `MTLDataType`, which `metal` re-exports), from the
macOS SDK's `Metal.framework` headers.

Only enumerator names and numeric values are extracted: they are the Metal API's wire contract
(guest command streams and Metal-shaped descriptors carry them as raw ordinals). No header text is
copied. Each listed `NS_ENUM` becomes a `metal_enum!` invocation (a closed Rust enum whose
`TryFrom<u32>` rejects unknown ordinals) and each listed `NS_OPTIONS` a `metal_flags!` invocation
(a bit set whose `TryFrom<u32>` rejects unknown bits); the macros live in `crates/metal2vulkan/src/metal/mod.rs` and,
for the front end's enums, `crates/metal2vulkan-air/src/metal/mod.rs`.
Deprecated spellings with the same value are recorded as aliases. Value expressions other than
integer literals, shifts, ors, and references to earlier enumerators fail generation.

The output records the SDK version it came from, so `--check` passes only against that SDK. To add
a type, append its Metal name to `ENUMS`, `AIR_ENUMS` or `OPTIONS` and regenerate. Never hand-edit
the output.

```sh
# regenerate from the active Xcode SDK (xcrun), or pass --headers DIR --sdk-version V
python3 scripts/regen-metal-enums/regen_metal_enums.py
# idempotence: fail if a committed file differs from a fresh generation
python3 scripts/regen-metal-enums/regen_metal_enums.py --check
# generator unit tests (no SDK needed)
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s scripts/regen-metal-enums
```

Requires macOS with Xcode (or the Command Line Tools SDK) and Python 3.
