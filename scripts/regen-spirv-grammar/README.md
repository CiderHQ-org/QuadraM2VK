# regen-spirv-grammar

Generates the translator's SPIR-V grammar items from **Khronos's machine-readable SPIR-V grammar**
(SPIRV-Headers, MIT-licensed) with our own generator, `generate.py`. Nothing else feeds it.

| Output | Role |
|---|---|
| `crates/metal2vulkan/src/spv.rs` | The opcodes (`Op`), enumerants (`Capability`, `Decoration`, `BuiltIn`, ...), masks and `GlslStd450Op` the translator names, with their grammar values |
| `crates/metal2vulkan/src/spirv_binary/table.rs` | Test only: every core instruction's operands and every enumerant's parameters, which the test parser (`spirv_binary/`) reads |

## What `spv.rs` holds

Only what `items.txt` lists: each kind the code names, the values it names, and `from_u32` where
the code converts words back. Values that only tests name are `cfg(test)`. The compiler checks the
list: a value the code names that is missing fails to build, and a listed value nothing uses is
dead code where rustc can see it. When code needs a new opcode or enumerant, add it to
`items.txt` and regenerate.

Names follow the grammar's, with a few changes Rust's and Clippy's naming rules need (see
`variant_name` and `const_name` in `generate.py`): an opcode drops its `Op` prefix, leading digits
move to the end (`Dim::D2`), a name of four or more capitals only is written with one capital
(`Capability::Fmakhr`), `SpecConstantOp` is `SpecConstantOperation`, and a mask's bits are
capitalized consts (`FPFastMathMode::NOT_NAN`). `Debug` always prints the grammar's own name.

## Revision

SPIRV-Headers tag `vulkan-sdk-1.4.341.0` (SPIR-V 1.6 revision 4; GLSL.std.450 version 100
revision 2). `HEADERS_REF=<tag>` selects another. Every generated file's header names the source,
its revision and Khronos's MIT notice.

## Run

```sh
./scripts/regen-spirv-grammar/regen-spirv-grammar.sh
cargo nextest run -p metal2vulkan --all-features
```

The script fetches `spirv.core.grammar.json`, `extinst.glsl.std.450.grammar.json` and `LICENSE`
once per tag into `.cache/spirv-headers/<tag>/` (gitignored; `METAL2VULKAN_GRAMMAR_CACHE` names
another directory), runs `generate.py` and formats both outputs with `rustfmt`. It needs `curl`,
Python 3 and `rustfmt`, and network access for a tag not yet fetched.
