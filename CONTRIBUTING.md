# Contributing

How to build, test and change metal2vulkan. [AGENTS.md](AGENTS.md) holds the project's rules;
they bind people too.

## Prerequisites

- Rust stable, at least the `rust-version` in the crates' manifests (1.98).
- [cargo-nextest](https://nexte.st): `cargo install cargo-nextest --locked`.
- The external tools the tests and validation use (the libraries run none):
  - `llvm-as` (LLVM 21 or newer) for the `.ll` fixtures, and `llvm-dis` for validation's harvest
    and fixture tooling;
  - `spirv-val` and `spirv-as` from SPIRV-Tools;
  - `xcrun metal-as` on macOS, for the `.air.ll` fixtures (tests that need it skip elsewhere);
  - `spirv-cross`, for the checks of the MSL MoltenVK would write. Without it they print `SKIPPED`;
    `METAL2VULKAN_TEST_REQUIRE_SPIRV_CROSS=1` makes its absence a failure.

  Each is found on `PATH` or through `METAL2VULKAN_<TOOL>` (`METAL2VULKAN_LLVM_AS`,
  `METAL2VULKAN_SPIRV_VAL`, ...). With Homebrew on macOS, LLVM is keg-only:
  `PATH=/opt/homebrew/opt/llvm/bin:$PATH`.
- A Vulkan driver or a Mac with Metal only for the device tests and authored-case runs
  ([docs/VALIDATION.md](docs/VALIDATION.md)).

## Layout

```text
.
├── crates/
│   ├── metal2vulkan/             # the translator library and the CLI (feature `cli`)
│   │   ├── src/                  #   src/tests/: tests that reach internal items
│   │   ├── tests/                #   integration tests through the public API, golden tables
│   │   └── examples/             #   api_preview.rs, which docs/API.md quotes
│   ├── metal2vulkan-air/         # the AIR front end
│   ├── metal2vulkan-runtime/     # the Metal API on Vulkan (README, examples)
│   ├── metal2vulkan-validation/  # unpublished: cases, executors, corpus tools
│   │   ├── fixtures/public/      #   our own synthetic fixtures (.ll, .air.ll, .metal, .spvasm)
│   │   ├── profiles/             #   committed Khronos device profiles
│   │   └── corpus/               #   authored cases and observations (local/ is private, ignored)
│   └── metal2vulkan-devtools/    # unpublished: tool runner, fixture assemblers
├── docs/                         # guides and generated references
├── scripts/                      # developer utilities and generators
├── Cargo.toml                    # the virtual workspace
├── LICENSE                       # the notice: GPL-2.0-or-later OR LGPL-3.0-or-later
└── LICENSE-*                     # the texts it names: GPL v2, LGPL v3, and GPL v3
```

## The loop

```sh
cargo fmt --all
cargo clippy --workspace --all-targets --all-features -- -D warnings
RUSTDOCFLAGS='-D warnings' cargo doc --workspace --all-features --no-deps
scripts/api-docs   # the translator's API reference; fails on a link to a hidden item
cargo nextest run -p metal2vulkan -p metal2vulkan-air -p metal2vulkan-runtime -p metal2vulkan-devtools --all-features
cargo nextest run -p metal2vulkan-validation --all-features
cargo test --workspace --all-features --doc
```

None of these opens a device. Device tests compile only under `--cfg m2v_gpu` or
`--cfg m2v_cpu_icd` ([docs/VALIDATION.md](docs/VALIDATION.md)).

To build the command:

```sh
cargo build --release -p metal2vulkan --features cli
```

## Where tests go

- A test that names an internal item is a unit test of its crate: under `src/tests/` (or its
  module's own `tests`), naming items through `crate::`. Run one file with
  `cargo nextest run -p metal2vulkan --all-features --lib -E 'test(/^tests::<file>::/)'`.
- `crates/metal2vulkan/tests/` holds integration tests through the public API only.
- `crates/metal2vulkan/tests/api_surface.rs` pins every public name of the translator. A new public
  item gets its row there.
- Fixtures are synthetic and ours, under `crates/metal2vulkan-validation/fixtures/public/`. A new
  fixture lands with its rows in `crates/metal2vulkan/tests/public_fixture_expectations_engine.tsv`
  (in inventory order) and its golden printout under `tests/air_input/`, blessed with
  `cargo test -p metal2vulkan --all-features --lib tests::air_input -- --ignored` (on macOS, for
  the `.air.ll` fixtures).
- The manifests set `autobins = false`: a new binary needs its own `[[bin]]` table.

## Golden output hashes and `SEMANTICS_VERSION`

`src/tests/golden_hashes.rs` pins a hash of the final SPIR-V (or the refusal class) of every public
fixture entry point on every committed profile, in `crates/metal2vulkan/tests/golden_hashes.tsv`,
and of every fixture that samples, gathers or computes a level of detail under every exact form
(Metal's sampler emulated and `DevicePolicy::exact_answers`), in
`crates/metal2vulkan/tests/golden_hashes_exact.tsv`: the default policy samples natively, so only
that table holds the sampler's emulation.
`cache::SEMANTICS_VERSION` is part of every cache key, so a change that moves a cell while its
fixture and profile are unchanged must bump `SEMANTICS_VERSION`
(`crates/metal2vulkan/src/cache/mod.rs`) in the same change. New, removed or edited fixtures and
profiles need only the bless. The table records cells only, no version: a `SEMANTICS_VERSION` bump
alone leaves it unchanged, and the bless on `development` refuses a cell that moved under an
unchanged fixture and profile when `SEMANTICS_VERSION` is still its value at the table's last bless.

The tables are re-blessed on `development` after merges, so a change need not carry its own bless.
One command blesses both (on macOS, which has `xcrun metal-as`):

```sh
PATH=/opt/homebrew/opt/llvm/bin:$PATH cargo nextest run -p metal2vulkan --all-features \
  --run-ignored ignored-only -E 'test(bless_golden_hashes)'
```

Never merge a table by hand: after a merge take `development`'s tables and bless again. The other
versions move the same way when their behavior changes: `ARTIFACT_FORMAT_VERSION` (the artifact
encoding) and `RUNTIME_VERSION` (what the runtime does with an unchanged artifact).

## Generated docs

Edit the source, then regenerate:

| Doc | Generator | Source |
|---|---|---|
| `docs/REFUSALS.md` | `python3 scripts/refusal-surface/refusal_surface.py` (`--check` to verify) | the code's refusal types; `reach.tsv` beside the script |
| `docs/METAL-COVERAGE.md` | `python3 scripts/metal-coverage/coverage.py` (`--check` to verify) | `scripts/metal-coverage/coverage.tsv` |
| `crates/metal2vulkan/src/spv.rs`, `crates/metal2vulkan/src/spirv_binary/table.rs` | `scripts/regen-spirv-grammar/regen-spirv-grammar.sh` | Khronos SPIRV-Headers' grammar JSON, and `items.txt` beside the script |
| `crates/metal2vulkan-validation/profiles/*.json` | `scripts/regen-device-profiles/` | gpuinfo reports and local captures |

Three docs are checked by tests: `docs/FAST-PATHS.md` must name every `DevicePolicy` accuracy
choice and the notes it records (`tests::fast_paths`); `docs/API.md` must list every degrade code
with its category and quote `examples/api_preview.rs` verbatim in its `rust` blocks; and
`docs/METAL-COVERAGE.md` must carry the coverage table's totals. A new degrade code also goes in
`crates/metal2vulkan-runtime/src/tests/degradation_codes.txt`, every code ever issued, which a test
holds equal to the live codes and `Degradation::RETIRED_CODES`; a removed kind's code moves to
`RETIRED_CODES` and to `docs/API.md`'s retired lines, and stays in the list.

## Changes

- Follow [AGENTS.md](AGENTS.md): structure over names, honest refusals, fast defaults with exact
  switches (and their row in `docs/FAST-PATHS.md`), tests for every fixed bug class.
- One concern per commit when practical; run `cargo fmt --all` first.
- Update the docs a consumer reads when what they describe changes: `docs/API.md` and
  `examples/api_preview.rs` for the translator's API, the runtime's README and rustdoc for the
  runtime, `docs/HOWTO.md` for integration, `docs/ARCHITECTURE.md` for the translator's stages,
  `README.md` for the command line.
- `CHANGELOG.md` lists releases in a few lines each; add to the top, unreleased version.

## License

metal2vulkan is dual-licensed, `GPL-2.0-or-later OR LGPL-3.0-or-later`, at the recipient's choice
([LICENSE](LICENSE)). Contributions are accepted under the same dual license: a change you submit
can be redistributed and modified under either.

The license files sit at the root and, because cargo packages each crate from its own directory,
as copies in every published crate (`crates/metal2vulkan`, `crates/metal2vulkan-air` and
`crates/metal2vulkan-runtime`). Change them at the root and copy them over; a validation test (`published_crates`) holds every copy equal to the root's file
and the notice's SPDX line equal to the manifests' `license`.
