#!/bin/sh
# Regenerate crates/metal2vulkan/src/spv.rs (the SPIR-V grammar items the translator names) and
# crates/metal2vulkan/src/spirv_binary/table.rs (the test parser's instruction table) from
# Khronos's machine-readable SPIR-V grammar in SPIRV-Headers, with generate.py beside this script.
#
# usage: scripts/regen-spirv-grammar/regen-spirv-grammar.sh
#        HEADERS_REF=<SPIRV-Headers tag> scripts/regen-spirv-grammar/regen-spirv-grammar.sh
#
# The grammar files are fetched once per tag into .cache/spirv-headers/<tag>/ at the repository's
# root (gitignored); METAL2VULKAN_GRAMMAR_CACHE names another directory. Needs curl, python3 and
# rustfmt.
set -eu

HEADERS_REF="${HEADERS_REF:-vulkan-sdk-1.4.341.0}"
here=$(cd -- "$(dirname -- "$0")" && pwd)
root=$(cd -- "$here/../.." && pwd)
cache="${METAL2VULKAN_GRAMMAR_CACHE:-$root/.cache/spirv-headers}/$HEADERS_REF"
base="https://raw.githubusercontent.com/KhronosGroup/SPIRV-Headers/$HEADERS_REF"

mkdir -p "$cache"
for file in include/spirv/unified1/spirv.core.grammar.json \
  include/spirv/unified1/extinst.glsl.std.450.grammar.json LICENSE; do
  name=${file##*/}
  if [ ! -s "$cache/$name" ]; then
    curl -sSfL -o "$cache/$name.part" "$base/$file"
    mv "$cache/$name.part" "$cache/$name"
  fi
done

spv="$root/crates/metal2vulkan/src/spv.rs"
table="$root/crates/metal2vulkan/src/spirv_binary/table.rs"
python3 -I "$here/generate.py" --grammar "$cache" --tag "$HEADERS_REF" \
  --items "$here/items.txt" --spv "$spv" --table "$table"
rustfmt --edition 2021 "$spv" "$table"
