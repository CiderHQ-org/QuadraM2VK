#!/usr/bin/env bash
# GPL-3.0-only. Authored preload/memory/cache regressions on the explicitly selected Vulkan ICD.
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
out=${QM_PRELOAD_OUTPUT:-$root/artifacts/preload}
library=${QM_HOST_LIBRARY:-$root/build/metal/libquadra_metal_host.so}
mkdir -p "$out"
out=$(cd -- "$out" && pwd -P)
printf 'RUNNING\n' >"$out/preload-tests.status"
cache=$(mktemp -d "$out/cache.XXXXXX")
trap 'result=$?; rm -rf -- "$cache"; if (( result != 0 )); then printf "FAIL (%s)\n" "$result" >"$out/preload-tests.status"; fi' EXIT
export VK_DRIVER_FILES=${VK_DRIVER_FILES:-/usr/share/vulkan/icd.d/lvp_icd.x86_64.json}
export LP_NUM_THREADS=${LP_NUM_THREADS:-1}
export XDG_CACHE_HOME=${XDG_CACHE_HOME:-$root/.cache/qualification}
export VK_LOADER_LAYERS_DISABLE='~implicit~'
if [[ -d $root/.tools/validation/usr/share/vulkan/explicit_layer.d ]]; then
  export VK_LAYER_PATH="$root/.tools/validation/usr/share/vulkan/explicit_layer.d"
fi
export VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
export VK_LOADER_DEBUG=layer
export MACOBLOX_METAL_SHADER_CACHE="$cache"
export MACOBLOX_METAL_PRELOAD=1
export MACOBLOX_METAL_PRELOAD_KEY=authored-preload-qualification
export MACOBLOX_METAL_DIAGNOSTICS=1
export MACOBLOX_METAL_GPU_TIMESTAMPS=1
export PYTHONPATH="$root/integration/quadra/launcher:$root/integration/quadra/tests"
tests="$root/integration/quadra/tests"
clang -O2 -Wall -Wextra -Werror "$tests/metal_preload_test.c" -ldl -o "$out/metal-preload-test"
python3 "$tests/assemble_metal_test.py" "$tests/metal_draw.ll" "$out/draw.metallib" bench_vertex:0 bench_fragment:1
python3 "$tests/assemble_metal_test.py" "$tests/metal_dispatch.ll" "$out/dispatch.metallib"
MACOBLOX_METAL_PRELOAD_IN_PROCESS=0 MACOBLOX_METAL_DIAGNOSTICS_FILE="$out/cold-trace.jsonl" \
  timeout 120s "$out/metal-preload-test" "$library" "$out/draw.metallib" "$out/dispatch.metallib" >"$out/cold.stdout.log" 2>"$out/cold.log"
MACOBLOX_METAL_PRELOAD_IN_PROCESS=0 MACOBLOX_METAL_DIAGNOSTICS=0 \
  timeout 120s python3 "$root/integration/quadra/launcher/macoblox/metal_preload.py" --worker "$library" 4096 >"$out/preload.log" 2>&1
MACOBLOX_METAL_PRELOAD_IN_PROCESS=1 MACOBLOX_METAL_DIAGNOSTICS_FILE="$out/warm-trace.jsonl" \
  timeout 120s "$out/metal-preload-test" "$library" "$out/draw.metallib" "$out/dispatch.metallib" >"$out/warm.stdout.log" 2>"$out/warm.log"
printf 'blocked cache path\n' >"$out/blocked-cache"
MACOBLOX_METAL_PRELOAD=0 MACOBLOX_METAL_PRELOAD_IN_PROCESS=0 MACOBLOX_METAL_SHADER_CACHE="$out/blocked-cache" \
  MACOBLOX_METAL_DIAGNOSTICS_FILE="$out/fallback-trace.jsonl" \
  timeout 120s "$out/metal-preload-test" "$library" "$out/draw.metallib" "$out/dispatch.metallib" >"$out/fallback.stdout.log" 2>"$out/fallback.log"
python3 - "$out" "$library" <<'PY'
import json, os, sys
from pathlib import Path
from macoblox import metal_preload
out, library = Path(sys.argv[1]), sys.argv[2]
def snapshots(name):
    return {line.split(' ',1)[0]: json.loads(line.split(' ',1)[1])
            for line in (out/name.replace('.log','.stdout.log')).read_text().splitlines() if line.startswith(('BEFORE ','CREATED ','AFTER '))}
warm = snapshots('warm.log')
assert warm['BEFORE']['preloaded_states_remaining'] == 2, warm['BEFORE']
assert warm['CREATED']['preloaded_states_remaining'] == 0, warm['CREATED']
assert warm['CREATED']['work']['creation_pipelines_made'] == warm['BEFORE']['work']['creation_pipelines_made'], warm
assert warm['AFTER']['work']['draws'] >= 1
assert warm['AFTER']['work']['submits'] >= 1
for phase in ['buffer_create','buffer_upload','texture_create','texture_upload']:
    assert warm['AFTER']['phases'][phase]['count'] > 0, phase
assert not snapshots('fallback.log')['AFTER']['shader_cache']['persistent']
assert snapshots('fallback.log')['AFTER']['shader_cache']['write_failures'] == 0
progress = []
result = metal_preload.run(library, dict(os.environ), progress=lambda done,total: progress.append([done,total]))
assert result['status'] == 'complete' and result['total'] == 2 and result['failed'] == 0, result
assert progress[-1] == [2,2], progress
# A bad learned recipe is a miss. The current game can recreate and record it again.
path = next((Path(os.environ['MACOBLOX_METAL_SHADER_CACHE'])/'quadra-preload-v1').glob('pipeline-*.qmc'))
data = bytearray(path.read_bytes()); data[-1] ^= 1; path.write_bytes(data)
damaged = metal_preload.run(library, dict(os.environ))
assert damaged['status'] == 'partial' and damaged['failed'] == 1, damaged
assert not path.exists()
other = metal_preload.run(library, dict(os.environ, MACOBLOX_METAL_PRELOAD_KEY='another-client'))
assert other['status'] == 'empty', other
for name in ['cold-trace.jsonl','warm-trace.jsonl','fallback-trace.jsonl']:
    records = [json.loads(line) for line in (out/name).read_text().splitlines()]
    assert records and all(record['schema'] == 1 for record in records)
    assert (out/name).stat().st_size <= 16*1024*1024
summary = {'status':'PASS','warm':result,'corrupt_recipe':damaged,'other_client':other,
           'resident_before':2,'resident_after_creation':0,'extra_creation_pipelines':0,
           'gpu_output':'exact authored render + 65-thread dispatch', 'memory_only_fallback':True}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: shader preload replay, same-device Vulkan-object reuse, corrupt recipe recovery and client isolation')
PY
# A separate client cache exercises a scene with more than 64 distinct pipeline descriptions.
for mode in cold warm; do
  resident=0
  if [[ $mode == warm ]]; then resident=1; fi
  MACOBLOX_METAL_SHADER_CACHE="$cache/many" MACOBLOX_METAL_PRELOAD_IN_PROCESS="$resident" \
    MACOBLOX_METAL_DIAGNOSTICS_FILE="$out/many-$mode-trace.jsonl" \
    timeout 120s "$out/metal-preload-test" "$library" "$out/draw.metallib" "$out/dispatch.metallib" --many-pipelines \
      >"$out/many-$mode.stdout.log" 2>"$out/many-$mode.log"
done
python3 - "$out" <<'PY'
import json,sys
from pathlib import Path
out=Path(sys.argv[1])
def snapshots(mode):
    return {line.split(' ',1)[0]:json.loads(line.split(' ',1)[1])
            for line in (out/f'many-{mode}.stdout.log').read_text().splitlines()
            if line.startswith(('BEFORE ','CREATED ','AFTER '))}
cold,warm=snapshots('cold'),snapshots('warm')
expected=min(98,warm['BEFORE']['preloaded_state_limit'])
assert warm['BEFORE']['preloaded_states_remaining']==expected, warm['BEFORE']
assert warm['CREATED']['preloaded_states_remaining']==0, warm['CREATED']
extra=warm['CREATED']['work']['creation_pipelines_made']-warm['BEFORE']['work']['creation_pipelines_made']
if expected==98:
    assert extra==0, (cold,warm)
summary=json.loads((out/'results.json').read_text())
summary['large_scene']={'authored_states':98,'resident_limit':warm['BEFORE']['preloaded_state_limit'],
    'resident_before':expected,'resident_after_creation':0,'extra_creation_pipelines':extra,
    'full_retention_tested':expected==98,'exact_gpu_output':True,
    'scope':'authored lavapipe/selected ICD pipeline reuse, not game FPS'}
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print(f'PASS: {expected}/98 authored states retained; {extra} additional creation pipelines')
PY
# First-launch preparation must work without learned pipeline descriptions. Use only our
# authored libraries and resource bytes; the real client's content is never a test fixture.
MACOBLOX_METAL_SHADER_CACHE="$cache/startup" python3 - "$out" "$library" <<'PY'
import json,os,sys
from pathlib import Path
from macoblox import metal_preload
from startup_preload_test import shader_pack
out,host=Path(sys.argv[1]),sys.argv[2]
pack=out/'authored-shaders.pack'
pack.write_bytes(shader_pack([b'wrapper'+(out/'draw.metallib').read_bytes(),
                              b'other wrapper'+(out/'dispatch.metallib').read_bytes()]))
resources=out/'authored-resources'
(resources/'content').mkdir(parents=True,exist_ok=True)
(resources/'content/font.ttf').write_bytes(b'authored local preload bytes'*1024)
progress=[]
result=metal_preload.run(host,dict(os.environ),library_pack=pack,resource_root=resources,
                        stage_progress=lambda *row:progress.append(row))
assert result['status']=='complete',result
assert result['stages']['libraries']['prepared']==2,result
assert result['stages']['libraries']['failed']==0,result
assert result['stages']['resources']['read_bytes']==(resources/'content/font.ttf').stat().st_size,result
assert {'libraries','pipelines','resources'}<={row[0] for row in progress},progress
summary=json.loads((out/'results.json').read_text())
summary['first_launch_stages']=result
(out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
print('PASS: first-launch shader-library preparation and bounded local-resource warming')
PY
XDG_CONFIG_HOME="$out/launcher/config" XDG_DATA_HOME="$out/launcher/data" \
  python3 -m unittest metal_preload_test startup_preload_test graphics_test session_housekeeping_test >"$out/launcher-tests.log" 2>&1
for log in cold preload warm fallback many-cold many-warm; do
  if ! rg -q 'Insert instance layer.*VK_LAYER_KHRONOS_validation' "$out/$log.log" || rg -n 'VUID-|Validation Error' "$out/$log.log"; then
    printf 'Vulkan validation evidence failed: %s\n' "$log" >&2
    exit 1
  fi
done
printf 'PASS\n' >"$out/preload-tests.status"
printf 'PASS: shader preloading, live memory, cache recovery and launcher controls (%s)\n' "$out"
