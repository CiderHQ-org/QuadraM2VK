// metal-std: metal4.0
// metal-macos-min: 27.0
// Owned fixture: the AIR 2.9 twin of
// validation/fixtures/public/engine/icb_replay_grid_template.metal. Below the next blank line
// is that file, unchanged; the directives above compile it as MSL 4.0 for macOS 27, where Apple's
// compiler writes AIR 2.9 (each memory atomic with its `mem_flags`, each barrier with its
// `memory_order`). It compiles on every committed profile to the same words as its original
// (tests/engine_air_2_9.rs), so the original's cases and rows speak for it.

// Owned fixture: a compute template whose output observes the region of
// the dispatch each of its threads runs in. Each thread stages its index in its threadgroup in
// threadgroup memory, waits for its group, and reads its neighbour's (so a threadgroup of another
// size reads another index). It then adds, atomically, a weighted sum of its threadgroup's size,
// the dispatch's threadgroup size, its threadgroup's position, the grid's threadgroups and
// threads, its position in its threadgroup, and its neighbour's index, plus `params[0]`, to the
// output slot `index % params[1]`, `index` being its position in the grid, linearized. Commands
// that share the output commute. Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void icb_replay_grid_template(
    constant uint *params [[buffer(0)]],
    device atomic_uint *out [[buffer(1)]],
    threadgroup uint *scratch [[threadgroup(0)]],
    uint3 tid [[thread_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 size [[threads_per_threadgroup]],
    uint3 nominal [[dispatch_threads_per_threadgroup]],
    uint3 groups [[threadgroups_per_grid]],
    uint3 grid [[threads_per_grid]],
    uint lin [[thread_index_in_threadgroup]])
{
    uint n = size.x * size.y * size.z;
    scratch[lin] = lin;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint neighbour = scratch[(lin + 1) % n];
    uint v = params[0]
        + 3 * size.x + 5 * size.y + 7 * nominal.x + 11 * nominal.y
        + 13 * group.x + 17 * group.y + 19 * groups.x + 23 * groups.y
        + 29 * grid.x + 31 * grid.y + 37 * lid.x + 41 * lid.y + 43 * neighbour;
    uint index = tid.x + tid.y * grid.x + tid.z * grid.x * grid.y;
    atomic_fetch_add_explicit(&out[index % params[1]], v, memory_order_relaxed);
}
