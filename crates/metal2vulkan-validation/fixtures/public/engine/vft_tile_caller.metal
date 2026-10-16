// Owned MSL: a tile function that calls visible
// functions through a `visible_function_table`, as `vft_vertex_caller`'s vertex function does.
// Thread `t` of tile `k` (`x + y * width` of its `threadgroup_position_in_grid` over
// `threadgroups_per_grid`) is `i = k * count + t`, `count` the tile's threads: it calls entry
// `i % size` of the `uint(uint, uint)` table at tile buffer 1 with (i, 3) and writes the result
// to word `i` of tile buffer 0 (0xfffffffd for a null entry, 0xfffffffe for an empty table,
// 0xffffffff for a null table). It touches no attachment. The functions come from
// `visible_table_functions.metal`, which the pipeline's tile stage links. Loop-free.
#include <metal_stdlib>
using namespace metal;

using WordOp = uint(uint, uint);

kernel void vft_tile_dispatch(device uint *out [[buffer(0)]],
                              visible_function_table<WordOp> ops [[buffer(1)]],
                              uint t [[thread_index_in_threadgroup]],
                              uint2 size [[threads_per_threadgroup]],
                              uint2 g [[threadgroup_position_in_grid]],
                              uint2 n [[threadgroups_per_grid]]) {
    uint i = (g.x + g.y * n.x) * (size.x * size.y) + t;
    uint r = 0xffffffffu;
    if (!is_null_visible_function_table(ops)) {
        uint k = ops.size();
        r = 0xfffffffeu;
        if (k > 0u) {
            WordOp *f = ops[i % k];
            r = f == nullptr ? 0xfffffffdu : f(i, 3u);
        }
    }
    out[i] = r;
}
