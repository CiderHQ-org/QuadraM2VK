// Owned MSL: a vertex function that calls
// visible functions through a `visible_function_table`, as `visible_table_caller`'s kernel does.
// Vertex `vid` calls entry `vid % size` of the `uint(uint, uint)` table at vertex buffer 1 with
// (vid, 3) and writes the result to word `vid` of vertex buffer 0 (0xfffffffd for a null entry,
// 0xfffffffe for an empty table, 0xffffffff for a null table). It returns nothing, so its
// pipeline draws points with rasterization disabled. The functions come from
// `visible_table_functions.metal`, which the pipeline's vertex stage links. Loop-free.
#include <metal_stdlib>
using namespace metal;

using WordOp = uint(uint, uint);

vertex void vft_vertex_dispatch(device uint *out [[buffer(0)]],
                                visible_function_table<WordOp> ops [[buffer(1)]],
                                uint vid [[vertex_id]]) {
    uint r = 0xffffffffu;
    if (!is_null_visible_function_table(ops)) {
        uint n = ops.size();
        r = 0xfffffffeu;
        if (n > 0u) {
            WordOp *f = ops[vid % n];
            r = f == nullptr ? 0xfffffffdu : f(vid, 3u);
        }
    }
    out[vid] = r;
}
