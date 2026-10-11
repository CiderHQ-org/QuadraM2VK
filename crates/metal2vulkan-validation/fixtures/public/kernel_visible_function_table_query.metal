#include <metal_stdlib>
using namespace metal;

using WordFunction = uint(uint);

kernel void kernel_visible_function_table_query(
    device uint *out [[buffer(0)]],
    visible_function_table<WordFunction> few [[buffer(1)]],
    visible_function_table<WordFunction> many [[buffer(2)]])
{
    out[0] = is_null_visible_function_table(few) ? 7u : 3u;
    out[1] = few.size();
    out[2] = few.empty() ? 11u : 5u;
    out[3] = is_null_visible_function_table(many) ? 17u : 13u;
    out[4] = many.size();
    out[5] = many.size() + few.size();
}
