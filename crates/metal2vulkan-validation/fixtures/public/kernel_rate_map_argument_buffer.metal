// Sixteen physical coordinates mapped to screen coordinates through one of two rasterization-rate
// maps that an argument buffer holds.
//
// The map pointers are members of an argument buffer, as in the corpus rows that keep their maps
// beside their other per-pass resources: `left` at byte 8 and `right` at byte 16, after a
// `coordinates` buffer at byte 0 that holds the inputs as plain bytes. Row y of the grid reads
// `left` and row y = 1 reads `right`; the compiler selects the member's address and then loads the
// pointer, the shape `air.map_*_coordinates` reads its map through in those rows. A case authors
// each map member as `rasterization_rate_map`: the Metal oracle encodes a buffer holding what
// `copyParameterDataToBuffer:offset:` writes for the map, and a Vulkan executor one holding this
// crate's parameter data of the same map (`bind::RasterizationRateMap`).
//
// The maps are the two whose tiles this crate compiles as Apple M3 does: `left` is
// a 64x64 map of uniform quality 1.0 (the identity), and `right` is 37x23 with horizontal samples
// [0.5, 1] and vertical samples [1, 0.5], whose physical size is 36x23 (in this crate's compile,
// x maps 33 screen pixels onto the first 32 physical ones and 4 onto the last 4, and y is the
// identity). The eight coordinates lie inside both physical extents, at exact binary fractions, so
// the two rows differ wherever `right`'s x scale does: row 0 answers each coordinate itself.
#include <metal_stdlib>
using namespace metal;

struct MapArguments {
    constant float2* coordinates [[id(0)]];
    constant rasterization_rate_map_data* left [[id(1)]];
    constant rasterization_rate_map_data* right [[id(2)]];
};

kernel void kernel_rate_map_argument_buffer(
    constant MapArguments& arguments [[buffer(0)]],
    device uint* out [[buffer(1)]],
    uint2 tid [[thread_position_in_grid]])
{
    constant rasterization_rate_map_data& data = tid.y == 0 ? *arguments.left : *arguments.right;
    rasterization_rate_map_decoder decoder(data);
    float2 screen = decoder.map_physical_to_screen_coordinates(arguments.coordinates[tid.x]);
    uint lane = tid.y * 8 + tid.x;
    out[lane * 2 + 0] = as_type<uint>(screen.x);
    out[lane * 2 + 1] = as_type<uint>(screen.y);
}
