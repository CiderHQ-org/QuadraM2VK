// Owned fixture: recovers the weights of Metal's bicubic filter
// (`filter::bicubic`), which decides whether Vulkan's cubic filter reproduces it.
//
// Each texture is a pattern of single-texel impulses, so a filtered sample returns filter weights
// rather than a blend of colours:
//   h[[texture(0)]]  texel (x, y) is the unit vector e(x mod 4): channel c holds the weight of the
//                    one horizontal tap whose column is congruent to c mod 4 (a 4-tap footprint
//                    covers each residue once), times the sum of the vertical weights;
//   v[[texture(1)]]  texel (x, y) is e(y mod 4): the vertical weights, times the horizontal sum;
//   p[[texture(2)]]  e(x mod 4) where y mod 4 == 0, else zero: channel c is the product of one
//                    horizontal and one vertical weight (the 2D kernel's separability).
// The sample point comes from the input buffer, so each lane chooses its sub-texel fraction and
// base texel. Explicit level 0 (magnification), clamp-to-edge, normalized coordinates.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

constant sampler bicubic_clamp(filter::bicubic, address::clamp_to_edge, coord::normalized);

kernel void a2_bicubic_weights(texture2d<float, access::sample> h [[texture(0)]],
                               texture2d<float, access::sample> v [[texture(1)]],
                               texture2d<float, access::sample> p [[texture(2)]],
                               const device float2 *coord [[buffer(0)]],
                               device float4 *out [[buffer(1)]],
                               uint tid [[thread_position_in_grid]])
{
    float2 uv = coord[tid];
    out[3 * tid + 0] = h.sample(bicubic_clamp, uv, level(0));
    out[3 * tid + 1] = v.sample(bicubic_clamp, uv, level(0));
    out[3 * tid + 2] = p.sample(bicubic_clamp, uv, level(0));
}
