// Owned fixture: reads and the width query of 1D texture arrays
// (`air.read_texture_1d_array`, `air.get_width_texture_1d_array`). Lane i reads texel
// (i mod w, (i / w) mod n) of three 1D arrays of w texels and n slices: `float` through a read
// texture at 32-bit coordinates, `uint` through a read texture at 16-bit coordinates, and `half`
// through a sampled texture (the read sampler's form). It writes the three texels' words to
// out[16i .. 16i + 11] and the three widths and the array size to out[16i + 12 .. 16i + 15].
// Loop-free.
#include <metal_stdlib>
using namespace metal;

kernel void unk_read_1d_array(texture1d_array<float, access::read> f [[texture(0)]],
                              texture1d_array<uint, access::read> u [[texture(1)]],
                              texture1d_array<half> h [[texture(2)]],
                              device uint4 *out [[buffer(0)]],
                              uint i [[thread_position_in_grid]]) {
    uint w = f.get_width();
    uint n = f.get_array_size();
    uint x = i % w;
    uint l = (i / w) % n;
    out[4 * i] = as_type<uint4>(f.read(x, l));
    out[4 * i + 1] = u.read(ushort(x), ushort(l));
    out[4 * i + 2] = as_type<uint4>(float4(h.read(x, l)));
    out[4 * i + 3] = uint4(w, u.get_width(), h.get_width(), n);
}
