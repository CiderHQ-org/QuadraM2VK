#include <metal_stdlib>
using namespace metal;

// The saturating, halving and wide-product integer intrinsics at the inputs where a plausible
// lowering and a correct one part company: `INT_MIN`/`INT_MAX`, where `addsat`/`subsat` have to
// pick WHICH end to clamp to (the sign of the first operand, not of the wrapped result); `-1`,
// where a signed `hadd` must round toward -inf rather than toward zero, and where an unsigned one
// must not sign-extend; and products that overflow 32 bits, which is the only thing `mulhi`/`madhi`
// report. `madhi`'s addend joins the high half in 32-bit arithmetic and is allowed to wrap, while
// `madsat` clamps the 64-bit sum, so the two disagree exactly at the edges.
// Rows 10-14 repeat the halving and saturating families at 16 bits, 8 bits and three 8-bit lanes:
// those lowerings carry no widening, so the width and the lane count are the whole risk.
// Every operand is read from a buffer so nothing folds, and the input row is rotated per thread so
// the 12 edge values meet each other in every pairing.
kernel void kernel_integer_saturating_grid(
    device const uint *in  [[buffer(0)]],
    device uint       *out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    const uint N = 12;
    uint  x = in[i % N], y = in[(i + 1) % N], z = in[(i + 2) % N];
    int  sx = as_type<int>(x), sy = as_type<int>(y), sz = as_type<int>(z);
    uint o = i * 15;
    out[o +  0] = as_type<uint>(mulhi(sx, sy));
    out[o +  1] = as_type<uint>(madhi(sx, sy, sz));
    out[o +  2] = madhi(x, y, z);
    out[o +  3] = madsat(x, y, z);
    out[o +  4] = as_type<uint>(hadd(sx, sy));
    out[o +  5] = hadd(x, y);
    out[o +  6] = as_type<uint>(rhadd(sx, sy));
    out[o +  7] = rhadd(x, y);
    out[o +  8] = as_type<uint>(addsat(sx, sy));
    out[o +  9] = as_type<uint>(subsat(sx, sy));
    short hx = short(x & 0xFFFFu), hy = short(y & 0xFFFFu);
    out[o + 10] = uint(as_type<ushort>(addsat(hx, hy)));
    out[o + 11] = uint(as_type<ushort>(hadd(hx, hy)));
    out[o + 12] = uint(as_type<ushort>(rhadd(hx, hy)));
    char3 va = char3(char(x & 0xFFu), char(y & 0xFFu), char(z & 0xFFu));
    char3 vb = char3(char(y & 0xFFu), char(z & 0xFFu), char(x & 0xFFu));
    char3 vs = subsat(va, vb);
    out[o + 13] = uint(as_type<uchar>(vs.x))
                | (uint(as_type<uchar>(vs.y)) << 8)
                | (uint(as_type<uchar>(vs.z)) << 16);
    char3 vh = rhadd(va, vb);
    out[o + 14] = uint(as_type<uchar>(vh.x))
                | (uint(as_type<uchar>(vh.y)) << 8)
                | (uint(as_type<uchar>(vh.z)) << 16);
}
