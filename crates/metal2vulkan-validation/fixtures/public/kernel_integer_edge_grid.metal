#include <metal_stdlib>
using namespace metal;

// Seventeen integer intrinsics at the inputs that separate a correct hand-rolled lowering from a
// plausible one: zero (`clz`/`ctz` answer the WIDTH, while SPIR-V's `FindUMsb`/`FindILsb` answer -1),
// `INT_MIN` (`abs` has no positive image), an all-ones mask, a shift count at and past the width, a
// bitfield whose offset plus count reaches exactly the top bit (rows 11 and 12).
// The bitfield offsets and counts are masked so their sum never exceeds 32, and the `clamp` bounds
// are ordered before the call: past either boundary the operation is undefined and a case that
// compares it is asserting on nothing.
// Every operand is read from a buffer so nothing folds, and the input row is rotated per thread so
// the 12 edge values meet each other in every pairing.
kernel void kernel_integer_edge_grid(
    device const uint *in  [[buffer(0)]],
    device uint       *out [[buffer(1)]],
    uint i [[thread_position_in_grid]])
{
    const uint N = 12;
    uint  x = in[i % N], y = in[(i + 1) % N], z = in[(i + 2) % N];
    int  sx = as_type<int>(x), sy = as_type<int>(y), sz = as_type<int>(z);
    uint o = i * 17;
    out[o +  0] = clz(x);
    out[o +  1] = ctz(x);
    out[o +  2] = popcount(x);
    out[o +  3] = as_type<uint>(abs(sx));
    out[o +  4] = reverse_bits(x);
    out[o +  5] = rotate(x, y & 31u);
    out[o +  6] = mulhi(x, y);
    out[o +  7] = as_type<uint>(madsat(sx, sy, sz));
    out[o +  8] = extract_bits(x, y & 15u, 1u + (z & 15u));
    out[o +  9] = as_type<uint>(extract_bits(sx, y & 15u, 1u + (z & 15u)));
    out[o + 10] = insert_bits(x, y, z & 15u, 1u + (x & 7u));
    out[o + 11] = extract_bits(x, 16u, 16u);
    out[o + 12] = insert_bits(x, y, 24u, 8u);
    out[o + 13] = uint(rhadd(ushort(x & 0xFFFFu), ushort(y & 0xFFFFu)));
    out[o + 14] = uint(addsat(uchar(x & 0xFFu), uchar(y & 0xFFu)));
    out[o + 15] = uint(subsat(uchar(x & 0xFFu), uchar(y & 0xFFu)));
    out[o + 16] = as_type<uint>(clamp(sx, min(sy, sz), max(sy, sz)));
}
