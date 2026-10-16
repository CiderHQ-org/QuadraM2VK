// Owned fixture: an integer read-modify-write of the color attachment. Each
// fragment multiplies the count it reads by three and adds `k + 1` in lane `k` (wrapping), so `n`
// fragments over one pixel in order leave `c * 3^n + (k + 1) * (3^n - 1) / 2` in lane `k`, and a
// fragment that read the attachment before an earlier one wrote it leaves a different integer.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

fragment uint4 exf_fetch_count(uint4 count [[color(0)]])
{
    return count * 3u + uint4(1u, 2u, 3u, 4u);
}
