// Owned fixture: a fragment function that reads a color attachment it does
// not write. It reads color attachments 0 and 1 and writes only color attachment 0,
// `count * 3 + tag` (wrapping), so `n` fragments over one pixel in order leave
// `c * 3^n + t * (3^n - 1) / 2` in each lane of attachment 0 while attachment 1 keeps `t`; a
// fragment that read an attachment 1 an earlier fragment had changed leaves a different integer.
// Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

struct Count {
    uint4 count [[color(0)]];
};

fragment Count exf2_fetch_unwritten(uint4 count [[color(0)]], uint4 tag [[color(1)]])
{
    Count out;
    out.count = count * 3u + tag;
    return out;
}
