// Owned fixture: a writer of color 0 that declares color 0 as an input it never reads, and whose
// discard alone keeps it from early fragment tests. On a device that orders fetch through the
// pixel interlock, the declared input puts color 0 in the interlocked storage form though the
// function reads nothing there; the write it returns to color 0 is then a storage write made
// before the depth and stencil tests, and one under a write mask that covers no channel stores
// nothing. It returns `tint` (buffer 0) and discards the fragments of every fourth column (`x` in
// `[4k, 4k + 1)`), as `smw_writers`' `smw_discard` does. Loop-free.
// metal-fast-math: off
// metal-macos-min: 27.0
#include <metal_stdlib>
using namespace metal;

fragment half4 smw_unread(float4 p [[position]], half4 unread [[color(0)]],
                          constant half4 &tint [[buffer(0)]]) {
    if ((uint(p.x) & 3u) == 0u) {
        discard_fragment();
    }
    return tint;
}
