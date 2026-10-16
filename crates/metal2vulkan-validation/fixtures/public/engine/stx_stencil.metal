// Owned fixture: a fragment function's `[[stencil]]` output, which Metal
// uses as the stencil reference of both faces (its low 8 bits) when it tests and writes the
// stencil under the encoder's depth/stencil state. Each function returns the stencil (one
// returns its depth too); the executors' fullscreen triangle covers every pixel once per instance.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

struct StencilOut {
    uint stencil [[stencil]];
};

// Word `y * width + x` of `values` at pixel (x, y).
fragment StencilOut stx_stencil_by_pixel(float4 pos [[position]],
                                         constant uint *values [[buffer(0)]],
                                         constant uint &width [[buffer(1)]]) {
    uint i = uint(pos.y) * width + uint(pos.x);
    return { values[i] };
}

// As `stx_stencil_by_pixel`, except that a pixel whose word of `discards` is nonzero discards
// its fragment before returning.
fragment StencilOut stx_stencil_discard(float4 pos [[position]],
                                        constant uint *values [[buffer(0)]],
                                        constant uint &width [[buffer(1)]],
                                        constant uint *discards [[buffer(2)]]) {
    uint i = uint(pos.y) * width + uint(pos.x);
    if (discards[i] != 0u) {
        discard_fragment();
    }
    return { values[i] };
}

struct StencilDepthOut {
    float depth [[depth(any)]];
    uint stencil [[stencil]];
};

// As `stx_stencil_by_pixel`, with word `y * width + x` of `depths` as the fragment's depth.
fragment StencilDepthOut stx_stencil_with_depth(float4 pos [[position]],
                                                constant uint *values [[buffer(0)]],
                                                constant uint &width [[buffer(1)]],
                                                constant float *depths [[buffer(2)]]) {
    uint i = uint(pos.y) * width + uint(pos.x);
    return { depths[i], values[i] };
}

// The overlapping fragments of pixel `p = y * shape.x + x` take the words of its run in turn:
// `taken[p]`, in raster order group 0, counts the fragments that came before, and the fragment
// returns word `p * shape.y + taken[p]` of `values`, then counts itself.
fragment StencilOut stx_stencil_in_order(float4 pos [[position]],
                                         constant uint *values [[buffer(0)]],
                                         constant uint2 &shape [[buffer(1)]],
                                         device uint *taken [[buffer(2), raster_order_group(0)]]) {
    uint p = uint(pos.y) * shape.x + uint(pos.x);
    uint k = taken[p];
    taken[p] = k + 1u;
    return { values[p * shape.y + k] };
}
