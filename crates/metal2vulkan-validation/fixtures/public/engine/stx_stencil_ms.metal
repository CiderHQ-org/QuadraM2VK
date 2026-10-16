// Owned fixture: a fragment function's `[[stencil]]` output in a
// multisampled pass, where Metal tests and writes each covered sample against the output (its low
// 8 bits). The functions that read `[[sample_id]]` run once per sample, so each sample of a pixel
// takes its own reference; the others run once per pixel. The executors' fullscreen triangle
// covers every sample of every pixel once per instance. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct StencilOut {
    uint stencil [[stencil]];
};

// Sample `s` of pixel `p = y * shape[0] + x` returns word `p * shape[1] + s` of `values`
// (`shape[1]` is the pass's sample count).
fragment StencilOut stx_stencil_ms_by_sample(float4 pos [[position]],
                                             uint s [[sample_id]],
                                             constant uint *values [[buffer(0)]],
                                             constant uint *shape [[buffer(1)]]) {
    uint p = uint(pos.y) * shape[0] + uint(pos.x);
    return { values[p * shape[1] + s] };
}

// The overlapping fragments of sample `s` of pixel `p = y * shape[0] + x` take the words of its
// run in turn: `taken[t]`, `t = p * shape[1] + s`, in raster order group 0, counts the fragments
// that came before, and the fragment returns word `t * shape[2] + taken[t]` of `values`, then
// counts itself (`shape[1]` is the pass's sample count, `shape[2]` the run's length).
fragment StencilOut stx_stencil_ms_in_order(float4 pos [[position]],
                                            uint s [[sample_id]],
                                            constant uint *values [[buffer(0)]],
                                            constant uint *shape [[buffer(1)]],
                                            device uint *taken [[buffer(2), raster_order_group(0)]]) {
    uint t = (uint(pos.y) * shape[0] + uint(pos.x)) * shape[1] + s;
    uint k = taken[t];
    taken[t] = k + 1u;
    return { values[t * shape[2] + k] };
}

struct ColorStencilOut {
    uint color [[color(0)]];
    uint stencil [[stencil]];
};

// Pixel `p = y * width + x` returns word `p` of `colors` as its color and word `p` of `values` as
// its stencil, once for every sample of the pixel.
fragment ColorStencilOut stx_stencil_ms_color(float4 pos [[position]],
                                              constant uint *values [[buffer(0)]],
                                              constant uint *colors [[buffer(1)]],
                                              constant uint &width [[buffer(2)]]) {
    uint p = uint(pos.y) * width + uint(pos.x);
    return { colors[p], values[p] };
}

// Sample `s` of pixel `p = y * shape[0] + x` returns word `t = p * shape[1] + s` of `colors` as
// its color and word `t` of `values` as its stencil.
fragment ColorStencilOut stx_stencil_ms_color_by_sample(float4 pos [[position]],
                                                        uint s [[sample_id]],
                                                        constant uint *values [[buffer(0)]],
                                                        constant uint *colors [[buffer(1)]],
                                                        constant uint *shape [[buffer(2)]]) {
    uint t = (uint(pos.y) * shape[0] + uint(pos.x)) * shape[1] + s;
    return { colors[t], values[t] };
}
