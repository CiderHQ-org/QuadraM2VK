// Owned fixture: fragment functions that read and write a render pass's
// per-sample state, drawn into attachments of 1, 2, and 4 samples. Each reads or writes one
// sample-rate feature: the pass's sample count, the coverage mask in and out, and depth per
// sample. Loop-free; nothing here runs on a GPU when compiled.
#include <metal_stdlib>
using namespace metal;

// The pass's raster sample count, at pixel rate.
fragment uint sample_rate_num_samples() {
    return get_num_samples();
}

// The coverage the rasterizer gave the invocation (`[[sample_mask]]` in) beside its sample index,
// which runs the function once per sample.
fragment uint4 sample_rate_mask_in(uint mask [[sample_mask]], uint sample [[sample_id]]) {
    return uint4(mask, sample, 0u, 1u);
}

// The coverage the rasterizer gave the pixel, at pixel rate.
fragment uint4 sample_rate_coverage(uint mask [[sample_mask]]) {
    return uint4(mask, 0u, 0u, 1u);
}

struct MaskedColor {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// A written coverage mask: samples 0 and 2 take the color, the others keep the attachment's
// contents.
fragment MaskedColor sample_rate_mask_out() {
    MaskedColor out;
    out.color = float4(1.0, 0.0, 1.0, 1.0);
    out.mask = 0x5u;
    return out;
}

struct SampleDepth {
    float depth [[depth(any)]];
};

// A depth written per sample: an eighth for each sample index past the first.
fragment SampleDepth sample_rate_depth(uint sample [[sample_id]]) {
    SampleDepth out;
    out.depth = 0.125 * float(sample + 1u);
    return out;
}
