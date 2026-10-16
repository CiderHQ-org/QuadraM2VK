// metal-std: macos-metal2.2
// Owned fixture: `discard_fragment` with terminate semantics (before
// MSL 2.3), compiled with the standard named on the first line. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct Surface {
    float4 position [[position]];
    float4 color;
};

// Nothing tangled follows the discard: it ends the invocation (`OpTerminateInvocation` or
// `OpKill`).
fragment float4 discard_terminate(Surface in [[stage_in]]) {
    if (in.color.a < 0.5) {
        discard_fragment();
    }
    return in.color;
}

// A derivative follows the discard, and no loop or non-quad subgroup operation: demote or
// predication keeps the quad's derivatives defined.
fragment float4 discard_derivative(Surface in [[stage_in]]) {
    float4 c = in.color;
    if (c.a < 0.5) {
        discard_fragment();
        c = c * 0.25;
    }
    return float4(dfdx(c.x), dfdy(c.y), c.z, 1.0);
}

// A derivative inside a loop follows the discard: demote or predication, as at MSL 2.3, since
// Metal runs a discarded MSL 2.2 thread on as a helper through the loop.
fragment float4 discard_loop(Surface in [[stage_in]], constant uint &n [[buffer(0)]]) {
    float4 c = in.color;
    if (c.a < 0.5) {
        discard_fragment();
    }
    float acc = 0.0;
    for (uint i = 0; i < n; ++i) {
        acc += dfdx(c.x * float(i));
    }
    return float4(acc, c.y, c.z, 1.0);
}

// A derivative and a simdgroup reduction follow the discard: demote, as at MSL 2.3 (the
// discarded thread stays a lane of the reduction on Metal); refused without it.
fragment float4 discard_simd(Surface in [[stage_in]]) {
    float4 c = in.color;
    if (c.a < 0.5) {
        discard_fragment();
        c = c * 0.25;
    }
    return float4(dfdx(c.x), simd_sum(c.y), c.z, 1.0);
}
