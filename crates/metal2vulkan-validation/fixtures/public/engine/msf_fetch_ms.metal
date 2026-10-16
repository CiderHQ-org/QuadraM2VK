// Owned fixture: framebuffer fetch in a multisampled pass. Each function
// reads color attachment 0 and writes it back. Its cases draw into passes of 2 or 4 samples whose
// samples start distinct, so what each sample ends with says whether the function ran once per
// sample and which sample's color it read. The probes cover [[sample_id]], the coverage mask in
// and out, and a discard. No multiply feeds an add, so nothing contracts, and every lane is
// exact for the cases' small integer-valued inputs. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

// The color read, beside the invocation's sample index.
fragment float4 msf_fetch_sample(float4 dst [[color(0)]], uint s [[sample_id]])
{
    return float4(dst.x + 1.0f, float(s), dst.z + dst.z, dst.w);
}

// The color read, beside the coverage mask the function is given.
fragment float4 msf_fetch_mask_in(float4 dst [[color(0)]], uint mask [[sample_mask]])
{
    return float4(dst.x + 1.0f, float(mask), dst.z, dst.w);
}

struct MsfMasked {
    float4 color [[color(0)]];
    uint mask [[sample_mask]];
};

// The color read, written back under the coverage mask 0b0101: samples 0 and 2 only.
fragment MsfMasked msf_fetch_mask_out(float4 dst [[color(0)]])
{
    MsfMasked out;
    out.color = (dst + float4(1.0f, 2.0f, 3.0f, 4.0f)) * 2.0f;
    out.mask = 0x5u;
    return out;
}

// The color read, discarded where its alpha is negative.
fragment float4 msf_fetch_discard(float4 dst [[color(0)]])
{
    if (dst.w < 0.0f) {
        discard_fragment();
    }
    return float4(dst.x + 1.0f, dst.y, dst.z, dst.w);
}
