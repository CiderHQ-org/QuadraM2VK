// Owned fixture: the kernel the argument-encoder
// GPU test runs. argument_buffers' kernel, but for its nested argument buffer, which holds a
// pointer and an inline constant here in place of a texture: the translator does not implement a
// texture or sampler loaded from a nested argument buffer. The other members
// and their ids are argument_buffers', whose encoders Metal measured. Written for this repository;
// no third-party source. Loop-free.
#include <metal_stdlib>
using namespace metal;

struct ReflectionArgInner {
    uint x;
    float2 y;
};

struct ReflectionSub {
    device float *p [[id(0)]];
    uint n [[id(1)]];
};

struct ReflectionPair {
    texture2d<float> a;
    sampler b;
};

struct ReflectionArgs {
    texture2d<float> tex [[id(0)]];
    sampler smp [[id(1)]];
    device float *data [[id(2)]];
    array<texture2d<float, access::read>, 3> texs [[id(3)]];
    constant float4 *consts [[id(6)]];
    uint count [[id(7)]];
    float4 color [[id(8)]];
    command_buffer icb [[id(9)]];
    texture2d<float, access::write> wtex [[id(10)]];
    depth2d<float> dtex [[id(11)]];
    device ReflectionArgInner *inner [[id(12)]];
    compute_pipeline_state cps [[id(13)]];
    constant ReflectionSub *sub [[id(15)]];
    array<sampler, 2> smps [[id(16)]];
    device float *ptrs [[id(18)]] [2];
    uint vals [[id(20)]] [2];
    ReflectionPair pair [[id(22)]];
    ReflectionPair pairs [[id(24)]] [2];
};

struct ReflectionImplicit {
    device uint *a;
    texture2d<half> t;
    sampler s;
    uint n;
};

kernel void encoder_reads(device ReflectionArgs &args [[buffer(0)]],
                          constant ReflectionImplicit &imp [[buffer(1)]],
                          device float4 *out [[buffer(2)]],
                          uint2 gid [[thread_position_in_grid]]) {
    float2 c = float2(0.5);
    float4 v = args.tex.sample(args.smp, c, level(0)) + args.data[gid.x]
             + args.texs[2].read(gid) + args.consts[0] + float(args.count) + args.color
             + float4(args.dtex.sample(args.smp, c, level(0))) + float(args.inner->x)
             + *args.sub->p + float(args.sub->n)
             + args.pair.a.sample(args.smps[1], c, level(0)) + *args.ptrs[1] + float(args.vals[1])
             + args.pairs[1].a.sample(args.pairs[1].b, c, level(0));
    args.wtex.write(v, gid);
    imp.a[gid.x] = imp.n + uint(imp.t.sample(imp.s, c, level(0)).x);
    out[gid.x] = v;
}
