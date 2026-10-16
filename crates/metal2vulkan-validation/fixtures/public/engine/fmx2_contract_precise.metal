// Owned fixture: which multiply-add pairs Metal fuses without fast math (`-fno-fast-math`), where
// an operation carries only the flags a `#pragma clang fp` grants. A Metal probe measured
// `llvm.fmuladd`, the written `a * b + c`, which Metal rounds once. Each shape k reads its own
// operands `v = x + 8k` (`a, b, c, d = v[0 .. 3]`):
//
//   k  word   shape
//   0  0      p = a * b; p + c                     no flags (two statements)
//   1  1      a * b + c                            `llvm.fmuladd`, the control
//   2  2      a * b + c under `contract(fast)`     `contract` on both
//   3  3, 4   p = a * b; p + c; p + d under        a `contract` product used by two additions
//             `contract(fast)`
//   4  5      c - a * b under `contract(fast)`
//   5  6      a * b + c * d under `contract(fast)`
//   6  7      p = a * b; p + c under               `reassoc` on both, no `contract`
//             `reassociate(on)`
//   7  8      p = a * b; p + c under both pragmas  `reassoc contract` on both
//   8  9      p = a * b under `contract(fast)`;    only the product carries `contract`
//             p + c
//   9  10     p = a * b; p + c under               only the addition carries it
//             `contract(fast)`
//
// once for `float` (words 0-10 of the thread's 32) and once for `half` (words 16-26, each a `half`
// zero-extended). Thread i's `x` is `fin + 80 i` or `hin + 80 i`. Words no shape writes keep the
// output's initial bytes. Loop-free.
// metal-fast-math: off
#include <metal_stdlib>
using namespace metal;

static uint word(float v) { return as_type<uint>(v); }
static uint word(half v) { return uint(as_type<ushort>(v)); }

template <typename T>
static void shapes(device const T *x, device uint *o)
{
    {
        device const T *v = x + 0;
        T p = v[0] * v[1];
        o[0] = word(p + v[2]);
    }
    {
        device const T *v = x + 8;
        o[1] = word(v[0] * v[1] + v[2]);
    }
    {
#pragma clang fp contract(fast)
        device const T *v = x + 16;
        o[2] = word(v[0] * v[1] + v[2]);
    }
    {
#pragma clang fp contract(fast)
        device const T *v = x + 24;
        T p = v[0] * v[1];
        o[3] = word(p + v[2]);
        o[4] = word(p + v[3]);
    }
    {
#pragma clang fp contract(fast)
        device const T *v = x + 32;
        o[5] = word(v[2] - v[0] * v[1]);
    }
    {
#pragma clang fp contract(fast)
        device const T *v = x + 40;
        o[6] = word(v[0] * v[1] + v[2] * v[3]);
    }
    {
#pragma clang fp reassociate(on)
        device const T *v = x + 48;
        T p = v[0] * v[1];
        o[7] = word(p + v[2]);
    }
    {
#pragma clang fp contract(fast)
#pragma clang fp reassociate(on)
        device const T *v = x + 56;
        T p = v[0] * v[1];
        o[8] = word(p + v[2]);
    }
    {
        device const T *v = x + 64;
        T p;
        {
#pragma clang fp contract(fast)
            p = v[0] * v[1];
        }
        o[9] = word(p + v[2]);
    }
    {
        device const T *v = x + 72;
        T p = v[0] * v[1];
        {
#pragma clang fp contract(fast)
            o[10] = word(p + v[2]);
        }
    }
}

kernel void fmx2_contract_precise(device const float *fin [[buffer(0)]],
                                  device const half *hin [[buffer(1)]],
                                  device uint *out [[buffer(2)]],
                                  uint i [[thread_position_in_grid]])
{
    device uint *o = out + 32 * i;
    shapes<float>(fin + 80 * i, o);
    shapes<half>(hin + 80 * i, o + 16);
}
