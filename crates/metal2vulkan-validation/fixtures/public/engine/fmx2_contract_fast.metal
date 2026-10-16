// Owned fixture: which multiply-add pairs Metal fuses, compiled with the
// default fast math. A Metal probe measured one shape: a `fast` product used once by a
// `fast` addition, which Metal rounds once. This kernel writes the shapes around it, each shape k
// on its own operands `v = x + 8k` (`a, b, c, d, e, s = v[0 .. 5]`), so the front end shares no
// product between shapes:
//
//   k   words   shape
//   0   0       a * b + c                          the measured shape, the control
//   1   1, 2    p = a * b; p + c; p + d            a product used by two additions
//   2   3, 4    p = a * b; p + c; p                a product also stored
//   3   5       a * b - c                          a subtraction of the addend
//   4   6       c - a * b                          a subtraction of the product
//   5   7       -(a * b + c)                       a negated addition
//   6   8       -(a * b) - c                       a product of a negation, subtracted from
//   7   9       a * b + c * d                      two products (the later one first in the AIR)
//   8   10      q = c * d; p = a * b; p + q        two products (the earlier one first)
//   9   11      a * b - c * d
//   10  12, 13  p = a * b; p + c * d; p            two products, the first also stored
//   11  14      a * b + c * d + e                  a chain
//   12  15      (a * b + c) * d + e                a nested pair
//   13  16      s > 0 ? a * b : d, plus c          a product across a select
//   14  17      the same with the operands loaded  a product across a phi
//               in the arms
//   15  18, 19  a marker (1 where s > 0) and       a product across a phi whose arm stores
//               (s > 0 ? a * b : d) + c
//   16  20      a * b + c under `contract(off)`    neither operation carries `contract`
//   17  21      p = a * b under `contract(off)`;   only the addition carries it
//               p + c
//   18  22      p = a * b; p + c under             only the product carries it
//               `contract(off)`
//   19  23      a * b + c under                    `contract` without `reassoc`
//               `reassociate(off)`
//   20  24-27   a * b + c on four lanes            a vector pair (operands x[160 .. 175])
//
// once for `float` (words 0-27 of the thread's 64) and once for `half` (words 32-59, each a
// `half` zero-extended), and word 63 is `float(ha * hb) + c`: a `half` product widened before a
// `float` addition (`ha`, `hb` = `hin[192 i + 176 ..]`, `c` = `fin[192 i + 176]`). Thread i's
// `x` is `fin + 192 i` or `hin + 192 i`. Words no shape writes keep the output's initial bytes.
// Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint word(float v) { return as_type<uint>(v); }
static uint word(half v) { return uint(as_type<ushort>(v)); }

template <typename T, typename T4>
static void shapes(device const T *x, device uint *o)
{
    {
        device const T *v = x + 0;
        o[0] = word(v[0] * v[1] + v[2]);
    }
    {
        device const T *v = x + 8;
        T p = v[0] * v[1];
        o[1] = word(p + v[2]);
        o[2] = word(p + v[3]);
    }
    {
        device const T *v = x + 16;
        T p = v[0] * v[1];
        o[3] = word(p + v[2]);
        o[4] = word(p);
    }
    {
        device const T *v = x + 24;
        o[5] = word(v[0] * v[1] - v[2]);
    }
    {
        device const T *v = x + 32;
        o[6] = word(v[2] - v[0] * v[1]);
    }
    {
        device const T *v = x + 40;
        o[7] = word(-(v[0] * v[1] + v[2]));
    }
    {
        device const T *v = x + 48;
        o[8] = word(-(v[0] * v[1]) - v[2]);
    }
    {
        device const T *v = x + 56;
        o[9] = word(v[0] * v[1] + v[2] * v[3]);
    }
    {
        device const T *v = x + 64;
        T a = v[0], b = v[1], c = v[2], d = v[3];
        T q = c * d;
        T p = a * b;
        o[10] = word(p + q);
    }
    {
        device const T *v = x + 72;
        o[11] = word(v[0] * v[1] - v[2] * v[3]);
    }
    {
        device const T *v = x + 80;
        T p = v[0] * v[1];
        o[12] = word(p + v[2] * v[3]);
        o[13] = word(p);
    }
    {
        device const T *v = x + 88;
        o[14] = word(v[0] * v[1] + v[2] * v[3] + v[4]);
    }
    {
        device const T *v = x + 96;
        o[15] = word((v[0] * v[1] + v[2]) * v[3] + v[4]);
    }
    {
        device const T *v = x + 104;
        T a = v[0], b = v[1], c = v[2], d = v[3], s = v[5];
        T q = s > T(0) ? a * b : d;
        o[16] = word(q + c);
    }
    {
        device const T *v = x + 112;
        T q = v[5] > T(0) ? v[0] * v[1] : v[3];
        o[17] = word(q + v[2]);
    }
    {
        device const T *v = x + 120;
        T q;
        if (v[5] > T(0)) {
            q = v[0] * v[1];
            o[18] = 1u;
        } else {
            q = v[3];
        }
        o[19] = word(q + v[2]);
    }
    {
#pragma clang fp contract(off)
        device const T *v = x + 128;
        o[20] = word(v[0] * v[1] + v[2]);
    }
    {
        device const T *v = x + 136;
        T p;
        {
#pragma clang fp contract(off)
            p = v[0] * v[1];
        }
        o[21] = word(p + v[2]);
    }
    {
        device const T *v = x + 144;
        T p = v[0] * v[1];
        {
#pragma clang fp contract(off)
            o[22] = word(p + v[2]);
        }
    }
    {
#pragma clang fp reassociate(off)
        device const T *v = x + 152;
        o[23] = word(v[0] * v[1] + v[2]);
    }
    {
        device const T *v = x + 160;
        T4 a = T4(v[0], v[1], v[2], v[3]);
        T4 b = T4(v[4], v[5], v[6], v[7]);
        T4 c = T4(v[8], v[9], v[10], v[11]);
        T4 r = a * b + c;
        o[24] = word(r.x);
        o[25] = word(r.y);
        o[26] = word(r.z);
        o[27] = word(r.w);
    }
}

kernel void fmx2_contract_fast(device const float *fin [[buffer(0)]],
                               device const half *hin [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint i [[thread_position_in_grid]])
{
    device const float *f = fin + 192 * i;
    device const half *h = hin + 192 * i;
    device uint *o = out + 64 * i;
    shapes<float, float4>(f, o);
    shapes<half, half4>(h, o + 32);
    o[63] = word(float(h[176] * h[177]) + f[176]);
}
