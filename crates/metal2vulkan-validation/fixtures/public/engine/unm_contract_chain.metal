// Owned fixture: how Metal groups a chain of fast-math additions over `contract`
// products. A Metal probe (`fmx2_contract_fast`) read one chain, `a * b + c * d + e`, which Metal
// answers as `fma(a, b, fma(c, d, e))` where the pair rule alone gives `fma(c, d, a * b) + e`. This
// kernel writes the chains around it, compiled with the default fast math. Each shape k reads its
// own operands `v = x + 8k` (`a .. h = v[0 .. 7]`) unless the table says otherwise:
//
//   k   words    shape
//   0   0        a * b + c * d + e                    the control (fmx2_contract_fast word 14)
//   1   1        e + a * b + c * d                    the addend first
//   2   2        a * b + e + c * d                    the addend between
//   3   3        a * b + (c * d + e)                  grouped in the source
//   4   4        c * d + a * b + e                    the products in the other order
//   5   5        a * b + c * d + e, e loaded first    the load order e, a, b, c, d
//   6   6        a * b + c * d + e, c d loaded first  the load order c, d, a, b, e
//   7   7        a * b + c * d + e, apart             a, b, c, d, e = x[224], x[232], x[240],
//                                                      x[248], x[255]: no two in one 8-block
//   8   8        a * b + c * d + 1.5                  a constant addend
//   9   9        a * b + c * d + (i & 15) - 7         an addend converted from the thread index
//   10  10       a * b + c * d + e * f + g            three products and an addend
//   11  11       a * b + c * d + e * f                three products
//   12  12       (a * b + c * d) + (e * f + g * h)    two pairs grouped in the source
//   13  13       a * b + c + d                        one product, two addends
//   14  14       a + b + c * d                        two addends, then the product
//   15  15       a * b + c * d - e                    a subtracted addend
//   16  16       a * b - c * d + e                    a subtracted product
//   17  17       e - a * b - c * d                    both products subtracted
//   18  18-21    a * b + c * d + e on four lanes      a vector chain (a = x[144 .. 147], b, c, d,
//                                                      e the next four each)
//   19  22, 23   p = a * b; p + c * d + e; p          a chain whose product is also stored
//                                                      (operands x + 176)
//   20  24       a * b + c * d + e + f                two addends after the products
//                                                      (operands x + 184)
//   21  25       a * a + b * b + c                    squares (operands x + 200)
//   22  26       a * b + c * d + e * f + g * h        four products (operands x + 208)
//   23  27       a * b + c * d + (e + f)              an addend that is a sum (operands x + 216)
//
// and the addend's kinds after two products, each shape k from 24 on reading `v = x + 256 + 8 (k -
// 24)`:
//
//   24  28       a * b + c * d + T(i & 15)            converted from the thread index alone
//   25  29       a * b + c * d + abs(e)               an absolute value
//   26  30       a * b + c * d + min(e, f)            a minimum
//   27  31       a * b + c * d + (i & 1 ? e : f)      a select
//   28  32       a * b + c * d + T(y[300])            the other type's value converted (`y` is
//                                                      the other buffer's `x`)
//   29  33, 34   a * b + c * d + e; e                 a loaded addend also stored
//   30  35       a * b + c * d + fma(e, f, g)         an explicit fused multiply-add
//   31  36, 37   p = e * f; a * b + c * d + p; p      a product addend also stored
//   32  38       a * b + c * d + e + f * g            a product after the addend
//
// once for `float` (words 0-38 of the thread's 128) and once for `half` (words 64-102, each a
// `half` zero-extended). Thread i's `x` is `fin + 512 i` or `hin + 512 i`. Words no shape writes
// keep the output's initial bytes. Loop-free.
#include <metal_stdlib>
using namespace metal;

static uint word(float v) { return as_type<uint>(v); }
static uint word(half v) { return uint(as_type<ushort>(v)); }

template <typename T, typename T4, typename U>
static void chains(device const T *x, device const U *y, device uint *o, uint i)
{
    {
        device const T *v = x + 0;
        o[0] = word(v[0] * v[1] + v[2] * v[3] + v[4]);
    }
    {
        device const T *v = x + 8;
        o[1] = word(v[4] + v[0] * v[1] + v[2] * v[3]);
    }
    {
        device const T *v = x + 16;
        o[2] = word(v[0] * v[1] + v[4] + v[2] * v[3]);
    }
    {
        device const T *v = x + 24;
        o[3] = word(v[0] * v[1] + (v[2] * v[3] + v[4]));
    }
    {
        device const T *v = x + 32;
        o[4] = word(v[2] * v[3] + v[0] * v[1] + v[4]);
    }
    {
        device const T *v = x + 40;
        T e = v[4];
        T a = v[0], b = v[1], c = v[2], d = v[3];
        o[5] = word(a * b + c * d + e);
    }
    {
        device const T *v = x + 48;
        T c = v[2], d = v[3];
        T a = v[0], b = v[1];
        T e = v[4];
        o[6] = word(a * b + c * d + e);
    }
    {
        o[7] = word(x[224] * x[232] + x[240] * x[248] + x[255]);
    }
    {
        device const T *v = x + 64;
        o[8] = word(v[0] * v[1] + v[2] * v[3] + T(1.5));
    }
    {
        device const T *v = x + 72;
        o[9] = word(v[0] * v[1] + v[2] * v[3] + T(int(i & 15u) - 7));
    }
    {
        device const T *v = x + 80;
        o[10] = word(v[0] * v[1] + v[2] * v[3] + v[4] * v[5] + v[6]);
    }
    {
        device const T *v = x + 88;
        o[11] = word(v[0] * v[1] + v[2] * v[3] + v[4] * v[5]);
    }
    {
        device const T *v = x + 96;
        o[12] = word((v[0] * v[1] + v[2] * v[3]) + (v[4] * v[5] + v[6] * v[7]));
    }
    {
        device const T *v = x + 104;
        o[13] = word(v[0] * v[1] + v[2] + v[3]);
    }
    {
        device const T *v = x + 112;
        o[14] = word(v[0] + v[1] + v[2] * v[3]);
    }
    {
        device const T *v = x + 120;
        o[15] = word(v[0] * v[1] + v[2] * v[3] - v[4]);
    }
    {
        device const T *v = x + 128;
        o[16] = word(v[0] * v[1] - v[2] * v[3] + v[4]);
    }
    {
        device const T *v = x + 136;
        o[17] = word(v[4] - v[0] * v[1] - v[2] * v[3]);
    }
    {
        device const T *v = x + 144;
        T4 a = T4(v[0], v[1], v[2], v[3]);
        T4 b = T4(v[4], v[5], v[6], v[7]);
        T4 c = T4(v[8], v[9], v[10], v[11]);
        T4 d = T4(v[12], v[13], v[14], v[15]);
        T4 e = T4(v[16], v[17], v[18], v[19]);
        T4 r = a * b + c * d + e;
        o[18] = word(r.x);
        o[19] = word(r.y);
        o[20] = word(r.z);
        o[21] = word(r.w);
    }
    {
        device const T *v = x + 176;
        T p = v[0] * v[1];
        o[22] = word(p + v[2] * v[3] + v[4]);
        o[23] = word(p);
    }
    {
        device const T *v = x + 184;
        o[24] = word(v[0] * v[1] + v[2] * v[3] + v[4] + v[5]);
    }
    {
        device const T *v = x + 200;
        o[25] = word(v[0] * v[0] + v[1] * v[1] + v[2]);
    }
    {
        device const T *v = x + 208;
        o[26] = word(v[0] * v[1] + v[2] * v[3] + v[4] * v[5] + v[6] * v[7]);
    }
    {
        device const T *v = x + 216;
        o[27] = word(v[0] * v[1] + v[2] * v[3] + (v[4] + v[5]));
    }
    {
        device const T *v = x + 256;
        o[28] = word(v[0] * v[1] + v[2] * v[3] + T(int(i & 15u)));
    }
    {
        device const T *v = x + 264;
        o[29] = word(v[0] * v[1] + v[2] * v[3] + abs(v[4]));
    }
    {
        device const T *v = x + 272;
        o[30] = word(v[0] * v[1] + v[2] * v[3] + min(v[4], v[5]));
    }
    {
        device const T *v = x + 280;
        o[31] = word(v[0] * v[1] + v[2] * v[3] + ((i & 1u) ? v[4] : v[5]));
    }
    {
        device const T *v = x + 288;
        o[32] = word(v[0] * v[1] + v[2] * v[3] + T(y[300]));
    }
    {
        device const T *v = x + 304;
        T e = v[4];
        o[33] = word(v[0] * v[1] + v[2] * v[3] + e);
        o[34] = word(e);
    }
    {
        device const T *v = x + 312;
        o[35] = word(v[0] * v[1] + v[2] * v[3] + fma(v[4], v[5], v[6]));
    }
    {
        device const T *v = x + 320;
        T p = v[4] * v[5];
        o[36] = word(v[0] * v[1] + v[2] * v[3] + p);
        o[37] = word(p);
    }
    {
        device const T *v = x + 328;
        o[38] = word(v[0] * v[1] + v[2] * v[3] + v[4] + v[5] * v[6]);
    }
}

kernel void unm_contract_chain(device const float *fin [[buffer(0)]],
                               device const half *hin [[buffer(1)]],
                               device uint *out [[buffer(2)]],
                               uint i [[thread_position_in_grid]])
{
    device uint *o = out + 128 * i;
    chains<float, float4, half>(fin + 512 * i, hin + 512 * i, o, i);
    chains<half, half4, float>(hin + 512 * i, fin + 512 * i, o + 64, i);
}
