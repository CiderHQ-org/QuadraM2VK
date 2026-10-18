// Owned fixture: `mlo_half_trig_inf` compiled with the
// default fast math. The `half` calls carry LLVM's fast flags, under which an infinite operand is
// undefined (MSL section 7.1); the case reads which answers Metal gives there all the same (a Metal
// probe measured them equal to the answers without fast math).
//
// Thread i reads the `half` pattern `in[i]` as `x` and writes one `uint4`: `sin(x) | cos(x) << 16`,
// `tan(x) | s << 16`, `c`, and the pattern of `x`, where `s = sincos(x, c)`. The operands are
// `mlo_half_trig_inf`'s. Loop-free.
//
// sincos's cosine, `c`, is declared ahead of the switch's only label (on i >> 31, zero for every
// thread), so every path into the switch bypasses its declaration and Clang emits no lifetime
// marker for it (as in trn_math_fast.metal and fmx2_divide_precise.metal).
#include <metal_stdlib>
using namespace metal;

kernel void mlo_half_trig_inf_fast(device const ushort *in [[buffer(0)]],
                                   device uint4 *out [[buffer(1)]],
                                   uint i [[thread_position_in_grid]])
{
    switch (i >> 31) {
        half c;
    default:
        half x = as_type<half>(in[i]);
        half s = sincos(x, c);
        out[i] = uint4(uint(as_type<ushort>(sin(x))) | uint(as_type<ushort>(cos(x))) << 16,
                       uint(as_type<ushort>(tan(x))) | uint(as_type<ushort>(s)) << 16,
                       uint(as_type<ushort>(c)), uint(in[i]));
    }
}
