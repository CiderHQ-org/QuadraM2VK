//! The arith family end to end: owned kernels go through the whole new pipeline (decode →
//! specialize → analyze → select → emit → verify → serialize → link) on committed profiles and pass
//! L3, and their typed modules compute the Metal/LLVM result on the CPU SPIR-V interpreter.
//! Profiles without `shaderInt64` hold 64-bit integers as two words, a switch on one included, and
//! profiles without `shaderFloat64` refuse `double` with the typed capability; `half` and `bfloat`
//! are held widened on every profile.
//!
//! Each kernel stores one word to buffer 0 (the skeleton's static-root store). Canonicalization
//! folds integer arithmetic on constants but never float arithmetic, so each kernel makes its
//! inputs opaque as `fadd <constant>, -0.0` (exact for every non-NaN input) and a bitcast to
//! an integer where it needs one. `llvm-as` assembles the generated module; without it the test
//! skips visibly.
//!
//! Every profile here takes the translation's routines of Metal's maths
//! (`FloatMathsPolicy::Exact`, `exact_maths`), whose answers these tests check; native device
//! maths, the default, is `engine_native_maths`'.

use crate::air::{self, SpecRequest};
use crate::device::profile::{DeviceProfile, DriverWorkaround, FloatFunctions};
use crate::device::requirements::VkFeature;
use crate::device::{Device, DeviceDescription, DevicePolicy};
use crate::error::{Capability, Error, Feature, LowerFamily, Needs};
use crate::float16::f16_bits_to_f32;
use crate::library::FunctionConstantValues;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::compile::{plan_bitcode, EntryRef};
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, exact_maths, link_and_validate, llvm_assemble, meter, profile, tiny_check_scales,
    without_math_routines, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};

/// The profiles: the M1 set plus one with 64-bit floats.
const PROFILES: [&str; 4] = [
    "moltenvk-m3",
    "lunarg-desktop-baseline-2023",
    "vk12-floor",
    "amd-rdna3-radv",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Need {
    Float64,
}

struct Kernel {
    name: &'static str,
    /// The body; it defines `%r` (`i32` or `float`), which the kernel stores.
    body: String,
    /// The result type of `%r`.
    float: bool,
    /// The stored word, or `None` when the interpreter does not model the kernel
    /// (transcendentals).
    want: Option<u32>,
    needs: &'static [Need],
    note: Option<PrecisionNote>,
}

/// `%name` = an opaque `float` with the value `v`.
fn of(name: &str, v: f32) -> String {
    format!(
        "  %{name} = fadd float 0x{:016X}, -0.0\n",
        f64::from(v).to_bits()
    )
}

/// `%name` = an opaque `i32` with bits `b`. A NaN or denormal pattern is not carried through
/// `float` arithmetic, which answers Metal's canonical NaN and reads a denormal as the zero of its
/// sign, but through `llvm.umin` with `-1`.
fn oi(name: &str, b: u32) -> String {
    let f = f32::from_bits(b);
    if f.is_nan() || f.is_subnormal() {
        return format!(
            "  %{name} = call i32 @llvm.umin.i32(i32 {}, i32 -1)\n",
            b as i32
        );
    }
    format!(
        "{}  %{name} = bitcast float %{name}.f to i32\n",
        of(&format!("{name}.f"), f32::from_bits(b))
    )
}

/// `%name` = an opaque `i<bits>` with the low bits of `b`.
fn on(name: &str, bits: u8, b: u32) -> String {
    format!(
        "{}  %{name} = trunc i32 %{name}.w to i{bits}\n",
        oi(&format!("{name}.w"), b)
    )
}

fn k(name: &'static str, body: String, want: u32) -> Kernel {
    Kernel {
        name,
        body,
        float: false,
        want: Some(want),
        needs: &[],
        note: None,
    }
}

fn kernels() -> Vec<Kernel> {
    let mut v = vec![
        k(
            "add_i8_wraps",
            format!(
                "{}{}  %s = add i8 %a, %b\n  %r = zext i8 %s to i32\n",
                on("a", 8, 250),
                on("b", 8, 10)
            ),
            4,
        ),
        k(
            "sdiv_i16_sign_extends",
            format!(
                "{}  %q = sdiv i16 %a, 3\n  %r = sext i16 %q to i32\n",
                on("a", 16, 0xFFF6)
            ),
            0xFFFF_FFFD,
        ),
        k(
            "ashr_i5_keeps_the_sign",
            format!(
                "{}  %s = ashr i5 %a, 2\n  %r = zext i5 %s to i32\n",
                on("a", 5, 0x10)
            ),
            0x1C,
        ),
        k(
            "icmp_signed_and_unsigned",
            format!(
                "{}{}  %lt = icmp slt i32 %a, %b\n  %ult = icmp ult i32 %a, %b\n  %x = select i1 %lt, i32 7, i32 9\n  %y = select i1 %ult, i32 16, i32 32\n  %r = or i32 %x, %y\n",
                oi("a", 5),
                oi("b", 0xFFFF_FFFD)
            ),
            9 | 16,
        ),
        // An integer to `bool` is `x != 0`, whatever the low bit or byte (lanes 2, 0x80, 0, 1, and
        // the scalar 0x100).
        k(
            "convert_integers_to_bool_is_nonzero",
            format!(
                "{}{}{}{}{}  %v0 = insertelement <4 x i8> zeroinitializer, i8 %a, i32 0\n  %v1 = insertelement <4 x i8> %v0, i8 %b, i32 1\n  %v2 = insertelement <4 x i8> %v1, i8 %c, i32 2\n  %v3 = insertelement <4 x i8> %v2, i8 %d, i32 3\n  %bv = call <4 x i1> @air.convert.u.v4i1.u.v4i8(<4 x i8> %v3)\n  %e0 = extractelement <4 x i1> %bv, i32 0\n  %e1 = extractelement <4 x i1> %bv, i32 1\n  %e2 = extractelement <4 x i1> %bv, i32 2\n  %e3 = extractelement <4 x i1> %bv, i32 3\n  %s = call i1 @air.convert.u.i1.s.i32(i32 %w)\n  %x0 = select i1 %e0, i32 1, i32 0\n  %x1 = select i1 %e1, i32 2, i32 0\n  %x2 = select i1 %e2, i32 4, i32 0\n  %x3 = select i1 %e3, i32 8, i32 0\n  %x4 = select i1 %s, i32 16, i32 0\n  %y0 = or i32 %x0, %x1\n  %y1 = or i32 %y0, %x2\n  %y2 = or i32 %y1, %x3\n  %r = or i32 %y2, %x4\n",
                on("a", 8, 2),
                on("b", 8, 0x80),
                on("c", 8, 0),
                on("d", 8, 1),
                oi("w", 0x100)
            ),
            1 | 2 | 8 | 16,
        ),
        k(
            "fcmp_uno_sees_nan",
            format!(
                "{}{}  %u = fcmp uno float %a, %b\n  %o = fcmp olt float %b, %b\n  %x = zext i1 %u to i32\n  %y = zext i1 %o to i32\n  %y2 = shl i32 %y, 1\n  %r = or i32 %x, %y2\n",
                of("a", f32::NAN),
                of("b", 1.0)
            ),
            1,
        ),
        k(
            "fptosi_and_sitofp",
            format!(
                "{}  %i = fptosi float %a to i8\n  %n = sub i8 0, %i\n  %f = sitofp i8 %n to float\n  %r = bitcast float %f to i32\n",
                of("a", 3.7)
            ),
            (-3.0f32).to_bits(),
        ),
        k(
            "vector_shuffle_and_bitcast",
            format!(
                "{}  %v0 = insertelement <4 x i8> <i8 1, i8 2, i8 3, i8 0>, i8 %a, i64 3\n  %v1 = shufflevector <4 x i8> %v0, <4 x i8> poison, <4 x i32> <i32 3, i32 2, i32 1, i32 0>\n  %r = bitcast <4 x i8> %v1 to i32\n",
                on("a", 8, 4)
            ),
            0x0102_0304,
        ),
        k(
            "dynamic_extract",
            format!(
                "{}  %e = extractelement <4 x i32> <i32 10, i32 20, i32 30, i32 40>, i32 %i\n  %r = add i32 %e, 1\n",
                oi("i", 2)
            ),
            31,
        ),
        // An element index may be any integer type, `i1` too, read as unsigned: a compare's `bool`
        // picks lane 1 or 0. The SPIR-V builder rejected the `bool` index.
        k(
            "i1_index_extract_set",
            format!(
                "{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %r = extractelement <2 x i32> <i32 10, i32 20>, i1 %c\n",
                of("x", 0.75)
            ),
            20,
        ),
        k(
            "i1_index_extract_clear",
            format!(
                "{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %r = extractelement <2 x i32> <i32 10, i32 20>, i1 %c\n",
                of("x", 0.25)
            ),
            10,
        ),
        k(
            "i1_index_insert_set",
            format!(
                "{}{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %w = insertelement <2 x i32> <i32 10, i32 20>, i32 %a, i1 %c\n  %l0 = extractelement <2 x i32> %w, i32 0\n  %l1 = extractelement <2 x i32> %w, i32 1\n  %s = mul i32 %l0, 1000\n  %r = add i32 %s, %l1\n",
                of("x", 0.75),
                oi("a", 7)
            ),
            10_007,
        ),
        k(
            "i1_index_insert_clear",
            format!(
                "{}{}  %c = fcmp ogt float %x, 0x3FE0000000000000\n  %w = insertelement <2 x i32> <i32 10, i32 20>, i32 %a, i1 %c\n  %l0 = extractelement <2 x i32> %w, i32 0\n  %l1 = extractelement <2 x i32> %w, i32 1\n  %s = mul i32 %l0, 1000\n  %r = add i32 %s, %l1\n",
                of("x", 0.25),
                oi("a", 7)
            ),
            7_020,
        ),
        k(
            "aggregate_round_trip",
            format!(
                "{}{}  %s0 = insertvalue {{ i32, float }} undef, i32 %a, 0\n  %s1 = insertvalue {{ i32, float }} %s0, float %f, 1\n  %x = extractvalue {{ i32, float }} %s1, 0\n  %g = extractvalue {{ i32, float }} %s1, 1\n  %gi = fptosi float %g to i32\n  %r = add i32 %x, %gi\n",
                oi("a", 40),
                of("f", 2.0)
            ),
            42,
        ),
        k(
            "llvm_integer_intrinsics",
            format!(
                "{}{}{}  %m = call i32 @llvm.umin.i32(i32 %a, i32 %b)\n  %p = call i16 @llvm.ctpop.i16(i16 %h)\n  %pw = zext i16 %p to i32\n  %l = call i32 @llvm.ctlz.i32(i32 %m, i1 false)\n  %f = call i32 @llvm.fshl.i32(i32 %a, i32 %b, i32 8)\n  %s0 = add i32 %m, %pw\n  %s1 = add i32 %s0, %l\n  %r = xor i32 %s1, %f\n",
                oi("a", 5),
                oi("b", 0x0900_0000),
                on("h", 16, 0xF0F0),
            ),
            // umin 5; ctpop 8; ctlz(5) 29; fshl(5, 0x09000000, 8) = 0x509.
            (5 + 8 + 29) ^ 0x509,
        ),
        k(
            "abs_and_saturation",
            format!(
                "{}  %x = call i8 @llvm.abs.i8(i8 %a, i1 false)\n  %s = call i8 @air.add_sat.s.i8(i8 %x, i8 125)\n  %r = zext i8 %s to i32\n",
                on("a", 8, 0xFB)
            ),
            // abs(-5) = 5; 5 + 125 saturates at 127.
            127,
        ),
        k(
            "air_convert_saturates",
            format!(
                "{}{}  %i = call i32 @air.convert.s.i32.f.f32(float %a)\n  %u = call i8 @air.convert.u.i8.f.f32(float %b)\n  %uw = zext i8 %u to i32\n  %r = xor i32 %i, %uw\n",
                of("a", 3.0e9),
                of("b", -7.0)
            ),
            0x7FFF_FFFF,
        ),
        k(
            "air_float_functions",
            format!(
                "{}  %c = call float @air.clamp.f32(float %a, float 0.0, float 1.0)\n  %n = fneg float %a\n  %ro = call float @air.round.f32(float %n)\n  %fr = call float @air.fract.f32(float %a)\n  %s = fadd float %ro, %fr\n  %t = fadd float %s, %c\n  %r = bitcast float %t to i32\n",
                of("a", 2.5)
            ),
            // clamp 1, round(-2.5) = -3, fract 0.5: -1.5.
            (-1.5f32).to_bits(),
        ),
        k(
            "phi_of_a_widened_byte",
            format!(
                "{}  %c = icmp ugt i8 %a, 3\n  br i1 %c, label %t, label %e\nt:\n  %x = add i8 %a, 100\n  br label %j\ne:\n  br label %j\nj:\n  %p = phi i8 [ %x, %t ], [ 7, %e ]\n  %r = zext i8 %p to i32\n",
                on("a", 8, 200)
            ),
            44,
        ),
    ];
    // A switch on a narrow integer selects on its zero-extended container.
    v.push(k(
        "switch_on_a_widened_byte",
        format!(
            "{}  switch i8 %a, label %d [ i8 1, label %o\n i8 -56, label %n ]\no:\n  br label %d\nn:\n  br label %d\nd:\n  %r = phi i32 [ 7, %entry ], [ 11, %o ], [ 13, %n ]\n",
            on("a", 8, 200)
        ),
        13,
    ));
    // Without `shaderInt64` the selector is two words; a nonzero high word matches no (32-bit) case
    // literal.
    v.push(k(
        "switch_on_an_i40",
        format!(
            "{}  %w = zext i32 %a to i40\n  %s = shl i40 %w, 4\n  switch i40 %s, label %d [ i40 16, label %o ]\no:\n  br label %d\nd:\n  %r = phi i32 [ 7, %entry ], [ 11, %o ]\n",
            oi("a", 1)
        ),
        11,
    ));
    v.push(k(
        "switch_on_an_i64_high_word",
        format!(
            "{}  %w = zext i32 %a to i64\n  %s = shl i64 %w, 32\n  %t = or i64 %s, 16\n  switch i64 %t, label %d [ i64 16, label %o ]\no:\n  br label %d\nd:\n  %p = phi i32 [ 7, %entry ], [ 11, %o ]\n  %u = lshr i64 %t, 32\n  switch i64 %u, label %e [ i64 5, label %f ]\nf:\n  br label %e\ne:\n  %q = phi i32 [ 100, %d ], [ 200, %f ]\n  %r = add i32 %p, %q\n",
            oi("a", 5)
        ),
        207,
    ));
    // `mad_sat` on 64-bit integers saturates the exact 128-bit result.
    v.push(k(
        "mad_sat_i64_saturates_the_exact_result",
        format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 31\n  %s = call i64 @air.mad_sat.s.i64(i64 %y, i64 %y, i64 -5)\n  %u = call i64 @air.mad_sat.u.i64(i64 %x, i64 %x, i64 7)\n  %n = sub i64 0, %y\n  %m = call i64 @air.mad_sat.s.i64(i64 %n, i64 %y, i64 3)\n  %sh = lshr i64 %s, 32\n  %uh = lshr i64 %u, 32\n  %mh = lshr i64 %m, 32\n  %a0 = trunc i64 %s to i32\n  %a1 = trunc i64 %sh to i32\n  %a2 = trunc i64 %u to i32\n  %a3 = trunc i64 %uh to i32\n  %a4 = trunc i64 %mh to i32\n  %x1 = xor i32 %a0, %a1\n  %x2 = xor i32 %x1, %a2\n  %x3 = xor i32 %x2, %a3\n  %r = xor i32 %x3, %a4\n",
            oi("a", 0xFFFF_FFFF)
        ),
        0xFFFF_FFF6,
    ));
    v.push(k(
        "convert_round_and_repack",
        format!(
            "{}  %c1 = call i32 @air.convert.s.i32.f.f32(float %a)\n  %f1 = call float @air.round.f32(float %a)\n  %c2 = call i32 @air.convert.s.i32.f.f32(float %f1)\n  %v = bitcast i32 %c1 to <4 x i8>\n  %v2 = add <4 x i8> %v, <i8 1, i8 1, i8 1, i8 1>\n  %w = bitcast <4 x i8> %v2 to i32\n  %s = add i32 %c1, %c2\n  %r = add i32 %s, %w\n",
            of("a", 1.5)
        ),
        // trunc 1, round 2, <2, 1, 1, 1> = 0x01010102.
        3 + 0x0101_0102,
    ));
    v.push(Kernel {
        name: "mul_i40_masks",
        body: format!(
            "{}  %w = zext i32 %a to i40\n  %m = mul i40 %w, 4096\n  %s = lshr i40 %m, 8\n  %r = trunc i40 %s to i32\n",
            oi("a", 0x1234_5678)
        ),
        float: false,
        want: Some(0x2345_6780),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "i64_carry",
        body: format!(
            "{}  %w = zext i32 %a to i64\n  %s = add i64 %w, 1\n  %h = lshr i64 %s, 32\n  %r = trunc i64 %h to i32\n",
            oi("a", 0xFFFF_FFFF)
        ),
        float: false,
        want: Some(1),
        needs: &[],
        note: None,
    });
    // Without `shaderInt64` 64-bit integers are two words; division is the exact word division
    // (`udivrem`), with no loop.
    v.push(k(
        "i64_division_as_words",
        format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 20\n  %q = udiv i64 %y, 1000003\n  %n = sub i64 0, %y\n  %s = srem i64 %n, 77\n  %ql = trunc i64 %q to i32\n  %qs = lshr i64 %q, 32\n  %qh = trunc i64 %qs to i32\n  %x1 = xor i32 %ql, %qh\n  %sl = trunc i64 %s to i32\n  %r = add i32 %x1, %sl\n",
            oi("a", 0x1234_5678)
        ),
        0x1316_B402,
    ));
    v.push(k(
        WIDE_DIVISIONS,
        format!(
            "{}{}  %x = zext i32 %a to i64\n  %y0 = zext i32 %b to i64\n  %y = shl i64 %y0, 24\n  %d = or i64 %x, 1\n  %q = udiv i64 %y, %d\n  %m = urem i64 %y, %d\n  %n = sub i64 0, %y\n  %s = sdiv i64 %n, %d\n  %t = srem i64 %n, 77\n  %v0 = insertelement <2 x i64> zeroinitializer, i64 %y, i32 0\n  %v = insertelement <2 x i64> %v0, i64 %n, i32 1\n  %w = udiv <2 x i64> %v, <i64 3, i64 1000003>\n  %e0 = extractelement <2 x i64> %w, i32 0\n  %e1 = extractelement <2 x i64> %w, i32 1\n  %s1 = add i64 %q, %m\n  %s2 = add i64 %s1, %s\n  %s3 = add i64 %s2, %t\n  %s4 = add i64 %s3, %e0\n  %s5 = add i64 %s4, %e1\n  %h = lshr i64 %s5, 32\n  %hl = trunc i64 %h to i32\n  %ll = trunc i64 %s5 to i32\n  %r = xor i32 %ll, %hl\n",
            oi("a", 0x1234_5678),
            oi("b", 0x0ABC_DEF1)
        ),
        0x925B_1770,
    ));
    v.push(k(
        "i64_phi_across_blocks",
        format!(
            "{}  %x = zext i32 %a to i64\n  %c = icmp ugt i32 %a, 5\n  br i1 %c, label %t, label %f\nt:\n  %u = shl i64 %x, 33\n  br label %d\nf:\n  %v = add i64 %x, 1\n  br label %d\nd:\n  %p = phi i64 [ %u, %t ], [ %v, %f ]\n  %h = lshr i64 %p, 32\n  %r = trunc i64 %h to i32\n",
            oi("a", 0x1234_5678)
        ),
        0x2468_ACF0,
    ));
    v.push(k(
        "fptosi_to_i64_as_words",
        format!(
            "{}  %i = fptosi float %f to i64\n  %h = lshr i64 %i, 32\n  %l = trunc i64 %i to i32\n  %hh = trunc i64 %h to i32\n  %r = xor i32 %l, %hh\n",
            of("f", -12_345_678.0)
        ),
        0x00BC_614D,
    ));
    // Chunked vectors of 64-bit integers are chunks of two-word lanes without `shaderInt64`:
    // lane-wise operations per chunk, constant and `i64` dynamic lane moves, a shuffle into a
    // five-lane vector (its last chunk one lane), a lane-wise intrinsic, a word bitcast, and a
    // truncation.
    v.push(k(
        "chunked_i64_lanes",
        format!(
            "{}{}  %x = zext i32 %a to i64\n  %h = shl i64 %x, 28\n  %v = insertelement <8 x i64> <i64 1, i64 2, i64 3, i64 4, i64 5, i64 6, i64 7, i64 8>, i64 %h, i32 5\n  %w = add <8 x i64> %v, <i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295, i64 4294967295>\n  %m = mul <8 x i64> %w, <i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3, i64 3>\n  %c = icmp ugt <8 x i64> %m, <i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900, i64 12884901900>\n  %s = select <8 x i1> %c, <8 x i64> %m, <8 x i64> %v\n  %i = zext i32 %b to i64\n  %d = extractelement <8 x i64> %s, i64 %i\n  %t = insertelement <8 x i64> %s, i64 %x, i64 %i\n  %q = shufflevector <8 x i64> %t, <8 x i64> %s, <5 x i32> <i32 5, i32 13, i32 0, i32 7, i32 2>\n  %u = call <5 x i64> @llvm.umax.v5i64(<5 x i64> %q, <5 x i64> <i64 100, i64 100, i64 100, i64 100, i64 100>)\n  %bw = bitcast <5 x i64> %u to <10 x i32>\n  %e0 = extractelement <10 x i32> %bw, i32 1\n  %e1 = extractelement <10 x i32> %bw, i32 2\n  %tr = trunc <5 x i64> %u to <5 x i32>\n  %e2 = extractelement <5 x i32> %tr, i32 4\n  %dl = trunc i64 %d to i32\n  %dh64 = lshr i64 %d, 32\n  %dh = trunc i64 %dh64 to i32\n  %x0 = xor i32 %e0, %e1\n  %x1 = xor i32 %x0, %e2\n  %x2 = xor i32 %x1, %dl\n  %r = xor i32 %x2, %dh\n",
            oi("a", 0x89AB_CDEF),
            oi("b", 5)
        ),
        0x19D0_36FB,
    ));
    v.push(k(
        "chunked_i64_converts_and_one_lane",
        format!(
            "{}  %x = zext i32 %a to i64\n  %n = sub i64 0, %x\n  %v = insertelement <6 x i64> <i64 -3, i64 7, i64 1099511627776, i64 -1099511627777, i64 5, i64 6>, i64 %n, i32 4\n  %f = sitofp <6 x i64> %v to <6 x float>\n  %g = extractelement <6 x float> %f, i32 4\n  %k = extractelement <6 x float> %f, i32 3\n  %gi = bitcast float %g to i32\n  %ki = bitcast float %k to i32\n  %one = insertelement <1 x i64> poison, i64 %x, i32 0\n  %two = add <1 x i64> %one, <i64 -1>\n  %e = extractelement <1 x i64> %two, i32 0\n  %el = trunc i64 %e to i32\n  %y = xor i32 %gi, %ki\n  %r = xor i32 %y, %el\n",
            oi("a", 0x89AB_CDEF)
        ),
        0x9522_6620,
    ));
    // Metal's fast division reads the device's reciprocal; its precise division is a routine, exact
    // on every profile.
    v.push(Kernel {
        name: "arcp_fdiv_notes_device_precision",
        body: format!("{}  %r = fdiv arcp float %a, 2.0\n", of("a", 7.0)),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[],
        note: Some(PrecisionNote::DeviceArithmetic),
    });
    v.push(Kernel {
        name: "fdiv_is_metals_precise_division",
        body: format!("{}  %r = fdiv float %a, 2.0\n", of("a", 7.0)),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    // Apple's math library defines precise `air.exp2`, `air.log2`, and `air.sqrt` as these `llvm.*`
    // calls, so they lower as those functions with their notes (precise `sqrt`, whose answer is the
    // one correctly rounded value, with none); `llvm.copysign` moves the sign bit exactly.
    v.push(Kernel {
        name: "llvm_sqrt_is_the_correctly_rounded_root",
        body: format!(
            "{}  %r = call float @llvm.sqrt.f32(float %a)\n",
            of("a", 2.25)
        ),
        float: true,
        want: Some(1.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    for (name, f) in [
        ("llvm_exp2_notes_device_transcendentals", "exp2"),
        ("llvm_log2_notes_device_transcendentals", "log2"),
    ] {
        v.push(Kernel {
            name,
            body: format!(
                "{}  %r = call float @llvm.{f}.f32(float %a)\n",
                of("a", 3.0)
            ),
            float: true,
            want: None,
            needs: &[],
            note: Some(PrecisionNote::DeviceTranscendentals),
        });
    }
    v.push(Kernel {
        name: "llvm_copysign_takes_the_sign_bit",
        body: format!(
            "{}{}  %r = call float @llvm.copysign.f32(float %a, float %b)\n",
            of("a", 2.5),
            of("b", -0.0)
        ),
        float: true,
        want: Some((-2.5f32).to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "sin_notes_device_transcendentals",
        body: format!("{}  %r = call float @air.sin.f32(float %a)\n", of("a", 0.5)),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    // `rgb565` rounds the float product with ties to even (0.5 * 63 = 31.5 -> 32, exact in f32),
    // and `rg11b10f` unpacks the half with the field's bits (0x3C0 -> red 1.0).
    v.push(k(
        "rgb565_pack_rounds_the_exact_product",
        format!(
            "{}  %v = insertelement <3 x float> <float 0.0, float 0.5, float 0.0>, float %a, i32 0\n  %p = call i16 @air.pack.unorm.rgb565.v3f32(<3 x float> %v)\n  %r = zext i16 %p to i32\n",
            of("a", 1.0)
        ),
        31 | (32 << 5),
    ));
    v.push(Kernel {
        name: "rg11b10f_unpacks_half_fields",
        body: format!(
            "{}  %v = call <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32 %w)\n  %x = extractelement <3 x float> %v, i32 0\n  %y = extractelement <3 x float> %v, i32 1\n  %r = fadd float %x, %y\n",
            oi("w", 0x3C0 | (0x180 << 11))
        ),
        float: true,
        want: Some((1.0f32 + 2f32.powi(-9)).to_bits()),
        needs: &[],
        note: None,
    });
    // A 64-bit integer just above a bfloat tie rounds up (through a round-to-odd f32).
    v.push(Kernel {
        name: "int64_to_bfloat_rounds_once",
        body: format!(
            "{}  %x = zext i32 %a to i64\n  %y = shl i64 %x, 32\n  %z = or i64 %y, 1\n  %b = uitofp i64 %z to bfloat\n  %r = fpext bfloat %b to float\n",
            oi("a", 0x0101_0000)
        ),
        float: true,
        want: Some(0x5B81_0000),
        needs: &[],
        note: None,
    });
    // Vectors of other than 2 to 4 lanes are chunked: lane-wise operations per chunk, lane moves
    // through chunk members, bitcasts through words.
    v.push(k(
        "chunked_add_and_constant_lane",
        format!(
            "{}  %v = insertelement <8 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8>, i32 %a, i32 5\n  %w = add <8 x i32> %v, <i32 10, i32 20, i32 30, i32 40, i32 50, i32 60, i32 70, i32 80>\n  %x = extractelement <8 x i32> %w, i32 5\n  %y = extractelement <8 x i32> %w, i32 7\n  %r = add i32 %x, %y\n",
            oi("a", 3)
        ),
        3 + 60 + 88,
    ));
    v.push(Kernel {
        name: "chunked_dynamic_lanes_compare_and_select",
        body: format!(
            "{}{}  %v = insertelement <8 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000, float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, float %a, i32 %i\n  %x = fmul <8 x float> %v, <float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000, float 0x4000000000000000>\n  %c = fcmp ogt <8 x float> %x, <float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000, float 0x4014000000000000>\n  %s = select <8 x i1> %c, <8 x float> %x, <8 x float> zeroinitializer\n  %j = add i32 %i, 1\n  %e = extractelement <8 x float> %s, i32 %i\n  %g = extractelement <8 x float> %s, i32 %j\n  %h = extractelement <8 x float> %s, i32 1\n  %t = fadd float %e, %g\n  %r = fadd float %t, %h\n",
            of("a", 3.0),
            oi("i", 6)
        ),
        float: true,
        want: Some(22.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "chunked_bytes_bitcast_both_ways",
        format!(
            "{}  %v = insertelement <4 x i32> zeroinitializer, i32 %a, i32 2\n  %b = bitcast <4 x i32> %v to <16 x i8>\n  %x = add <16 x i8> %b, <i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1, i8 1>\n  %y = extractelement <16 x i8> %x, i32 9\n  %z = bitcast <16 x i8> %x to <4 x i32>\n  %w = extractelement <4 x i32> %z, i32 2\n  %yw = zext i8 %y to i32\n  %r = xor i32 %yw, %w\n",
            oi("a", 0x0403_0201)
        ),
        0x0504_0302 ^ 3,
    ));
    v.push(k(
        "chunked_half_lanes_bitcast_through_words",
        format!(
            "{}  %v = insertelement <4 x float> zeroinitializer, float %f, i32 1\n  %h = bitcast <4 x float> %v to <8 x half>\n  %g = fadd <8 x half> %h, %h\n  %b = bitcast <8 x half> %g to <4 x i32>\n  %r = extractelement <4 x i32> %b, i32 1\n",
            of("f", f32::from_bits(0x3C00_4000))
        ),
        0x4000_4400,
    ));
    v.push(Kernel {
        name: "chunked_shuffles_widen_and_narrow",
        body: format!(
            "{}  %p = insertelement <4 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000>, float %a, i32 0\n  %q = shufflevector <4 x float> %p, <4 x float> <float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, <8 x i32> <i32 7, i32 6, i32 5, i32 4, i32 3, i32 2, i32 1, i32 0>\n  %s = shufflevector <8 x float> %q, <8 x float> poison, <3 x i32> <i32 7, i32 0, i32 4>\n  %e = extractelement <3 x float> %s, i32 0\n  %g = extractelement <3 x float> %s, i32 1\n  %r = fadd float %e, %g\n",
            of("a", 1.5)
        ),
        float: true,
        want: Some(9.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "one_lane_vector",
        body: format!(
            "{}  %v = insertelement <1 x float> poison, float %a, i32 0\n  %w = fmul <1 x float> %v, <float 3.0>\n  %r = extractelement <1 x float> %w, i32 0\n",
            of("a", 1.25)
        ),
        float: true,
        want: Some(3.75f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "five_lane_signed_division",
        format!(
            "{}  %v = insertelement <5 x i16> <i16 -10, i16 20, i16 30, i16 40, i16 -50>, i16 %a, i32 4\n  %w = sdiv <5 x i16> %v, <i16 3, i16 3, i16 3, i16 3, i16 3>\n  %x = extractelement <5 x i16> %w, i32 0\n  %y = extractelement <5 x i16> %w, i32 4\n  %xw = zext i16 %x to i32\n  %yw = zext i16 %y to i32\n  %ys = shl i32 %yw, 16\n  %r = or i32 %xw, %ys\n",
            on("a", 16, 7)
        ),
        0x0002_FFFD,
    ));
    v.push(Kernel {
        name: "chunked_math_convert_and_whole_select",
        body: format!(
            "{}{}  %v = insertelement <8 x float> <float 0x3FF0000000000000, float 0x4000000000000000, float 0x4008000000000000, float 0x4010000000000000, float 0x4014000000000000, float 0x4018000000000000, float 0x401C000000000000, float 0x4020000000000000>, float %a, i32 5\n  %m = call <8 x float> @air.fabs.v8f32(<8 x float> %v)\n  %n = fptosi <8 x float> %m to <8 x i32>\n  %lt = icmp slt i32 %i, 0\n  %s = select i1 %lt, <8 x i32> zeroinitializer, <8 x i32> %n\n  %z = freeze <8 x i32> %s\n  %f = sitofp <8 x i32> %z to <8 x float>\n  %r = extractelement <8 x float> %f, i32 5\n",
            of("a", -4.5),
            oi("i", 1)
        ),
        float: true,
        want: Some(4.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // `sincos` stores its cosine through the access family's store into the private cell the
    // kernel reads back; the interpreter does not model `Sin`/`Cos`.
    v.push(Kernel {
        name: "sincos_stores_the_cosine",
        body: format!(
            "  %c = alloca float, align 4\n{}  %s = call float @air.sincos.f32(float %a, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd float %s, %cv\n",
            of("a", 0.5)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(Kernel {
        name: "sincos_half_vector_rounds_both",
        body: format!(
            "  %c = alloca <2 x half>, align 4\n{}  %h = fptrunc float %a to half\n  %v0 = insertelement <2 x half> zeroinitializer, half %h, i32 0\n  %v = insertelement <2 x half> %v0, half %h, i32 1\n  %s = call <2 x half> @air.sincos.v2f16(<2 x half> %v, ptr %c)\n  %cv = load <2 x half>, ptr %c, align 4\n  %t = fadd <2 x half> %s, %cv\n  %e = extractelement <2 x half> %t, i32 1\n  %r = fpext half %e to float\n",
            of("a", 0.25)
        ),
        float: true,
        want: None,
        needs: &[],
        note: None,
    });
    // `dot` of chunked vectors, under the exact maths these kernels compile with, is Metal's fused
    // chain over every lane (`wide_dot_is_metals_fused_chain_in_metals_order`), with no note;
    // `sincos` of one computes per chunk and stores its cosine as one chunked leaf.
    v.push(Kernel {
        name: "chunked_dot_sums_every_chunk",
        body: format!(
            "{}  %v = insertelement <8 x float> <float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0>, float %a, i32 7\n  %w = shufflevector <8 x float> %v, <8 x float> poison, <5 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7>\n  %d8 = call float @air.dot.v8f32(<8 x float> %v, <8 x float> %v)\n  %d5 = call float @air.dot.v5f32(<5 x float> %w, <5 x float> %w)\n  %r = fadd float %d8, %d5\n",
            of("a", 0.5)
        ),
        float: true,
        // 1 + 4 + ... + 49 + 0.25 = 140.25, and 1 + 4 + 9 + 16 + 0.25 = 30.25.
        want: Some(170.5f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "chunked_half_dot",
        body: format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <5 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, half %h, i32 4\n  %d = call half @air.dot.v5f16(<5 x half> %v, <5 x half> %v)\n  %r = fpext half %d to float\n",
            of("a", 1.5)
        ),
        float: true,
        // 4 * 1 + 1.5^2 = 6.25, a half.
        want: Some(6.25f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "chunked_sincos_stores_the_cosine",
        body: format!(
            "  %c = alloca <5 x float>, align 32\n{}  %v = insertelement <5 x float> zeroinitializer, float %a, i32 4\n  %s = call <5 x float> @air.sincos.v5f32(<5 x float> %v, ptr %c)\n  %cv = load <5 x float>, ptr %c, align 32\n  %t = fadd <5 x float> %s, %cv\n  %r = extractelement <5 x float> %t, i32 4\n",
            of("a", 0.5)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(Kernel {
        name: "pow_notes_device_transcendentals",
        body: format!(
            "{}  %r = call float @air.pow.f32(float %a, float 3.0)\n",
            of("a", -2.0)
        ),
        float: true,
        want: None,
        needs: &[],
        note: Some(PrecisionNote::DeviceTranscendentals),
    });
    v.push(k(
        "mad_sat_saturates_the_exact_result",
        format!(
            "{}{}  %s = call i8 @air.mad_sat.s.i8(i8 %a, i8 %a, i8 -100)\n  %u = call i32 @air.mad_sat.u.i32(i32 %b, i32 %b, i32 7)\n  %sw = zext i8 %s to i32\n  %r = xor i32 %sw, %u\n",
            on("a", 8, 15),
            oi("b", 0x0001_0000)
        ),
        // 15 * 15 - 100 = 125; 2^32 + 7 saturates at 0xFFFFFFFF.
        125 ^ 0xFFFF_FFFF,
    ));
    v.push(Kernel {
        name: "extract_bits_on_i64",
        body: format!(
            "{}  %w = zext i32 %a to i64\n  %h = shl i64 %w, 28\n  %e = call i64 @air.extract_bits.s.i64(i64 %h, i32 28, i32 8)\n  %r = trunc i64 %e to i32\n",
            oi("a", 0x80)
        ),
        float: false,
        want: Some(0xFFFF_FF80),
        needs: &[],
        note: None,
    });
    // 0.25 * 255 = 63.75 -> 64; 0.5 * 255 = 127.5 -> 128 (ties to even).
    v.push(k(
        "pack_unorm4x8_rounds_to_even",
        format!(
            "{}  %v = insertelement <4 x float> <float 0.0, float 0.5, float 1.0, float 1.0>, float %a, i32 0\n  %r = call i32 @air.pack.unorm4x8.v4f32(<4 x float> %v)\n",
            of("a", 0.25)
        ),
        0xFFFF_8040,
    ));
    // 0x7FFF is 1.0; 0x8000 is -32768 / 32767, clamped to -1.0.
    v.push(Kernel {
        name: "unpack_snorm2x16_clamps",
        body: format!(
            "{}  %u = call <2 x float> @air.unpack.snorm2x16.v2f32(i32 %b)\n  %x = extractelement <2 x float> %u, i32 0\n  %y = extractelement <2 x float> %u, i32 1\n  %r = fadd float %x, %y\n",
            oi("b", 0x8000_7FFF)
        ),
        float: true,
        want: Some(0),
        needs: &[],
        // The division is exact.
        note: None,
    });
    // An 8-bit float to an integer is its value under the saturating rule (448 to `char` is 127,
    // -1.0 to `uint` is 0, 57344 and -2.0 to 64 bits, held as two words where the profile lacks
    // `shaderInt64`), and records `Fp8Value`; a widening to a float does not.
    for (name, call, bits, wide, want) in [
        (
            "fp8_e4m3fn_to_char_saturates",
            "i8 @air.convert.s.i8.f.f8e4m3fn",
            0x7Eu32,
            false,
            127u32,
        ),
        (
            "fp8_e4m3_to_uint_clamps_at_zero",
            "i32 @air.convert.u.i32.f.f8e4m3",
            0xB8,
            false,
            0,
        ),
        (
            "fp8_e5m2_to_ulong",
            "i64 @air.convert.u.i64.f.f8e5m2",
            0x7B,
            true,
            57344,
        ),
        (
            "fp8_e5m2_to_long",
            "i64 @air.convert.s.i64.f.f8e5m2",
            0xC0,
            true,
            0xFFFF_FFFE,
        ),
    ] {
        let width = call.split(' ').next().unwrap_or_default();
        let narrow = if wide {
            format!("  %r = trunc {width} %c to i32\n")
        } else if width == "i32" {
            "  %r = add i32 %c, 0\n".to_owned()
        } else {
            format!("  %r = sext {width} %c to i32\n")
        };
        v.push(Kernel {
            name,
            body: format!("{}  %c = call {call}(i8 %a)\n{narrow}", on("a", 8, bits)),
            float: false,
            want: Some(want),
            needs: &[],
            note: Some(PrecisionNote::Fp8Value),
        });
    }
    v.push(Kernel {
        name: "fp8_e5m2_to_float_is_exact",
        body: format!(
            "{}  %r = call float @air.convert.f.f32.f.f8e5m2(i8 %a)\n",
            on("a", 8, 0x3C)
        ),
        float: true,
        want: Some(1.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // Metal's `snorm` `rgb10a2` words (Apple M3 Max): code 65 unpacks to its product with
    // fl(1/511), one ULP below 65/511; just below 1.5/511 packs to 2 in each 10-bit field (the f32
    // product rounds onto the tie); a NaN packs to -1 in every field.
    v.push(Kernel {
        name: "unpack_snorm_rgb10a2_multiplies_by_the_reciprocal",
        body: format!(
            "{}  %u = call <4 x float> @air.unpack.snorm.rgb10a2.v4f32(i32 %b)\n  %r = extractelement <4 x float> %u, i32 0\n",
            oi("b", 0x4411_0441)
        ),
        float: true,
        want: Some(0x3E02_4120),
        needs: &[],
        note: None,
    });
    for (name, x, want) in [
        (
            "pack_snorm_rgb10a2_rounds_the_float_product",
            "0x3F680C0600000000",
            0x0020_0802u32,
        ),
        (
            "pack_snorm_rgb10a2_packs_nan_as_minus_one",
            "0x7FF8000000000000",
            0xE018_0601,
        ),
    ] {
        v.push(k(
            name,
            format!(
                "  %a = fadd float {x}, -0.0\n  %v0 = insertelement <4 x float> undef, float %a, i32 0\n  %v1 = insertelement <4 x float> %v0, float %a, i32 1\n  %v2 = insertelement <4 x float> %v1, float %a, i32 2\n  %v = insertelement <4 x float> %v2, float %a, i32 3\n  %r = call i32 @air.pack.snorm.rgb10a2.v4f32(<4 x float> %v)\n"
            ),
            want,
        ));
    }
    // Metal's sRGB tables decode every byte to a step that encodes back to the byte (the alpha byte
    // through `a / 255` and the exact product), exactly on every device.
    v.push(k(
        "srgb_unpack_and_pack_round_trip",
        format!(
            "{}  %u = call <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32 %b)\n  %r = call i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float> %u)\n",
            oi("b", 0x8901_37FF)
        ),
        0x8901_37FF,
    ));
    v.push(Kernel {
        name: "double_needs_float64",
        body: format!(
            "{}  %d = fpext float %a to double\n  %s = fadd double %d, 0x3FF8000000000000\n  %r = fptrunc double %s to float\n",
            of("a", 2.0)
        ),
        float: true,
        want: Some(3.5f32.to_bits()),
        needs: &[Need::Float64],
        note: None,
    });
    // Half and bfloat are held widened in f32 on every profile, and each operation rounds to the
    // format: 2049 rounds to 2048 and 2048 + 1 rounds back to 2048 (ties to even), where f32
    // arithmetic would give 2050.
    v.push(Kernel {
        name: "half_rounds_each_operation",
        body: format!(
            "{}  %h = fptrunc float %a to half\n  %s = fadd half %h, 0xH3C00\n  %r = fpext half %s to float\n",
            of("a", 2049.0)
        ),
        float: true,
        want: Some(2048.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(Kernel {
        name: "bfloat_rounds_each_operation",
        body: format!(
            "{}  %b = fptrunc float %a to bfloat\n  %s = fadd bfloat %b, 0xR3F80\n  %r = fpext bfloat %s to float\n",
            of("a", 256.0)
        ),
        float: true,
        want: Some(256.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    v.push(k(
        "half_vector_bitcast",
        format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <2 x half> <half 0xH3C00, half 0xH4000>, half %h, i32 1\n  %r = bitcast <2 x half> %v to i32\n",
            of("a", 3.0)
        ),
        0x4200_3C00,
    ));
    // Three 16-bit lanes (48 bits) move lane by lane, with no 64-bit integer.
    v.push(k(
        "half3_bitcast_to_short3",
        format!(
            "{}  %h = fptrunc float %a to half\n  %v = insertelement <3 x half> <half 0xH3C00, half 0xH4000, half 0xH4400>, half %h, i32 1\n  %b = bitcast <3 x half> %v to <3 x i16>\n  %x = extractelement <3 x i16> %b, i32 1\n  %y = extractelement <3 x i16> %b, i32 2\n  %xw = zext i16 %x to i32\n  %yw = zext i16 %y to i32\n  %s = shl i32 %yw, 16\n  %r = or i32 %s, %xw\n",
            of("a", 3.0)
        ),
        0x4400_4200,
    ));
    v.push(Kernel {
        name: "short3_bitcast_to_half3",
        body: format!(
            "{}  %v = insertelement <3 x i16> <i16 15360, i16 0, i16 17408>, i16 %x, i32 1\n  %h = bitcast <3 x i16> %v to <3 x half>\n  %e = extractelement <3 x half> %h, i32 1\n  %f = extractelement <3 x half> %h, i32 2\n  %s = fadd half %e, %f\n  %r = fpext half %s to float\n",
            on("x", 16, 0x4200)
        ),
        float: true,
        want: Some(7.0f32.to_bits()),
        needs: &[],
        note: None,
    });
    // 2^24 + 2^16 + 1 is just above a bfloat tie; rounding through a plain f32 would land on
    // the tie and round down to 2^24.
    v.push(Kernel {
        name: "int_to_bfloat_rounds_once",
        body: format!(
            "{}  %b = sitofp i32 %a to bfloat\n  %r = fpext bfloat %b to float\n",
            oi("a", 0x0101_0001)
        ),
        float: true,
        want: Some(0x4B81_0000),
        needs: &[],
        note: None,
    });
    // 1 + 2^-11 + 2^-40 is just above a half tie; rounding through a plain f32 would land on
    // the tie and round down to 1.
    v.push(Kernel {
        name: "double_to_half_rounds_once",
        body: format!(
            "{}  %d = fpext float %a to double\n  %e = fadd double %d, 0x3D70000000000000\n  %h = fptrunc double %e to half\n  %r = fpext half %h to float\n",
            of("a", 1.0 + 2f32.powi(-11))
        ),
        float: true,
        want: Some((1.0f32 + 2f32.powi(-10)).to_bits()),
        needs: &[Need::Float64],
        note: None,
    });
    v
}

const DECLS: &str = "
declare i32 @llvm.umin.i32(i32, i32)
declare <4 x i1> @air.convert.u.v4i1.u.v4i8(<4 x i8>)
declare i1 @air.convert.u.i1.s.i32(i32)
declare i16 @llvm.ctpop.i16(i16)
declare i32 @llvm.ctlz.i32(i32, i1)
declare i32 @llvm.fshl.i32(i32, i32, i32)
declare i8 @llvm.abs.i8(i8, i1)
declare i8 @air.add_sat.s.i8(i8, i8)
declare i32 @air.convert.s.i32.f.f32(float)
declare i8 @air.convert.u.i8.f.f32(float)
declare float @air.clamp.f32(float, float, float)
declare float @air.round.f32(float)
declare float @air.fract.f32(float)
declare float @air.sin.f32(float)
declare float @air.sincos.f32(float, ptr)
declare <8 x float> @air.fabs.v8f32(<8 x float>)
declare i16 @air.pack.unorm.rgb565.v3f32(<3 x float>)
declare <3 x float> @air.unpack.unorm.rg11b10f.v3f32(i32)
declare <2 x half> @air.sincos.v2f16(<2 x half>, ptr)
declare float @air.dot.v8f32(<8 x float>, <8 x float>)
declare float @air.dot.v5f32(<5 x float>, <5 x float>)
declare half @air.dot.v5f16(<5 x half>, <5 x half>)
declare <5 x float> @air.sincos.v5f32(<5 x float>, ptr)
declare float @air.pow.f32(float, float)
declare i8 @air.mad_sat.s.i8(i8, i8, i8)
declare i32 @air.mad_sat.u.i32(i32, i32, i32)
declare i64 @air.mad_sat.s.i64(i64, i64, i64)
declare i64 @air.mad_sat.u.i64(i64, i64, i64)
declare <5 x i64> @llvm.umax.v5i64(<5 x i64>, <5 x i64>)
declare i64 @air.extract_bits.s.i64(i64, i32, i32)
declare i32 @air.pack.unorm4x8.v4f32(<4 x float>)
declare <2 x float> @air.unpack.snorm2x16.v2f32(i32)
declare <4 x float> @air.unpack.unorm4x8.srgb.v4f32(i32)
declare i32 @air.pack.unorm4x8.srgb.v4f32(<4 x float>)
declare float @llvm.fmuladd.f32(float, float, float)
declare float @llvm.sqrt.f32(float)
declare float @llvm.exp2.f32(float)
declare float @llvm.log2.f32(float)
declare float @llvm.copysign.f32(float, float)
declare i8 @air.convert.s.i8.f.f8e4m3fn(i8)
declare i32 @air.convert.u.i32.f.f8e4m3(i8)
declare i64 @air.convert.u.i64.f.f8e5m2(i8)
declare i64 @air.convert.s.i64.f.f8e5m2(i8)
declare float @air.convert.f.f32.f.f8e5m2(i8)
declare <4 x float> @air.unpack.snorm.rgb10a2.v4f32(i32)
declare i32 @air.pack.snorm.rgb10a2.v4f32(<4 x float>)
";

/// The generated module with every kernel of `ks`.
fn module(ks: &[Kernel]) -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for kk in ks {
        let ty = if kk.float { "float" } else { "i32" };
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{}  store {ty} %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
            kk.name, kk.body
        ));
    }
    s.push_str(DECLS);
    let refs: Vec<String> = (0..ks.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, kk) in ks.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, kk.name));
    }
    s
}

/// Assemble `text`, or `None` (skipped visibly) without `llvm-as`.
fn assemble(tag: &str, text: &str) -> Option<Vec<u8>> {
    let scratch = Scratch::new(&format!("m2v-arith-{tag}"));
    let dir = scratch.path();
    let ll = dir.join("arith.ll");
    std::fs::write(&ll, text).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the arith module: {e}"),
    }
}

/// Four scalar and one two-lane 64-bit division.
const WIDE_DIVISIONS: &str = "wide_divisions_share_their_routine";

fn has(p: &DeviceProfile, n: Need) -> bool {
    let l = &p.fields().leaf;
    match n {
        Need::Float64 => l.float64,
    }
}

/// The typed module of `entry` on `p`, run on the interpreter: the stored word.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile) -> u32 {
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, p, &mut m).expect("memory");
    let plan = select(&spec, &memory, p, &LowerInputs::new(), &mut m).expect("select");
    let module = emit_module(&spec, &plan, &mut m).expect("emit").module;
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0; 4],
        }],
        push: vec![0; 4],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = out.buffers.first().expect("buffer");
    u32::from_le_bytes([b[0], b[1], b[2], b[3]])
}

#[test]
fn arith_kernels_validate_and_compute_on_every_profile() {
    let ks = kernels();
    let Some(bitcode) = assemble("kernels", &module(&ks)) else {
        return;
    };
    // Every need is both met (and computed) and refused on some profile.
    let (mut met, mut refused) = (Vec::new(), Vec::new());
    let mut computed = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let missing = kk.needs.iter().copied().find(|n| !has(&p, *n));
            match (compile(&bitcode, kk.name, &p), missing) {
                (Ok(a), None) => {
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                    // The plan's estimate bounds the emitted words.
                    let plan = plan_bitcode(
                        &bitcode,
                        EntryRef::Name(kk.name),
                        &FunctionConstantValues::defaults(),
                        &p,
                        &LowerInputs::new(),
                        &mut meter(),
                    )
                    .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
                    let words = a.module.words.len() as u64;
                    assert!(
                        plan.estimate.words >= words,
                        "{what}: estimate {} < emitted {words}",
                        plan.estimate.words
                    );
                    if let Some(n) = kk.note {
                        assert!(a.precision.contains(&n), "{what}: {:?}", a.precision);
                    }
                    // Only an 8-bit float to integer conversion reads the unverified value.
                    assert_eq!(
                        a.precision.contains(&PrecisionNote::Fp8Value),
                        kk.note == Some(PrecisionNote::Fp8Value),
                        "{what}: {:?}",
                        a.precision
                    );
                    assert!(
                        a.strategies.entries.keys().any(|n| n.starts_with("arith.")),
                        "{what}: {:?}",
                        a.strategies.entries
                    );
                    if let Some(want) = kk.want {
                        let got = interpret(&bitcode, kk.name, &p);
                        assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
                        computed += 1;
                    }
                    met.extend(kk.needs.iter().copied());
                }
                (Err(Error::Unsupported(u)), Some(n)) => {
                    let (feature, needs) = match n {
                        Need::Float64 => (
                            Feature::Float64Arithmetic,
                            Needs::AnyOf(vec![vec![Capability::Feature(VkFeature::ShaderFloat64)]]),
                        ),
                    };
                    assert_eq!((u.feature, u.needs), (feature, needs), "{what}");
                    assert!(u.location.is_some(), "{what}: located");
                    refused.push(n);
                }
                (other, missing) => panic!("{what}: missing {missing:?}, got {other:?}"),
            }
        }
    }
    assert!(
        met.contains(&Need::Float64) && refused.contains(&Need::Float64),
        "Float64 met and refused"
    );
    assert!(
        computed >= PROFILES.len() * 10,
        "{computed} kernels computed"
    );
}

/// The number of `op` instructions in module `words`.
fn count_op(words: &[u32], op: crate::spv::Op) -> usize {
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        n += usize::from(opcode == op as u32);
        i += len;
    }
    n
}

/// The function control of each `OpFunction` in module `words`, in order.
fn function_controls(words: &[u32]) -> Vec<u32> {
    let mut i = 5;
    let mut controls = Vec::new();
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Function as u32 {
            controls.push(words[i + 3]);
        }
        i += len;
    }
    controls
}

/// Without `shaderInt64` the wide division is one function per lane count, which each wide division
/// on that many lanes calls, so a module holds it once per lane count, not once per use; with
/// `shaderInt64` there is no routine. No function takes a control on these profiles; on the same
/// device under MoltenVK 1.4.1, whose SPIRV-Cross writes every function inlined at each call
/// (`DriverWorkaround::FunctionsInlinedAtEveryCall`), each routine is declared `DontInline` and the
/// entry is not.
#[test]
fn wide_divisions_share_one_routine_per_form() {
    let ks: Vec<Kernel> = kernels()
        .into_iter()
        .filter(|kk| kk.name == WIDE_DIVISIONS)
        .collect();
    let Some(bitcode) = assemble("divisions", &module(&ks)) else {
        return;
    };
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles/vk12-floor.json");
    let mut desc = DeviceDescription::from_khronos_json(
        &std::fs::read_to_string(&path).expect("profile"),
        None,
    )
    .expect("description");
    desc.driver_id = Some(crate::vk::DriverId::MOLTENVK);
    desc.driver_version = 10401;
    let converted = Device::new(desc, exact_maths())
        .expect("device")
        .profile()
        .clone();
    assert!(converted
        .fields()
        .workarounds
        .contains(&DriverWorkaround::FunctionsInlinedAtEveryCall));
    let cases = [
        (
            "vk12-floor",
            profile("vk12-floor", exact_maths()).expect("profile"),
            3,
            5,
            0,
        ),
        ("vk12-floor under MoltenVK 1.4.1", converted, 3, 5, 0x2),
        (
            "amd-rdna3-radv",
            profile("amd-rdna3-radv", exact_maths()).expect("profile"),
            1,
            0,
            0,
        ),
    ];
    for (stem, p, functions, calls, routine_control) in cases {
        let a = compile(&bitcode, WIDE_DIVISIONS, &p).unwrap_or_else(|e| panic!("{stem}: {e}"));
        let words = &a.module.words;
        assert_eq!(
            (
                count_op(words, crate::spv::Op::Function),
                count_op(words, crate::spv::Op::FunctionCall)
            ),
            (functions, calls),
            "{stem}: the entry and one routine per lane count (i64, <2 x i64>), one call per division"
        );
        let mut controls = function_controls(words);
        controls.sort_unstable();
        let mut want = vec![routine_control; functions - 1];
        want.insert(0, 0);
        assert_eq!(
            controls, want,
            "{stem}: each routine's control, and none on the entry"
        );
    }
}

/// `float` and `half` fused multiply-adds (`air.fma`, `llvm.fma`, `llvm.fmuladd`) round once on
/// every profile, by the profile's route: `OpFmaKHR` where `shaderFmaFloat32` is enabled
/// (moltenvk-m3), else one shared `double` routine (lavapipe) or integer routine (vk12-floor) per
/// lane count, which every use on that many lanes calls; called on one and two lanes, the two-lane
/// routine calls the one-lane routine on each lane. A1 case 52343cc4's lane (`a = 1 + 2^-23`, `b =
/// 1 - 2^-24`, `c = -RN(a * b)`), on one lane and on a vector's first, gives the product's rounding
/// error, which an unfused `Fma` loses; a `half` triple whose `float` sum is a `half` midpoint
/// gives the once-rounded `half`: through the shared `half` routine (lavapipe, vk12-floor), or
/// inline as the device's 16-bit `Fma`, which the driver computes as Metal's `half` `fma`
/// (moltenvk-m3). No module notes `DeviceArithmetic` or `F16FmaDoubleRounding`.
#[test]
fn fused_multiply_adds_round_once_on_every_route() {
    let (a, b) = (1.0f32 + f32::EPSILON, 1.0f32 - f32::EPSILON / 2.0);
    let c = -(a * b);
    let fused = a.mul_add(b, c);
    assert_ne!(fused, 0.0);
    let inputs = format!("{}{}{}", of("a", a), of("b", b), of("c", c));
    // `<2 x float>` lanes (a, 2), (b, 3), (c, 1): the fused residue and 7.
    let vector = "  %va0 = insertelement <2 x float> <float 0.0, float 2.0>, float %a, i32 0\n  %vb0 = insertelement <2 x float> <float 0.0, float 3.0>, float %b, i32 0\n  %vc0 = insertelement <2 x float> <float 0.0, float 1.0>, float %c, i32 0\n";
    let float_kernel = Kernel {
        name: "fma_float_rounds_once",
        body: format!(
            "{inputs}{vector}  %s = call float @air.fma.f32(float %a, float %b, float %c)\n  %m = call float @llvm.fmuladd.f32(float %a, float %b, float %c)\n  %v = call <2 x float> @air.fma.v2f32(<2 x float> %va0, <2 x float> %vb0, <2 x float> %vc0)\n  %v0 = extractelement <2 x float> %v, i32 0\n  %v1 = extractelement <2 x float> %v, i32 1\n  %si = bitcast float %s to i32\n  %mi = bitcast float %m to i32\n  %v0i = bitcast float %v0 to i32\n  %v1i = bitcast float %v1 to i32\n  %x = xor i32 %si, %mi\n  %y = xor i32 %si, %v0i\n  %z = or i32 %x, %y\n  %r = xor i32 %z, %v1i\n"
        ),
        float: false,
        // s, m and the vector's first lane are the same fused residue: their xors are 0,
        // leaving 7.0's bits.
        want: Some(7.0f32.to_bits()),
        needs: &[],
        note: None,
    };
    // 0x400a * 0x0bec + 0xbc9f: the exact sum rounds to 0xbc9f, the float sum to a half
    // midpoint that rounds away.
    let half_kernel = Kernel {
        name: "fma_half_rounds_once",
        body: format!(
            "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n  %x = call half @air.fma.f16(half %h, half %g, half %u)\n  %r = fpext half %x to float\n",
            on("t", 16, 0x400A),
            on("s", 16, 0x0BEC),
            on("v", 16, 0xBC9F)
        ),
        float: true,
        want: Some((-1183.0f32 / 1024.0).to_bits()),
        needs: &[],
        note: None,
    };
    let ks = [float_kernel, half_kernel];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.fma.f32(float, float, float)\ndeclare <2 x float> @air.fma.v2f32(<2 x float>, <2 x float>, <2 x float>)\ndeclare half @air.fma.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("fma-routes", &text) else {
        return;
    };
    // (profile, OpFmaKHR, functions, calls) of the float kernel: the entry and one routine per
    // lane count (scalar, two lanes), one call per use, and the two-lane routine's call per lane.
    // And (functions, calls) of the half kernel: the entry and the `half` routine's call, or the
    // entry alone.
    for (stem, khr, functions, calls, half) in [
        ("moltenvk-m3", 3, 1, 0, (1, 0)),
        ("lavapipe", 0, 3, 5, (2, 1)),
        ("vk12-floor", 0, 3, 5, (2, 1)),
    ] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for n in [
                PrecisionNote::DeviceArithmetic,
                PrecisionNote::F16FmaDoubleRounding,
                PrecisionNote::FastMathRewrites,
            ] {
                assert!(!a.precision.contains(&n), "{what}: {:?}", a.precision);
            }
            if kk.name == "fma_float_rounds_once" {
                assert_eq!(
                    (
                        count_op(&words, crate::spv::Op::FmaKHR),
                        count_op(&words, crate::spv::Op::Function),
                        count_op(&words, crate::spv::Op::FunctionCall)
                    ),
                    (khr, functions, calls),
                    "{what}: OpFmaKHR, functions, calls"
                );
            } else {
                assert_eq!(
                    (
                        count_op(&words, crate::spv::Op::Function),
                        count_op(&words, crate::spv::Op::FunctionCall)
                    ),
                    half,
                    "{what}: functions, calls"
                );
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// `<n x ty>` `%name` of opaque lanes `vals` (each `fadd <v>, -0.0`, a `half` lane truncated
/// from its exact `float`).
fn opaque_vector(name: &str, half: bool, vals: &[f32]) -> String {
    let ty = if half { "half" } else { "float" };
    let n = vals.len();
    let mut s = String::new();
    let mut prev = "poison".to_owned();
    for (i, v) in vals.iter().enumerate() {
        let lane = format!("{name}.{i}");
        s.push_str(&of(&lane, *v));
        let mut at = format!("%{lane}");
        if half {
            s.push_str(&format!("  %{lane}.h = fptrunc float %{lane} to half\n"));
            at = format!("%{lane}.h");
        }
        let next = if i + 1 == n {
            format!("%{name}")
        } else {
            format!("%{name}.v{i}")
        };
        s.push_str(&format!(
            "  {next} = insertelement <{n} x {ty}> {prev}, {ty} {at}, i32 {i}\n"
        ));
        prev = next;
    }
    s
}

/// Metal's `dot` on `float` and `half` is the products' fused chain, `fma(x[n-1], y[n-1], ...
/// fma(x[1], y[1], x[0] * y[0]))`, with and without fast math, on every profile: the multiply and
/// `OpFmaKHR` (amd-rdna3-radv), or the shared routine of the profile's route (lavapipe's `double`
/// one, vk12-floor's and lunarg-desktop-baseline-2023's integer one), and on moltenvk-m3, whose
/// `float` functions are Metal's, the device's `OpDot` on `float`, which reaches Metal's own `dot`.
/// A `half` is summed in `half` by the chain on every profile. The words are Metal's (a probe, the
/// M3 Max and the M4 alike, at lanes where no other order, tree, unfused or exact sum gives them),
/// and fast math's zero signs where a device that regroups `dot(float2(a, b), float2(b, c))` as
/// `b * (a + c)` misses them (`nan_denormals_fast` with zeros for its denormals;
/// `nan_functions_fast`'s `dot((-0, +0), (+0, -0))`). No `dot` notes `DeviceArithmetic` or
/// `DeviceDenormals`.
#[test]
fn dot_is_metals_fused_chain_on_every_profile() {
    let f = |b: u32| f32::from_bits(b);
    let x = [0x3F2121E3, 0xC1504426, 0x40D96C9D, 0x3EC6C559].map(f);
    let y = [0xBFD689B8, 0xC0608ADE, 0xC0D7159F, 0x402DE9ED].map(f);
    let hx = [0xB3DD, 0xB0C3, 0xBB51, 0x3CD8].map(f16_bits_to_f32);
    let hy = [0xC76B, 0x4A20, 0x31C7, 0xBA80].map(f16_bits_to_f32);
    let mut ks = Vec::new();
    let names = [
        ["dot2_float", "dot3_float", "dot4_float"],
        ["dot2_float_fast", "dot3_float_fast", "dot4_float_fast"],
    ];
    for (fast, row) in names.iter().enumerate() {
        for (k, (name, want)) in row
            .iter()
            .copied()
            .zip([0x42327452u32, 0xBF87085A, 0x37A4C7B4])
            .enumerate()
        {
            let n = k + 2;
            let flag = if fast == 1 { "fast " } else { "" };
            ks.push(Kernel {
                name,
                body: format!(
                    "{}{}  %d = call {flag}float @air.dot.v{n}f32(<{n} x float> %x, <{n} x float> %y)\n  %r = bitcast float %d to i32\n",
                    opaque_vector("x", false, &x[..n]),
                    opaque_vector("y", false, &y[..n])
                ),
                float: false,
                want: Some(want),
                needs: &[],
                note: None,
            });
        }
    }
    for (k, (name, want)) in ["dot2_half", "dot3_half", "dot4_half"]
        .into_iter()
        .zip([0x0E80u32, 0xB145, 0xBC98])
        .enumerate()
    {
        let n = k + 2;
        ks.push(Kernel {
            name,
            body: format!(
                "{}{}  %d = call half @air.dot.v{n}f16(<{n} x half> %x, <{n} x half> %y)\n  %u = bitcast half %d to i16\n  %r = zext i16 %u to i32\n",
                opaque_vector("x", true, &hx[..n]),
                opaque_vector("y", true, &hy[..n])
            ),
            float: false,
            want: Some(want),
            needs: &[],
            note: None,
        });
    }
    // fma(-2^-126, -0, +0 * -2^-126) is +0; `b * (a + c)` is -0.
    let tiny = f(0x8080_0000);
    ks.push(Kernel {
        name: "dot_fast_zero_sign",
        body: format!(
            "{}{}  %d = call fast float @air.dot.v2f32(<2 x float> %x, <2 x float> %y)\n  %r = bitcast float %d to i32\n",
            opaque_vector("x", false, &[0.0, tiny]),
            opaque_vector("y", false, &[tiny, -0.0])
        ),
        float: false,
        want: Some(0),
        needs: &[],
        note: None,
    });
    // fma(+0, -0, -0 * +0) is -0.
    ks.push(Kernel {
        name: "dot_fast_negative_zero",
        body: format!(
            "{}{}  %d = call fast float @air.dot.v2f32(<2 x float> %x, <2 x float> %y)\n  %r = bitcast float %d to i32\n",
            opaque_vector("x", false, &[-0.0, 0.0]),
            opaque_vector("y", false, &[0.0, -0.0])
        ),
        float: false,
        want: Some(0x8000_0000),
        needs: &[],
        note: None,
    });
    let decls: String = [2, 3, 4]
        .iter()
        .map(|n| {
            format!(
                "declare float @air.dot.v{n}f32(<{n} x float>, <{n} x float>)\ndeclare half @air.dot.v{n}f16(<{n} x half>, <{n} x half>)\n"
            )
        })
        .collect();
    let text = module(&ks).replace(DECLS, &format!("{DECLS}{decls}"));
    let Some(bitcode) = assemble("dot-chain", &text) else {
        return;
    };
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "lavapipe",
        "vk12-floor",
        "lunarg-desktop-baseline-2023",
    ] {
        let p = profile(stem, exact_maths()).expect("profile");
        let metal = p.fields().float_functions == FloatFunctions::Metal;
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for n in [
                PrecisionNote::DeviceArithmetic,
                PrecisionNote::DeviceDenormals,
            ] {
                assert!(!a.precision.contains(&n), "{what}: {:?}", a.precision);
            }
            let device_dot = metal && !kk.name.ends_with("_half");
            assert_eq!(
                count_op(&words, crate::spv::Op::Dot) > 0,
                device_dot,
                "{what}: OpDot"
            );
            // The interpreter is no Metal: moltenvk-m3's `OpDot` is its device's.
            if !device_dot {
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
        }
    }
}

/// Metal's `dot` of more than four lanes, on `float` and `half`, is the same fused chain over
/// every lane in order, with and without fast math: an Apple M3 Max under macOS
/// 27.0.1, 16,384 rows of seeded random lanes each on 5, 8 and 16 `float` lanes and 8 and 16 `half`
/// lanes, every answer the chain's. Each row here is one where the chain backwards, the chunks'
/// sums added and the exact sum rounded once all miss Metal's word. Where Metal's order is the
/// profile's (`RoundingOrderPolicy::Exact`; moltenvk-m3 under every policy) every profile answers it
/// by the chain ([`ArithStrategy::ChunkedChain`]): no `OpDot`, no `DeviceArithmetic` and no
/// `DeviceDenormals`. Under the default, a profile of the device's order keeps each chunk's
/// `OpDot`, noted `DeviceArithmetic`.
///
/// [`ArithStrategy::ChunkedChain`]: crate::lower::arith::ArithStrategy::ChunkedChain
#[test]
fn wide_dot_is_metals_fused_chain_in_metals_order() {
    let f = |b: &[u32]| b.iter().map(|&w| f32::from_bits(w)).collect::<Vec<_>>();
    let h = |b: &[u16]| b.iter().map(|&w| f16_bits_to_f32(w)).collect::<Vec<_>>();
    let x5 = f(&[0x3C4706A1, 0xC5D9EEB5, 0x3D6EA9B2, 0x3A2F23BF, 0xC603B91B]);
    let y5 = f(&[0x40986206, 0xBE6B6C10, 0xBC124CDB, 0xB8E3B629, 0x3E3C8570]);
    let x8 = f(&[
        0xBFC457BC, 0xBD86A520, 0x35B57EFC, 0x43A8E94D, 0x3ABA1FAF, 0xBF68F7B9, 0x411A7DC7,
        0xC0010B78,
    ]);
    let y8 = f(&[
        0xBAC18975, 0xB7E5CCAA, 0x3DCAA0D4, 0xB6B402AC, 0xC2D6EDAA, 0x397CD0F5, 0xC0DB6522,
        0xC2007DCD,
    ]);
    let x16 = f(&[
        0x3E0AEC21, 0xBDE4C98F, 0x48F7CE82, 0xC74A1CB5, 0x3FEFB46A, 0xB8CDB9CA, 0x367C7539,
        0xB7A71EB2, 0xC8A058F0, 0x41F42DA4, 0xBC9A925A, 0xC66A6596, 0xC0E26B92, 0x42AA4631,
        0x433758BE, 0xB93AF022,
    ]);
    let y16 = f(&[
        0x448360E6, 0xBC3FB11E, 0xBFED97A2, 0x3BBB547E, 0x3C051C11, 0xC4EE106E, 0x4521C04F,
        0x3B630B77, 0x3A529217, 0xC2FE1081, 0xC656A359, 0xC6FFF097, 0x3EB192E9, 0xC2147C43,
        0xC20B70CB, 0x40D2FDA7,
    ]);
    let hx8 = h(&[
        0x32B2, 0xC743, 0xAD26, 0x36C6, 0xC7E3, 0xBFF2, 0x43C1, 0x4AA1,
    ]);
    let hy8 = h(&[
        0xBD37, 0x321E, 0x3C57, 0x4039, 0xCC4E, 0x4F9E, 0xB7DC, 0xAC3B,
    ]);
    let hx16 = h(&[
        0x4391, 0x403F, 0xBC34, 0xC571, 0xB6BC, 0xB7B3, 0xB7EF, 0x3BA4, 0x2FD5, 0xB227, 0xB58C,
        0x2AFB, 0xAACF, 0x357C, 0x4DB6, 0xB40F,
    ]);
    let hy16 = h(&[
        0x2D47, 0xB043, 0xB15B, 0xAB19, 0x36CE, 0x3634, 0x45F8, 0x4F21, 0xAF75, 0xC50E, 0x2B33,
        0x2BFF, 0xCBB9, 0x3595, 0x3C83, 0xACEA,
    ]);
    let float = |name: &'static str, fast: bool, x: &[f32], y: &[f32], want: u32| {
        let n = x.len();
        let flag = if fast { "fast " } else { "" };
        Kernel {
            name,
            body: format!(
                "{}{}  %d = call {flag}float @air.dot.v{n}f32(<{n} x float> %x, <{n} x float> %y)\n  %r = bitcast float %d to i32\n",
                opaque_vector("x", false, x),
                opaque_vector("y", false, y)
            ),
            float: false,
            want: Some(want),
            needs: &[],
            note: None,
        }
    };
    let half = |name: &'static str, x: &[f32], y: &[f32], want: u32| {
        let n = x.len();
        Kernel {
            name,
            body: format!(
                "{}{}  %d = call half @air.dot.v{n}f16(<{n} x half> %x, <{n} x half> %y)\n  %u = bitcast half %d to i16\n  %r = zext i16 %u to i32\n",
                opaque_vector("x", true, x),
                opaque_vector("y", true, y)
            ),
            float: false,
            want: Some(want),
            needs: &[],
            note: None,
        }
    };
    let ks = [
        float("dot5_float", false, &x5, &y5, 0x424D5858),
        float("dot8_float", false, &x8, &y8, 0xBFCA9608),
        float("dot8_float_fast", true, &x8, &y8, 0xBFCA9608),
        float("dot16_float", false, &x16, &y16, 0x4DE9E2D2),
        half("dot8_half", &hx8, &hy8, 0x547C),
        half("dot16_half", &hx16, &hy16, 0x5284),
    ];
    // `DECLS` declares the dots of 5 and 8 `float` lanes.
    let decls = "declare float @air.dot.v16f32(<16 x float>, <16 x float>)\n\
                 declare half @air.dot.v8f16(<8 x half>, <8 x half>)\n\
                 declare half @air.dot.v16f16(<16 x half>, <16 x half>)\n";
    let text = module(&ks).replace(DECLS, &format!("{DECLS}{decls}"));
    let Some(bitcode) = assemble("wide-dot-chain", &text) else {
        return;
    };
    let device = [
        PrecisionNote::DeviceArithmetic,
        PrecisionNote::DeviceDenormals,
    ];
    for stem in [
        "moltenvk-m3",
        "amd-rdna3-radv",
        "lavapipe",
        "vk12-floor",
        "lunarg-desktop-baseline-2023",
    ] {
        for (policy, exact) in [(exact_maths(), true), (DevicePolicy::default(), false)] {
            let p = profile(stem, policy).expect("profile");
            let chain = exact || p.fields().float_functions == FloatFunctions::Metal;
            for kk in &ks {
                let what = format!("{} on {stem} (exact: {exact})", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let device_dot = count_op(&words, crate::spv::Op::Dot) > 0;
                if chain {
                    assert!(!device_dot, "{what}: OpDot");
                    for n in device {
                        assert!(!a.precision.contains(&n), "{what}: {:?}", a.precision);
                    }
                    let got = interpret(&bitcode, kk.name, &p);
                    assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
                } else {
                    assert!(device_dot, "{what}: OpDot");
                    assert!(
                        a.precision.contains(&PrecisionNote::DeviceArithmetic),
                        "{what}: {:?}",
                        a.precision
                    );
                }
            }
        }
    }
}

/// `air.fast_fmod` on `float` is Metal's `fma(-trunc(x / y), y, x)` of its fast quotient (`x` times
/// the reciprocal of `y`), on every profile's fused multiply-add: `OpFmaKHR` (moltenvk-m3) or the
/// shared routine (lavapipe, vk12-floor). A1 case dd3b3b99's `fmod(1e10, 3)` is Metal's `-512` (the
/// quotient one ulp above the correctly rounded one), where `OpFRem`'s expression on the correctly
/// rounded quotient gives `0`. The module notes `DeviceArithmetic` (the reciprocal's rounding). On
/// moltenvk-m3, whose float functions and division are Metal's, the reciprocal is the device's
/// `OpFDiv(1, y)`; a profile with `OpFmaKHR` and without that fact refines it to `RN(1 / y)` by
/// four more `OpFmaKHR` (`fast_division_follows_the_metal_float_functions_fact`). The quotient is a
/// call of the fast division's routine, one function beside the entry (and the fused multiply-add's
/// routine where it has one).
#[test]
fn fast_fmod_is_metals_composition() {
    let kernel = Kernel {
        name: "fast_fmod_composes",
        body: format!(
            "{}{}  %f = call fast float @air.fast_fmod.f32(float %x, float %y)\n  %r = fadd float %f, 0.0\n",
            of("x", 1e10),
            of("y", 3.0)
        ),
        float: true,
        want: Some((-512.0f32).to_bits()),
        needs: &[],
        note: None,
    };
    let ks = [kernel];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fast_fmod.f32(float, float)\n"),
    );
    let Some(bitcode) = assemble("fast-fmod", &text) else {
        return;
    };
    for (stem, khr, functions) in [
        ("moltenvk-m3", 1, 2),
        ("lavapipe", 0, 3),
        ("vk12-floor", 0, 3),
    ] {
        let p = profile(stem, exact_maths()).expect("profile");
        let kk = &ks[0];
        let what = format!("{} on {stem}", kk.name);
        let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        assert!(
            a.precision.contains(&PrecisionNote::DeviceArithmetic),
            "{what}: {:?}",
            a.precision
        );
        assert_eq!(
            (
                count_op(&words, crate::spv::Op::FmaKHR),
                count_op(&words, crate::spv::Op::Function),
                count_op(&words, crate::spv::Op::FRem)
            ),
            (khr, functions, 0),
            "{what}: OpFmaKHR, functions, OpFRem"
        );
        let got = interpret(&bitcode, kk.name, &p);
        assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
    }
}

/// `fcmp one` is the negation of `OpFUnordEqual`, never `OpFOrdNotEqual`, on every profile:
/// SPIRV-Cross, which MoltenVK uses, renders `OpFOrdNotEqual` as MSL's `!=`, true for a NaN
/// operand, where `one` (and Metal, authored case df9e364a) answers false. The kernel's bits are
/// `one(NaN, 1)` (0), `one(1, 2)` (1), `one(2, 2)` (0), `ueq(NaN, 1)` (1), and the two lanes of
/// `one(<NaN, 1>, <1, 2>)` (0, 1).
#[test]
fn an_ordered_not_equal_is_the_negated_unordered_equal() {
    let compares = "  %c0 = fcmp one float %n, %one\n  %c1 = fcmp one float %one, %two\n  %c2 = fcmp one float %two, %two\n  %c3 = fcmp ueq float %n, %one\n  %va = insertelement <2 x float> <float 0.0, float 1.0>, float %n, i32 0\n  %vb = insertelement <2 x float> <float 0.0, float 2.0>, float %one, i32 0\n  %vc = fcmp one <2 x float> %va, %vb\n  %c4 = extractelement <2 x i1> %vc, i32 0\n  %c5 = extractelement <2 x i1> %vc, i32 1\n";
    let mut bits = String::from("  %b0 = zext i1 %c0 to i32\n");
    for i in 1..6 {
        bits.push_str(&format!(
            "  %z{i} = zext i1 %c{i} to i32\n  %s{i} = shl i32 %z{i}, {i}\n  %b{i} = or i32 %b{}, %s{i}\n",
            i - 1
        ));
    }
    let kernel = k(
        "ordered_not_equal",
        format!(
            "{}{}{}{compares}{bits}  %r = add i32 %b5, 0\n",
            of("n", f32::NAN),
            of("one", 1.0),
            of("two", 2.0)
        ),
        0b10_1010,
    );
    let ks = [kernel];
    let Some(bitcode) = assemble("ordered-not-equal", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        let kk = &ks[0];
        let what = format!("{} on {stem}", kk.name);
        let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        assert_eq!(
            count_op(&words, crate::spv::Op::FOrdNotEqual),
            0,
            "{what}: OpFOrdNotEqual"
        );
        // One per `one` and `ueq`; the vector `one` may be split into its lanes.
        assert!(
            count_op(&words, crate::spv::Op::FUnordEqual) >= 4,
            "{what}: OpFUnordEqual"
        );
        let got = interpret(&bitcode, kk.name, &p);
        assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
    }
}

/// Metal fuses a `contract` multiply into the `contract` addition or subtraction that uses it, in
/// the same block, when every use of the product does so (a probe cases e761c0aa and c3cc7397):
/// `a * a + c` with `a = 1 + 2^-12` and `c = -(1 + 2^-11)` is the product's rounding error `2^-24`
/// fused and 0 rounded twice. A product used by two additions is fused into both; `c - a * b` and
/// `a * b - c` negate a factor and the addend; of two products the first operand's is fused; a
/// `half` pair and a `half` product widened to `float` fuse too (`(1 + 2^-6)^2 - (1 + 2^-5) =
/// 2^-12`). A stored product, one reaching the addition through a `select`, and an addition without
/// `contract` stay two roundings. Every fused `float` pair is one `OpFmaKHR` on moltenvk-m3; none
/// of these shapes notes `FastMathRewrites`, while a negation between the product and the addition,
/// which the probe did not measure, does.
#[test]
fn contract_pairs_fuse_as_metal_fuses_them() {
    let a = 1.0f32 + 2f32.powi(-12);
    let c = -(1.0f32 + 2f32.powi(-11));
    let tiny = 2f32.powi(-24);
    assert_eq!(a.mul_add(a, c), tiny);
    assert_eq!(a * a + c, 0.0);
    let inputs = format!(
        "{}{}{}{}",
        of("a", a),
        of("c", c),
        of("e", -c),
        of("one", -1.0)
    );
    let p = "  %p = fmul contract float %a, %a\n";
    let half_inputs = format!(
        "{}{}  %ha = bitcast i16 %ta to half\n  %hc = bitcast i16 %tc to half\n",
        on("ta", 16, 0x3C10),
        on("tc", 16, 0xBC20)
    );
    // (name, body defining %r as a float, the stored bits, OpFmaKHR on moltenvk-m3, notes
    // FastMathRewrites)
    let shapes: [(&'static str, String, u32, usize, bool); 11] = [
        (
            "contract_pair_fuses",
            format!("{inputs}{p}  %r = fadd contract float %p, %c\n"),
            tiny.to_bits(),
            1,
            false,
        ),
        (
            "contract_product_used_twice_fuses_into_both",
            format!("{inputs}{p}  %s = fadd contract float %p, %c\n  %t = fadd contract float %p, %one\n  %r = fadd float %s, %t\n"),
            (2f32.powi(-11) + 2f32.powi(-23)).to_bits(),
            2,
            false,
        ),
        (
            "contract_subtrahend_product_negates_a_factor",
            format!("{inputs}{p}  %r = fsub contract float %e, %p\n"),
            (-tiny).to_bits(),
            1,
            false,
        ),
        (
            "contract_product_minus_addend_negates_the_addend",
            format!("{inputs}{p}  %r = fsub contract float %p, %e\n"),
            tiny.to_bits(),
            1,
            false,
        ),
        (
            "contract_two_products_fuse_the_first",
            format!("{inputs}{p}  %q = fmul contract float %e, %one\n  %r = fadd contract float %p, %q\n"),
            tiny.to_bits(),
            1,
            false,
        ),
        (
            "contract_stored_product_rounds_twice",
            format!("{inputs}{p}  store float %p, ptr addrspace(1) %0, align 4\n  %r = fadd contract float %p, %c\n"),
            0,
            0,
            false,
        ),
        (
            "contract_product_through_a_select_rounds_twice",
            format!("{inputs}{p}  %k = fcmp olt float %a, 2.0\n  %q = select i1 %k, float %p, float %c\n  %r = fadd contract float %q, %c\n"),
            0,
            0,
            false,
        ),
        (
            "contract_needs_both_operations",
            format!("{inputs}{p}  %r = fadd float %p, %c\n"),
            0,
            0,
            false,
        ),
        (
            "contract_half_pair_fuses",
            format!("{half_inputs}  %hp = fmul contract half %ha, %ha\n  %hs = fadd contract half %hp, %hc\n  %r = fpext half %hs to float\n"),
            2f32.powi(-12).to_bits(),
            0,
            false,
        ),
        (
            "contract_widened_half_product_fuses_in_float",
            format!("{half_inputs}{}  %hp = fmul contract half %ha, %ha\n  %w = fpext half %hp to float\n  %r = fadd contract float %w, %fc\n", of("fc", -(1.0 + 2f32.powi(-5)))),
            2f32.powi(-12).to_bits(),
            1,
            false,
        ),
        (
            "contract_through_a_negation_is_not_measured",
            format!("{inputs}{p}  %n = fneg float %p\n  %r = fadd contract float %n, %e\n"),
            0,
            0,
            true,
        ),
    ];
    let ks: Vec<Kernel> = shapes
        .iter()
        .map(|(name, body, want, _, _)| Kernel {
            name,
            body: body.clone(),
            float: true,
            want: Some(*want),
            needs: &[],
            note: None,
        })
        .collect();
    let Some(bitcode) = assemble("contract", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let pr = profile(stem, exact_maths()).expect("profile");
        for ((name, _, want, khr, noted), kk) in shapes.iter().zip(&ks) {
            let what = format!("{name} on {stem}");
            let art = compile(&bitcode, kk.name, &pr).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&art, &pr).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                art.precision.contains(&PrecisionNote::FastMathRewrites),
                *noted,
                "{what}: {:?}",
                art.precision
            );
            if stem == "moltenvk-m3" {
                assert_eq!(
                    count_op(&words, crate::spv::Op::FmaKHR),
                    *khr,
                    "{what}: OpFmaKHR"
                );
            }
            let got = interpret(&bitcode, kk.name, &pr);
            assert_eq!(got, *want, "{what}: {got:#x}");
        }
    }
}

/// A fused `c - a * b` negates a factor that is not a sum where the other is one. The A1
/// regularizer step `w - lr * (sign(w) * s + g)` of f9a12252 at lane 16 (`sign(w) = +0`, `s =
/// 1`, `g = -0`, `lr = 0.5`, `w = -0`) is `-0` exactly; `fma(-fma(+0, 1, -0), 0.5, -0)` is that
/// value, but MoltenVK's fast-math Metal compile moved the negation into the inner sum and
/// answered `+0`, where Metal answers `-0`. `fma(-0.5, fma(+0, 1, -0), -0)` is the same value
/// and leaves it no sum to move it into. Both factor orders of the product; every profile route.
#[test]
fn a_fused_subtraction_negates_a_factor_that_is_not_a_sum() {
    let inputs = format!(
        "{}{}{}{}{}  %lr = bitcast i32 %lrb to float\n  %m = fmul contract float %s, %x\n  %t = fadd contract float %m, %g\n",
        of("s", 0.0),
        of("x", 1.0),
        of("g", -0.0),
        oi("lrb", 0.5f32.to_bits()),
        of("w", -0.0)
    );
    let ks = [
        Kernel {
            name: "fused_subtraction_of_sum_times_factor",
            body: format!(
                "{inputs}  %p = fmul contract float %t, %lr\n  %r = fsub contract float %w, %p\n"
            ),
            float: true,
            want: Some(0x8000_0000),
            needs: &[],
            note: None,
        },
        Kernel {
            name: "fused_subtraction_of_factor_times_sum",
            body: format!(
                "{inputs}  %p = fmul contract float %lr, %t\n  %r = fsub contract float %w, %p\n"
            ),
            float: true,
            want: Some(0x8000_0000),
            needs: &[],
            note: None,
        },
    ];
    let Some(bitcode) = assemble("negate-factor", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            // On moltenvk-m3 both pairs are `OpFmaKHR` and the negation is the module's only
            // one; the routes that call routines negate the same operand.
            if stem == "moltenvk-m3" {
                let negated = negated_operands(&words);
                assert_eq!(negated.len(), 1, "{what}: one OpFNegate");
                let sums: Vec<u32> = [
                    crate::spv::Op::FmaKHR,
                    crate::spv::Op::FAdd,
                    crate::spv::Op::FSub,
                ]
                .into_iter()
                .flat_map(|op| results_and_no_contraction(&words, op).0)
                .collect();
                assert!(
                    !sums.contains(&negated[0]),
                    "{what}: the negated factor is a sum"
                );
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Metal regroups a tree of fast-math additions over `contract` products as the owned probe
/// `engine/unm_contract_chain` measured (Apple M4 and M3 Max, every thread), not pair by pair:
/// `(a * b + c * d) + e` with `e` loaded is `fma(c, d, fma(a, b, e))`, and with `e` computed
/// without reading memory `fma(a, b, fma(c, d, e))`, where the pair rule would give `fma(a, b, c *
/// d) + e`; `e - (a * b + c * d)` with `e` loaded is `fma(-c, d, fma(-a, b, e))`; a third product
/// is fused outermost and its addend moved innermost; and `(a * b + e1) + e2` is `fma(a, b, e1) +
/// e2`. The operands are chosen so that the groupings differ. A tree whose regrouping turns on an
/// addend of a kind the probe did not measure (a value computed without reading memory other than a
/// converted integer) notes `FastMathRewrites`; the others do not. Every link is one `OpFmaKHR` on
/// moltenvk-m3.
#[test]
fn contract_chains_regroup_as_metal_does() {
    let (a, b, c, d, e) = (
        -1.204_375_f32,
        -1.529_513_6_f32,
        1.322_877_2_f32,
        -1.410_43_f32,
        0.023_719_51_f32,
    );
    let loaded = c.mul_add(d, a.mul_add(b, e));
    let kept = a.mul_add(b, c.mul_add(d, e));
    let pairwise = a.mul_add(b, c * d) + e;
    assert!(loaded != kept && loaded != pairwise && kept != pairwise);
    let subtracted = (-c).mul_add(d, (-a).mul_add(b, e));
    assert!(subtracted != (-a).mul_add(b, (-c).mul_add(d, e)));
    assert!(subtracted != e - a.mul_add(b, c * d));
    let inputs = format!(
        "{}{}{}{}{}  %p = fmul fast float %a, %b\n  %q = fmul fast float %c, %d\n",
        of("a", a),
        of("b", b),
        of("c", c),
        of("d", d),
        of("ev", e)
    );
    // The addend loaded back from the output word.
    let load = "  store float %ev, ptr addrspace(1) %0, align 4\n  %e = load float, ptr addrspace(1) %0, align 4\n";
    // (name, body defining %r, the stored value, OpFmaKHR on moltenvk-m3, notes
    // FastMathRewrites)
    let shapes: [(&'static str, String, f32, usize, bool); 5] = [
        (
            "chain_loaded_addend_regroups",
            format!("{inputs}{load}  %s = fadd fast float %p, %q\n  %r = fadd fast float %s, %e\n"),
            loaded,
            2,
            false,
        ),
        (
            "chain_computed_addend_keeps_its_order",
            format!("{inputs}  %s = fadd fast float %p, %q\n  %r = fadd fast float %s, %ev\n"),
            kept,
            2,
            true,
        ),
        (
            "chain_subtracted_sum_regroups",
            format!("{inputs}{load}  %s = fadd fast float %p, %q\n  %r = fsub fast float %e, %s\n"),
            subtracted,
            2,
            false,
        ),
        (
            "chain_third_product_fuses_outermost",
            format!("{inputs}{load}  %t = fmul fast float %c, %a\n  %s = fadd fast float %p, %q\n  %u = fadd fast float %s, %t\n  %r = fadd fast float %u, %e\n"),
            c.mul_add(a, a.mul_add(b, c.mul_add(d, e))),
            3,
            false,
        ),
        (
            "chain_second_addend_stays_an_addition",
            format!("{inputs}  %s = fadd fast float %p, %ev\n  %r = fadd fast float %s, %c\n"),
            a.mul_add(b, e) + c,
            1,
            false,
        ),
    ];
    let ks: Vec<Kernel> = shapes
        .iter()
        .map(|(name, body, want, _, _)| Kernel {
            name,
            body: body.clone(),
            float: true,
            want: Some(want.to_bits()),
            needs: &[],
            note: None,
        })
        .collect();
    let Some(bitcode) = assemble("chains", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let pr = profile(stem, exact_maths()).expect("profile");
        for ((name, _, want, khr, noted), kk) in shapes.iter().zip(&ks) {
            let what = format!("{name} on {stem}");
            let art = compile(&bitcode, kk.name, &pr).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&art, &pr).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                art.precision.contains(&PrecisionNote::FastMathRewrites),
                *noted,
                "{what}: {:?}",
                art.precision
            );
            if stem == "moltenvk-m3" {
                assert_eq!(
                    count_op(&words, crate::spv::Op::FmaKHR),
                    *khr,
                    "{what}: OpFmaKHR"
                );
            }
            let got = interpret(&bitcode, kk.name, &pr);
            assert_eq!(got, want.to_bits(), "{what}: {got:#x}");
        }
    }
}

/// The operand of every `OpFNegate` in `words`.
fn negated_operands(words: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::FNegate as u32 {
            out.push(words[i + 3]);
        }
        i += len;
    }
    out
}

/// Metal divides `float` under `arcp` as `x` times its fast reciprocal of `y` (a probe cases
/// 9496749c and 3921d3e2, every flag set with `arcp`, every lane). The probe's lane 0, `0x42aa3a21
/// / 0x44c80075`, is `0x3d59e381` under `arcp` (`x` times `1 / y` rounded; without a flag it is
/// `0x3d59e382`, [`a_division_without_arcp_is_metals_precise_division`]). The reciprocal flushes: a
/// divisor past `2^126` gives a zero and a denormal divisor the infinity. On moltenvk-m3, whose
/// float functions and division are Metal's, the reciprocal is the device's `OpFDiv(1, y)`, Metal's
/// own under its fast-math compile, with no `OpFmaKHR`. A profile with `OpFmaKHR` and without that
/// fact takes `RN(1 / y)`, the device's estimate refined by four `OpFmaKHR`
/// (`fast_division_follows_the_metal_float_functions_fact`), so `6 / 3` is `2` whatever the
/// device's estimate (RC3 case f8494e2f read `0x3fffffff` from RADV's). The others take the
/// device's. Metal's own reciprocal is not on every lane of any of them in every math mode, and the
/// note stays.
#[test]
fn an_arcp_division_is_metals_fast_division() {
    let xy = format!(
        "{}{}",
        of("x", f32::from_bits(0x42AA_3A21)),
        of("y", f32::from_bits(0x44C8_0075))
    );
    let one = of("one", 1.0);
    let far = format!("{one}{}", of("far", 2f32.powi(127)));
    let tiny = format!(
        "{one}{}  %tiny = bitcast i32 %tb to float\n",
        oi("tb", 0x0040_0000)
    );
    let kernel = |name: &'static str, body: String, want: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(want),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "arcp_divide",
            format!("{xy}  %r = fdiv arcp float %x, %y\n"),
            0x3D59_E381,
        ),
        kernel(
            "fast_divide",
            format!("{xy}  %r = fdiv fast float %x, %y\n"),
            0x3D59_E381,
        ),
        kernel(
            "arcp_divide_past_2_126",
            format!("{far}  %r = fdiv arcp float %one, %far\n"),
            0,
        ),
        kernel(
            "arcp_divide_by_a_denormal",
            format!("{tiny}  %r = fdiv arcp float %one, %tiny\n"),
            0x7F80_0000,
        ),
        kernel(
            "arcp_six_by_three",
            format!(
                "{}{}  %r = fdiv arcp float %six, %three\n",
                of("six", 6.0),
                of("three", 3.0)
            ),
            0x4000_0000,
        ),
    ];
    let Some(bitcode) = assemble("fast-divide", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                count_op(&words, crate::spv::Op::FmaKHR),
                0,
                "{what}: OpFmaKHR"
            );
            assert!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                "{what}: {:?}",
                a.precision
            );
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Every committed profile's stem (`validation/profiles`), sorted.
fn every_profile() -> Vec<String> {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles");
    let mut out: Vec<String> = std::fs::read_dir(&dir)
        .unwrap_or_else(|e| panic!("{}: {e}", dir.display()))
        .map(|e| e.expect("profile directory entry").path())
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .map(|p| p.file_stem().expect("stem").to_string_lossy().into_owned())
        .collect();
    out.sort();
    out
}

/// Metal's fast division answers its NaNs and denormals on every committed profile, whatever the
/// device's NaN encoding and denormal mode: `0x7FC00000` for a NaN divisor of either sign (RADV
/// answered `0xffc00000`, a probe cases 9496749c and 3921d3e2), a NaN numerator, `0 / 0`, `inf /
/// inf`, and `inf · 0` (`inf / 2^127`, A1 case a919193f's `tanh` past `x = 44.4`); a denormal
/// numerator as a zero of its sign, so a denormal over a denormal is `0 · inf = NaN` (A1 case
/// f9a12252, where lavapipe kept the numerator and answered `+inf`); and a quotient whose 24-bit
/// rounding is below `2^-126` as a zero of its sign, `(2^24 - 1) · 2^-150` included, which the
/// denormal grid rounds up to `2^-126`. The operands are integer bits, so the division is the
/// kernel's only float operation, and it records the device's arithmetic (its reciprocal) and no
/// denormal or NaN note. The interpreter keeps denormals and NaN payloads, so each answer is the
/// lowering's.
#[test]
fn a_fast_division_answers_metal_at_nans_and_denormals_on_every_profile() {
    let bits = |name: &str, b: u32| {
        format!("  %{name}.i = call i32 @llvm.umin.i32(i32 {b}, i32 -1)\n  %{name} = bitcast i32 %{name}.i to float\n")
    };
    let kernel = |name: &'static str, x: u32, y: u32, want: u32| Kernel {
        name,
        body: format!(
            "{}{}  %r = fdiv arcp float %x, %y\n",
            bits("x", x),
            bits("y", y)
        ),
        float: true,
        want: Some(want),
        needs: &[],
        note: None,
    };
    let nan = 0x7FC0_0000;
    let ks = [
        kernel("fast_one_by_nan", 0x3F80_0000, 0x7FC0_0000, nan),
        kernel("fast_inf_by_nan", 0x7F80_0000, 0x7FC0_0000, nan),
        kernel("fast_one_by_negative_nan", 0x3F80_0000, 0xFFC0_0000, nan),
        kernel("fast_nan_by_two", 0x7FC0_0001, 0x4000_0000, nan),
        kernel("fast_zero_by_zero", 0, 0, nan),
        kernel("fast_inf_by_inf", 0x7F80_0000, 0x7F80_0000, nan),
        kernel("fast_inf_by_2_127", 0x7F80_0000, 0x7F00_0000, nan),
        kernel("fast_denormal_by_one", 0x0040_0000, 0x3F80_0000, 0),
        kernel(
            "fast_negative_denormal_by_two",
            0x8040_0000,
            0x4000_0000,
            0x8000_0000,
        ),
        kernel(
            "fast_denormal_by_minus_one",
            0x0000_0001,
            0xBF80_0000,
            0x8000_0000,
        ),
        kernel("fast_denormal_by_denormal", 0x0040_0000, 0x0040_0000, nan),
        kernel("fast_tiny_quotient", 0x0D80_0000, 0x4E80_0000, 0),
        kernel(
            "fast_negative_tiny_quotient",
            0x8D80_0000,
            0x4E80_0000,
            0x8000_0000,
        ),
        kernel(
            "fast_tiny_quotient_negative_divisor",
            0x0D80_0000,
            0xCE80_0000,
            0x8000_0000,
        ),
        kernel(
            "fast_quotient_below_the_grid_rounding",
            0x00FF_FFFF,
            0x4000_0000,
            0,
        ),
        kernel(
            "fast_negative_quotient_below_the_grid_rounding",
            0x80FF_FFFF,
            0x4000_0000,
            0x8000_0000,
        ),
        kernel(
            "fast_least_normal_quotient",
            0x0100_0000,
            0x4000_0000,
            0x0080_0000,
        ),
        kernel("fast_six_by_three", 0x40C0_0000, 0x4040_0000, 0x4000_0000),
    ];
    let Some(bitcode) = assemble("fast-divide-specials", &module(&ks)) else {
        return;
    };
    for stem in every_profile() {
        let p = profile(&stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                "{what}: {:?}",
                a.precision
            );
            for n in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert!(
                    !a.precision.contains(&n),
                    "{what}: {n:?} in {:?}",
                    a.precision
                );
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// An `fdiv` without `arcp` on `float` is Metal's precise division on every profile (a probe cases
/// 9496749c, 3921d3e2, and c10cba13): the correctly rounded quotient (lane 0x42aa3a21 / 0x44c80075
/// gives 0x3D59E382, where `arcp` gives 0x3D59E381), a denormal operand read as a zero, a denormal
/// quotient flushed, also `(2 - 2^-23) / 2^127`, whose 24-bit quotient `2^-126 - 2^-150` the
/// denormal grid would round up to the smallest normal (c10cba13: Metal answers 0), and
/// `0x7FC00000` for `0 / 0`. The interpreter's `OpFDiv` keeps denormals (it answers `2^-127` for `1
/// / 2^127`), so these lanes fail on the device's division. A `half` division is the routine's
/// under every flag set, `arcp` included, rounded once to `half` (4c363561, 54f0fe8e). The routine
/// is exact: no `DeviceArithmetic` note.
#[test]
fn a_division_without_arcp_is_metals_precise_division() {
    let xy = format!(
        "{}{}",
        of("x", f32::from_bits(0x42AA_3A21)),
        of("y", f32::from_bits(0x44C8_0075))
    );
    let one = of("one", 1.0);
    // 1 / 3 in `half`: 0x3555, whose `float` value is the stored word.
    let hxy = format!(
        "{}{}  %hx = bitcast i16 %hxb to half\n  %hy = bitcast i16 %hyb to half\n",
        on("hxb", 16, 0x3C00),
        on("hyb", 16, 0x4200)
    );
    let half_q = (f32::from_bits(0x3EAA_A000)).to_bits();
    let kernel = |name: &'static str, body: String, want: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(want),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "plain_divide",
            format!("{xy}  %r = fdiv float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "divide_without_arcp",
            format!("{xy}  %r = fdiv nnan ninf nsz contract float %x, %y\n"),
            0x3D59_E382,
        ),
        kernel(
            "divide_to_a_denormal",
            format!(
                "{one}{}  %r = fdiv float %one, %far\n",
                of("far", 2f32.powi(127))
            ),
            0,
        ),
        kernel(
            "divide_a_denormal",
            format!(
                "{one}{}  %tiny = bitcast i32 %tb to float\n  %r = fdiv float %tiny, %one\n",
                oi("tb", 0x0040_0000)
            ),
            0,
        ),
        kernel(
            "divide_to_just_below_the_smallest_normal",
            format!(
                "{}{}  %r = fdiv float %x, %y\n",
                of("x", f32::from_bits(0x3FFF_FFFF)),
                of("y", -(2f32.powi(127)))
            ),
            0x8000_0000,
        ),
        kernel(
            "divide_zero_by_zero",
            format!("{}  %r = fdiv float %z, %z\n", of("z", 0.0)),
            0x7FC0_0000,
        ),
        kernel(
            "divide_half",
            format!("{hxy}  %q = fdiv half %hx, %hy\n  %r = fpext half %q to float\n"),
            half_q,
        ),
        kernel(
            "divide_half_under_arcp",
            format!("{hxy}  %q = fdiv fast half %hx, %hy\n  %r = fpext half %q to float\n"),
            half_q,
        ),
    ];
    let Some(bitcode) = assemble("precise-divide", &module(&ks)) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceArithmetic),
                "{what}: {:?}",
                a.precision
            );
            assert!(
                a.strategies.entries.contains_key("arith.fdiv.precise"),
                "{what}: {:?}",
                a.strategies.entries
            );
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// `air.fast_sincos` on `float` answers as `air.fast_sin` and `air.fast_cos` do (a probe cases
/// 9496749c and 3921d3e2, every lane): `+0` for both past `float(π/2) · 2^22` (`0x4AC90FDB`), which
/// the module compares against, while `air.sincos` keeps the device's functions there. (The
/// interpreter does not model `Sin`; the shape is the module's.)
#[test]
fn fast_sincos_flushes_past_the_last_quadrant() {
    let body = |call: &str| {
        format!(
            "  %c = alloca float, align 4\n{}  %s = call float @{call}(float %a, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd float %s, %cv\n",
            of("a", 2.0)
        )
    };
    let kernel = |name: &'static str, call: &str| Kernel {
        name,
        body: body(call),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("fast_sincos_flushes", "air.fast_sincos.f32"),
        kernel("sincos_keeps_the_device", "air.sincos.f32"),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fast_sincos.f32(float, ptr)\n"),
    );
    let Some(bitcode) = assemble("fast-sincos", &text) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe", "vk12-floor"] {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let limits = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::Constant as u32
                    && w.len() == 4
                    && w[3] == 0x4AC9_0FDB
            });
            assert_eq!(
                limits > 0,
                kk.name == "fast_sincos_flushes",
                "{what}: the flush limit"
            );
        }
    }
}

/// `float` `+ - *`, a comparison and `OpFmaKHR` record neither `DeviceDenormals` nor
/// `DeviceNanEncoding`: their denormals are read as Metal reads them, the device's where the
/// profile states it reads them so and flushed elsewhere, and their NaN is Metal's, the device's
/// where the profile states it and selected elsewhere; so does a remainder, Metal's composition of
/// those; the functions that answer Metal at denormals and NaNs themselves (`sign`), `half`
/// arithmetic, and the fused multiply-add routines record neither; nor does `fast::atan2`, whose
/// device quotient reads its operands as Metal does, flushed off the deviation.
/// The operands come from bits (`llvm.umin` of constants, bitcast), not from `float` arithmetic.
#[test]
fn device_float_environment_notes_follow_the_operations() {
    // 0x3fc03c00: a float near 1.5 whose low half is half 1.0.
    let w = "  %w = call i32 @llvm.umin.i32(i32 1069562880, i32 2147483647)\n  %x = bitcast i32 %w to float\n";
    let x = f32::from_bits(0x3FC0_3C00);
    let none = &[][..];
    let ks = [
        k("env_fadd", format!("{w}  %s = fadd float %x, %x\n  %r = bitcast float %s to i32\n"), (x + x).to_bits()),
        // Metal's remainder: `x - x * trunc(x / x)`.
        k("env_frem", format!("{w}  %s = frem float %x, %x\n  %r = bitcast float %s to i32\n"), 0),
        k("env_sign", format!("{w}  %s = call float @air.sign.f32(float %x)\n  %r = bitcast float %s to i32\n"), 1.0f32.to_bits()),
        k("env_fcmp", format!("{w}  %c = fcmp olt float %x, 0.0\n  %r = zext i1 %c to i32\n"), 0),
        k("env_half", format!("{w}  %t = trunc i32 %w to i16\n  %h = bitcast i16 %t to half\n  %s = fadd half %h, %h\n  %u = bitcast half %s to i16\n  %r = zext i16 %u to i32\n"), 0x4000),
        k("env_fma", format!("{w}  %s = call float @air.fma.f32(float %x, float %x, float %x)\n  %r = bitcast float %s to i32\n"), x.mul_add(x, x).to_bits()),
        // `fast::atan2` selects Metal's NaN but returns the `atan` of the device's quotient:
        // the device's `Atan` only in a fast-math entry on a device whose functions are
        // Metal's, and the translation's routine here, an entry that keeps NaNs. The value is not
        // checked here.
        Kernel {
            want: None,
            ..k("env_fast_atan2", format!("{w}  %s = call float @air.fast_atan2.f32(float %x, float %x)\n  %r = bitcast float %s to i32\n"), 0)
        },
    ];
    // The notes each kernel records on moltenvk-m3 and on lavapipe. A `float` sum answers
    // Metal's NaN and denormals on both: the device's on moltenvk-m3, selected and flushed on
    // lavapipe.
    let notes: [(&[PrecisionNote], &[PrecisionNote]); 7] = [
        (none, none),
        (none, none),
        (none, none),
        (none, none),
        (none, none),
        (none, none),
        (none, none),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.sign.f32(float)\ndeclare float @air.fma.f32(float, float, float)\ndeclare float @air.fast_atan2.f32(float, float)\n"),
    );
    let Some(bitcode) = assemble("float-environment", &text) else {
        return;
    };
    for (stem, column) in [("moltenvk-m3", 0), ("lavapipe", 1)] {
        let p = profile(stem, exact_maths()).expect("profile");
        for (kk, (m3, lp)) in ks.iter().zip(&notes) {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let want = if column == 0 { *m3 } else { *lp };
            for n in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert_eq!(
                    a.precision.contains(&n),
                    want.contains(&n),
                    "{what}: {n:?} in {:?}",
                    a.precision
                );
            }
            if kk.want.is_some() {
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
        }
    }
}

/// An `frem` on `float` is Metal's. Without every fast-math flag it is `x - y * trunc(x / y)` of
/// Metal's precise quotient, each step rounded (`arith.frem`): `frem(1, 0.1)` is `+0`, as on the M3
/// Max, where the exact remainder is `0x3DCCCCCB`; with every flag it is Metal's `fast::fmod`
/// (`arith.fmod.fast`). Neither leaves the remainder to the device's `OpFRem`, and the one without
/// flags records no note.
#[test]
fn a_float_remainder_is_metals() {
    let one = "  %w = call i32 @llvm.umin.i32(i32 1065353216, i32 2147483647)\n  %x = bitcast i32 %w to float\n";
    let tenth = "  %v = call i32 @llvm.umin.i32(i32 1036831949, i32 2147483647)\n  %y = bitcast i32 %v to float\n";
    let ks = [
        k(
            "frem_precise",
            format!("{one}{tenth}  %s = frem float %x, %y\n  %r = bitcast float %s to i32\n"),
            0,
        ),
        Kernel {
            want: None,
            ..k(
                "frem_fast",
                format!(
                    "{one}{tenth}  %s = frem fast float %x, %y\n  %r = bitcast float %s to i32\n"
                ),
                0,
            )
        },
    ];
    let Some(bitcode) = assemble("frem", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(count_op(&words, crate::spv::Op::FRem), 0, "{what}: OpFRem");
            let census = if kk.name == "frem_precise" {
                "arith.frem"
            } else {
                "arith.fmod.fast"
            };
            assert!(
                a.strategies.entries.contains_key(census),
                "{what}: {:?}",
                a.strategies.entries
            );
            if kk.name == "frem_precise" {
                for n in [
                    PrecisionNote::DeviceArithmetic,
                    PrecisionNote::DeviceDenormals,
                    PrecisionNote::DeviceNanEncoding,
                ] {
                    assert!(!a.precision.contains(&n), "{what}: {n:?}");
                }
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(got, 0, "{what}: {got:#x}");
            }
        }
    }
}

/// Everything the arithmetic family leaves unlowered is a typed, located `NotImplemented` refusal
/// of the arith family, never an internal error or wrong output.
#[test]
fn unlowered_arithmetic_refuses_as_not_implemented() {
    let cases = [
        k(
            "pack_to_a_packed_float",
            format!(
                "{}  %v = insertelement <3 x float> zeroinitializer, float %a, i32 0\n  %r = call i32 @air.pack.unorm.rg11b10f.v3f32(<3 x float> %v)\n",
                of("a", 1.0)
            ),
            0,
        ),
    ];
    let text = module(&cases).replace(
        DECLS,
        &format!("{DECLS}declare i32 @air.pack.unorm.rg11b10f.v3f32(<3 x float>)\n"),
    );
    let Some(bitcode) = assemble("refusals", &text) else {
        return;
    };
    let p = profile("moltenvk-m3", exact_maths()).expect("profile");
    for kk in &cases {
        match compile(&bitcode, kk.name, &p) {
            Err(Error::Unsupported(u)) => {
                assert_eq!(
                    u.feature,
                    Feature::Lowering(LowerFamily::Arith),
                    "{}",
                    kk.name
                );
                assert_eq!(u.needs, Needs::NotImplemented, "{}", kk.name);
                assert!(u.location.is_some(), "{}: located", kk.name);
            }
            other => panic!(
                "{}: expected a NotImplemented refusal, got {other:?}",
                kk.name
            ),
        }
    }
}

/// The result ids of every `op` instruction in `words`, and the ids decorated `NoContraction`.
fn results_and_no_contraction(words: &[u32], op: crate::spv::Op) -> (Vec<u32>, Vec<u32>) {
    let (mut results, mut decorated) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            decorated.push(words[i + 1]);
        }
        if opcode == op as u32 {
            results.push(words[i + 2]);
        }
        i += len;
    }
    (results, decorated)
}

/// The results of every `GLSL.std.450 Fma` whose addend is a `-0.0` `float` constant.
fn glsl_fma_results(words: &[u32]) -> Vec<u32> {
    const GLSL_FMA: u32 = 50;
    let (mut results, mut negative_zeros) = (Vec::new(), Vec::new());
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::Constant as u32 && len == 4 && words[i + 3] == 0x8000_0000 {
            negative_zeros.push(words[i + 2]);
        }
        if opcode == crate::spv::Op::ExtInst as u32
            && words[i + 4] == GLSL_FMA
            && negative_zeros.contains(&words[i + 7])
        {
            results.push(words[i + 2]);
        }
        i += len;
    }
    results
}

/// An undecorated SPIR-V float operation lets the device fuse `a * b + c` (which rounds
/// differently) and regroup it with its neighbours, as MoltenVK's fast-math Metal compile does.
/// LLVM grants the two separately (`contract`, `reassoc`), so only an operation carrying both is
/// left undecorated; every other addition is decorated `NoContraction`, and every other multiply is
/// the product rounded on its own, decorated `NoContraction`: an `OpFMul` where the profile's
/// `float` functions are the device's, and `GLSL.std.450 Fma(a, b, -0.0)` where they are Metal's,
/// whose SPIR-V reaches Metal through SPIRV-Cross. Under `FloatMathsPolicy::Exact` the product's
/// check of a profile whose check is fused (`TinyCheck::Fused`) scales an operand by its own
/// multiply, `NoContraction` too, counted apart. The decoration matters where the addition may be
/// fused: Mesa splits a contractable `Fma` into a multiply and an add and folds the `-0.0` away,
/// and the multiply may then be fused with the `reassoc contract` add of `mul_strict_add_fast`
/// (NIR's `fuse` option on older Mesa, or a backend such as ACO) unless the multiply is
/// `NoContraction`. The interpreter rounds each operation on its own: here `(1 + 2^-12)^2 - (1 +
/// 2^-11)` is 0 unfused and 2^-24 fused. A pair that both carry `contract` is the one Metal fuses:
/// one fused multiply-add, the multiply no operation of its own, 2^-24 on every profile.
#[test]
fn float_ops_without_contract_and_reassoc_are_no_contraction() {
    use crate::lower::arith::denormal::{tiny_check, TinyCheck};
    let inputs = format!(
        "{}{}",
        of("a", 1.0 + 2f32.powi(-12)),
        of("c", -(1.0 + 2f32.powi(-11)))
    );
    let kernel = |name: &'static str, mul: &str, add: &str| Kernel {
        name,
        body: format!("{inputs}  %p = fmul {mul}float %a, %a\n  %r = fadd {add}float %p, %c\n"),
        float: true,
        want: Some(0),
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("mul_add_strict", "", ""),
        kernel("mul_add_contract", "contract ", "contract "),
        kernel("mul_add_reassoc", "reassoc ", "reassoc "),
        kernel(
            "mul_add_reassoc_contract",
            "reassoc contract ",
            "reassoc contract ",
        ),
        kernel("mul_add_fast", "fast ", "fast "),
        kernel("mul_strict_add_fast", "", "reassoc contract "),
    ];
    let Some(bitcode) = assemble("contract", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            if matches!(
                kk.name,
                "mul_add_contract" | "mul_add_reassoc_contract" | "mul_add_fast"
            ) {
                assert_eq!(
                    interpret(&bitcode, kk.name, &p),
                    2f32.powi(-24).to_bits(),
                    "{what}: fused"
                );
                continue;
            }
            let (muls, decorated) = results_and_no_contraction(&words, crate::spv::Op::FMul);
            let (adds, _) = results_and_no_contraction(&words, crate::spv::Op::FAdd);
            let fmas = glsl_fma_results(&words);
            // The product's check, where it is fused, scales an operand by its own multiply.
            let scales = tiny_check_scales(&words);
            let fused = tiny_check(&p) == Some(TinyCheck::Fused);
            assert_eq!(
                scales.len(),
                usize::from(fused),
                "{what}: the check's scaling"
            );
            assert!(
                scales.iter().all(|s| decorated.contains(s)),
                "{what}: the check's scaling NoContraction: {scales:?}"
            );
            let muls: Vec<u32> = muls.into_iter().filter(|m| !scales.contains(m)).collect();
            let add_rewritable = kk.name == "mul_strict_add_fast";
            // No unfused multiply here carries both flags: each is the profile's product with
            // `NoContraction`.
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            let mul = match (muls.as_slice(), fmas.as_slice(), metal) {
                ([], [m], true) | ([m], [], false) => *m,
                _ => panic!("{what}: one multiply in the profile's form: {muls:?} {fmas:?}"),
            };
            assert!(decorated.contains(&mul), "{what}: %{mul} NoContraction");
            // The input fadds carry no flags; the last fadd is the kernel's.
            let add = *adds.last().expect("an fadd");
            assert_eq!(
                decorated.contains(&add),
                !add_rewritable,
                "{what}: %{add} NoContraction"
            );
            assert_eq!(interpret(&bitcode, kk.name, &p), 0, "{what}: unfused");
        }
    }
}

/// Float arithmetic the AIR lets a compiler rewrite with its neighbour notes `FastMathRewrites`:
/// two chained operations that both carry `reassoc` (through a negation), and a `contract` pair
/// whose fusion the probe did not measure, a negation between the multiply and the addition.
/// Arithmetic whose neighbours carry no matching flag does not: a lone `fast` operation on
/// unflagged inputs, a `contract` multiply feeding a `reassoc` addition, and strict arithmetic; nor
/// does a `float` `llvm.fmuladd`, which is the fused multiply-add Metal computes, nor a `contract`
/// multiply fused into its one `contract` addition or subtraction, which is the pair Metal fuses.
#[test]
fn arithmetic_the_air_lets_a_compiler_rewrite_notes_fast_math_rewrites() {
    let inputs = format!("{}{}{}", of("a", 1.5), of("b", 2.5), of("c", -3.0));
    let kernel = |name: &'static str, body: &str| Kernel {
        name,
        body: format!("{inputs}{body}"),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let noted = [
        kernel(
            "contract_mul_neg_add",
            "  %p = fmul contract float %a, %b\n  %n = fneg float %p\n  %r = fadd contract float %n, %c\n",
        ),
        kernel(
            "reassoc_sum",
            "  %s = fadd reassoc float %a, %b\n  %r = fadd reassoc float %s, %c\n",
        ),
        kernel(
            "reassoc_product_through_neg",
            "  %p = fmul reassoc float %a, %b\n  %n = fneg float %p\n  %r = fmul reassoc float %n, %c\n",
        ),
    ];
    let quiet = [
        kernel(
            "fast_mul_add",
            "  %p = fmul fast float %a, %b\n  %r = fadd fast float %p, %c\n",
        ),
        kernel(
            "contract_mul_sub",
            "  %p = fmul contract float %a, %b\n  %r = fsub contract float %c, %p\n",
        ),
        kernel(
            "fmuladd",
            "  %r = call float @llvm.fmuladd.f32(float %a, float %b, float %c)\n",
        ),
        kernel("lone_fast", "  %r = fmul fast float %a, %b\n"),
        kernel(
            "contract_mul_reassoc_add",
            "  %p = fmul contract float %a, %b\n  %r = fadd reassoc float %p, %c\n",
        ),
        kernel(
            "strict_mul_add",
            "  %p = fmul float %a, %b\n  %r = fadd float %p, %c\n",
        ),
    ];
    let noting = noted.len();
    let ks: Vec<Kernel> = noted.into_iter().chain(quiet).collect();
    let Some(bitcode) = assemble("rewrites", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (i, kk) in ks.iter().enumerate() {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::FastMathRewrites),
                i < noting,
                "{what}: {:?}",
                a.precision
            );
        }
    }
}

/// Growth report: the same chain of 16 float operations on `float`, `half`, and `bfloat`. Each
/// widened `half` operation adds its rounding; each `bfloat` operation adds its integer rounding.
/// The narrow chains grow over the `float` chain without the check of a product at `2^-126`
/// (native float maths, `FloatMathsPolicy::Native`); under exact float maths each of its eight
/// products adds at most that check (`denormal::tiny_product_words`). The measured words are
/// printed (`--nocapture`) and bounded so a growth regression fails.
#[test]
fn narrow_float_module_growth_is_bounded() {
    use crate::device::FloatMathsPolicy;
    use crate::lower::arith::denormal::{tiny_product_words, TinyCheck};
    use crate::lower::arith::legal::{FloatFmt, Form, Held};
    let chain = |ty: &str| -> String {
        let (enter, leave) = match ty {
            "float" => (
                "  %x0 = fadd float %a, -0.0\n".to_owned(),
                "  %r = fadd float %x16, -0.0\n".to_owned(),
            ),
            t => (
                format!("  %x0 = fptrunc float %a to {t}\n"),
                format!("  %r = fpext {t} %x16 to float\n"),
            ),
        };
        let mut s = format!("{}{enter}", of("a", 1.5));
        for i in 0..16 {
            let op = if i % 2 == 0 { "fadd" } else { "fmul" };
            s.push_str(&format!("  %x{} = {op} {ty} %x{i}, %x0\n", i + 1));
        }
        s.push_str(&leave);
        s
    };
    let kernel = |name: &'static str, ty: &str| Kernel {
        name,
        body: chain(ty),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("chain_float", "float"),
        kernel("chain_half", "half"),
        kernel("chain_bfloat", "bfloat"),
    ];
    let Some(bitcode) = assemble("growth", &module(&ks)) else {
        return;
    };
    let plain = DevicePolicy {
        float_maths: FloatMathsPolicy::Native,
        ..exact_maths()
    };
    let float = Form {
        held: Held::Float(FloatFmt::F32),
        lanes: 1,
    };
    let checks = 8 * tiny_product_words(float, TinyCheck::Bits) as usize;
    for stem in PROFILES {
        let words_on = |policy: DevicePolicy, kk: &Kernel| {
            let p = profile(stem, policy).expect("profile");
            let a = compile(&bitcode, kk.name, &p)
                .unwrap_or_else(|e| panic!("{} on {stem}: {e:?}", kk.name));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{} on {stem}: L3: {e}", kk.name));
            a.module.words.len()
        };
        let p = profile(stem, exact_maths()).expect("profile");
        let checked = words_on(exact_maths(), &ks[0]);
        let words = [
            words_on(plain, &ks[0]),
            words_on(exact_maths(), &ks[1]),
            words_on(exact_maths(), &ks[2]),
        ];
        eprintln!(
            "half and bfloat growth on {stem}: float {} words ({checked} with its products' \
             checks), half {} ({:+}), bfloat {} ({:+})",
            words[0],
            words[1],
            words[1] as i64 - words[0] as i64,
            words[2],
            words[2] as i64 - words[0] as i64
        );
        assert!(
            words[0] <= checked && checked <= words[0] + checks,
            "{stem}: float {} words, {checked} with its products' checks",
            words[0]
        );
        // Half: through the device's measured quantize, per multiply and per narrowing
        // conversion a six-instruction rounding with the denormal arithmetic (at most 32 words),
        // and per sum a three-instruction one without it (at most 16 words), plus the conversions'
        // share and the shared constants. On the bits, per multiply and conversion seventeen
        // integer instructions and the three of the denormal arithmetic (at most 104 words), per
        // sum the seventeen (at most 88), and ten shared constants. Through the conversion, fewer
        // than either.
        let bound = if p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfQuantize) {
            9 * 32 + 8 * 16 + 64
        } else {
            9 * 104 + 8 * 88 + 128
        };
        assert!(words[1] <= words[0] + bound, "{stem}: half {words:?}");
        // Bfloat: a bounded integer rounding (at most 18 instructions) per operation.
        assert!(
            words[2] <= words[0] + 16 * 18 * 6 + 256,
            "{stem}: bfloat {words:?}"
        );
    }
}

/// A `half` result rounds through `OpQuantizeToF16` on a profile whose quantize is measured
/// Metal's, through the conversion on one whose conversion to 16 bits is, and on the bits in
/// integer arithmetic elsewhere; the operation decides the rest, never a name: only a product and a
/// narrowing conversion carry the denormal arithmetic (one `RoundEven` each); a sum, exact below
/// the smallest normal half, carries none. A profile that takes the measured
/// `DriverDeviation::MetalHalfConversion` (moltenvk-m3) rounds through a 16-bit conversion, each
/// rounding that conversion of `x + -0.0` decorated `NoContraction` (the barrier Metal's compiler
/// cannot fold the conversion through) and its widening, requiring the fact and declaring no 16-bit
/// float mode or `RoundingModeRTE`: the round trip first taken is not exact on lavapipe, whose
/// conversion to 16 bits truncates and, under `RoundingModeRTE`, wraps an overflow into the sign. A
/// profile whose float controls make that conversion Metal's (amd-rdna3-radv) rounds through the
/// conversion of `x` itself and its widening, each decorated `NoContraction`, in an entry declaring
/// `RoundingModeRTE`, `DenormPreserve` and `SignedZeroInfNanPreserve` for 16 bits and the modes
/// its independence ties to them (`narrow::controlled_modes`), requiring no fact. No other module
/// declares a 16-bit float mode or `RoundingModeRTE` for it. A module that quantizes requires
/// `DriverDeviation::MetalHalfQuantize`'s fact, since Vulkan leaves the quantize's rounding mode to
/// the device; no module notes the device's rounding.
///
/// The chain: one `fptrunc`, eight `fadd`, and eight `fmul` on `half`, strict and `fast`. The
/// `fast` chain's seven multiplies whose one use is the next addition are fused into it: each pair
/// is a call of the module's one `half` fused multiply-add routine, whose single rounding carries
/// the denormal arithmetic as a product's does, so the chain keeps the first addition's, the last
/// multiply's, the conversion's, and the routine's roundings. On a profile whose driver computes a
/// 16-bit `Fma` as Metal's `half` `fma` (moltenvk-m3) each pair is that `Fma` instead, inline, on
/// its operands converted to 16 bits and its result widened, all exactly (four `OpFConvert`, the
/// module's one 16-bit float type), so the chain keeps the other three roundings. On a profile
/// whose driver's 16-bit `+ - *`, kept apart, is Metal's (moltenvk-m3), each `fadd` and `fmul` that
/// rounds is that instruction, between two narrowings and a widening (three `OpFConvert`), and
/// rounds nothing itself: the conversion's rounding alone is left to the module.
#[test]
fn half_rounding_follows_the_operation() {
    let chain = |fast: bool| -> String {
        let flag = if fast { "fast " } else { "" };
        let mut s = format!(
            "{}  %x0 = fptrunc float %a to half\n",
            of("a", 1.5).replace("fadd float", &format!("fadd {flag}float"))
        );
        for i in 0..16 {
            let op = if i % 2 == 0 { "fadd" } else { "fmul" };
            s.push_str(&format!("  %x{} = {op} {flag}half %x{i}, %x0\n", i + 1));
        }
        s.push_str("  %r = fpext half %x16 to float\n");
        s
    };
    let kernel = |name: &'static str, fast: bool| Kernel {
        name,
        body: chain(fast),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [kernel("strict_chain", false), kernel("fast_chain", true)];
    let Some(bitcode) = assemble("half-rounding", &module(&ks)) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let quantize = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::QuantizeToF16 as u32
            });
            let fconvert = count_ops(&words, |w| w[0] & 0xFFFF == crate::spv::Op::FConvert as u32);
            // `GLSL.std.450 RoundEven` is instruction 2 of the set.
            let round_even = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32 && w.len() > 4 && w[4] == 2
            });
            // The fused pairs on 16-bit floats: the device's `Fma` measured Metal's, or `OpFmaKHR`
            // under the 16-bit float controls; each converts three operands and widens its result.
            let f16 = kk.name == "fast_chain"
                && matches!(
                    crate::lower::arith::fma::half_route(&p),
                    crate::lower::arith::fma::Route::F16 | crate::lower::arith::fma::Route::F16Khr
                );
            let (q, c, r) = match (kk.name == "fast_chain", f16) {
                (true, true) => (3, 7 * 4, 2),
                (true, false) => (4, 0, 3),
                (false, _) => (17, 0, 9),
            };
            // Where the device's conversion to 16 bits is measured Metal's (moltenvk-m3) each
            // rounding is that conversion and its widening, with no quantize and no denormal
            // arithmetic.
            let converted = p
                .takes_driver_deviation(crate::error::DriverDeviation::MetalHalfConversion)
                && p.fields().leaf.float16;
            // Where the device's float controls make the conversion Metal's, each rounding is
            // that conversion and its widening too, with no barrier.
            let controlled = crate::lower::arith::narrow::quantize_form(&p)
                == crate::spirv::builder::QuantizeForm::Controlled;
            // Where the device's quantize is measured Metal's each rounding is that quantize;
            // elsewhere it is integer arithmetic on the bits, with no quantize.
            let device = p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfQuantize)
                && !converted
                && !controlled;
            // Where the device's 16-bit `+ - *` is Metal's, measured or under the 16-bit float
            // controls, the operations among the roundings: sixteen of the strict chain's
            // seventeen, the first addition and the last multiply of the fast chain's three. The
            // conversion's rounding stays the module's.
            let ha = (p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfArithmetic)
                && p.fields().leaf.float16)
                || controlled;
            assert!(
                !ha || converted || controlled,
                "{what}: the half arithmetic beside the conversion"
            );
            let kept = match (ha, kk.name == "fast_chain") {
                (false, _) => 0,
                (true, true) => 2,
                (true, false) => 16,
            };
            let want = match (converted || controlled, device) {
                (true, _) => (0, c + 2 * (q - kept) + 3 * kept, 0),
                (false, true) => (q, c, r),
                (false, false) => (0, c, r),
            };
            assert_eq!(
                (quantize, fconvert, round_even),
                want,
                "{what}: OpQuantizeToF16, OpFConvert, RoundEven"
            );
            // Each rounding through the measured conversion converts `x + -0.0` decorated
            // `NoContraction`, the barrier Metal's compiler cannot fold the conversion through:
            // without it the folded conversion rounds toward zero. The controlled conversion adds
            // none: only the strict chain's own `%a = fadd float 1.5, -0.0`, decorated
            // `NoContraction` without fast math, reads as one where the `fptrunc` narrows it.
            let own_barrier = usize::from(kk.name == "strict_chain");
            assert_eq!(
                barred_roundings(&words),
                (
                    if converted || controlled { q - kept } else { 0 },
                    if controlled {
                        q - kept - own_barrier
                    } else {
                        0
                    }
                ),
                "{what}: conversion roundings, and those without the barrier"
            );
            let half_type = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::TypeFloat as u32 && w.len() > 2 && w[2] == 16
            });
            assert_eq!(
                half_type,
                usize::from(f16 || converted || controlled),
                "{what}: a 16-bit float type"
            );
            // The controlled form's modes: `RoundingModeRTE` at 16 bits and the widths tied to it.
            let rte: Vec<u32> = crate::lower::arith::narrow::controlled_modes(&p)
                .filter(|_| controlled)
                .unwrap_or_default()
                .iter()
                .filter_map(|m| match *m {
                    crate::spirv::ir::ExecMode::RoundingModeRte { width } => Some(width.into()),
                    _ => None,
                })
                .collect();
            assert_eq!(
                (
                    declares_signed_zero_inf_nan(&words, 16),
                    rounds_even_at(&words)
                ),
                (controlled, rte),
                "{what}: a 16-bit float mode and RoundingModeRTE"
            );
            // A module that quantizes requires the quantize's measured fact; one that rounds
            // through the conversion requires the conversion's; one that rounds on the bits
            // requires neither.
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalHalfQuantize),
                device,
                "{what}"
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalHalfConversion),
                converted,
                "{what}"
            );
        }
    }
}

/// Only a module that rounds to `half` through the device's conversion to 16 bits and back requires
/// `VkProperty::MetalHalfConversion`, wherever the profile takes the measured
/// `DriverDeviation::MetalHalfConversion` (moltenvk-m3). A `half` sum rounds so there. A `half`
/// fused multiply-add on the device's 16-bit `Fma` converts its operands, each a `half` already,
/// exactly and widens the `Fma`'s result, which rounds nothing: it requires `MetalHalfFma` alone.
/// Where the `half` routine computes the fused sum instead, its rounding takes the conversion like
/// any other. Where the device's 16-bit `+ - *` is Metal's, the sum is that instruction, which
/// rounds nothing of the module's: it requires `MetalHalfArithmetic` alone.
#[test]
fn only_a_rounding_through_the_conversion_requires_the_conversion_fact() {
    let halves = format!(
        "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n",
        on("t", 16, 0x3C01),
        on("s", 16, 0x4100),
        on("v", 16, 0x3800)
    );
    let ks = [
        Kernel {
            name: "half_sum",
            body: format!("{halves}  %x = fadd half %h, %g\n  %r = fpext half %x to float\n"),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
        Kernel {
            name: "half_fma",
            body: format!(
                "{halves}  %x = call half @llvm.fma.f16(half %h, half %g, half %u)\n  %r = fpext half %x to float\n"
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare half @llvm.fma.f16(half, half, half)\n"),
    );
    let Some(bitcode) = assemble("half-conversion-fact", &text) else {
        return;
    };
    let mut converting = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        let converted = p
            .takes_driver_deviation(crate::error::DriverDeviation::MetalHalfConversion)
            && p.fields().leaf.float16;
        let f16 = p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfFma)
            && p.fields().leaf.float16;
        let ha = p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfArithmetic)
            && p.fields().leaf.float16;
        converting += usize::from(converted);
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let rounds = match kk.name {
                "half_fma" => !f16,
                _ => !ha,
            };
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalHalfArithmetic),
                ha && kk.name == "half_sum",
                "{what}: {:?}",
                a.requirements.properties
            );
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalHalfConversion),
                converted && rounds,
                "{what}: {:?}",
                a.requirements.properties
            );
        }
    }
    assert!(converting > 0, "no profile rounds through the conversion");
}

/// Only a `half` result that is a value of the format wherever it lies below the smallest normal
/// half skips the denormal arithmetic. A square root and `fract`'s difference round through
/// `OpQuantizeToF16` alone (on bits alone where the device's quantize is not measured Metal's), and
/// an integer converted to `half` rounds by integer arithmetic with neither; a transcendental,
/// `ldexp`, a quotient, and the one rounding of a fused sum (`fmuladd`, and `mix`'s two fused
/// multiply-adds, which call the module's one `half` routine) keep one `RoundEven` each, since
/// their `f32` result below `2^-14` is off the `2^-24` grid (half `exp(-10)` is about `4.54e-5`).
/// The counts pin which steps may take `round_exact_below_normal`. On a profile whose driver
/// computes a 16-bit `Fma` as Metal's `half` `fma` (moltenvk-m3), a fused sum rounds nothing
/// itself: the device's `Fma` rounds it. On a profile whose conversion to 16 bits is measured
/// Metal's (moltenvk-m3) no rounding holds either: each is the conversion and its widening.
#[test]
fn only_an_exact_half_result_skips_the_denormal_rounding() {
    let halves = format!(
        "{}{}{}  %h = bitcast i16 %t to half\n  %g = bitcast i16 %s to half\n  %u = bitcast i16 %v to half\n",
        on("t", 16, 0x3C01),
        on("s", 16, 0x4100),
        on("v", 16, 0x3800)
    );
    // (name, the operation defining `%x`, `OpQuantizeToF16` count, `RoundEven` count)
    let cases: [(&'static str, &str, usize, usize); 9] = [
        ("half_exp", "call half @air.exp.f16(half %h)", 1, 1),
        ("half_log", "call half @air.log.f16(half %h)", 1, 1),
        (
            "half_ldexp",
            "call half @air.ldexp.f16(half %h, i32 -20)",
            1,
            1,
        ),
        ("half_fdiv", "fdiv half %h, %g", 1, 1),
        ("half_sqrt", "call half @air.sqrt.f16(half %h)", 1, 0),
        ("half_sitofp", "sitofp i32 %i to half", 0, 0),
        ("half_fract", "call half @air.fract.f16(half %h)", 1, 0),
        (
            "half_mix",
            "call half @air.mix.f16(half %h, half %g, half %u)",
            1,
            1,
        ),
        (
            "half_fmuladd",
            "call half @llvm.fmuladd.f16(half %h, half %g, half %u)",
            1,
            1,
        ),
    ];
    let ks: Vec<Kernel> = cases
        .iter()
        .map(|&(name, op, _, _)| Kernel {
            name,
            body: format!(
                "{halves}{}  %x = {op}\n  %r = fpext half %x to float\n",
                oi("i", 0x4040_0000)
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        })
        .collect();
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @air.exp.f16(half)\ndeclare half @air.log.f16(half)\ndeclare half @air.ldexp.f16(half, i32)\ndeclare half @air.sqrt.f16(half)\ndeclare half @air.fract.f16(half)\ndeclare half @air.mix.f16(half, half, half)\ndeclare half @llvm.fmuladd.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("half-exact-below-normal", &text) else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (kk, &(_, _, quantizes, rounds_even)) in ks.iter().zip(&cases) {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let quantize = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::QuantizeToF16 as u32
            });
            // `GLSL.std.450 RoundEven` is instruction 2 of the set.
            let round_even = count_ops(&words, |w| {
                w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32 && w.len() > 4 && w[4] == 2
            });
            let fused = matches!(kk.name, "half_mix" | "half_fmuladd")
                && p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfFma);
            // A rounding through the conversion holds neither.
            let converted = p
                .takes_driver_deviation(crate::error::DriverDeviation::MetalHalfConversion)
                && p.fields().leaf.float16;
            // Nor does one through the conversion under the float controls.
            let controlled = crate::lower::arith::narrow::quantize_form(&p)
                == crate::spirv::builder::QuantizeForm::Controlled;
            // A rounding on the bits holds no quantize.
            let device = p.takes_driver_deviation(crate::error::DriverDeviation::MetalHalfQuantize);
            let want = match (fused || converted || controlled, device) {
                (true, _) => (0, 0),
                (false, true) => (quantizes, rounds_even),
                (false, false) => (0, rounds_even),
            };
            assert_eq!(
                (quantize, round_even),
                want,
                "{what}: OpQuantizeToF16, RoundEven"
            );
        }
    }
}

/// The widths at which `words` (a module) declares `OpExecutionMode <entry> RoundingModeRTE`, in
/// declaration order.
fn rounds_even_at(words: &[u32]) -> Vec<u32> {
    let mut widths = Vec::new();
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::RoundingModeRTE as u32
        {
            widths.push(words[i + 3]);
        }
        i += len;
    }
    widths
}

/// The instructions of `words` (a module) for which `pick` holds of their words.
fn count_ops(words: &[u32], pick: impl Fn(&[u32]) -> bool) -> usize {
    let mut i = 5;
    let mut n = 0;
    while i < words.len() {
        let len = (words[i] >> 16) as usize;
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        n += usize::from(pick(&words[i..i + len]));
        i += len;
    }
    n
}

/// The `half` roundings in integer arithmetic in `words` (`narrow`'s `QuantizeForm::Integer`):
/// the `OpBitwiseAnd`s that keep a value's top 19 bits, `0xFFFFE000` (a constant, or a composite
/// of it), one per rounding.
fn integer_roundings(words: &[u32]) -> usize {
    let mut keep = std::collections::BTreeSet::new();
    let mut anded = 0;
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let w = &words[i..i + len];
        match opcode {
            o if o == crate::spv::Op::Constant as u32 && len == 4 && w[3] == 0xFFFF_E000 => {
                keep.insert(w[2]);
            }
            o if o == crate::spv::Op::ConstantComposite as u32
                && len > 3
                && w[3..].iter().all(|c| keep.contains(c)) =>
            {
                keep.insert(w[2]);
            }
            o if o == crate::spv::Op::BitwiseAnd as u32
                && w[3..].iter().any(|v| keep.contains(v)) =>
            {
                anded += 1;
            }
            _ => {}
        }
        i += len;
    }
    anded
}

/// The `half` roundings through the conversion in `words` (an `OpFConvert` of an `f32` to 16 bits
/// whose result another `OpFConvert` widens back, `narrow::converted`), and how many of them
/// convert anything but the barrier: an `OpFAdd` decorated `NoContraction` whose second operand
/// is `-0.0` (a constant, or a composite of them).
fn barred_roundings(words: &[u32]) -> (usize, usize) {
    use std::collections::{BTreeMap, BTreeSet};
    let mut widths = BTreeMap::new(); // float or vector type -> its lanes' width
    let mut minus_zero = BTreeSet::new(); // constants that are -0.0 in every lane
    let mut no_contraction = BTreeSet::new();
    let mut fadd = BTreeMap::new(); // result -> second operand
    let mut fconvert = Vec::new(); // (result type, result, operand)
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        let w = &words[i..i + len];
        match opcode {
            o if o == crate::spv::Op::TypeFloat as u32 => {
                widths.insert(w[1], w[2]);
            }
            o if o == crate::spv::Op::TypeVector as u32 => {
                if let Some(&width) = widths.get(&w[2]) {
                    widths.insert(w[1], width);
                }
            }
            o if o == crate::spv::Op::Constant as u32 && len == 4 && w[3] == 0x8000_0000 => {
                minus_zero.insert(w[2]);
            }
            o if o == crate::spv::Op::ConstantComposite as u32
                && w[3..].iter().all(|c| minus_zero.contains(c)) =>
            {
                minus_zero.insert(w[2]);
            }
            o if o == crate::spv::Op::Decorate as u32
                && w[2] == crate::spv::Decoration::NoContraction as u32 =>
            {
                no_contraction.insert(w[1]);
            }
            o if o == crate::spv::Op::FAdd as u32 => {
                fadd.insert(w[2], w[4]);
            }
            o if o == crate::spv::Op::FConvert as u32 => fconvert.push((w[1], w[2], w[3])),
            _ => {}
        }
        i += len;
    }
    let narrowed: BTreeMap<u32, u32> = fconvert
        .iter()
        .filter(|(ty, _, _)| widths.get(ty) == Some(&16))
        .map(|&(_, r, operand)| (r, operand))
        .collect();
    let roundings: Vec<u32> = fconvert
        .iter()
        .filter(|(ty, _, operand)| widths.get(ty) == Some(&32) && narrowed.contains_key(operand))
        .filter_map(|(_, _, n)| narrowed.get(n).copied())
        .collect();
    let unbarred = roundings
        .iter()
        .filter(|x| {
            !(no_contraction.contains(*x) && fadd.get(*x).is_some_and(|b| minus_zero.contains(b)))
        })
        .count();
    (roundings.len(), unbarred)
}

/// Whether `words` declares `OpExecutionMode <entry> SignedZeroInfNanPreserve <width>`.
pub(super) fn declares_signed_zero_inf_nan(words: &[u32], width: u32) -> bool {
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::ExecutionMode as u32
            && len == 4
            && words[i + 2] == crate::spv::ExecutionMode::SignedZeroInfNanPreserve as u32
            && words[i + 3] == width
        {
            return true;
        }
        i += len;
    }
    false
}

/// A float operation without `nnan`/`ninf`/`nsz` makes the entry declare `SignedZeroInfNanPreserve
/// 32` where the profile supports it, and note `DeviceSignedZeroInfNan` where it does not; `fast`
/// operations need neither, but for a `fast` product's check at `±2^-126` where the profile checks
/// one, which answers Metal's zero of its sign. Half arithmetic is held in `f32`, so it needs the
/// 32-bit mode.
#[test]
fn float_ops_without_fast_flags_preserve_signed_zero_inf_nan() {
    let inputs = format!("{}{}", of("a", -0.0), of("b", 0.0));
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body: format!("{inputs}{body}"),
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel("strict_mul", "  %r = fmul float %a, %b\n".to_owned()),
        Kernel {
            // The opaque inputs are `fast` too, so nothing in the entry needs preservation.
            body: format!("{inputs}  %r = fmul fast float %a, %b\n")
                .replace("fadd float", "fadd fast float"),
            ..kernel("fast_mul", String::new())
        },
        kernel(
            "strict_half",
            "  %h = fptrunc float %a to half\n  %s = fmul half %h, %h\n  %r = fpext half %s to float\n"
                .to_owned(),
        ),
    ];
    let Some(bitcode) = assemble("fpenv", &module(&ks)) else {
        return;
    };
    let (mut declared, mut noted) = (0, 0);
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        let supported = p.fields().float_controls.f32.signed_zero_inf_nan_preserve;
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let note = a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan);
            let mode = declares_signed_zero_inf_nan(&words, 32);
            if kk.name == "fast_mul" {
                let checked = crate::lower::arith::denormal::checks_tiny_products(&p);
                assert_eq!(
                    (mode, note),
                    (checked && supported, checked && !supported),
                    "{what}: fast needs preservation only for its product's check"
                );
            } else {
                assert_eq!((mode, note), (supported, !supported), "{what}");
                declared += usize::from(mode);
                noted += usize::from(note);
            }
        }
    }
    assert!(
        declared > 0 && noted > 0,
        "{declared} declared, {noted} noted"
    );
}

/// Under `FloatMathsPolicy::Exact` a fused multiply-add the translation computes by a routine (in
/// `double` or in integers without `OpFmaKHR`, a `half` one through `float`, and a checked
/// `OpFmaKHR`'s small results) answers Metal's zero of its sign through float instructions, so a
/// `fast` entry that calls one declares `SignedZeroInfNanPreserve` at 32 bits, and at 64 for a
/// routine in `double`, where the profile supports it, else notes `DeviceSignedZeroInfNan`: without
/// the mode Vulkan lets a device answer a zero of either sign (Metal's `fma(+0, 100, +0)` is `+0`).
/// Nothing else in the kernels needs the mode: their operands are bits, their calls `fast`. Where
/// the profile's functions are Metal's own (MoltenVK), and under the default policy, neither kernel
/// declares it.
#[test]
fn fused_multiply_add_routines_keep_zero_signs() {
    use crate::lower::arith::fma::{self, Route};
    use crate::spirv::testing::committed_profiles_with;
    let ks = [
        Kernel {
            name: "fast_fma_float",
            body: format!(
                "{}{}{}  %r = call fast float @llvm.fma.f32(float %a, float %b, float %c)\n",
                ob("a", 0),
                ob("b", 100f32.to_bits()),
                ob("c", 0)
            ),
            float: true,
            want: None,
            needs: &[],
            note: None,
        },
        Kernel {
            name: "fast_fma_half",
            body: format!(
                "{}{}{}  %x = call fast half @llvm.fma.f16(half %h, half %g, half %u)\n  %xb = bitcast half %x to i16\n  %r = zext i16 %xb to i32\n",
                oh("h", 0),
                oh("g", 0x5640),
                oh("u", 0)
            ),
            float: false,
            want: None,
            needs: &[],
            note: None,
        },
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @llvm.fma.f32(float, float, float)\ndeclare half @llvm.fma.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("fma-zero-signs", &text) else {
        return;
    };
    // The routes and widths some profile keeps zero signs at (by the mode or the note).
    let mut kept = std::collections::BTreeSet::new();
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        let exact = p.fields().float_functions == FloatFunctions::Device;
        let r = fma::routes(&p);
        let fc = p.fields().float_controls;
        let supported = [
            fc.f32.signed_zero_inf_nan_preserve,
            fc.f64.signed_zero_inf_nan_preserve,
        ];
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            // The kernel's route, whether it calls a routine, and whether that one is in `double`.
            let (route, routine) = if kk.float {
                let routine = matches!(r.float, Route::Wide | Route::Soft) || r.tiny.is_some();
                let double = r.float == Route::Wide || r.tiny == Some(Route::Wide);
                (r.float, [routine, double])
            } else {
                (r.half, [r.half == Route::HalfOdd, false])
            };
            let wanted = routine.map(|w| w && exact);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for (i, width) in [32, 64].into_iter().enumerate() {
                let mode = declares_signed_zero_inf_nan(&words, width);
                assert_eq!(mode, wanted[i] && supported[i], "{what}: mode {width}");
                if wanted[i] {
                    kept.insert((route, width));
                }
            }
            let unsupported = (0..2).any(|i| wanted[i] && !supported[i]);
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan),
                unsupported,
                "{what}: note"
            );
        }
    }
    for want in [
        (Route::Wide, 32),
        (Route::Wide, 64),
        (Route::Soft, 32),
        (Route::HalfOdd, 32),
    ] {
        assert!(kept.contains(&want), "no profile keeps {want:?}: {kept:?}");
    }
    for (stem, p) in committed_profiles_with(DevicePolicy::default()).expect("profiles") {
        for kk in &ks {
            let what = format!("{} on {stem} by default", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            for width in [32, 64] {
                assert!(
                    !declares_signed_zero_inf_nan(&words, width),
                    "{what}: mode {width}"
                );
            }
        }
    }
}

/// A float to integer `air.convert` converts NaN to 0, telling it by `OpIsNan` only where the entry
/// declares `SignedZeroInfNanPreserve` for `float`, and by its bits where it does not (a device may
/// assume there that no operand is a NaN, in `OpIsNan` and `NClamp` alike): a `fast` entry has no
/// `OpIsNan`. The NaN is a sum's: a product's check at `±2^-126` would keep zero signs under
/// `FloatMathsPolicy::Exact`. (AIR has no `double` conversion; `math::convert`'s unit tests read a
/// 64-bit container's NaN by its two words.)
#[test]
fn a_convert_tells_a_nan_by_its_bits_where_the_entry_may_assume_none() {
    let float = |name: &'static str, flags: &str| {
        k(
            name,
            format!(
                "{}  %f = bitcast i32 %a to float\n  %g = fadd {flags}float %f, %f\n  %i = call i32 @air.convert.s.i32.f.f32(float %g)\n  %u = call i8 @air.convert.u.i8.f.f32(float %g)\n  %uw = zext i8 %u to i32\n  %s = or i32 %i, %uw\n  %r = add i32 %s, 7\n",
                oi("a", 0x7FC0_0001)
            ),
            7,
        )
    };
    let ks = [
        float("strict_nan_convert", ""),
        float("fast_nan_convert", "fast "),
    ];
    let Some(bitcode) = assemble("nanconvert", &module(&ks)) else {
        return;
    };
    let mut bits = 0;
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let mode = declares_signed_zero_inf_nan(&words, 32);
            let is_nan = count_op(&words, crate::spv::Op::IsNan);
            assert_eq!(is_nan > 0, mode, "{what}: {is_nan} OpIsNan, mode {mode}");
            if kk.name != "strict_nan_convert" {
                assert!(!mode, "{what}: nothing needs preservation");
            }
            bits += usize::from(!mode);
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(got, 7, "{what}: {got:#x}");
        }
    }
    assert!(bits >= PROFILES.len(), "{bits} by bits");
}

/// The profile of committed profile `stem` with its driver's `MetalFloatFunctions` deviation
/// taken or not as `metal` says, and nothing else changed.
fn with_float_functions(stem: &str, metal: bool) -> DeviceProfile {
    use crate::device::{Device, DeviceDescription};
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    let deviation = crate::error::DriverDeviation::MetalFloatFunctions;
    Device::with_deviation(desc, exact_maths(), deviation, metal)
        .expect("device")
        .profile()
        .clone()
}

/// A precise `sinpi`, `cospi`, or `tanpi` call (and a `half` one) is Metal's own form, a routine
/// the entry calls, with no `GLSL.std.450 Sin` or `Cos`, whatever the profile's typed fact says of
/// the device's `float` functions (its driver's deviation `MetalFloatFunctions`): Metal's are none
/// of the instructions SPIRV-Cross emits, so no device answers them. A fast call (`air.fast_sinpi`)
/// is the routine too without the fact; with it, the exact reduction around the device's `Sin` and
/// `Cos`, within Apple's fast bound (the orchestrator's ruling of 2026-10-07). On two profiles that
/// differ only in the fact; the routines' values are within their few ulp on the interpreter (which
/// leaves a device's `Sin` and `Cos` unmodelled).
#[test]
fn pi_scaled_functions_call_metals_routines_but_a_fast_call_on_metals_device() {
    let fast_inputs = |s: String| s.replace("fadd float", "fadd fast float");
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body,
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let x = of("x", 0.3);
    let ks = [
        kernel(
            "pi_sinpi_precise_entry",
            format!("{x}  %r = call float @air.sinpi.f32(float %x)\n"),
        ),
        kernel(
            "pi_cospi_precise_entry",
            format!("{x}  %r = call float @air.cospi.f32(float %x)\n"),
        ),
        kernel(
            "pi_tanpi_precise_entry",
            format!("{x}  %r = call float @air.tanpi.f32(float %x)\n"),
        ),
        kernel(
            "pi_sinpi_fast_entry",
            fast_inputs(format!(
                "{x}  %r = call fast float @air.fast_sinpi.f32(float %x)\n"
            )),
        ),
        kernel(
            "pi_half_sinpi_fast_entry",
            fast_inputs(format!(
                "{x}  %h = fptrunc float %x to half\n  %s = call fast half @air.sinpi.f16(half %h)\n  %r = fpext half %s to float\n"
            )),
        ),
    ];
    // The `half` call's operand is 0.3 rounded to `half`, so its answer is within a `half` ulp
    // (2^13 `float` ulps) of the `float` one.
    let pi03 = std::f64::consts::PI * f64::from(0.3f32);
    let exact = [
        (pi03.sin(), 3.0),
        (pi03.cos(), 3.0),
        (pi03.tan(), 5.0),
        (pi03.sin(), 3.0),
        (pi03.sin(), 8192.0),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.sinpi.f32(float)\ndeclare float @air.cospi.f32(float)\ndeclare float @air.tanpi.f32(float)\ndeclare float @air.fast_sinpi.f32(float)\ndeclare half @air.sinpi.f16(half)\n"
        ),
    );
    let Some(bitcode) = assemble("pi_scaled", &text) else {
        return;
    };
    // `GLSL.std.450 Sin` and `Cos` are instructions 13 and 14 of the set.
    let sin_cos = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.len() > 4
                && (w[4] == 13 || w[4] == 14)
        })
    };
    for metal in [true, false] {
        let p = with_float_functions("moltenvk-m3", metal);
        for (kk, (v, tolerance)) in ks.iter().zip(exact) {
            let what = format!("{} with the fact {metal}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let calls = count_op(&words, crate::spv::Op::FunctionCall);
            if metal && kk.name == "pi_sinpi_fast_entry" {
                assert_eq!(calls, 0, "{what}: calls a routine");
                assert!(sin_cos(&words) >= 2, "{what}: no device Sin and Cos");
                // The device's `Sin` and `Cos` are not the interpreter's to model.
                continue;
            }
            assert!(calls >= 1, "{what}: does not call the routine");
            assert_eq!(sin_cos(&words), 0, "{what}: a device Sin or Cos");
            let got = f64::from(f32::from_bits(interpret(&bitcode, kk.name, &p)));
            let ulp = f64::from(f32::EPSILON) * 2f64.powi(v.abs().log2().floor() as i32);
            assert!(
                (got - v).abs() <= tolerance * ulp,
                "{what}: {got} against {v} ({} ulp)",
                (got - v).abs() / ulp
            );
        }
    }
}

/// The fast reciprocal that an `arcp` division, `air.fast_tanh`, and `air.fast_fmod` divide by is
/// chosen by the profile's typed fact alone. Where the device's `float` functions and division are
/// Metal's own (its driver's deviation `MetalFloatFunctions`), it is the device's `OpFDiv(1, y)`:
/// Metal's fast reciprocal where Metal compiles the module with fast math, and `RN(1 / y)` where
/// the module turns it off. Elsewhere, on a profile with `OpFmaKHR`, it is `RN(1 / y)` refined from
/// that estimate by four more `OpFmaKHR`, counted outside the math library's routines of one
/// operand (`fast::tanh`'s `exp2` is a routine without the fact, whose exact products are
/// `OpFmaKHR` too). The test uses two `moltenvk-m3` profiles that differ only in the fact. The
/// division's and `fmod`'s values are the interpreter's on both: at these operands the device's
/// reciprocal and `RN(1 / y)` agree.
#[test]
fn fast_division_follows_the_metal_float_functions_fact() {
    let kernel = |name: &'static str, body: String, want: Option<f32>| Kernel {
        name,
        body,
        float: true,
        want: want.map(f32::to_bits),
        needs: &[],
        note: None,
    };
    let (x, y) = (of("x", 1e10), of("y", 3.0));
    let ks = [
        kernel(
            "fast_division_arcp",
            format!("{x}{y}  %r = fdiv arcp float %x, %y\n"),
            Some(1e10f32 * (1.0f32 / 3.0)),
        ),
        kernel(
            "fast_division_tanh",
            format!(
                "{}  %r = call fast float @air.fast_tanh.f32(float %x)\n",
                of("x", 0.5)
            ),
            None,
        ),
        kernel(
            "fast_division_fmod",
            format!("{x}{y}  %r = call fast float @air.fast_fmod.f32(float %x, float %y)\n"),
            Some(-512.0),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.fast_tanh.f32(float)\ndeclare float @air.fast_fmod.f32(float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("fast-division-fact", &text) else {
        return;
    };
    for kk in &ks {
        let mut khr = [0usize; 2];
        for (i, metal) in [true, false].into_iter().enumerate() {
            let p = with_float_functions("moltenvk-m3", metal);
            let what = format!("{} with the fact {metal}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                "{what}: {:?}",
                a.precision
            );
            khr[i] = count_op(&without_math_routines(&words), crate::spv::Op::FmaKHR);
            // `fast::tanh`'s `exp2` is the device's, which the interpreter leaves unmodelled.
            if kk.want.is_some() {
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
        }
        // The refinement's four `OpFmaKHR`, and nothing else, tell the two apart.
        assert_eq!(
            khr[1],
            khr[0] + 4,
            "{}: OpFmaKHR with the fact and without",
            kk.name
        );
    }
}

/// Where the device's `float` division is Metal's (its driver's deviation `MetalFloatFunctions`)
/// and Metal compiles the entry with fast math (it declares no `SignedZeroInfNanPreserve`), an
/// `arcp` division is the device's `OpFDiv(x, y)` itself, which MoltenVK hands Metal's compiler as
/// the AIR's own fast `x / y`. Metal's compiler does not treat `x * (1.0 / y)` the same way: case
/// 24c263bb's `(a * 255) / a` was 255 on Metal and 254 as `x` times the device's `OpFDiv(1, y)`. In
/// an entry that declares the mode, and on a profile without the fact, the
/// quotient stays `x` times the profile's reciprocal, a call of the fast division's routine. The
/// interpreter's `OpFDiv` is correctly rounded, so `1e10 / 3` tells the forms apart: `RN(1e10 / 3)`
/// is `0x4f46aea1` and `1e10 · RN(1 / 3)` is `0x4f46aea2`. The device's quotient, Metal's own,
/// records no note; the routine's records `DeviceArithmetic`.
#[test]
fn a_fast_division_is_the_devices_in_a_fast_math_entry_where_division_is_metals() {
    // The fast-math entries' opaque inputs are `fast` too, so nothing there needs preservation.
    let fast_inputs = |s: String| s.replace("fadd float", "fadd fast float");
    let (x, y) = (of("x", 1e10), of("y", 3.0));
    let vector = "  %vx = insertelement <4 x float> undef, float %x, i32 0\n  \
                  %vy = insertelement <4 x float> undef, float %y, i32 0\n  \
                  %vq = fdiv fast <4 x float> %vx, %vy\n  \
                  %r = extractelement <4 x float> %vq, i32 0\n";
    let scalar = "  %r = fdiv fast float %x, %y\n";
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body,
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let ks = [
        kernel(
            "fast_quotient_fast_math",
            fast_inputs(format!("{x}{y}{scalar}")),
        ),
        kernel(
            "fast_quotient_fast_math_v4",
            fast_inputs(format!("{x}{y}{vector}")),
        ),
        kernel("fast_quotient_preserving", format!("{x}{y}{scalar}")),
        kernel("fast_quotient_preserving_v4", format!("{x}{y}{vector}")),
    ];
    // Whether each kernel's entry is compiled with fast math.
    let fast_math = [true, true, false, false];
    let Some(bitcode) = assemble("fast-quotient", &module(&ks)) else {
        return;
    };
    for (kk, fast_math) in ks.iter().zip(fast_math) {
        for metal in [true, false] {
            let p = with_float_functions("moltenvk-m3", metal);
            let what = format!("{} with the fact {metal}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(
                declares_signed_zero_inf_nan(&words, 32),
                !fast_math,
                "{what}: SignedZeroInfNanPreserve"
            );
            // Metal's own quotient in a fast entry records no note; the routine's does.
            let device = metal && fast_math;
            assert_eq!(
                a.precision.contains(&PrecisionNote::DeviceArithmetic),
                !device,
                "{what}: {:?}",
                a.precision
            );
            // The device's form is the entry's one `OpFDiv`, and the module defines no routine;
            // the other is one call of the routine, which holds the reciprocal's `OpFDiv`.
            let form = (
                count_op(&words, crate::spv::Op::Function),
                count_op(&words, crate::spv::Op::FunctionCall),
                count_op(&words, crate::spv::Op::FDiv),
            );
            assert_eq!(form, if device { (1, 0, 1) } else { (2, 1, 1) }, "{what}");
            let got = interpret(&bitcode, kk.name, &p);
            let want: u32 = if device { 0x4F46_AEA1 } else { 0x4F46_AEA2 };
            assert_eq!(got, want, "{what}: {got:#x}");
        }
    }
}

/// On a profile whose `float` functions are Metal's own (`moltenvk-m3` states it), a call in the
/// math mode Metal compiles the entry in is the device's instruction, and a call in the other mode
/// (`precise::atan` in a fast-math entry, `fast::atan` or `fast::acos` in an entry that keeps NaNs)
/// calls the translation's routine; `atan2` is the device's in either, since MoltenVK's is always
/// `precise::atan2`. The same description without the fact takes the routine for every call, as
/// `lavapipe` does: the fact flips the emission. `exp10`, which no instruction is, is Metal's
/// composition of the device's `exp2` with the fact and its routine without, and `fast::rsqrt`, the
/// device's everywhere, is the same word for word with and without the fact. Precise `sin`, `sqrt`,
/// `rsqrt`, and `fmod`, whose fast calls are their own functions, are the device's only in an entry
/// that keeps NaNs; `fast::rsqrt` is the device's everywhere; `fast::sin` and `fast::cos` are the
/// device's in either mode where its functions are Metal's (and elsewhere call the routine that
/// answers past `π`), and so is `fast::tan`'s formula of them in an entry that keeps NaNs, whose
/// quotient calls the fast division's routine on every profile. A `half` `atan` calls its own
/// form's routine on every profile, and so does a precise `sinpi`
/// (`pi_scaled_functions_call_metals_routines_but_a_fast_call_on_metals_device`). A routine's value
/// is within an ulp of the exact one on the interpreter (a correctly rounded root within half of
/// one, `fmod` exact, `fast::atan` Metal's own word), and the plan's estimate bounds the module's
/// words.
#[test]
fn math_library_routines_follow_the_device_fact_and_the_entry_mode() {
    let fast_inputs = |s: String| s.replace("fadd float", "fadd fast float");
    let kernel = |name: &'static str, body: String| Kernel {
        name,
        body,
        float: true,
        want: None,
        needs: &[],
        note: None,
    };
    let x = of("x", 0.5);
    let yx = format!("{}{}", of("y", 0.5), of("x", 0.75));
    let ks = [
        kernel(
            "libm_atan_precise_entry",
            format!("{x}  %r = call float @air.atan.f32(float %x)\n"),
        ),
        kernel(
            "libm_fast_atan_fast_entry",
            fast_inputs(format!(
                "{x}  %r = call fast float @air.fast_atan.f32(float %x)\n"
            )),
        ),
        kernel(
            "libm_fast_atan_precise_entry",
            format!("{x}  %r = call fast float @air.fast_atan.f32(float %x)\n"),
        ),
        kernel(
            "libm_atan_fast_entry",
            fast_inputs(format!(
                "{x}  %r = call fast float @air.atan.f32(float %x)\n"
            )),
        ),
        kernel(
            "libm_asin_precise_entry",
            format!("{x}  %r = call float @air.asin.f32(float %x)\n"),
        ),
        kernel(
            "libm_fast_acos_precise_entry",
            format!("{x}  %r = call fast float @air.fast_acos.f32(float %x)\n"),
        ),
        kernel(
            "libm_atan2_fast_entry",
            fast_inputs(format!(
                "{yx}  %r = call fast float @air.atan2.f32(float %y, float %x)\n"
            )),
        ),
        kernel(
            "libm_exp10_precise_entry",
            format!("{x}  %r = call float @air.exp10.f32(float %x)\n"),
        ),
        // A `half` function is precise in both modes: the device's only in an entry that keeps
        // NaNs (the widenings carry no flags and need nothing preserved).
        kernel(
            "libm_half_atan_fast_entry",
            fast_inputs(format!(
                "{x}  %h = fptrunc float %x to half\n  %s = call fast half @air.atan.f16(half %h)\n  %r = fpext half %s to float\n"
            )),
        ),
        kernel(
            "libm_half_atan_precise_entry",
            format!(
                "{x}  %h = fptrunc float %x to half\n  %s = call half @air.atan.f16(half %h)\n  %r = fpext half %s to float\n"
            ),
        ),
        kernel(
            "libm_sin_precise_entry",
            format!("{x}  %r = call float @air.sin.f32(float %x)\n"),
        ),
        kernel(
            "libm_sin_fast_entry",
            fast_inputs(format!("{x}  %r = call fast float @air.sin.f32(float %x)\n")),
        ),
        kernel(
            "libm_sincos_precise_entry",
            format!(
                "  %c = alloca float, align 4\n{x}  %s = call float @air.sincos.f32(float %x, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd float %s, %cv\n"
            ),
        ),
        // `fast::sincos`: Metal's fast functions, the device's where its functions are Metal's in
        // either mode, and elsewhere the device's inside `[-π, π]` and the routine past it.
        kernel(
            "libm_fast_sincos_fast_entry",
            fast_inputs(format!(
                "  %c = alloca float, align 4\n{x}  %s = call fast float @air.fast_sincos.f32(float %x, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = fadd fast float %s, %cv\n"
            )),
        ),
        // `fast::tan`, not the precise routine: Apple's formula of the fast functions, or the
        // device's `Tan` where it is Metal's fast one.
        kernel(
            "libm_fast_tan_fast_entry",
            fast_inputs(format!(
                "{x}  %r = call fast float @air.fast_tan.f32(float %x)\n"
            )),
        ),
        kernel(
            "libm_sinpi_precise_entry",
            format!(
                "{}  %r = call float @air.sinpi.f32(float %x)\n",
                of("x", 0.3)
            ),
        ),
        kernel(
            "libm_sqrt_precise_entry",
            format!(
                "{}  %r = call float @air.sqrt.f32(float %x)\n",
                of("x", 2.0)
            ),
        ),
        kernel(
            "libm_sqrt_fast_entry",
            fast_inputs(format!(
                "{}  %r = call fast float @air.sqrt.f32(float %x)\n",
                of("x", 2.0)
            )),
        ),
        kernel(
            "libm_rsqrt_fast_entry",
            fast_inputs(format!(
                "{}  %r = call fast float @air.rsqrt.f32(float %x)\n",
                of("x", 2.0)
            )),
        ),
        // `fast::rsqrt`: Metal's fast function, the device's within its bound.
        kernel(
            "libm_fast_rsqrt_fast_entry",
            fast_inputs(format!(
                "{}  %r = call fast float @air.fast_rsqrt.f32(float %x)\n",
                of("x", 2.0)
            )),
        ),
        kernel(
            "libm_fast_sin_precise_entry",
            format!("{x}  %r = call fast float @air.fast_sin.f32(float %x)\n"),
        ),
        kernel(
            "libm_fast_sin_past_pi_fast_entry",
            fast_inputs(format!(
                "{}  %r = call fast float @air.fast_sin.f32(float %x)\n",
                of("x", 4.0)
            )),
        ),
        kernel(
            "libm_fast_tan_precise_entry",
            format!("{x}  %r = call fast float @air.fast_tan.f32(float %x)\n"),
        ),
        kernel(
            "libm_fmod_fast_entry",
            fast_inputs(format!(
                "{}{}  %r = call fast float @air.fmod.f32(float %x, float %y)\n",
                of("x", 7.5),
                of("y", 2.0)
            )),
        ),
    ];
    // Per kernel: the exact value (`None`: not measured here), its tolerance in `float` ulps (a
    // `half` ulp is 2^13 of them), and its routine (`Mode(m)`: a device function, whose routine
    // `moltenvk-m3` calls when `m`, `sinpi`'s on every profile; `Never`: no routine, the same
    // module on both sides of the fact; `Divides`: `fast::tan`'s formula in an entry that keeps
    // NaNs, which calls the fast division's routine on every profile, and the fast sine's and
    // cosine's routines only without the fact).
    #[derive(Clone, Copy)]
    enum Routine {
        Mode(bool),
        Never,
        Divides,
    }
    let sin_pi_03 = (std::f64::consts::PI * f64::from(0.3f32)).sin();
    // Metal's fast `atan` of 0.5, its own word: 1.83 ulp above the exact value.
    let fast_atan_half = f64::from(f32::from_bits(0x3EED_633A));
    let expect: [(Option<f64>, f64, Routine); 24] = [
        (Some(0.5f64.atan()), 1.0, Routine::Mode(false)),
        (Some(fast_atan_half), 0.0, Routine::Mode(false)),
        (Some(fast_atan_half), 0.0, Routine::Mode(true)),
        (Some(0.5f64.atan()), 1.0, Routine::Mode(true)),
        (Some(0.5f64.asin()), 1.0, Routine::Mode(false)),
        (Some(0.5f64.acos()), 1.0, Routine::Mode(true)),
        // Metal's precise `atan2` is its own form, 1.31 ulp from the exact value here.
        (Some(0.5f64.atan2(0.75)), 2.0, Routine::Mode(false)),
        (Some(10f64.powf(0.5)), 1.0, Routine::Mode(false)),
        // A `half` `atan` is Metal's own `half` form's routine on every profile.
        (Some(0.5f64.atan()), 4096.0, Routine::Mode(true)),
        (Some(0.5f64.atan()), 4096.0, Routine::Mode(true)),
        (Some(0.5f64.sin()), 1.0, Routine::Mode(false)),
        (Some(0.5f64.sin()), 1.0, Routine::Mode(true)),
        (Some(0.5f64.sin() + 0.5f64.cos()), 2.0, Routine::Mode(false)),
        (None, 0.0, Routine::Mode(false)),
        (None, 0.0, Routine::Mode(false)),
        // Metal's own form's routine on every profile.
        (Some(sin_pi_03), 3.0, Routine::Mode(true)),
        (Some(2f64.sqrt()), 0.5, Routine::Mode(false)),
        (Some(2f64.sqrt()), 0.5, Routine::Mode(true)),
        (Some(1.0 / 2f64.sqrt()), 0.5, Routine::Mode(true)),
        (None, 0.0, Routine::Never),
        (None, 0.0, Routine::Mode(false)),
        (Some(4f64.sin()), 1.0, Routine::Mode(false)),
        (None, 0.0, Routine::Divides),
        (Some(1.5), 0.0, Routine::Mode(true)),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.atan.f32(float)\ndeclare float @air.fast_atan.f32(float)\ndeclare float @air.asin.f32(float)\ndeclare float @air.fast_acos.f32(float)\ndeclare float @air.atan2.f32(float, float)\ndeclare float @air.exp10.f32(float)\ndeclare half @air.atan.f16(half)\ndeclare float @air.fast_sincos.f32(float, ptr)\ndeclare float @air.fast_tan.f32(float)\ndeclare float @air.fast_sin.f32(float)\ndeclare float @air.sinpi.f32(float)\ndeclare float @air.sqrt.f32(float)\ndeclare float @air.rsqrt.f32(float)\ndeclare float @air.fast_rsqrt.f32(float)\ndeclare float @air.fmod.f32(float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("libm", &text) else {
        return;
    };
    let stated = profile("moltenvk-m3", exact_maths()).expect("profile");
    assert_eq!(
        stated.fields().float_functions,
        with_float_functions("moltenvk-m3", true)
            .fields()
            .float_functions
    );
    let profiles = [
        ("moltenvk-m3", stated),
        (
            "moltenvk-m3 without the fact",
            with_float_functions("moltenvk-m3", false),
        ),
        (
            "lavapipe",
            profile("lavapipe", exact_maths()).expect("profile"),
        ),
    ];
    let mut composition: Vec<(&str, usize, Vec<u32>)> = Vec::new();
    for (column, (label, p)) in profiles.iter().enumerate() {
        for (kk, (exact, tolerance, routine)) in ks.iter().zip(expect) {
            let what = format!("{} on {label}", kk.name);
            let a = compile(&bitcode, kk.name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) = link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let plan = plan_bitcode(
                &bitcode,
                EntryRef::Name(kk.name),
                &FunctionConstantValues::defaults(),
                p,
                &LowerInputs::new(),
                &mut meter(),
            )
            .unwrap_or_else(|e| panic!("{what}: plan: {e}"));
            assert!(
                plan.estimate.words >= a.module.words.len() as u64,
                "{what}: estimate {} < emitted {}",
                plan.estimate.words,
                a.module.words.len()
            );
            // The fact is the device's: with it, Metal's mode rule; without it, every routine.
            let metal = column == 0;
            let calls = match routine {
                Routine::Mode(on_metal) => !metal || on_metal,
                Routine::Divides => true,
                Routine::Never => false,
            };
            assert_eq!(
                count_op(&words, crate::spv::Op::FunctionCall) > 0,
                calls,
                "{what}: calls a routine"
            );
            if matches!(routine, Routine::Divides) {
                // The division's call; and the sine's and cosine's where they are routines, with
                // the fused step's from inside them where the profile has no `OpFmaKHR`.
                let n = count_op(&words, crate::spv::Op::FunctionCall);
                if metal {
                    assert_eq!(n, 1, "{what}: calls");
                } else {
                    assert!(n >= 3, "{what}: {n} calls");
                }
            }
            if matches!(routine, Routine::Never) {
                composition.push((kk.name, column, a.module.words.clone()));
            }
            match exact {
                Some(v) if calls => {
                    let got = f64::from(f32::from_bits(interpret(&bitcode, kk.name, p)));
                    let ulp = f64::from(f32::EPSILON) * 2f64.powi(v.abs().log2().floor() as i32);
                    assert!(
                        (got - v).abs() <= tolerance * ulp,
                        "{what}: {got} against {v} ({} ulp)",
                        (got - v).abs() / ulp
                    );
                }
                _ => {}
            }
        }
    }
    // A composition's module is the same on both sides of the fact.
    for (name, column, words) in &composition {
        if *column == 0 {
            let without = composition
                .iter()
                .find(|(n, c, _)| n == name && *c == 1)
                .expect("the module without the fact");
            assert!(words == &without.2, "{name} changes with the fact");
        }
    }
}

// ---- Metal's NaN ----

/// `%x` = an opaque `float` with bits `b` (any pattern, NaNs included): `llvm.umin` of a constant
/// and `-1`, bitcast.
fn ob(name: &str, b: u32) -> String {
    format!(
        "  %{name}.w = call i32 @llvm.umin.i32(i32 {}, i32 -1)\n  %{name} = bitcast i32 %{name}.w to float\n",
        b as i32
    )
}

/// Committed profile `stem` with its driver's `MetalCanonicalNans` deviation taken or not as
/// `fact` says.
fn profile_with_canonical_nans(stem: &str, fact: bool) -> DeviceProfile {
    profile_with_fact(
        stem,
        crate::error::DriverDeviation::MetalCanonicalNans,
        fact,
    )
}

/// Committed profile `stem` with its driver's deviation `deviation` taken or not as `fact` says.
fn profile_with_fact(
    stem: &str,
    deviation: crate::error::DriverDeviation,
    fact: bool,
) -> DeviceProfile {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/profiles")
        .join(format!("{stem}.json"));
    let text = std::fs::read_to_string(&path).expect("profile");
    let desc = DeviceDescription::from_khronos_json(&text, None).expect("description");
    Device::with_deviation(desc, exact_maths(), deviation, fact)
        .expect("device")
        .profile()
        .clone()
}

/// The bits tests' `OpUGreaterThan` in the entry point's own function of module `words` (one
/// entry point): a module-scope routine tests by the bits in entries of every float mode, and
/// compares integers so (a precise division's).
fn entry_bits_tests(words: &[u32]) -> usize {
    let (mut entry, mut inside, mut n) = (None, false, 0);
    let mut i = 5;
    while i < words.len() {
        let (len, opcode) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if opcode == crate::spv::Op::EntryPoint as u32 {
            entry = Some(words[i + 2]);
        } else if opcode == crate::spv::Op::Function as u32 {
            inside = Some(words[i + 2]) == entry;
        } else if opcode == crate::spv::Op::FunctionEnd as u32 {
            inside = false;
        } else if inside && opcode == crate::spv::Op::UGreaterThan as u32 {
            n += 1;
        }
        i += len;
    }
    n
}

/// The NaN tests in module `words`: `OpIsNan` and the bits test's `OpUGreaterThan`.
fn nan_tests(words: &[u32]) -> (usize, usize) {
    (
        count_op(words, crate::spv::Op::IsNan),
        count_op(words, crate::spv::Op::UGreaterThan),
    )
}

/// Whether module `words` decorates any value `FPFastMathMode`.
fn decorates_fast_math(words: &[u32]) -> bool {
    count_ops(words, |inst| {
        inst[0] & 0xFFFF == crate::spv::Op::Decorate as u32
            && inst.get(2) == Some(&(crate::spv::Decoration::FPFastMathMode as u32))
    }) > 0
}

/// The kernels of the NaN-model tests: their declarations beside [`DECLS`].
const NAN_DECLS: &str = "declare float @air.fast_round.f32(float)\ndeclare float @air.fast_asin.f32(float)\ndeclare float @air.fast_tan.f32(float)\ndeclare float @air.asin.f32(float)\n";

/// A kernel storing `%s`'s bits, `float` `%s` defined by `body`.
fn nan_kernel(name: &'static str, body: String, want: Option<u32>) -> Kernel {
    Kernel {
        want,
        ..k(name, format!("{body}  %r = bitcast float %s to i32\n"), 0)
    }
}

/// The canonical NaN of `float` arithmetic is left to the device exactly where the profile takes
/// its driver's deviation `MetalCanonicalNans`, that its arithmetic answers it, and selected
/// everywhere else; flipping the fact on either device flips that emission and nothing else. The
/// other classes are selected whatever the fact: `round` returns a NaN operand's own bits. The
/// interpreter keeps a NaN operand's payload, as such a device would.
#[test]
fn the_canonical_nan_follows_the_device_fact() {
    let ks = [
        nan_kernel(
            "nan_fadd",
            format!("{}  %s = fadd float %x, 1.0\n", ob("x", 0xFFC0_1234)),
            None,
        ),
        nan_kernel(
            "nan_round",
            format!(
                "{}  %s = call float @air.round.f32(float %x)\n",
                ob("x", 0x7F80_0001)
            ),
            Some(0x7F80_0001),
        ),
    ];
    let text = module(&ks);
    let Some(bitcode) = assemble("nan-fact", &text) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        let mut emitted: Vec<Vec<Vec<u32>>> = Vec::new();
        for fact in [true, false] {
            let p = profile_with_canonical_nans(stem, fact);
            let mut per = Vec::new();
            for kk in &ks {
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let got = interpret(&bitcode, kk.name, &p);
                let (is_nan, bits) = nan_tests(&words);
                if kk.name == "nan_fadd" {
                    assert!(f32::from_bits(got).is_nan(), "{what}: {got:#x}");
                    assert_eq!(got == 0x7FC0_0000, !fact, "{what}: {got:#x}");
                    assert_eq!(is_nan + bits, usize::from(!fact), "{what}");
                } else {
                    assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
                }
                assert!(
                    !a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                    "{what}: {:?}",
                    a.precision
                );
                per.push(words);
            }
            emitted.push(per);
        }
        assert_ne!(emitted[0][0], emitted[1][0], "nan_fadd on {stem}");
        assert_eq!(emitted[0][1], emitted[1][1], "nan_round on {stem}");
    }
    // The committed profiles state the fact for the measured device only.
    for (stem, fact) in [("moltenvk-m3", true), ("lavapipe", false)] {
        let p = profile(stem, exact_maths()).expect("profile");
        assert_eq!(
            p.fingerprint(),
            profile_with_canonical_nans(stem, fact).fingerprint(),
            "{stem}"
        );
    }
}

// ---- Metal's float denormals ----

/// Metal reads an `f32` denormal operand of `+ - *`, a comparison, `sqrt` and `ldexp` as the zero
/// of its sign and flushes a denormal result, and `floor` and `ceil` read their operand so. Where
/// the profile takes its driver's measured deviation `MetalFloatDenormals` the device's own reading
/// is left to it and the module requires the fact; elsewhere the module flushes them itself, so the
/// interpreter, which keeps denormals as lavapipe does, gives Metal's answer. `floor` and `ceil`
/// are flushed whatever the fact: no device measured reads their operand as Metal does; and `ldexp`
/// is Metal's whatever the fact, computed on the operand's bits for every exponent. Flipping the
/// fact flips that and nothing else; neither side notes `DeviceDenormals`. The operands are bits
/// (`llvm.umin` of constants), so each kernel's float operation is its only one.
#[test]
fn float_denormals_follow_the_device_fact() {
    // The kernel's float `%r` from `body`, Metal's word, and the word of a device that keeps
    // denormals (the interpreter, where the module leaves them to the device).
    let float = |name: &'static str, body: String, metal: u32| Kernel {
        name,
        body,
        float: true,
        want: Some(metal),
        needs: &[],
        note: None,
    };
    let den = f32::from_bits(0x0040_0000);
    let (ks, devices): (Vec<Kernel>, Vec<u32>) = [
        (
            float(
                "den_fadd",
                format!(
                    "{}{}  %r = fadd float %x, %y\n",
                    ob("x", 0x0040_0000),
                    ob("y", 0x0040_0000)
                ),
                0,
            ),
            0x0080_0000,
        ),
        (
            float(
                "den_fmul_tiny",
                format!(
                    "{}{}  %r = fmul float %x, %y\n",
                    ob("x", 0x1F80_0000),
                    ob("y", 0x1F80_0000)
                ),
                0,
            ),
            0x0020_0000,
        ),
        (
            float(
                "den_fmul_tiny_negative",
                format!(
                    "{}{}  %r = fmul float %x, %y\n",
                    ob("x", 0x9F80_0000),
                    ob("y", 0x1F80_0000)
                ),
                0x8000_0000,
            ),
            0x8020_0000,
        ),
        (
            k(
                "den_fcmp",
                format!(
                    "{}{}  %c = fcmp ogt float %x, %y\n  %r = zext i1 %c to i32\n",
                    ob("x", 2),
                    ob("y", 1)
                ),
                0,
            ),
            1,
        ),
        (
            float(
                "den_floor",
                format!(
                    "{}  %r = call float @air.floor.f32(float %x)\n",
                    ob("x", 0x8000_0001)
                ),
                0x8000_0000,
            ),
            0x8000_0000,
        ),
        (
            float(
                "den_ceil",
                format!("{}  %r = call float @air.ceil.f32(float %x)\n", ob("x", 1)),
                0,
            ),
            0,
        ),
        (
            float(
                "den_sqrt",
                format!(
                    "{}  %r = call float @air.sqrt.f32(float %x)\n",
                    ob("x", 0x0040_0000)
                ),
                0,
            ),
            den.sqrt().to_bits(),
        ),
        (
            float(
                "den_ldexp",
                format!(
                    "{}  %r = call float @air.ldexp.f32(float %x, i32 1)\n",
                    ob("x", 0x0040_0000)
                ),
                0,
            ),
            0x0080_0000,
        ),
        (
            float(
                "den_ldexp_tiny",
                format!(
                    "{}  %r = call float @air.ldexp.f32(float %x, i32 -1)\n",
                    ob("x", 0x0080_0000)
                ),
                0,
            ),
            0x0040_0000,
        ),
    ]
    .into_iter()
    .unzip();
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.floor.f32(float)\ndeclare float @air.ceil.f32(float)\ndeclare float @air.sqrt.f32(float)\ndeclare float @air.ldexp.f32(float, i32)\n"),
    );
    let Some(bitcode) = assemble("denormal-fact", &text) else {
        return;
    };
    let deviation = crate::error::DriverDeviation::MetalFloatDenormals;
    for stem in ["moltenvk-m3", "lavapipe"] {
        let mut emitted: Vec<Vec<Vec<u32>>> = Vec::new();
        for fact in [true, false] {
            let p = profile_with_fact(stem, deviation, fact);
            let mut per = Vec::new();
            for (kk, device) in ks.iter().zip(&devices) {
                let what = format!("{} on {stem}, MetalFloatDenormals {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let own = matches!(
                    kk.name,
                    "den_floor" | "den_ceil" | "den_ldexp" | "den_ldexp_tiny"
                );
                let left = fact && !own;
                assert_eq!(
                    a.requirements
                        .properties
                        .contains(&crate::device::requirements::VkProperty::MetalFloatDenormals),
                    left,
                    "{what}"
                );
                assert!(
                    !a.precision.contains(&PrecisionNote::DeviceDenormals),
                    "{what}: {:?}",
                    a.precision
                );
                let got = interpret(&bitcode, kk.name, &p);
                let want = if left {
                    *device
                } else {
                    kk.want.expect("Metal's word")
                };
                assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
                per.push(words);
            }
            emitted.push(per);
        }
        for (i, kk) in ks.iter().enumerate() {
            let own = matches!(
                kk.name,
                "den_floor" | "den_ceil" | "den_ldexp" | "den_ldexp_tiny"
            );
            assert_eq!(
                emitted[0][i] == emitted[1][i],
                own,
                "{} on {stem}: the module with and without the fact",
                kk.name
            );
        }
    }
    // The committed profiles state the fact for the measured device only.
    for (stem, fact) in [("moltenvk-m3", true), ("lavapipe", false)] {
        let p = profile(stem, exact_maths()).expect("profile");
        assert_eq!(
            p.fingerprint(),
            profile_with_fact(stem, deviation, fact).fingerprint(),
            "{stem}"
        );
    }
}

/// `air.sincos` answers the NaN its `sin` and `cos` answer as calls of their own, since Metal's
/// `sincos` is its `sin` and `cos` word for word (cases 7b283404 and f6524207, every pair): precise
/// `sin` and `cos` without fast-math flags answer the canonical NaN, so each result selects it
/// where the device fact does not leave it to the device, and the call notes no
/// `DeviceNanEncoding`. The cosine is stored, so it is selected even where the sine is not read. On
/// `lavapipe`, whose sine and cosine are the translation's routines, the interpreter runs both
/// results: each is `0x7FC00000` for an operand NaN with a payload and the sign bit, which the
/// routines alone pass on (NVIDIA answered `0x7FFFFFFF`).
#[test]
fn sincos_answers_the_nan_of_sin_and_cos() {
    let kernel = |name: &'static str, read: &str| {
        Kernel {
        name,
        body: format!(
            "{}  %c = alloca float, align 4\n  %t = call float @air.sincos.f32(float %x, ptr %c)\n  %cv = load float, ptr %c, align 4\n  %r = bitcast float {read} to i32\n",
            ob("x", 0xFFC0_1234)
        ),
        float: false,
        want: None,
        needs: &[],
        note: None,
    }
    };
    let ks = [
        kernel("sincos_nan_sine", "%t"),
        kernel("sincos_nan_cosine", "%cv"),
    ];
    // The selects each kernel's results add where the fact leaves the NaN to the module: the
    // sine's and the cosine's, then the cosine's alone (the sine is not read).
    let selects = [2, 1];
    let text = module(&ks);
    let Some(bitcode) = assemble("sincos-nan", &text) else {
        return;
    };
    for stem in ["moltenvk-m3", "lavapipe"] {
        for (kk, selects) in ks.iter().zip(selects) {
            let mut tests = Vec::new();
            for fact in [true, false] {
                let p = profile_with_canonical_nans(stem, fact);
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                assert!(
                    !a.precision.contains(&PrecisionNote::DeviceNanEncoding),
                    "{what}: {:?}",
                    a.precision
                );
                let (is_nan, bits) = nan_tests(&words);
                tests.push(is_nan + bits);
                if stem == "lavapipe" && !fact {
                    let got = interpret(&bitcode, kk.name, &p);
                    assert_eq!(got, 0x7FC0_0000, "{what}: {got:#x}");
                }
            }
            assert_eq!(tests[1], tests[0] + selects, "{} on {stem}", kk.name);
        }
    }
}

/// A result whose every use drops its NaN's bits (a float comparison, or arithmetic that answers
/// the canonical NaN itself) needs no select, so a chain of `float` arithmetic tests once, at its
/// end, and a result only compared tests never. The chain has no division: the precise division
/// routine answers the canonical NaN through bit comparisons of its own, which the count would read
/// as the model's tests.
#[test]
fn a_nan_is_selected_only_where_its_bits_are_observed() {
    let xyz = format!(
        "{}{}{}",
        ob("x", 0xFFC0_1234),
        ob("y", 0x4000_0000),
        ob("z", 0x7F80_0001)
    );
    let ks = [
        nan_kernel(
            "nan_single",
            format!("{xyz}  %s = fadd float %x, %y\n"),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "nan_chain",
            format!("{xyz}  %p = fmul float %x, %y\n  %q = fsub float %p, %y\n  %s = fadd float %q, %z\n"),
            Some(0x7FC0_0000),
        ),
        k(
            "nan_compared",
            format!("{xyz}  %p = fmul float %x, %y\n  %c = fcmp olt float %p, 0.0\n  %r = zext i1 %c to i32\n"),
            0,
        ),
    ];
    let Some(bitcode) = assemble("nan-observed", &module(&ks)) else {
        return;
    };
    let p = profile("lavapipe", exact_maths()).expect("profile");
    for (kk, tests) in ks.iter().zip([1, 1, 0]) {
        let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{}: {e:?}", kk.name));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{}: {e}", kk.name));
        let (is_nan, bits) = nan_tests(&words);
        assert_eq!(is_nan + bits, tests, "{}", kk.name);
        let got = interpret(&bitcode, kk.name, &p);
        assert_eq!(Some(got), kk.want, "{}: {got:#x}", kk.name);
    }
}

/// The select's NaN test is `OpIsNan` only in an entry that declares `SignedZeroInfNanPreserve 32`;
/// where every float operation carries `nnan`/`ninf`/`nsz` the entry declares nothing, a device may
/// fold `OpIsNan`, and the test is the bits instead, which still answers Metal's NaN. No module
/// decorates a value `FPFastMathMode`, so nothing tells a device a tested value is no NaN; nor do
/// the NaN-model probes' fast-math fixtures.
#[test]
fn a_nan_test_no_float_mode_folds() {
    let x = ob("x", 0xFFC0_1234);
    let ks = [
        nan_kernel(
            "strict_fadd",
            format!("{x}  %s = fadd float %x, 1.0\n"),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "fast_fadd",
            format!("{x}  %s = fadd fast float %x, 1.0\n"),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "fast_round",
            format!("{x}  %s = call fast float @air.fast_round.f32(float %x)\n"),
            Some(0xFFC0_1234),
        ),
        // The device's `asin` and `tan`, which the interpreter does not evaluate.
        nan_kernel(
            "fast_asin",
            format!("{x}  %s = call fast float @air.fast_asin.f32(float %x)\n"),
            None,
        ),
        nan_kernel(
            "fast_tan",
            format!("{x}  %s = call fast float @air.fast_tan.f32(float %x)\n"),
            None,
        ),
        nan_kernel(
            "strict_asin",
            format!("{x}  %s = call float @air.asin.f32(float %x)\n"),
            None,
        ),
    ];
    let text = module(&ks).replace(DECLS, &format!("{DECLS}{NAN_DECLS}"));
    let Some(bitcode) = assemble("nan-test", &text) else {
        return;
    };
    let (mut bits_tested, mut is_nan_tested) = (0, 0);
    for stem in PROFILES.into_iter().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(!decorates_fast_math(&words), "{what}");
            let (is_nan, bits) = nan_tests(&words);
            let mode = declares_signed_zero_inf_nan(&words, 32);
            assert_eq!(
                mode,
                kk.name.starts_with("strict")
                    && p.fields().float_controls.f32.signed_zero_inf_nan_preserve,
                "{what}"
            );
            if !mode {
                assert_eq!(
                    is_nan, 0,
                    "{what}: OpIsNan without SignedZeroInfNanPreserve"
                );
            } else {
                assert_eq!(
                    entry_bits_tests(&words),
                    0,
                    "{what}: the bits test where OpIsNan is safe"
                );
            }
            bits_tested += usize::from(bits > 0);
            is_nan_tested += usize::from(is_nan > 0);
            if let Some(want) = kk.want {
                let got = interpret(&bitcode, kk.name, &p);
                // moltenvk-m3 leaves a sum's NaN to its device; the interpreter is not one.
                if stem != "moltenvk-m3" || kk.name == "fast_round" {
                    assert_eq!(got, want, "{what}: {got:#x}");
                }
            }
        }
    }
    assert!(
        bits_tested > 0 && is_nan_tested > 0,
        "{bits_tested} {is_nan_tested}"
    );
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine");
    for entry in ["nan_arith_fast", "nan_functions_fast", "nan_functions"] {
        let bitcode = match llvm_assemble(&dir.join(format!("{entry}.ll"))) {
            Ok(b) => b,
            Err(AssembleError::Missing(t)) => {
                eprintln!("SKIPPED: {t}");
                return;
            }
            Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {entry}: {e}"),
        };
        for stem in PROFILES.into_iter().chain(["lavapipe"]) {
            let p = profile(stem, exact_maths()).expect("profile");
            let what = format!("{entry} on {stem}");
            let a = compile(&bitcode, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert!(!decorates_fast_math(&words), "{what}");
        }
    }
}

/// `%name` = an opaque `half` with the bits `b`.
fn oh(name: &str, b: u16) -> String {
    format!(
        "  %{name}.w = call i32 @llvm.umin.i32(i32 {b}, i32 -1)\n  %{name}.t = trunc i32 %{name}.w to i16\n  %{name} = bitcast i16 %{name}.t to half\n"
    )
}

/// A kernel storing the bits of `half` `%s`, defined by `body`, zero-extended.
fn half_nan_kernel(name: &'static str, body: String, want: Option<u32>) -> Kernel {
    Kernel {
        want,
        ..k(
            name,
            format!("{body}  %b = bitcast half %s to i16\n  %r = zext i16 %b to i32\n"),
            0,
        )
    }
}

/// Class B: Metal's `half` `+`, `*`, and `air.fma.f16`, with or without fast-math flags, and its
/// narrowing of a `float` to `half` (`fptrunc`, and `air.convert` from class E's version) answer
/// the canonical `half` NaN `0x7E00`, which the lowering selects (one NaN test per result, none
/// where the device fact leaves the canonical class to the device); a chain selects once, at its
/// end. `half` `-` is class K's, selected the same way. The interpreter's `OpQuantizeToF16` answers
/// `0x7E00` itself, so the selects are counted against the same kernel compiled with the fact.
#[test]
fn half_nans_answer_metals_half_nan() {
    let hx = oh("hx", 0xFE01);
    let x = ob("x", 0xFFC0_1234);
    let ks = [
        half_nan_kernel(
            "half_fadd",
            format!("{hx}  %s = fadd half %hx, 0xH3C00\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fmul",
            format!("{hx}  %s = fmul half %hx, 0xH4000\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fma",
            format!("{hx}  %s = call half @air.fma.f16(half %hx, half 0xH3C00, half 0xH0000)\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "narrow",
            format!("{x}  %s = fptrunc float %x to half\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "narrow_then_add",
            format!("{x}  %n = fptrunc float %x to half\n  %s = fadd half %n, 0xH3C00\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "narrow_convert",
            format!(
                "{x}  %v = insertelement <4 x float> <float 1.0, float 1.0, float 1.0, float 1.0>, float %x, i32 2\n  %w = call <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float> %v)\n  %s = extractelement <4 x half> %w, i32 2\n"
            ),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fadd_fast",
            format!("{hx}  %s = fadd fast half %hx, 0xH3C00\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fsub",
            format!("{hx}  %s = fsub half %hx, 0xH3C00\n"),
            Some(0x7E00),
        ),
    ];
    let selects = [1, 1, 1, 1, 1, 1, 1, 1];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare half @air.fma.f16(half, half, half)\ndeclare <4 x half> @air.convert.f.v4f16.f.v4f32(<4 x float>)\n"),
    );
    let Some(bitcode) = assemble("nan-half", &text) else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        for (kk, n) in ks.iter().zip(selects) {
            let mut tests = Vec::new();
            for fact in [false, true] {
                let p = profile_with_canonical_nans(stem, fact);
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (is_nan, bits) = nan_tests(&words);
                tests.push(is_nan + bits);
                if let Some(want) = kk.want {
                    let got = interpret(&bitcode, kk.name, &p);
                    assert_eq!(got, want, "{what}: {got:#x}");
                }
            }
            assert_eq!(tests[0], tests[1] + n, "{} on {stem}", kk.name);
        }
    }
}

/// Class C: Metal widens a `half` NaN to the canonical `float` NaN `0x7FC00000`, sign and payload
/// dropped, by `fpext` and by its intrinsic form `air.convert`. A `half` is held widened, so the
/// widening emits no device operation and every device keeps the payload, MoltenVK included: the
/// select is emitted whatever the device fact states. The interpreter keeps the payload as a device
/// would, so the stored bits show the select. A widening only compared needs none.
#[test]
fn a_widened_half_nan_is_canonical_on_every_profile() {
    let hx = oh("hx", 0xFE01);
    let ks = [
        nan_kernel(
            "widen",
            format!("{hx}  %s = fpext half %hx to float\n"),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "widen_vector",
            format!(
                "{hx}  %v = insertelement <2 x half> <half 0xH3C00, half 0xH3C00>, half %hx, i32 1\n  %w = fpext <2 x half> %v to <2 x float>\n  %s = extractelement <2 x float> %w, i32 1\n"
            ),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "widen_convert",
            format!(
                "{hx}  %v = insertelement <4 x half> <half 0xH3C00, half 0xH3C00, half 0xH3C00, half 0xH3C00>, half %hx, i32 1\n  %w = call <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half> %v)\n  %s = extractelement <4 x float> %w, i32 1\n"
            ),
            Some(0x7FC0_0000),
        ),
        k(
            "widen_compared",
            format!("{hx}  %s = fpext half %hx to float\n  %c = fcmp olt float %s, 0.0\n  %r = zext i1 %c to i32\n"),
            0,
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare <4 x float> @air.convert.f.v4f32.f.v4f16(<4 x half>)\n"),
    );
    let Some(bitcode) = assemble("nan-widen", &text) else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        for fact in [false, true] {
            let p = profile_with_canonical_nans(stem, fact);
            for kk in &ks {
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (is_nan, bits) = nan_tests(&words);
                let selects = usize::from(kk.name != "widen_compared");
                assert!(is_nan + bits >= selects, "{what}");
                if kk.name == "widen_compared" {
                    assert_eq!(is_nan + bits, 0, "{what}");
                }
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
        }
    }
}

/// Class D: Metal's `half` `fneg` and `fabs` answer the canonical `half` NaN `0x7E00`, with or
/// without fast-math flags. A device computes both exactly on the held value's sign, so the select
/// is emitted whatever the device fact states: one NaN test more than the same value stored
/// unchanged, on either profile and in either fact state. An `fadd` whose only use is the `fneg`
/// selects nothing of its own, and a `fneg` only compared none. (The `fadd`'s rounding to `half` in
/// integer arithmetic, `narrow`'s `QuantizeForm::Integer` on lavapipe, makes two unsigned
/// comparisons of its own, which [`integer_roundings`] finds and the count leaves out.) The
/// interpreter keeps a `half` NaN's payload through the held value, as a device does (the value
/// stored unchanged is `0xFE01`), so the stored bits show the select.
#[test]
fn half_sign_operations_answer_the_canonical_nan_on_every_profile() {
    let hx = oh("hx", 0xFE01);
    let ks = [
        half_nan_kernel(
            "half_kept",
            format!("{hx}  %s = freeze half %hx\n"),
            Some(0xFE01),
        ),
        half_nan_kernel(
            "half_fneg",
            format!("{hx}  %s = fneg half %hx\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fneg_fast",
            format!("{hx}  %s = fneg fast half %hx\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fabs",
            format!("{hx}  %s = call half @air.fabs.f16(half %hx)\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fabs_fast",
            format!("{hx}  %s = call fast half @air.fabs.f16(half %hx)\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_fadd_then_fneg",
            format!("{hx}  %n = fadd half %hx, 0xH3C00\n  %s = fneg half %n\n"),
            Some(0x7E00),
        ),
        k(
            "half_fneg_compared",
            format!("{hx}  %n = fneg half %hx\n  %c = fcmp olt half %n, 0xH0000\n  %r = zext i1 %c to i32\n"),
            0,
        ),
    ];
    let text = module(&ks).replace(DECLS, &format!("{DECLS}declare half @air.fabs.f16(half)\n"));
    let Some(bitcode) = assemble("nan-half-sign", &text) else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        for fact in [false, true] {
            let p = profile_with_canonical_nans(stem, fact);
            let mut tests = Vec::new();
            for kk in &ks {
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (is_nan, bits) = nan_tests(&words);
                tests.push(is_nan + bits - 2 * integer_roundings(&words));
                let got = interpret(&bitcode, kk.name, &p);
                assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
            }
            let what = format!("on {stem}, MetalCanonicalNans {fact}: {tests:?}");
            for t in &tests[1..6] {
                assert_eq!(*t, tests[0] + 1, "{what}");
            }
            assert_eq!(tests[6], 0, "{what}");
        }
    }
}

/// Class E: Metal's `float` `fneg` and `fabs` (`air.fabs`, `air.fast_fabs`, and `llvm.fabs`, which
/// takes `air.fabs`'s answer) flip or clear the sign bit alone, with and without fast-math flags: a
/// NaN keeps its payload, signaling or not, and a denormal stays. The lowering moves the sign as an
/// integer on the bits, so no device's `OpFNegate` or `FAbs` (RADV's `fneg` keeps a NaN's sign,
/// NVIDIA's `fabs` answers `0x7FFFFFFF`, RADV and ANV flush a denormal) decides it, on every
/// profile: no float negation or absolute value is emitted, no NaN is selected, and no device note
/// is recorded.
#[test]
fn float_sign_operations_move_the_bits_on_every_profile() {
    let snan = ob("x", 0x7F80_0001);
    let neg_snan = ob("x", 0xFF80_0001);
    let ks = [
        nan_kernel(
            "float_kept",
            format!("{neg_snan}  %s = freeze float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fneg",
            format!("{snan}  %s = fneg float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fneg_fast",
            format!("{snan}  %s = fneg fast float %x\n"),
            Some(0xFF80_0001),
        ),
        nan_kernel(
            "float_fabs",
            format!("{neg_snan}  %s = call float @air.fabs.f32(float %x)\n"),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "float_fabs_fast",
            format!("{neg_snan}  %s = call fast float @air.fast_fabs.f32(float %x)\n"),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "float_llvm_fabs",
            format!(
                "{}  %s = call float @llvm.fabs.f32(float %x)\n",
                ob("x", 0xFFC0_1234)
            ),
            Some(0x7FC0_1234),
        ),
        nan_kernel(
            "float_fneg_denormal",
            format!("{}  %s = fneg float %x\n", ob("x", 1)),
            Some(0x8000_0001),
        ),
        nan_kernel(
            "float_fabs_denormal",
            format!(
                "{}  %s = call fast float @air.fast_fabs.f32(float %x)\n",
                ob("x", 0x8000_0001)
            ),
            Some(1),
        ),
        nan_kernel(
            "float_fneg_vector",
            format!("{snan}  %v = insertelement <2 x float> <float 1.0, float 1.0>, float %x, i32 1\n  %w = fneg <2 x float> %v\n  %s = extractelement <2 x float> %w, i32 1\n"),
            Some(0xFF80_0001),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!("{DECLS}declare float @air.fabs.f32(float)\ndeclare float @air.fast_fabs.f32(float)\ndeclare float @llvm.fabs.f32(float)\n"),
    );
    let Some(bitcode) = assemble("nan-float-sign", &text) else {
        return;
    };
    let fabs = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::FAbs as u32))
        })
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(count_op(&words, crate::spv::Op::FNegate), 0, "{what}");
            assert_eq!(fabs(&words), 0, "{what}");
            assert_eq!(nan_tests(&words), (0, 0), "{what}");
            for note in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class F: Metal's `float` `fmin`, `fmax`, `min3`, `max3`, and `clamp`, precise and `fast_` alike,
/// and `llvm.minnum`/`llvm.maxnum` with every flag (Metal's fast-math `a < b ? a : b`), with none
/// and with some, answer an operand bit for bit on every profile (`nan_minmax`, `nan_minmax_fast`,
/// and their zeros-and-denormals cases, every lane; for the other flag states): a NaN loses, two
/// NaNs and a tie answer the second operand, and a denormal compares as zero but is returned
/// itself. Off a profile that takes `DriverDeviation::MetalMinMax` no device `NMin`, `NMax`, or
/// `NClamp` decides it; on one those instructions do, measured to be Metal's (the interpreter
/// models them so), and the module requires `VkProperty::MetalMinMax`. No
/// device note is recorded.
#[test]
fn float_min_max_answer_metals_operand_on_every_profile() {
    let two = |a: u32, b: u32| format!("{}{}", ob("x", a), ob("y", b));
    let three = |a: u32, b: u32, c: u32| format!("{}{}{}", ob("x", a), ob("y", b), ob("z", c));
    let ks = [
        nan_kernel(
            "fmin_two_nans",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0xFFC0_1234, 0x7F80_0001)
            ),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "fmin_a_nan_loses",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0xFFC0_1234, 0x4000_0000)
            ),
            Some(0x4000_0000),
        ),
        nan_kernel(
            "fmax_tie_of_zeros",
            format!(
                "{}  %s = call float @air.fmax.f32(float %x, float %y)\n",
                two(0, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "fmin_denormal_compares_as_zero",
            format!(
                "{}  %s = call float @air.fmin.f32(float %x, float %y)\n",
                two(0, 1)
            ),
            Some(1),
        ),
        nan_kernel(
            "fast_fmin_tie_of_zeros",
            format!(
                "{}  %s = call fast float @air.fast_fmin.f32(float %x, float %y)\n",
                two(0x8000_0000, 0)
            ),
            Some(0),
        ),
        nan_kernel(
            "minnum_fast_a_nan_loses",
            format!(
                "{}  %s = call fast float @llvm.minnum.f32(float %x, float %y)\n",
                two(0x7FC0_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "maxnum_fast_denormal_tie",
            format!(
                "{}  %s = call fast float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0x8000_0000, 0x8000_0001)
            ),
            Some(0x8000_0001),
        ),
        nan_kernel(
            "minnum_unflagged_denormal_tie",
            format!(
                "{}  %s = call float @llvm.minnum.f32(float %x, float %y)\n",
                two(0, 1)
            ),
            Some(1),
        ),
        nan_kernel(
            "maxnum_nsz_arcp_denormal_tie",
            format!(
                "{}  %s = call nsz arcp float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0x8000_0001, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "minnum_nnan_a_nan_loses",
            format!(
                "{}  %s = call nnan float @llvm.minnum.f32(float %x, float %y)\n",
                two(0x7FC0_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "maxnum_unflagged_two_nans",
            format!(
                "{}  %s = call float @llvm.maxnum.f32(float %x, float %y)\n",
                two(0xFFC0_0001, 0x7F80_0001)
            ),
            Some(0x7F80_0001),
        ),
        nan_kernel(
            "clamp_of_a_nan",
            format!(
                "{}  %s = call float @air.clamp.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x3F80_0000, 0x4000_0000)
            ),
            Some(0x3F80_0000),
        ),
        nan_kernel(
            "fast_min3_of_two_nans",
            format!(
                "{}  %s = call fast float @air.fast_fmin3.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x7F80_0001, 0x8000_0000)
            ),
            Some(0x8000_0000),
        ),
        nan_kernel(
            "max3_keeps_a_nan_payload",
            format!(
                "{}  %s = call float @air.fmax3.f32(float %x, float %y, float %z)\n",
                three(0xFFC0_1234, 0x7F80_0001, 0x7FA0_0000)
            ),
            Some(0x7FA0_0000),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.fmin.f32(float, float)\ndeclare float @air.fmax.f32(float, float)\ndeclare float @air.fast_fmin.f32(float, float)\ndeclare float @llvm.minnum.f32(float, float)\ndeclare float @llvm.maxnum.f32(float, float)\ndeclare float @air.fast_fmin3.f32(float, float, float)\ndeclare float @air.fmax3.f32(float, float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-float-min-max", &text) else {
        return;
    };
    let device_pick = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && [
                    crate::spv::GlslStd450Op::NMin as u32,
                    crate::spv::GlslStd450Op::NMax as u32,
                    crate::spv::GlslStd450Op::NClamp as u32,
                ]
                .iter()
                .any(|op| w.get(4) == Some(op))
        })
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let device = p.takes_driver_deviation(crate::error::DriverDeviation::MetalMinMax);
            assert_eq!(device_pick(&words) > 0, device, "{what}");
            assert_eq!(
                a.requirements
                    .properties
                    .contains(&crate::device::requirements::VkProperty::MetalMinMax),
                device,
                "{what}"
            );
            for note in [
                PrecisionNote::DeviceDenormals,
                PrecisionNote::DeviceNanEncoding,
            ] {
                assert!(!a.precision.contains(&note), "{what}: {:?}", a.precision);
            }
            // The interpreter reads a denormal operand of `NMin` by its value; Metal's reading, a
            // zero, is the device's, measured by the driver facts and the gate lane
            // (`runtime_lane::driver_deviations`).
            if device && kk.name.contains("denormal") {
                continue;
            }
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class H: Metal's `half` `fmod` answers `+0` for a zero remainder of a nonzero dividend and the
/// dividend itself for a zero one, with or without fast-math flags (`nan_fmod_zero`,
/// `nan_fmod_zero_fast`: `fmod(-2, 0.5)` is `+0`, `fmod(-0, 1)` is `-0`), on every profile; a
/// nonzero remainder keeps the dividend's sign, and a `float` zero remainder too (Metal's
/// `fmod(-2, 0.5)` on `float` is `-0`). The interpreter's `OpFRem` is C's `fmod`, the dividend's
/// sign, so the stored bits show the select.
#[test]
fn half_fmod_answers_metals_zero_remainder_on_every_profile() {
    let two = |a: u16, b: u16| format!("{}{}", oh("hx", a), oh("hy", b));
    let ks = [
        half_nan_kernel(
            "half_fmod_negative_dividend",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xC000, 0x3800)
            ),
            Some(0),
        ),
        half_nan_kernel(
            "half_fmod_negative_dividend_fast",
            format!(
                "{}  %s = call fast half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xFBFF, 0x7BFF)
            ),
            Some(0),
        ),
        half_nan_kernel(
            "half_fmod_of_negative_zero",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0x8000, 0x3C00)
            ),
            Some(0x8000),
        ),
        half_nan_kernel(
            "half_fmod_nonzero_remainder",
            format!(
                "{}  %s = call half @air.fmod.f16(half %hx, half %hy)\n",
                two(0xC200, 0x4000)
            ),
            Some(0xBC00),
        ),
        nan_kernel(
            "float_fmod_negative_dividend",
            format!(
                "{}{}  %s = call float @air.fmod.f32(float %x, float %y)\n",
                ob("x", 0xC000_0000),
                ob("y", 0x3F00_0000)
            ),
            Some(0x8000_0000),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @air.fmod.f16(half, half)\ndeclare float @air.fmod.f32(float, float)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-half-fmod", &text) else {
        return;
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class I: Metal's `mix(x, y, t)` on `float` and `half` is `fma(t, y, fma(-t, x, x))`, with and
/// without fast-math flags, on every profile: no `FMix` is emitted, and the interpreter answers
/// Metal's recorded words (`nan_functions`: `mix(inf, 1, 0)` is the canonical NaN, `mix(-0, -0, 0)`
/// is `+0`; `nan_half`: `mix(-0.3, 0.7, 0.2)` is `0xAE68`, where the blend gives `0xAE66`). The
/// driver's deviation `MetalCanonicalNans` is not taken, so a device that leaves the NaN to
/// `OpFmaKHR` (MoltenVK's, whose arithmetic answers Metal's) shows the select here too.
#[test]
fn mix_is_metals_fused_form_on_every_profile() {
    let three = |a: u32, b: u32, c: u32| format!("{}{}{}", ob("x", a), ob("y", b), ob("z", c));
    let halves = |a: u16, b: u16, c: u16| format!("{}{}{}", oh("hx", a), oh("hy", b), oh("hz", c));
    let ks = [
        nan_kernel(
            "mix_infinity_weighted_out",
            format!(
                "{}  %s = call float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x7F80_0000, 0x3F80_0000, 0)
            ),
            Some(0x7FC0_0000),
        ),
        nan_kernel(
            "mix_negative_zeros_fast",
            format!(
                "{}  %s = call fast float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x8000_0000, 0x8000_0000, 0)
            ),
            Some(0),
        ),
        nan_kernel(
            "mix_one_endpoint",
            format!(
                "{}  %s = call float @air.mix.f32(float %x, float %y, float %z)\n",
                three(0x501502F9, 0x3F80_0000, 0x3F80_0000)
            ),
            Some(0x3F80_0000),
        ),
        half_nan_kernel(
            "half_mix",
            format!(
                "{}  %s = call half @air.mix.f16(half %hx, half %hy, half %hz)\n",
                halves(0xB4CD, 0x399A, 0x3266)
            ),
            Some(0xAE68),
        ),
        half_nan_kernel(
            "half_mix_fast",
            format!(
                "{}  %s = call fast half @air.mix.f16(half %hx, half %hy, half %hz)\n",
                halves(0xB4CD, 0x399A, 0x3266)
            ),
            Some(0xAE68),
        ),
    ];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare float @air.mix.f32(float, float, float)\ndeclare half @air.mix.f16(half, half, half)\n"
        ),
    );
    let Some(bitcode) = assemble("mix-fused", &text) else {
        return;
    };
    let blend = |words: &[u32]| {
        count_ops(words, |w| {
            w[0] & 0xFFFF == crate::spv::Op::ExtInst as u32
                && w.get(4) == Some(&(crate::spv::GlslStd450Op::FMix as u32))
        })
    };
    for stem in PROFILES.iter().copied().chain(["lavapipe"]) {
        let p = profile_with_canonical_nans(stem, false);
        for kk in &ks {
            let what = format!("{} on {stem}", kk.name);
            let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            assert_eq!(blend(&words), 0, "{what}");
            let got = interpret(&bitcode, kk.name, &p);
            assert_eq!(Some(got), kk.want, "{what}: {got:#x}");
        }
    }
}

/// Class K: Metal's `half` functions answer the canonical `half` NaN `0x7E00` with and without
/// fast-math flags (`nan_half`, `nan_half_fast`): `llvm.fmuladd`, `ldexp` (unlike `float`'s, which
/// returns the operand), and `sqrt` take one select more where the device fact does not leave the
/// canonical class to the device. `round` returns a NaN operand itself, a signaling one kept, and
/// `+0` for a zero result; `asin`'s domain error takes the operand's sign (a select whatever the
/// fact); and `fract` with every flag answers `0x3BFF` for an infinity, where without flags it
/// answers `+0`. The interpreter's `OpQuantizeToF16` answers `0x7E00` itself, so the canonical
/// selects are counted against the same kernel compiled with the fact.
#[test]
fn half_functions_answer_metals_half_nan() {
    let hx = oh("hx", 0xFE01);
    let snan = oh("hx", 0x7C01);
    let inf = oh("hx", 0x7C00);
    let tenth = oh("hx", 0xB4CD);
    let ks = [
        half_nan_kernel(
            "half_fmuladd",
            format!(
                "{hx}  %s = call half @llvm.fmuladd.f16(half %hx, half 0xH3C00, half 0xH0000)\n"
            ),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_ldexp",
            format!("{hx}  %s = call half @air.ldexp.f16(half %hx, i32 2)\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_sqrt_fast",
            format!("{hx}  %s = call fast half @air.sqrt.f16(half %hx)\n"),
            Some(0x7E00),
        ),
        half_nan_kernel(
            "half_round_snan",
            format!("{snan}  %s = call half @air.round.f16(half %hx)\n"),
            Some(0x7C01),
        ),
        half_nan_kernel(
            "half_round_to_zero",
            format!("{tenth}  %s = call fast half @air.round.f16(half %hx)\n"),
            Some(0),
        ),
        half_nan_kernel(
            "half_fract_flagged",
            format!("{inf}  %s = call fast half @air.fract.f16(half %hx)\n"),
            Some(0x3BFF),
        ),
        half_nan_kernel(
            "half_fract",
            format!("{inf}  %s = call half @air.fract.f16(half %hx)\n"),
            Some(0),
        ),
        half_nan_kernel(
            "half_asin",
            format!("{hx}  %s = call half @air.asin.f16(half %hx)\n"),
            None,
        ),
    ];
    // The canonical selects the device fact leaves to the device, per kernel.
    let left = [1, 1, 1, 0, 0, 0, 0, 0];
    let text = module(&ks).replace(
        DECLS,
        &format!(
            "{DECLS}declare half @llvm.fmuladd.f16(half, half, half)\ndeclare half @air.ldexp.f16(half, i32)\ndeclare half @air.sqrt.f16(half)\ndeclare half @air.round.f16(half)\ndeclare half @air.fract.f16(half)\ndeclare half @air.asin.f16(half)\n"
        ),
    );
    let Some(bitcode) = assemble("nan-half-functions", &text) else {
        return;
    };
    for stem in ["lavapipe", "moltenvk-m3"] {
        for (kk, n) in ks.iter().zip(left) {
            let mut tests = Vec::new();
            for fact in [false, true] {
                let p = profile_with_canonical_nans(stem, fact);
                let what = format!("{} on {stem}, MetalCanonicalNans {fact}", kk.name);
                let a = compile(&bitcode, kk.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                let (_, words) =
                    link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                let (is_nan, bits) = nan_tests(&words);
                tests.push(is_nan + bits);
                if let Some(want) = kk.want {
                    let got = interpret(&bitcode, kk.name, &p);
                    assert_eq!(got, want, "{what}: {got:#x}");
                }
            }
            assert_eq!(tests[0], tests[1] + n, "{} on {stem}: {tests:?}", kk.name);
            if kk.name == "half_asin" {
                assert!(
                    tests[1] > 0,
                    "half_asin on {stem}: its domain sign is selected"
                );
            }
        }
    }
}
