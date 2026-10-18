//! Metal's math where `GLSL.std.450` is a different function, end to end: a call's `air` symbol
//! variant and fast-math flags select what it computes. On every committed M1 profile each kernel
//! compiles, links, and passes L3, and its module holds the instructions of the function Metal
//! computes:
//!
//! - `air.fast_tanh` on `float` is the MSL fast-math formula `(t - 1) / (t + 1)`,
//!   `t = exp2(x * 2 log2 e)` of Metal's `exp2` (no `Tanh`), every step of it `NoContraction`; the
//!   precise `air.tanh` is Metal's `tanh`, and a `half` `fast_tanh` Metal's `half` `tanh`, `(t - 1)
//!   rcp(t + 1)` of the same `t` with Metal's reciprocal (no `Tanh`). Metal's `exp2` and `tanh` are
//!   the device's `Exp2` and `Tanh` where the profile's `float` functions are Metal's
//!   (`FloatFunctions::Metal`) and the translation's routines elsewhere, and so is its reciprocal
//!   in a fast-math entry, the device's `1 / y` (`OpFDiv`); the checks below leave the routines out
//!   of the kernel's own instructions, with the precise division they divide through
//!   (`without_math_routines`);
//! - `air.sign` is Metal's `sign` with and without fast math (no `FSign`, which keeps the sign
//!   of a zero where Metal answers `+0`), and both answer `-1` for `-3` on the interpreter;
//! - the precise `air.sinpi` calls Metal's routine on every profile, so the entry's own
//!   instructions hold no `Sin` and no `RoundEven`; the fast `air.fast_sinpi` reduces its argument
//!   (`RoundEven`) and takes the sine of the residue from the device's `Sin` where the profile's
//!   `float` functions are Metal's (`FloatFunctions::Metal`), and calls the routine elsewhere.
//!
//! The values of these functions are the unit tests' (`lower::arith::tests`); no GPU is used.
//!
//! The sRGB conversions of `unorm4x8_srgb` run end to end on the interpreter: A1's committed
//! fixtures `a1_srgb_decode_all_bytes` (every byte in every channel, `float` and `half` results)
//! and `a1_srgb_encode_sweep` (every multiple of 2⁻¹⁴ in `[0, 1.0078]` and every `half` up to 2.25)
//! compile on every profile above, declare Metal's twelve-bit table once per module, read no `Pow`,
//! note no device transcendentals, and write the table's word for every thread.
//!
//! `llvm-as` assembles the modules; without it the tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::profile::FloatFunctions;
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::interface::PushLoc;
use crate::pipeline::sym::{Axis, PushField, ResourceKind, ResourceSlot, SpecKey, Stage};
use crate::pipeline::testing::{
    compile, engine_dir, exact_maths, link_and_validate, llvm_assemble, meter, profile,
    tiny_check_scales, typed_module, without_math_routines, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::StorageClass;

const PROFILES: [&str; 3] = ["moltenvk-m3", "lunarg-desktop-baseline-2023", "vk12-floor"];

/// `GLSL.std.450` instruction numbers.
const FSIGN: u32 = 6;
const SIN: u32 = 13;
const TANH: u32 = 21;
const EXP2: u32 = 29;
const ROUND_EVEN: u32 = 2;
const POW: u32 = 26;

/// One kernel: its body defines `%r` (`float`), which it stores.
struct Kernel {
    name: &'static str,
    body: &'static str,
    /// `GLSL.std.450` instructions the module must and must not contain.
    has: &'static [u32],
    lacks: &'static [u32],
    /// `GLSL.std.450` instructions the module contains exactly where the profile's `float`
    /// functions are Metal's.
    metal_functions_only: &'static [u32],
    /// Whether the module divides (`OpFDiv`) by the fast division's routine.
    divides: bool,
    /// Whether the module divides (`OpFDiv`) exactly where the profile's `float` functions are
    /// Metal's: the device's reciprocal in a fast-math entry.
    metal_divides: bool,
    /// The stored word on the interpreter, when it models the kernel.
    want: Option<u32>,
}

const KERNELS: [Kernel; 8] = [
    Kernel {
        name: "fast_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.fast_tanh.f32(float %a)\n",
        has: &[],
        lacks: &[TANH],
        metal_functions_only: &[EXP2],
        divides: true,
        metal_divides: false,
        want: None,
    },
    // RC3 cases e72f550f and 2d5f3949's GELU tail at x = -8: 0.5 * x * (1 + tanh(...)).
    Kernel {
        name: FAST_TANH_IN_GELU,
        body: "  %a = fadd fast float 0xC038A66660000000, -0.0\n  %x = fadd fast float 0xC020000000000000, -0.0\n  %t = call fast float @air.fast_tanh.f32(float %a)\n  %s = fadd fast float %t, 1.000000e+00\n  %h = fmul fast float %x, 5.000000e-01\n  %r = fmul fast float %h, %s\n",
        has: &[],
        lacks: &[TANH],
        metal_functions_only: &[EXP2],
        divides: true,
        metal_divides: false,
        want: None,
    },
    Kernel {
        name: "precise_tanh_float",
        body: "  %a = fadd fast float 0x4046000000000000, -0.0\n  %r = call fast float @air.tanh.f32(float %a)\n",
        has: &[],
        lacks: &[EXP2],
        metal_functions_only: &[TANH],
        divides: false,
        metal_divides: false,
        want: None,
    },
    Kernel {
        name: "fast_tanh_half",
        body: "  %a = fadd fast float 0x4000000000000000, -0.0\n  %h = fptrunc float %a to half\n  %t = call fast half @air.fast_tanh.f16(half %h)\n  %r = fpext half %t to float\n",
        has: &[],
        lacks: &[TANH],
        metal_functions_only: &[EXP2],
        divides: false,
        metal_divides: true,
        want: None,
    },
    Kernel {
        name: "fast_sign",
        body: "  %a = fadd fast float 0xC008000000000000, -0.0\n  %r = call fast float @air.sign.f32(float %a)\n",
        has: &[],
        lacks: &[FSIGN],
        metal_functions_only: &[],
        divides: false,
        metal_divides: false,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sign",
        body: "  %a = fadd float 0xC008000000000000, -0.0\n  %r = call float @air.sign.f32(float %a)\n",
        has: &[],
        lacks: &[FSIGN],
        metal_functions_only: &[],
        divides: false,
        metal_divides: false,
        want: Some(0xBF80_0000),
    },
    Kernel {
        name: "precise_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.sinpi.f32(float %a)\n",
        has: &[],
        lacks: &[SIN, ROUND_EVEN],
        metal_functions_only: &[],
        divides: false,
        metal_divides: false,
        want: None,
    },
    Kernel {
        name: "fast_sinpi",
        body: "  %a = fadd fast float 0x3FF0000000000000, -0.0\n  %r = call fast float @air.fast_sinpi.f32(float %a)\n",
        has: &[],
        lacks: &[],
        metal_functions_only: &[SIN, ROUND_EVEN],
        divides: false,
        metal_divides: false,
        want: None,
    },
];

/// The kernel whose `fast::tanh` feeds the caller's `1 + tanh` ([`fast_tanh_is_one_operation`]).
const FAST_TANH_IN_GELU: &str = "fast_tanh_in_gelu";

/// The `GLSL.std.450 Fma` instruction number.
const FMA: u32 = 50;

fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in &KERNELS {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n{}  store float %r, ptr addrspace(1) %0, align 4\n  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    s.push_str("declare float @air.fast_tanh.f32(float)\ndeclare float @air.tanh.f32(float)\ndeclare half @air.fast_tanh.f16(half)\ndeclare float @air.sign.f32(float)\ndeclare float @air.sinpi.f32(float)\ndeclare float @air.fast_sinpi.f32(float)\n");
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-metal-math");
    let dir = scratch.path();
    let ll = dir.join("metal_math.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the module: {e}"),
    }
}

/// The `GLSL.std.450` instruction numbers of every `OpExtInst` in `words`, and whether an
/// `OpFDiv` occurs.
fn ext_insts(words: &[u32]) -> (Vec<u32>, bool) {
    let (mut ext, mut div) = (Vec::new(), false);
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::ExtInst as u32 {
            ext.push(words[i + 4]);
        }
        div |= op == crate::spv::Op::FDiv as u32;
        i += len;
    }
    (ext, div)
}

/// The stored word of `entry` on the interpreter.
fn interpret(bitcode: &[u8], entry: &str, stem: &str) -> u32 {
    let p = profile(stem, exact_maths()).expect("profile");
    let mut m = meter();
    let decoded = air::decode(bitcode, &mut m).expect("decode");
    let req = SpecRequest {
        call_stack_depth: 1,
        entry: entry.to_owned(),
        ..SpecRequest::default()
    };
    let spec = air::specialize(decoded, &req, &mut m).expect("specialize");
    let memory = analyze_memory(&spec, &p, &mut m).expect("memory");
    let plan = select(&spec, &memory, &p, &LowerInputs::new(), &mut m).expect("select");
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
fn metal_math_variants_select_metals_functions() {
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for k in &KERNELS {
            let what = format!("{} on {stem}", k.name);
            let a = compile(&bitcode, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, div) = ext_insts(&without_math_routines(&words));
            for i in k.has {
                assert!(
                    ext.contains(i),
                    "{what}: GLSL.std.450 {i} missing ({ext:?})"
                );
            }
            for i in k.lacks {
                assert!(
                    !ext.contains(i),
                    "{what}: GLSL.std.450 {i} present ({ext:?})"
                );
            }
            let metal = p.fields().float_functions == FloatFunctions::Metal;
            for i in k.metal_functions_only {
                assert_eq!(
                    ext.contains(i),
                    metal,
                    "{what}: GLSL.std.450 {i} with float functions Metal's = {metal} ({ext:?})"
                );
            }
            assert_eq!(
                div,
                k.divides || (k.metal_divides && metal),
                "{what}: OpFDiv"
            );
            // The fast-math entry's `fast_tanh` notes the routines it calls, except where every
            // step is Metal's own function on the device (`FloatFunctions::Metal`), which notes
            // nothing in such an entry.
            if k.divides {
                assert_eq!(
                    a.precision.contains(&PrecisionNote::DeviceArithmetic)
                        && a.precision.contains(&PrecisionNote::DeviceTranscendentals),
                    !metal,
                    "{what}: {:?}",
                    a.precision
                );
            }
            if let Some(want) = k.want {
                let got = interpret(&bitcode, k.name, stem);
                assert_eq!(got, want, "{what}: {got:#x} != {want:#x}");
            }
        }
    }
}

/// Each float arithmetic result of `words` (`OpFAdd`, `OpFSub`, `OpFMul`, `OpFDiv`, and
/// `GLSL.std.450 Fma`) but those of `apart`, as its opcode and whether it is decorated
/// `NoContraction`.
fn float_arithmetic(words: &[u32], apart: &[u32]) -> Vec<(u32, bool)> {
    let (mut results, mut exact) = (Vec::new(), Vec::new());
    let arith = [
        crate::spv::Op::FAdd,
        crate::spv::Op::FSub,
        crate::spv::Op::FMul,
        crate::spv::Op::FDiv,
    ]
    .map(|op| op as u32);
    let mut i = 5;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == crate::spv::Op::Decorate as u32
            && words[i + 2] == crate::spv::Decoration::NoContraction as u32
        {
            exact.push(words[i + 1]);
        }
        if arith.contains(&op) || (op == crate::spv::Op::ExtInst as u32 && words[i + 4] == FMA) {
            results.push((op, words[i + 2]));
        }
        i += len;
    }
    results
        .into_iter()
        .filter(|(_, id)| !apart.contains(id))
        .map(|(op, id)| (op, exact.contains(&id)))
        .collect()
}

/// `air.fast_tanh` is one operation to Metal, so the device may not fuse or regroup any of its
/// steps with the caller's arithmetic. Every step is `NoContraction` whatever the call's flags: `x
/// · 2 log2 e`, the quotient's product and the same product `2^24` larger that decides its flush
/// (each in the profile's form: `GLSL.std.450 Fma(a, b, -0.0)` where its `float` functions are
/// Metal's, an `OpFMul` elsewhere), `t - 1`, `t + 1`, and the reciprocal's estimate. The caller's
/// `fast` arithmetic stays undecorated. Without the decorations, Mesa's `lower_flrp` rule turned
/// GELU's `1 + tanh` into `t` on RADV and x86 lavapipe (RC3 cases e72f550f and 2d5f3949,
/// `gelu(-8)`). Under `FloatMathsPolicy::Exact` each of the caller's two products is checked; a
/// fused check (`TinyCheck::Fused`) scales an operand by its own multiply, `NoContraction`,
/// counted apart.
#[test]
fn fast_tanh_is_one_operation() {
    use crate::lower::arith::denormal::{tiny_check, TinyCheck};
    let Some(bitcode) = assemble() else {
        return;
    };
    for stem in PROFILES.into_iter().chain(["amd-rdna3-radv", "lavapipe"]) {
        let p = profile(stem, exact_maths()).expect("profile");
        let what = format!("{FAST_TANH_IN_GELU} on {stem}");
        let a =
            compile(&bitcode, FAST_TANH_IN_GELU, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let own = without_math_routines(&words);
        let scales = tiny_check_scales(&own);
        let fused = tiny_check(&p) == Some(TinyCheck::Fused);
        assert_eq!(
            scales.len(),
            2 * usize::from(fused),
            "{what}: the checks' scaling"
        );
        let ops = float_arithmetic(&own, &scales);
        let exact_muls = |ops: &[(u32, bool)]| {
            let mul = (crate::spv::Op::FMul as u32, true);
            ops.iter().filter(|&&o| o == mul).count()
        };
        assert_eq!(
            exact_muls(&float_arithmetic(&own, &[])),
            exact_muls(&ops) + scales.len(),
            "{what}: the checks' scaling, NoContraction"
        );
        let count = |op: crate::spv::Op, exact: bool| {
            ops.iter()
                .filter(|(o, e)| *o == op as u32 && *e == exact)
                .count()
        };
        let fma = |exact: bool| {
            ops.iter()
                .filter(|(o, e)| *o == crate::spv::Op::ExtInst as u32 && *e == exact)
                .count()
        };
        // The tanh's products are in the profile's form, and none in the other.
        let metal = p.fields().float_functions == FloatFunctions::Metal;
        let (products, other) = if metal {
            (fma(true), count(crate::spv::Op::FMul, true))
        } else {
            (count(crate::spv::Op::FMul, true), fma(true))
        };
        assert_eq!(
            (
                products,
                count(crate::spv::Op::FSub, true),
                count(crate::spv::Op::FAdd, true),
                count(crate::spv::Op::FDiv, true)
            ),
            (4, 1, 1, 1),
            "{what}: the tanh's steps, NoContraction: {ops:?}"
        );
        assert_eq!(
            (
                fma(false),
                count(crate::spv::Op::FSub, false),
                count(crate::spv::Op::FDiv, false),
                count(crate::spv::Op::FMul, false),
                other
            ),
            (0, 0, 0, 2, 0),
            "{what}: only the caller's multiplies are left free: {ops:?}"
        );
        assert!(
            count(crate::spv::Op::FAdd, false) >= 1,
            "{what}: the caller's 1 + tanh stays free: {ops:?}"
        );
    }
}

/// The byte Metal's twelve-bit table encodes `x` to: the index `i = round(x * 4095)` of the exact
/// product, then the MSL transfer with the exponent `0.41666` (which reproduces every one of A1's
/// 4096 observed indices; `1/2.4` misses four), rounded to nearest even.
fn srgb_encode(x: f64) -> u32 {
    let i = (x.clamp(0.0, 1.0) * 4095.0).round_ties_even();
    let c = i / 4095.0;
    let v = if c < 0.0031308 {
        12.92 * c
    } else {
        1.055 * c.powf(0.41666) - 0.055
    };
    (255.0 * v).round_ties_even() as u32
}

/// A linear (alpha) channel: the exact product `x * 255` rounded to nearest even.
fn unorm_encode(x: f64) -> u32 {
    (x.clamp(0.0, 1.0) * 255.0).round_ties_even() as u32
}

/// Metal's decode of color byte `b`: the table step `k = round(decode(b / 255) * 4095)` of the MSL
/// transfer, answered as `k / 4095` correctly rounded.
fn srgb_decode(b: u32) -> f32 {
    let c = f64::from(b) / 255.0;
    let v = if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    };
    ((v * 4095.0).round_ties_even() / 4095.0) as f32
}

/// The `half` nearest `x` (ties to even), for `x` in `[0, 65504]`.
fn half_bits(x: f32) -> u32 {
    let v = f64::from(x);
    assert!((0.0..=65504.0).contains(&v), "{x}");
    if v < 2f64.powi(-14) {
        // Subnormal: multiples of 2^-24 (1024 is the least normal, 0x400).
        return (v * 2f64.powi(24)).round_ties_even() as u32;
    }
    let mut e = ((x.to_bits() >> 23) & 0xff) as i32 - 127;
    let mut m = (v / 2f64.powi(e - 10)).round_ties_even() as u32;
    if m == 2048 {
        e += 1;
        m = 1024;
    }
    (((e + 15) as u32) << 10) | (m - 1024)
}

/// The value of `half` bit pattern `h` (finite, non-negative).
fn half_value(h: u32) -> f64 {
    let (e, m) = ((h >> 10) & 0x1f, h & 0x3ff);
    if e == 0 {
        f64::from(m) * 2f64.powi(-24)
    } else {
        f64::from(1024 + m) * 2f64.powi(e as i32 - 25)
    }
}

/// The words fixture `entry` writes to `[[buffer(0)]]` over `threads` threads (32-lane
/// threadgroups), `per_thread` words each, on the interpreter.
fn run_fixture(bc: &[u8], entry: &str, stem: &str, threads: u32, per_thread: u32) -> Vec<u32> {
    let p = profile(stem, exact_maths()).expect("profile");
    let (plan, m) = typed_module(bc, entry, &p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    // Zero push bytes: a zero bind residual and a dispatch region based at the origin.
    for f in &plan.push.fields {
        assert!(
            matches!(
                (f.field, f.loc),
                (
                    PushField::SlotBindOffset(_) | PushField::DispatchRegion,
                    PushLoc::Push { .. }
                )
            ),
            "{entry}: {:?}",
            f.field
        );
    }
    // One `Private` variable with an initializer: the table, declared once however many
    // conversions read it (float and half).
    let tables = m
        .globals()
        .iter()
        .filter(|g| g.class == StorageClass::Private && g.init.is_some())
        .count();
    assert_eq!(tables, 1, "{entry} on {stem}: the sRGB table");
    let mut res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes: vec![0xab; 4 * (threads * per_thread) as usize],
        }],
        push: vec![0; plan.push.push_range.map_or(0, |(_, len)| len as usize)],
        spec: Default::default(),
    };
    for (axis, n) in [(Axis::X, 32u32), (Axis::Y, 1), (Axis::Z, 1)] {
        res.spec.insert(SpecKey::LocalSize(axis), n.into());
    }
    res.spec.insert(SpecKey::FlatLocalSize, 32u32.into());
    let config = Config {
        workgroups: [threads / 32, 1, 1],
        ..Config::default()
    };
    let out = run(&m, &config, res).unwrap_or_else(|e| panic!("{entry} on {stem}: {e}"));
    out.buffers[0]
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// End to end: the sRGB fixtures through select, emit, and link on every profile above: L3 passes,
/// the module reads no `Pow` and notes no device transcendentals, and on the interpreter every
/// thread writes Metal's table words.
///
/// - `a1_srgb_decode_all_bytes`: thread `t` unpacks the word of bytes `t`, `255 - t`,
///   `t ^ 0x5a`, `(7t + 3) & 255` and writes the four `float` results, then the four `half`
///   results (each rounded from the `float` quotient) as two words. Metal's own steps of bytes 1,
///   137, and 255 (`0x39800801`, `0x3E800801`, `1.0`) are among them.
/// - `a1_srgb_encode_sweep`: thread `t` packs `3t`, `3t + 1`, `3t + 2` times 2⁻¹⁴ (alpha `3t`
///   times 2⁻¹⁴, linear), then the `half` patterns `3t`, `3t + 1`, `3t + 2` (alpha `3t`).
#[test]
fn srgb_fixtures_answer_with_metals_tables_on_the_interpreter() {
    const DECODE: u32 = 256;
    const ENCODE: u32 = 5504;
    let assembled = |stem: &str| match llvm_assemble(&engine_dir().join(format!("{stem}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {stem}: {e}"),
    };
    let (Some(decode_bc), Some(encode_bc)) = (
        assembled("a1_srgb_decode_all_bytes"),
        assembled("a1_srgb_encode_sweep"),
    ) else {
        return;
    };

    let mut decoded = Vec::new();
    for t in 0..DECODE {
        let bytes = [t, 255 - t, t ^ 0x5a, (7 * t + 3) & 255];
        let lanes = [
            srgb_decode(bytes[0]),
            srgb_decode(bytes[1]),
            srgb_decode(bytes[2]),
            bytes[3] as f32 / 255.0,
        ];
        decoded.extend(lanes.iter().map(|x| x.to_bits()));
        decoded.push(half_bits(lanes[0]) | (half_bits(lanes[1]) << 16));
        decoded.push(half_bits(lanes[2]) | (half_bits(lanes[3]) << 16));
    }
    // Metal's steps (authored case `srgb-transfer-and-its-twelve-bit-table`): bytes 1, 137, and
    // 255 in the red channel of threads 1, 137, and 255.
    for (t, want) in [
        (1usize, 0x3980_0801u32),
        (137, 0x3E80_0801),
        (255, 0x3F80_0000),
    ] {
        assert_eq!(decoded[6 * t], want, "byte {t}");
    }
    let mut encoded = Vec::new();
    for t in 0..ENCODE {
        let k = 3 * t;
        let x = |j: u32| f64::from(k + j) * 2f64.powi(-14);
        encoded.push(
            srgb_encode(x(0))
                | (srgb_encode(x(1)) << 8)
                | (srgb_encode(x(2)) << 16)
                | (unorm_encode(x(0)) << 24),
        );
        let h = |j: u32| half_value(k + j);
        encoded.push(
            srgb_encode(h(0))
                | (srgb_encode(h(1)) << 8)
                | (srgb_encode(h(2)) << 16)
                | (unorm_encode(h(0)) << 24),
        );
    }

    for stem in PROFILES {
        let p = profile(stem, exact_maths()).expect("profile");
        for (bc, entry, threads, per_thread, want) in [
            (&decode_bc, "a1_srgb_decode_all_bytes", DECODE, 6, &decoded),
            (&encode_bc, "a1_srgb_encode_sweep", ENCODE, 2, &encoded),
        ] {
            let what = format!("{entry} on {stem}");
            let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let (_, words) =
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
            let (ext, _) = ext_insts(&words);
            assert!(!ext.contains(&POW), "{what}: reads Pow ({ext:?})");
            assert!(
                !a.precision.contains(&PrecisionNote::DeviceTranscendentals),
                "{what}: {:?}",
                a.precision
            );
            let got = run_fixture(bc, entry, stem, threads, per_thread);
            for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
                assert_eq!(
                    g,
                    w,
                    "{what}: thread {} word {}: {g:#010x} != {w:#010x}",
                    i as u32 / per_thread,
                    i as u32 % per_thread
                );
            }
            assert_eq!(got.len(), want.len(), "{what}");
        }
    }
}
