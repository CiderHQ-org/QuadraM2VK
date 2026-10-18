//! The device's forms of Metal's maths: on a device whose `float` functions are not Metal's own,
//! three of the policy's choices each take the device's instructions where Metal's answer would be
//! computed, each recording its kinds, and each set on or off alone: the maths functions, divisions
//! and fused multiply-adds (`FloatMathsPolicy`: `NativeTranscendentals`, `NativeDivision`,
//! `NativeFma`), the NaN and minimum and maximum (`NanMinMaxPolicy`: `NativeNanEncoding`,
//! `NativeMinMax`), and the rounding order of `dot`, `mix` and `fast::sqrt` (`RoundingOrderPolicy`:
//! `NativeDot`, `NativeMix`, `NativeTranscendentals`). Off, each takes Metal's answer: the
//! translation's routines, its selects of Metal's NaN and operand, and Metal's fused forms.
//!
//! The kernels read three `float`s from buffer 0 and store one result after them: a precise
//! `sin`, `exp`, `atan2` and `pow`, a precise and an `arcp` division, a `half` `fma`, an addition,
//! an `fmin`, a `saturate`, a `fast::sqrt`, a `dot` and a `float` and a `half` `mix`. On lavapipe
//! each records its kind exactly where its own choice is on, whatever the others are, and is
//! smaller there than with it off; the interpreter's answers stay within the bound the kind's
//! docs state when on, and are Metal's when off. `llvm-as` assembles the module; without it the
//! tests skip visibly.

use crate::air::{self, SpecRequest};
use crate::device::profile::{
    DeviceProfile, FloatFunctions, MinMaxRoute, NanEncoding, RoundingOrderRoute,
};
use crate::device::{DevicePolicy, FloatMathsPolicy, NanMinMaxPolicy, RoundingOrderPolicy};
use crate::lower::emit::emit_module;
use crate::lower::select::{select, LowerInputs};
use crate::memory::static_root::analyze_memory;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, meter, profile, AssembleError, Scratch,
};
use crate::pipeline::PrecisionNote;
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::{committed_profiles_with, is_moltenvk_capture};

const OP_FUNCTION: u32 = 54;

/// One of the policy's choices between the device's form and Metal's that native device maths
/// was split into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Choice {
    /// The maths functions, divisions and fused multiply-adds (`FloatMathsPolicy`).
    Maths,
    /// The NaN and minimum and maximum (`NanMinMaxPolicy`).
    NanMinMax,
    /// The rounding order of `dot`, `mix` and `fast::sqrt` (`RoundingOrderPolicy`).
    RoundingOrder,
}

impl Choice {
    const ALL: [Choice; 3] = [Choice::Maths, Choice::NanMinMax, Choice::RoundingOrder];

    /// `policy` with this choice on (the device's form) or off (Metal's).
    fn set(self, policy: DevicePolicy, on: bool) -> DevicePolicy {
        match self {
            Choice::Maths => DevicePolicy {
                float_maths: match on {
                    true => FloatMathsPolicy::Native,
                    false => FloatMathsPolicy::Exact,
                },
                ..policy
            },
            Choice::NanMinMax => DevicePolicy {
                nan_and_min_max: match on {
                    true => NanMinMaxPolicy::Native,
                    false => NanMinMaxPolicy::Exact,
                },
                ..policy
            },
            Choice::RoundingOrder => DevicePolicy {
                rounding_order: match on {
                    true => RoundingOrderPolicy::Native,
                    false => RoundingOrderPolicy::Exact,
                },
                ..policy
            },
        }
    }

    /// The default policy with every choice on or off.
    fn every(on: bool) -> DevicePolicy {
        Choice::ALL
            .iter()
            .fold(DevicePolicy::default(), |p, c| c.set(p, on))
    }

    /// Whether profile `p` takes this choice's device forms.
    fn taken(self, p: &DeviceProfile) -> bool {
        let f = p.fields();
        match self {
            Choice::Maths => f.float_functions == FloatFunctions::Native,
            Choice::NanMinMax => {
                f.nan_encoding == NanEncoding::Native && f.min_max == MinMaxRoute::Native
            }
            Choice::RoundingOrder => f.rounding_order == RoundingOrderRoute::Native,
        }
    }
}

/// The kinds of the device's forms of Metal's maths.
const NATIVES: [PrecisionNote; 7] = [
    PrecisionNote::NativeTranscendentals,
    PrecisionNote::NativeDivision,
    PrecisionNote::NativeFma,
    PrecisionNote::NativeNanEncoding,
    PrecisionNote::NativeMinMax,
    PrecisionNote::NativeDot,
    PrecisionNote::NativeMix,
];

/// A kernel: its name, the body defining `%r` from `%a`, `%b` and `%c`, the note its device form
/// records, and the choice that takes it.
struct Kernel {
    name: &'static str,
    body: &'static str,
    note: PrecisionNote,
    choice: Choice,
}

const KERNELS: [Kernel; 14] = [
    Kernel {
        name: "native_sin",
        body: "  %r = tail call float @air.sin.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_exp",
        body: "  %r = tail call float @air.exp.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_atan2",
        body: "  %r = tail call float @air.atan2.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeTranscendentals,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_pow",
        body: "  %r = tail call float @air.pow.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeTranscendentals,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_fdiv",
        body: "  %r = fdiv float %a, %b\n",
        note: PrecisionNote::NativeDivision,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_fdiv_arcp",
        body: "  %r = fdiv arcp float %a, %b\n",
        note: PrecisionNote::NativeDivision,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_half_fma",
        body: "  %ha = fptrunc float %a to half\n  %hb = fptrunc float %b to half\n  %hc = fptrunc float %c to half\n  %hr = tail call half @air.fma.f16(half %ha, half %hb, half %hc)\n  %r = fpext half %hr to float\n",
        note: PrecisionNote::NativeFma,
        choice: Choice::Maths,
    },
    Kernel {
        name: "native_fadd",
        body: "  %r = fadd float %a, %b\n",
        note: PrecisionNote::NativeNanEncoding,
        choice: Choice::NanMinMax,
    },
    Kernel {
        name: "native_fmin",
        body: "  %r = tail call float @air.fmin.f32(float %a, float %b)\n",
        note: PrecisionNote::NativeMinMax,
        choice: Choice::NanMinMax,
    },
    Kernel {
        name: "native_fast_sqrt",
        body: "  %r = tail call fast float @air.fast_sqrt.f32(float %a)\n",
        note: PrecisionNote::NativeTranscendentals,
        choice: Choice::RoundingOrder,
    },
    Kernel {
        name: "native_dot",
        body: "  %u0 = insertelement <2 x float> undef, float %a, i32 0\n  %u = insertelement <2 x float> %u0, float %b, i32 1\n  %v0 = insertelement <2 x float> undef, float %c, i32 0\n  %v = insertelement <2 x float> %v0, float %a, i32 1\n  %r = tail call float @air.dot.v2f32(<2 x float> %u, <2 x float> %v)\n",
        note: PrecisionNote::NativeDot,
        choice: Choice::RoundingOrder,
    },
    Kernel {
        name: "native_mix",
        body: "  %r = tail call float @air.mix.f32(float %a, float %b, float %c)\n",
        note: PrecisionNote::NativeMix,
        choice: Choice::RoundingOrder,
    },
    Kernel {
        name: "native_saturate",
        body: "  %r = tail call float @air.saturate.f32(float %a)\n",
        note: PrecisionNote::NativeMinMax,
        choice: Choice::NanMinMax,
    },
    Kernel {
        name: "native_half_mix",
        body: "  %ha = fptrunc float %a to half\n  %hb = fptrunc float %b to half\n  %hc = fptrunc float %c to half\n  %hr = tail call half @air.mix.f16(half %ha, half %hb, half %hc)\n  %r = fpext half %hr to float\n",
        note: PrecisionNote::NativeMix,
        choice: Choice::RoundingOrder,
    },
];

/// The module of every kernel.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    for k in &KERNELS {
        s.push_str(&format!(
            "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0) {{\nentry:\n  %p1 = getelementptr inbounds float, ptr addrspace(1) %0, i64 1\n  %p2 = getelementptr inbounds float, ptr addrspace(1) %0, i64 2\n  %p3 = getelementptr inbounds float, ptr addrspace(1) %0, i64 3\n  %a = load float, ptr addrspace(1) %0, align 4\n  %b = load float, ptr addrspace(1) %p1, align 4\n  %c = load float, ptr addrspace(1) %p2, align 4\n{}  store float %r, ptr addrspace(1) %p3, align 4\n  ret void\n}}\n\n",
            k.name, k.body
        ));
    }
    s.push_str("declare float @air.sin.f32(float)\ndeclare float @air.exp.f32(float)\ndeclare float @air.atan2.f32(float, float)\ndeclare float @air.pow.f32(float, float)\ndeclare half @air.fma.f16(half, half, half)\ndeclare float @air.fmin.f32(float, float)\ndeclare float @air.dot.v2f32(<2 x float>, <2 x float>)\ndeclare float @air.fast_sqrt.f32(float)\ndeclare float @air.mix.f32(float, float, float)\ndeclare half @air.mix.f16(half, half, half)\ndeclare float @air.saturate.f32(float)\n");
    let refs: Vec<String> = (0..KERNELS.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!llvm.module.flags = !{{!0}}\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str("!0 = !{i32 7, !\"air.max_device_buffers\", i32 31}\n!10 = !{}\n!11 = !{!12}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", i32 0, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"float\", !\"air.arg_name\", !\"io\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, i32 0, i32 0}\n");
    for (i, k) in KERNELS.iter().enumerate() {
        s.push_str(&format!("!{} = !{{ptr @{}, !10, !11}}\n", 20 + i, k.name));
    }
    s
}

/// The module's bitcode, or `None` (skipped visibly) without `llvm-as`.
fn bitcode() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-native-maths");
    let dir = scratch.path();
    let ll = dir.join("native.ll");
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

/// `entry` on `p` on the interpreter with inputs `x`: the stored result.
fn interpret(bitcode: &[u8], entry: &str, p: &DeviceProfile, x: [f32; 3]) -> f32 {
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
    let bytes = [x[0], x[1], x[2], 0.0]
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let res = Resources {
        buffers: vec![Buffer {
            slot: Some(ResourceSlot {
                stage: Stage::Kernel,
                kind: ResourceKind::Buffer,
                index: 0,
            }),
            address: None,
            bytes,
        }],
        push: vec![0; plan.push.push_range.map_or(4, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let out = run(&module, &Config::default(), res)
        .unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    let b = &out.buffers[0];
    f32::from_le_bytes([b[12], b[13], b[14], b[15]])
}

/// The functions a module defines, its entry included.
fn functions(words: &[u32]) -> usize {
    let mut i = 5;
    let mut n = 0;
    while let Some(&w) = words.get(i) {
        n += usize::from(w & 0xFFFF == OP_FUNCTION);
        i += (w >> 16).max(1) as usize;
    }
    n
}

/// The units in the last place of `want` (an `f32`) that `got` lies from it.
fn ulps(got: f32, want: f64) -> f64 {
    let w = want as f32;
    let ulp = f64::from(f32::from_bits(w.abs().to_bits() + 1) - w.abs());
    (f64::from(got) - want).abs() / ulp
}

/// On every committed profile but the MoltenVK captures, whose functions, NaN and minimum and
/// maximum are Metal's own, each choice takes its device forms exactly where it is on, and moves
/// the fingerprint; on the MoltenVK captures neither setting takes them, and only the NaN choice
/// moves the fingerprint: a texel's NaN, the image's decode, follows it on every device
/// (`TexelNanRoute`).
#[test]
fn each_choice_takes_the_device_s_forms_where_it_is_on() {
    for choice in Choice::ALL {
        let base = Choice::every(true);
        let on = committed_profiles_with(choice.set(base, true)).expect("profiles");
        let off = committed_profiles_with(choice.set(base, false)).expect("profiles");
        assert!(on.len() >= 10, "{} profiles", on.len());
        for ((stem, p), (_, q)) in on.iter().zip(&off) {
            let metal = is_moltenvk_capture(stem);
            assert_eq!(choice.taken(p), !metal, "{stem} {choice:?} on");
            assert!(!choice.taken(q), "{stem} {choice:?} off");
            let moved = p.fingerprint() != q.fingerprint();
            assert_eq!(
                moved,
                !metal || choice == Choice::NanMinMax,
                "{stem} {choice:?}"
            );
        }
    }
}

/// On lavapipe each kernel records its kind exactly where its own choice is on, under every setting
/// of the others (all of them on and all of them off); where it is on the module is smaller than
/// where it is off, passes L3 either way, and records no kind of a choice that is off. With every
/// choice on, each module is its entry alone (no routine).
#[test]
fn each_choice_records_its_kinds_only_when_on() {
    let Some(bc) = bitcode() else {
        return;
    };
    for others in [true, false] {
        let base = Choice::every(others);
        for k in &KERNELS {
            let words = |on: bool| {
                let policy = k.choice.set(base, on);
                let p = profile("lavapipe", policy).expect("profile");
                let what = format!("{} {:?} on {on}, others {others}", k.name, k.choice);
                let a = compile(&bc, k.name, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
                link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
                assert_eq!(
                    a.precision.contains(&k.note),
                    on,
                    "{what}: {:?}",
                    a.precision
                );
                for c in Choice::ALL {
                    if c != k.choice && !others {
                        let theirs: Vec<_> = KERNELS
                            .iter()
                            .filter(|o| o.choice == c && o.note != k.note)
                            .map(|o| o.note)
                            .collect();
                        assert!(
                            theirs.iter().all(|n| !a.precision.contains(n)),
                            "{what}: {:?}",
                            a.precision
                        );
                    }
                }
                if on && others {
                    assert_eq!(functions(&a.module.words), 1, "{what}: no routine");
                }
                if !on && !others {
                    assert!(
                        NATIVES.iter().all(|n| !a.precision.contains(n)),
                        "{what}: {:?}",
                        a.precision
                    );
                }
                a.module.words.len()
            };
            let (on, off) = (words(true), words(false));
            assert!(
                on < off,
                "{} others {others}: on {on} words, off {off}",
                k.name
            );
        }
    }
}

/// On lavapipe the maths functions' device forms compute within the bounds their notes state on the
/// interpreter (which models the divisions and the `half` `fma`, not the transcendental
/// instructions), and Metal's off: the correctly rounded quotient, `x` times `RN(1 / y)` for an
/// `arcp` one, the correctly rounded `half` `fma`.
#[test]
fn the_maths_functions_are_within_their_bounds_on_and_metal_s_off() {
    let Some(bc) = bitcode() else {
        return;
    };
    let on = profile("lavapipe", Choice::Maths.set(Choice::every(false), true)).expect("on");
    let off = profile("lavapipe", Choice::every(false)).expect("off");
    for (a, b) in [(1.0f32, 3.0f32), (-3.0, 0.7), (0.75, -4.1), (1e-30, 3e7)] {
        for div in ["native_fdiv", "native_fdiv_arcp"] {
            let q = interpret(&bc, div, &on, [a, b, 0.0]);
            assert!(
                ulps(q, f64::from(a) / f64::from(b)) <= 2.5,
                "{div}({a}, {b})"
            );
        }
        let exact = interpret(&bc, "native_fdiv", &off, [a, b, 0.0]);
        assert_eq!(exact, (f64::from(a) / f64::from(b)) as f32, "{a} / {b}");
        let fast = interpret(&bc, "native_fdiv_arcp", &off, [a, b, 0.0]);
        assert_eq!(fast, a * (1.0 / b), "{a} / {b} arcp");
    }
    // A precise division's divisor past `2^126` is scaled into Vulkan's range.
    let big = 2f32.powi(127);
    assert_eq!(interpret(&bc, "native_fdiv", &on, [big, big, 0.0]), 1.0);
    // `half` `fma(1.5, 2.25, 0.125)` is exact: 3.5.
    for p in [&on, &off] {
        assert_eq!(
            interpret(&bc, "native_half_fma", p, [1.5, 2.25, 0.125]),
            3.5
        );
    }
}

/// On lavapipe with the NaN and minimum and maximum on, an addition answers the device's NaN (a
/// NaN, here the operand's payload) where off selects Metal's canonical `0x7fc00000`, and `fmin` is
/// the device's `NMin` and `saturate` its `NClamp(x, 0, 1)`, answering the operand Metal does
/// beside a NaN and between two numbers either way. The maths functions' device divisions take the
/// same choice's NaN: Metal's canonical one off, selected (the routine they replace answers it),
/// the device's on.
#[test]
fn the_nan_and_min_max_are_the_device_s_on_and_metal_s_off() {
    let Some(bc) = bitcode() else {
        return;
    };
    let base = Choice::every(true);
    let on = profile("lavapipe", Choice::NanMinMax.set(base, true)).expect("on");
    let off = profile("lavapipe", Choice::NanMinMax.set(base, false)).expect("off");
    let nan = f32::from_bits(0x7fc0_1234);
    for name in ["native_fadd", "native_fdiv", "native_fdiv_arcp"] {
        let run = |p: &DeviceProfile| interpret(&bc, name, p, [nan, 1.0, 0.0]);
        assert!(run(&on).is_nan(), "{name}");
        assert_eq!(run(&off).to_bits(), 0x7fc0_0000, "{name}");
    }
    for p in [&on, &off] {
        let min = |a: f32, b: f32| interpret(&bc, "native_fmin", p, [a, b, 0.0]);
        assert_eq!(min(nan, 2.0), 2.0);
        assert_eq!(min(2.0, nan), 2.0);
        assert_eq!(min(1.0, -3.0), -3.0);
        // `saturate` is Metal's `clamp(x, +0, 1)`, the device's `NClamp` on.
        let saturate = |x: f32| interpret(&bc, "native_saturate", p, [x, 0.0, 0.0]);
        for (x, want) in [
            (nan, 0.0f32),
            (2.0, 1.0),
            (-3.0, 0.0),
            (0.25, 0.25),
            (f32::INFINITY, 1.0),
            (f32::NEG_INFINITY, 0.0),
        ] {
            assert_eq!(saturate(x), want, "saturate({x})");
        }
        // Either zero on (`NativeMinMax`'s tie), `+0` off.
        assert_eq!(saturate(-0.0), 0.0);
    }
    assert_eq!(
        interpret(&bc, "native_saturate", &off, [-0.0, 0.0, 0.0]).to_bits(),
        0
    );
}

/// On lavapipe with the rounding order on, `dot` is the device's `OpDot` and `mix` its `FMix` (a
/// `half` one rounded once), each within the bound its note states, and `fast::sqrt` the device's
/// `Sqrt`, within Vulkan's `1 / InverseSqrt` (4.5 ulp); off, `dot` is Metal's fused chain and `mix`
/// its `fma(t, y, fma(-t, x, x))`, word for word.
#[test]
fn the_rounding_order_is_the_device_s_on_and_metal_s_off() {
    let Some(bc) = bitcode() else {
        return;
    };
    let base = Choice::every(true);
    let on = profile("lavapipe", Choice::RoundingOrder.set(base, true)).expect("on");
    let off = profile("lavapipe", Choice::RoundingOrder.set(base, false)).expect("off");
    for [a, b, c] in [
        [1.0f32, 3.0, 0.5],
        [-3.0, 0.7, 1.9],
        [0.75, -4.1, 2.5e-3],
        [1e-3, 3e7, -0.25],
    ] {
        let (a64, b64, c64) = (f64::from(a), f64::from(b), f64::from(c));
        // `dot(float2(a, b), float2(c, a))`: two lanes, `(3n - 1) * 2^-24 * sum |a_i * b_i|`.
        let sum = (a64 * c64).abs() + (b64 * a64).abs();
        let dot = interpret(&bc, "native_dot", &on, [a, b, c]);
        assert!(
            (f64::from(dot) - (a64 * c64 + b64 * a64)).abs() <= 5.0 * 2f64.powi(-24) * sum,
            "dot({a}, {b}, {c}) = {dot}"
        );
        assert_eq!(
            interpret(&bc, "native_dot", &off, [a, b, c]),
            b.mul_add(a, a * c),
            "Metal's dot({a}, {b}, {c})"
        );
        // `mix(a, b, c)`: within `2^-21 * (|a * (1 - c)| + |b * c|)`.
        let scale = (a64 * (1.0 - c64)).abs() + (b64 * c64).abs();
        let mix = interpret(&bc, "native_mix", &on, [a, b, c]);
        assert!(
            (f64::from(mix) - (a64 + c64 * (b64 - a64))).abs() <= 2f64.powi(-21) * scale,
            "mix({a}, {b}, {c}) = {mix}"
        );
        assert_eq!(
            interpret(&bc, "native_mix", &off, [a, b, c]),
            c.mul_add(b, (-c).mul_add(a, a)),
            "Metal's mix({a}, {b}, {c})"
        );
    }
    // `half` `mix(0.5, 1.5, 0.25)` is exact: 0.75.
    for p in [&on, &off] {
        assert_eq!(interpret(&bc, "native_half_mix", p, [0.5, 1.5, 0.25]), 0.75);
    }
    // On, `fast::sqrt`'s zero and `+inf` are IEEE's, as Metal's (the interpreter leaves off's
    // `InverseSqrt` unmodelled).
    let sqrt = |x: f32| interpret(&bc, "native_fast_sqrt", &on, [x, 0.0, 0.0]);
    for x in [2.0f32, 0.75, 1e-30, 3e30] {
        assert!(ulps(sqrt(x), f64::from(x).sqrt()) <= 4.5, "fast::sqrt({x})");
    }
    assert_eq!(sqrt(0.0).to_bits(), 0);
    assert_eq!(sqrt(f32::INFINITY), f32::INFINITY);
}
