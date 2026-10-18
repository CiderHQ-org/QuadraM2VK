//! How the translation reads `f32` denormals in `mix`, two chained `fma`s, `atan2`, `sinpi`,
//! `cospi`, `tanpi` and `fast::fmod`, precise and fast, against Metal's words on an Apple M3 Max
//! under macOS 27.0.1 (`metal_denormals_m3.txt`). Metal reads every denormal operand of these as
//! the zero of its sign and answers no denormal; `mix(a, b, c)` is `fma(c, b, fma(-c, a, a))` with
//! each `fma` flushing its operands and its result.
//!
//! Each kernel loops over its rows (three operand words a row in `[[buffer(0)]]`) and stores one
//! answer word a row in `[[buffer(1)]]`. The CPU SPIR-V interpreter keeps denormals, as a device
//! without `DriverDeviation::MetalFloatDenormals` may, so on each such committed profile (one per
//! fused multiply-add route and routine of its small results) an entry that records no
//! `PrecisionNote::DeviceDenormals` must answer every row as Metal did, but, where the device's
//! fused multiply-add is unchecked (`fma::tiny_route`), for its rounding before the flush
//! ([`device_fma_gap`]). `llvm-as` assembles the module; without it the tests skip visibly.

use crate::device::{DevicePolicy, DeviceProfile, FloatMathsPolicy, RoundingOrderPolicy};
use crate::error::DriverDeviation;
use crate::lower::arith::fma;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, link_and_validate, llvm_assemble, typed_module, AssembleError, Scratch,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles_with;
use std::collections::BTreeSet;

/// Metal's words.
const WORDS: &str = include_str!("metal_denormals_m3.txt");

/// A function of the probe.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fun {
    Mix,
    /// `fma(c, b, fma(-c, a, a))`, which Metal answers as its `mix` on every row.
    FmaFma,
    Atan2,
    SinPi,
    CosPi,
    TanPi,
    /// `fast::fmod`, which Metal compiles to `air.fast_fmod` in both modes.
    Fmod,
}

const FUNS: [Fun; 7] = [
    Fun::Mix,
    Fun::FmaFma,
    Fun::Atan2,
    Fun::SinPi,
    Fun::CosPi,
    Fun::TanPi,
    Fun::Fmod,
];

impl Fun {
    /// The fixture's name for the function whose words this one answers.
    fn table(self) -> &'static str {
        match self {
            Fun::Mix | Fun::FmaFma => "mix",
            Fun::Atan2 => "atan2",
            Fun::SinPi => "sinpi",
            Fun::CosPi => "cospi",
            Fun::TanPi => "tanpi",
            Fun::Fmod => "fmod",
        }
    }

    fn arity(self) -> usize {
        match self {
            Fun::Mix | Fun::FmaFma => 3,
            Fun::Atan2 | Fun::Fmod => 2,
            Fun::SinPi | Fun::CosPi | Fun::TanPi => 1,
        }
    }

    /// The kernel's name.
    fn name(self, fast: bool) -> String {
        let mode = if fast { "fast" } else { "precise" };
        format!("dn_{self:?}_{mode}").to_lowercase()
    }

    /// The call of `%a`, `%b` and `%c` into `%s` as Metal compiles it in each mode: fast math
    /// takes the `fast_` functions and every flag.
    fn body(self, fast: bool) -> String {
        let flags = if fast { "fast " } else { "" };
        let prefix = if fast { "fast_" } else { "" };
        match self {
            Fun::Mix => {
                format!("  %s = call {flags}float @air.mix.f32(float %a, float %b, float %c)\n")
            }
            Fun::FmaFma => format!(
                "  %nc = fneg {flags}float %c\n  %t = call {flags}float @air.fma.f32(float %nc, \
                 float %a, float %a)\n  %s = call {flags}float @air.fma.f32(float %c, float %b, \
                 float %t)\n"
            ),
            Fun::Atan2 => {
                format!("  %s = call {flags}float @air.{prefix}atan2.f32(float %a, float %b)\n")
            }
            Fun::SinPi | Fun::CosPi | Fun::TanPi => format!(
                "  %s = call {flags}float @air.{prefix}{}.f32(float %a)\n",
                self.table()
            ),
            Fun::Fmod => {
                format!("  %s = call {flags}float @air.fast_fmod.f32(float %a, float %b)\n")
            }
        }
    }
}

/// The fixture's rows of `fun`: the operands, padded with zeros to three, and Metal's answers
/// without and with fast math.
fn rows(fun: Fun) -> Vec<([u32; 3], [u32; 2])> {
    let hex = |w: &str| u32::from_str_radix(w, 16).expect("hex word");
    WORDS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .filter_map(|l| {
            let w: Vec<&str> = l.split_whitespace().collect();
            (w[0] == fun.table()).then(|| {
                let n = fun.arity();
                let mut ops = [0; 3];
                for (o, x) in ops.iter_mut().zip(&w[1..=n]) {
                    *o = hex(x);
                }
                (ops, [hex(w[n + 1]), hex(w[n + 2])])
            })
        })
        .collect()
}

/// The kernel of `fun` in mode `fast` over `n` rows.
fn kernel(fun: Fun, fast: bool, n: usize) -> String {
    let mut b = format!(
        "define void @{}(ptr addrspace(1) noundef \"air-buffer-no-alias\" %0, ptr addrspace(1) \
         noundef \"air-buffer-no-alias\" %1) {{\nentry:\n  br label %loop\nloop:\n  %g = phi i32 \
         [ 0, %entry ], [ %g1, %loop ]\n  %base = mul i32 %g, 3\n",
        fun.name(fast)
    );
    for (j, v) in ["a", "b", "c"].iter().enumerate() {
        b.push_str(&format!(
            "  %i{j} = add i32 %base, {j}\n  %z{j} = zext i32 %i{j} to i64\n  %p{j} = \
             getelementptr inbounds i32, ptr addrspace(1) %0, i64 %z{j}\n  %w{j} = load i32, ptr \
             addrspace(1) %p{j}, align 4\n  %{v} = bitcast i32 %w{j} to float\n"
        ));
    }
    b.push_str(&fun.body(fast));
    b.push_str(&format!(
        "  %r = bitcast float %s to i32\n  %zg = zext i32 %g to i64\n  %q = getelementptr \
         inbounds i32, ptr addrspace(1) %1, i64 %zg\n  store i32 %r, ptr addrspace(1) %q, align \
         4\n  %g1 = add i32 %g, 1\n  %done = icmp eq i32 %g1, {n}\n  br i1 %done, label %exit, \
         label %loop\nexit:\n  ret void\n}}\n\n"
    ));
    b
}

/// Every kernel, as `(fun, fast)`.
fn kernels() -> Vec<(Fun, bool)> {
    FUNS.iter().flat_map(|&f| [(f, false), (f, true)]).collect()
}

/// The module of every kernel.
fn module() -> String {
    let mut s = String::from(
        "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n",
    );
    let ks = kernels();
    for &(fun, fast) in &ks {
        s.push_str(&kernel(fun, fast, rows(fun).len()));
    }
    s.push_str(
        "declare float @air.mix.f32(float, float, float)\ndeclare float @air.fma.f32(float, \
         float, float)\ndeclare float @air.atan2.f32(float, float)\ndeclare float \
         @air.fast_atan2.f32(float, float)\ndeclare float @air.fast_fmod.f32(float, float)\n",
    );
    for f in ["sinpi", "cospi", "tanpi"] {
        s.push_str(&format!(
            "declare float @air.{f}.f32(float)\ndeclare float @air.fast_{f}.f32(float)\n"
        ));
    }
    let refs: Vec<String> = (0..ks.len()).map(|i| format!("!{}", 20 + i)).collect();
    s.push_str(&format!(
        "\n!air.kernel = !{{{}}}\n!air.version = !{{!17}}\n!air.language_version = !{{!18}}\n\n",
        refs.join(", ")
    ));
    s.push_str(
        "!10 = !{}\n!11 = !{!12, !13}\n!12 = !{i32 0, !\"air.buffer\", !\"air.location_index\", \
         i32 0, i32 1, !\"air.read\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", i32 \
         4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"s\"}\n!13 = !{i32 1, !\"air.buffer\", !\"air.location_index\", i32 \
         1, i32 1, !\"air.read_write\", !\"air.address_space\", i32 1, !\"air.arg_type_size\", \
         i32 4, !\"air.arg_type_align_size\", i32 4, !\"air.arg_type_name\", !\"uint\", \
         !\"air.arg_name\", !\"out\"}\n!17 = !{i32 2, i32 8, i32 0}\n!18 = !{!\"Metal\", i32 4, \
         i32 0, i32 0}\n",
    );
    for (i, &(fun, fast)) in ks.iter().enumerate() {
        s.push_str(&format!(
            "!{} = !{{ptr @{}, !10, !11}}\n",
            20 + i,
            fun.name(fast)
        ));
    }
    s
}

/// Assemble the module, or `None` (skipped visibly) without `llvm-as`.
fn assemble() -> Option<Vec<u8>> {
    let scratch = Scratch::new("m2v-metal-denormals");
    let dir = scratch.path();
    let ll = dir.join("denormals.ll");
    std::fs::write(&ll, module()).expect("write");
    let assembled = llvm_assemble(&ll);
    match assembled {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected the denormals module: {e}"),
    }
}

/// The words kernel `name` stores for `rows` on `p`, run on the interpreter.
fn interpret(bc: &[u8], name: &str, p: &DeviceProfile, rows: &[([u32; 3], [u32; 2])]) -> Vec<u32> {
    let (plan, m) = typed_module(bc, name, p).unwrap_or_else(|e| panic!("{name}: {e}"));
    let input: Vec<u8> = rows
        .iter()
        .flat_map(|(ops, _)| ops.iter().flat_map(|w| w.to_le_bytes()))
        .collect();
    let slot = |index| {
        Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index,
        })
    };
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: slot(0),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: slot(1),
                address: None,
                bytes: vec![0xab; 4 * rows.len()],
            },
        ],
        push: vec![0; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let config = Config {
        step_limit: 1 << 40,
        ..Config::default()
    };
    let out = run(&m, &config, res).unwrap_or_else(|e| panic!("{name}: interpreter: {e}"));
    out.buffers[1]
        .chunks(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// The committed profiles under `policy` whose device may keep denormals (no
/// `DriverDeviation::MetalFloatDenormals`), one per fused multiply-add route and routine of its
/// small results ([`fma::tiny_route`]).
fn profiles(policy: DevicePolicy) -> Vec<(String, DeviceProfile)> {
    let mut seen = BTreeSet::new();
    committed_profiles_with(policy)
        .expect("profiles")
        .into_iter()
        .filter(|(_, p)| !p.takes_driver_deviation(DriverDeviation::MetalFloatDenormals))
        .filter(|(_, p)| seen.insert((fma::float_route(p), fma::tiny_route(p))))
        .collect()
}

/// An `f32` denormal as the zero of its sign.
fn ftz(x: u32) -> u32 {
    if x & 0x7F80_0000 == 0 {
        x & 0x8000_0000
    } else {
        x
    }
}

/// The device's fused multiply-add (`OpFmaKHR`, or native device maths' `Fma`) on a device that
/// keeps denormals, read as Metal reads them around it: the operands and the rounded result
/// flushed, a NaN Metal's.
fn device_fma(a: u32, b: u32, c: u32) -> u32 {
    let f = |x: u32| f32::from_bits(ftz(x));
    let r = ftz(f(a).mul_add(f(b), f(c)).to_bits());
    if f32::from_bits(r).is_nan() {
        0x7FC0_0000
    } else {
        r
    }
}

/// The device's fused multiply-add unchecked (native device maths, which the runtime announces as
/// `float-tininess-after-rounding`; docs/FAST-PATHS.md): Metal flushes a result whose exact value
/// is below `2^-126` before rounding it, where the device has rounded it up to `2^-126`, which the
/// flush after it keeps. `fun`'s answer with each `fma` so, where it is a composition of them.
fn device_fma_gap(fun: Fun, [a, b, c]: [u32; 3]) -> Option<u32> {
    match fun {
        Fun::Mix | Fun::FmaFma => Some(device_fma(c, b, device_fma(c ^ 0x8000_0000, a, a))),
        _ => None,
    }
}

/// What every kernel on one profile answered.
#[derive(Default)]
struct Report {
    /// The kernels that record `DeviceDenormals`.
    noted: Vec<String>,
    /// The rows of [`device_fma_gap`]'s known gap, per kernel.
    gaps: Vec<(String, usize)>,
    /// Every other answer that is not Metal's word.
    wrong: Vec<String>,
}

/// Every kernel on `p` against Metal's words.
fn check(bc: &[u8], stem: &str, p: &DeviceProfile) -> Report {
    let device = fma::float_route(p).device_float() && fma::tiny_route(p).is_none();
    let mut report = Report::default();
    for (fun, fast) in kernels() {
        let name = fun.name(fast);
        let what = format!("{name} on {stem}");
        let a = compile(bc, &name, p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        if a.precision.contains(&PrecisionNote::DeviceDenormals) {
            report.noted.push(name);
            continue;
        }
        // A kernel that records the device's NaN encoding answers any NaN for Metal's.
        let nan = |w: u32| w & 0x7FFF_FFFF > 0x7F80_0000;
        let device_nan = [
            PrecisionNote::DeviceNanEncoding,
            PrecisionNote::NativeNanEncoding,
        ]
        .iter()
        .any(|n| a.precision.contains(n));
        let rows = rows(fun);
        let got = interpret(bc, &name, p, &rows);
        let mut gaps = 0;
        for ((ops, metal), g) in rows.iter().zip(&got) {
            let want = metal[usize::from(fast)];
            if *g == want || (device_nan && nan(*g) && nan(want)) {
                continue;
            }
            if device && device_fma_gap(fun, *ops) == Some(*g) {
                gaps += 1;
                continue;
            }
            report.wrong.push(format!(
                "{what}: {:08x?}: {g:08x}, Metal {want:08x}",
                &ops[..fun.arity()]
            ));
        }
        if gaps > 0 {
            report.gaps.push((name, gaps));
        }
    }
    report
}

fn exact_answers() -> DevicePolicy {
    DevicePolicy::default().exact_answers()
}

fn exact_float_maths() -> DevicePolicy {
    DevicePolicy {
        float_maths: FloatMathsPolicy::Exact,
        ..DevicePolicy::default()
    }
}

fn exact_rounding_order() -> DevicePolicy {
    DevicePolicy {
        rounding_order: RoundingOrderPolicy::Exact,
        ..DevicePolicy::default()
    }
}

/// Each policy's kernels on each route, against Metal's words: no answer off Metal's but
/// [`device_fma_gap`]'s on the device's fused multiply-add unchecked, and `DeviceDenormals`
/// recorded by exactly the kernels `noted` names (`mix`: the device's `FMix` under the device's
/// order; the others: native device maths).
fn run_policy(bc: &[u8], policy: DevicePolicy, what: &str, noted: &[Fun]) -> Vec<String> {
    let want: Vec<String> = kernels()
        .into_iter()
        .filter(|(f, _)| noted.contains(f))
        .map(|(f, fast)| f.name(fast))
        .collect();
    let mut wrong = Vec::new();
    for (stem, p) in profiles(policy) {
        let report = check(bc, &stem, &p);
        eprintln!(
            "{what} on {stem} ({:?}, checked by {:?}): DeviceDenormals on {:?}; the device fma's \
             gap on {:?}; {} other answers off Metal's",
            fma::float_route(&p),
            fma::tiny_route(&p),
            report.noted,
            report.gaps,
            report.wrong.len()
        );
        assert_eq!(report.noted, want, "{what} on {stem}: DeviceDenormals");
        wrong.extend(report.wrong.into_iter().map(|w| format!("{what}: {w}")));
    }
    wrong
}

/// The accurate forms read every probed function's denormals as Metal does: under
/// `exact_answers()` no kernel records `DeviceDenormals`, under `FloatMathsPolicy::Exact` only
/// `mix` by the device's `FMix` (the device's order), under `RoundingOrderPolicy::Exact` only the
/// functions native device maths answers; on every route of a device that may keep a denormal,
/// every other kernel answers every row with Metal's word. Under `FloatMathsPolicy::Exact`
/// (`exact_answers()` too) the device's fused multiply-add is checked and has no gap; under
/// `RoundingOrderPolicy::Exact` alone, `mix` by it rounds before its flush.
#[test]
fn the_accurate_forms_read_denormals_as_metal() {
    let Some(bc) = assemble() else {
        return;
    };
    let native = [Fun::Atan2, Fun::SinPi, Fun::CosPi, Fun::TanPi, Fun::Fmod];
    let mut wrong = run_policy(&bc, exact_answers(), "exact_answers", &[]);
    wrong.extend(run_policy(
        &bc,
        exact_float_maths(),
        "FloatMathsPolicy::Exact",
        &[Fun::Mix],
    ));
    wrong.extend(run_policy(
        &bc,
        exact_rounding_order(),
        "RoundingOrderPolicy::Exact",
        &native,
    ));
    assert!(
        wrong.is_empty(),
        "{} answers off Metal's, the first: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(40)]
    );
}

/// The default forms answer as Metal does at denormals wherever they record no `DeviceDenormals`
/// (but the unchecked device `fma`'s rounding before its flush); native device maths and the
/// device's `FMix` record it.
#[test]
fn the_default_forms_note_what_they_leave_to_the_device() {
    let Some(bc) = assemble() else {
        return;
    };
    let noted = [
        Fun::Mix,
        Fun::Atan2,
        Fun::SinPi,
        Fun::CosPi,
        Fun::TanPi,
        Fun::Fmod,
    ];
    let wrong = run_policy(&bc, DevicePolicy::default(), "default", &noted);
    assert!(
        wrong.is_empty(),
        "{} answers off Metal's, the first: {:#?}",
        wrong.len(),
        &wrong[..wrong.len().min(40)]
    );
}
