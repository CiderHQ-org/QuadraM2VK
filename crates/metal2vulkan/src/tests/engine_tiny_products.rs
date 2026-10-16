//! Metal's `float` multiply flushes a product whose exact value lies below `2^-126` to the zero of
//! its sign before rounding it (tininess before rounding), on vector and scalar lanes, with and
//! without fast math, and as the first product of `dot`'s fused chain: the owned fixtures
//! `tiny_products_flush` and `tiny_products_flush_fast`, whose words on an Apple M3 Max under
//! macOS 27.0.1 are `tiny_products_m3.txt` (the authored cases of the fixtures,
//! `float-products-flushed-before-rounding` and its fast-math twin).
//! A Vulkan device rounds first, so an exact product within half a unit below `2^-126` answers
//! `2^-126` there.
//!
//! The CPU SPIR-V interpreter rounds first and keeps denormals, as a device without
//! `DriverDeviation::MetalFloatDenormals` may, so on each such committed profile (one per fused
//! multiply-add route and routine of its small results) it shows what each policy answers: under
//! `FloatMathsPolicy::Exact` every word is Metal's; under the default a product is the device's,
//! rounded first (announced once per device, `float-tininess-after-rounding`). On every committed
//! profile the check of a product (`arith.float.tiny`) is compiled exactly where the profile's
//! functions are to answer as Metal's (`denormal::checks_tiny_products`), inline and branchless.
//! `llvm-as` assembles the fixtures; without it the tests skip visibly.

use crate::device::profile::FloatFunctions;
use crate::device::{DevicePolicy, DeviceProfile};
use crate::error::DriverDeviation;
use crate::lower::arith::fma;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::pipeline::testing::{
    compile, exact_maths, link_and_validate, llvm_assemble, typed_module, AssembleError,
};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::testing::committed_profiles_with;
use std::collections::BTreeSet;

/// Metal's words (`tiny_products_m3.txt`).
const WORDS: &str = include_str!("tiny_products_m3.txt");

/// The fixtures' entries: without and with fast math.
const ENTRIES: [&str; 2] = ["tiny_products_flush", "tiny_products_flush_fast"];

/// The census name of a product's check.
const TINY: &str = "arith.float.tiny";

/// A thread past the authored cases', its words measured on the same M3 Max under macOS 27.0.1 by
/// the fixtures' own source: `(2 - 2^-23) · 2^-126`, a normal `float`, times `±0.5` on each lane,
/// whose exact product `±(2^-126 - 2^-150)` a device that rounds first ties to `±2^-126`, where
/// Metal answers the zero of its sign. A product by a constant of magnitude at least 1 is never
/// checked (its product is a zero or at least the other operand's magnitude), but one by `0.5` is.
const HALF_A_NORMAL: &str = concat!(
    "00ffffff 80ffffff 3f000000 00ffffff 3f000000 3f000000 00ffffff bf000000 ",
    "00000000 80000000 00000000 80000000 00000000 80000000 ",
    "00000000 80000000 80000000 80000000 80000000 80000000 ",
    "00000000 80000000 00000000 80000000 00000000 80000000 ",
    "00000000 80000000 80000000 80000000 80000000 80000000",
);

/// One thread's operands (`a` then `b`, four lanes each) and Metal's twelve words for each entry:
/// `a * b`, the scalar products, and the four `dot`s.
struct Thread {
    input: [u32; 8],
    metal: [[u32; 12]; 2],
}

/// The threads of `tiny_products_m3.txt`, then [`HALF_A_NORMAL`].
fn threads() -> Vec<Thread> {
    let hex = |w: &str| u32::from_str_radix(w, 16).expect("hex word");
    let threads: Vec<Thread> = WORDS
        .lines()
        .filter(|l| !l.starts_with('#'))
        .chain([HALF_A_NORMAL])
        .map(|l| {
            let w: Vec<u32> = l.split_whitespace().map(hex).collect();
            assert_eq!(w.len(), 32, "a thread's line: {l}");
            let mut t = Thread {
                input: [0; 8],
                metal: [[0; 12]; 2],
            };
            t.input.copy_from_slice(&w[..8]);
            t.metal[0].copy_from_slice(&w[8..20]);
            t.metal[1].copy_from_slice(&w[20..]);
            t
        })
        .collect();
    assert_eq!(threads.len(), 65, "the fixtures' threads and HALF_A_NORMAL");
    threads
}

/// Fixture `entry` assembled, or `None` (skipped visibly) without `llvm-as`.
fn assemble(entry: &str) -> Option<Vec<u8>> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../metal2vulkan-validation/fixtures/public/engine")
        .join(format!("{entry}.ll"));
    match llvm_assemble(&path) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(t)) => {
            eprintln!("SKIPPED: {t}");
            None
        }
        Err(AssembleError::Rejected(e)) => panic!("llvm-as rejected {entry}: {e}"),
    }
}

/// The words `entry` stores for `threads` on `p`, run on the interpreter, a thread a workgroup.
fn interpret(bc: &[u8], entry: &str, p: &DeviceProfile, threads: &[Thread]) -> Vec<[u32; 12]> {
    let (plan, m) = typed_module(bc, entry, p).unwrap_or_else(|e| panic!("{entry}: {e}"));
    let input: Vec<u8> = threads
        .iter()
        .flat_map(|t| t.input.iter().flat_map(|w| w.to_le_bytes()))
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
                bytes: vec![0xab; 48 * threads.len()],
            },
        ],
        push: vec![0; plan.push.push_range.map_or(0, |(_, end)| end as usize)],
        spec: Default::default(),
    };
    let config = Config {
        workgroups: [u32::try_from(threads.len()).expect("threads"), 1, 1],
        ..Config::default()
    };
    let out = run(&m, &config, res).unwrap_or_else(|e| panic!("{entry}: interpreter: {e}"));
    out.buffers[1]
        .as_chunks::<48>()
        .0
        .iter()
        .map(|t| {
            let mut words = [0; 12];
            for (w, c) in words.iter_mut().zip(t.as_chunks::<4>().0) {
                *w = u32::from_le_bytes(*c);
            }
            words
        })
        .collect()
}

/// The committed profiles under `policy` whose device may keep denormals (no
/// `DriverDeviation::MetalFloatDenormals`), as the interpreter does, one per fused multiply-add
/// route and routine of its small results ([`fma::tiny_route`]).
fn keeping(policy: DevicePolicy) -> Vec<(String, DeviceProfile)> {
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

/// The product of `a` and `b` on a device that rounds first and keeps denormals, its operands and
/// result read as Metal reads a denormal (the module flushes them there).
fn device_mul(a: u32, b: u32) -> u32 {
    let f = |x: u32| f32::from_bits(ftz(x));
    ftz((f(a) * f(b)).to_bits())
}

/// Whether every committed profile under `policy` compiles a product's check exactly where its
/// functions are to answer as Metal's, and how many do.
fn checks_where_device_functions(bc: &[u8], entry: &str, policy: DevicePolicy) -> usize {
    let mut checking = 0;
    for (stem, p) in committed_profiles_with(policy).expect("profiles") {
        let what = format!("{entry} on {stem}");
        let a = compile(bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        let device = p.fields().float_functions == FloatFunctions::Device;
        assert_eq!(a.strategies.entries.contains_key(TINY), device, "{what}");
        checking += usize::from(device);
    }
    checking
}

/// Under `FloatMathsPolicy::Exact` every committed profile whose functions are not Metal's own
/// checks its products (MoltenVK's multiply is Metal's), and on each one that may keep a denormal
/// every word, multiplies and `dot`s alike, is Metal's.
#[test]
fn exact_float_maths_flushes_a_tiny_product_as_metal() {
    let threads = threads();
    for (i, entry) in ENTRIES.into_iter().enumerate() {
        let Some(bc) = assemble(entry) else {
            return;
        };
        let checking = checks_where_device_functions(&bc, entry, exact_maths());
        assert!(checking > 0, "{entry}: no profile checks a product");
        let profiles = keeping(exact_maths());
        assert!(!profiles.is_empty(), "no profile keeps denormals");
        for (stem, p) in profiles {
            let got = interpret(&bc, entry, &p, &threads);
            let wrong: Vec<String> = threads
                .iter()
                .zip(&got)
                .enumerate()
                .flat_map(|(t, (thread, words))| {
                    (0..12)
                        .filter(|&w| words[w] != thread.metal[i][w])
                        .map(move |w| {
                            format!(
                                "thread {t} word {w}: {:08x}, Metal {:08x}",
                                words[w], thread.metal[i][w]
                            )
                        })
                })
                .collect();
            assert!(
                wrong.is_empty(),
                "{entry} on {stem}: {} words off Metal's: {:#?}",
                wrong.len(),
                &wrong[..wrong.len().min(20)]
            );
        }
    }
}

/// Under the default policy no profile checks a product: on a device that may keep a denormal
/// each product is the device's, rounded first, which answers `2^-126` of its sign on lanes where
/// Metal answers the zero of its sign (some do), and Metal's word on every other lane.
#[test]
fn the_default_leaves_a_tiny_product_to_the_device() {
    let threads = threads();
    for (i, entry) in ENTRIES.into_iter().enumerate() {
        let Some(bc) = assemble(entry) else {
            return;
        };
        let checking = checks_where_device_functions(&bc, entry, DevicePolicy::default());
        assert_eq!(
            checking, 0,
            "{entry}: a profile checks a product by default"
        );
        for (stem, p) in keeping(DevicePolicy::default()) {
            let got = interpret(&bc, entry, &p, &threads);
            let mut rounded_first = 0;
            for (t, (thread, words)) in threads.iter().zip(&got).enumerate() {
                // The vector products, then the scalar ones: lane `l` of each is `a[l] * b[l]`.
                let products = words.iter().zip(&thread.metal[i]).take(8);
                for (w, (&got, &metal)) in products.enumerate() {
                    let l = w % 4;
                    let device = device_mul(thread.input[l], thread.input[4 + l]);
                    assert_eq!(
                        got, device,
                        "{entry} on {stem}: thread {t} word {w}, Metal {metal:08x}"
                    );
                    if device != metal {
                        assert_eq!(
                            (device & 0x7FFF_FFFF, device & 0x8000_0000),
                            (0x0080_0000, metal),
                            "{entry} on {stem}: thread {t} word {w}: only a tiny product differs"
                        );
                        rounded_first += 1;
                    }
                }
            }
            assert!(
                rounded_first > 0,
                "{entry} on {stem}: no product rounded up to 2^-126"
            );
        }
    }
}

/// The `UMulExtended` instructions inside the entry point's own function of module `words`.
fn entry_umul_extended(words: &[u32]) -> usize {
    use crate::spv::Op;
    let instructions = || {
        let mut i = 5;
        std::iter::from_fn(move || {
            let len = (*words.get(i)? >> 16) as usize;
            assert!(len > 0 && i + len <= words.len(), "malformed module");
            let at = &words[i..i + len];
            i += len;
            Some(at)
        })
    };
    let entry = instructions()
        .find(|at| at[0] & 0xFFFF == Op::EntryPoint as u32)
        .map(|at| at[2])
        .expect("an entry point");
    let (mut inside, mut n) = (false, 0);
    for at in instructions() {
        let op = at[0] & 0xFFFF;
        if op == Op::Function as u32 {
            inside = at[2] == entry;
        } else if op == Op::FunctionEnd as u32 {
            inside = false;
        } else if op == Op::UMulExtended as u32 && inside {
            n += 1;
        }
    }
    n
}

/// Under `FloatMathsPolicy::Exact` each product's check is inline and branchless, in the form the
/// profile takes (`denormal::tiny_check`): where its `float` fused multiply-add is `OpFmaKHR`, one
/// `OpFmaKHR` on the operands' magnitudes and no integer product; elsewhere the operands' bits,
/// one `UMulExtended` a check (nine in each fixture: the vector product, the four scalar ones and
/// each `dot`'s first). The plan's size estimate bounds the module.
#[test]
fn a_product_s_check_is_inline_in_the_profile_s_form() {
    use crate::lower::arith::denormal::{tiny_check, TinyCheck};
    for entry in ENTRIES {
        let Some(bc) = assemble(entry) else {
            return;
        };
        let mut forms = BTreeSet::new();
        for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
            let Some(check) = tiny_check(&p) else {
                continue;
            };
            let what = format!("{entry} on {stem}");
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
            let products = entry_umul_extended(&a.module.words);
            let want = match check {
                TinyCheck::Fused => 0,
                TinyCheck::Bits => 9,
            };
            assert_eq!(products, want, "{what}: {check:?}");
            let (plan, _) = typed_module(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e}"));
            let words = a.module.words.len() as u64;
            assert!(
                plan.estimate.words >= words,
                "{what}: estimate {} < emitted {words}",
                plan.estimate.words
            );
            forms.insert(format!("{check:?}"));
        }
        assert!(
            forms.contains("Fused"),
            "{entry}: no profile checks by OpFmaKHR: {forms:?}"
        );
    }
}

/// The functions of module `words` declared `DontInline` (`OpFunction`'s control `0x2`).
fn dont_inline_functions(words: &[u32]) -> usize {
    use crate::spv::Op;
    let mut i = 5;
    let mut found = 0;
    while i < words.len() {
        let (len, op) = ((words[i] >> 16) as usize, words[i] & 0xFFFF);
        assert!(len > 0 && i + len <= words.len(), "malformed module");
        if op == Op::Function as u32 && words[i + 3] & 0x2 != 0 {
            found += 1;
        }
        i += len;
    }
    found
}

/// Under `FloatMathsPolicy::Exact` a checked `OpFmaKHR` (`dot`'s chain in the fixtures) answers
/// its small results by a plain routine, which a driver may inline: NVIDIA keeps a call declared
/// `DontInline`, and on the RTX 5080 (615.71.09) the call cost more than the inlined routine, in
/// pipeline creation and in GPU time. No function of the fixture is declared so, on any profile.
#[test]
fn the_routine_of_small_fma_results_is_not_declared_dont_inline() {
    use crate::lower::arith::fma::tiny_route;
    let entry = ENTRIES[0];
    let Some(bc) = assemble(entry) else {
        return;
    };
    let mut checked = 0;
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        if tiny_route(&p).is_none() {
            continue;
        }
        let what = format!("{entry} on {stem}");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        assert_eq!(dont_inline_functions(&a.module.words), 0, "{what}");
        checked += 1;
    }
    assert!(checked > 0, "no profile checks OpFmaKHR");
}

/// A check that answers Metal's zero of its sign keeps zero signs, in a fast-math entry too: a
/// product's at `±2^-126` and `dot`'s chain, a checked `OpFmaKHR` or a fused multiply-add routine,
/// make the entry declare `SignedZeroInfNanPreserve 32`, and the chain whose routine computes in
/// `double` (`fma::small`, the `double` route) `64`, each where the profile supports it, else note
/// `DeviceSignedZeroInfNan`. Without the mode Vulkan
/// lets a device answer a zero of either sign from any float operation: NVIDIA's 615.71.09 on the
/// RTX 3090 answered `-0` for a checked `fma(+0, 100, +0)`, where Metal answers `+0`. Under the
/// default policy nothing is checked, and the fast entry declares neither.
#[test]
fn checks_that_answer_a_zero_s_sign_keep_zero_signs() {
    use super::engine_arith::declares_signed_zero_inf_nan;
    use crate::pipeline::artifact::PrecisionNote;
    let entry = ENTRIES[1];
    let Some(bc) = assemble(entry) else {
        return;
    };
    let mut declared = [0; 2];
    for (stem, p) in committed_profiles_with(exact_maths()).expect("profiles") {
        let what = format!("{entry} on {stem}");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        // The fixture's products are checked where the profile checks one, and `dot`'s chain is
        // then a checked `OpFmaKHR` or a routine, in `double` on the `double` route.
        let tiny = fma::tiny_route(&p);
        let exact = p.fields().float_functions == FloatFunctions::Device;
        let wide = fma::float_route(&p) == fma::Route::Wide || tiny == Some(fma::Route::Wide);
        let wanted = [
            a.strategies.entries.contains_key(TINY) || tiny.is_some(),
            exact && wide,
        ];
        let fc = p.fields().float_controls;
        let supported = [
            fc.f32.signed_zero_inf_nan_preserve,
            fc.f64.signed_zero_inf_nan_preserve,
        ];
        for (i, width) in [32, 64].into_iter().enumerate() {
            let mode = declares_signed_zero_inf_nan(&words, width);
            assert_eq!(mode, wanted[i] && supported[i], "{what}: mode {width}");
            declared[i] += usize::from(mode);
        }
        let unsupported = (0..2).any(|i| wanted[i] && !supported[i]);
        assert_eq!(
            a.precision.contains(&PrecisionNote::DeviceSignedZeroInfNan),
            unsupported,
            "{what}: note"
        );
    }
    assert!(
        declared.iter().all(|&n| n > 0),
        "no profile declares a width: {declared:?}"
    );
    for (stem, p) in committed_profiles_with(DevicePolicy::default()).expect("profiles") {
        let what = format!("{entry} on {stem} by default");
        let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{what}: {e:?}"));
        let (_, words) = link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{what}: L3: {e}"));
        for width in [32, 64] {
            assert!(
                !declares_signed_zero_inf_nan(&words, width),
                "{what}: mode {width}"
            );
        }
    }
}
