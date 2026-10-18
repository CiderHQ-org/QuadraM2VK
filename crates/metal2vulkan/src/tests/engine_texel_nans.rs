//! Metal's NaN of a `half` texel under the NaN switch: plain Metal (an Apple M3 Max
//! under macOS 26.7) hands back a 16-bit float texel's own bits, its sign, payload and signalling
//! bit kept, for a read, a nearest sample or a gather, and `0x7e00` for every other NaN: a 32-bit
//! float texel's, and every linear sample's.
//!
//! On the lavapipe profile (native half rounding and native device sampling; the device's own NaN
//! by default), a kernel reads, samples (nearest and linear, constexpr samplers) or gathers a
//! `texture2d<half>` and stores the `half4`:
//!
//! - a read runs on the CPU interpreter over a uniform texture whose texels the device decodes
//!   exactly (each `half` widened, as NVIDIA's texture unit decodes them): under
//!   `NanMinMaxPolicy::Exact` every NaN lane is Metal's, the texel's own bits where the slot's
//!   class (`TexelFormat`, read as data) is a 16-bit float format's and `0x7e00` where it is a
//!   32-bit float format's, every other lane the texel's value; under `NanMinMaxPolicy::Native`,
//!   the default, the module reads no class, and a signalling NaN texel does not read back as
//!   its bits;
//! - a sample or gather (the interpreter models no device sampler) is checked in its module:
//!   under the switch a nearest sample and a gather read the class and hold `0x7e00`, a linear
//!   sample holds `0x7e00` and reads no class; under the default none does.
//!
//! Every module validates, and the plan's word estimate bounds its emission. `llvm-as` assembles
//! the kernels; without it the tests skip visibly.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::budget::SPIRV_WORDS;
use crate::device::profile::DeviceProfile;
use crate::device::{DevicePolicy, NanMinMaxPolicy};
use crate::library::FunctionConstantValues;
use crate::lower::select::{LowerInputs, Plan};
use crate::metal::texel_format::{texel_format, TexelFormat};
use crate::metal::PixelFormat;
use crate::pipeline::compile::{compile_bitcode, plan_bitcode, EntryRef};
use crate::pipeline::facts::RuntimeFact;
use crate::pipeline::sym::{Axis, ResourceKind, ResourceSlot, SlotClass, SlotRef, SpecKey, Stage};
use crate::pipeline::testing::{
    link_and_validate, llvm_assemble, meter, profile, typed_module_with, AssembleError,
};
use crate::spirv::interp::{run_with_images, Buffer, Config, Image, Order, Resources};
use crate::spirv::ir::{GlobalKind, Inst, TypedModule};

const HEADER: &str = "target datalayout = \"e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024-n8:16:32\"\ntarget triple = \"air64_v28-apple-macosx26.0.0\"\n\n";

/// The texture's side: 4 x 4 texels, one thread each.
const SIDE: u32 = 4;

const TEXTURE: SlotRef = SlotRef {
    stage: Stage::Kernel,
    class: SlotClass::Texture,
    index: 0,
};

/// Bitcode of the module `body` (after the AIR header), or `None` without `llvm-as`.
fn assemble(name: &str, body: &str) -> Option<Vec<u8>> {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let dir = std::env::temp_dir().join(format!(
        "m2v-texel-nans-{}-{}-{}",
        std::process::id(),
        NEXT.fetch_add(1, Ordering::Relaxed),
        name.replace(' ', "-")
    ));
    std::fs::create_dir_all(&dir).expect("scratch");
    let ll = dir.join("k.ll");
    std::fs::write(&ll, format!("{HEADER}{body}")).expect("write");
    let r = llvm_assemble(&ll);
    let _ = std::fs::remove_dir_all(&dir);
    match r {
        Ok(x) => Some(x),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("{name}: {why}"),
    }
}

/// How the kernel accesses `texture2d<half>` slot 0 at its thread's texel `gid`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Access {
    /// `t.read(gid)` of an `access::read` texture: one texel.
    Read,
    /// `t.sample(s, coord)` at the texel's centre through a nearest constexpr sampler: one texel.
    Nearest,
    /// `t.sample(s, coord)` at the texel's centre through a linear constexpr sampler: filtered.
    Linear,
    /// `t.gather(s, coord)` (component 0) through a nearest constexpr sampler: one texel a lane.
    Gather,
}

impl Access {
    /// Whether Metal hands back a 16-bit float texel's own NaN for the access.
    fn one_texel(self) -> bool {
        self != Access::Linear
    }
}

/// The words of a constexpr sampler of normalized coordinates, every axis clamped to its edge,
/// no mip filter, min and mag filter `filter` (0 nearest, 1 linear).
fn sampler_words(filter: u64) -> u64 {
    const BASE: u64 = 0x007b_ff00_0008_0000;
    let edge = 1;
    BASE | edge | edge << 3 | edge << 6 | filter << 9 | filter << 11
}

/// A kernel accessing slot 0 as `access` at thread `gid` of the `SIDE` x `SIDE` grid, storing
/// the `half4` to `out[gid.x + SIDE gid.y]` (buffer 0).
fn kernel(access: Access) -> String {
    let (vec4, suffix) = ("<4 x half>", "v4f16");
    let (tex_access, call, decl) = match access {
        Access::Read => (
            "read",
            format!(
                "%rs = call ptr addrspace(2) @air.get_read_sampler()\n  %r = call {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) %rs, <2 x i32> %gid, <2 x i32> zeroinitializer, i32 0, i32 1)"
            ),
            format!(
                "declare ptr addrspace(2) @air.get_read_sampler()\ndeclare {{ {vec4}, i8 }} @air.read_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x i32>, <2 x i32>, i32, i32)"
            ),
        ),
        Access::Nearest | Access::Linear => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i1 false, float 0.0, float 0.0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.sample_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i1, float, float, i32)"
            ),
        ),
        Access::Gather => (
            "sample",
            format!(
                "%r = call {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1) %t, ptr addrspace(2) @__air_sampler_state, <2 x float> %uv, i1 true, <2 x i32> zeroinitializer, i32 0, i32 0)"
            ),
            format!(
                "declare {{ {vec4}, i8 }} @air.gather_texture_2d.{suffix}(ptr addrspace(1), ptr addrspace(2), <2 x float>, i1, <2 x i32>, i32, i32)"
            ),
        ),
    };
    let filter = u64::from(access == Access::Linear);
    let (state, states, state_md) = match access {
        Access::Read => (String::new(), "", ""),
        Access::Nearest | Access::Linear | Access::Gather => (
            format!(
                "@__air_sampler_state = internal addrspace(2) constant [2 x i64] [i64 {}, i64 0], align 8\n\n",
                sampler_words(filter)
            ),
            "!air.sampler_states = !{!6}\n",
            "!6 = !{!\"air.sampler_state\", ptr addrspace(2) @__air_sampler_state}\n",
        ),
    };
    let scale = 1.0 / f64::from(SIDE);
    format!(
        r#"{state}define void @k(ptr addrspace(1) %t, ptr addrspace(1) %out, <2 x i32> %gid) {{
  %x = extractelement <2 x i32> %gid, i64 0
  %y = extractelement <2 x i32> %gid, i64 1
  %row = mul i32 %y, {SIDE}
  %n = add i32 %x, %row
  %i = zext i32 %n to i64
  %fx = uitofp i32 %x to float
  %fy = uitofp i32 %y to float
  %cx = fadd float %fx, 0.5
  %cy = fadd float %fy, 0.5
  %c0 = insertelement <2 x float> zeroinitializer, float %cx, i64 0
  %c1 = insertelement <2 x float> %c0, float %cy, i64 1
  %uv = fmul <2 x float> %c1, <float {scale}, float {scale}>
  {call}
  %v = extractvalue {{ {vec4}, i8 }} %r, 0
  %q = getelementptr inbounds {vec4}, ptr addrspace(1) %out, i64 %i
  store {vec4} %v, ptr addrspace(1) %q, align 8
  ret void
}}

{decl}

!air.kernel = !{{!0}}
{states}!0 = !{{ptr @k, !1, !2}}
!1 = !{{}}
!2 = !{{!3, !4, !5}}
!3 = !{{i32 0, !"air.texture", !"air.location_index", i32 0, i32 1, !"air.{tex_access}", !"air.arg_type_name", !"texture2d<half, {tex_access}>", !"air.arg_name", !"t"}}
!4 = !{{i32 1, !"air.buffer", !"air.location_index", i32 0, i32 1, !"air.read_write", !"air.address_space", i32 1, !"air.arg_type_size", i32 8, !"air.arg_type_align_size", i32 8, !"air.arg_type_name", !"half4", !"air.arg_name", !"out"}}
!5 = !{{i32 2, !"air.thread_position_in_grid", !"air.arg_type_name", !"uint2", !"air.arg_name", !"gid"}}
{state_md}"#
    )
}

/// The `f32` bits of half `h` widened exactly, as a device that decodes a half-float texel
/// exactly hands it over: a NaN keeps its sign and its payload at the top of the significand.
fn widened(h: u16) -> u32 {
    let sign = u32::from(h & 0x8000) << 16;
    let exp = u32::from(h >> 10 & 0x1F);
    let man = u32::from(h & 0x3FF);
    match exp {
        0x1F => sign | 0x7F80_0000 | man << 13,
        0 if man == 0 => sign,
        0 => {
            // A half subnormal: a normal `f32`.
            let shift = man.leading_zeros() - 21;
            sign | (113 - shift) << 23 | (man << shift & 0x3FF) << 13
        }
        _ => sign | (exp + 112) << 23 | man << 13,
    }
}

/// Whether half bits `h` are a NaN.
fn is_nan(h: u16) -> bool {
    h & 0x7C00 == 0x7C00 && h & 0x3FF != 0
}

/// The texels of the uniform texture, four channels each: quiet and signalling NaNs of both
/// signs and payload edges, among finite values and infinities (the probe's halves).
const TEXELS: [[u16; 4]; 4] = [
    [0x7e00, 0x7c01, 0x3c00, 0xfe00],
    [0xfc01, 0x7dff, 0x7fff, 0x0001],
    [0xffff, 0x7d55, 0x7c00, 0xfe01],
    [0x7f00, 0x8000, 0xff00, 0x7bff],
];

/// The module's texture resource at slot 0 (a combined binding's, for a sampled slot).
fn resource(m: &TypedModule) -> ResourceSlot {
    m.globals()
        .iter()
        .find_map(|g| match g.kind {
            GlobalKind::Resource(s)
                if s.index == 0
                    && matches!(
                        s.kind,
                        ResourceKind::Texture | ResourceKind::CombinedTextureSampler(_)
                    ) =>
            {
                Some(s)
            }
            _ => None,
        })
        .expect("the texture's resource")
}

/// Compile the kernel `bc` on `p`, check that the plan's word estimate bounds the emission, and
/// validate the linked module: the emitted words.
fn validate(bc: &[u8], p: &DeviceProfile, what: &str) -> u64 {
    let consts = FunctionConstantValues::defaults();
    let entry = EntryRef::Name("k");
    let inputs = LowerInputs::new();
    let plan = plan_bitcode(bc, entry, &consts, p, &inputs, &mut meter())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let a = compile_bitcode(bc, entry, &consts, p, &inputs, &mut meter(), SPIRV_WORDS)
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let words = a.module.words.len() as u64;
    assert!(
        plan.estimate.words >= words,
        "{what}: estimate {} < emitted {words}",
        plan.estimate.words
    );
    link_and_validate(&a, p).unwrap_or_else(|e| panic!("{what}: {e}"));
    words
}

/// Resources of a dispatch of one-thread threadgroups over the grid, the slot's class `class`.
fn resources(plan: &Plan, class: u32) -> Resources {
    let mut spec = BTreeMap::new();
    for axis in [Axis::X, Axis::Y, Axis::Z] {
        spec.insert(SpecKey::LocalSize(axis), 1);
    }
    spec.insert(SpecKey::FlatLocalSize, 1);
    spec.insert(SpecKey::TexelFormat(TEXTURE), u64::from(class));
    let push = vec![0u8; plan.push.push_range.map_or(0, |(_, end)| end as usize)];
    let out = Buffer {
        slot: Some(ResourceSlot {
            stage: Stage::Kernel,
            kind: ResourceKind::Buffer,
            index: 0,
        }),
        address: None,
        bytes: vec![0xab; (8 * SIDE * SIDE) as usize],
    };
    Resources {
        buffers: vec![out],
        push,
        spec,
    }
}

/// The `half4` every thread of kernel `bc` on `p` stores over the uniform texture of `texel`
/// with the slot's class `class`, in both scheduling orders, which must agree.
fn run(bc: &[u8], p: &DeviceProfile, texel: [u16; 4], class: u32, what: &str) -> Vec<[u16; 4]> {
    run_words(bc, p, texel.map(widened), class, what)
}

/// [`run`] over a texture whose texel the device decodes to the `f32` words `texel`.
fn run_words(
    bc: &[u8],
    p: &DeviceProfile,
    texel: [u32; 4],
    class: u32,
    what: &str,
) -> Vec<[u16; 4]> {
    let (plan, m) = typed_module_with(bc, "k", p, &LowerInputs::new())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    let mut img = Image::new(resource(&m), [SIDE, SIDE, 1], 1, 1, 1);
    for n in 0..SIDE * SIDE {
        let at = img.index(0, 0, 0, [n % SIDE, n / SIDE, 0]).expect("texel");
        img.texels[at] = texel;
    }
    let res = resources(&plan, class);
    let mut outs = Vec::new();
    for order in [Order::Forward, Order::Reverse] {
        let cfg = Config {
            workgroups: [SIDE, SIDE, 1],
            order,
            ..Config::default()
        };
        let (o, _) = run_with_images(&m, &cfg, res.clone(), vec![img.clone()])
            .unwrap_or_else(|e| panic!("{what}: interpreter: {e:?}"));
        outs.push(o.buffers.into_iter().next().expect("out"));
    }
    assert_eq!(outs[0], outs[1], "{what}: the scheduling orders disagree");
    outs[0]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|t| {
            let h = |i: usize| u16::from_le_bytes([t[2 * i], t[2 * i + 1]]);
            [h(0), h(1), h(2), h(3)]
        })
        .collect()
}

/// The `half` lanes Metal hands back for `texel` through `access` from a texture whose channels
/// are 16-bit floats (`halves`) or 32-bit floats: a gather's lanes are each its texel's red.
fn metal(access: Access, texel: [u16; 4], halves: bool) -> [u16; 4] {
    let lanes = match access {
        Access::Gather => [texel[0]; 4],
        _ => texel,
    };
    lanes.map(|h| match is_nan(h) && !(halves && access.one_texel()) {
        true => 0x7e00,
        false => h,
    })
}

/// The profiles of lavapipe under the NaN switch (`NanMinMaxPolicy::Exact`) and the default.
fn profiles() -> (DeviceProfile, DeviceProfile) {
    let switched = DevicePolicy {
        nan_and_min_max: NanMinMaxPolicy::Exact,
        ..DevicePolicy::default()
    };
    (
        profile("lavapipe", switched).expect("profile"),
        profile("lavapipe", DevicePolicy::default()).expect("profile"),
    )
}

/// Whether `plan` reads texture slot 0's class (`RuntimeFact::TexelFormat`, read as data).
fn reads_class(plan: &Plan) -> bool {
    plan.facts
        .contains(&RuntimeFact::TexelFormat { slot: TEXTURE })
}

/// The constants of module `m` that are `0x7fc00000` in every lane: Metal's `0x7e00` held.
fn canonical_nans(m: &TypedModule) -> usize {
    use crate::spirv::ir::{ConstKind, Constant};
    let canonical = |bits: u64| bits == 0x7fc0_0000;
    m.consts()
        .iter()
        .filter(|(_, c)| match &c.kind {
            ConstKind::Plain(Constant::Composite { parts, .. }) => parts
                .iter()
                .all(|p| m.consts().scalar_bits(*p).is_some_and(|b| canonical(b.0))),
            _ => false,
        })
        .count()
}

/// A read on the interpreter: under `NanMinMaxPolicy::Exact` every NaN texel of a 16-bit float
/// texture (or of a class whose conversion is not measured, `TexelFormat::OTHER`, the device's
/// decode) reads back as its own bits and of a 32-bit float texture as `0x7e00`, every other lane
/// its value; under the default (`NanMinMaxPolicy::Native`) the module reads no class and is
/// smaller, and a signalling NaN texel does not read back as its bits.
#[test]
fn a_half_texel_reads_metals_nan_under_the_nan_switch() {
    let (exact, native) = profiles();
    let half16 = texel_format(PixelFormat::RGBA16Float).0;
    let float32 = texel_format(PixelFormat::RGBA32Float).0;
    let access = Access::Read;
    let what = format!("{access:?}");
    let Some(bc) = assemble(&what, &kernel(access)) else {
        return;
    };
    let words_exact = validate(&bc, &exact, &format!("{what} exact"));
    let words_native = validate(&bc, &native, &format!("{what} native"));
    assert!(
        words_native < words_exact,
        "{what}: the default's {words_native} words, the switch's {words_exact}"
    );
    let plan = |p: &DeviceProfile| {
        typed_module_with(&bc, "k", p, &LowerInputs::new())
            .unwrap_or_else(|e| panic!("{what}: {e}"))
            .0
    };
    assert!(
        reads_class(&plan(&exact)),
        "{what}: the switch reads no class"
    );
    assert!(
        !reads_class(&plan(&native)),
        "{what}: the default reads a class"
    );
    let mut native_differs = false;
    for texel in TEXELS {
        // A class whose conversion is not measured keeps the device's decode, here exact.
        let other = TexelFormat::OTHER.0;
        for (class, halves) in [(half16, true), (float32, false), (other, true)] {
            let at = format!("{what} {texel:04x?} halves {halves}");
            let want = metal(access, texel, halves);
            for got in run(&bc, &exact, texel, class, &at) {
                assert_eq!(got, want, "{at}: got {got:04x?}, Metal {want:04x?}");
            }
            if halves {
                native_differs |= run(&bc, &native, texel, class, &at)
                    .iter()
                    .any(|got| *got != want);
            }
        }
    }
    assert!(
        native_differs,
        "{what}: the default answered Metal's NaNs too"
    );
}

/// NaNs of a 32-bit float texel: signalling NaNs of the smallest payload, the quiet NaN, and
/// NaNs whose payload's top bits a narrowing to `half` would keep.
const FLOAT_NANS: [u32; 6] = [
    0x7f80_0001,
    0x7fc0_0000,
    0xff80_0001,
    0x7fbf_ffff,
    0x7fc1_2345,
    0xffff_ffff,
];

/// A read of a 32-bit float texel on the interpreter, on lavapipe's and the RTX 5080's profiles:
/// under `NanMinMaxPolicy::Exact` its every NaN, in any lane, signalling NaNs of the smallest
/// payload included, reads back as `0x7e00`, beside a `1.0` and beside the largest `float`, which
/// rounds to infinity. (On the devices themselves the second differs: item 5.)
#[test]
fn a_32_bit_float_texels_nans_read_metals_canonical_nan_in_every_lane() {
    let float32 = texel_format(PixelFormat::RGBA32Float).0;
    let access = Access::Read;
    let what = format!("{access:?}");
    let Some(bc) = assemble(&what, &kernel(access)) else {
        return;
    };
    let profiles = ["lavapipe", "nvidia-rtx5080-laptop"].map(|stem| {
        let switched = DevicePolicy {
            nan_and_min_max: NanMinMaxPolicy::Exact,
            ..DevicePolicy::default()
        };
        (stem, profile(stem, switched).expect("profile"))
    });
    // Beside a `1.0`, and beside the largest `float`, which overflows a `half`.
    for (beside, rounded) in [(0x3f80_0000, 0x3c00), (0x7f7f_ffff, 0x7c00)] {
        for nan in FLOAT_NANS {
            for lane in 0..4 {
                let mut texel = [beside; 4];
                texel[lane] = nan;
                let at = format!("{what} {texel:08x?}");
                for (stem, p) in &profiles {
                    for got in run_words(&bc, p, texel, float32, &at) {
                        let mut want = [rounded; 4];
                        want[lane] = 0x7e00;
                        assert_eq!(got, want, "{stem} {at}: got {got:04x?}, Metal {want:04x?}");
                    }
                }
            }
        }
    }
}

/// The image fetches and gathers of module `m`.
fn fetches_and_gathers(m: &TypedModule) -> (usize, usize) {
    let insts = || m.blocks().iter().flat_map(|b| b.insts.iter());
    let fetches = insts()
        .filter(|n| matches!(n.inst, Inst::ImageFetch { .. }))
        .count();
    let gathers = insts()
        .filter(|n| matches!(n.inst, Inst::ImageGather { .. }))
        .count();
    (fetches, gathers)
}

/// A gather under the NaN switch: where the bound class is a 16-bit float one the shader fetches
/// the four texels Metal's gather reads, so each lane is its texel's own bits, NaNs included, on
/// the interpreter (which models fetches, not a device's gather); any other class keeps the
/// device's gather. Under the default the module fetches nothing.
#[test]
fn a_half_gather_fetches_its_16_bit_texels_under_the_nan_switch() {
    let (exact, native) = profiles();
    let half16 = texel_format(PixelFormat::RGBA16Float).0;
    let access = Access::Gather;
    let what = format!("{access:?}");
    let Some(bc) = assemble(&what, &kernel(access)) else {
        return;
    };
    validate(&bc, &exact, &format!("{what} exact"));
    validate(&bc, &native, &format!("{what} native"));
    let module = |p: &DeviceProfile| {
        typed_module_with(&bc, "k", p, &LowerInputs::new())
            .unwrap_or_else(|e| panic!("{what}: {e}"))
            .1
    };
    let (fetches, gathers) = fetches_and_gathers(&module(&exact));
    assert_eq!(
        (fetches, gathers),
        (4, 1),
        "{what}: the switch's fetches and gather"
    );
    let (fetches, gathers) = fetches_and_gathers(&module(&native));
    assert_eq!(
        (fetches, gathers),
        (0, 1),
        "{what}: the default's fetches and gather"
    );
    for texel in TEXELS {
        let at = format!("{what} {texel:04x?}");
        let want = metal(access, texel, true);
        for got in run(&bc, &exact, texel, half16, &at) {
            assert_eq!(got, want, "{at}: got {got:04x?}, Metal {want:04x?}");
        }
    }
}

/// Samples and gathers (the interpreter models no device sampler): under `NanMinMaxPolicy::Exact` a
/// nearest sample and a gather read the slot's class and hold Metal's `0x7e00`, a linear sample,
/// which never hands back a texel's own NaN, holds `0x7e00` and reads no class; under the default
/// each reads no class, holds no `0x7e00` and is smaller; every module validates and the plan's
/// estimate bounds it.
#[test]
fn a_half_texels_sample_and_gather_take_metals_nan_under_the_nan_switch() {
    let (exact, native) = profiles();
    for access in [Access::Nearest, Access::Linear, Access::Gather] {
        let what = format!("{access:?}");
        let Some(bc) = assemble(&what, &kernel(access)) else {
            return;
        };
        let words_exact = validate(&bc, &exact, &format!("{what} exact"));
        let words_native = validate(&bc, &native, &format!("{what} native"));
        assert!(
            words_native < words_exact,
            "{what}: the default's {words_native} words, the switch's {words_exact}"
        );
        let typed = |p: &DeviceProfile| {
            typed_module_with(&bc, "k", p, &LowerInputs::new())
                .unwrap_or_else(|e| panic!("{what}: {e}"))
        };
        let (exact_plan, exact_module) = typed(&exact);
        let (native_plan, native_module) = typed(&native);
        assert_eq!(
            reads_class(&exact_plan),
            access.one_texel(),
            "{what}: the switch's class"
        );
        assert!(
            !reads_class(&native_plan),
            "{what}: the default reads a class"
        );
        assert!(
            canonical_nans(&exact_module) > 0,
            "{what}: the switch holds no 0x7e00"
        );
        assert_eq!(
            canonical_nans(&native_module),
            0,
            "{what}: the default holds 0x7e00"
        );
    }
}

/// On the MoltenVK captures, whose arithmetic answers Metal's canonical NaN
/// (`NanEncoding::MetalCanonical`), a texel's NaN is still the image's decode, which the access
/// narrows itself (through MoltenVK on the Apple M3 Max a 32-bit float texel's NaN read as `half`
/// kept its payload's top bits). So under `NanMinMaxPolicy::Exact` the access takes Metal's rule
/// there too (`TexelNanRoute::Metal`): a read, on the interpreter, hands back a 16-bit float
/// texel's NaN as its own bits and every NaN of a 32-bit float texel, in any lane, as `0x7e00`,
/// where under the default some such NaN reads back otherwise; a read, a nearest sample and a
/// gather read the slot's class, and every access holds more `0x7e00` than under the default (the
/// default's half texel conversion reads the class too, so its reading proves nothing there).
/// Every module validates and the plan's estimate bounds it.
#[test]
fn a_half_texels_nan_is_metals_under_the_nan_switch_where_the_arithmetic_answers_metals() {
    use crate::device::profile::{NanEncoding, TexelNanRoute};
    let half16 = texel_format(PixelFormat::RGBA16Float).0;
    let float32 = texel_format(PixelFormat::RGBA32Float).0;
    let switched = DevicePolicy {
        nan_and_min_max: NanMinMaxPolicy::Exact,
        ..DevicePolicy::default()
    };
    for stem in ["moltenvk-m3", "moltenvk-m4"] {
        let exact = profile(stem, switched).expect("profile");
        let native = profile(stem, DevicePolicy::default()).expect("profile");
        for p in [&exact, &native] {
            assert_eq!(
                p.fields().nan_encoding,
                NanEncoding::MetalCanonical,
                "{stem}"
            );
        }
        assert_eq!(exact.fields().texel_nans, TexelNanRoute::Metal, "{stem}");
        assert_eq!(native.fields().texel_nans, TexelNanRoute::Device, "{stem}");
        for access in [
            Access::Read,
            Access::Nearest,
            Access::Linear,
            Access::Gather,
        ] {
            let what = format!("{access:?} on {stem}");
            let Some(bc) = assemble(&what, &kernel(access)) else {
                return;
            };
            validate(&bc, &exact, &format!("{what} exact"));
            validate(&bc, &native, &format!("{what} native"));
            let typed = |p: &DeviceProfile| {
                typed_module_with(&bc, "k", p, &LowerInputs::new())
                    .unwrap_or_else(|e| panic!("{what}: {e}"))
            };
            let (exact_plan, exact_module) = typed(&exact);
            let (_, native_module) = typed(&native);
            if access.one_texel() {
                assert!(
                    reads_class(&exact_plan),
                    "{what}: the switch reads no class"
                );
            }
            let (held, held_by_default) = (
                canonical_nans(&exact_module),
                canonical_nans(&native_module),
            );
            assert!(
                held > held_by_default,
                "{what}: the switch holds {held} 0x7e00, the default {held_by_default}"
            );
            if access != Access::Read {
                continue;
            }
            for texel in TEXELS {
                for (class, halves) in [(half16, true), (float32, false)] {
                    let at = format!("{what} {texel:04x?} halves {halves}");
                    let want = metal(access, texel, halves);
                    for got in run(&bc, &exact, texel, class, &at) {
                        assert_eq!(got, want, "{at}: got {got:04x?}, Metal {want:04x?}");
                    }
                }
            }
            let mut native_differs = false;
            for nan in FLOAT_NANS {
                for lane in 0..4 {
                    let mut texel = [0x3f80_0000; 4];
                    texel[lane] = nan;
                    let at = format!("{what} {texel:08x?}");
                    let mut want = [0x3c00; 4];
                    want[lane] = 0x7e00;
                    for got in run_words(&bc, &exact, texel, float32, &at) {
                        assert_eq!(got, want, "{at}: got {got:04x?}, Metal {want:04x?}");
                    }
                    native_differs |= run_words(&bc, &native, texel, float32, &at)
                        .iter()
                        .any(|got| *got != want);
                }
            }
            assert!(
                native_differs,
                "{what}: the default answered Metal's NaNs too"
            );
        }
    }
}
