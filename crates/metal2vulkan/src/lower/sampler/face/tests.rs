//! `metal_ties` on the interpreter against Metal's rule. Each kernel invocation reads a direction's
//! bits and writes the bits of the direction the device is handed. `snap` checks `snapped`, and
//! `fold` a linear cube filter's taps across the faces' edges. `anisotropic` checks
//! `emit::cube_footprint`'s gradients through a constexpr anisotropic sampler.

use super::metal_ties;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn storage_buffer(b: &mut ModuleBuilder, index: u32, elem: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

/// Word `k` of invocation `i`'s `per` in `buf`.
fn word(
    f: &mut FunctionBuilder<'_>,
    buf: GlobalId,
    (i, per): (Value, u32),
    k: u32,
) -> BResult<Value> {
    let per = f.u32_const(per)?;
    let per = f.module().const_value(per)?;
    let at = f.binary(BinOp::IMul, i, per)?;
    let k = f.u32_const(k)?;
    let k = f.module().const_value(k)?;
    let at = f.binary(BinOp::IAdd, at, k)?;
    let zero = f.u32_const(0)?;
    let zero = f.module().const_value(zero)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads a direction's three bit patterns from buffer 0 and writes
/// those of `metal_ties`'s direction to buffer 1 (`ties`), or of the direction itself.
fn kernel(ties: bool) -> BResult<TypedModule> {
    kernel_on(ModuleBuilder::new(), 3, &|f, p, _| match ties {
        true => metal_ties(f, p),
        false => Ok(p),
    })
}

/// How a kernel hands a direction on: from the direction and the record's word after it.
type Hand<'a> = &'a dyn Fn(&mut FunctionBuilder<'_>, Value, Value) -> BResult<Value>;

/// A kernel of module `b` whose invocation `i` reads a record of `per` words from buffer 0, a
/// direction's three bit patterns and, past them, a `u32`, and writes the bits of `hand`'s
/// direction to buffer 1.
fn kernel_on(mut b: ModuleBuilder, per: u32, hand: Hand<'_>) -> BResult<TypedModule> {
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let vec3 = b.ty(Type::Vector { elem: f32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let mut c = Vec::new();
    for k in 0..3 {
        let at = word(&mut f, input, (i, per), k)?;
        let bits = f.load(at, MemAccess::default())?;
        c.push(f.convert(ConvOp::Bitcast, bits, f32_)?);
    }
    let p = f.composite_construct(vec3, &c)?;
    let extra = match per {
        3 => {
            let zero = f.u32_const(0)?;
            f.module().const_value(zero)?
        }
        _ => {
            let at = word(&mut f, input, (i, per), 3)?;
            f.load(at, MemAccess::default())?
        }
    };
    let q = hand(&mut f, p, extra)?;
    for k in 0..3 {
        let x = f.composite_extract(q, &[k])?;
        let bits = f.convert(ConvOp::Bitcast, x, u32_)?;
        let at = word(&mut f, output, (i, 3), k)?;
        f.store(at, bits, MemAccess::default())?;
    }
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The kernel's direction per lane.
fn run(lanes: &[[f32; 3]]) -> Vec<[f32; 3]> {
    let records: Vec<u32> = lanes.iter().flat_map(|d| d.map(f32::to_bits)).collect();
    run_records(&kernel(true).expect("kernel"), &records, lanes.len())
}

/// Module `m`'s direction per lane of `lanes` lanes, from the records `records`.
fn run_records(m: &TypedModule, records: &[u32], lanes: usize) -> Vec<[f32; 3]> {
    let bytes: Vec<u8> = records.iter().flat_map(|x| x.to_le_bytes()).collect();
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0xAB; lanes * 12],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<12>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                f32::from_bits(u32::from_le_bytes([
                    c[4 * k],
                    c[4 * k + 1],
                    c[4 * k + 2],
                    c[4 * k + 3],
                ]))
            })
        })
        .collect()
}

/// The magnitudes Metal selects a face by: a subnormal one as zero.
fn selecting(d: [f32; 3]) -> [f32; 3] {
    d.map(|x| {
        if x.abs() < f32::MIN_POSITIVE {
            0.0
        } else {
            x.abs()
        }
    })
}

/// Metal's major axis of the magnitudes `m`: `x` over `y` over `z` at equal
/// magnitudes.
fn metal_axis(m: [f32; 3]) -> usize {
    if m[0] >= m[1] && m[0] >= m[2] {
        0
    } else if m[1] >= m[2] {
        1
    } else {
        2
    }
}

/// `x` one ulp smaller in magnitude, its sign kept.
fn shrunk(x: f32) -> f32 {
    f32::from_bits(x.to_bits() - 1)
}

/// `2^-125`, below which Metal reads a face coordinate's component as zero.
const LEAST_READ: f32 = f32::from_bits(0x0100_0000);

/// `2^126`, past which (as a major magnitude) Metal reads every face coordinate as zero.
const MOST_READ: f32 = f32::from_bits(0x7e80_0000);

/// `2^125`, the largest major magnitude the device is handed.
const MOST_DIVIDED: f32 = f32::from_bits(0x7e00_0000);

/// What a zero direction is handed: face `+x`, texel `(0, 0)`.
const ZERO_DIRECTION: [f32; 3] = [
    1.0,
    f32::from_bits(0x3f7f_ffff),
    f32::from_bits(0x3f7f_ffff),
];

/// Whether Metal reads minor component `c` beside major magnitude `top` as zero.
fn read_as_zero(c: f32, top: f32) -> bool {
    c.abs() < LEAST_READ || top > MOST_READ
}

/// `x` one ulp larger in magnitude, its sign kept.
fn grown(x: f32) -> f32 {
    f32::from_bits(x.to_bits() + 1)
}

/// The direction Metal's rule hands the device: one with an infinite or NaN component as it is;
/// a direction left zero `ZERO_DIRECTION`; else each minor component `+0` where Metal reads it as
/// zero, and the ties broken: each minor component tying the major magnitude one ulp smaller in
/// magnitude, or, at a tie at `2^-125` itself, the major component one ulp larger instead; every
/// other component as it is; then the whole scaled by `1/2` where the major magnitude is above
/// `2^125`, or by `1/8` where it is above `2^126`.
fn reference(d: [f32; 3]) -> [f32; 3] {
    if d.iter().any(|x| !x.is_finite()) {
        return d;
    }
    let m = selecting(d);
    let major = metal_axis(m);
    let top = m[major];
    if top == 0.0 {
        return ZERO_DIRECTION;
    }
    let tied = |k: usize| k != major && !read_as_zero(d[k], top) && d[k].abs() == top;
    let grow = top == LEAST_READ && (0..3).any(tied);
    let scale = match top {
        t if t > MOST_READ => 0.125,
        t if t > MOST_DIVIDED => 0.5,
        _ => 1.0,
    };
    std::array::from_fn(|k| {
        let c = if k == major {
            match grow {
                true => grown(d[k]),
                false => d[k],
            }
        } else if read_as_zero(d[k], top) {
            0.0
        } else if tied(k) && !grow {
            shrunk(d[k])
        } else {
            d[k]
        };
        c * scale
    })
}

/// A face and its two coordinates, each minor axis's component over the major magnitude.
type Reading = (usize, bool, [(usize, f64); 2]);

/// What Apple's texture unit reads along a finite direction `d`: the major axis by `x` over `y`
/// over `z` on the magnitudes with a subnormal one read as zero, and each minor component over the
/// major magnitude, zero where Metal reads it as zero; a direction left zero at face `+x`'s texel
/// `(0, 0)`, where `y` and `z` read `+1`.
fn metal_reading(d: [f32; 3]) -> Reading {
    let m = selecting(d);
    let major = metal_axis(m);
    let top = m[major];
    if top == 0.0 {
        return (0, true, [(1, 1.0), (2, 1.0)]);
    }
    let coordinate = |k: usize| match read_as_zero(d[k], top) {
        true => 0.0,
        false => f64::from(d[k]) / f64::from(top),
    };
    let minors = [(major + 1) % 3, (major + 2) % 3];
    (major, d[major] > 0.0, minors.map(|k| (k, coordinate(k))))
}

/// What a Vulkan device reads along a finite direction `q` with one largest magnitude, in exact
/// arithmetic: that axis's face, and each minor component over its magnitude.
fn device_reading(q: [f32; 3]) -> Reading {
    let a = q.map(f32::abs);
    let major = metal_axis(a);
    let minors = [(major + 1) % 3, (major + 2) % 3];
    for k in minors {
        assert!(a[k] < a[major], "{q:?}: axis {k} ties {major}");
    }
    let top = f64::from(a[major]);
    (
        major,
        q[major] > 0.0,
        minors.map(|k| (k, f64::from(q[k]) / top)),
    )
}

/// Whether readings `want` and `got` name the same face and, on every size in `SIZES`, the same
/// texel (nearest, clamped) and 1/256 position (`floor(256 X + 1/2)`) on each axis.
fn same_texels(want: Reading, got: Reading) -> bool {
    let position = |c: f64, n: u32| (c + 1.0) / 2.0 * f64::from(n);
    want.0 == got.0
        && want.1 == got.1
        && want.2.iter().zip(got.2).all(|((k, w), (j, g))| {
            *k == j
                && SIZES.iter().all(|n| {
                    let (w, g) = (position(*w, *n), position(g, *n));
                    let texel = |x: f64| x.floor().clamp(0.0, f64::from(n - 1));
                    let fixed = |x: f64| (256.0 * x + 0.5).floor();
                    texel(w) == texel(g) && fixed(w) == fixed(g)
                })
        })
}

/// The magnitudes the lanes are built from: ordinary ones; both ends of the normal range and the
/// floats around `2^-125`, `2^125`, `2^126`, and `2^127`; subnormal, infinite, and large ones.
const MAGNITUDES: [f32; 22] = [
    1.0,
    0.885,
    0.5,
    3.0,
    f32::MIN_POSITIVE,
    f32::from_bits(0x0080_0001),
    f32::from_bits(0x00ff_ffff),
    LEAST_READ,
    f32::from_bits(0x0100_0001),
    f32::from_bits(1),
    f32::from_bits(0x0040_0000),
    MOST_DIVIDED,
    f32::from_bits(0x7e00_0001),
    f32::from_bits(0x7e40_0000),
    MOST_READ,
    f32::from_bits(0x7e80_0001),
    f32::from_bits(0x7f00_0000),
    f32::MAX,
    f32::INFINITY,
    1.0e-30,
    123_456.79,
    1.0e30,
];

/// Every pair and the triple of tied axes in every sign combination, with the third component at
/// several fractions of the magnitude, over `MAGNITUDES`; one-ulp controls; a component of each
/// magnitude beside a major one of each; and directions with no tie, a zero one, and NaN.
fn lanes() -> Vec<[f32; 3]> {
    let mut out = Vec::new();
    for m in MAGNITUDES {
        for (a, b, c) in [(0, 1, 2), (0, 2, 1), (1, 2, 0)] {
            for sa in [1.0f32, -1.0] {
                for sb in [1.0f32, -1.0] {
                    for t in [0.0f32, -0.0, 0.25, -0.25, 0.885, -0.885] {
                        let mut d = [0.0f32; 3];
                        d[a] = sa * m;
                        d[b] = sb * m;
                        d[c] = t * m;
                        out.push(d);
                        // The control: one tied component an ulp larger in magnitude.
                        if m.is_finite() && m < f32::MAX {
                            for k in [a, b] {
                                let mut e = d;
                                e[k] = f32::from_bits(e[k].to_bits() + 1);
                                out.push(e);
                            }
                        }
                    }
                }
            }
        }
        for s in 0..8u32 {
            let sign = |k: u32| if s >> k & 1 == 1 { -1.0f32 } else { 1.0 };
            out.push([sign(0) * m, sign(1) * m, sign(2) * m]);
        }
        for minor in MAGNITUDES {
            out.push([m, -minor, 0.5 * minor]);
            out.push([-minor, 0.0, m]);
        }
    }
    out.extend([
        [0.0, 0.0, 0.0],
        [-0.0, 0.0, -0.0],
        [0.3, -0.7, 0.2],
        [-0.1, 0.2, -0.9],
        [f32::NAN, 1.0, 1.0],
        [1.0, f32::NAN, 1.0],
        [-1.0, -0.885, -1.0],
    ]);
    out
}

/// The interpreter's direction equals the rule on every lane, bit for bit.
#[test]
fn the_direction_is_handed_as_metal_reads_it() {
    let lanes = lanes();
    let got = run(&lanes);
    let (mut ties, mut zeroed) = (0, 0);
    for (d, q) in lanes.iter().zip(&got) {
        let want = reference(*d);
        assert_eq!(
            q.map(f32::to_bits),
            want.map(f32::to_bits),
            "direction {d:?}: {q:?}, Metal's rule {want:?}"
        );
        for (x, w) in d.iter().zip(want) {
            ties += usize::from(*x != 0.0 && (w == shrunk(*x) || w == grown(*x)));
            zeroed += usize::from(*x != 0.0 && w.to_bits() == 0);
        }
    }
    assert!(ties > 900, "{ties} components moved to break a tie");
    assert!(zeroed > 1000, "{zeroed} components read as zero");
}

/// Metal's largest cube face and a range of smaller ones, odd and even.
const SIZES: [u32; 9] = [1, 2, 3, 5, 8, 13, 255, 1024, 16_384];

/// On every finite direction, the direction handed to the device has one largest magnitude, Metal's
/// major axis, so no device's own tie order is consulted; and a device reading it in exact
/// arithmetic reads Metal's face, texel, and 1/256 position on every face size up to 16,384.
#[test]
fn every_device_reads_metals_face_and_texel() {
    let lanes = lanes();
    let got = run(&lanes);
    let mut checked = 0;
    for (d, q) in lanes.iter().zip(&got) {
        if d.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let (want, have) = (metal_reading(*d), device_reading(*q));
        assert!(
            same_texels(want, have),
            "direction {d:?} handed {q:?}: Metal {want:?}, the device {have:?}"
        );
        checked += usize::from(q != d);
    }
    assert!(checked > 2000, "{checked} changed directions");
}

/// Apple's texture unit (MoltenVK samples there) reads the handed direction by Metal's own rule and
/// still reads Metal's face, texel, and 1/256 position along the original: no shrunk component
/// falls below `2^-125` and no grown major rises past `2^126`.
#[test]
fn apples_texture_unit_reads_the_handed_direction_alike() {
    let lanes = lanes();
    let got = run(&lanes);
    for (d, q) in lanes.iter().zip(&got) {
        if d.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let (want, have) = (metal_reading(*d), metal_reading(*q));
        assert!(
            same_texels(want, have),
            "direction {d:?} handed {q:?}: Metal {want:?}, then {have:?}"
        );
    }
}

/// What a device reads along a finite direction `q` with one largest magnitude when it divides
/// by the reciprocal of the major magnitude doubled, in `f32`, with a subnormal reciprocal flushed
/// to zero, as AMD's cube instructions (`v_cubema` takes the major axis doubled) and RADV read
/// `(2^126, 2^125, 0)`: at the face's centre past a major magnitude of `2^125`, else exactly.
fn flushing_reading(q: [f32; 3]) -> Reading {
    let (major, positive, minors) = device_reading(q);
    let reciprocal = 1.0 / (2.0 * q[major].abs());
    match reciprocal < f32::MIN_POSITIVE {
        true => (major, positive, minors.map(|(k, _)| (k, 0.0))),
        false => (major, positive, minors),
    }
}

/// The device is handed no major magnitude above `2^125` (RADV misread `2^126`, ANV and NVIDIA
/// `2^127` and up, the probe lanes), and a device whose reciprocal of the doubled major
/// magnitude flushes reads Metal's face, texel, and 1/256 position along it, the directions whose
/// major magnitude is above `2^125` included.
#[test]
fn a_huge_major_is_scaled_below_the_devices_reach() {
    let lanes = lanes();
    let got = run(&lanes);
    let mut huge = 0;
    for (d, q) in lanes.iter().zip(&got) {
        if d.iter().any(|x| !x.is_finite()) {
            continue;
        }
        let top = q.map(f32::abs).into_iter().fold(0.0, f32::max);
        assert!(top <= MOST_DIVIDED, "direction {d:?} handed {q:?}");
        let (want, have) = (metal_reading(*d), flushing_reading(*q));
        assert!(
            same_texels(want, have),
            "direction {d:?} handed {q:?}: Metal {want:?}, the device {have:?}"
        );
        huge += usize::from(selecting(*d).into_iter().fold(0.0, f32::max) > MOST_DIVIDED);
    }
    assert!(
        huge > 400,
        "{huge} directions with a major magnitude above 2^125"
    );
}

/// A direction whose components are all zero or subnormal, in any signs, is handed as
/// `ZERO_DIRECTION`, which reads face `+x` (one major axis) at texel `(0, 0)` on every face size up
/// to 16,384, as Metal reads it; and a subnormal component beside a normal
/// major one is handed as `+0`.
#[test]
fn a_direction_left_zero_reads_metals_corner() {
    let tiny = [
        0.0,
        -0.0,
        f32::from_bits(1),
        -f32::from_bits(1),
        f32::from_bits(0x0040_0000),
        -f32::from_bits(0x007f_ffff),
    ];
    let mut lanes = Vec::new();
    for x in tiny {
        for y in tiny {
            for z in tiny {
                lanes.push([x, y, z]);
                lanes.push([x, 0.5, z]);
            }
        }
    }
    let got = run(&lanes);
    for (d, q) in lanes.iter().zip(&got) {
        if d[1] == 0.5 {
            let want = [0.0, 0.5, 0.0];
            assert_eq!(q.map(f32::to_bits), want.map(f32::to_bits), "{d:?}");
            continue;
        }
        assert_eq!(
            q.map(f32::to_bits),
            ZERO_DIRECTION.map(f32::to_bits),
            "{d:?}"
        );
        // Face `+x`, where `s_c = -z` and `t_c = -y`: texel `(0, 0)` reads `y` and `z` as `+1`.
        assert!(same_texels(metal_reading(*d), device_reading(*q)), "{d:?}");
        for n in SIZES {
            for c in [q[1], q[2]] {
                let t = (f64::from(-c) / f64::from(q[0]) + 1.0) / 2.0 * f64::from(n);
                assert_eq!(t.floor(), 0.0, "{d:?}, size {n}");
            }
        }
    }
}

/// A direction with an infinite or NaN component is handed bit for bit: no infinity becomes finite,
/// no NaN moves, and nothing beside them changes.
#[test]
fn infinities_and_nans_pass_unchanged() {
    let parts = [
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        -f32::NAN,
        f32::from_bits(0x7fc0_1234),
        f32::MAX,
        f32::MIN_POSITIVE,
        1.0,
        -0.0,
        f32::from_bits(3),
    ];
    let mut lanes = Vec::new();
    for x in parts {
        for y in parts {
            for z in parts {
                let d = [x, y, z];
                if d.iter().any(|c| !c.is_finite()) {
                    lanes.push(d);
                }
            }
        }
    }
    let got = run(&lanes);
    for (d, q) in lanes.iter().zip(&got) {
        assert_eq!(
            q.map(f32::to_bits),
            d.map(f32::to_bits),
            "{d:?} handed {q:?}"
        );
    }
    assert!(lanes.len() > 500, "{} lanes", lanes.len());
}

/// `TIE_WORDS` bounds the words `metal_ties` adds to a module, which is valid on every committed
/// profile.
#[test]
fn the_estimate_bounds_the_emitted_words() {
    let profiles = crate::spirv::testing::committed_profiles().expect("profiles");
    for (stem, profile) in &profiles {
        let with = crate::spirv::testing::assert_valid(kernel(true).expect("kernel"), profile);
        let without = crate::spirv::testing::assert_valid(kernel(false).expect("kernel"), profile);
        let added = with.len() - without.len();
        assert!(
            added <= super::TIE_WORDS as usize,
            "{stem}: metal_ties adds {added} words, estimated {}",
            super::TIE_WORDS
        );
    }
}

mod anisotropic;
mod fold;
mod routines;
mod snap;
