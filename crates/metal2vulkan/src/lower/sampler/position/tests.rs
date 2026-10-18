//! Metal's position and the snapped coordinates, run on the interpreter against the rule computed
//! exactly. Each kernel reads `(bits(u), n)` per invocation and writes the bits of [`position`]'s
//! `X` and of [`snapped_axis`]'s gather and nearest coordinates, for one addressing, and of its
//! gather and footprint-cell coordinates reading `u` as a pixel coordinate.
//!
//! The reference wraps `u` with the same `f32` operations Metal's rule names (they are exact or
//! rounded once, as on the device), forms the product `u' n` exactly in `f64` (at most 46
//! significant bits), and rounds it to `f32` with ties away from zero.

use super::{
    celled_by_word, gathered_by_word, offset_dropped, pixel_centre_by_word, position, snapped_axis,
    snapped_by_word, AxisSnap,
};
use crate::lower::sampler::bicubic::TapAddress;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{self, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass, Type,
    TypedModule,
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

fn u32_const(f: &mut FunctionBuilder<'_>, x: u32) -> BResult<Value> {
    let c = f.u32_const(x)?;
    f.module().const_value(c)
}

/// Word `k` of invocation `i`'s record of `per` words in `buf`.
fn word(f: &mut FunctionBuilder<'_>, buf: GlobalId, i: Value, per: u32, k: u32) -> BResult<Value> {
    let per = u32_const(f, per)?;
    let at = f.binary(crate::spirv::ir::BinOp::IMul, i, per)?;
    let k = u32_const(f, k)?;
    let at = f.binary(crate::spirv::ir::BinOp::IAdd, at, k)?;
    let zero = u32_const(f, 0)?;
    let base = f.module().global_value(buf)?;
    f.access_chain(base, &[zero, at])
}

/// A kernel whose invocation `i` reads `(bits(u), n)` from buffer 0 and writes `X`, the gather's
/// coordinate, the nearest sample's, and, reading `u` as a pixel coordinate, the gather's and the
/// footprint cell's, as bits, to buffer 1, for an axis addressed by `address` whose access the
/// device moves by the constant texel offset `offset` (0: none).
fn kernel(address: TapAddress, offset: i32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 2, 0)?;
    let ub = f.load(at, MemAccess::default())?;
    let u = f.convert(ConvOp::Bitcast, ub, f32_)?;
    let at = word(&mut f, input, i, 2, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let (x, _) = position(&mut f, u, n, address)?;
    let o = if offset == 0 {
        None
    } else {
        // Metal's offsets lie in [-8, 7], each exact as an `f32`.
        Some(super::f32_const(&mut f, offset as f32)?)
    };
    let at = (address, o, None);
    let g = snapped_axis(&mut f, (AxisSnap::Gather, false), u, n, at)?;
    let near = snapped_axis(&mut f, (AxisSnap::Nearest, false), u, n, at)?;
    let pg = snapped_axis(&mut f, (AxisSnap::Gather, true), u, n, at)?;
    let cell = snapped_axis(&mut f, (AxisSnap::Cell, true), u, n, at)?;
    for (k, v) in [x, g, near, pg, cell].into_iter().enumerate() {
        let bits = f.convert(ConvOp::Bitcast, v, u32_)?;
        let at = word(&mut f, output, i, 5, k as u32)?;
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

/// The kernel's five words per lane `(u, n)`.
fn run(address: TapAddress, lanes: &[(f32, u32)]) -> Vec<[u32; 5]> {
    run_offset(address, 0, lanes)
}

/// The kernel's five words per lane `(u, n)`, under the constant texel offset `offset`.
fn run_offset(address: TapAddress, offset: i32, lanes: &[(f32, u32)]) -> Vec<[u32; 5]> {
    let m = kernel(address, offset).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(u, n)| [u.to_bits().to_le_bytes(), n.to_le_bytes()])
        .flatten()
        .collect();
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
                bytes: vec![0xAB; lanes.len() * 20],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<20>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// How [`wrap`] rounds the subtraction.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Rounding {
    Up,
    Down,
    Even,
}

/// `u` wrapped into one period under `address`, its subtraction rounded by `rounding`.
fn wrap(u: f32, address: TapAddress, rounding: Rounding) -> f32 {
    let k = match address {
        TapAddress::Edge | TapAddress::Border => return u,
        TapAddress::Repeat => u.floor(),
        TapAddress::Mirror => {
            let fl = (u * 0.5).floor();
            fl + fl
        }
    };
    // Exact in `f64` but where `|u| < 2^-29`, where it lies just below `-k` and rounds to `-k`
    // (to nearest) either way. So `s` is a neighbour of the exact difference, which `s + k` (exact
    // in `f64`) against `u` places on one side of it or on it.
    let exact = f64::from(u) - f64::from(k);
    let s = exact as f32;
    let z = f64::from(s) + f64::from(k);
    match rounding {
        Rounding::Up if z < f64::from(u) => f32::from_bits(s.to_bits() + 1),
        Rounding::Down if z > f64::from(u) => f32::from_bits(s.to_bits() - 1),
        _ => s,
    }
}

/// `u` wrapped into one period under `address` on an axis of size `n` as Metal wraps it: in
/// `f32`, the subtraction rounded toward positive infinity, or toward negative infinity (as the
/// exact wrap reads, module docs) where `n` is a power of two.
fn wrapped(u: f32, n: u32, address: TapAddress) -> f32 {
    wrap(
        u,
        address,
        if n.is_power_of_two() {
            Rounding::Down
        } else {
            Rounding::Up
        },
    )
}

/// The exact real `x` (an `f64` holding it exactly) rounded to `f32` with ties away from zero.
fn ties_away(x: f64) -> f32 {
    let r = x as f32;
    let d = x - f64::from(r);
    if d == 0.0 || !r.is_finite() {
        return r;
    }
    // The other neighbour, on `x`'s side of `r`.
    let other = if (d > 0.0) == (r >= 0.0) {
        f32::from_bits(r.to_bits() + 1)
    } else if r == 0.0 {
        return r;
    } else {
        f32::from_bits(r.to_bits() - 1)
    };
    let e = (f64::from(other) - x).abs();
    if e == d.abs() && f64::from(other).abs() > f64::from(r).abs() {
        other
    } else {
        r
    }
}

/// Metal's position (module docs).
fn metal_x(u: f32, n: u32, address: TapAddress) -> f32 {
    ties_away(f64::from(wrapped(u, n, address)) * f64::from(n))
}

/// The position were the product rounded to nearest even (the rule before the ties).
fn even_x(u: f32, n: u32, address: TapAddress) -> f32 {
    (f64::from(wrapped(u, n, address)) * f64::from(n)) as f32
}

/// The position were the wrap rounded by `rounding` on every size.
fn x_rounded(u: f32, n: u32, address: TapAddress, rounding: Rounding) -> f32 {
    ties_away(f64::from(wrap(u, address, rounding)) * f64::from(n))
}

/// Whether positions `a` and `b` read another footprint: a gather's `i0`, a linear or bicubic
/// filter's `q`, or a nearest texel.
fn footprints_differ(a: f32, b: f32) -> bool {
    let q = |x: f32| (256.0 * f64::from(x) + 0.5).floor() as i64;
    q(a) != q(b) || a.floor() != b.floor()
}

/// The `i0` at position `x`.
fn gather_i0(x: f32) -> i64 {
    let q = (256.0 * f64::from(x) + 0.5).floor() as i64 - 128;
    q >> 8
}

/// The gather's snapped coordinate (module docs), computed from the exact `i0`.
fn gather_coordinate(x: f32, n: u32, address: TapAddress) -> f32 {
    let mut i0 = gather_i0(x);
    if matches!(address, TapAddress::Edge | TapAddress::Border) {
        i0 = i0.clamp(-10, i64::from(n) + 8);
    }
    (i0 as f32 + 1.0) / n as f32
}

/// The nearest sample's snapped coordinate.
fn nearest_coordinate(x: f32, n: u32) -> f32 {
    (x.floor() + 0.5) / n as f32
}

const ADDRESSES: [TapAddress; 4] = [
    TapAddress::Edge,
    TapAddress::Border,
    TapAddress::Repeat,
    TapAddress::Mirror,
];

/// `j 2^k` for odd `j` with `n`'s odd part times `j` in `[2^24, 2^25)`: the product `u n` is then
/// exactly halfway between two `f32`s. Several `j` per size, at magnitudes from a fraction of a
/// period to a few periods, both signs.
fn ties(n: u32) -> Vec<f32> {
    let odd = n >> n.trailing_zeros();
    if odd == 1 {
        return Vec::new();
    }
    let lo = (1u64 << 24).div_ceil(u64::from(odd));
    let hi = ((1u64 << 25) - 1) / u64::from(odd);
    let mut js: Vec<u64> = Vec::new();
    for j in [
        lo,
        lo + 1,
        lo + 2,
        lo + 3,
        (lo + hi) / 2,
        (lo + hi) / 2 + 1,
        hi - 1,
        hi,
    ] {
        let j = j | 1;
        if j >= lo && j <= hi && !js.contains(&j) && j < (1 << 24) {
            js.push(j);
        }
    }
    let mut out = Vec::new();
    for j in js {
        for target in [0.3f64, 0.99, 1.7, 2.6] {
            // `u = j 2^k` near `target`, an exact `f32` (`j` has at most 24 bits).
            let k = (target / j as f64).log2().floor() as i32;
            let u = (j as f64 * 2f64.powi(k)) as f32;
            assert_eq!(f64::from(u), j as f64 * 2f64.powi(k));
            out.push(u);
            out.push(-u);
        }
    }
    out
}

/// Coordinates within three ulps of each footprint step from 6 texels below the level to 2 above
/// it (a gather's `i0` steps where `X` crosses `i + 1/2 - 1/512`, a nearest texel at each
/// integer) whose product `u n` is a tie the even and ties-away roundings put on either side of
/// the step (the tie probe's lanes).
fn step_ties(n: u32) -> Vec<f32> {
    let nf = f64::from(n);
    let mut steps: Vec<f64> = (-6 * 256..(i64::from(n) + 2) * 256)
        .map(|z| (2 * z - 1) as f64 / 512.0)
        .collect();
    steps.extend((-6..i64::from(n) + 2).map(|k| k as f64));
    let mut out = Vec::new();
    for step in steps {
        if step == 0.0 {
            continue;
        }
        let u0 = (step / nf) as f32;
        for d in -3i32..=3 {
            let u = f32::from_bits(u0.to_bits().wrapping_add_signed(d));
            let exact = f64::from(u) * nf;
            let (even, away) = ((exact as f32), ties_away(exact));
            if even != away
                && (gather_i0(even) != gather_i0(away) || even.floor() != away.floor())
                && !out.contains(&u)
            {
                out.push(u);
            }
        }
    }
    out
}

/// Coordinates for size `n`: texel edges and centres and a ulp either side of each, through
/// several periods both ways, every tie [`ties`] makes, and for the small sizes every tie at a
/// footprint step ([`step_ties`]).
fn lanes(n: u32) -> Vec<(f32, u32)> {
    let mut us: Vec<f32> = Vec::new();
    let nf = f64::from(n);
    for k in -2 * i64::from(n) - 3..=3 * i64::from(n) + 3 {
        for frac in [0.0, 0.5, 0.25, 1.0 / 512.0, 0.5 - 1.0 / 512.0] {
            let u = ((k as f64 + frac) / nf) as f32;
            us.push(u);
            us.push(f32::from_bits(u.to_bits().wrapping_add(1)));
            us.push(f32::from_bits(u.to_bits().wrapping_sub(1)));
        }
    }
    us.extend(ties(n));
    if n <= 255 {
        us.extend(step_ties(n));
    }
    us.retain(|u| u.is_finite());
    us.into_iter().map(|u| (u, n)).collect()
}

/// On every addressing and every size from 1 to Metal's largest, `X` is Metal's position bit for
/// bit (ties included), and the lanes include, on the clamping axes, ties where the even product
/// reads another footprint (a gather's `i0` or a nearest texel), and on the repeating and mirrored
/// axes coordinates outside one period where multiplying first reads another footprint, and where
/// the wrap's rounding does: up at a size that is not a power of two, where rounding to nearest
/// even reads another footprint, and down at a power of two, where rounding up does.
#[test]
fn the_position_is_metals_wrap_first_and_ties_away() {
    let sizes = [
        1u32, 2, 3, 5, 7, 8, 10, 12, 13, 16, 255, 256, 1000, 4095, 16383, 16384,
    ];
    for address in ADDRESSES {
        let (mut ties_seen, mut wraps_seen) = (0usize, 0usize);
        let (mut ups_seen, mut downs_seen) = (0usize, 0usize);
        for n in sizes {
            // Every tie, and up to 200 lanes where the wrap's rounding up or down decides; the
            // other lanes sampled to at most about 1500 per size.
            let special: Vec<f32> = ties(n)
                .into_iter()
                .chain(if n <= 255 { step_ties(n) } else { Vec::new() })
                .collect();
            let all = lanes(n);
            let step = (all.len() / 1500).max(1);
            let decides = |u: f32| {
                footprints_differ(
                    x_rounded(u, n, address, Rounding::Up),
                    x_rounded(u, n, address, Rounding::Down),
                )
            };
            let mut deciding = 0;
            let picked: Vec<(f32, u32)> = all
                .iter()
                .enumerate()
                .filter(|(k, (u, _))| {
                    let take = deciding < 200 && decides(*u);
                    deciding += usize::from(take);
                    take || k % step == 0 || special.contains(u)
                })
                .map(|(_, l)| *l)
                .collect();
            let out = run(address, &picked);
            for (&(u, n), got) in picked.iter().zip(&out) {
                let want = metal_x(u, n, address);
                assert_eq!(
                    got[0],
                    want.to_bits(),
                    "{address:?} n {n} u {u:e} ({:#010x}): X {:e} vs {want:e}",
                    u.to_bits(),
                    f32::from_bits(got[0])
                );
                let even = even_x(u, n, address);
                if gather_i0(want) != gather_i0(even) || want.floor() != even.floor() {
                    ties_seen += 1;
                }
                if n.is_power_of_two() {
                    let up = x_rounded(u, n, address, Rounding::Up);
                    downs_seen += usize::from(footprints_differ(want, up));
                } else {
                    let nearest_even = x_rounded(u, n, address, Rounding::Even);
                    ups_seen += usize::from(footprints_differ(want, nearest_even));
                }
                let first = (f64::from(u) * f64::from(n)) as f32;
                let period = match address {
                    TapAddress::Mirror => 2 * i64::from(n),
                    _ => i64::from(n),
                };
                if matches!(address, TapAddress::Repeat | TapAddress::Mirror)
                    && (gather_i0(first) - gather_i0(want)).rem_euclid(period) != 0
                {
                    wraps_seen += 1;
                }
            }
        }
        // A tie moves a footprint only at a negative position, or at one of `2^14` or more (where
        // a step of `i0` is odd in `f32`): elsewhere the even neighbour is the step itself
        // (module docs). A repeating or mirrored axis's position is never negative.
        if matches!(address, TapAddress::Repeat | TapAddress::Mirror) {
            assert!(
                wraps_seen > 0,
                "{address:?}: no lane where the wrap decides"
            );
            assert!(
                ups_seen > 0,
                "{address:?}: no lane where the wrap's rounding up decides"
            );
            assert!(
                downs_seen > 0,
                "{address:?}: no lane where a power of two's rounding down decides"
            );
        } else {
            assert!(ties_seen > 0, "{address:?}: no lane where the ties decide");
        }
    }
}

/// The integer tie test's bound, `n < 2^22`: at `n = 2^22 - 1` every tie on every addressing is
/// Metal's.
#[test]
fn the_ties_hold_up_to_the_size_bound() {
    let n = (1u32 << 22) - 1;
    let us: Vec<(f32, u32)> = ties(n).into_iter().map(|u| (u, n)).collect();
    assert!(!us.is_empty());
    for address in ADDRESSES {
        let out = run(address, &us);
        let mut decided = 0;
        for (&(u, n), got) in us.iter().zip(&out) {
            let want = metal_x(u, n, address);
            assert_eq!(got[0], want.to_bits(), "{address:?} n {n} u {u:e}");
            decided += usize::from(want != even_x(u, n, address));
        }
        assert!(decided > 0, "{address:?}: no tie rounded the other way");
    }
}

/// The snapped coordinates, against `i0` and `floor(X)` computed exactly, on every lane of Metal's
/// sizes; each lies within `2^-7` of a texel of its target (the block's centre, the texel's
/// centre), half a texel from every edge the device's rounding could cross.
#[test]
fn the_snapped_coordinates_centre_metals_texels() {
    let sizes = [1u32, 2, 3, 7, 12, 13, 255, 1000, 16383, 16384];
    for address in ADDRESSES {
        for n in sizes {
            let all = lanes(n);
            let step = (all.len() / 1500).max(1);
            let picked: Vec<(f32, u32)> = all.into_iter().step_by(step).collect();
            let out = run(address, &picked);
            for (&(u, n), got) in picked.iter().zip(&out) {
                let x = metal_x(u, n, address);
                let g = gather_coordinate(x, n, address);
                assert_eq!(
                    got[1],
                    g.to_bits(),
                    "{address:?} n {n} u {u:e}: gather {:e} vs {g:e}",
                    f32::from_bits(got[1])
                );
                let target = (g as f64 * f64::from(n)).round();
                assert!((f64::from(g) * f64::from(n) - target).abs() <= 2f64.powi(-7));
                if matches!(address, TapAddress::Repeat | TapAddress::Mirror) {
                    let c = nearest_coordinate(x, n);
                    assert_eq!(
                        got[2],
                        c.to_bits(),
                        "{address:?} n {n} u {u:e}: nearest {:e} vs {c:e}",
                        f32::from_bits(got[2])
                    );
                    let centre = f64::from(x.floor()) + 0.5;
                    assert!((f64::from(c) * f64::from(n) - centre).abs() <= 2f64.powi(-7));
                }
            }
        }
    }
}

/// The gather's `i0` in `f32` is exact below `|X| = 2^15`, which bounds every position of a
/// mirrored axis of Metal's largest size (`N = 2^14`, a power of two, whose wrap rounds down, so a
/// mirrored `u` a ulp below 0 lies at `X = 2^15 - 2^-9`): positions at and just below
/// each step of `i0` near the bound, and that largest position.
#[test]
fn the_gather_base_holds_up_to_the_position_bound() {
    let n = 1u32 << 14;
    let mut us = vec![-f32::from_bits(1 << 23)];
    for i0 in [32_760i64, 32_765, 32_766, 32_767] {
        // `i0` steps where `X` crosses `i0 + 1/2 - 1/512`.
        let edge = i0 as f64 + 0.5 - 1.0 / 512.0;
        for d in [-2.0, -1.0, 0.0, 1.0] {
            let x = edge + d / 512.0;
            us.push((x / f64::from(n)) as f32);
        }
    }
    let lanes: Vec<(f32, u32)> = us.iter().map(|u| (*u, n)).collect();
    let out = run(TapAddress::Mirror, &lanes);
    let mut top = false;
    for (&(u, n), got) in lanes.iter().zip(&out) {
        let x = metal_x(u, n, TapAddress::Mirror);
        assert!(f64::from(x) < 32_768.0);
        top |= f64::from(x) == 32_768.0 - 1.0 / 512.0;
        let g = gather_coordinate(x, n, TapAddress::Mirror);
        assert_eq!(got[1], g.to_bits(), "u {u:e} X {x}");
    }
    assert!(top, "no lane at X = 2^15 - 2^-9");
}

/// A pixel coordinate `x` is Metal's position as given. Its gather's snapped coordinate is the
/// gather's at `X = x`, and its footprint cell's, `(q + 128) / 256 / n` for `q = floor(256 x -
/// 127.5)`, reads as a normalized coordinate (Metal's own sampler's position of it, ties away from
/// zero) the same `q`: every lane a ulp either side of each step and texel edge and centre, on the
/// clamping axes (a pixel sampler only clamps) of Metal's sizes.
#[test]
fn pixel_coordinate_snaps_read_metals_footprint() {
    let q = |x: f32| (256.0 * f64::from(x) - 127.5).floor() as i64;
    for address in [TapAddress::Edge, TapAddress::Border] {
        for n in [1u32, 2, 3, 13, 255, 1000, 16383, 16384] {
            let mut xs = Vec::new();
            let top = i64::from(n).min(40) + 12;
            for k in -12..=top {
                for base in [
                    k as f32,
                    k as f32 + 0.5 - 1.0 / 512.0,
                    k as f32 + 0.5 - 1.0 / 256.0,
                ] {
                    xs.extend([base, base.next_down(), base.next_up(), base + 1.0 / 1024.0]);
                }
            }
            for k in [n - 1, n, n + 1] {
                let b = k as f32 - 0.5 - 1.0 / 512.0;
                xs.extend([b, b.next_down(), b.next_up(), k as f32 - 0.25]);
            }
            let lanes: Vec<(f32, u32)> = xs.iter().map(|x| (*x, n)).collect();
            let out = run(address, &lanes);
            for (&(x, n), got) in lanes.iter().zip(&out) {
                let g = gather_coordinate(x, n, address);
                assert_eq!(got[3], g.to_bits(), "{address:?} n {n} x {x:e}: gather");
                let c = f32::from_bits(got[4]);
                assert_eq!(
                    q(metal_x(c, n, address)),
                    q(x),
                    "{address:?} n {n} x {x:e}: cell {c:e}"
                );
            }
        }
    }
}

/// A kernel whose invocation `i` reads `(n, base, parent)` from buffer 0 and writes
/// [`offset_dropped`]'s answer for a level of size `n` under a base level of size `base`, and its
/// base-level form's for a base level of size `n`, each without a parent test and under the parent
/// test `parent != 0`, as `0` or `1`, to buffer 1.
fn rule_kernel() -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 1)?;
    let base = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let parent = f.load(at, MemAccess::default())?;
    let one = u32_const(&mut f, 1)?;
    let zero = u32_const(&mut f, 0)?;
    let parent = f.binary(crate::spirv::ir::BinOp::INotEqual, parent, zero)?;
    let forms = [None, Some(parent)]
        .into_iter()
        .flat_map(|p| [(Some(base), p), (None, p)]);
    for (k, (base, parent)) in forms.enumerate() {
        let dropped = offset_dropped(&mut f, n, base, parent)?;
        let bit = f.select(dropped, one, zero)?;
        let at = word(&mut f, output, i, 4, k as u32)?;
        f.store(at, bit, MemAccess::default())?;
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

/// The rule kernel's four answers per lane `(n, base)`, under no parent test.
fn run_rule(lanes: &[(u32, u32)]) -> Vec<[u32; 4]> {
    let lanes: Vec<(u32, u32, u32)> = lanes.iter().map(|&(n, b)| (n, b, 0)).collect();
    run_parent_rule(&lanes)
}

/// The rule kernel's four answers per lane `(n, base, parent)`.
fn run_parent_rule(lanes: &[(u32, u32, u32)]) -> Vec<[u32; 4]> {
    let m = rule_kernel().expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(n, b, p)| [n.to_le_bytes(), b.to_le_bytes(), p.to_le_bytes()])
        .flatten()
        .collect();
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
                bytes: vec![0xAB; lanes.len() * 16],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<16>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// Whether Metal dropped a constant texel offset on a repeating or mirrored axis whose level size
/// is `.0` under a base level of size `.1`: the offset probe, round 3 (Apple M4,
/// macOS 27.0.1), every level of 18 bases at an explicit level and through a view of one level,
/// under `repeat` and `mirrored_repeat` alike, offsets -8 to 7.
const MEASURED_RULE: [(u32, u32, bool); 85] = [
    (2, 2, false),
    (2, 4, false),
    (2, 5, true),
    (2, 8, false),
    (2, 9, true),
    (2, 10, true),
    (2, 11, true),
    (2, 16, false),
    (2, 17, true),
    (2, 18, true),
    (2, 20, true),
    (2, 32, false),
    (2, 33, true),
    (2, 36, true),
    (2, 40, true),
    (2, 64, false),
    (2, 66, true),
    (3, 3, true),
    (3, 6, true),
    (3, 7, true),
    (3, 12, true),
    (3, 13, true),
    (3, 24, true),
    (3, 48, true),
    (4, 4, false),
    (4, 8, false),
    (4, 9, true),
    (4, 16, false),
    (4, 17, true),
    (4, 18, true),
    (4, 32, false),
    (4, 33, true),
    (4, 36, true),
    (4, 64, false),
    (4, 66, true),
    (5, 5, true),
    (5, 10, true),
    (5, 11, true),
    (5, 20, true),
    (5, 40, true),
    (6, 6, true),
    (6, 12, true),
    (6, 13, true),
    (6, 24, true),
    (6, 48, true),
    (7, 7, true),
    (8, 8, false),
    (8, 16, false),
    (8, 17, false),
    (8, 32, false),
    (8, 33, false),
    (8, 64, false),
    (8, 66, false),
    (9, 9, false),
    (9, 18, false),
    (9, 36, false),
    (10, 10, false),
    (10, 20, false),
    (10, 40, false),
    (11, 11, false),
    (12, 12, false),
    (12, 24, false),
    (12, 48, false),
    (13, 13, false),
    (16, 16, false),
    (16, 32, false),
    (16, 33, false),
    (16, 64, false),
    (16, 66, false),
    (17, 17, false),
    (18, 18, false),
    (18, 36, false),
    (20, 20, false),
    (20, 40, false),
    (24, 24, false),
    (24, 48, false),
    (32, 32, false),
    (32, 64, false),
    (33, 33, false),
    (33, 66, false),
    (36, 36, false),
    (40, 40, false),
    (48, 48, false),
    (64, 64, false),
    (66, 66, false),
];

/// [`offset_dropped`] is Metal's rule, a level size from 2 to 7 under a base-level size that is not
/// a power of two: on every pair round 3 measured, on every pair of a level size up to 64 and a
/// base up to 600, and its base-level form (sizes 3, 5, 6 and 7) wherever the level is the base.
#[test]
fn the_offset_rule_is_metals_by_level_and_base() {
    let rule = |n: u32, base: u32| (2..=7).contains(&n) && !base.is_power_of_two();
    for &(n, base, dropped) in &MEASURED_RULE {
        assert_eq!(rule(n, base), dropped, "the measured pair ({n}, {base})");
    }
    let mut lanes: Vec<(u32, u32)> = MEASURED_RULE.iter().map(|&(n, b, _)| (n, b)).collect();
    for base in 1u32..=600 {
        let mut n = base;
        loop {
            lanes.push((n, base));
            if n == 1 {
                break;
            }
            n /= 2;
        }
    }
    lanes.extend((0u32..=70).map(|n| (n, n)));
    lanes.extend([(u32::MAX, u32::MAX), (1 << 14, 1 << 14), (16383, 16383)]);
    let out = run_rule(&lanes);
    for (&(n, base), got) in lanes.iter().zip(&out) {
        assert_eq!(got[0], u32::from(rule(n, base)), "level {n} of base {base}");
        let at_base = matches!(n, 3 | 5 | 6 | 7);
        assert_eq!(got[1], u32::from(at_base), "base level {n}");
        if n == base && n != 0 {
            assert_eq!(rule(n, n), at_base, "the base-level form of {n}");
        }
    }
}

/// The residue: a view whose first level lies `level` levels above its parent's base reads Metal's
/// rule by its parent's base-level size. Under the bound texture's `TextureFacts::offset_base_axes`
/// bit (the binder's `SpecKey::OffsetBaseAxes`), [`offset_dropped`] of each of the view's levels,
/// by the view's own base-level size and in its base-level form, answers as Metal's rule by the
/// parent's, on every parent size up to 300 and view level up to 9; without the bit it reads the
/// view's own, as before.
#[test]
fn the_offset_rule_reads_the_parent_s_base_through_the_view_bit() {
    use crate::bind::TextureFacts;
    use crate::metal::{PixelFormat, TextureType};
    let rule = |n: u32, base: u32| (2..=7).contains(&n) && !base.is_power_of_two();
    let mut lanes = Vec::new();
    let mut cases = Vec::new();
    for parent in 1u32..=300 {
        for level in 0..10u32 {
            let own = (parent >> level).max(1);
            let mut facts = TextureFacts::new(TextureType::Type1D, PixelFormat::RGBA8Unorm);
            facts.parent_relative_level = level;
            facts.parent_base_size = [parent, 0, 0];
            facts.mipmap_level_count = 32 - own.leading_zeros();
            let bit = u32::from(facts.offset_base_axes() & 1);
            let mut n = own;
            loop {
                lanes.push((n, own, bit));
                cases.push((parent, level));
                if n == 1 {
                    break;
                }
                n /= 2;
            }
        }
    }
    let out = run_parent_rule(&lanes);
    for ((&(n, own, bit), &(parent, level)), got) in lanes.iter().zip(&cases).zip(&out) {
        let at = format!("level {n} of a view at {level} of {parent} (own base {own}, bit {bit})");
        let metal = u32::from(rule(n, parent));
        assert_eq!(got[2], metal, "{at}");
        if n == own {
            assert_eq!(got[3], metal, "{at}: the base-level form");
        }
        assert_eq!(got[0], u32::from(rule(n, own)), "{at}: no parent test");
    }
}

/// On a repeating or mirrored axis whose size is 3, 5, 6 or 7, Metal drops a texel offset the
/// device adds, so each snapped coordinate moves back by the offset there: the device's index at it
/// (`floor(c n - 1/2)` for a gather, `floor(c n)` for a nearest sample), plus the offset, is the
/// unmoved snap's. On every other size, and on every clamping axis, the offset leaves the snap as
/// it is (Metal adds it as the device does). Reading `u` as a pixel coordinate, the gather and the
/// footprint cell move back the same way.
#[test]
fn a_snap_moves_back_by_the_offset_metal_drops() {
    let mut moved = 0usize;
    for address in ADDRESSES {
        for n in (1u32..=10).chain([13, 16, 255]) {
            let all = lanes(n);
            let step = (all.len() / 200).max(1);
            let picked: Vec<(f32, u32)> = all.into_iter().step_by(step).collect();
            let plain = run(address, &picked);
            let dropped = matches!(address, TapAddress::Repeat | TapAddress::Mirror)
                && matches!(n, 3 | 5 | 6 | 7);
            for o in [-8i32, -1, 1, 7] {
                let out = run_offset(address, o, &picked);
                let back = if dropped { o as f32 } else { 0.0 };
                let nf = n as f32;
                for ((&(u, _), got), was) in picked.iter().zip(&out).zip(&plain) {
                    let at = format!("{address:?} n {n} o {o} u {u:e}");
                    if !dropped {
                        assert_eq!(got, was, "{at}: moved where Metal adds the offset");
                        continue;
                    }
                    assert_eq!(got[0], was[0], "{at}: X");
                    let x = metal_x(u, n, address);
                    let g = (gather_i0(x) as f32 + 1.0 - back) / nf;
                    assert_eq!(got[1], g.to_bits(), "{at}: gather");
                    let c = (x.floor() + 0.5 - back) / nf;
                    assert_eq!(got[2], c.to_bits(), "{at}: nearest");
                    let pg = (gather_i0(u) as f32 + 1.0 - back) / nf;
                    assert_eq!(got[3], pg.to_bits(), "{at}: pixel gather");
                    let y = 256.0 * u;
                    let fl = y.floor();
                    let cell = (fl + if y - fl >= 0.5 { 1.0 } else { 0.0 }) * (1.0 / 256.0);
                    let cell = (cell - back) / nf;
                    assert_eq!(got[4], cell.to_bits(), "{at}: cell");
                    // The device's index at the moved coordinate, plus the offset, is the
                    // unmoved one's.
                    let index = |bits: u32, gather: bool| {
                        let y = f64::from(f32::from_bits(bits)) * f64::from(n);
                        (if gather { y - 0.5 } else { y }).floor() as i64
                    };
                    assert_eq!(
                        index(got[1], true) + i64::from(o),
                        index(was[1], true),
                        "{at}"
                    );
                    assert_eq!(
                        index(got[2], false) + i64::from(o),
                        index(was[2], false),
                        "{at}"
                    );
                    moved += 1;
                }
            }
        }
    }
    assert!(moved > 0, "no lane moved back");
}

/// A kernel whose invocation `i` reads `(bits(u), n, word)` from buffer 0 and writes the bits of
/// [`snapped_by_word`]'s coordinate of a one-axis sample of size `n` (the size of the level it
/// reads) through a runtime sampler of word `word`, at a level the function knows or not
/// (`leveled`), to buffer 1.
fn word_kernel(leveled: bool) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let ub = f.load(at, MemAccess::default())?;
    let u = f.convert(ConvOp::Bitcast, ub, f32_)?;
    let at = word(&mut f, input, i, 3, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let w = f.load(at, MemAccess::default())?;
    let c = snapped_by_word(&mut f, (w, leveled), (u, &[], None), 1, (n, None, 1))?;
    let bits = f.convert(ConvOp::Bitcast, c, u32_)?;
    let at = word(&mut f, output, i, 1, 0)?;
    f.store(at, bits, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// The word kernel's coordinate bits per lane `(u, n, word)`, at a level the function knows or
/// not (`leveled`).
fn run_word(lanes: &[(f32, u32, u32)], leveled: bool) -> Vec<u32> {
    let m = word_kernel(leveled).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(u, n, w)| [u.to_bits().to_le_bytes(), n.to_le_bytes(), w.to_le_bytes()])
        .flatten()
        .collect();
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
                bytes: vec![0xAB; lanes.len() * 4],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// A runtime sampler's word snap hands the device the centre of Metal's texel, bit for bit as the
/// constexpr nearest snap does, on an axis whose word repeats or mirrors, or mirrors once and
/// clamps (a clamping axis's position), where the word reads the base level (normalized, nearest,
/// no mip filter, isotropic, whatever else it states); the coordinate as given on a clamping axis,
/// and for any word that filters linearly, takes pixel coordinates, a mip filter, or anisotropy.
#[test]
fn the_word_snap_centres_metals_texels_where_the_word_reads_the_base_level() {
    word_snap_reaches(false);
}

/// At a level the function knows, a runtime sampler's word snap reaches a nearest mip filter's
/// sample too (the size is that level's, `level::snap_level`), and nothing else more: a linear mip
/// filter's two levels keep the coordinate.
#[test]
fn a_leveled_word_snap_centres_metals_texels_where_the_word_reads_one_level() {
    word_snap_reaches(true);
}

/// The word snap's coordinates against the host's, for words that reach it and words that do not,
/// at a level the function knows or not (`leveled`).
fn word_snap_reaches(leveled: bool) {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let modes = [
        (M::ClampToEdge, None),
        (M::MirrorClampToEdge, Some(TapAddress::Edge)),
        (M::Repeat, Some(TapAddress::Repeat)),
        (M::MirrorRepeat, Some(TapAddress::Mirror)),
        (M::ClampToZero, None),
        (M::ClampToBorderColor, None),
    ];
    // Base-level words: the default, and one with every field the reach ignores set.
    let ignored = SamplerWord::ADDRESS[1]
        | SamplerWord::ADDRESS[2]
        | SamplerWord::BORDER
        | SamplerWord::COMPARE
        | SamplerWord::REDUCTION;
    let nearest_mip = 1 << SamplerWord::MIP_FILTER_SHIFT;
    let others = [
        SamplerWord::PIXEL_COORDINATES,
        SamplerWord::MIN_LINEAR,
        SamplerWord::MAG_LINEAR,
        nearest_mip,
        2 << SamplerWord::MIP_FILTER_SHIFT,
        SamplerWord::ANISOTROPIC,
        nearest_mip | SamplerWord::MIN_LINEAR,
        SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX,
    ];
    let mut checked = 0;
    for n in [3u32, 13, 16, 255] {
        let all = lanes(n);
        let step = (all.len() / 300).max(1);
        let us: Vec<f32> = all.into_iter().step_by(step).map(|(u, _)| u).collect();
        for (mode, address) in modes {
            let m = mode.as_raw() << SamplerWord::ADDRESS_SHIFT[0];
            let mut words: Vec<(u32, bool)> = vec![(m, true), (m | ignored, true)];
            words.extend(others.iter().map(|&o| {
                // A clamp's levels are ignored; a nearest mip filter's one level reaches a
                // leveled snap alone.
                let reach = o == SamplerWord::LEVEL_MIN | SamplerWord::LEVEL_MAX
                    || (leveled && o == nearest_mip);
                (m | o, reach)
            }));
            let picked: Vec<(f32, u32, u32)> = words
                .iter()
                .flat_map(|&(w, _)| us.iter().map(move |&u| (u, n, w)))
                .collect();
            let out = run_word(&picked, leveled);
            for (&(u, n, w), &got) in picked.iter().zip(&out) {
                let base = words.iter().any(|&(x, b)| x == w && b);
                let want = match (address, base) {
                    (Some(a), true) => nearest_coordinate(metal_x(u, n, a), n),
                    _ => u,
                };
                assert_eq!(
                    got,
                    want.to_bits(),
                    "{mode:?} n {n} word {w:#x} u {u:e}: {:e} vs {want:e}",
                    f32::from_bits(got)
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 10_000, "{checked}");
}

/// The coordinate the word kernels hand the device where their snap does not reach a lane: a
/// marker no snap yields.
const DEVICE: f32 = -12345.0;

/// A kernel whose invocation `i` reads `(bits(u), n, word)` from buffer 0 and writes the bits of
/// [`gathered_by_word`]'s coordinate of a one-axis gather of size `n` through a runtime sampler of
/// word `word`, on a device with Metal's own sampler or not (`metal`), whose access the device
/// moves by the constant texel offset `offset` (0: none), and of [`celled_by_word`]'s, each
/// handing [`DEVICE`] as the coordinate the device reads otherwise, to buffer 1.
fn word_gather_kernel(metal: bool, offset: i32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let ub = f.load(at, MemAccess::default())?;
    let u = f.convert(ConvOp::Bitcast, ub, f32_)?;
    let at = word(&mut f, input, i, 3, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let w = f.load(at, MemAccess::default())?;
    let device = super::f32_const(&mut f, DEVICE)?;
    let o = match offset {
        0 => None,
        o => Some(super::f32_const(&mut f, o as f32)?),
    };
    let g = gathered_by_word(&mut f, (w, metal), (u, device, &[o], None), 1, (n, 1))?;
    let c = celled_by_word(&mut f, w, (u, device), 1, (n, 1))?;
    for (k, v) in [g, c].into_iter().enumerate() {
        let bits = f.convert(ConvOp::Bitcast, v, u32_)?;
        let at = word(&mut f, output, i, 2, k as u32)?;
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

/// The word gather kernel's two coordinates' bits per lane `(u, n, word)`.
fn run_word_gather(lanes: &[(f32, u32, u32)], metal: bool, offset: i32) -> Vec<[u32; 2]> {
    let m = word_gather_kernel(metal, offset).expect("kernel");
    let bytes: Vec<u8> = lanes
        .iter()
        .flat_map(|(u, n, w)| [u.to_bits().to_le_bytes(), n.to_le_bytes(), w.to_le_bytes()])
        .flatten()
        .collect();
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
                bytes: vec![0xAB; lanes.len() * 8],
            },
        ],
        ..Resources::default()
    };
    let cfg = Config {
        workgroups: [u32::try_from(lanes.len()).expect("lanes"), 1, 1],
        ..Config::default()
    };
    let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
    out.buffers[1]
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| {
            std::array::from_fn(|k| {
                u32::from_le_bytes([c[4 * k], c[4 * k + 1], c[4 * k + 2], c[4 * k + 3]])
            })
        })
        .collect()
}

/// A runtime sampler's gather hands the device the centre of the 2x2 block Metal reads, by its
/// word: at Metal's position by the axis's mode for a normalized word (the constexpr gather's snap,
/// bit for bit, but for its wider clamp of `i0`, which reads the same edge texels or border; a
/// `MirrorClampToEdge` axis's a clamping axis's), at the coordinate as given for a
/// pixel word, its index moved back by an offset where a repeating, mirrored or mirror-clamped
/// axis's size drops it; on a device with Metal's own sampler it snaps a pixel word alone, never
/// moved by an offset, which that sampler places as Metal places a pixel coordinate's. A word's
/// other fields leave it as it is. The footprint-cell snap takes the centre of the step Metal reads
/// the coordinate as given at where the word states a linear pixel sampler, and the device's
/// coordinate for every other word.
#[test]
fn a_runtime_samplers_gather_centres_metals_block_by_its_word() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let modes = [
        (M::ClampToEdge, Some(TapAddress::Edge)),
        // Its position is a clamping axis's.
        (M::MirrorClampToEdge, Some(TapAddress::Edge)),
        (M::Repeat, Some(TapAddress::Repeat)),
        (M::MirrorRepeat, Some(TapAddress::Mirror)),
        (M::ClampToZero, Some(TapAddress::Border)),
        (M::ClampToBorderColor, Some(TapAddress::Border)),
    ];
    // Fields the gather ignores: the filters, the mip filter, the `t` and `r` modes, the border,
    // the comparison, the reduction, anisotropy and the level clamps.
    let ignored = SamplerWord::MIN_LINEAR
        | SamplerWord::MAG_LINEAR
        | SamplerWord::MIP_FILTER
        | SamplerWord::ADDRESS[1]
        | SamplerWord::ADDRESS[2]
        | SamplerWord::BORDER
        | SamplerWord::COMPARE
        | SamplerWord::REDUCTION
        | SamplerWord::ANISOTROPIC
        | SamplerWord::LEVEL_MIN
        | SamplerWord::LEVEL_MAX;
    let linear_pixel =
        SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR | SamplerWord::MAG_LINEAR;
    let mut checked = 0usize;
    for (metal, offset) in [
        (false, 0i32),
        (false, -3),
        (false, 2),
        (true, 0),
        (true, -3),
    ] {
        for n in [1u32, 3, 7, 13, 16, 255, 16383] {
            let all = lanes(n);
            let step = (all.len() / 60).max(1);
            let us: Vec<f32> = all.into_iter().step_by(step).map(|(u, _)| u).collect();
            let mut picked = Vec::new();
            for (mode, _) in modes {
                let m = mode.as_raw() << SamplerWord::ADDRESS_SHIFT[0];
                for w in [
                    m,
                    m | ignored,
                    m | SamplerWord::PIXEL_COORDINATES,
                    m | linear_pixel,
                    m | SamplerWord::PIXEL_COORDINATES | SamplerWord::MIN_LINEAR,
                ] {
                    picked.extend(us.iter().map(|&u| (u, n, w)));
                }
            }
            // Pixel coordinates around the steps and edges of the texels near the level.
            for k in -12i64..=i64::from(n.min(40)) + 12 {
                for base in [k as f32, k as f32 + 0.5 - 1.0 / 512.0, k as f32 + 0.25] {
                    for x in [base, base.next_down(), base.next_up()] {
                        for w in [SamplerWord::PIXEL_COORDINATES, linear_pixel] {
                            picked.push((x, n, w));
                        }
                    }
                }
            }
            let out = run_word_gather(&picked, metal, offset);
            let nf = n as f32;
            for (&(u, n, w), got) in picked.iter().zip(&out) {
                let at = format!("metal {metal} offset {offset} n {n} word {w:#x} u {u:e}");
                let pixel = w & SamplerWord::PIXEL_COORDINATES != 0;
                let raw = (w >> SamplerWord::ADDRESS_SHIFT[0]) & 0b111;
                let address = modes
                    .iter()
                    .find(|(m, _)| m.as_raw() == raw)
                    .and_then(|&(_, a)| a);
                let snapped = match (pixel, address) {
                    (true, _) => Some(u),
                    (false, Some(a)) if !metal => Some(metal_x(u, n, a)),
                    (false, _) => None,
                };
                let want = match snapped {
                    Some(x) => {
                        let i0 = gather_i0(x).clamp(-2 * i64::from(n) - 10, 2 * i64::from(n) + 8);
                        let wraps =
                            matches!(address, Some(TapAddress::Repeat | TapAddress::Mirror))
                                || raw == M::MirrorClampToEdge.as_raw();
                        // Metal's own sampler places the offset itself.
                        let back = if offset != 0 && wraps && !metal && matches!(n, 3 | 5 | 6 | 7) {
                            offset as f32
                        } else {
                            0.0
                        };
                        (i0 as f32 + 1.0 - back) / nf
                    }
                    None => DEVICE,
                };
                assert_eq!(
                    got[0],
                    want.to_bits(),
                    "{at}: gather {:e} vs {want:e}",
                    f32::from_bits(got[0])
                );
                if let Some(x) = snapped {
                    // The device reads the block at `i0`, as Metal does.
                    let c = f64::from(f32::from_bits(got[0])) * f64::from(n);
                    let i0 = gather_i0(x).clamp(-2 * i64::from(n) - 10, 2 * i64::from(n) + 8);
                    assert!((c - (i0 + 1) as f64).abs() <= 2f64.powi(-7) + 8.0, "{at}");
                }
                let cell = if w & linear_pixel == linear_pixel {
                    let y = 256.0 * u;
                    let fl = y.floor();
                    ((fl + if y - fl >= 0.5 { 1.0 } else { 0.0 }) * (1.0 / 256.0)) / nf
                } else {
                    DEVICE
                };
                assert_eq!(
                    got[1],
                    cell.to_bits(),
                    "{at}: cell {:e} vs {cell:e}",
                    f32::from_bits(got[1])
                );
                checked += 1;
            }
        }
    }
    assert!(checked > 30_000, "{checked}");
}

/// A kernel whose invocation `i` reads `(bits(c), n, word)` from buffer 0 and writes the bits of
/// [`pixel_centre_by_word`]'s centre of a one-axis texel centre `c` on a level of `n` texels
/// through a runtime pixel sampler of word `word`, whose access the device moves by the constant
/// texel offset `offset`, to buffer 1.
fn pixel_centre_kernel(offset: i32) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = storage_buffer(&mut b, 0, u32_)?;
    let output = storage_buffer(&mut b, 1, u32_)?;
    let id = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let gid = f.module().global_value(id)?;
    let gid = f.load(gid, MemAccess::default())?;
    let i = f.composite_extract(gid, &[0])?;
    let at = word(&mut f, input, i, 3, 0)?;
    let cb = f.load(at, MemAccess::default())?;
    let c = f.convert(ConvOp::Bitcast, cb, f32_)?;
    let at = word(&mut f, input, i, 3, 1)?;
    let n = f.load(at, MemAccess::default())?;
    let at = word(&mut f, input, i, 3, 2)?;
    let w = f.load(at, MemAccess::default())?;
    let o = match offset {
        0 => None,
        o => Some(super::f32_const(&mut f, o as f32)?),
    };
    let moved = pixel_centre_by_word(&mut f, w, (c, &[o], None), 1, (n, 1))?;
    let bits = f.convert(ConvOp::Bitcast, moved, u32_)?;
    let at = word(&mut f, output, i, 1, 0)?;
    f.store(at, bits, MemAccess::default())?;
    f.ret(None)?;
    let main = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        main,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// A runtime pixel sampler's nearest texel centre moves back by a texel offset Metal drops, which
/// the device adds: on an axis the word states `MirrorClampToEdge` whose size is 3, 5, 6 or 7;
/// every other axis, size, offset and mode keeps its centre.
#[test]
fn a_pixel_centre_moves_back_by_an_offset_metal_drops_on_a_mirror_clamped_axis() {
    use crate::metal::sampler::SamplerWord;
    use crate::metal::SamplerAddressMode as M;
    let modes = [
        M::ClampToEdge,
        M::MirrorClampToEdge,
        M::ClampToZero,
        M::ClampToBorderColor,
    ];
    let mut checked = 0usize;
    for offset in [0i32, -3, 2, -7, 5] {
        let mut picked = Vec::new();
        for n in [1u32, 2, 3, 4, 5, 6, 7, 8, 13, 255] {
            for mode in modes {
                let w = SamplerWord::PIXEL_COORDINATES
                    | mode.as_raw() << SamplerWord::ADDRESS_SHIFT[0]
                    | M::MirrorClampToEdge.as_raw() << SamplerWord::ADDRESS_SHIFT[1];
                for k in -20i32..=20 {
                    picked.push((k as f32 + 0.5, n, w));
                }
            }
        }
        let m = pixel_centre_kernel(offset).expect("kernel");
        let bytes: Vec<u8> = picked
            .iter()
            .flat_map(|(c, n, w)| [c.to_bits().to_le_bytes(), n.to_le_bytes(), w.to_le_bytes()])
            .flatten()
            .collect();
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
                    bytes: vec![0xAB; picked.len() * 4],
                },
            ],
            ..Resources::default()
        };
        let cfg = Config {
            workgroups: [u32::try_from(picked.len()).expect("lanes"), 1, 1],
            ..Config::default()
        };
        let (out, _) = interp::run_with_images(&m, &cfg, res, Vec::new()).expect("run");
        for (&(c, n, w), got) in picked.iter().zip(out.buffers[1].as_chunks::<4>().0.iter()) {
            let got = f32::from_le_bytes([got[0], got[1], got[2], got[3]]);
            let raw = (w >> SamplerWord::ADDRESS_SHIFT[0]) & 0b111;
            let dropped = raw == M::MirrorClampToEdge.as_raw() && matches!(n, 3 | 5 | 6 | 7);
            let want = if dropped { c - offset as f32 } else { c };
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "offset {offset} n {n} word {w:#x} c {c}: {got} vs {want}"
            );
            checked += 1;
        }
    }
    assert!(checked > 5_000, "{checked}");
}
