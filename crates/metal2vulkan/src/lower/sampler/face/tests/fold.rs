//! `define_fold` on the interpreter against the cube's geometry. Each invocation reads a tap (its
//! face, its indices on the face's axes, the face size, and the corner face it picks) and writes
//! the 2D-array coordinate the fold names; the reference finds the texel whose centre on the cube
//! is nearest the tap's, and at the cube's corner, where the tap lies past both edges of its face,
//! the corner texel of each of the three faces there.

use super::super::define_fold;
use super::{kernel_on, run_records};
use crate::spirv::builder::ModuleBuilder;
use crate::spirv::ir::{ConvOp, Type};

/// The direction of tap `(i, j)` of face `face` (Vulkan's order) in the doubled frame where a
/// face spans `[-n, n]`: Vulkan's face table at `sc = 2 i + 1 - n` and `tc = 2 j + 1 - n`, the
/// major component `±n`.
fn direction(face: usize, i: i64, j: i64, n: i64) -> [i64; 3] {
    let (s, t) = (2 * i + 1 - n, 2 * j + 1 - n);
    match face {
        0 => [n, -t, -s],
        1 => [-n, -t, s],
        2 => [s, n, t],
        3 => [s, -n, -t],
        4 => [s, -t, n],
        _ => [-s, -t, -n],
    }
}

/// Every texel of a cube of `n`-texel faces: its face, its indices, and its centre.
fn centres(n: i64) -> Vec<(usize, i64, i64, [i64; 3])> {
    let mut out = Vec::new();
    for face in 0..6 {
        for j in 0..n {
            for i in 0..n {
                out.push((face, i, j, direction(face, i, j, n)));
            }
        }
    }
    out
}

/// The texel tap `(i, j)` of face `face` reads (picking face axis `pick` at the cube's corner):
/// within the face its own; past one edge the texel whose centre is nearest the tap's (one
/// texel, across the edge); past both, the texel at the cube's corner of the face of axis `pick`,
/// whose centre lies one step in from the corner on the two other axes.
fn reference(face: usize, (i, j): (i64, i64), n: i64, pick: usize) -> (usize, i64, i64) {
    let d = direction(face, i, j, n);
    let all = centres(n);
    let past_i = !(0..n).contains(&i);
    let past_j = !(0..n).contains(&j);
    if past_i && past_j {
        let mut c = d.map(|v| v.signum() * (n - 1));
        c[pick] = d[pick].signum() * n;
        let found: Vec<_> = all.iter().filter(|t| t.3 == c).collect();
        assert_eq!(found.len(), 1, "the corner texel of {c:?}");
        return (found[0].0, found[0].1, found[0].2);
    }
    let distance = |c: &[i64; 3]| -> i64 { (0..3).map(|k| (c[k] - d[k]).pow(2)).sum() };
    let least = all.iter().map(|t| distance(&t.3)).min().expect("texels");
    let found: Vec<_> = all.iter().filter(|t| distance(&t.3) == least).collect();
    assert_eq!(found.len(), 1, "the nearest texel to {d:?}");
    (found[0].0, found[0].1, found[0].2)
}

/// The fold names the reference's texel, as the 2D-array coordinate `(i, j, base + face)`, for
/// every tap from one texel before each face's edges to one past, every face, faces of 1, 2, 3,
/// 4, 5, and 8 texels, and every corner pick; and every coordinate lies on the cube.
#[test]
fn the_fold_reads_the_texel_across_the_edge() {
    let mut b = ModuleBuilder::new();
    let fold = define_fold(&mut b).expect("fold");
    // A record: the face, `i`, and `j` as the direction's bits, and the size, the pick, and the
    // first layer packed after them.
    let m = kernel_on(b, 4, &|f, p, extra| {
        let u = f.module().ty(Type::Int {
            width: 32,
            signed: false,
        })?;
        let i = f.module().ty(Type::Int {
            width: 32,
            signed: true,
        })?;
        let fl = f.module().ty(Type::Float { width: 32 })?;
        let vec3 = f.module().ty(Type::Vector { elem: fl, len: 3 })?;
        let face = f.composite_extract(p, &[0])?;
        let face = f.convert(ConvOp::Bitcast, face, u)?;
        let ti = f.composite_extract(p, &[1])?;
        let ti = f.convert(ConvOp::Bitcast, ti, i)?;
        let tj = f.composite_extract(p, &[2])?;
        let tj = f.convert(ConvOp::Bitcast, tj, i)?;
        let field = |f: &mut crate::spirv::builder::FunctionBuilder<'_>, shift: u32| {
            let s = f.u32_const(shift)?;
            let s = f.module().const_value(s)?;
            let m = f.u32_const(0xff)?;
            let m = f.module().const_value(m)?;
            let v = f.binary(crate::spirv::ir::BinOp::ShiftRightLogical, extra, s)?;
            f.binary(crate::spirv::ir::BinOp::BitwiseAnd, v, m)
        };
        let n = field(f, 0)?;
        let n = f.convert(ConvOp::Bitcast, n, i)?;
        let pick = field(f, 8)?;
        let base = field(f, 16)?;
        let at = f.call(fold, &[face, ti, tj, n, pick, base])?.expect("fold");
        f.convert(ConvOp::Bitcast, at, vec3)
    })
    .expect("kernel");
    let mut records = Vec::new();
    let mut want = Vec::new();
    for n in [1i64, 2, 3, 4, 5, 8] {
        for face in 0..6usize {
            for i in -1..=n {
                for j in -1..=n {
                    for pick in 0..3usize {
                        let base = 6 * (n as u32 % 3);
                        records.extend([
                            face as u32,
                            i as u32,
                            j as u32,
                            n as u32 | (pick as u32) << 8 | base << 16,
                        ]);
                        let (g, ri, rj) = reference(face, (i, j), n, pick);
                        want.push([ri as u32, rj as u32, base + g as u32]);
                    }
                }
            }
        }
    }
    let got = run_records(&m, &records, want.len());
    let mut corners = 0;
    for (k, (g, w)) in got.iter().zip(&want).enumerate() {
        let g = g.map(f32::to_bits);
        let r = &records[4 * k..4 * k + 4];
        assert_eq!(
            g,
            *w,
            "face {} tap ({}, {}) size {} pick {}",
            r[0],
            r[1] as i32,
            r[2] as i32,
            r[3] & 0xff,
            (r[3] >> 8) & 0xff
        );
        let n = r[3] & 0xff;
        assert!(g[0] < n && g[1] < n, "{g:?} off a face of {n}");
        corners +=
            usize::from((r[1] as i32 == -1 || r[1] == n) && (r[2] as i32 == -1 || r[2] == n));
    }
    assert_eq!(
        corners,
        6 * 3 * 4 * 6,
        "every face's four corners at each size and pick"
    );
}
