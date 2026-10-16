//! The emulated tessellator (`lower::tessellation::tessellator`) against its model.
//!
//! [`model`] is the D3D11 reference tessellator reordered as Metal orders it, with Metal's batch
//! across a draw. Its digests over the draw sets below are the probe model's (the reference plus
//! the same order rule, checked on the M3 Max over 1550 random multi-patch draws, every point,
//! triangle, first corner and order bit for bit). The emitters run on the CPU SPIR-V interpreter
//! over random patches of every pipeline shape and must give the model's processed factors,
//! counts, batch fills, tile words and corners' coordinates, bit for bit.

pub mod model;

use model::{Draw, Model, Part, Rng};

use crate::lower::tessellation::tessellator::{
    emit_decode, emit_place, emit_process, emit_stream_step, emit_triangle_count,
    max_triangle_count, Facts,
};
use crate::metal::{TessellationPartitionMode, Winding};
use crate::pipeline::interface::PatchDomain;
use crate::pipeline::sym::{ResourceKind, ResourceSlot, Stage};
use crate::spirv::builder::{FunctionBuilder, ModuleBuilder};
use crate::spirv::error::BResult;
use crate::spirv::ids::{GlobalId, TypeId, Value};
use crate::spirv::interp::{run, Buffer, Config, Resources};
use crate::spirv::ir::{
    natural_layout, BinOp, ConvOp, ExecMode, ExecModel, IoDir, LayoutRule, MemAccess, StorageClass,
    Type, TypedModule,
};

/// `s_fixedReciprocal`, the reference's table.
const RECIPROCAL: [u32; 65] = [
    0xffff_ffff,
    0x10000,
    0x8000,
    0x5555,
    0x4000,
    0x3333,
    0x2aab,
    0x2492,
    0x2000,
    0x1c72,
    0x199a,
    0x1746,
    0x1555,
    0x13b1,
    0x1249,
    0x1111,
    0x1000,
    0xf0f,
    0xe39,
    0xd79,
    0xccd,
    0xc31,
    0xba3,
    0xb21,
    0xaab,
    0xa3d,
    0x9d9,
    0x97b,
    0x925,
    0x8d4,
    0x889,
    0x842,
    0x800,
    0x7c2,
    0x788,
    0x750,
    0x71c,
    0x6eb,
    0x6bd,
    0x690,
    0x666,
    0x63e,
    0x618,
    0x5f4,
    0x5d1,
    0x5b0,
    0x591,
    0x572,
    0x555,
    0x539,
    0x51f,
    0x505,
    0x4ec,
    0x4d5,
    0x4be,
    0x4a8,
    0x492,
    0x47e,
    0x46a,
    0x457,
    0x444,
    0x432,
    0x421,
    0x410,
    0x400,
];

#[test]
fn reciprocals_are_the_reference_table() {
    for (n, &r) in RECIPROCAL.iter().enumerate().skip(1) {
        assert_eq!(model::reciprocal(n as u32), r, "1/{n}");
    }
}

/// The probe model's digests over the same draws: the D3D11 reference plus the order rule, as
/// the probe checked it on the M3 Max.
#[test]
fn model_is_the_probe_model() {
    assert_eq!(
        model::digest(&model::uniform_draws(true)),
        0xa1cc_1dd4_52d5_6b9c
    );
    assert_eq!(
        model::digest(&model::uniform_draws(false)),
        0xfe37_b8db_a400_72e8
    );
    assert_eq!(
        model::digest(&model::random_draws(20_261_016, 1000)),
        0x3991_4371_a9e9_0d35
    );
}

fn facts_of(m: &Model) -> Facts {
    Facts {
        domain: if m.quad {
            PatchDomain::Quad
        } else {
            PatchDomain::Triangle
        },
        partition: match m.part {
            Part::Pow2 => TessellationPartitionMode::Pow2,
            Part::Integer => TessellationPartitionMode::Integer,
            Part::FractionalOdd => TessellationPartitionMode::FractionalOdd,
            Part::FractionalEven => TessellationPartitionMode::FractionalEven,
        },
        winding: if m.cw {
            Winding::Clockwise
        } else {
            Winding::CounterClockwise
        },
        max_factor: m.max_factor,
        scale_enabled: m.scale_enabled,
    }
}

#[test]
fn counts_stay_within_the_maximum() {
    let draws = model::random_draws(7, 3000);
    for d in &draws {
        let max = max_triangle_count(&facts_of(&d.model));
        for halves in &d.patches {
            let p = d.model.process(halves, d.scale_value());
            let n = d.model.triangle_count(&p);
            assert!(n <= max, "{} triangles past {max}: {}", n, d.line());
        }
    }
    // Every factor at the most the pipeline allows reaches the bound.
    for quad in [true, false] {
        for part in [
            Part::Pow2,
            Part::Integer,
            Part::FractionalOdd,
            Part::FractionalEven,
        ] {
            for max_factor in [64, 48, 16, 15, 3] {
                let m = Model {
                    quad,
                    part,
                    cw: true,
                    max_factor,
                    scale_enabled: false,
                };
                let p = m.process(&[0x7bff; 6], 1.0);
                assert_eq!(
                    m.triangle_count(&p),
                    max_triangle_count(&facts_of(&m)),
                    "{part:?} max {max_factor} quad {quad}"
                );
            }
        }
    }
}

/// The binder pushes an infinite scale as `2^32`: every factor's level is the one an infinite
/// scale gives.
#[test]
fn infinite_scale_is_two_to_the_32() {
    for quad in [true, false] {
        for part in [
            Part::Pow2,
            Part::Integer,
            Part::FractionalOdd,
            Part::FractionalEven,
        ] {
            let m = Model {
                quad,
                part,
                cw: true,
                max_factor: 64,
                scale_enabled: true,
            };
            for h in 0..=u16::MAX {
                let halves = [h, 0x4400, 0x4400, 0x4400, h, 0x4400];
                assert_eq!(
                    m.process(&halves, f32::INFINITY),
                    m.process(&halves, 4_294_967_296.0),
                    "{part:?} {h:04x}"
                );
            }
        }
    }
}

// ---- The emitters on the interpreter ----

/// Input words per case: 6 halves, the scale's bits, the fill before, a triangle index.
const IN: usize = 9;
/// Output words per case: 6 processed factors, the count, the fill after, the tile word, and the
/// triangle's corners' `u`, `v`, `w` bits.
const OUT: usize = 18;

fn slot(index: u32) -> ResourceSlot {
    ResourceSlot {
        stage: Stage::Kernel,
        kind: ResourceKind::Buffer,
        index,
    }
}

fn words(b: &mut ModuleBuilder, index: u32, u32_: TypeId) -> BResult<GlobalId> {
    let rta = b.ty(Type::RuntimeArray {
        elem: u32_,
        stride: Some(4),
    })?;
    let layout = natural_layout(b.types(), LayoutRule::Std430, true, &[rta])?;
    let block = b.ty(Type::Struct {
        members: [rta].into(),
        layout: Some(layout),
    })?;
    b.resource(slot(index), StorageClass::StorageBuffer, block)
}

fn konst(f: &mut FunctionBuilder<'_>, v: u32) -> BResult<Value> {
    let c = f.u32_const(v)?;
    f.module().const_value(c)
}

/// One invocation per case: process, count, step, decode the case's triangle and place its
/// corners.
fn kernel(facts: &Facts) -> BResult<TypedModule> {
    let mut b = ModuleBuilder::new();
    let u32_ = b.ty(Type::Int {
        width: 32,
        signed: false,
    })?;
    let f32_ = b.ty(Type::Float { width: 32 })?;
    let uvec3 = b.ty(Type::Vector { elem: u32_, len: 3 })?;
    let input = words(&mut b, 0, u32_)?;
    let output = words(&mut b, 1, u32_)?;
    let invocation = b.builtin(crate::spv::BuiltIn::GlobalInvocationId, IoDir::Input, uvec3)?;
    let void = b.ty(Type::Void)?;
    let mut f = b.function(void)?;
    let zero = konst(&mut f, 0)?;
    let inv = f.module().global_value(invocation)?;
    let inv = f.load(inv, MemAccess::default())?;
    let g = f.composite_extract(inv, &[0])?;
    let input = f.module().global_value(input)?;
    let output = f.module().global_value(output)?;
    let in_base = konst(&mut f, IN as u32)?;
    let in_base = f.binary(BinOp::IMul, g, in_base)?;
    let out_base = konst(&mut f, OUT as u32)?;
    let out_base = f.binary(BinOp::IMul, g, out_base)?;
    let mut read = Vec::new();
    for n in 0..IN as u32 {
        let n = konst(&mut f, n)?;
        let at = f.binary(BinOp::IAdd, in_base, n)?;
        let p = f.access_chain(input, &[zero, at])?;
        read.push(f.load(p, MemAccess::default())?);
    }
    let scale = f.convert(ConvOp::Bitcast, read[6], f32_)?;
    let p = emit_process(&mut f, facts, &read[..6], scale)?;
    let count = emit_triangle_count(&mut f, facts, &p)?;
    let (after, tiles) = emit_stream_step(&mut f, facts, read[7], &p)?;
    let mut out: Vec<Value> = p.to_vec();
    out.extend([count, after, tiles]);
    // A triangle past the patch (a culled one) is not decoded.
    let ids = emit_decode(&mut f, facts, read[8], &p, tiles)?;
    for id in ids {
        for x in emit_place(&mut f, facts, id, &p)? {
            out.push(f.convert(ConvOp::Bitcast, x, u32_)?);
        }
    }
    for (n, v) in out.into_iter().enumerate() {
        let n = konst(&mut f, n as u32)?;
        let at = f.binary(BinOp::IAdd, out_base, n)?;
        let ptr = f.access_chain(output, &[zero, at])?;
        f.store(ptr, v, MemAccess::default())?;
    }
    f.ret(None)?;
    let id = f.finish()?;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x: 1, y: 1, z: 1 }],
    )?;
    b.finalize()
}

/// One case: a patch, the fill before it, and one of its triangles.
struct Case {
    halves: [u16; 6],
    used: u32,
    l: u32,
}

/// What the model says of a case.
fn expect(m: &Model, scale: f32, c: &Case) -> [u32; OUT] {
    let p = m.process(&c.halves, scale);
    let count = m.triangle_count(&p);
    let (after, tiles) = m.stream_step(c.used, &p);
    let mut out = [0u32; OUT];
    out[..6].copy_from_slice(&p);
    out[6] = count;
    out[7] = after;
    out[8] = tiles;
    if c.l < count {
        for (n, point) in m.decode(c.l, &p, tiles).into_iter().enumerate() {
            let xs: [f32; 3] = m.place(point, &p);
            for (k, x) in xs.into_iter().enumerate() {
                out[9 + 3 * n + k] = x.to_bits();
            }
        }
    }
    out
}

/// Run `cases` of pipeline `m` at encoder scale `scale` through the emitters and compare.
fn check(m: &Model, scale: f32, cases: &[Case]) {
    let facts = facts_of(m);
    let module = kernel(&facts).expect("tessellator kernel builds");
    let mut input = Vec::new();
    for c in cases {
        let h = c.halves;
        let words = [
            u32::from(h[0]),
            u32::from(h[1]),
            u32::from(h[2]),
            u32::from(h[3]),
            u32::from(h[4]),
            u32::from(h[5]),
            scale.to_bits(),
            c.used,
            c.l,
        ];
        input.extend(words.iter().flat_map(|w| w.to_le_bytes()));
    }
    let res = Resources {
        buffers: vec![
            Buffer {
                slot: Some(slot(0)),
                address: None,
                bytes: input,
            },
            Buffer {
                slot: Some(slot(1)),
                address: None,
                bytes: vec![0; cases.len() * OUT * 4],
            },
        ],
        push: Vec::new(),
        spec: Default::default(),
    };
    let cfg = Config {
        workgroups: [cases.len() as u32, 1, 1],
        step_limit: u64::MAX,
        ..Config::default()
    };
    let outcome = run(&module, &cfg, res).expect("tessellator kernel runs");
    let (words, _) = outcome.buffers[1].as_chunks::<4>();
    let got: Vec<u32> = words.iter().map(|&b| u32::from_le_bytes(b)).collect();
    for (n, c) in cases.iter().enumerate() {
        let want = expect(m, scale, c);
        let mut have = [0u32; OUT];
        have.copy_from_slice(&got[n * OUT..(n + 1) * OUT]);
        if want[6] <= c.l {
            // No triangle to decode: only the patch's words.
            have[9..].fill(0);
        }
        assert_eq!(
            have, want,
            "{m:?} scale {scale}: halves {:04x?} fill {} triangle {}",
            c.halves, c.used, c.l
        );
    }
}

/// Random cases of `d`'s pipeline: each patch at a random fill (and at the fill its draw
/// carries), with random triangles, plus every triangle of a few patches.
fn cases_of(rng: &mut Rng, d: &Draw, every: bool) -> Vec<Case> {
    let m = &d.model;
    let mut out = Vec::new();
    let mut used = 0;
    for halves in &d.patches {
        let mut h = [0u16; 6];
        h[..halves.len()].copy_from_slice(halves);
        let p = m.process(&h, d.scale_value());
        let count = m.triangle_count(&p);
        let fills = [used, rng.below(513), rng.pick(&[0, 1, 500, 508, 511, 512])];
        for (n, &fill) in fills.iter().enumerate() {
            if every && n == 0 && count <= 600 {
                for l in 0..count {
                    out.push(Case {
                        halves: h,
                        used: fill,
                        l,
                    });
                }
            } else {
                out.push(Case {
                    halves: h,
                    used: fill,
                    l: if count == 0 { 0 } else { rng.below(count) },
                });
            }
        }
        used = m.stream_step(used, &p).0;
    }
    out
}

#[test]
fn emitters_match_the_model() {
    let mut rng = Rng(0x7e55);
    let draws = model::random_draws(0x5eed, 300);
    // Group the draws by pipeline: one kernel each.
    let mut groups: Vec<(Model, Option<u16>, Vec<&Draw>)> = Vec::new();
    for d in &draws {
        match groups
            .iter_mut()
            .find(|(m, s, _)| *m == d.model && *s == d.scale)
        {
            Some((_, _, ds)) => ds.push(d),
            None => groups.push((d.model, d.scale, vec![d])),
        }
    }
    for (m, _, ds) in &groups {
        let scale = ds[0].scale_value();
        let mut cases = Vec::new();
        for (n, d) in ds.iter().enumerate() {
            cases.extend(cases_of(&mut rng, d, n == 0 && d.patches.len() <= 2));
        }
        check(m, scale, &cases);
    }
}

/// A culled patch draws nothing, leaves the fill as it was and has tile word 0 (Metal draws
/// nothing for a draw whose only patch is culled: the probe ran no vertex).
#[test]
fn culled_patches_draw_nothing() {
    for quad in [true, false] {
        for part in [
            Part::Pow2,
            Part::Integer,
            Part::FractionalOdd,
            Part::FractionalEven,
        ] {
            let m = Model {
                quad,
                part,
                cw: false,
                max_factor: 64,
                scale_enabled: false,
            };
            let mut cases = Vec::new();
            for edge in [0x0000, 0x8000, 0xbc00, 0x7e00, 0xfc00, 0xfe00] {
                for used in [0, 17, 512] {
                    let mut h = [0x4800; 6];
                    h[1] = edge;
                    cases.push(Case {
                        halves: h,
                        used,
                        l: 0,
                    });
                }
            }
            for c in &cases {
                let p = m.process(&c.halves, 1.0);
                assert_eq!(p, [0; 6]);
                assert_eq!(m.triangle_count(&p), 0);
                assert_eq!(m.stream_step(c.used, &p), (c.used, 0));
            }
            check(&m, 1.0, &cases);
        }
    }
}
