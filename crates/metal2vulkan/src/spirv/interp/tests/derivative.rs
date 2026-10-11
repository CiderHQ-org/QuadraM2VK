//! Compute derivatives under `DerivativeGroupLinearKHR`: groups of four consecutive local indices
//! at both subgroup widths and in both orders, the unqualified ones (the only ones the lowering
//! emits in a kernel) exact exactly when every implementation choice agrees, the fine ones each
//! lane's own row's and column's, `Fwidth` as `|dx| + |dy|`, vectors per component, and the
//! non-uniform-group and partial-group refusals.

use super::*;
use crate::spirv::ir::{DerivOp, ExecMode};

const WIDTHS: [SubgroupSize; 2] = [SubgroupSize::W32, SubgroupSize::W64];

const ALL: [DerivOp; 3] = [DerivOp::DPdx, DerivOp::DPdy, DerivOp::Fwidth];

fn f32_ty(b: &mut ModuleBuilder) -> TypeId {
    b.ty(Type::Float { width: 32 }).expect("f32")
}

/// A compute module with local size `local` under derivative group `mode`, one `u32` storage
/// buffer, and entry body `body`.
fn deriv_kernel(
    local: [u32; 3],
    mode: ExecMode,
    body: impl FnOnce(&mut FunctionBuilder<'_>, GlobalId) -> BResult<()>,
) -> TypedModule {
    let mut b = ModuleBuilder::new();
    let buf = buffer(&mut b, 0);
    let void = b.ty(Type::Void).expect("void");
    let mut f = b.function(void).expect("function");
    body(&mut f, buf).expect("body");
    if f.current_block().is_some() {
        f.ret(None).expect("ret");
    }
    let id = f.finish().expect("finish");
    let [x, y, z] = local;
    b.entry(
        ExecModel::GLCompute,
        id,
        vec![ExecMode::LocalSize { x, y, z }, mode],
    )
    .expect("entry");
    b.finalize().expect("finalize")
}

/// `P(li)` as an `f32` value, from `li` as `f32` (`p` builds it).
type Operand = fn(&mut FunctionBuilder<'_>, Value) -> BResult<Value>;

/// Each lane stores `op(P)` for every op of [`ALL`] (as bits) at `out[li * ALL.len() + k]`.
fn all_ops(local: u32, p: Operand) -> TypedModule {
    deriv_kernel([local, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
        let li = lid(f);
        let f32_ = f32_ty(f.module());
        let u32_ = u32_ty(f.module());
        let x = f.convert(ConvOp::ConvertUToF, li, f32_)?;
        let v = p(f, x)?;
        let n = c32(f, ALL.len() as u32);
        let base = f.binary(BinOp::IMul, li, n)?;
        for (k, op) in ALL.into_iter().enumerate() {
            let d = f.derivative(op, v)?;
            let bits = f.convert(ConvOp::Bitcast, d, u32_)?;
            let kk = c32(f, k as u32);
            let i = f.binary(BinOp::IAdd, base, kk)?;
            store_at(f, out, i, bits)?;
        }
        Ok(())
    })
}

/// The value of `op` at group position `i` over the group's `P` (its own row and column).
fn want(op: DerivOp, i: usize, q: [f32; 4]) -> f32 {
    let rows = [q[1] - q[0], q[3] - q[2]];
    let cols = [q[2] - q[0], q[3] - q[1]];
    let (dx, dy) = (rows[i >> 1], cols[i & 1]);
    match op {
        DerivOp::DPdx => dx,
        DerivOp::DPdy => dy,
        _ => dx.abs() + dy.abs(),
    }
}

/// Run `m` over one workgroup of `local` lanes at every width and in both orders; the words of
/// buffer 0.
fn run_words(m: &TypedModule, local: u32) -> Vec<u32> {
    let mut out = None;
    for sg in WIDTHS {
        let o = run_both(
            m,
            cfg(1, sg, Order::Forward),
            &buffers(&[local as usize * ALL.len()]),
        );
        let w = words(&o.buffers[0]);
        if let Some(prev) = &out {
            assert_eq!(prev, &w, "the widths disagree");
        }
        out = Some(w);
    }
    out.expect("a width")
}

/// An operand affine in the group position (`P = 3·li + 0.5`): every row differs by 3 and every
/// column by 6, so every kind is determined.
#[test]
fn affine_operands_are_exact_for_every_kind() {
    let m = all_ops(64, |f, x| {
        let f32_ = f32_ty(f.module());
        let three = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(3f32.to_bits())),
        })?;
        let three = f.module().const_value(three)?;
        let half = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(0.5f32.to_bits())),
        })?;
        let half = f.module().const_value(half)?;
        let t = f.binary(BinOp::FMul, x, three)?;
        f.binary(BinOp::FAdd, t, half)
    });
    let w = run_words(&m, 64);
    for li in 0..64usize {
        let q = std::array::from_fn(|k| 3.0 * ((li & !3) + k) as f32 + 0.5);
        for (k, op) in ALL.into_iter().enumerate() {
            let got = f32::from_bits(w[li * ALL.len() + k]);
            assert_eq!(got, want(op, li & 3, q), "lane {li} {op:?}");
        }
    }
    assert_eq!(f32::from_bits(w[0]), 3.0);
    assert_eq!(f32::from_bits(w[1]), 6.0);
    assert_eq!(f32::from_bits(w[2]), 9.0);
}

/// `P = li²`: rows and columns differ, so every kind the implementation chooses traps.
#[test]
fn chosen_derivatives_trap_when_rows_differ() {
    let square: Operand = |f, x| f.binary(BinOp::FMul, x, x);
    for op in ALL {
        let m = deriv_kernel([8, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
            let li = lid(f);
            let f32_ = f32_ty(f.module());
            let u32_ = u32_ty(f.module());
            let x = f.convert(ConvOp::ConvertUToF, li, f32_)?;
            let v = square(f, x)?;
            let d = f.derivative(op, v)?;
            let bits = f.convert(ConvOp::Bitcast, d, u32_)?;
            store_at(f, out, li, bits)
        });
        let e = run(&m, &Config::default(), buffers(&[8])).expect_err("an ambiguous derivative");
        assert!(
            matches!(e.trap_kind(), Some(TrapKind::ImplementationChoice(_))),
            "{op:?}: {e:?}"
        );
    }
}

/// `P = li²` under the fine kinds: rows and columns differ, and each lane reads its own row's x
/// difference and its own column's y difference, with no implementation choice to trap on.
#[test]
fn fine_derivatives_read_their_own_row_and_column() {
    const FINE: [DerivOp; 3] = [DerivOp::DPdxFine, DerivOp::DPdyFine, DerivOp::FwidthFine];
    for (k, op) in FINE.into_iter().enumerate() {
        let m = deriv_kernel([16, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
            let li = lid(f);
            let f32_ = f32_ty(f.module());
            let u32_ = u32_ty(f.module());
            let x = f.convert(ConvOp::ConvertUToF, li, f32_)?;
            let v = f.binary(BinOp::FMul, x, x)?;
            let d = f.derivative(op, v)?;
            let bits = f.convert(ConvOp::Bitcast, d, u32_)?;
            store_at(f, out, li, bits)
        });
        for sg in WIDTHS {
            let o = run_both(&m, cfg(1, sg, Order::Forward), &buffers(&[16]));
            let w = words(&o.buffers[0]);
            for (li, &got) in w.iter().enumerate().take(16) {
                let q: [f32; 4] = std::array::from_fn(|j| (((li & !3) + j) as f32).powi(2));
                let i = li & 3;
                let (dx, dy) = (
                    [q[1] - q[0], q[3] - q[2]][i >> 1],
                    [q[2] - q[0], q[3] - q[1]][i & 1],
                );
                let want = [dx, dy, dx.abs() + dy.abs()][k];
                assert_eq!(f32::from_bits(got), want, "lane {li} {op:?} at {sg:?}");
            }
        }
    }
}

/// A vector operand is differenced per component: `(li, 5)` has y derivative `(2, 0)`.
#[test]
fn vector_operands_are_per_component() {
    let m = deriv_kernel([16, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
        let li = lid(f);
        let f32_ = f32_ty(f.module());
        let u32_ = u32_ty(f.module());
        let vec2 = f.module().ty(Type::Vector { elem: f32_, len: 2 })?;
        let x = f.convert(ConvOp::ConvertUToF, li, f32_)?;
        let five = f.module().constant(Constant::Scalar {
            ty: f32_,
            bits: ConstBits(u64::from(5f32.to_bits())),
        })?;
        let five = f.module().const_value(five)?;
        let v = f.composite_construct(vec2, &[x, five])?;
        let d = f.derivative(DerivOp::DPdy, v)?;
        let two = c32(f, 2);
        let base = f.binary(BinOp::IMul, li, two)?;
        for k in 0..2u32 {
            let c = f.composite_extract(d, &[k])?;
            let bits = f.convert(ConvOp::Bitcast, c, u32_)?;
            let kk = c32(f, k);
            let i = f.binary(BinOp::IAdd, base, kk)?;
            store_at(f, out, i, bits)?;
        }
        Ok(())
    });
    let o = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[32]),
    );
    let w = words(&o.buffers[0]);
    for li in 0..16 {
        assert_eq!(
            (f32::from_bits(w[li * 2]), f32::from_bits(w[li * 2 + 1])),
            (2.0, 0.0),
            "lane {li}"
        );
    }
}

/// Derivatives must run in control flow uniform within the group: half a group in the tangle
/// traps, and so does a group the dispatch does not fill (a local size of 6).
#[test]
fn a_derivative_needs_its_whole_group() {
    let divergent = deriv_kernel([8, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
        let li = lid(f);
        let one = c32(f, 1);
        let zero = c32(f, 0);
        let low = f.binary(BinOp::BitwiseAnd, li, one)?;
        let even = f.binary(BinOp::IEqual, low, zero)?;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let f32_ = f32_ty(fb.module());
            let u32_ = u32_ty(fb.module());
            let x = fb.convert(ConvOp::ConvertUToF, li, f32_)?;
            let d = fb.derivative(DerivOp::DPdx, x)?;
            let bits = fb.convert(ConvOp::Bitcast, d, u32_)?;
            store_at(fb, out, li, bits)?;
            Ok(ArmEnd::FallThrough(vec![]))
        });
        f.if_(&mut (), even, then, None)?;
        Ok(())
    });
    let e = run(&divergent, &Config::default(), buffers(&[8])).expect_err("a split group");
    assert_eq!(e.trap_kind(), Some(&TrapKind::InactiveLane { lane: 1 }));

    let partial = deriv_kernel([6, 1, 1], ExecMode::DerivativeGroupLinear, |f, out| {
        let li = lid(f);
        let f32_ = f32_ty(f.module());
        let u32_ = u32_ty(f.module());
        let x = f.convert(ConvOp::ConvertUToF, li, f32_)?;
        let d = f.derivative(DerivOp::DPdx, x)?;
        let bits = f.convert(ConvOp::Bitcast, d, u32_)?;
        store_at(f, out, li, bits)
    });
    let e = run(&partial, &Config::default(), buffers(&[8])).expect_err("a partial group");
    assert_eq!(e.trap_kind(), Some(&TrapKind::InactiveLane { lane: 6 }));
}
