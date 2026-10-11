//! Subgroup operations at widths 16, 32, and 64: ballots, reductions, scans, shuffles, and quads over
//! the active tangle, maximal reconvergence after selections and loops, and the
//! inactive-lane and non-uniform-barrier traps.

use super::*;
use crate::spirv::ir::QuadDirection;

const WIDTHS: [SubgroupSize; 3] = [SubgroupSize::W16, SubgroupSize::W32, SubgroupSize::W64];

fn to_u32(f: &mut FunctionBuilder<'_>, b: Value) -> BResult<Value> {
    let one = c32(f, 1);
    let zero = c32(f, 0);
    f.select(b, one, zero)
}

/// `out[li * stride + k] = values[k]`.
fn store_row(f: &mut FunctionBuilder<'_>, out: GlobalId, li: Value, vals: &[Value]) -> BResult<()> {
    let stride = c32(f, vals.len() as u32);
    let base = f.binary(BinOp::IMul, li, stride)?;
    for (k, v) in vals.iter().enumerate() {
        let kk = c32(f, k as u32);
        let i = f.binary(BinOp::IAdd, base, kk)?;
        store_at(f, out, i, *v)?;
    }
    Ok(())
}

#[test]
fn uniform_subgroup_operations() {
    let m = kernel([64, 1, 1], 1, |f, b| {
        let li = lid(f);
        let three = c32(f, 3);
        let zero = c32(f, 0);
        let r = f.binary(BinOp::UMod, li, three)?;
        let pred = f.binary(BinOp::IEqual, r, zero)?;
        let ballot = f.group_ballot(pred)?;
        let w0 = f.composite_extract(ballot, &[0])?;
        let w1 = f.composite_extract(ballot, &[1])?;
        let sum = f.group_arith(GroupArith::IAdd, GroupMode::Reduce, li)?;
        let one = c32(f, 1);
        let incl = f.group_arith(GroupArith::IAdd, GroupMode::InclusiveScan, one)?;
        let excl = f.group_arith(GroupArith::IAdd, GroupMode::ExclusiveScan, li)?;
        let xor = f.group_shuffle(ShuffleOp::Xor, li, one)?;
        let first = f.group_broadcast_first(li)?;
        let elect = f.group_elect()?;
        let elect = to_u32(f, elect)?;
        let quad = f.group_arith(GroupArith::IAdd, GroupMode::Clustered(4), li)?;
        let swap = f.group_quad_swap(li, QuadDirection::Diagonal)?;
        let umax = f.group_arith(GroupArith::UMax, GroupMode::Reduce, li)?;
        store_row(
            f,
            b[0],
            li,
            &[w0, w1, sum, incl, excl, xor, first, elect, quad, swap, umax],
        )
    });
    for sg in WIDTHS {
        let n = sg.lanes();
        let out = run_both(&m, cfg(1, sg, Order::Forward), &buffers(&[64 * 11]));
        let w = words(&out.buffers[0]);
        for li in 0..64u32 {
            let base = li / n * n;
            let lane = li % n;
            let members: Vec<u32> = (base..base + n).collect();
            let mut ballot = 0u64;
            for m in &members {
                if m % 3 == 0 {
                    ballot |= 1 << (m - base);
                }
            }
            let row = &w[(li * 11) as usize..(li * 11 + 11) as usize];
            let want = [
                ballot as u32,
                (ballot >> 32) as u32,
                members.iter().sum::<u32>(),
                lane + 1,
                (base..li).sum::<u32>(),
                li ^ 1,
                base,
                u32::from(lane == 0),
                (li & !3..(li & !3) + 4).sum::<u32>(),
                li ^ 3,
                base + n - 1,
            ];
            assert_eq!(row, &want, "lane {li} at width {n}");
        }
    }
}

#[test]
fn divergent_selection_sees_only_active_lanes_and_reconverges() {
    let m = kernel([64, 1, 1], 1, |f, b| {
        let li = lid(f);
        let one = c32(f, 1);
        let zero = c32(f, 0);
        let low = f.binary(BinOp::BitwiseAnd, li, one)?;
        let even = f.binary(BinOp::IEqual, low, zero)?;
        let then: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let one = c32(fb, 1);
            let n = fb.group_arith(GroupArith::IAdd, GroupMode::Reduce, one)?;
            Ok(ArmEnd::FallThrough(vec![n]))
        });
        let els: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let two = c32(fb, 2);
            let n = fb.group_arith(GroupArith::IAdd, GroupMode::Reduce, two)?;
            Ok(ArmEnd::FallThrough(vec![n]))
        });
        let arm = f.if_(&mut (), even, then, Some(els))?.0[0];
        let after = f.group_arith(GroupArith::IAdd, GroupMode::Reduce, one)?;
        store_row(f, b[0], li, &[arm, after])
    });
    for sg in WIDTHS {
        let n = sg.lanes();
        let out = run_both(&m, cfg(1, sg, Order::Forward), &buffers(&[128]));
        let w = words(&out.buffers[0]);
        for li in 0..64usize {
            let arm = if li % 2 == 0 { n / 2 } else { n };
            assert_eq!(&w[li * 2..li * 2 + 2], &[arm, n], "lane {li} width {n}");
        }
    }
}

#[test]
fn lanes_leaving_a_loop_early_wait_at_its_merge() {
    // Lane li runs li % 4 iterations; each iteration adds the count of lanes still looping.
    let m = kernel([64, 1, 1], 1, |f, b| {
        let li = lid(f);
        let u32_ = u32_ty(f.module());
        let four = c32(f, 4);
        let n = f.binary(BinOp::UMod, li, four)?;
        let acc = f.local_var(u32_, None)?;
        let accp = f.var_ptr(acc)?;
        let zero = c32(f, 0);
        f.store(accp, zero, MemAccess::default())?;
        counted(f, &mut (), n, move |fb, _, _| {
            let one = c32(fb, 1);
            let active = fb.group_arith(GroupArith::IAdd, GroupMode::Reduce, one)?;
            let a = fb.load(accp, MemAccess::default())?;
            let s = fb.binary(BinOp::IAdd, a, active)?;
            fb.store(accp, s, MemAccess::default())
        })?;
        let a = f.load(accp, MemAccess::default())?;
        let one = c32(f, 1);
        let after = f.group_arith(GroupArith::IAdd, GroupMode::Reduce, one)?;
        store_row(f, b[0], li, &[a, after])
    });
    for sg in WIDTHS {
        let n = sg.lanes();
        let q = n / 4;
        let out = run_both(&m, cfg(1, sg, Order::Forward), &buffers(&[128]));
        let w = words(&out.buffers[0]);
        for li in 0..64usize {
            // Iteration k runs the lanes with li % 4 > k: 3q, 2q, then q lanes.
            let want = [0, 3 * q, 5 * q, 6 * q][li % 4];
            assert_eq!(&w[li * 2..li * 2 + 2], &[want, n], "lane {li} width {n}");
        }
    }
}

#[test]
fn reading_an_inactive_lane_traps() {
    let m = kernel([32, 1, 1], 1, |f, b| {
        let li = lid(f);
        let sixteen = c32(f, 16);
        let low = f.binary(BinOp::ULessThan, li, sixteen)?;
        let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
            let sixteen = c32(fb, 16);
            let v = fb.group_shuffle(ShuffleOp::Xor, li, sixteen)?;
            store_at(fb, b[0], li, v)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), low, arm, None)?;
        Ok(())
    });
    assert_eq!(
        trap_of(run(&m, &Config::default(), buffers(&[32]))),
        TrapKind::InactiveLane { lane: 16 }
    );
}

/// Under `MaximallyReconvergesKHR` a subgroup barrier orders its tangle, the lanes that reach it
/// together, wherever they diverged (SPIR-V requires every invocation of the scope only at
/// `Workgroup` scope and wider): the low half's barrier runs, and each low lane reads its
/// neighbour's word as lockstep does, before any is overwritten.
#[test]
fn a_divergent_subgroup_barrier_orders_its_tangle_under_maximal_reconvergence() {
    let m = kernel_with(
        [32, 1, 1],
        1,
        vec![ExecMode::MaximallyReconverges],
        |f, b| {
            let li = lid(f);
            store_at(f, b[0], li, li)?;
            let sixteen = c32(f, 16);
            let low = f.binary(BinOp::ULessThan, li, sixteen)?;
            let buf = b[0];
            let arm: ArmFn<'_, ()> = Box::new(move |fb, _| {
                let one = c32(fb, 1);
                let other = fb.binary(BinOp::BitwiseXor, li, one)?;
                let v = load_at(fb, buf, other)?;
                fb.control_barrier(Scope::Subgroup, Scope::Subgroup, Semantics::RELAXED)?;
                store_at(fb, buf, li, v)?;
                Ok(ArmEnd::FallThrough(Vec::new()))
            });
            f.if_(&mut (), low, arm, None)?;
            Ok(())
        },
    );
    let out = run_both(&m, Config::default(), &buffers(&[32]));
    let w = words(&out.buffers[0]);
    for li in 0..32u32 {
        let want = if li < 16 { li ^ 1 } else { li };
        assert_eq!(w[li as usize], want, "lane {li}");
    }
}

#[test]
fn a_divergent_subgroup_barrier_traps() {
    let m = kernel([32, 1, 1], 0, |f, _| {
        let li = lid(f);
        let sixteen = c32(f, 16);
        let low = f.binary(BinOp::ULessThan, li, sixteen)?;
        let arm: ArmFn<'_, ()> = Box::new(|fb, _| {
            fb.control_barrier(Scope::Subgroup, Scope::Subgroup, Semantics::RELAXED)?;
            Ok(ArmEnd::FallThrough(Vec::new()))
        });
        f.if_(&mut (), low, arm, None)?;
        Ok(())
    });
    assert_eq!(
        trap_of(run(&m, &Config::default(), Resources::default())),
        TrapKind::NonUniformBarrier
    );
}

#[test]
fn bool_reductions() {
    let m = kernel([64, 1, 1], 1, |f, b| {
        let li = lid(f);
        let forty = c32(f, 40);
        let p = f.binary(BinOp::ULessThan, li, forty)?;
        let all = f.group_vote(true, p)?;
        let any = f.group_vote(false, p)?;
        let all = to_u32(f, all)?;
        let any = to_u32(f, any)?;
        store_row(f, b[0], li, &[all, any])
    });
    let out = run_both(
        &m,
        cfg(1, SubgroupSize::W32, Order::Forward),
        &buffers(&[128]),
    );
    let w = words(&out.buffers[0]);
    for li in 0..64usize {
        let want: [u32; 2] = if li < 32 { [1, 1] } else { [0, 1] };
        assert_eq!(&w[li * 2..li * 2 + 2], &want);
    }
}
