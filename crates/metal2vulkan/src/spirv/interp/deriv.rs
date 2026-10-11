//! Compute derivatives over the active tangle of one SIMT step
//! (`SPV_KHR_compute_shader_derivatives`).
//!
//! **Derivative groups.** Under `DerivativeGroupLinearKHR` (the mode the lowering emits) a
//! derivative group is four invocations with consecutive local invocation indices `4k + i`, and
//! `i` is the invocation's position in the Vulkan quad (0 top-left, 1 top-right, 2 bottom-left,
//! 3 bottom-right). Subgroups are runs of 32 or 64 consecutive local indices (`sched`), so a group
//! is the aligned four lanes `L & !3` of one subgroup. Explicit derivatives must execute in
//! control flow uniform within the group: a group lane outside the tangle, or one the dispatch
//! does not launch (a local size that is not a multiple of four), is
//! [`TrapKind::InactiveLane`]. `DerivativeGroupQuadsKHR` groups 2×2 local ids, which may span
//! subgroups the scheduler steps apart, so it is [`Unmodelled`]; a compute derivative without
//! either mode is invalid SPIR-V.
//!
//! **Values** (Vulkan "Derivative Operations"). With `P0..P3` the operand in group positions
//! `0..3` and every difference rounded to `f32`, fine derivatives along x are `P1 − P0` in the
//! top row (positions 0, 1) and `P3 − P2` in the bottom row, and along y `P2 − P0` in the left
//! column (positions 0, 2) and `P3 − P1` in the right one. A coarse derivative may reuse one
//! row's (or column's) difference for the whole group, and the unqualified `OpDPdx`, `OpDPdy`,
//! and `OpFwidth` may be either kind, so their result is either row's (column's) difference at
//! the implementation's choice. The interpreter is an oracle: it returns the value when every
//! allowed choice gives the same bits (both rows differ equally, as for any operand affine in
//! the group position) and traps [`TrapKind::ImplementationChoice`] otherwise. `OpDPdxFine`,
//! `OpDPdyFine` and `OpFwidthFine` have one answer, the position's own row's (column's)
//! difference. `Fwidth*` is `|dx| + |dy|` of the same kind, with the unqualified kinds chosen
//! consistently.

use crate::spirv::ids::Value;
use crate::spirv::ir::{DerivOp, ExecMode};

use super::float;
use super::sched::{Lane, Prog};
use super::value::{Kind, Val};
use super::{malformed, IResult, TrapKind};

/// Execute derivative `op` of `a` for `lanes`, the tangle in scheduling order. Returns one
/// result per lane, in the same order.
pub(super) fn exec(p: &Prog<'_>, op: DerivOp, a: Value, lanes: &[&Lane]) -> IResult<Vec<Val>> {
    let modes = &p.m.entry().modes;
    if !modes.contains(&ExecMode::DerivativeGroupLinear) {
        return Err(malformed(
            "a compute derivative without a derivative group mode",
        ));
    }
    let n = p.t.num_of(a)?;
    if n.kind != Kind::Float || n.width != 32 {
        return Err(malformed(
            "a derivative of a value that is not 32-bit float",
        ));
    }
    // Operand components per subgroup lane.
    let vals: Vec<(u32, Vec<u64>)> = lanes
        .iter()
        .map(|l| Ok((l.sg_lane, l.get(p, a)?.comps()?)))
        .collect::<IResult<_>>()?;
    let at = |lane: u32| -> IResult<&[u64]> {
        vals.iter()
            .find(|(id, _)| *id == lane)
            .map(|(_, c)| c.as_slice())
            .ok_or_else(|| {
                TrapKind::InactiveLane {
                    lane: i64::from(lane),
                }
                .err()
            })
    };
    vals.iter()
        .map(|(id, _)| {
            let base = id & !3;
            let (p0, p1, p2, p3) = (at(base)?, at(base + 1)?, at(base + 2)?, at(base + 3)?);
            let comps = p0
                .iter()
                .zip(p1)
                .zip(p2)
                .zip(p3)
                .map(|(((x0, x1), x2), x3)| component(op, [*x0, *x1, *x2, *x3], id & 3))
                .collect::<IResult<Vec<u64>>>()?;
            Ok(Val::from_comps(comps))
        })
        .collect()
}
/// One component of `op` over the group's operand bits `q` at group position `at`: a fine
/// derivative's is its own row's (column's) difference, an unqualified one's the same at every
/// group position.
fn component(op: DerivOp, q: [u64; 4], at: u32) -> IResult<u64> {
    let sub = |x: u64, y: u64| float::sub(32, x, y);
    let [p0, p1, p2, p3] = q;
    let rows = [sub(p1, p0)?, sub(p3, p2)?];
    let cols = [sub(p2, p0)?, sub(p3, p1)?];
    let (x, y, fine) = match op {
        DerivOp::DPdx => (true, false, false),
        DerivOp::DPdy => (false, true, false),
        DerivOp::Fwidth => (true, true, false),
        DerivOp::DPdxFine => (true, false, true),
        DerivOp::DPdyFine => (false, true, true),
        DerivOp::FwidthFine => (true, true, true),
    };
    // The allowed differences along each axis: either row's (column's), as the implementation
    // chooses for an unqualified derivative; the position's own for a fine one (positions 0 and
    // 1 are the top row, 0 and 2 the left column).
    let (row, col) = ((at >> 1) as usize, (at & 1) as usize);
    let (xs, ys): (&[u64], &[u64]) = match fine {
        true => (&rows[row..=row], &cols[col..=col]),
        false => (&rows, &cols),
    };
    let abs = |v: u64| v & !(1u64 << 31);
    let mut candidates = Vec::with_capacity(4);
    match (x, y) {
        (true, false) => candidates.extend_from_slice(xs),
        (false, true) => candidates.extend_from_slice(ys),
        _ => {
            for &dx in xs {
                for &dy in ys {
                    candidates.push(float::add(32, abs(dx), abs(dy))?);
                }
            }
        }
    }
    let first = candidates
        .first()
        .copied()
        .ok_or_else(|| malformed("a derivative without candidates"))?;
    let nan = |v: u64| f32::from_bits(v as u32).is_nan();
    if candidates
        .iter()
        .all(|&v| v == first || (nan(v) && nan(first)))
    {
        Ok(first)
    } else {
        Err(
            TrapKind::ImplementationChoice("a coarse or unqualified derivative whose rows differ")
                .err(),
        )
    }
}
