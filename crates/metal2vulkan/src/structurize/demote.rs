//! P4: demotion of block-crossing SSA values.
//!
//! After P4 no SSA value crosses a block: every value the Plan lists in
//! [`CfgView::cross_block_values`] lives in its own `Function` variable (stored right after its
//! definition, loaded once at the top of each using block), and every phi in
//! [`CfgView::phis`] gets an incoming variable that each predecessor writes as a parallel copy
//! just before its terminator. That is what lets P5–P7 add blocks and reroute edges as plain graph
//! rewrites: nothing they do can break SSA dominance.
//!
//! The structurizer never sees instructions, so P4 is the demotion *plan*: one [`DemoteVar`] per
//! crossing value and one [`PhiVar`] per phi, with the Plan's demotion type. P8's `CodeEmitter`
//! emits the loads and stores while it emits each block. Every [`DemoteTy`] is demotable by
//! construction, so control's `NonDemotableMerge` cannot arise here. A view that lists a value
//! twice breaks its contract: `Error::Internal(Invariant)`.

use super::cfg::{invariant, with_scratch, Scratch};
use super::loops::Kept;
use super::CfgView;
use crate::air::ir::ValueId;
use crate::budget::Meter;
use crate::error::Error;
use crate::spirv::ids::DemoteTy;

/// The `Function` variable of one block-crossing value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DemoteVar {
    pub value: ValueId,
    pub ty: DemoteTy,
}

/// The incoming variable of one phi.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PhiVar {
    pub phi: ValueId,
    pub ty: DemoteTy,
}

/// P4's result: the variables in the view's order.
#[derive(Debug)]
pub(super) struct Demoted {
    pub(super) demoted: Vec<DemoteVar>,
    pub(super) phis: Vec<PhiVar>,
    pub(super) ledger_bytes: usize,
}

/// P4. On error nothing stays charged to `meter`'s ledger.
pub(super) fn demote(view: &impl CfgView, meter: &mut Meter) -> Result<Demoted, Error> {
    let (cross, phis) = (view.cross_block_values(), view.phis());
    let mut kept = Kept::default();
    let result = with_scratch(meter, |meter, scratch| {
        distinct(
            cross.iter().map(|(v, _)| *v),
            cross.len(),
            "demoted",
            meter,
            scratch,
        )?;
        distinct(
            phis.iter().map(|(v, _)| *v),
            phis.len(),
            "phi",
            meter,
            scratch,
        )?;
        let mut stack = Vec::new();
        let mut demoted = kept.with_capacity(meter, cross.len())?;
        for (value, ty) in cross {
            let ty = kept_ty(ty, &mut stack, &mut kept, meter, scratch)?;
            demoted.push(DemoteVar { value: *value, ty });
        }
        let mut phi_vars = kept.with_capacity(meter, phis.len())?;
        for (phi, ty) in phis {
            let ty = kept_ty(ty, &mut stack, &mut kept, meter, scratch)?;
            phi_vars.push(PhiVar { phi: *phi, ty });
        }
        Ok((demoted, phi_vars))
    });
    match result {
        Ok((demoted, phis)) => Ok(Demoted {
            demoted,
            phis,
            ledger_bytes: kept.bytes(),
        }),
        Err(e) => {
            meter.release(kept.bytes());
            Err(e)
        }
    }
}

/// Refuse a value listed twice.
fn distinct(
    values: impl Iterator<Item = ValueId>,
    len: usize,
    what: &str,
    meter: &mut Meter,
    scratch: &mut Scratch,
) -> Result<(), Error> {
    let mut sorted = scratch.with_capacity(meter, len)?;
    sorted.extend(values);
    meter.tick(u32::try_from(len).unwrap_or(u32::MAX))?;
    sorted.sort_unstable();
    if let Some(w) = sorted.windows(2).find(|w| w.first() == w.get(1)) {
        return Err(invariant(format!(
            "{what} value {:?} listed twice",
            w.first()
        )));
    }
    Ok(())
}

/// Clone `ty`, charging the heap of its composite members first (measured with an explicit
/// stack, so a deep type does not recurse here).
fn kept_ty<'t>(
    ty: &'t DemoteTy,
    stack: &mut Vec<&'t DemoteTy>,
    kept: &mut Kept,
    meter: &mut Meter,
    scratch: &mut Scratch,
) -> Result<DemoteTy, Error> {
    meter.tick(1)?;
    let mut heap = 0usize;
    stack.clear();
    scratch.push(meter, stack, ty)?;
    while let Some(t) = stack.pop() {
        if let DemoteTy::Composite(members) = t {
            meter.tick(1)?;
            heap = heap.saturating_add(std::mem::size_of_val::<[DemoteTy]>(members));
            for m in members.iter() {
                scratch.push(meter, stack, m)?;
            }
        }
    }
    kept.charge(meter, heap)?;
    Ok(ty.clone())
}
