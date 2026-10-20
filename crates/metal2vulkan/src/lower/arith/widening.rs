//! The `half` loads read from their bits by the device's widening: where the device's widening is
//! Metal's and no use of the value observes a NaN's bits.
//!
//! A `half` is held as the `float` of its value. Its decode from loaded bits ([`from_bits`])
//! is exact integer arithmetic on every device, about 20 instructions a vector, and keeps a NaN's
//! payload, so a value moved to memory stores the bits Metal moves. The device's widening (an
//! `OpFConvert` to 32 bits of the bits read as a 16-bit float) is a few, and takes every `half` but
//! a NaN to its value where the profile takes `DriverDeviation::MetalHalfArithmetic`: that
//! measurement reads every `half` sum the device computes through the device's widening, `x + -0`
//! of every `x` among them, and Metal's own widening, read over every bit pattern on the M3 Max in
//! fast and safe math, is the same (a NaN gives `0x7FC00000`). Only a NaN's payload differs.
//!
//! So a load of `half`s (a scalar or a vector) takes the device's widening
//! ([`from_bits`] under [`ModuleBuilder::set_widens_halves`]) where every use of its value,
//! followed through the moves that carry it (an element extract, insert or shuffle, either arm of
//! a select, a phi, a freeze), answers the same for every NaN:
//!
//! - a `half` `fadd`, `fsub` or `fmul`: its NaN result is `0x7E00` whatever the operands' NaNs (the
//!   same measurement, every pair of `half`s; a sum the module computes in `float` instead is the
//!   device's canonical `float` NaN, `DriverDeviation::MetalCanonicalNans`, which rounds to
//!   `0x7E00`);
//! - a compare, which reads no NaN's bits;
//! - an `fpext` to `float`, which Metal answers `0x7FC00000` for every NaN, as the widening does.
//!
//! No result of these carries the loaded NaN's payload (the `fpext`'s is Metal's own answer), so
//! the value is followed through its moves alone: a later store, bitcast or sign test of a result
//! reads what Metal's reads. Any other use of the value (a store, a bitcast, a call, a sign
//! operation, a division, a return, an aggregate) keeps the exact decode. The module then requires
//! `VkProperty::MetalHalfArithmetic` (`pipeline::compile`).
//!
//! [`from_bits`]: super::narrow::from_bits
//! [`ModuleBuilder::set_widens_halves`]: crate::spirv::builder::ModuleBuilder::set_widens_halves

use std::collections::BTreeSet;

use crate::air::ir::{BinOp as AirBin, CastOp, InstId, InstKind, Type};
use crate::budget::Meter;
use crate::error::{DriverDeviation, Error};
use crate::lower::select::SelectCx;
use crate::lower::strategy::Strategy;
use crate::lower::users::Users;

/// The loads of `half`s of `cx`'s function that take the device's widening (the module docs),
/// by their strategies `strategy` (one per value): none unless the profile takes
/// `DriverDeviation::MetalHalfArithmetic` and `DriverDeviation::MetalCanonicalNans` and has
/// `shaderFloat16`.
pub fn widened_loads(
    cx: &SelectCx<'_>,
    strategy: &[Option<Strategy>],
    m: &mut Meter,
) -> Result<BTreeSet<InstId>, Error> {
    let p = cx.profile;
    let applies = p.takes_driver_deviation(DriverDeviation::MetalHalfArithmetic)
        && p.takes_driver_deviation(DriverDeviation::MetalCanonicalNans)
        && p.fields().leaf.float16;
    let mut out = BTreeSet::new();
    if !applies {
        return Ok(out);
    }
    let types = cx.entry.module.types();
    let halves = |ty| match types.get(ty) {
        Some(Type::Half) => true,
        Some(Type::Vector { elem, .. }) => matches!(types.get(*elem), Some(Type::Half)),
        _ => false,
    };
    let loads: Vec<InstId> = cx
        .body
        .insts
        .iter()
        .filter(|(at, inst)| {
            matches!(inst.kind, InstKind::Load { atomic: None, .. })
                && halves(inst.ty)
                && cx
                    .values
                    .inst(*at)
                    .and_then(|v| strategy.get(v.index()))
                    .is_some_and(|s| matches!(s, Some(Strategy::Access(_))))
        })
        .map(|(at, _)| at)
        .collect();
    if loads.is_empty() {
        return Ok(out);
    }
    let users = Users::of(cx.body);
    for load in loads {
        if nan_bits_unread(cx, &users, load, m)? {
            out.insert(load);
        }
    }
    Ok(out)
}

/// Whether no use of `load`'s value reads a NaN's bits (the module docs), followed through the
/// moves that carry it.
fn nan_bits_unread(
    cx: &SelectCx<'_>,
    users: &Users,
    load: InstId,
    m: &mut Meter,
) -> Result<bool, Error> {
    let mut seen = BTreeSet::from([load]);
    let mut stack = vec![load];
    while let Some(at) = stack.pop() {
        for &u in users.get(at) {
            m.tick(1)?;
            let Some(inst) = cx.body.insts.get(u) else {
                return Ok(false);
            };
            let carries = match &inst.kind {
                InstKind::BinOp {
                    op: AirBin::FAdd | AirBin::FSub | AirBin::FMul,
                    ..
                }
                | InstKind::FCmp { .. } => false,
                InstKind::Cast {
                    op: CastOp::FPExt, ..
                } => false,
                // The moves: a value of `half`s is only ever their value operand (a select's
                // condition is a `bool`, an element index an integer).
                InstKind::ExtractElement { .. }
                | InstKind::InsertElement { .. }
                | InstKind::ShuffleVector { .. }
                | InstKind::Select { .. }
                | InstKind::Phi { .. }
                | InstKind::Freeze(_) => true,
                _ => return Ok(false),
            };
            if carries && seen.insert(u) {
                stack.push(u);
            }
        }
    }
    Ok(true)
}
