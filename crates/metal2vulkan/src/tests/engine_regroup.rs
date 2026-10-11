//! Metal's pairing of a fast-math multiply chain's lane-uniform factors (`lower::arith::regroup`),
//! on the owned fixture `engine/regroup_chain`.

use crate::air::ir::{BinOp as AirBin, InstId, InstKind, Value};
use crate::lower::arith::regroup::{paired, Ir};
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, meter, specialized_entry, AssembleError,
};
use crate::spirv::testing::committed_profiles;

fn fixture() -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join("regroup_chain.ll")) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected regroup_chain: {why}"),
    }
}

/// What the entry `entry` holds after the rewrite (on the first committed profile): the
/// multiplies whose lane-uniform factors are paired ([`paired`]), and those that read another
/// multiply through a lane selection or whole (a chain the rewrite left).
fn chains(bc: &[u8], entry: &str) -> (usize, usize) {
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .next()
        .expect("a profile");
    let e = specialized_entry(
        bc,
        entry,
        &p,
        &crate::library::FunctionConstantValues::defaults(),
        &[],
        &mut meter(),
    )
    .unwrap_or_else(|err| panic!("{entry}: {err}"));
    let body = e
        .module
        .func(e.func)
        .and_then(|f| f.body.as_ref())
        .expect("a body");
    let ir = Ir {
        module: &e.module,
        func: e.func,
        body,
    };
    let product = |v: Value| match v {
        Value::Inst(id) => match body.insts.get(id).map(|i| &i.kind) {
            Some(InstKind::BinOp {
                op: AirBin::FMul, ..
            }) => true,
            Some(InstKind::ShuffleVector {
                a: Value::Inst(src),
                ..
            }) => matches!(
                body.insts.get(*src).map(|i| &i.kind),
                Some(InstKind::BinOp {
                    op: AirBin::FMul,
                    ..
                })
            ),
            _ => false,
        },
        _ => false,
    };
    let muls: Vec<(InstId, Value, Value)> = body
        .insts
        .iter()
        .filter_map(|(at, inst)| match inst.kind {
            InstKind::BinOp {
                op: AirBin::FMul,
                a,
                b,
                ..
            } => Some((at, a, b)),
            _ => None,
        })
        .collect();
    let pairs = muls.iter().filter(|&&(at, _, _)| paired(ir, at)).count();
    let left = muls
        .iter()
        .filter(|&&(at, a, b)| !paired(ir, at) && (product(a) || product(b)))
        .count();
    (pairs, left)
}

/// Each entry's chain as the rewrite leaves it: the dither fragment's and the float kernel's become
/// `x * (u * w)`, one paired multiply each and no chain left; the chain whose inner lanes are
/// stored too, and the one whose third factor differs per lane, keep their grouping; the one
/// already paired is left as it is.
#[test]
fn chains_with_two_lane_uniform_factors_take_metal_s_pairing() {
    let Some(bc) = fixture() else {
        return;
    };
    assert_eq!(chains(&bc, "dither"), (1, 0), "dither");
    assert_eq!(chains(&bc, "pair_lanes"), (1, 0), "pair_lanes");
    assert_eq!(chains(&bc, "lanes_used_twice"), (0, 1), "lanes_used_twice");
    assert_eq!(chains(&bc, "per_lane_third"), (0, 1), "per_lane_third");
    assert_eq!(chains(&bc, "already_paired"), (1, 0), "already_paired");
}

/// The notes follow: a chain Metal pairs, rewritten or written so, notes no `FastMathRewrites`; a
/// chain that keeps its grouping notes it, through a lane selection too (`lanes_used_twice`, which
/// once noted nothing), on every committed profile. The dither fragment notes it for another
/// operation: its noise's subtraction under the scale's product.
#[test]
fn a_chain_metal_regroups_otherwise_notes_the_rewrites() {
    let Some(bc) = fixture() else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, noted) in [
            ("dither", true),
            ("pair_lanes", false),
            ("already_paired", false),
            ("lanes_used_twice", true),
            ("per_lane_third", true),
        ] {
            let a = compile(&bc, entry, &p).unwrap_or_else(|e| panic!("{stem} {entry}: {e}"));
            link_and_validate(&a, &p).unwrap_or_else(|e| panic!("{stem} {entry}: L3: {e}"));
            assert_eq!(
                a.precision.contains(&PrecisionNote::FastMathRewrites),
                noted,
                "{stem} {entry}: {:?}",
                a.precision
            );
        }
    }
}
