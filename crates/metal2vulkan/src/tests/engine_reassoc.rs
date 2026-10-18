//! Metal's regrouping of a fast-math sum (`lower::arith::reassoc`), on the owned fixtures
//! `engine/sum_regroup` and `engine/sum_regroup_precise`.

use crate::air::intrinsics::AirIntrinsic;
use crate::air::ir::{BinOp as AirBin, Body, Callee, FastMath, InstId, InstKind, Type, Value};
use crate::air::Entry;
use crate::pipeline::artifact::PrecisionNote;
use crate::pipeline::testing::{
    compile, engine_dir, link_and_validate, llvm_assemble, meter, specialized_entry, AssembleError,
};
use crate::spirv::testing::committed_profiles;
use std::collections::BTreeSet;

fn fixture(name: &str) -> Option<Vec<u8>> {
    match llvm_assemble(&engine_dir().join(format!("{name}.ll"))) {
        Ok(b) => Some(b),
        Err(AssembleError::Missing(why)) => {
            eprintln!("SKIPPED: {why}");
            None
        }
        Err(AssembleError::Rejected(why)) => panic!("llvm-as rejected {name}: {why}"),
    }
}

/// Signed leaves: each leaf's name and whether the sum subtracts it, sorted.
type Signed = Vec<(String, bool)>;

/// A sum's grouping: the signed leaves of its value, and those of each addition it makes, each
/// up to its sign (rounding to nearest is symmetric, so a sum and its negation round alike). Two
/// sums of one grouping round alike on every input.
type Grouping = (Signed, BTreeSet<Signed>);

fn negated(s: Signed) -> Signed {
    let mut s: Signed = s.into_iter().map(|(n, neg)| (n, !neg)).collect();
    s.sort();
    s
}

fn joined(a: Signed, b: Signed) -> Signed {
    let mut s: Signed = a.into_iter().chain(b).collect();
    s.sort();
    s
}

/// `s` up to its sign: the first leaf added.
fn unsigned(s: &Signed) -> Signed {
    if s.first().is_some_and(|(_, neg)| *neg) {
        negated(s.clone())
    } else {
        s.clone()
    }
}

/// The grouping of an expression over named leaves: `+`, binary and unary `-`, parentheses;
/// binary operators group from the left.
fn parse(text: &str) -> Grouping {
    fn unary(t: &[String], i: &mut usize, sums: &mut BTreeSet<Signed>) -> Signed {
        let tok = t.get(*i).cloned().expect("an operand");
        *i += 1;
        match tok.as_str() {
            "-" => negated(unary(t, i, sums)),
            "(" => {
                let s = expr(t, i, sums);
                assert_eq!(t.get(*i).map(String::as_str), Some(")"), "{t:?}");
                *i += 1;
                s
            }
            _ => vec![(tok, false)],
        }
    }
    fn expr(t: &[String], i: &mut usize, sums: &mut BTreeSet<Signed>) -> Signed {
        let mut s = unary(t, i, sums);
        while let Some(op) = t.get(*i).filter(|o| *o == "+" || *o == "-").cloned() {
            *i += 1;
            let r = unary(t, i, sums);
            s = joined(s, if op == "-" { negated(r) } else { r });
            sums.insert(unsigned(&s));
        }
        s
    }
    let spaced = text.replace('(', " ( ").replace(')', " ) ");
    let tokens: Vec<String> = spaced
        .split_whitespace()
        .flat_map(|w| match w.strip_prefix('-').filter(|r| !r.is_empty()) {
            Some(rest) => vec!["-".to_string(), rest.to_string()],
            None => vec![w.to_string()],
        })
        .collect();
    let mut sums = BTreeSet::new();
    let mut i = 0;
    let s = expr(&tokens, &mut i, &mut sums);
    assert_eq!(i, tokens.len(), "{text}");
    (s, sums)
}

/// The `half` sums of `entry` (after the rewrite, on the first committed profile) as groupings
/// over its leaves, each leaf named by the latest texture call it reads, in program order
/// (`names`), and whether every addition and subtraction of them still carries `reassoc`.
fn sums(bc: &[u8], entry: &str, names: &[&str]) -> (BTreeSet<Grouping>, Vec<bool>) {
    let (_, p) = committed_profiles()
        .expect("profiles")
        .into_iter()
        .next()
        .expect("a profile");
    let e: Entry = specialized_entry(
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
    let half = |id: InstId| {
        body.insts
            .get(id)
            .and_then(|i| e.module.types().get(i.ty))
            .is_some_and(|t| matches!(t, Type::Half))
    };
    let sum = |id: InstId| {
        half(id)
            && matches!(
                body.insts.get(id).map(|i| &i.kind),
                Some(
                    InstKind::BinOp {
                        op: AirBin::FAdd | AirBin::FSub,
                        ..
                    } | InstKind::FNeg { .. }
                )
            )
    };
    let calls: Vec<InstId> = body
        .insts
        .iter()
        .filter(|(_, i)| {
            matches!(
                &i.kind,
                InstKind::Call(c) if matches!(c.callee, Callee::Air { op: AirIntrinsic::Texture(_), .. })
            )
        })
        .map(|(id, _)| id)
        .collect();
    assert_eq!(calls.len(), names.len(), "{entry}: texture calls");
    let mut read = BTreeSet::new();
    for (id, i) in body.insts.iter() {
        if sum(id) {
            i.kind.for_each_value(|v| {
                if let Value::Inst(o) = v {
                    read.insert(o);
                }
            });
        }
    }
    let mut out = BTreeSet::new();
    let mut flags = Vec::new();
    for (id, i) in body.insts.iter() {
        if let InstKind::BinOp {
            op: AirBin::FAdd | AirBin::FSub,
            fmf,
            ..
        } = &i.kind
        {
            if half(id) {
                flags.push(fmf.contains(FastMath::REASSOC));
            }
        }
        if sum(id) && !read.contains(&id) {
            let mut adds = BTreeSet::new();
            let s = signed(body, Value::Inst(id), &sum, &calls, names, &mut adds);
            out.insert((s, adds));
        }
    }
    (out, flags)
}

/// The signed leaves of `v`, a sum's value, collecting each addition's in `adds`.
fn signed(
    body: &Body,
    v: Value,
    sum: &dyn Fn(InstId) -> bool,
    calls: &[InstId],
    names: &[&str],
    adds: &mut BTreeSet<Signed>,
) -> Signed {
    let id = match v {
        Value::Inst(id) if sum(id) => id,
        _ => return vec![(leaf(body, v, calls, names), false)],
    };
    match body.insts.get(id).map(|i| &i.kind) {
        Some(InstKind::FNeg { v, .. }) => negated(signed(body, *v, sum, calls, names, adds)),
        Some(InstKind::BinOp { op, a, b, .. }) => {
            let x = signed(body, *a, sum, calls, names, adds);
            let y = signed(body, *b, sum, calls, names, adds);
            let s = joined(x, if *op == AirBin::FSub { negated(y) } else { y });
            adds.insert(unsigned(&s));
            s
        }
        _ => unreachable!("a sum"),
    }
}

/// The name of the latest texture call `v` reads.
fn leaf(body: &Body, v: Value, calls: &[InstId], names: &[&str]) -> String {
    let mut seen = BTreeSet::new();
    let mut work = vec![v];
    let mut latest = None;
    while let Some(v) = work.pop() {
        let Value::Inst(id) = v else {
            continue;
        };
        if !seen.insert(id) {
            continue;
        }
        if let Some(k) = calls.iter().position(|c| *c == id) {
            latest = latest.max(Some(k));
            continue;
        }
        if let Some(i) = body.insts.get(id) {
            i.kind.for_each_value(|o| work.push(o));
        }
    }
    let k = latest.unwrap_or_else(|| panic!("{v:?} reads no texture"));
    names.get(k).expect("a name").to_string()
}

const IN_ORDER: [&str; 9] = ["C", "wL", "wR", "wU", "wD", "TL", "TR", "BL", "BR"];
const TIED: [&str; 9] = ["TL", "wU", "BL", "C", "wL", "BR", "wR", "TR", "wD"];

fn expect(sums: &[&str]) -> BTreeSet<Grouping> {
    sums.iter().map(|s| parse(s)).collect()
}

/// The pinned sums take Metal's grouping, as measured on the fixture's own AIR:
/// `regroup` and `tied`, whose leaves tie in rank as their reads are ordered. Every addition and
/// subtraction rewritten carries no `reassoc`. The sum the rank does not pin (`unpinned`) keeps the
/// AIR's grouping and its flags.
#[test]
fn pinned_sums_take_metal_s_grouping() {
    let Some(bc) = fixture("sum_regroup") else {
        return;
    };
    let (got, flags) = sums(&bc, "regroup", &IN_ORDER);
    assert_eq!(
        got,
        expect(&[
            "(((wR - (wL + TL)) + TR) - BL) + BR",
            "((wD - ((wU + TL) + TR)) + BL) + BR",
        ]),
        "regroup"
    );
    assert!(flags.iter().all(|r| !r), "regroup: {flags:?}");
    let (got, flags) = sums(&bc, "tied", &TIED);
    assert_eq!(
        got,
        expect(&[
            "(((-TL - (BL + wL)) + BR) + wR) + TR",
            "(((BL - (wU + TL)) + BR) - TR) + wD",
        ]),
        "tied"
    );
    assert!(flags.iter().all(|r| !r), "tied: {flags:?}");
    // The AIR's grouping (the frontend's); the top-left leaf is named by its latest read, the
    // top-left sample.
    let (got, flags) = sums(&bc, "unpinned", &IN_ORDER);
    assert_eq!(
        got,
        expect(&[
            "((wR + TR) - ((wL + TL) + BL)) + BR",
            "((wD + BL) - ((wU + TR) + TL)) + BR",
        ]),
        "unpinned"
    );
    assert!(flags.iter().all(|r| *r), "unpinned: {flags:?}");
}

/// Strict-math sums stay as written: without fast math no operation carries `reassoc`, and each sum
/// keeps the source's grouping.
#[test]
fn strict_math_sums_stay_as_written() {
    let Some(bc) = fixture("sum_regroup_precise") else {
        return;
    };
    let (got, flags) = sums(&bc, "strict", &IN_ORDER);
    assert_eq!(
        got,
        expect(&[
            "((wR + TR) + BR) - (BL + (wL + TL))",
            "((wD + BL) + BR) - (TR + (wU + TL))",
        ])
    );
    assert!(flags.iter().all(|r| !r), "strict: {flags:?}");
}

/// The notes follow: a pinned sum is Metal's grouping and notes no `FastMathRewrites`; one the rank
/// does not pin notes it; on every committed profile, and each links and validates.
#[test]
fn a_sum_left_as_written_notes_the_rewrites() {
    let Some(bc) = fixture("sum_regroup") else {
        return;
    };
    for (stem, p) in committed_profiles().expect("profiles") {
        for (entry, noted) in [("regroup", false), ("tied", false), ("unpinned", true)] {
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
