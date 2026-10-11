//! P1: make the working graph reducible by multi-entry dispatch.
//!
//! Each round builds the loop forest of the current graph. Every outermost irreducible loop
//! (a cycle with entries `e_0` (its DFS-first header) .. `e_{k-1}`) gets a selector `sel` and a
//! dispatch header `D: switch sel { i => e_i }`. Every edge entering the loop at `e_i` goes to a
//! fresh `enter_i: sel = i; br D`, and every retreating edge inside the loop to `e_i` goes to a
//! fresh `back_i: sel = i; br D`. `D` is then the loop's only entry, so the loop is natural and
//! headed by `D`; its body (without `D`) is resolved by the next round. The dispatch path runs no
//! original code and reads only the selector value stored on the same path, so every invocation
//! executes the same sequence of original blocks.
//!
//! A round adds at most `2k + 1` blocks per irreducible loop, so the output is O((n + e)·d) for
//! irreducible nesting depth `d`. Refusals, all before the round allocates anything:
//!
//! - a tangled original block inside an irreducible loop: `Unsupported{IrreducibleConvergent}`
//!   (`NotRepresentable`: dispatch changes which dynamic instances invocations share, and the
//!   source defines no reconvergence to preserve), located at that block;
//! - nesting deeper than `Limit::IrreducibleDepth` (the loop forest refuses it);
//! - more nodes than `Limit::OutputBlocks` allows for the pruned input (`4(n+e)+1024`).
//!
//! Node splitting is never used (worst case exponential, and it would duplicate tangled code).

use super::cfg::{invariant, Block, Content, Node, StructCfg, Term};
use super::CfgView;
use crate::budget::{Meter, IRREDUCIBLE_DEPTH};
use crate::error::{Error, Feature, Limit, Unsupported};
use crate::graph::{Cfg, LoopForest, LoopId};

/// P1 with the standard output cap for the graph as P0 left it. On error, the ledger holds
/// exactly what `sc` holds (`sc` may have grown before a later round refused).
#[cfg(test)]
pub fn resolve(sc: &mut StructCfg, view: &impl CfgView, meter: &mut Meter) -> Result<(), Error> {
    let cap = crate::budget::output_blocks_cap(sc.len() as u64, sc.edge_count() as u64);
    resolve_capped(sc, view, meter, cap)
}

/// P1 with an explicit output-node cap.
pub fn resolve_capped(
    sc: &mut StructCfg,
    view: &impl CfgView,
    meter: &mut Meter,
    cap: u64,
) -> Result<(), Error> {
    // Each round removes one level of irreducible nesting; the forest refuses deeper nests.
    for _ in 0..=IRREDUCIBLE_DEPTH {
        let cfg = Cfg::build(&*sc, meter)?;
        let step = match LoopForest::compute(&cfg, meter) {
            Ok(forest) => {
                let step = if forest.is_reducible_graph() {
                    Ok(true)
                } else {
                    round(sc, view, &cfg, &forest, meter, cap).map(|()| false)
                };
                meter.release(forest.ledger_bytes());
                step
            }
            Err(e) => Err(e.into()),
        };
        meter.release(cfg.ledger_bytes());
        if step? {
            return Ok(());
        }
    }
    Err(invariant(
        "irreducible resolution did not converge within the depth cap",
    ))
}

/// Resolve every outermost irreducible loop of `forest`.
fn round(
    sc: &mut StructCfg,
    view: &impl CfgView,
    cfg: &Cfg<Node>,
    forest: &LoopForest<Node>,
    meter: &mut Meter,
    cap: u64,
) -> Result<(), Error> {
    let mut resolved: Option<LoopId> = None;
    for l in forest.loops() {
        meter.tick(1)?;
        if forest.is_reducible(l) || resolved.is_some_and(|r| forest.loop_contains(r, l)) {
            continue;
        }
        resolved = Some(l);
        refuse_tangled(sc, view, forest, l, meter)?;
        let k = forest.entries(l).len();
        let added = (sc.len() as u64)
            .saturating_add(2u64.saturating_mul(k as u64))
            .saturating_add(1);
        meter.bound(Limit::OutputBlocks, added, cap)?;
        dispatch(sc, cfg, forest, l, meter)?;
    }
    Ok(())
}

fn refuse_tangled(
    sc: &StructCfg,
    view: &impl CfgView,
    forest: &LoopForest<Node>,
    l: LoopId,
    meter: &mut Meter,
) -> Result<(), Error> {
    for b in forest.blocks(l) {
        meter.tick(1)?;
        if let Content::Code(original) = sc.block(b)?.content {
            if view.has_tangled(original) {
                return Err(Error::Unsupported(
                    Unsupported::not_representable(Feature::IrreducibleConvergent)
                        .at(view.location(original)),
                ));
            }
        }
    }
    Ok(())
}

/// Route every entry of irreducible loop `l` through a fresh dispatch header.
fn dispatch(
    sc: &mut StructCfg,
    cfg: &Cfg<Node>,
    forest: &LoopForest<Node>,
    l: LoopId,
    meter: &mut Meter,
) -> Result<(), Error> {
    let sel = sc.new_selector()?;
    let targets: Box<[Node]> = forest.entries(l).collect();
    let d = sc.push(
        meter,
        Block {
            content: Content::Empty,
            term: Term::Dispatch {
                sel,
                targets: targets.clone(),
            },
        },
    )?;
    for (i, &e) in targets.iter().enumerate() {
        let value = u32::try_from(i).map_err(|_| invariant("entry count exceeds u32"))?;
        let set = Block {
            content: Content::SetSel { sel, value },
            term: Term::Br(d),
        };
        let (mut enter, mut back) = (None, None);
        let mut previous = None;
        for u in cfg.predecessors(e) {
            meter.tick(1)?;
            // Predecessors are listed by ascending source, once per parallel edge; a retarget
            // moves every edge `u -> e` at once.
            if previous == Some(u) || !cfg.is_reachable(u) {
                continue;
            }
            previous = Some(u);
            let slot = if !forest.contains(l, u) {
                &mut enter
            } else if cfg.is_retreating(u, e) {
                &mut back
            } else {
                continue;
            };
            let via = match *slot {
                Some(via) => via,
                None => {
                    let via = sc.push(meter, set.clone())?;
                    *slot = Some(via);
                    via
                }
            };
            sc.retarget(u, e, via)?;
        }
    }
    Ok(())
}
