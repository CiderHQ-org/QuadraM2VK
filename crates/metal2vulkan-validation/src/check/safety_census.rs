//! CPU-only execution-safety census of installed cases (typed): mechanical loop-freedom over
//! decoded AIR and the case-schema caps.
//!
//! Each installed case is classified from its own entry module plus its linked modules; pipeline
//! templates are installed cases of their own and appear as their own rows. The census reads each
//! private source shard at most once and never executes anything. A source shard it needs that is
//! missing or unreadable fails the whole census, naming the shard (the corpus root lacks the
//! private sources); only a case whose AIR its existing shard does not hold is an unclassified row.

use super::caps::cap_errors;
use super::loop_free::{
    classify_sources_with_fuel, Classification, CycleEvidence, FuelScope, LoopClass,
};
use crate::case::{AuthoredCase, ExecutionSafety};
use crate::source::{
    public_sources, read_source_shard, shard_index_for_hash, source_shard_path, SourceRow,
};
use crate::store::CorpusStore;
use std::collections::{BTreeMap, HashMap};
use std::path::Path;

/// One installed case, its authored declaration, the mechanical class of its own shader, and the
/// caps it exceeds.
pub struct SafetyCensusRow {
    pub case_id: String,
    pub name: String,
    pub declared: ExecutionSafety,
    /// `Err` when the source or a linked module cannot be loaded.
    pub class: Result<LoopClass, String>,
    /// Whether the Metal oracle's fuel transform bounds every cycle of the shader; `None` when
    /// unclassified.
    pub fuel: Option<FuelScope>,
    pub cap_errors: Vec<String>,
}

impl SafetyCensusRow {
    /// A `loop_free` declaration the classifier contradicts. The other direction (declared
    /// `authored_bounded`, mechanically loop-free) is conservative and is only counted.
    pub fn contradicts_declaration(&self) -> bool {
        self.declared == ExecutionSafety::LoopFree
            && matches!(&self.class, Ok(LoopClass::Cyclic(_)))
    }

    /// The cycle the Metal oracle's fuel transform cannot bound, when the row is cyclic and fuel
    /// does not cover it: such a case cannot run on the oracle.
    pub fn fuel_gap(&self) -> Option<&CycleEvidence> {
        match (&self.class, &self.fuel) {
            (Ok(LoopClass::Cyclic(_)), Some(fuel)) => fuel.evidence(),
            _ => None,
        }
    }
}

/// Aggregate counts over a census.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct SafetyCensusSummary {
    pub total: usize,
    pub loop_free: usize,
    pub cyclic: usize,
    pub unclassified: usize,
    /// Cyclic rows by evidence kind.
    pub cyclic_by_evidence: BTreeMap<&'static str, usize>,
    pub declared_loop_free_but_cyclic: usize,
    /// Cyclic rows the Metal oracle may run fuel-instrumented.
    pub cyclic_fuel_covered: usize,
    /// Cyclic rows fuel cannot bound, by the evidence of the cycle it cannot reach.
    pub fuel_uncovered_by_evidence: BTreeMap<&'static str, usize>,
    pub declared_bounded_but_loop_free: usize,
    pub over_caps: usize,
}

pub fn summarize(rows: &[SafetyCensusRow]) -> SafetyCensusSummary {
    let mut summary = SafetyCensusSummary {
        total: rows.len(),
        ..SafetyCensusSummary::default()
    };
    for row in rows {
        if !row.cap_errors.is_empty() {
            summary.over_caps += 1;
        }
        match &row.class {
            Ok(LoopClass::LoopFree) => {
                summary.loop_free += 1;
                if row.declared == ExecutionSafety::AuthoredBounded {
                    summary.declared_bounded_but_loop_free += 1;
                }
            }
            Ok(LoopClass::Cyclic(evidence)) => {
                summary.cyclic += 1;
                *summary
                    .cyclic_by_evidence
                    .entry(evidence.kind())
                    .or_default() += 1;
                if row.declared == ExecutionSafety::LoopFree {
                    summary.declared_loop_free_but_cyclic += 1;
                }
                if row.fuel == Some(FuelScope::Covered) {
                    summary.cyclic_fuel_covered += 1;
                }
                if let Some(evidence) = row.fuel_gap() {
                    *summary
                        .fuel_uncovered_by_evidence
                        .entry(evidence.kind())
                        .or_default() += 1;
                }
            }
            Err(_) => summary.unclassified += 1,
        }
    }
    summary
}

/// Classify every installed case and check it against the caps.
pub fn safety_census(root: &Path, store: &CorpusStore) -> Result<Vec<SafetyCensusRow>, String> {
    let mut cases = store.read_all_cases()?;
    cases.sort_by(|left, right| left.air_sha256.cmp(&right.air_sha256));
    // A public row by its module and entry, which a case names.
    let public = public_sources()?
        .into_iter()
        .map(|row| ((row.air_sha256.clone(), row.entry.clone()), row))
        .collect::<HashMap<_, _>>();
    let mut loaded_shard = None::<(usize, HashMap<String, SourceRow>)>;
    let mut rows = Vec::with_capacity(cases.len());
    for case in cases {
        let source = match public.get(&(case.air_sha256.clone(), case.entry.clone())) {
            Some(source) => Some(source),
            None => private_source(root, &case.air_sha256, &mut loaded_shard)?,
        };
        let classified = source
            .ok_or_else(|| {
                format!(
                    "AIR {} is not in its source shard or public fixtures",
                    case.air_sha256
                )
            })
            .and_then(|source| classify_case(root, &case, source));
        let (class, fuel) = match classified {
            Ok(Classification { class, fuel }) => (Ok(class), Some(fuel)),
            Err(error) => (Err(error), None),
        };
        let cap_errors = cap_errors(&case);
        rows.push(SafetyCensusRow {
            case_id: case.case_id,
            name: case.name,
            declared: case.execution_safety,
            class,
            fuel,
            cap_errors,
        });
    }
    Ok(rows)
}

/// The private source of `air_sha256` from its shard, loading the shard into `loaded` unless it is
/// the one already there. `Ok(None)` when the shard does not hold the hash; an error when the shard
/// is missing or unreadable.
fn private_source<'a>(
    root: &Path,
    air_sha256: &str,
    loaded: &'a mut Option<(usize, HashMap<String, SourceRow>)>,
) -> Result<Option<&'a SourceRow>, String> {
    let shard = shard_index_for_hash(air_sha256)?;
    if loaded.as_ref().is_none_or(|(index, _)| *index != shard) {
        let path = source_shard_path(root, shard);
        if !path.is_file() {
            return Err(format!(
                "source shard {} for AIR {air_sha256} does not exist",
                path.display()
            ));
        }
        *loaded = Some((
            shard,
            read_source_shard(&path)?
                .into_iter()
                .map(|row| (row.air_sha256.clone(), row))
                .collect(),
        ));
    }
    let (_, sources) = loaded.as_ref().expect("shard loaded above");
    Ok(sources.get(air_sha256))
}

fn classify_case(
    root: &Path,
    case: &AuthoredCase,
    source: &SourceRow,
) -> Result<Classification, String> {
    let linked = crate::library_module::resolve_linked_functions(root, case)
        .map_err(|errors| errors.join("; "))?;
    Ok(classify_sources_with_fuel(
        &case.entry,
        super::executed_modules(source, &linked),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::shard_index_for_hash;
    use crate::ScratchDir;

    #[test]
    fn fuel_gaps_name_the_cyclic_rows_fuel_cannot_bound() {
        let row = |name: &str, class: LoopClass, fuel: FuelScope| SafetyCensusRow {
            case_id: format!("id-{name}"),
            name: name.into(),
            declared: ExecutionSafety::AuthoredBounded,
            class: Ok(class),
            fuel: Some(fuel),
            cap_errors: Vec::new(),
        };
        let cycle = CycleEvidence::CfgCycle {
            function: "main".into(),
        };
        let unresolved = CycleEvidence::UnresolvedCall {
            function: "main".into(),
            callee: "helper".into(),
        };
        let rows = [
            row("loop-free", LoopClass::LoopFree, FuelScope::Covered),
            row(
                "covered",
                LoopClass::Cyclic(cycle.clone()),
                FuelScope::Covered,
            ),
            row(
                "uncovered",
                LoopClass::Cyclic(cycle),
                FuelScope::Uncovered(unresolved.clone()),
            ),
        ];
        let gaps = rows
            .iter()
            .filter_map(|row| row.fuel_gap().map(|gap| (row.name.as_str(), gap)))
            .collect::<Vec<_>>();
        assert_eq!(gaps, [("uncovered", &unresolved)]);
        let summary = summarize(&rows);
        assert_eq!((summary.cyclic, summary.cyclic_fuel_covered), (2, 1));
        assert_eq!(
            summary.fuel_uncovered_by_evidence,
            [("unresolved_call", 1)].into_iter().collect()
        );
    }

    #[test]
    fn a_missing_source_shard_fails_the_census_naming_it() {
        let scratch = ScratchDir::new("safety-census-missing-shard").unwrap();
        let store = CorpusStore::new(scratch.path());
        let hash = "22".repeat(32);
        let mut case = crate::case::attachmentless_fragment_test_case(hash.clone(), "main".into());
        case.case_id = case.computed_case_id().unwrap();
        store.put_case(case.clone()).unwrap();
        let path = source_shard_path(scratch.path(), shard_index_for_hash(&hash).unwrap());

        let error = safety_census(scratch.path(), &store).err().unwrap();
        assert!(
            error.contains(&path.display().to_string()) && error.contains(&hash),
            "{error}"
        );

        // With the shard present, a case whose AIR it does not hold is one unclassified row.
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, b"").unwrap();
        let rows = safety_census(scratch.path(), &store).unwrap();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].case_id, case.case_id);
        assert!(rows[0]
            .class
            .as_ref()
            .is_err_and(|error| error.contains("is not in its source shard")));
        assert_eq!(summarize(&rows).unclassified, 1);
    }
}
