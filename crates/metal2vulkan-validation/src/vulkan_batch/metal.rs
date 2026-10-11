//! The Metal oracle's qualifications on the shared batch driver (`corpus-metal --batch`): what
//! differs from a candidate batch. A Metal run is one [`Pass::Metal`] per case, through
//! `corpus-metal` on the GPU, recording the case's Metal row in the batch's Metal environment; the
//! driver ([`run_batch`](super::run_batch)) runs, records, stops, and resumes it as it does a
//! candidate run. Here are the selection, the install, and the report.

use super::{
    family, read_done_list, read_records, scope, BatchBackend, BatchBuild, Class, DoneRun,
    Installed, Outcome, Pass, DONE_LIST, ROWS,
};
use crate::case::AuthoredCase;
use crate::observation::MetalObservation;
use crate::store::CorpusStore;
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

/// The outcomes a Metal run comes to, in report order.
const METAL_OUTCOMES: [Outcome; 7] = [
    Outcome::Qualified,
    Outcome::ReferenceGap,
    Outcome::Refused,
    Outcome::DeviceCrash,
    Outcome::WaitTimeout,
    Outcome::Timeout,
    Outcome::Failed,
];

/// The cases a Metal batch qualifies: every case of `cases`, or each of `case_ids` (every one a
/// case of `cases`), in case-ID order.
pub fn selected_cases(
    mut cases: Vec<AuthoredCase>,
    case_ids: &[String],
) -> Result<Vec<AuthoredCase>, String> {
    if !case_ids.is_empty() {
        let requested = case_ids.iter().collect::<BTreeSet<_>>();
        cases.retain(|case| requested.contains(&case.case_id));
        if cases.len() != requested.len() {
            let found = cases
                .iter()
                .map(|case| &case.case_id)
                .collect::<BTreeSet<_>>();
            let missing = requested
                .difference(&found)
                .map(|id| id.as_str())
                .collect::<Vec<_>>();
            return Err(format!("no case for: {}", missing.join(", ")));
        }
    }
    cases.sort_by(|left, right| left.case_id.cmp(&right.case_id));
    Ok(cases)
}

/// The Metal rows a Metal batch's rows file holds, in file order (a missing file holds none; a
/// last line without its newline is a row whose append was cut short, and is dropped).
pub fn read_rows(path: &Path) -> Result<Vec<MetalObservation>, String> {
    read_records(path, "Metal row")
}

/// Write into `store` the Metal row of each case's last run in the out directory `out` that
/// recorded one, through the store's own validated write ([`CorpusStore::upsert_metal`]), in
/// case order, as the single-case qualification writes it into its store. Only a qualification
/// and a reference gap record a row; a row no finished run lists (an append cut before its
/// done-list line) is not installed. The out directory must be a Metal batch's, and each row of its
/// environment. The record is returned, as a candidate install returns it.
pub fn install(out: &Path, store: &CorpusStore) -> Result<Installed, String> {
    let build = BatchBuild::read(out)?;
    let environment_id = match &build {
        Some(build) if build.backend == BatchBackend::Metal => build.environment_id.clone(),
        recorded => {
            return Err(format!(
                "{} holds runs of the {} backend, not metal",
                out.display(),
                recorded
                    .as_ref()
                    .map_or(BatchBackend::Vulkan, |build| build.backend)
                    .directory()
            ))
        }
    };
    let recorded = read_done_list(&out.join(DONE_LIST))?
        .into_iter()
        .filter(|run| matches!(run.outcome, Outcome::Qualified | Outcome::ReferenceGap))
        .map(|run| run.case_id)
        .collect::<BTreeSet<_>>();
    let mut last = BTreeMap::new();
    for row in read_rows(&out.join(ROWS))? {
        if row.environment_id != environment_id {
            return Err(format!(
                "Metal row {} is of {}, not the batch's {environment_id}",
                row.case_id, row.environment_id
            ));
        }
        last.insert(row.case_id.clone(), row);
    }
    let mut installed = Installed {
        build,
        ..Installed::default()
    };
    for (case_id, row) in last {
        if recorded.contains(&case_id) {
            store.upsert_metal(row)?;
            installed.rows += 1;
        }
    }
    Ok(installed)
}

/// The class's name in a Metal report: a match is a qualification.
fn class_label(class: Class) -> &'static str {
    match class {
        Class::Match => "qualified",
        class => class.label(),
    }
}

/// The Metal batch's report over the out directory's `done` (the last run of each case wins),
/// against `cases`, the cases the batch selects ([`selected_cases`]). Lines:
///
/// - `batch-report pass=metal selected=N ran=N pending=N qualified=N refusal=N failure=N`, then
///   the count of each refusal and failure [`Outcome`] (`reference_gap=N refused=N
///   device_crash=N wait_timeout=N timeout=N failed=N`);
/// - `batch-family pass=metal family=F cases=N qualified=N refusal=N failure=N` per [`family`],
///   and `batch-scope pass=metal scope=S ...` the same per [`scope`];
/// - `batch-run case=ID pass=metal scope=S family=F outcome=O seconds=S name=NAME detail=D` for
///   every run that did not qualify;
/// - `batch-slowest case=ID pass=metal seconds=S outcome=O` for the slowest run.
pub fn report(cases: &[AuthoredCase], done: &[DoneRun]) -> Vec<String> {
    let by_id = cases
        .iter()
        .map(|case| (case.case_id.as_str(), case))
        .collect::<HashMap<_, _>>();
    let mut latest = BTreeMap::<&str, &DoneRun>::new();
    for run in done
        .iter()
        .filter(|run| run.pass == Pass::Metal && by_id.contains_key(run.case_id.as_str()))
    {
        latest.insert(run.case_id.as_str(), run);
    }
    let classes = [Class::Match, Class::Refusal, Class::Failure];
    let mut outcomes = BTreeMap::<Outcome, usize>::new();
    let mut groups = BTreeMap::<(&str, String), BTreeMap<Class, usize>>::new();
    for (case_id, run) in &latest {
        *outcomes.entry(run.outcome).or_default() += 1;
        let case = by_id[case_id];
        for group in [("scope", scope(case).to_string()), ("family", family(case))] {
            *groups
                .entry(group)
                .or_default()
                .entry(run.outcome.class())
                .or_default() += 1;
        }
    }
    let class_count = |class: Class| {
        outcomes
            .iter()
            .filter(|(outcome, _)| outcome.class() == class)
            .map(|(_, count)| count)
            .sum::<usize>()
    };
    let mut line = format!(
        "batch-report pass=metal selected={} ran={} pending={}",
        cases.len(),
        latest.len(),
        cases.len() - latest.len()
    );
    for class in classes {
        line.push_str(&format!(" {}={}", class_label(class), class_count(class)));
    }
    for outcome in METAL_OUTCOMES
        .into_iter()
        .filter(|outcome| *outcome != Outcome::Qualified)
    {
        line.push_str(&format!(
            " {}={}",
            outcome.label(),
            outcomes.get(&outcome).copied().unwrap_or(0)
        ));
    }
    let mut lines = vec![line];
    for ((kind, name), counts) in &groups {
        let mut line = format!(
            "batch-{kind} pass=metal {kind}={name} cases={}",
            counts.values().sum::<usize>()
        );
        for class in classes {
            line.push_str(&format!(
                " {}={}",
                class_label(class),
                counts.get(&class).copied().unwrap_or(0)
            ));
        }
        lines.push(line);
    }
    for (case_id, run) in latest
        .iter()
        .filter(|(_, run)| run.outcome != Outcome::Qualified)
    {
        let case = by_id[case_id];
        lines.push(format!(
            "batch-run case={case_id} pass=metal scope={} family={} outcome={} seconds={:.3} \
             name={} detail={}",
            scope(case),
            family(case),
            run.outcome.label(),
            run.seconds,
            case.name,
            run.detail
        ));
    }
    if let Some(slowest) = latest
        .values()
        .max_by(|left, right| left.seconds.total_cmp(&right.seconds))
    {
        lines.push(format!(
            "batch-slowest case={} pass=metal seconds={:.3} outcome={}",
            slowest.case_id,
            slowest.seconds,
            slowest.outcome.label()
        ));
    }
    lines
}
