//! The harness's CPU tests: the committed records are current, and the records file and the
//! comparison behave as `record` says, on records these tests make.

use super::record::{compare, read_lines, stale, write_line, Fingerprint, Line, Refusal, Value};
use super::{fingerprint, records_path, HARNESS, SCENARIOS};
use crate::runtime_lane::METAL_ENVIRONMENT;

/// Every scenario has a line of [`METAL_ENVIRONMENT`], the lane's reference; every committed
/// line is current (its fingerprint the scenario's now, so no source, version or fixture
/// changed since Metal answered), and its records are of the scenario's cases, every case
/// recorded, each record named once.
#[test]
fn committed_records_are_current() {
    let mut problems = Vec::new();
    for scenario in &SCENARIOS {
        let lines = read_lines(&records_path(scenario.name)).unwrap();
        // What a line depends on beside its environment, which `stale` does not compare.
        let current = fingerprint(scenario, METAL_ENVIRONMENT).unwrap();
        if !lines
            .iter()
            .any(|line| line.fingerprint.metal_environment == METAL_ENVIRONMENT)
        {
            problems.push(format!("{}: no line of {METAL_ENVIRONMENT}", scenario.name));
        }
        for line in &lines {
            let environment = &line.fingerprint.metal_environment;
            let stale = stale(&line.fingerprint, &current);
            if !stale.is_empty() {
                problems.push(format!(
                    "{} on {environment}: stale ({}): record it again",
                    scenario.name,
                    stale.join(", ")
                ));
            }
            for case in scenario.cases {
                let prefix = format!("{case}/");
                if !line
                    .records
                    .iter()
                    .any(|(name, _)| name.starts_with(&prefix))
                {
                    problems.push(format!(
                        "{} on {environment}: no record of {case}",
                        scenario.name
                    ));
                }
            }
            let mut names = std::collections::BTreeSet::new();
            for (name, _) in &line.records {
                if !names.insert(name) {
                    problems.push(format!(
                        "{} on {environment}: {name} is recorded twice",
                        scenario.name
                    ));
                }
                let known = scenario
                    .cases
                    .iter()
                    .any(|case| name.starts_with(&format!("{case}/")));
                if !known {
                    problems.push(format!(
                        "{} on {environment}: {name} is of no case",
                        scenario.name
                    ));
                }
            }
        }
    }
    assert!(problems.is_empty(), "{}", problems.join("\n"));
}

/// A record depends on its own scenario's source and the shared harness, never on another
/// scenario: no harness source names a scenario, or holds a scenario's source, so a scenario
/// added or changed stales its own records alone.
#[test]
fn no_harness_source_names_a_scenario() {
    for scenario in &SCENARIOS {
        for source in HARNESS {
            assert!(
                !source.contains(scenario.name) && source != scenario.source,
                "a harness source names scenario {}",
                scenario.name
            );
        }
    }
}

fn print(environment: &str) -> Fingerprint {
    Fingerprint {
        metal_environment: environment.to_owned(),
        scenario: "s".to_owned(),
        scenario_version: 1,
        sources_sha256: "aa".to_owned(),
        fixtures: Vec::new(),
    }
}

fn done(name: &str) -> (String, Value) {
    (name.to_owned(), Value::Done)
}

fn refused(name: &str, refusal: Refusal, detail: &str) -> (String, Value) {
    (
        name.to_owned(),
        Value::Refused {
            refusal,
            detail: detail.to_owned(),
        },
    )
}

#[test]
fn a_line_replaces_its_environments_and_lines_sort_by_environment() {
    let dir = crate::ScratchDir::new("parity-records").unwrap();
    let path = dir.path().join("s.jsonl");
    assert!(read_lines(&path).unwrap().is_empty());
    let line = |environment: &str, records| Line {
        fingerprint: print(environment),
        records,
    };
    write_line(&path, line("m4", vec![done("c/a")])).unwrap();
    write_line(&path, line("m3", vec![done("c/a")])).unwrap();
    write_line(
        &path,
        line(
            "m4",
            vec![done("c/a"), ("c/b".to_owned(), Value::bytes(&[1, 2]))],
        ),
    )
    .unwrap();
    let lines = read_lines(&path).unwrap();
    let environments: Vec<&str> = lines
        .iter()
        .map(|line| line.fingerprint.metal_environment.as_str())
        .collect();
    assert_eq!(environments, ["m3", "m4"]);
    assert_eq!(lines[1].records.len(), 2);
    assert!(compare(&lines, &print("m4"), &lines[1].records).is_ok());
}

#[test]
fn a_stale_line_fails_as_stale_naming_what_changed() {
    let lines = [Line {
        fingerprint: print("m3"),
        records: vec![done("c/a")],
    }];
    let records = [done("c/a")];
    type Change = (&'static str, fn(&mut Fingerprint));
    let changes: [Change; 4] = [
        ("scenario_version", |f| f.scenario_version = 2),
        ("sources", |f| f.sources_sha256 = "bb".to_owned()),
        ("fixtures", |f| {
            f.fixtures.push(super::record::FixtureDigest {
                case: "c".to_owned(),
                air_sha256: "cc".to_owned(),
                metal_input_sha256: "dd".to_owned(),
            });
        }),
        ("scenario", |f| f.scenario = "t".to_owned()),
    ];
    for (what, change) in changes {
        let mut current = print("m3");
        change(&mut current);
        let error = compare(&lines, &current, &records).unwrap_err();
        assert!(error.contains("stale") && error.contains(what), "{error}");
    }
    let error = compare(&lines, &print("m4"), &records).unwrap_err();
    assert!(error.contains("no record of m4"), "{error}");
}

#[test]
fn the_compare_is_exact_record_by_record() {
    let metal = vec![
        done("c/a"),
        refused("c/b", Refusal::Stopped, "Metal's words"),
        ("c/c".to_owned(), Value::bytes(&[1, 2, 3])),
        (
            "c/d".to_owned(),
            Value::Completed {
                status: "completed".to_owned(),
            },
        ),
    ];
    let lines = [Line {
        fingerprint: print("m3"),
        records: metal.clone(),
    }];
    let current = print("m3");
    // The refusal's words are not compared.
    let mut same = metal.clone();
    same[1] = refused("c/b", Refusal::Stopped, "the runtime's words");
    assert!(compare(&lines, &current, &same).is_ok());
    // Each kind of difference is named.
    let cases: [(Vec<(String, Value)>, &str); 6] = [
        (
            vec![
                metal[0].clone(),
                done("c/b"),
                metal[2].clone(),
                metal[3].clone(),
            ],
            "c/b: Metal",
        ),
        (
            vec![
                metal[0].clone(),
                metal[1].clone(),
                ("c/c".to_owned(), Value::bytes(&[1, 2, 4])),
                metal[3].clone(),
            ],
            "c/c: Metal",
        ),
        (
            vec![
                metal[0].clone(),
                metal[1].clone(),
                metal[2].clone(),
                (
                    "c/d".to_owned(),
                    Value::Completed {
                        status: "error".to_owned(),
                    },
                ),
            ],
            "c/d: Metal",
        ),
        (metal[..3].to_vec(), "c/d: Metal Completed"),
        (
            [metal.clone(), vec![done("c/e")]].concat(),
            "c/e: no Metal record",
        ),
        (
            vec![
                done("c/x"),
                metal[1].clone(),
                metal[2].clone(),
                metal[3].clone(),
            ],
            "record 0: Metal c/a",
        ),
    ];
    for (records, named) in cases {
        let error = compare(&lines, &current, &records).unwrap_err();
        assert!(error.contains(named), "{named}: {error}");
    }
}

#[test]
fn a_gap_matches_nothing_not_even_a_gap() {
    let gap = refused("c/a", Refusal::Gap, "a gap");
    let lines = [Line {
        fingerprint: print("m3"),
        records: vec![gap.clone()],
    }];
    assert!(compare(&lines, &print("m3"), &[gap]).is_err());
}
