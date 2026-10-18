//! The harness's device tests. On the device the cfg selects, each scenario runs on the runtime
//! and is compared with Metal's records (`<scenario>_matches_metal`). On a Mac with
//! `--cfg m2v_gpu`, [`record_metal`] records Metal's answers into the committed records files,
//! each case in a child process ([`metal_child`]) under API validation (`MTL_DEBUG_LAYER=1`), for
//! the scenarios whose line is missing or stale.

use super::api::Observation;
use super::record::{compare, read_lines, Value};
use super::runtime::RuntimeApi;
use super::{fingerprint, records_path, scenarios, Scenario, SCENARIOS};
use crate::runtime_lane::METAL_ENVIRONMENT;

/// Run every case of `scenario` on the runtime, each on a device of its own, and compare the
/// records with Metal's of [`METAL_ENVIRONMENT`]. Fails on a difference, a stale record, or
/// any message of the validation layer.
fn matches_metal(name: &str) {
    let scenario = SCENARIOS
        .iter()
        .find(|scenario| scenario.name == name)
        .expect("a scenario");
    let current = fingerprint(scenario, METAL_ENVIRONMENT).unwrap();
    let lines = read_lines(&records_path(scenario.name)).unwrap();
    let mut records = Vec::new();
    for &case in scenario.cases {
        records.extend(run_on_runtime(scenario, case).unwrap());
    }
    if let Err(differences) = compare(&lines, &current, &records) {
        panic!("{differences}");
    }
    eprintln!(
        "parity: {} matches {METAL_ENVIRONMENT}, {} records",
        scenario.name,
        records.len()
    );
}

/// One case's records on a runtime device, which reported no validation message.
fn run_on_runtime(scenario: &Scenario, case: &'static str) -> Result<Vec<(String, Value)>, String> {
    let api = RuntimeApi::open()?;
    let mut observation = Observation::new(case, false);
    scenarios::run(&api, scenario.name, case, &mut observation)?;
    let records = observation.finish(scenario.name);
    let messages = api.messages.clone();
    drop(api);
    let messages = messages.take();
    if messages.is_empty() {
        Ok(records)
    } else {
        Err(format!(
            "{}/{case}: the validation layer reported: {messages:#?}",
            scenario.name
        ))
    }
}

#[test]
fn buffer_copies_match_metal() {
    matches_metal("buffer_copies");
}

#[test]
fn buffer_texture_copies_match_metal() {
    matches_metal("buffer_texture_copies");
}

#[test]
fn hazards_match_metal() {
    matches_metal("hazards");
}

#[test]
fn command_buffer_times_match_metal() {
    matches_metal("command_buffer_times");
}

/// Metal's side: the recorder and its child.
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal_recording {
    use super::super::api::{Observation, AT, RECORD};
    use super::super::metal::MetalApi;
    use super::super::record::{read_lines, stale, write_line, Line, Refusal, Value};
    use super::super::{fingerprint, records_path, scenarios, SCENARIOS};
    use std::os::unix::process::ExitStatusExt as _;
    use std::process::Command;

    /// The variable that names the child's scenario and case, `scenario/case`.
    const CASE_VAR: &str = "M2V_PARITY_CASE";

    /// SIGABRT: how Metal's API validation ends a process it stops.
    const ABORT: i32 = 6;

    /// Record Metal's answers to every scenario whose line for this Mac's environment is missing
    /// or stale into its committed records file, as that environment's line. A current line is
    /// kept, and its scenario does not run: only what changed runs on the Mac.
    #[test]
    #[ignore = "writes the committed parity records; a recording run on a Mac takes it"]
    fn record_metal() {
        let environment = crate::metal::identity().unwrap().environment_id();
        for scenario in &SCENARIOS {
            let fingerprint = fingerprint(scenario, &environment).unwrap();
            let lines = read_lines(&records_path(scenario.name)).unwrap();
            let current = lines.iter().any(|line| {
                line.fingerprint.metal_environment == environment
                    && stale(&line.fingerprint, &fingerprint).is_empty()
            });
            if current {
                eprintln!("parity: {} on {environment} is current", scenario.name);
                continue;
            }
            let mut records = Vec::new();
            for &case in scenario.cases {
                records.extend(child(scenario.name, case).unwrap());
            }
            eprintln!(
                "parity: recorded {} on {environment}, {} records",
                scenario.name,
                records.len()
            );
            let line = Line {
                fingerprint,
                records,
            };
            write_line(&records_path(scenario.name), line).unwrap();
        }
    }

    /// Run one case in a child process under API validation: the records it printed, and, when
    /// API validation stopped it, the call it stopped, refused.
    fn child(scenario: &str, case: &str) -> Result<Vec<(String, Value)>, String> {
        let output = Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args([
                "--exact",
                "parity::lane::metal_recording::metal_child",
                "--ignored",
                "--nocapture",
                "--test-threads=1",
            ])
            .env(CASE_VAR, format!("{scenario}/{case}"))
            .env("MTL_DEBUG_LAYER", "1")
            .output()
            .map_err(|error| format!("{scenario}/{case}: spawn: {error}"))?;
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        let mut records: Vec<(String, Value)> = Vec::new();
        let mut at = None;
        for line in stdout.lines() {
            if let Some(name) = line.strip_prefix(AT) {
                at = Some(name.to_owned());
            } else if let Some(json) = line.strip_prefix(RECORD) {
                let record: (String, Value) = serde_json::from_str(json)
                    .map_err(|error| format!("{scenario}/{case}: {error}: {json}"))?;
                records.push(record);
            }
        }
        for line in stderr.lines().filter(|l| l.starts_with("parity time:")) {
            eprintln!("{line}");
        }
        let passed = output.status.success() && stdout.contains("test result: ok. 1 passed");
        if passed {
            return Ok(records);
        }
        let stopped = if output.status.signal() == Some(ABORT) {
            stderr
                .find("failed assertion")
                .map(|start| assertion(&stderr[start..]))
        } else {
            None
        };
        let pending = at.filter(|name| records.last().is_none_or(|(last, _)| last != name));
        match (stopped, pending) {
            (Some(detail), Some(name)) => {
                records.push((
                    name,
                    Value::Refused {
                        refusal: Refusal::Stopped,
                        detail,
                    },
                ));
                Ok(records)
            }
            _ => Err(format!(
                "{scenario}/{case}: the child failed ({}):\n{stdout}\n{stderr}",
                output.status
            )),
        }
    }

    /// The text of a failed assertion of Metal's API validation, between its quotes.
    fn assertion(text: &str) -> String {
        let body = text.split_once('`').map_or(text, |(_, rest)| rest);
        body.split_once('\'')
            .map_or(body, |(quoted, _)| quoted)
            .trim()
            .to_owned()
    }

    /// One case on Metal, its calls and records printed as they are made.
    #[test]
    #[ignore = "a child process of record_metal"]
    fn metal_child() {
        let named = std::env::var(CASE_VAR).expect("the case to run");
        let (scenario, case) = named.split_once('/').expect("scenario/case");
        let scenario = SCENARIOS
            .iter()
            .find(|s| s.name == scenario)
            .expect("a scenario");
        let case = *scenario.cases.iter().find(|c| **c == case).expect("a case");
        let api = MetalApi::open().unwrap();
        let mut observation = Observation::new(case, true);
        scenarios::run(&api, scenario.name, case, &mut observation).unwrap();
        observation.finish(scenario.name);
    }
}
