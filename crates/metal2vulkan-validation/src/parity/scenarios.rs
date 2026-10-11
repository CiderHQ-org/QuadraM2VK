//! The scenarios: Metal programs written once against [`Api`], each a set of cases
//! (`parity::SCENARIOS` lists them), and the dispatch from a scenario's name to its module. The
//! steps they share are [`steps`], which every record's fingerprint covers (`parity::HARNESS`);
//! this file, which names every scenario, is in no fingerprint, so adding a scenario stales no
//! other scenario's records.

use super::api::{Api, Observation};

mod buffer_copies;
mod buffer_texture_copies;
mod command_buffer_times;
mod hazards;
pub(super) mod steps;

/// Run `case` of the scenario named `scenario` on `api`, recording into `observation`. An error
/// is the harness's own (a fixture that did not load, a wait that failed), not a record.
pub(crate) fn run<A: Api>(
    api: &A,
    scenario: &str,
    case: &str,
    observation: &mut Observation,
) -> Result<(), String> {
    match scenario {
        "buffer_copies" => buffer_copies::run(api, case, observation),
        "buffer_texture_copies" => buffer_texture_copies::run(api, case, observation),
        "hazards" => hazards::run(api, case, observation),
        "command_buffer_times" => command_buffer_times::run(api, case, observation),
        other => Err(format!("no scenario {other}")),
    }
}
