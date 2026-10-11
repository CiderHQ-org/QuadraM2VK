//! `corpus-metal --batch`, `--batch-report`, and `--batch-install`: the Metal oracle's
//! qualifications on the shared batch driver ([`metal2vulkan_validation::vulkan_batch`]), which
//! runs this executable once per case.

use crate::required;
use metal2vulkan_validation::gpu::tripwire::TripwirePaths;
use metal2vulkan_validation::gpu::wait::parse_wait_timeout;
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use metal2vulkan_validation::vulkan_batch::{
    metal, plan, read_done_list, run_batch, BatchBackend, BatchBuild, BatchConfig, BatchDevice,
    Class, Pass, DONE_LIST,
};
use std::path::PathBuf;
use std::time::Duration;

/// The per-run timeout when none is given: well past the oracle's compile deadline (60 s by
/// default) and its device waits' bound, so only a run stuck outside both reaches it.
const DEFAULT_RUN_TIMEOUT_SECONDS: u64 = 300;

const USAGE: &str =
    "usage: corpus-metal --batch --device-type gpu --out DIR --environment-id ID \\\n\
    [--corpus DIR] [--run-timeout SECONDS] [--limit CASES] [--case-id HASH]... \\\n\
    [--gpu-wait-timeout SECONDS] [--source-index FILE]\n\
  corpus-metal --batch-report --out DIR --environment-id ID [--corpus DIR] [--case-id HASH]...\n\
  corpus-metal --batch-install STORE --out DIR\n\
--batch qualifies every committed case (or each --case-id, which must be one) in the Metal\n\
environment --environment-id through this executable, one process per case, each the run a\n\
single-case invocation makes, in its own session killed whole after --run-timeout seconds\n\
(default 300) and in its own scratch store (links to --corpus, which is never written). Metal has\n\
no CPU device, so one run goes at a time: the batch refuses to start a run while the host's GPU\n\
hang tripwire exists, and stops after a run whose device wait timed out, that a signal ended, or\n\
that it killed at --run-timeout (writing the tripwire if no one did). Each run reads its case\n\
from the one case shard its AIR selects, and its private sources through --source-index\n\
(default: --corpus's .index.sqlite, when it has one), falling back to the source's shard\n\
wherever the index is missing, stale, or wrong. DIR/build.json records the build and venue of\n\
DIR's runs; a batch refuses a DIR of another build or venue. Each finished run appends a line to\n\
DIR/done.tsv (case, metal, outcome, exit, seconds, detail) and each Metal row it recorded to\n\
DIR/rows.jsonl: a qualification's, or a reference gap's (outcome reference_gap: the device\n\
cannot run the case). A case done.tsv lists is skipped, so a rerun resumes.\n\
--limit bounds the cases of one invocation. Runs that did not qualify keep their output in\n\
DIR/logs/.\n\
--batch-report prints DIR's build record against this build (batch-build), then, over the cases\n\
--batch selects with the same --case-id list (other cases' runs are not counted), the runs by\n\
class (qualified, refusal, failure) and outcome, the same per family and per scope, and every\n\
run that did not qualify. --batch-install writes into the store at STORE the Metal row of each\n\
case's last run that recorded one through the store's validated write, and prints\n\
batch-install rows=N store=STORE this_build=yes|no|unrecorded.";

/// Run a batch mode if the command line names one (its first argument): `true` when it did (and
/// it exited the process on an error), `false` for a single-case invocation.
pub fn serve() -> bool {
    let mut args = std::env::args().skip(1);
    let Some(mode) = args.next().filter(|arg| {
        matches!(
            arg.as_str(),
            "--batch" | "--batch-report" | "--batch-install"
        )
    }) else {
        return false;
    };
    if let Err(error) = main(&mode, args.collect()) {
        eprintln!("corpus-metal {mode}: {error}");
        std::process::exit(1);
    }
    true
}

/// Run the batch mode `mode` (`--batch`, `--batch-report`, `--batch-install`) with `args`.
fn main(mode: &str, args: Vec<String>) -> Result<(), String> {
    let mut corpus = corpus_root();
    let mut out = None;
    let mut install = None;
    let mut environment_id = None;
    let mut run_timeout = DEFAULT_RUN_TIMEOUT_SECONDS;
    let mut limit = None;
    let mut case_ids = Vec::new();
    let mut gpu_wait_timeout = None;
    let mut device_type = None;
    let mut source_index = None;
    let mut args = args.into_iter();
    if mode == "--batch-install" {
        install = Some(PathBuf::from(required(&mut args, mode)?));
    }
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => corpus = PathBuf::from(required(&mut args, "--corpus")?),
            "--out" => out = Some(PathBuf::from(required(&mut args, "--out")?)),
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "--run-timeout" => {
                run_timeout =
                    positive(&required(&mut args, "--run-timeout")?, "--run-timeout")? as u64
            }
            "--limit" => limit = Some(positive(&required(&mut args, "--limit")?, "--limit")?),
            "--case-id" => case_ids.push(required(&mut args, "--case-id")?),
            "--gpu-wait-timeout" => {
                let value = required(&mut args, "--gpu-wait-timeout")?;
                gpu_wait_timeout = Some(parse_wait_timeout(&value)?.as_secs());
            }
            "--device-type" => device_type = Some(required(&mut args, "--device-type")?),
            "--source-index" => {
                source_index = Some(PathBuf::from(required(&mut args, "--source-index")?))
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}\n{USAGE}")),
        }
    }
    let out = out.ok_or("--out DIR is required")?;
    let store = CorpusStore::new(&corpus);
    match mode {
        "--batch-install" => {
            let target = install.ok_or("--batch-install STORE is required")?;
            let installed = metal::install(&out, &CorpusStore::new(&target))?;
            println!(
                "batch-install rows={} store={} this_build={}",
                installed.rows,
                target.display(),
                match &installed.build {
                    Some(build) if build.is_this_build() => "yes",
                    Some(_) => "no",
                    None => "unrecorded",
                }
            );
            Ok(())
        }
        "--batch-report" => {
            let environment_id = environment_id.ok_or("--environment-id is required")?;
            let recorded = BatchBuild::read(&out)?;
            // The report reads no device: the recorded device type is compared as it is.
            let this_build = match &recorded {
                Some(recorded) => Some(
                    *recorded
                        == BatchBuild::current(
                            BatchBackend::Metal,
                            &environment_id,
                            &environment_id,
                            None,
                            DeviceType::parse(&recorded.device_type)?,
                        ),
                ),
                None => None,
            };
            println!(
                "batch-build this_build={}{}",
                match this_build {
                    Some(true) => "yes",
                    Some(false) => "no",
                    None => "unrecorded",
                },
                recorded
                    .map(|recorded| format!(
                        " {}",
                        serde_json::to_string(&recorded).unwrap_or_default()
                    ))
                    .unwrap_or_default()
            );
            let cases = metal::selected_cases(store.read_all_cases()?, &case_ids)?;
            for line in metal::report(&cases, &read_done_list(&out.join(DONE_LIST))?) {
                println!("{line}");
            }
            Ok(())
        }
        "--batch" => {
            match DeviceType::parse(
                device_type
                    .as_deref()
                    .ok_or("--batch needs --device-type gpu")?,
            )? {
                DeviceType::Gpu => {}
                DeviceType::Cpu => {
                    return Err("Metal has no CPU device: pass --device-type gpu".into())
                }
            }
            let environment_id = environment_id.ok_or("--environment-id is required")?;
            let cases = metal::selected_cases(store.read_all_cases()?, &case_ids)?;
            let done = read_done_list(&out.join(DONE_LIST))?;
            let runs = plan(&cases, &[Pass::Metal], &done, limit);
            // The corpus's own index, when no other is named and it has one.
            let source_index = source_index.or_else(|| {
                Some(metal2vulkan_validation::index::default_index_path(&corpus))
                    .filter(|index| index.is_file())
            });
            let config = BatchConfig {
                executor: std::env::current_exe()
                    .map_err(|error| format!("this executable: {error}"))?,
                backend: BatchBackend::Metal,
                corpus,
                out,
                metal_environment_id: environment_id.clone(),
                environment_id,
                restricted_to: None,
                device: BatchDevice::Gpu(TripwirePaths::for_host()?),
                jobs: 1,
                timeout: Duration::from_secs(run_timeout),
                gpu_wait_timeout,
                source_index,
            };
            eprintln!(
                "batch: cases={} done_before={} runs={} device=gpu jobs=1 run_timeout={run_timeout}s \
                 pass=metal",
                cases.len(),
                done.len(),
                runs.len(),
            );
            let finished = run_batch(&config, &runs, |run| {
                println!(
                    "batch-run {} {} {} {} {:.3}s {}",
                    run.case_id,
                    run.pass.label(),
                    run.outcome.label(),
                    run.ended,
                    run.seconds,
                    run.detail
                );
            })?;
            let of = |class: Class| {
                finished
                    .iter()
                    .filter(|run| run.outcome.class() == class)
                    .count()
            };
            println!(
                "batch-summary pass=metal ran={} qualified={} refusal={} failure={}",
                finished.len(),
                of(Class::Match),
                of(Class::Refusal),
                of(Class::Failure)
            );
            if let Some(slowest) = finished
                .iter()
                .max_by(|left, right| left.seconds.total_cmp(&right.seconds))
            {
                println!(
                    "batch-slowest {} {} {:.3}s {}",
                    slowest.case_id,
                    slowest.pass.label(),
                    slowest.seconds,
                    slowest.outcome.label()
                );
            }
            let pending = plan(
                &cases,
                &[Pass::Metal],
                &read_done_list(&config.out.join(DONE_LIST))?,
                None,
            )
            .len();
            println!("batch-pending runs={pending}");
            Ok(())
        }
        _ => Err(format!("unknown batch mode {mode:?}")),
    }
}

fn positive(value: &str, flag: &str) -> Result<usize, String> {
    match value.parse::<usize>() {
        Ok(number) if number > 0 => Ok(number),
        _ => Err(format!(
            "{flag} {value:?}: expected a positive whole number"
        )),
    }
}
