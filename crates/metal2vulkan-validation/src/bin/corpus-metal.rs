mod metal_batch;

use metal2vulkan_validation::candidate::exit_status;
use metal2vulkan_validation::check::admission::ExecutorError;
use metal2vulkan_validation::gpu::compile::{set_compile_timeout, COMPILE_TIMEOUT_EXIT_CODE};
use metal2vulkan_validation::gpu::wait::{parse_wait_timeout, set_wait_timeout};
use metal2vulkan_validation::gpu::DeviceType;
use metal2vulkan_validation::metal::{
    identity, prepare, probe_compile, qualify, REFERENCE_GAP_EXIT,
};
use metal2vulkan_validation::observation::MetalStatus;
use metal2vulkan_validation::oracle_fuel::{LoopSelection, OracleBound};
use metal2vulkan_validation::source::corpus_root;
use metal2vulkan_validation::store::CorpusStore;
use std::path::PathBuf;

fn main() {
    // The batch modes run this executable once per case; every other invocation is the single-case
    // qualification.
    if metal_batch::serve() {
        return;
    }
    match run() {
        Ok(()) => {}
        // Refused by policy (`execution_refused{...}`), distinct from a failure (1).
        Err(ExecutorError::Refused(refused)) => {
            println!("{refused}");
            std::process::exit(exit_status::REFUSED);
        }
        Err(
            error @ (ExecutorError::Failed(_)
            | ExecutorError::NotSupported(_)
            | ExecutorError::DeviceCrash(_)
            | ExecutorError::NeedsMetal(_)
            | ExecutorError::VenueMismatch(_)),
        ) => {
            eprintln!("corpus-metal: {error}");
            std::process::exit(1);
        }
    }
}

fn run() -> Result<(), ExecutorError> {
    let mut root = corpus_root();
    let mut case_id = None;
    let mut air_sha256 = None;
    let mut environment_id = None;
    let mut device_type = None;
    let mut prepare_only = false;
    let mut print_identity = false;
    let mut compile_probe = None;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--corpus" => root = PathBuf::from(required(&mut args, "--corpus")?),
            "--private-corpus" => metal2vulkan_validation::source::use_private_corpus_root(
                PathBuf::from(required(&mut args, "--private-corpus")?),
            )?,
            "--case-id" => case_id = Some(required(&mut args, "--case-id")?),
            "--air-sha256" => air_sha256 = Some(required(&mut args, "--air-sha256")?),
            "--source-index" => metal2vulkan_validation::source::use_source_index(PathBuf::from(
                required(&mut args, "--source-index")?,
            ))?,
            "--environment-id" => environment_id = Some(required(&mut args, "--environment-id")?),
            "--device-type" => {
                device_type = Some(DeviceType::parse(&required(&mut args, "--device-type")?)?)
            }
            "--prepare-only" => prepare_only = true,
            "--identity" => print_identity = true,
            "--compile-probe" => {
                compile_probe = Some(parse_probe(&required(&mut args, "--compile-probe")?)?)
            }
            "--compile-timeout" => set_compile_timeout(&required(&mut args, "--compile-timeout")?)?,
            "--gpu-wait-timeout" => set_wait_timeout(parse_wait_timeout(&required(
                &mut args,
                "--gpu-wait-timeout",
            )?)?)?,
            "-h" | "--help" => {
                println!(
                    "usage: corpus-metal [--corpus DIR] [--private-corpus DIR] --case-id HASH --environment-id ID \\\n\
                     --device-type gpu [--gpu-wait-timeout SECONDS] [--compile-timeout SECONDS] \\\n\
                     [--air-sha256 HASH] [--source-index FILE]\n\
                     corpus-metal [--corpus DIR] [--private-corpus DIR] --case-id HASH --prepare-only \\\n\
                     [--air-sha256 HASH] [--source-index FILE]\n\
                     corpus-metal [--corpus DIR] [--private-corpus DIR] --case-id HASH \\\n\
                     --compile-probe VARIANT [--compile-timeout SECONDS]\n\
                     corpus-metal --batch|--batch-report|--batch-install ... (see corpus-metal --batch --help)\n\
                     corpus-metal --identity\n\
                     Metal has no CPU device.\n\
                     --air-sha256 HASH, the case's AIR, reads the case from only the case shard HASH\n\
                     selects (refused if the case there is over another AIR); without it every case\n\
                     shard is read. --source-index FILE reads private sources through that corpus\n\
                     index's byte locations, read only: a location that is missing, stale, or not the\n\
                     source falls back to parsing the source's shard.\n\
                     --identity prints this host's Metal identity (device, GPU cores, macOS version\n\
                     and build, Xcode build, Metal toolchain) as JSON, then the environment ID it\n\
                     produces, and exits. A row is recorded under an environment of\n\
                     the C3 reference list (validation/cutover/c3-references.json) only on a host\n\
                     of the identity the list declares for it; any other host fails the case.\n\
                     A case that is not mechanically loop-free runs fuel-instrumented AIR, and only\n\
                     when fuel bounds every cycle; otherwise it is refused (exit 3).\n\
                     --prepare-only stops before any device opens: it admits the case, bounds\n\
                     every module it compiles, instruments its AIR, and runs metal-opt and\n\
                     metallib for its library, its pipeline templates, and its linked modules,\n\
                     then prints the result and exits 0 (a refusal exits 3, any other failure 1).\n\
                     --compile-probe, a diagnostic for a kernel case without command buffers, times\n\
                     compute pipeline creation of the case's own module as compiled, or\n\
                     fuel-instrumented with only some loops checked, with its function constants and\n\
                     linked functions. VARIANT is as-compiled, oracle-loops (the oracle's own\n\
                     instrumentation: every loop but the counted nests), entry-loops (the entry's\n\
                     own), callee-loops (its callees'), all-loops (every loop, the oracle's before\n\
                     v48), barrier-loops, other-loops (no barrier), counted-loops (a constant bounds\n\
                     their header visits), or uncounted-loops. It opens the device, runs nothing,\n\
                     records nothing, and prints the time.\n\
                     Each host tool (metal, metal-as, metal-opt, metallib) is killed with its\n\
                     process group past --compile-timeout (default 60 s), failing the case; a\n\
                     pipeline creation that outlasts it ends the process with exit\n\
                     {COMPILE_TIMEOUT_EXIT_CODE}.\n\
                     A pipeline creation that names an Apple GPU family (GPUFamily<N>) the device\n\
                     does not support, in every run, records a reference gap: a row with no\n\
                     output, which no candidate compares against (exit {REFERENCE_GAP_EXIT})."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg:?}").into()),
        }
    }
    if print_identity {
        let identity = identity()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&identity).map_err(|error| error.to_string())?
        );
        println!("environment_id {}", identity.environment_id());
        return Ok(());
    }
    let case_id = case_id.ok_or("--case-id is required")?;
    let store = CorpusStore::new(&root);
    let find = || -> Result<_, ExecutorError> {
        Ok(match &air_sha256 {
            Some(air_sha256) => store.find_case_of_air(&case_id, air_sha256)?,
            None => store.find_case(&case_id)?,
        }
        .ok_or_else(|| format!("case_id {case_id} not found"))?)
    };
    if let Some(loops) = compile_probe {
        let probe = probe_compile(&root, find()?, loops)?;
        let fuel = probe.fuel.map_or_else(String::new, |report| {
            format!(
                " functions={} loops={} barrier_loops={} tail_calls={} exempt_loops={} \
                 exempt_nest={}",
                report.functions,
                report.loops,
                report.barrier_loops,
                report.tail_calls,
                report.exempt_loops,
                report.exempt_nest
            )
        });
        println!(
            "compile-probe {case_id} variant={} seconds={:.3}{fuel}",
            probe_name(loops),
            probe.seconds
        );
        return Ok(());
    }
    if prepare_only {
        let case = find()?;
        let prepared = prepare(&root, case)?;
        let bound = match prepared.bound {
            OracleBound::LoopFree => "loop_free".to_string(),
            OracleBound::Fuel { invocation_fuel } => format!("fuel({invocation_fuel})"),
        };
        let fuel = prepared.fuel.map_or_else(String::new, |report| {
            format!(
                " functions={} loops={} barrier_loops={} tail_calls={} exempt_loops={} \
                 exempt_nest={}",
                report.functions,
                report.loops,
                report.barrier_loops,
                report.tail_calls,
                report.exempt_loops,
                report.exempt_nest
            )
        });
        let fuel = match prepared.status_slot {
            Some(slot) => format!("{fuel} status_slot={slot}"),
            None => fuel,
        };
        println!(
            "prepared {case_id} bound={bound}{fuel} metallib_bytes={} libraries={} \
             library_sha256={} binding_sha256={}",
            prepared.metallib_bytes,
            prepared.libraries,
            prepared.library_sha256,
            prepared.binding_sha256
        );
        return Ok(());
    }
    let environment_id = environment_id.ok_or("--environment-id is required")?;
    match device_type.ok_or("--device-type is required")? {
        DeviceType::Gpu => {}
        DeviceType::Cpu => return Err("Metal has no CPU device; pass --device-type gpu".into()),
    }
    let observation = qualify(&root, find()?, &environment_id)?;
    if let MetalStatus::ReferenceGap(gap) = &observation.status {
        println!(
            "reference_gap {} apple_family={} message={:?}",
            observation.case_id, gap.apple_family, gap.message
        );
        std::process::exit(REFERENCE_GAP_EXIT);
    }
    println!(
        "qualified {} output_sha256={}",
        observation.case_id, observation.metal_output_sha256
    );
    Ok(())
}

/// The instrumentation a `--compile-probe` names: `None` for the module as compiled.
fn parse_probe(name: &str) -> Result<Option<LoopSelection>, String> {
    match name {
        "as-compiled" => Ok(None),
        "oracle-loops" => Ok(Some(LoopSelection::Oracle)),
        "entry-loops" => Ok(Some(LoopSelection::Entry)),
        "callee-loops" => Ok(Some(LoopSelection::Callees)),
        "all-loops" => Ok(Some(LoopSelection::All)),
        "barrier-loops" => Ok(Some(LoopSelection::Barrier)),
        "other-loops" => Ok(Some(LoopSelection::NonBarrier)),
        "counted-loops" => Ok(Some(LoopSelection::Counted)),
        "uncounted-loops" => Ok(Some(LoopSelection::Uncounted)),
        _ => Err(format!(
            "--compile-probe takes as-compiled, oracle-loops, entry-loops, callee-loops, \
             all-loops, barrier-loops, other-loops, counted-loops, or uncounted-loops, not \
             {name:?}"
        )),
    }
}

/// The name [`parse_probe`] reads for `loops`.
fn probe_name(loops: Option<LoopSelection>) -> &'static str {
    match loops {
        None => "as-compiled",
        Some(LoopSelection::Oracle) => "oracle-loops",
        Some(LoopSelection::Entry) => "entry-loops",
        Some(LoopSelection::Callees) => "callee-loops",
        Some(LoopSelection::All) => "all-loops",
        Some(LoopSelection::Barrier) => "barrier-loops",
        Some(LoopSelection::NonBarrier) => "other-loops",
        Some(LoopSelection::Counted) => "counted-loops",
        Some(LoopSelection::Uncounted) => "uncounted-loops",
    }
}

fn required(args: &mut impl Iterator<Item = String>, flag: &str) -> Result<String, String> {
    args.next()
        .ok_or_else(|| format!("{flag} requires a value"))
}
