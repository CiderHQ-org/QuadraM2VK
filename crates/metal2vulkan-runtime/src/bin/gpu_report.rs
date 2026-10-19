//! `metal2vulkan-gpu-report`: what a device decides about its driver, as Chrome's `chrome://gpu`
//! says per feature. It makes the device with its checks, waiting for every one as the device is
//! made (`DeviceChecksOptions::wait`), so it prints settled answers, in a scratch checks directory
//! unless `--checks-dir` names one, and prints the setup's keys, each fact's decision and why, and
//! what each check cost.
//!
//! ```sh
//! cargo install --git https://github.com/steelbrain/metal2vulkan metal2vulkan-runtime --features gpu-report
//! metal2vulkan-gpu-report [--device N | --all] [--json] [--checks-dir DIR]
//! ```

use metal2vulkan::device::{release_name, CheckAnswer, CheckOf, Decided, FactStatus};
use metal2vulkan_runtime::{
    CheckRun, CheckSource, Device, DeviceChecksOptions, DeviceOptions, FeatureStatus, MessageSink,
    Settle, Settled, ShaderCacheOptions,
};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const USAGE: &str = "\
usage: metal2vulkan-gpu-report [--device N | --all] [--json] [--checks-dir DIR]

Makes a device with its checks of its driver and prints its feature status: the setup's keys,
each driver fact's decision and why, and each check's cost.

  --device N        the Nth device the runtime can make, in the system default's order
                    (default: the system default device)
  --all             every device the runtime can make
  --json            print JSON instead of text
  --checks-dir DIR  keep the checks' answers and the learned records in DIR (default: a scratch
                    directory removed at exit, so every check runs and no record is read)
";

/// The id name of the feature status message, which the report prints in its own form.
const FEATURE_STATUS: &str = "metal2vulkan-feature-status";

/// Which devices to report.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Which {
    SystemDefault,
    Index(usize),
    All,
}

/// The command line.
#[derive(Debug)]
struct Args {
    which: Which,
    json: bool,
    checks_dir: Option<PathBuf>,
    help: bool,
}

impl Args {
    /// The arguments after the program's name, or why they do not parse.
    fn parse(mut args: impl Iterator<Item = std::ffi::OsString>) -> Result<Args, String> {
        let mut parsed = Args {
            which: Which::SystemDefault,
            json: false,
            checks_dir: None,
            help: false,
        };
        let mut chose = false;
        while let Some(arg) = args.next() {
            let text = arg.to_string_lossy().into_owned();
            match text.as_str() {
                "-h" | "--help" => parsed.help = true,
                "--json" => parsed.json = true,
                "--all" | "--device" if chose => {
                    return Err("give one of --device and --all, once".to_owned());
                }
                "--all" => {
                    parsed.which = Which::All;
                    chose = true;
                }
                "--device" => {
                    let index = args.next().ok_or("--device needs a number")?;
                    let index = index.to_string_lossy();
                    let index = index
                        .parse()
                        .map_err(|_| format!("--device needs a number, not {index}"))?;
                    parsed.which = Which::Index(index);
                    chose = true;
                }
                "--checks-dir" => {
                    let dir = args.next().ok_or("--checks-dir needs a directory")?;
                    parsed.checks_dir = Some(PathBuf::from(dir));
                }
                _ => return Err(format!("unknown argument {text}")),
            }
        }
        Ok(parsed)
    }
}

/// A directory of the run's own under the OS temp dir, removed when the run ends.
struct Scratch(PathBuf);

impl Scratch {
    fn new() -> std::io::Result<Scratch> {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.subsec_nanos());
        let path = std::env::temp_dir().join(format!(
            "metal2vulkan-gpu-report-{}-{nanos}",
            std::process::id()
        ));
        std::fs::create_dir(&path)?;
        Ok(Scratch(path))
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args_os().skip(1)) {
        Ok(args) => args,
        Err(why) => {
            eprintln!("metal2vulkan-gpu-report: {why}\n\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    if args.help {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }
    let scratch = match Scratch::new() {
        Ok(scratch) => scratch,
        Err(error) => {
            eprintln!("metal2vulkan-gpu-report: no scratch directory: {error}");
            return ExitCode::FAILURE;
        }
    };
    // The devices are gone before the scratch directory is.
    let result = report(&args, &scratch.0);
    drop(scratch);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(why) => {
            eprintln!("metal2vulkan-gpu-report: {why}");
            ExitCode::FAILURE
        }
    }
}

/// Make the devices `args` names, with their checks in `args`' directory or `scratch`, and print
/// their reports.
fn report(args: &Args, scratch: &Path) -> Result<(), String> {
    let checks_dir = match &args.checks_dir {
        Some(dir) => dir.clone(),
        None => scratch.join("device-checks"),
    };
    let mut device_checks = DeviceChecksOptions::default();
    device_checks.directory = Some(checks_dir.clone());
    // Every check runs as the device is made: the report prints settled answers, none pending.
    device_checks.wait = true;
    // The report's devices make no pipeline of the program's: their shader cache is the run's.
    let mut shader_cache = ShaderCacheOptions::default();
    shader_cache.directory = Some(scratch.join("shader-cache"));
    let mut options = DeviceOptions::default();
    options.device_checks = device_checks;
    options.shader_cache = shader_cache;
    // Every message but the feature status, which the report prints, goes to stderr.
    options.messages = Some(MessageSink::new(|message| {
        if message.id_name != FEATURE_STATUS {
            eprintln!(
                "{:?} {}: {}",
                message.severity, message.id_name, message.text
            );
        }
    }));
    let devices: Vec<(usize, Device)> = match args.which {
        Which::SystemDefault => {
            let device = Device::system_default_with(&options).map_err(|e| e.to_string())?;
            vec![(0, device)]
        }
        Which::All => (Device::all_with(&options).map_err(|e| e.to_string())?)
            .into_iter()
            .enumerate()
            .collect(),
        Which::Index(index) => {
            let mut devices = Device::all_with(&options).map_err(|e| e.to_string())?;
            let count = devices.len();
            if index >= count {
                return Err(format!(
                    "no device {index}: the runtime can make {count} (0 to {})",
                    count.saturating_sub(1)
                ));
            }
            vec![(index, devices.swap_remove(index))]
        }
    };
    if devices.is_empty() {
        return Err("the runtime can make no device here".to_owned());
    }
    let scratch_dir = args.checks_dir.is_none();
    if args.json {
        let reports: Vec<Value> = (devices.iter())
            .map(|(index, device)| json_report(*index, device, &checks_dir, scratch_dir))
            .collect();
        let text = serde_json::to_string_pretty(&reports).map_err(|e| e.to_string())?;
        println!("{text}");
    } else {
        for (at, (index, device)) in devices.iter().enumerate() {
            if at > 0 {
                println!();
            }
            print!("{}", text_report(*index, device, &checks_dir, scratch_dir));
        }
    }
    Ok(())
}

/// Milliseconds, to a tenth of a microsecond.
fn ms(duration: Duration) -> f64 {
    (duration.as_secs_f64() * 10_000_000.0).round() / 10_000.0
}

/// The device's driver: its id, release and PCI ids.
fn driver_text(device: &Device) -> String {
    let description = &device.properties().description;
    format!(
        "{:?} {} (driverVersion 0x{:08x}), {:04x}:{:04x}",
        description.driver_id,
        release_name(description.driver_id, description.driver_version),
        description.driver_version,
        description.vendor_id,
        description.device_id,
    )
}

/// How the device came by its checks' answers.
fn settled_text(status: &FeatureStatus) -> String {
    match status.settled {
        Settled::Distrusted => {
            "distrusted by the policy: none ran or was read, every fact as the curated tables have it"
                .to_owned()
        }
        Settled::AtCreation { wall } => format!(
            "settled at creation in {} ms ({} run, {} from the cache)",
            ms(wall),
            status.checks_run,
            status.checks_from_cache
        ),
        Settled::FromCache { checked_at } => format!(
            "read from the cache ({} checks, checked at {checked_at} s past the Unix epoch)",
            status.checks_from_cache
        ),
        other => format!("{other:?}"),
    }
}

/// One fact's name and kind.
fn fact_name(row: &FactStatus) -> String {
    format!("{} ({})", row.fact.name(), row.fact.kind())
}

/// Whether the device takes the fact, in the fact's words.
fn taken(row: &FactStatus) -> &'static str {
    match (row.holds, row.fact.is_defect()) {
        (true, false) => "taken",
        (false, false) => "not taken",
        (true, true) => "worked around",
        (false, true) => "not worked around",
    }
}

/// How a fact was decided: a word, and why where there is a reason.
fn decided(decided: &Decided) -> (&'static str, Option<String>) {
    match decided {
        Decided::CheckedMetals => ("checked-metals", None),
        Decided::CheckedNotMetals => ("checked-not-metals", None),
        Decided::Listed(entry) => ("listed", Some(entry.clone())),
        Decided::Assumed(entry) => ("assumed", Some(entry.clone())),
        Decided::Unlisted => ("unlisted", None),
        Decided::Exact(why) => ("exact", Some(why.clone())),
        Decided::LearnedOff(why) => ("learned-off", Some(why.clone())),
        other => ("other", Some(format!("{other:?}"))),
    }
}

/// What became of a fact's check: a word, and why where there is a reason.
fn check_of(check: &CheckOf) -> (&'static str, Option<String>) {
    match check {
        CheckOf::None => ("none", None),
        CheckOf::Pending => ("pending", None),
        CheckOf::Distrusted => ("distrusted", None),
        CheckOf::Answered(CheckAnswer::Metals) => ("metals", None),
        CheckOf::Answered(CheckAnswer::NotMetals(how)) => ("not-metals", Some(how.clone())),
        CheckOf::Answered(CheckAnswer::Unanswered(why)) => ("unanswered", Some(why.clone())),
        CheckOf::Answered(CheckAnswer::NotRun(why)) => ("not-run", Some(why.clone())),
        other => ("other", Some(format!("{other:?}"))),
    }
}

/// When a check's answers must be in.
fn settle_word(settle: Settle) -> &'static str {
    match settle {
        Settle::AtCreation => "at-creation",
        Settle::Background => "background",
        _ => "other",
    }
}

/// The facts a check decides, by name.
fn check_facts(check: &CheckRun) -> Vec<String> {
    check.facts.iter().map(|fact| fact.name()).collect()
}

/// The report of the device numbered `index`, in text, its checks kept in `checks_dir` (the run's
/// own where `scratch`).
fn text_report(index: usize, device: &Device, checks_dir: &Path, scratch: bool) -> String {
    let status = &device.properties().feature_status;
    let mut out = format!("device {index}: {}\n", device.name());
    out.push_str(&format!("  driver: {}\n", driver_text(device)));
    out.push_str(&format!("  setup key: {}\n", status.setup));
    out.push_str(&format!("  checks: {}\n", settled_text(status)));
    let kept = match scratch {
        true => " (scratch, removed at exit)",
        false => "",
    };
    out.push_str(&format!(
        "  checks directory: {}{kept}\n",
        checks_dir.display()
    ));
    if let Some(why) = &status.cache_error {
        out.push_str(&format!("  cache not used: {why}\n"));
    }
    let learning = &status.learning;
    let records = match (&learning.records, learning.read) {
        (None, _) => "kept nowhere (no checks directory)".to_owned(),
        (Some(dir), true) => format!("in {}", dir.display()),
        (Some(dir), false) => format!("in {} (not read: checks distrusted)", dir.display()),
    };
    let watched = match learning.watched {
        true => "watched for a lost device",
        false => "unwatched for a lost device (every watch slot taken)",
    };
    out.push_str(&format!(
        "  learned key: {}, records {records}; {watched}\n",
        learning.key
    ));
    out.push_str("  check costs:\n");
    for check in &status.checks {
        let source = match check.source {
            CheckSource::Ran(cost) => format!(
                "ran in {} ms wall, {} GPU (longest submission {}), {} pipelines",
                ms(cost.wall),
                cost.gpu
                    .map_or("no timestamps".to_owned(), |gpu| format!("{} ms", ms(gpu))),
                cost.longest
                    .map_or("no timestamps".to_owned(), |gpu| format!("{} ms", ms(gpu))),
                cost.pipelines
            ),
            CheckSource::Cache => "read from the cache".to_owned(),
            CheckSource::Skipped => "skipped (its answers say why)".to_owned(),
            other => format!("{other:?}"),
        };
        out.push_str(&format!(
            "    {} [{}]: {source}\n",
            check_facts(check).join(", "),
            settle_word(check.settle)
        ));
    }
    out.push_str("  facts:\n");
    for row in &status.facts {
        let (how, why) = decided(&row.decided);
        let why = why.map_or(String::new(), |why| format!(": {why}"));
        let (check, check_why) = check_of(&row.check);
        let check = match (check, check_why) {
            ("none", _) => String::new(),
            (word, None) => format!(" [check: {word}]"),
            (word, Some(why)) => format!(" [check: {word}: {why}]"),
        };
        out.push_str(&format!(
            "    {}: {}, {how}{why}{check}\n",
            fact_name(row),
            taken(row)
        ));
    }
    out
}

/// The report of the device numbered `index`, in JSON (as [`text_report`]).
fn json_report(index: usize, device: &Device, checks_dir: &Path, scratch: bool) -> Value {
    let status = &device.properties().feature_status;
    let description = &device.properties().description;
    let settled = match status.settled {
        Settled::Distrusted => json!({ "kind": "distrusted" }),
        Settled::AtCreation { wall } => json!({ "kind": "at-creation", "wall_ms": ms(wall) }),
        Settled::FromCache { checked_at } => {
            json!({ "kind": "from-cache", "checked_at": checked_at })
        }
        other => json!({ "kind": "other", "debug": format!("{other:?}") }),
    };
    let checks: Vec<Value> = (status.checks.iter())
        .map(|check| {
            let source = match check.source {
                CheckSource::Ran(cost) => json!({
                    "kind": "ran",
                    "wall_ms": ms(cost.wall),
                    "gpu_ms": cost.gpu.map(ms),
                    "longest_ms": cost.longest.map(ms),
                    "pipelines": cost.pipelines,
                }),
                CheckSource::Cache => json!({ "kind": "cache" }),
                CheckSource::Skipped => json!({ "kind": "skipped" }),
                other => json!({ "kind": "other", "debug": format!("{other:?}") }),
            };
            json!({
                "facts": check_facts(check),
                "settle": settle_word(check.settle),
                "source": source,
            })
        })
        .collect();
    let facts: Vec<Value> = (status.facts.iter())
        .map(|row| {
            let (how, why) = decided(&row.decided);
            let (check, check_why) = check_of(&row.check);
            json!({
                "fact": row.fact.name(),
                "kind": row.fact.kind(),
                "holds": row.holds,
                "taken": taken(row),
                "decided": how,
                "why": why,
                "check": check,
                "check_why": check_why,
            })
        })
        .collect();
    let learning = &status.learning;
    json!({
        "index": index,
        "name": device.name(),
        "driver_id": format!("{:?}", description.driver_id),
        "driver_version": description.driver_version,
        "release": release_name(description.driver_id, description.driver_version),
        "vendor_id": description.vendor_id,
        "device_id": description.device_id,
        "setup_key": status.setup,
        "settled": settled,
        "checks_run": status.checks_run,
        "checks_from_cache": status.checks_from_cache,
        "checks_directory": checks_dir,
        "checks_directory_is_scratch": scratch,
        "cache_error": status.cache_error,
        "learning": {
            "key": learning.key,
            "records": learning.records,
            "read": learning.read,
            "watched": learning.watched,
        },
        "checks": checks,
        "facts": facts,
    })
}
