//! Soak the generated-program differential fuzzers on the CPU interpreter: an ignored test of this
//! crate, because the interpreter is this crate's test support.
//!
//! ```sh
//! METAL2VULKAN_TEST_FUZZ_SOAK='--seeds 1000 --profile lavapipe' \
//!   cargo nextest run --release -p metal2vulkan --all-features --run-ignored only fuzz_soak
//! ```
//!
//! The ordinary test gate runs 24 seeds per generator (`fuzz_interp`); this runs `--seeds N` of
//! them, from `--first S` (default 1), through the engine on `CpuInterp` for one committed
//! profile and one dispatch width (`--threads`, 32 or 64). It never opens a device. The seed range
//! is split into chunks that `--jobs` worker threads (default: the host's logical CPUs) take in
//! turn, each with its own executor; every translation stays within its own budget, so the soak's
//! peak is about `jobs` translations at once. Every refusal, trap, or divergence is printed, in
//! seed order, with each generator's slowest program (assembly, compile, L3, and every interpreter
//! run), and the test fails when any generator reported one.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::fuzz_interp::{CpuInterp, PROFILE};
use metal2vulkan_devtools::fuzz::{
    access, inline, irreducible, pointer_select, structured, workgroup, Execute, Kernel, Report,
    Run, Target, THREADS,
};

/// The variable that holds the soak's arguments.
const ARGS: &str = "METAL2VULKAN_TEST_FUZZ_SOAK";

const USAGE: &str = "METAL2VULKAN_TEST_FUZZ_SOAK='--seeds N [--first S] [--profile STEM] \
                     [--threads T] [--jobs J] [--generator NAME]...'
  --seeds N         seeds per generator (required)
  --first S         first seed (default 1)
  --profile STEM    committed profile validation/profiles/STEM.json (default moltenvk-m3)
  --threads T       threads in the dispatch's one threadgroup, 32 or 64 (default 32)
  --jobs J          worker threads (default: the host's logical CPUs)
  --generator NAME  run only NAME; repeatable. Names: structured, irreducible, inline,
                    workgroup, access-load, access-store, access-race, pointer-select";

type Check = fn(&mut dyn Execute, std::ops::RangeInclusive<u64>) -> Report;

/// What a generator does with the seed range.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Seeds {
    /// One program per seed.
    Each,
    /// One fixed program: run once, whatever the range.
    Fixed,
}

const GENERATORS: [(&str, Check, Seeds); 8] = [
    ("structured", structured::check, Seeds::Each),
    ("irreducible", irreducible::check, Seeds::Each),
    ("inline", inline::check, Seeds::Each),
    ("workgroup", workgroup::check, Seeds::Each),
    ("access-load", access::check_load, Seeds::Each),
    ("access-store", access::check_store, Seeds::Each),
    ("pointer-select", pointer_select::check, Seeds::Each),
    (
        "access-race",
        |executor, _| access::check_race(executor),
        Seeds::Fixed,
    ),
];

/// Seeds one worker takes at a time: small enough that the workers finish together, large
/// enough that taking a chunk costs nothing.
const CHUNK: u64 = 8;

struct Args {
    seeds: u64,
    first: u64,
    profile: String,
    threads: u32,
    jobs: usize,
    generators: Vec<String>,
}

fn parse(mut args: impl Iterator<Item = String>) -> Result<Args, String> {
    let mut seeds = None;
    let mut first = 1u64;
    let mut profile = PROFILE.to_owned();
    let mut threads = THREADS;
    let mut jobs = None;
    let mut generators = Vec::new();
    while let Some(arg) = args.next() {
        let mut value = || args.next().ok_or_else(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--seeds" => {
                seeds = Some(
                    value()?
                        .parse::<u64>()
                        .map_err(|error| format!("--seeds: {error}"))?,
                )
            }
            "--first" => {
                first = value()?
                    .parse::<u64>()
                    .map_err(|error| format!("--first: {error}"))?
            }
            "--profile" => profile = value()?,
            "--threads" => {
                threads = value()?
                    .parse::<u32>()
                    .map_err(|error| format!("--threads: {error}"))?
            }
            "--jobs" => {
                jobs = Some(
                    value()?
                        .parse::<usize>()
                        .map_err(|error| format!("--jobs: {error}"))?,
                )
            }
            "--generator" => {
                let name = value()?;
                if !GENERATORS.iter().any(|(known, _, _)| *known == name) {
                    return Err(format!("unknown generator {name:?}"));
                }
                generators.push(name);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    let seeds = seeds.ok_or("--seeds is required")?;
    if seeds == 0 {
        return Err("--seeds must be at least 1".to_owned());
    }
    first
        .checked_add(seeds - 1)
        .ok_or("--first + --seeds overflows")?;
    let jobs = match jobs {
        Some(0) => return Err("--jobs must be at least 1".to_owned()),
        Some(jobs) => jobs,
        None => std::thread::available_parallelism().map_or(1, usize::from),
    };
    Ok(Args {
        seeds,
        first,
        profile,
        threads,
        jobs,
        generators,
    })
}

/// A new executor for `args`.
fn executor(args: &Args) -> Result<CpuInterp, String> {
    match CpuInterp::new(&args.profile)? {
        Some(executor) => executor.with_threads(args.threads),
        None => Err("llvm-as is not installed".to_owned()),
    }
}

/// The slowest program one worker ran: assembly, compile, L3, and every interpreter run.
#[derive(Default)]
struct Slowest {
    took: Duration,
    entry: String,
}

impl Slowest {
    fn keep(&mut self, other: Slowest) {
        if other.took > self.took {
            *self = other;
        }
    }
}

/// [`CpuInterp`] timing each program.
struct Timed {
    inner: CpuInterp,
    slowest: Slowest,
}

impl Execute for Timed {
    fn target(&self) -> Target {
        self.inner.target()
    }

    fn execute(&mut self, kernel: &Kernel<'_>) -> Result<Run, String> {
        let started = Instant::now();
        let run = self.inner.execute(kernel);
        self.slowest.keep(Slowest {
            took: started.elapsed(),
            entry: kernel.entry.to_owned(),
        });
        run
    }
}

/// Run `check` over `seeds` on `args.jobs` workers, chunk by chunk, and merge the chunks'
/// reports in seed order; also the slowest program.
fn soak(
    args: &Args,
    check: Check,
    seeds: std::ops::RangeInclusive<u64>,
) -> Result<(Report, Slowest), String> {
    let (first, last) = (*seeds.start(), *seeds.end());
    let next = AtomicU64::new(first);
    let chunks: Mutex<Vec<(u64, Report)>> = Mutex::new(Vec::new());
    let slowest = Mutex::new(Slowest::default());
    let workers = args
        .jobs
        .min(usize::try_from((last - first) / CHUNK + 1).unwrap_or(usize::MAX));
    std::thread::scope(|scope| {
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                scope.spawn(|| -> Result<(), String> {
                    let mut executor = Timed {
                        inner: executor(args)?,
                        slowest: Slowest::default(),
                    };
                    loop {
                        let start = next.fetch_add(CHUNK, Ordering::Relaxed);
                        if start > last {
                            slowest
                                .lock()
                                .map_err(|_| "a worker panicked".to_owned())?
                                .keep(executor.slowest);
                            return Ok(());
                        }
                        let end = start.saturating_add(CHUNK - 1).min(last);
                        let report = check(&mut executor, start..=end);
                        chunks
                            .lock()
                            .map_err(|_| "a worker panicked".to_owned())?
                            .push((start, report));
                    }
                })
            })
            .collect();
        handles
            .into_iter()
            .try_for_each(|handle| handle.join().map_err(|_| "a worker panicked".to_owned())?)
    })?;
    let mut chunks = chunks
        .into_inner()
        .map_err(|_| "a worker panicked".to_owned())?;
    chunks.sort_by_key(|(start, _)| *start);
    let mut report = Report::default();
    for (_, chunk) in chunks {
        report.absorb(chunk);
    }
    let slowest = slowest
        .into_inner()
        .map_err(|_| "a worker panicked".to_owned())?;
    Ok((report, slowest))
}

/// The soak over the arguments in [`ARGS`]; ignored, so only a run that asks for it soaks.
#[test]
#[ignore = "a soak: set METAL2VULKAN_TEST_FUZZ_SOAK and run ignored tests"]
fn fuzz_soak() {
    let text = std::env::var(ARGS).unwrap_or_else(|_| panic!("{ARGS} is not set\n{USAGE}"));
    let args = parse(text.split_whitespace().map(str::to_owned))
        .unwrap_or_else(|error| panic!("{ARGS}: {error}\n{USAGE}"));
    // Fail on a bad profile, width, or missing `llvm-as` before any generator runs.
    executor(&args).unwrap_or_else(|error| panic!("fuzz soak: {error}"));
    let seeds = args.first..=args.first + (args.seeds - 1);
    let mut unclean = Vec::new();
    for (name, check, per_seed) in GENERATORS {
        if !args.generators.is_empty() && !args.generators.iter().any(|chosen| chosen == name) {
            continue;
        }
        let started = Instant::now();
        let range = match per_seed {
            Seeds::Each => seeds.clone(),
            Seeds::Fixed => *seeds.start()..=*seeds.start(),
        };
        let (report, slowest) = soak(&args, check, range.clone())
            .unwrap_or_else(|error| panic!("fuzz soak: {name}: {error}"));
        println!(
            "{name}: seeds {}..={} on {} at {} threads ({} jobs): {} executed, {} declined, {} \
             failures ({:.1} s; slowest program {} in {:.3} s)",
            range.start(),
            range.end(),
            args.profile,
            args.threads,
            args.jobs,
            report.executed,
            report.declined,
            report.failures.len(),
            started.elapsed().as_secs_f64(),
            slowest.entry,
            slowest.took.as_secs_f64()
        );
        for failure in &report.failures {
            println!("  {failure}");
        }
        if !report.failures.is_empty() || report.executed == 0 {
            unclean.push(name);
        }
    }
    assert!(unclean.is_empty(), "fuzz soak: not clean: {unclean:?}");
}
