//! The host A/B benchmark: the same Metal workloads run two ways on one machine, natively on
//! Metal (macOS only, through objc2-metal) and through this runtime on the Vulkan device its
//! loader finds (on a Mac, MoltenVK), from one metallib that `xcrun metal` compiles from
//! `host_ab.metal`. The runtime takes it through `new_library_with_data`, so its functions run as
//! the translator's SPIR-V.
//!
//! ```sh
//! cargo run --release -p metal2vulkan-runtime --example host_ab -- [options]
//! ```
//!
//! Without `--side`, each round runs one process per side, native first, then prints a table of
//! each metric: the median of the rounds' medians per side, the ratio runtime / native, and the
//! spread of the rounds. `--side native|runtime` runs one side in this process and prints its
//! result lines (for a profiler).
//!
//! Workloads (`--only a,b,...`; all by default):
//!
//! - `encode`: CPU cost of encoding. A render pass of `--draws` small textured draws into a
//!   512x512 `BGRA8Unorm` target, `static` (nothing set between draws), `bind` (each draw another
//!   vertex buffer and offset, and another fragment texture) and `pipeline` (as `bind`, and the
//!   pipeline state switched between two), and a compute encoder of as many single-threadgroup
//!   dispatches, `static` and `churn` (the pipeline state and both buffers' offsets each
//!   dispatch). Per draw and dispatch: the encoding thread's wall time from the command buffer's
//!   creation through `endEncoding`, and the whole process's CPU time until the wait returned.
//! - `fixed`: a command buffer's creation, an encoder of one command each (render, compute,
//!   blit), and the commit call of a command buffer of 30 such encoders.
//! - `latency`: commit to `waitUntilCompleted`'s return, and commit to the completed handler, of
//!   a command buffer that is empty, one blit fill, one dispatch, or one draw; one in flight.
//! - `gpu`: a full-screen heavy fragment (`heavy_fragment`, 2048x2048 `RGBA16Float`) and a heavy
//!   kernel (`heavy_kernel`, 4096 threadgroups of 256), one command buffer in flight: commit to
//!   the wait's return, and native's `GPUEndTime - GPUStartTime`. `--heavy-iterations` sets
//!   the fragment's loop, and 16 times it the kernel's.
//! - `pipeline`: the library, then each of four states' creation and first use, then each made
//!   again in the process. Each round runs it in two processes per side over a metallib of a
//!   fresh `NONCE` (`host_ab.metal`): `cold_*` (no cache holds its functions) and `warm_*` (a
//!   second process over the same runtime shader cache directory, after Metal's own caches took
//!   the first).
//! - `memory`: `newBufferWithLength:` and `newBufferWithBytes:` of 4 KiB, 1 MiB and 64 MiB,
//!   `newBufferWithBytesNoCopy:` of 1 MiB and 64 MiB, and a 1024x1024 `RGBA8Unorm` `Shared`
//!   texture's `replaceRegion` and `getBytes`.
//!
//! Options: `--rounds N` (3), `--draws N` (2000), `--timed N` (30), `--latency N` (200),
//! `--heavy-iterations N` (64), `--heavy-runs N` (15), `--scratch DIR` (metallibs and the
//! runtime's shader caches; a directory under the OS's temporary directory by default, removed at
//! the end), `--metallib PATH` (skip the compile; not with `pipeline`), `--cache DIR` (the
//! runtime's shader cache, with `--side`).
//!
//! On a Mac, the runtime's device opens the Vulkan loader (`/opt/homebrew/lib` among the places
//! it looks) and MoltenVK's `MVK_CONFIG_*` variables apply. Run timings on an otherwise idle GPU.

mod gpu;
#[cfg(target_os = "macos")]
mod native;
mod runtime;
mod workloads;

use gpu::Gpu;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;
use workloads::Sizes;

const WORKLOADS: [&str; 6] = ["encode", "fixed", "latency", "gpu", "pipeline", "memory"];

struct Options {
    side: Option<String>,
    only: Vec<String>,
    rounds: usize,
    sizes: Sizes,
    scratch: Option<PathBuf>,
    metallib: Option<PathBuf>,
    cache: Option<PathBuf>,
    /// Arguments passed on to each child.
    passed: Vec<String>,
}

fn options() -> Options {
    let mut options = Options {
        side: None,
        only: WORKLOADS.iter().map(|w| w.to_string()).collect(),
        rounds: 3,
        sizes: Sizes {
            draws: 2000,
            timed: 30,
            latency: 200,
            heavy_iterations: 64,
            heavy_runs: 15,
        },
        scratch: None,
        metallib: None,
        cache: None,
        passed: vec![],
    };
    let mut args = std::env::args().skip(1);
    while let Some(flag) = args.next() {
        let mut value = || {
            args.next()
                .unwrap_or_else(|| panic!("{flag} takes a value"))
        };
        let mut number = |passed: &mut Vec<String>| {
            let v = value();
            passed.extend([flag.clone(), v.clone()]);
            v.parse::<usize>()
                .unwrap_or_else(|_| panic!("{flag}: a number"))
        };
        match flag.as_str() {
            "--side" => options.side = Some(value()),
            "--only" => options.only = value().split(',').map(str::to_owned).collect(),
            "--rounds" => options.rounds = value().parse().expect("--rounds: a number"),
            "--draws" => options.sizes.draws = number(&mut options.passed),
            "--timed" => options.sizes.timed = number(&mut options.passed),
            "--latency" => options.sizes.latency = number(&mut options.passed),
            "--heavy-iterations" => {
                options.sizes.heavy_iterations = number(&mut options.passed) as u32
            }
            "--heavy-runs" => options.sizes.heavy_runs = number(&mut options.passed),
            "--scratch" => options.scratch = Some(PathBuf::from(value())),
            "--metallib" => options.metallib = Some(PathBuf::from(value())),
            "--cache" => options.cache = Some(PathBuf::from(value())),
            "--help" | "-h" => {
                eprintln!("see the module docs of examples/host_ab/main.rs");
                std::process::exit(0);
            }
            other => panic!("unknown option {other}"),
        }
    }
    for workload in &options.only {
        assert!(
            WORKLOADS.contains(&workload.as_str()),
            "unknown workload {workload}"
        );
    }
    options
}

fn main() {
    let options = options();
    match options.side.as_deref() {
        None => parent(&options),
        Some(side) => child(side, &options),
    }
}

/// One side in this process: open its device, run the selected workloads, print result lines.
fn child(side: &str, options: &Options) {
    let metallib = options.metallib.as_deref().expect("--metallib with --side");
    let t = Instant::now();
    match side {
        #[cfg(target_os = "macos")]
        "native" => {
            let native = native::Native::open();
            run(&native, t, metallib, options);
        }
        "runtime" => {
            let cache = options
                .cache
                .clone()
                .unwrap_or_else(|| scratch_default().join("cache"));
            let runtime = runtime::Runtime::open(&cache);
            run(&runtime, t, metallib, options);
            let c = runtime.device().work_counts();
            eprintln!(
                "runtime work counts: queue_submissions {} command_buffers_submitted {} \
                 pipeline_barriers {} draws {} dispatches {} variant_pipelines_made {} \
                 device_memory_allocated {} host_pointer_imports_made {}",
                c.queue_submissions,
                c.command_buffers_submitted,
                c.commands_replayed.cmd_pipeline_barrier,
                c.commands_replayed.cmd_draw,
                c.commands_replayed.cmd_dispatch,
                c.variant_pipelines_made,
                c.device_memory_allocated,
                c.host_pointer_imports_made,
            );
        }
        other => panic!("--side native|runtime, not {other}"),
    }
}

fn run<G: Gpu>(g: &G, opened: Instant, metallib: &Path, options: &Options) {
    workloads::emit(
        "meta",
        "device_open",
        "ms",
        &mut [opened.elapsed().as_secs_f64() * 1e3],
    );
    eprintln!("{}", g.label());
    let sizes = &options.sizes;
    if options.only.iter().any(|w| w == "pipeline") {
        workloads::pipeline(g, metallib, sizes);
    }
    let rest: Vec<&str> = options
        .only
        .iter()
        .map(String::as_str)
        .filter(|w| *w != "pipeline")
        .collect();
    if rest.is_empty() {
        return;
    }
    let t = Instant::now();
    let library = g.library(metallib);
    let scene = workloads::Scene::new(g, &library, sizes);
    workloads::emit(
        "meta",
        "scene_setup",
        "ms",
        &mut [t.elapsed().as_secs_f64() * 1e3],
    );
    for workload in rest {
        match workload {
            "encode" => workloads::encode(g, &scene, sizes),
            "fixed" => workloads::fixed(g, &scene, sizes),
            "latency" => workloads::latency(g, &scene, sizes),
            "gpu" => workloads::gpu(g, &scene, sizes),
            "memory" => workloads::memory(g, sizes),
            _ => unreachable!(),
        }
    }
}

fn scratch_default() -> PathBuf {
    std::env::temp_dir().join(format!("host-ab-{}", std::process::id()))
}

/// The shaders' source, beside this file.
fn source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("examples/host_ab/host_ab.metal")
}

/// Compile the source with `NONCE` into `scratch`.
fn compile(scratch: &Path, nonce: u64) -> PathBuf {
    let out = scratch.join(format!("host_ab-{nonce}.metallib"));
    let status = Command::new("xcrun")
        .args(["-sdk", "macosx", "metal", &format!("-DNONCE={nonce}"), "-o"])
        .arg(&out)
        .arg(source())
        .status()
        .expect("run xcrun metal");
    assert!(status.success(), "xcrun metal failed");
    out
}

/// One side's results: (workload, metric) to (unit, median).
type Results = BTreeMap<(String, String), (String, f64)>;

/// Run this binary as one side and collect its result lines.
fn spawn(side: &str, metallib: &Path, cache: &Path, only: &[&str], options: &Options) -> Results {
    let output = Command::new(std::env::current_exe().expect("this executable"))
        .args(["--side", side, "--only", &only.join(",")])
        .arg("--metallib")
        .arg(metallib)
        .arg("--cache")
        .arg(cache)
        .args(&options.passed)
        .stderr(Stdio::inherit())
        .output()
        .expect("run a side");
    assert!(
        output.status.success(),
        "the {side} side failed: {}",
        output.status
    );
    let mut results = Results::new();
    for line in String::from_utf8_lossy(&output.stdout).lines() {
        let fields: Vec<&str> = line.split('\t').collect();
        if fields.first() == Some(&"R") && fields.len() >= 5 {
            let median = fields[4].parse().expect("a median");
            results.insert(
                (fields[1].to_owned(), fields[2].to_owned()),
                (fields[3].to_owned(), median),
            );
        }
    }
    results
}

/// Every round's results per side: (workload, metric) to (unit, [native medians], [runtime
/// medians]).
type Table = BTreeMap<(String, String), (String, Vec<f64>, Vec<f64>)>;

fn add(table: &mut Table, side: usize, results: Results, prefix: &str) {
    for ((workload, metric), (unit, median)) in results {
        let key = (workload, format!("{prefix}{metric}"));
        let entry = table.entry(key).or_insert_with(|| (unit, vec![], vec![]));
        [&mut entry.1, &mut entry.2][side].push(median);
    }
}

fn parent(options: &Options) {
    let scratch = options.scratch.clone().unwrap_or_else(scratch_default);
    let removes = options.scratch.is_none();
    std::fs::create_dir_all(&scratch).expect("make the scratch directory");
    let sides: &[&str] = if cfg!(target_os = "macos") {
        &["native", "runtime"]
    } else {
        &["runtime"]
    };
    let rest: Vec<&str> = options
        .only
        .iter()
        .map(String::as_str)
        .filter(|w| *w != "pipeline")
        .collect();
    let mut table = Table::new();
    let metallib = match &options.metallib {
        Some(path) => path.clone(),
        None => compile(&scratch, 0),
    };
    // Below 2^24 (host_ab.metal), and new each second.
    let nonce_base = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("a clock after 1970")
        .as_secs()
        % 100_000;
    for round in 0..options.rounds {
        eprintln!("round {} of {}", round + 1, options.rounds);
        if !rest.is_empty() {
            for (s, side) in sides.iter().enumerate() {
                let results = spawn(side, &metallib, &scratch.join("cache-main"), &rest, options);
                add(&mut table, s, results, "");
            }
        }
        if options.only.iter().any(|w| w == "pipeline") {
            for (s, side) in sides.iter().enumerate() {
                let nonce = nonce_base * 100 + (round * 2 + s) as u64 + 1;
                let fresh = compile(&scratch, nonce);
                let cache = scratch.join(format!("cache-{nonce}"));
                let cold = spawn(side, &fresh, &cache, &["pipeline"], options);
                add(&mut table, s, cold, "cold_");
                let warm = spawn(side, &fresh, &cache, &["pipeline"], options);
                add(&mut table, s, warm, "warm_");
                let _ = std::fs::remove_dir_all(&cache);
                let _ = std::fs::remove_file(&fresh);
            }
        }
    }
    print_table(&table, options.rounds);
    if removes {
        let _ = std::fs::remove_dir_all(&scratch);
    }
}

fn median(values: &[f64]) -> Option<f64> {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted.get(sorted.len() / 2).copied()
}

fn print_table(table: &Table, rounds: usize) {
    println!(
        "| workload | metric | unit | native | runtime | runtime/native | native rounds | runtime rounds |"
    );
    println!("|---|---|---|---:|---:|---:|---|---|");
    let spread = |values: &[f64]| match (
        values.iter().copied().reduce(f64::min),
        values.iter().copied().reduce(f64::max),
    ) {
        (Some(low), Some(high)) => format!("{low:.3}..{high:.3} ({})", values.len()),
        _ => "-".to_owned(),
    };
    for ((workload, metric), (unit, native, runtime)) in table {
        let (n, r) = (median(native), median(runtime));
        let show = |v: Option<f64>| v.map_or("-".to_owned(), |v| format!("{v:.3}"));
        let ratio = match (n, r) {
            (Some(n), Some(r)) if n > 0.0 => format!("{:.2}x", r / n),
            _ => "-".to_owned(),
        };
        println!(
            "| {workload} | {metric} | {unit} | {} | {} | {ratio} | {} | {} |",
            show(n),
            show(r),
            spread(native),
            spread(runtime)
        );
    }
    println!("\n{rounds} rounds; each cell is the median of the rounds' medians.");
}
