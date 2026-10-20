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
//!   vertex buffer and offset, and another fragment texture), `pipeline` (as `bind`, and the
//!   pipeline state switched between two) and `passes` (as `pipeline`, in render passes of three
//!   draws, as a window server draws), and a compute encoder of as many single-threadgroup
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
//!   the fragment's loop, and 16 times it the kernel's. Then each feature kernel (`k_*`, one
//!   feature of the heavy kernel each: fma, sin, half, constant buffer, device buffer, texture
//!   read as `half` and as `float`, threadgroup float, threadgroup half) on the kernel's grid
//!   and loop. With both sides, each round also compares every kernel's output buffer after its
//!   last run between the sides, and the last table counts the 32-bit words that differ.
//! - `latency_split` (with `latency`): the same command buffers' host-clock timeline, commit to
//!   `kernelStartTime` to `kernelEndTime` to `GPUStartTime` to `GPUEndTime` to the wait's return.
//!   The runtime's GPU times on MoltenVK are the host's (its docs): its GPU start is its kernel
//!   end and its GPU end the moment it saw the command buffer complete.
//! - `pipeline`: the library, then each of four states' creation and first use, then each made
//!   again in the process. Each round runs it in two processes per side over a metallib of a
//!   fresh `NONCE` (`host_ab.metal`): `cold_*` (no cache holds its functions) and `warm_*` (a
//!   second process over the same runtime shader cache directory, after Metal's own caches took
//!   the first).
//! - `frame`: a window server's frame of heap textures: eight `BGRA8Unorm` textures of 256x256
//!   (`allowGPUOptimizedContents` off) made at their offsets of a `Placement` heap of `Shared`
//!   storage, each rendered by a pass of
//!   three textured draws, then each sampled into the target by one draw of a last pass, the
//!   command buffer committed and waited for, and the textures dropped. Per frame: the encoding
//!   thread's wall time from the first texture's creation through the commit, the textures'
//!   creation alone, their drop, and the process's CPU time until the drop returned. The
//!   runtime's alone (`frame_no_copy_*`): the same over a heap of host pages
//!   (`new_heap_with_bytes_no_copy`), as a guest's heap is.
//! - `many_draws`: a frame of `--many-draws` small draws, shaped as m2vbench's draws mode: one
//!   render pass into a 1920x1080 `BGRA8Unorm` target with a cleared `Depth32Float` attachment,
//!   each draw a textured rectangle of its own uniforms (a vertex and a fragment buffer at the
//!   draw's offset of one of four buffers), two fragment textures of sixteen, 16 bytes of
//!   `setVertexBytes`, alternately indexed and not; the pipeline state (four) every 8 draws, the
//!   depth-stencil state (two) every 16, the fragment sampler (two) every 32 and the cull mode
//!   every 64. Per frame: the encoding thread's wall time from the command buffer's creation
//!   through `endEncoding`, the commit call's, from its return to the wait's, the whole frame's,
//!   the process's CPU time, and native's GPU time.
//! - `memory`: `newBufferWithLength:` and `newBufferWithBytes:` of 4 KiB, 1 MiB and 64 MiB,
//!   `newBufferWithBytesNoCopy:` of 1 MiB and 64 MiB, and a 1024x1024 `RGBA8Unorm` `Shared`
//!   texture's `replaceRegion` and `getBytes`.
//! - `churn`: each frame makes eight 1280x800 `Private` render targets, clears and releases them,
//!   the same size each frame (`same`) or one of 64 widths in turn (`varied`); and sixteen 64-KiB
//!   buffers, filled and released (`buffers`) (`workloads::churn`).
//! - `compositing`: frames of the consumer's compositing shape, paced at 60 Hz: 20 command
//!   buffers a frame, and 10 new textures made in a `Placement` heap of tracked `Shared` storage
//!   (as the guest makes heap textures each frame): 10 command buffers each clear one new
//!   texture, made just before it, and draw 2 quads into it, 8 each draw 8 blended quads into one
//!   of two 1024x1024 layers from 4 new textures and the other layer (each depending on the
//!   command buffers before it), one dispatches 15 small kernels and fills a buffer, and the last
//!   draws both layers and 4 new textures onto a screen texture. Per frame: the wall time from the
//!   first command buffer's creation to the last one's completion (its median, its 99th
//!   percentile and the frames over 16.7 ms), the encoding thread's time to encode and commit,
//!   the process's CPU time, and on the runtime's side its work counts per frame (`rt_*`:
//!   `vkQueueSubmit` calls, the `VkSubmitInfo`s they held, each a Metal command buffer under
//!   MoltenVK, and the rest).
//!
//! Options: `--rounds N` (3), `--draws N` (2000), `--timed N` (30), `--latency N` (200),
//! `--heavy-iterations N` (64), `--heavy-runs N` (15), `--many-draws N` (6000), `--frames N`
//! (120, the compositing workload's timed frames), `--scratch DIR` (metallibs and the runtime's
//! shader caches; a directory under the OS's temporary directory by default, removed at the end),
//! `--metallib PATH` (skip the compile; not with `pipeline`), `--cache DIR` (the runtime's shader
//! cache, with `--side`), `--variants a,b,...` (the encode workload's variants alone:
//! `draw_static`, `draw_bind`, `draw_pipeline`, `draw_passes`, `dispatch_static`,
//! `dispatch_churn`; for a profile of one).
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

const WORKLOADS: [&str; 10] = [
    "encode",
    "fixed",
    "latency",
    "gpu",
    "pipeline",
    "memory",
    "frame",
    "many_draws",
    "churn",
    "compositing",
];

struct Options {
    side: Option<String>,
    only: Vec<String>,
    rounds: usize,
    sizes: Sizes,
    scratch: Option<PathBuf>,
    metallib: Option<PathBuf>,
    cache: Option<PathBuf>,
    /// Where a side writes the gpu workload's kernel outputs (`--outputs`).
    outputs: Option<PathBuf>,
    /// The encode workload's variants to run (`--variants`); all when empty.
    variants: Vec<String>,
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
            many_draws: 6000,
            frames: 120,
        },
        scratch: None,
        metallib: None,
        cache: None,
        outputs: None,
        variants: vec![],
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
            "--many-draws" => options.sizes.many_draws = number(&mut options.passed),
            "--frames" => options.sizes.frames = number(&mut options.passed),
            "--scratch" => options.scratch = Some(PathBuf::from(value())),
            "--metallib" => options.metallib = Some(PathBuf::from(value())),
            "--cache" => options.cache = Some(PathBuf::from(value())),
            "--outputs" => options.outputs = Some(PathBuf::from(value())),
            "--variants" => {
                let v = value();
                options.variants = v.split(',').map(str::to_owned).collect();
                options.passed.extend(["--variants".to_owned(), v]);
            }
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
                 device_memory_allocated {} host_pointer_imports_made {} \
                 requirement_images_made {} arena_pools_made {} owned_descriptor_pools {} \
                 draws_settled {} stage_tables_read {} graphics_pipelines_looked_up {} \
                 draws_pushing_set_zero {} set_zero_writes_pushed {} set_zero_sets_reused {} \
                 signed_depth_switch_ons {} sign_strips {} magnitude_stand_in_passes {}",
                c.queue_submissions,
                c.command_buffers_submitted,
                c.commands_replayed.cmd_pipeline_barrier,
                c.commands_replayed.cmd_draw,
                c.commands_replayed.cmd_dispatch,
                c.variant_pipelines_made,
                c.device_memory_allocated,
                c.host_pointer_imports_made,
                c.requirement_images_made,
                c.arena_pools_made,
                c.owned_descriptor_pools,
                c.draws_settled,
                c.stage_tables_read,
                c.graphics_pipelines_looked_up,
                c.draws_pushing_set_zero,
                c.set_zero_writes_pushed,
                c.set_zero_sets_reused,
                c.signed_depth_switch_ons,
                c.sign_strips,
                c.magnitude_stand_in_passes,
            );
            eprintln!("runtime commands replayed: {:?}", c.commands_replayed);
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
            "encode" => workloads::encode(g, &scene, sizes, &options.variants),
            "fixed" => workloads::fixed(g, &scene, sizes),
            "latency" => workloads::latency(g, &scene, sizes),
            "gpu" => workloads::gpu(g, &scene, sizes, options.outputs.as_deref()),
            "memory" => workloads::memory(g, sizes),
            "frame" => workloads::frame(g, &scene, sizes),
            "many_draws" => workloads::many_draws(g, &library, sizes),
            "churn" => workloads::churn(g, sizes),
            "compositing" => workloads::compositing(g, &scene, sizes),
            _ => unreachable!(),
        }
    }
}

/// The host clock Metal's and the runtime's command buffer times count: `CLOCK_UPTIME_RAW`
/// (`mach_absolute_time`) on macOS.
pub fn host_seconds() -> f64 {
    metal2vulkan_runtime::host_time()
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

/// Run this binary as one side and collect its result lines; with `outputs`, the side writes its
/// kernel outputs there.
fn spawn(
    side: &str,
    metallib: &Path,
    (cache, outputs): (&Path, Option<&Path>),
    only: &[&str],
    options: &Options,
) -> Results {
    let mut command = Command::new(std::env::current_exe().expect("this executable"));
    command
        .args(["--side", side, "--only", &only.join(",")])
        .arg("--metallib")
        .arg(metallib)
        .arg("--cache")
        .arg(cache);
    if let Some(outputs) = outputs {
        command.arg("--outputs").arg(outputs);
    }
    let output = command
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

/// Each kernel output's 32-bit words, and per round how many differ between the sides.
type Compared = BTreeMap<String, (usize, Vec<usize>)>;

/// Compare the outputs both sides wrote (`<name>.bin` in each directory), then remove them.
fn compare(dirs: &[PathBuf], compared: &mut Compared) {
    let mut names: Vec<_> = std::fs::read_dir(&dirs[0])
        .expect("read the outputs")
        .map(|entry| entry.expect("an output").file_name())
        .collect();
    names.sort();
    for name in names {
        let [a, b] = [0, 1].map(|s| std::fs::read(dirs[s].join(&name)).expect("read an output"));
        assert_eq!(a.len(), b.len(), "{name:?}: the sides' lengths");
        let differ = a.chunks(4).zip(b.chunks(4)).filter(|(x, y)| x != y).count();
        let key = name.to_string_lossy().trim_end_matches(".bin").to_owned();
        let entry = compared.entry(key).or_insert((a.len() / 4, vec![]));
        entry.1.push(differ);
    }
    for dir in dirs {
        let _ = std::fs::remove_dir_all(dir);
    }
}

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
    let mut compared = Compared::new();
    // Both sides' kernel outputs, compared each round.
    let outputs: Vec<PathBuf> = if sides.len() == 2 && rest.contains(&"gpu") {
        sides
            .iter()
            .map(|side| scratch.join(format!("outputs-{side}")))
            .collect()
    } else {
        vec![]
    };
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
                let out = outputs.get(s).map(PathBuf::as_path);
                if let Some(dir) = out {
                    std::fs::create_dir_all(dir).expect("make an outputs directory");
                }
                let cache = scratch.join("cache-main");
                let results = spawn(side, &metallib, (&cache, out), &rest, options);
                add(&mut table, s, results, "");
            }
            if !outputs.is_empty() {
                compare(&outputs, &mut compared);
            }
        }
        if options.only.iter().any(|w| w == "pipeline") {
            for (s, side) in sides.iter().enumerate() {
                let nonce = nonce_base * 100 + (round * 2 + s) as u64 + 1;
                let fresh = compile(&scratch, nonce);
                let cache = scratch.join(format!("cache-{nonce}"));
                let cold = spawn(side, &fresh, (&cache, None), &["pipeline"], options);
                add(&mut table, s, cold, "cold_");
                let warm = spawn(side, &fresh, (&cache, None), &["pipeline"], options);
                add(&mut table, s, warm, "warm_");
                let _ = std::fs::remove_dir_all(&cache);
                let _ = std::fs::remove_file(&fresh);
            }
        }
    }
    print_table(&table, options.rounds);
    print_compared(&compared);
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

fn print_compared(compared: &Compared) {
    if compared.is_empty() {
        return;
    }
    println!("\n| kernel output | 32-bit words | words differing (native vs runtime), per round |");
    println!("|---|---:|---|");
    for (name, (words, differ)) in compared {
        let rounds: Vec<String> = differ.iter().map(usize::to_string).collect();
        println!("| {name} | {words} | {} |", rounds.join(", "));
    }
}
