//! Pipeline state creation times, four ways: cold (an empty shader cache), from the device's memo
//! (the same device again), from the shader cache on disk (a new device over the first one's
//! directory), and from a binary archive (a new device over an empty directory, the archive
//! loaded). Ignored: a timing run takes it in release mode on each venue, with the drivers' own
//! disk caches off, and records the lines it prints.
//!
//! The pipelines are the committed public kernel cases the runtime runs and the public render
//! cases it draws. Only the state's creation is timed, each function made beforehand: the
//! translation (or its lookup), the shader modules and the layouts. The `VkPipeline` is made at
//! a pipeline's first dispatch or draw, apart from these times. The devices have no validation
//! layer. Every pipeline is added to the archive.

use super::renders::{descriptor_of, render_cases, PublicRender};
use super::{
    commit_case, completed, compute_descriptor, needs_later_steps, pipeline_state, public_case,
    public_case_names, Messages, PublicCase, ScratchCache,
};
use crate::air_library::AirLibrary;
use crate::candidate::runtime_case::{open_device, validated_options, CaseResources};
use crate::gpu::TEST_DEVICE;
use metal2vulkan::bind::Dispatch;
use metal2vulkan::pipeline::{
    ComputePipelineDescriptor, LinkedFunctions, RenderPipelineDescriptor,
};
use metal2vulkan_runtime::{
    BinaryArchive, BinaryArchiveDescriptor, ComputePipelineState, Device, PipelineOptions,
    ShaderCacheStatistics,
};
use std::path::Path;
use std::time::{Duration, Instant};

/// A device over the shader cache in `cache`, without the validation layer.
fn timing_device(cache: &Path) -> Device {
    let mut options = validated_options(&Messages::default(), cache);
    options.layers.clear();
    open_device(TEST_DEVICE, options).unwrap()
}

/// A case whose pipeline is timed.
enum Case {
    Kernel(Box<PublicCase>),
    Render(Box<PublicRender>),
}

/// A case's pipeline on one device, its functions made.
enum Pipeline {
    Kernel(Box<ComputePipelineDescriptor>),
    Render(Box<RenderPipelineDescriptor>),
}

impl Case {
    /// The pipeline of the case on `device`, if its functions can be made there.
    fn on(&self, device: &Device) -> Option<Pipeline> {
        match self {
            Case::Kernel(case) => device
                .new_library_with_air(&case.bitcode)
                .and_then(|library| {
                    library.new_function_with_name_constant_values(
                        &case.compute.entry,
                        &case.compute.constants,
                    )
                })
                .ok()
                .map(|function| {
                    Pipeline::Kernel(Box::new(compute_descriptor(
                        &function,
                        LinkedFunctions::new(),
                        &case.compute,
                    )))
                }),
            Case::Render(case) => descriptor_of(device, case)
                .ok()
                .map(|descriptor| Pipeline::Render(Box::new(descriptor))),
        }
    }

    fn is_render(&self) -> bool {
        matches!(self, Case::Render(_))
    }
}

impl Pipeline {
    /// The time `device` takes to create the pipeline under `options`, or `None` when it
    /// refuses.
    fn create(&self, device: &Device, options: &PipelineOptions) -> Option<Duration> {
        let start = Instant::now();
        let made = match self {
            Pipeline::Kernel(descriptor) => device
                .new_compute_pipeline_state(descriptor, options)
                .is_ok(),
            Pipeline::Render(descriptor) => device
                .new_render_pipeline_state(descriptor, options)
                .is_ok(),
        };
        made.then(|| start.elapsed())
    }
}

/// Every public kernel and render case whose pipeline a device of the venue creates, by name.
fn cases(device: &Device) -> Vec<(String, Case)> {
    let mut out = Vec::new();
    for name in public_case_names().unwrap() {
        if let Ok(case) = public_case(&name) {
            if !needs_later_steps(&case.compute) {
                out.push((name, Case::Kernel(Box::new(case))));
            }
        }
    }
    for (name, case) in render_cases() {
        if let Ok(case) = case {
            out.push((name, Case::Render(Box::new(case))));
        }
    }
    let defaults = PipelineOptions::default();
    out.retain(|(_, case)| {
        case.on(device)
            .and_then(|pipeline| pipeline.create(device, &defaults))
            .is_some()
    });
    out
}

/// One way's creation times, by case.
#[derive(Default)]
struct Times(Vec<(Duration, String, bool)>);

impl Times {
    /// Create `case`'s pipeline on `device` under `options` and keep the time; the pipeline.
    fn time(
        &mut self,
        device: &Device,
        name: &str,
        case: &Case,
        options: &PipelineOptions,
    ) -> Pipeline {
        let pipeline = case.on(device).unwrap();
        let time = pipeline
            .create(device, options)
            .unwrap_or_else(|| panic!("{name}: the creation was refused"));
        self.0.push((time, name.to_string(), case.is_render()));
        pipeline
    }

    /// Print the times of the renders (`render`) or kernels: the count, median, 90th
    /// percentile, sum and slowest.
    fn report(&self, way: &str, render: bool) {
        let mut times: Vec<&(Duration, String, bool)> =
            self.0.iter().filter(|(_, _, r)| *r == render).collect();
        times.sort_by_key(|(time, _, _)| *time);
        let kind = if render { "renders" } else { "kernels" };
        let Some((slowest, name, _)) = times.last() else {
            println!("CREATION {kind} {way}: none");
            return;
        };
        let at = |fraction: f64| times[((times.len() - 1) as f64 * fraction).round() as usize].0;
        let sum: Duration = times.iter().map(|(time, _, _)| *time).sum();
        println!(
            "CREATION {kind} {way}: {} pipelines, median {} us, p90 {} us, sum {} ms, slowest {} \
             us ({name})",
            times.len(),
            at(0.5).as_micros(),
            at(0.9).as_micros(),
            sum.as_millis(),
            slowest.as_micros()
        );
    }
}

fn lookups(s: &ShaderCacheStatistics) -> (u64, u64, u64) {
    (s.hits, s.misses, s.writes)
}

#[test]
#[ignore = "a release-mode measurement: a timing run takes it per venue"]
fn creation_times_cold_from_the_memo_the_disk_and_an_archive() {
    let (listing, cache, empty) = (
        ScratchCache::new().unwrap(),
        ScratchCache::new().unwrap(),
        ScratchCache::new().unwrap(),
    );
    let files = ScratchCache::new().unwrap();
    std::fs::create_dir_all(files.path()).unwrap();
    let file = files.path().join("pipelines.archive");
    let defaults = PipelineOptions::default();
    // Listed on a device of its own, so each creation timed below is its pipeline's first on
    // its device and in its cache directory.
    let cases = cases(&timing_device(listing.path()));

    let device = timing_device(cache.path());
    let archive = device
        .new_binary_archive(&BinaryArchiveDescriptor::default())
        .unwrap();
    let (mut cold, mut memo) = (Times::default(), Times::default());
    for (name, case) in &cases {
        cold.time(&device, name, case, &defaults);
        match memo.time(&device, name, case, &defaults) {
            Pipeline::Kernel(descriptor) => archive.add_compute_pipeline_functions(&descriptor),
            Pipeline::Render(descriptor) => archive.add_render_pipeline_functions(&descriptor),
        }
        .unwrap();
    }
    archive.serialize_to_url(&file).unwrap();
    drop((archive, device));

    let device = timing_device(cache.path());
    let mut disk = Times::default();
    for (name, case) in &cases {
        disk.time(&device, name, case, &defaults);
    }
    let disk_lookups = lookups(&device.shader_cache_statistics());
    drop(device);

    let device = timing_device(empty.path());
    let loaded: BinaryArchive = device
        .new_binary_archive(&BinaryArchiveDescriptor::new(Some(file)))
        .unwrap();
    let mut from_archive = PipelineOptions::default();
    from_archive.binary_archives = vec![loaded];
    let mut archived = Times::default();
    for (name, case) in &cases {
        archived.time(&device, name, case, &from_archive);
    }
    let archive_lookups = lookups(&device.shader_cache_statistics());
    drop(device);

    for render in [false, true] {
        cold.report("cold", render);
        memo.report("memo", render);
        disk.report("disk", render);
        archived.report("archive", render);
    }
    println!(
        "CREATION lookups (hits, misses, writes): disk {disk_lookups:?}, archive {archive_lookups:?}"
    );
    // A creation that takes stand-ins misses its first attempt, which compiles nothing.
    assert_eq!(disk_lookups.2, 0, "the disk-warm device compiled nothing");
    assert_eq!(archive_lookups.2, 0, "the archive gave every pipeline");
}

/// How long a kernel's first dispatch waits for its `VkPipeline`, where Metal compiled at the
/// state's creation: per public kernel case, cold (an empty shader cache, the drivers' own disk
/// caches off), the state's creation, its first run (one command buffer of the case's one
/// dispatch) and its second (the pipeline made), each timed on the caller's thread to its commit
/// (`encode`: the resources, the encoding and the commit) and to its completion (`run`). Previously
/// the encoding made the region's pipeline on the caller's thread; since, the device's pool makes
/// it and the command buffer waits for it. Prints `FIRST-DISPATCH-STALL` lines, the cases with the
/// largest caller's stall (first encode less second), `FIRST-DISPATCH` lines, those with the
/// largest first-run excess (first run less second), and the count, median and sum of the stall and
/// of the excess.
///
/// (the trace of the 26-32 ms first encodes once seen on the RTX 3090.) Each case's resources are
/// made once apart first, on the caller's thread, and dropped unused (`resources`,
/// [`timed_resources`]): its textures with their bytes, a multisampled texture's sample fill with
/// its render pipeline state (`runtime_samples.rs`), its sampler states and argument buffers. So
/// the first encode times the dispatch's own first use, with the case's helpers made, and each line
/// counts what the resources and the first run made ([`Made`]). `FIRST-ENCODE-WITH-RESOURCES` lines
/// are the cases with the largest stall as measured before (the resources' time and the stall).
///
/// The cases of the tensor
/// matmul fixture (`tns_mm_flags`, the coordinator's compute-heavy kernel) run once more at twice
/// their threadgroup's width, the same threads: another size's first run and its second
/// (`SECOND-SIZE` lines), a measurement only, whatever it computes.
#[test]
#[ignore = "a release-mode measurement: a timing run takes it per venue"]
fn first_dispatch_times() {
    let (listing, cache) = (ScratchCache::new().unwrap(), ScratchCache::new().unwrap());
    let listed = timing_device(listing.path());
    let names: Vec<String> = cases(&listed)
        .into_iter()
        .filter(|(_, case)| !case.is_render())
        .map(|(name, _)| name)
        .collect();
    drop(listed);
    let device = timing_device(cache.path());
    let mut rows = Vec::new();
    for name in &names {
        let Ok(case) = public_case(name) else {
            continue;
        };
        if case.compute.tile.is_some() {
            continue;
        }
        let start = Instant::now();
        let Ok(pipeline) = pipeline_state(&device, &case) else {
            continue;
        };
        let created = start.elapsed();
        let Some(resources) = timed_resources(&device, &case) else {
            continue;
        };
        let before = Made::now(&device);
        let Some(first) = timed_run(&device, &case, &pipeline) else {
            continue;
        };
        let first_made = Made::now(&device).since(before);
        let Some(second) = timed_run(&device, &case, &pipeline) else {
            continue;
        };
        rows.push(Row {
            name: name.clone(),
            created,
            resources,
            first: (first, first_made),
            second,
        });
        if case.compute.entry == "tns_mm_flags" {
            if let Some(timed) = second_size(&device, name, &pipeline) {
                println!("SECOND-SIZE {name}: {timed}");
            }
        }
    }
    let excess = |row: &Row| row.first.0 .1.saturating_sub(row.second.1);
    let stall = |row: &Row| row.first.0 .0.saturating_sub(row.second.0);
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    let print = |label: &str, row: &Row| {
        let Row {
            name,
            created,
            resources: (resources, resources_made),
            first: ((first_encode, first_run), first_made),
            second: (second_encode, second_run),
        } = row;
        println!(
            "{label} {name}: created {:.1} ms, resources {:.1} ms ({resources_made}), \
             first encode {:.1} ms run {:.1} ms ({first_made}), \
             second encode {:.1} ms run {:.1} ms, stall {:.1} ms, excess {:.1} ms",
            ms(*created),
            ms(*resources),
            ms(*first_encode),
            ms(*first_run),
            ms(*second_encode),
            ms(*second_run),
            ms(stall(row)),
            ms(excess(row))
        );
    };
    rows.sort_by_key(|row| std::cmp::Reverse(stall(row)));
    for row in rows.iter().take(6) {
        print("FIRST-DISPATCH-STALL", row);
    }
    // The stall as the lane measured it before the resources were made apart: those whose
    // resources took the longest on the caller's thread.
    rows.sort_by_key(|row| std::cmp::Reverse(stall(row) + row.resources.0));
    for row in rows.iter().take(6) {
        print("FIRST-ENCODE-WITH-RESOURCES", row);
    }
    rows.sort_by_key(|row| std::cmp::Reverse(excess(row)));
    for row in rows.iter().take(12) {
        print("FIRST-DISPATCH", row);
    }
    for (what, of) in [
        ("stall", &stall as &dyn Fn(&Row) -> Duration),
        ("excess", &excess),
    ] {
        let mut times: Vec<Duration> = rows.iter().map(of).collect();
        times.sort();
        if let (Some(median), Some(max)) = (times.get(times.len() / 2), times.last()) {
            println!(
                "FIRST-DISPATCH {} kernels: median {what} {:.1} ms, max {:.1} ms, sum {:.0} ms",
                times.len(),
                ms(*median),
                ms(*max),
                ms(times.iter().sum())
            );
        }
    }
    assert!(!rows.is_empty(), "no kernel case ran");
}

/// One kernel case's row.
struct Row {
    name: String,
    /// Its state's creation.
    created: Duration,
    /// Its resources made apart ([`timed_resources`]), and what that made.
    resources: (Duration, Made),
    /// Its first run (to its commit and to its completion), and what the run made.
    first: ((Duration, Duration), Made),
    /// Its second run.
    second: (Duration, Duration),
}

/// What the device made over a step: the pipeline states' creations' `VkPipeline`s (a render
/// state's predicted key, made at its creation, the first on the creating thread: in a kernel
/// case only a multisampled texture's sample fill makes one), the variants' `VkPipeline`s, the
/// dispatches' makings handed to the pool, every making through the shader cache's door
/// (`vkCreate*Pipelines`, the runtime's own kernels' included: each reads its record once
/// finished, written, unchanged or refused), and the shader cache's misses (each compiled: a
/// translation or a making).
#[derive(Clone, Copy)]
struct Made {
    creation: u64,
    variant: u64,
    pooled: u64,
    makings: u64,
    misses: u64,
}

impl Made {
    /// The device's counts now.
    fn now(device: &Device) -> Made {
        let (counts, cache) = (device.work_counts(), device.shader_cache_statistics());
        Made {
            creation: counts.creation_pipelines_made,
            variant: counts.variant_pipelines_made,
            pooled: counts.dispatch_pipelines_pooled,
            makings: cache.driver_saves
                + cache.driver_data_unchanged
                + cache.driver_data_truncated
                + cache.driver_data_mismatched,
            misses: cache.misses,
        }
    }

    /// What was made since `before`.
    fn since(self, before: Made) -> Made {
        Made {
            creation: self.creation - before.creation,
            variant: self.variant - before.variant,
            pooled: self.pooled - before.pooled,
            makings: self.makings - before.makings,
            misses: self.misses - before.misses,
        }
    }
}

impl std::fmt::Display for Made {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "creation pipelines +{}, variant pipelines +{}, dispatch makings pooled +{}, \
             makings +{}, cache misses +{}",
            self.creation, self.variant, self.pooled, self.makings, self.misses
        )
    }
}

/// `case`'s resources made on `device` on the caller's thread ([`CaseResources::of`]) and dropped
/// unused: how long that took, and what it made. `None` where it fails.
fn timed_resources(device: &Device, case: &PublicCase) -> Option<(Duration, Made)> {
    let before = Made::now(device);
    let start = Instant::now();
    let resources = CaseResources::of(
        device,
        &case.compute,
        &case.texture_usage,
        &case.argument_texture_usage,
    )
    .ok()?;
    let took = start.elapsed();
    drop(resources);
    Some((took, Made::now(device).since(before)))
}

/// `case` run once on `pipeline`, timed on the caller's thread to its commit and to its
/// completion; `None` where it fails.
fn timed_run(
    device: &Device,
    case: &PublicCase,
    pipeline: &ComputePipelineState,
) -> Option<(Duration, Duration)> {
    let start = Instant::now();
    let committed = commit_case(
        device,
        &case.compute,
        &case.texture_usage,
        &case.argument_texture_usage,
        pipeline,
        std::collections::BTreeMap::new(),
    )
    .ok()?;
    let encoded = start.elapsed();
    completed(case, committed).ok()?;
    Some((encoded, start.elapsed()))
}

/// Case `name` run on `pipeline` at twice its threadgroup's width over the same threads, twice:
/// the first run (its regions' pipelines made for it) and the second, each to its commit and to
/// its completion, as a line; `None` where the dispatch is not of threads or the width would pass
/// 1024.
fn second_size(device: &Device, name: &str, pipeline: &ComputePipelineState) -> Option<String> {
    let mut case = public_case(name).ok()?;
    let Dispatch::Threads {
        threads,
        threads_per_group: [x, y, z],
    } = case.compute.dispatch
    else {
        return None;
    };
    if 2 * x * y * z > 1024 {
        return None;
    }
    case.compute.dispatch = Dispatch::Threads {
        threads,
        threads_per_group: [2 * x, y, z],
    };
    let first = timed_run(device, &case, pipeline)?;
    let second = timed_run(device, &case, pipeline)?;
    let ms = |d: Duration| d.as_secs_f64() * 1e3;
    Some(format!(
        "{}x{y}x{z}: first encode {:.1} ms run {:.1} ms, second encode {:.1} ms run {:.1} ms",
        2 * x,
        ms(first.0),
        ms(first.1),
        ms(second.0),
        ms(second.1)
    ))
}
