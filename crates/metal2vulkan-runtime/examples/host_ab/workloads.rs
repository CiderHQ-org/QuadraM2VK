//! The workloads, written once against [`Gpu`]. Each prints its measurements as result lines
//! ([`emit`]): `R`, the workload, the metric, the unit, the median, the 10th and 90th percentiles
//! and the sample count, tab-separated.

use crate::gpu::{Format, Gpu};
use std::path::Path;
use std::sync::mpsc;
use std::time::{Duration, Instant};

/// How hard the workloads run.
#[derive(Clone, Copy, Debug)]
pub struct Sizes {
    /// Draws in a render pass, and dispatches in a compute encoder, of the encode workload.
    pub draws: usize,
    /// Timed command buffers of the encode workload, and of each phase of `fixed`.
    pub timed: usize,
    /// Command buffers of each latency phase.
    pub latency: usize,
    /// Loop iterations of the heavy shaders.
    pub heavy_iterations: u32,
    /// Timed command buffers of each heavy shader.
    pub heavy_runs: usize,
    /// Draws in the many-draws workload's frame.
    pub many_draws: usize,
}

/// One result line (module docs) of `values`, which it sorts.
pub fn emit(workload: &str, metric: &str, unit: &str, values: &mut [f64]) {
    assert!(!values.is_empty(), "{workload} {metric}: no samples");
    values.sort_by(f64::total_cmp);
    let at = |p: f64| values[((values.len() - 1) as f64 * p).round() as usize];
    println!(
        "R\t{workload}\t{metric}\t{unit}\t{:.3}\t{:.3}\t{:.3}\t{}",
        at(0.5),
        at(0.1),
        at(0.9),
        values.len()
    );
}

fn us(d: Duration) -> f64 {
    d.as_secs_f64() * 1e6
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

/// The process's CPU time so far (user and system, every thread).
pub fn process_cpu() -> Duration {
    // SAFETY: `getrusage` writes the struct it is given.
    let usage = unsafe {
        let mut usage = std::mem::zeroed::<libc::rusage>();
        assert_eq!(libc::getrusage(libc::RUSAGE_SELF, &mut usage), 0);
        usage
    };
    let tv = |t: libc::timeval| Duration::new(t.tv_sec as u64, t.tv_usec as u32 * 1000);
    tv(usage.ru_utime) + tv(usage.ru_stime)
}

/// The rectangles' buffers: [`RECTS`] rectangles each, one every [`RECT_STRIDE`] bytes.
const RECT_BUFFERS: usize = 4;
const RECTS: usize = 1024;
const RECT_STRIDE: usize = 256;
/// The small textures the draws take in turn, and their side.
const TEXTURES: usize = 16;
const TEXTURE_SIDE: usize = 64;
/// The small draws' target's side.
const TARGET_SIDE: usize = 512;
/// The heavy fragment's target's side, its texture's side, and the heavy kernel's texture's.
const HEAVY_TARGET_SIDE: usize = 2048;
const HEAVY_TEXTURE_SIDE: usize = 1024;
const KERNEL_TEXTURE_SIDE: usize = 256;
/// The heavy kernel's grid: threadgroups of 256 threads.
const HEAVY_GROUPS: usize = 4096;
/// The feature kernels (`host_ab.metal`).
const FEATURES: [&str; 9] = [
    "k_fma",
    "k_sin",
    "k_half",
    "k_constant",
    "k_device",
    "k_texture",
    "k_texture_float",
    "k_threadgroup_float",
    "k_threadgroup_half",
];
const HEAVY_THREADS: usize = 256;
/// The heavy and feature kernels' output: one `float4` per thread.
const HEAVY_OUT_BYTES: usize = HEAVY_GROUPS * HEAVY_THREADS * 16;

/// What the workloads draw and dispatch with, made once per process.
pub struct Scene<G: Gpu> {
    /// `quad_vertex` + `textured_fragment` into `BGRA8Unorm`, without and with blending.
    pub textured: [G::RenderPipeline; 2],
    /// `quad_vertex` + `heavy_fragment` into `RGBA16Float`.
    pub heavy_fragment: G::RenderPipeline,
    pub heavy_kernel: G::ComputePipeline,
    /// The feature kernels (`host_ab.metal`), by name.
    pub features: Vec<(&'static str, G::ComputePipeline)>,
    /// `tiny_kernel` and `tiny_kernel_b`.
    pub tiny: [G::ComputePipeline; 2],
    pub textures: Vec<G::Texture>,
    pub heavy_texture: G::Texture,
    pub kernel_texture: G::Texture,
    pub sampler: G::Sampler,
    pub target: G::Texture,
    pub heavy_target: G::Texture,
    pub rects: Vec<G::Buffer>,
    /// One rectangle over the whole target.
    pub full: G::Buffer,
    /// The heavy fragment's parameters, and the heavy kernel's (16 times the iterations).
    pub params: G::Buffer,
    pub kernel_params: G::Buffer,
    pub tiny_out: G::Buffer,
    pub tiny_values: G::Buffer,
    pub heavy_out: G::Buffer,
    /// The feature kernels' `src`.
    pub heavy_in: G::Buffer,
    pub fill: G::Buffer,
}

fn texels(side: usize, seed: u8) -> Vec<u8> {
    (0..side * side)
        .flat_map(|i| {
            let (x, y) = ((i % side) as u8, (i / side) as u8);
            [x.wrapping_mul(3) ^ seed, y.wrapping_mul(5), x ^ y, 255]
        })
        .collect()
}

fn floats(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// The heavy shaders' `HeavyParams`: 64 `float4`, the iterations and a scale.
pub fn params_bytes(iterations: u32) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1040);
    for i in 0..64 {
        let f = i as f32;
        bytes.extend(floats(&[
            0.97 - f * 0.001,
            1.3 + f * 0.01,
            0.25 + f * 0.002,
            0.003 * f,
        ]));
    }
    bytes.extend(iterations.to_le_bytes());
    bytes.extend(0.5f32.to_le_bytes());
    bytes.resize(1040, 0);
    bytes
}

impl<G: Gpu> Scene<G> {
    pub fn new(g: &G, library: &G::Library, sizes: &Sizes) -> Scene<G> {
        let textured = [
            g.render_pipeline(
                library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                false,
            ),
            g.render_pipeline(
                library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                true,
            ),
        ];
        let heavy_fragment = g.render_pipeline(
            library,
            "quad_vertex",
            "heavy_fragment",
            Format::Rgba16Float,
            false,
        );
        let heavy_kernel = g.compute_pipeline(library, "heavy_kernel");
        let features = FEATURES
            .iter()
            .map(|&name| (name, g.compute_pipeline(library, name)))
            .collect();
        let tiny = [
            g.compute_pipeline(library, "tiny_kernel"),
            g.compute_pipeline(library, "tiny_kernel_b"),
        ];
        let shader_texture = |side: usize, seed: u8| {
            let texture = g.texture(Format::Rgba8, side, side, false);
            g.replace_region(&texture, (side, side), &texels(side, seed), side * 4);
            texture
        };
        let textures = (0..TEXTURES)
            .map(|n| shader_texture(TEXTURE_SIDE, n as u8 * 16))
            .collect();
        // A 32 by 32 grid of small rectangles over the target, each at its own 256-byte offset
        // (a constant buffer's offset alignment).
        let rects = (0..RECT_BUFFERS)
            .map(|b| {
                let mut bytes = vec![0u8; RECTS * RECT_STRIDE];
                for r in 0..RECTS {
                    let cell = (r + b * 7) % RECTS;
                    let (x, y) = ((cell % 32) as f32, (cell / 32) as f32);
                    let at = |v: f32| v / 16.0 - 1.0;
                    let rect = floats(&[at(x), at(y), at(x + 0.5), at(y + 0.5)]);
                    bytes[r * RECT_STRIDE..r * RECT_STRIDE + 16].copy_from_slice(&rect);
                }
                g.buffer_with(&bytes)
            })
            .collect();
        Scene {
            textured,
            heavy_fragment,
            heavy_kernel,
            features,
            tiny,
            textures,
            heavy_texture: shader_texture(HEAVY_TEXTURE_SIDE, 7),
            kernel_texture: shader_texture(KERNEL_TEXTURE_SIDE, 9),
            sampler: g.sampler(true),
            target: g.texture(Format::Bgra8, TARGET_SIDE, TARGET_SIDE, true),
            heavy_target: g.texture(
                Format::Rgba16Float,
                HEAVY_TARGET_SIDE,
                HEAVY_TARGET_SIDE,
                true,
            ),
            rects,
            full: g.buffer_with(&floats(&[-1.0, -1.0, 1.0, 1.0])),
            params: g.buffer_with(&params_bytes(sizes.heavy_iterations)),
            kernel_params: g.buffer_with(&params_bytes(sizes.heavy_iterations * 16)),
            tiny_out: g.buffer(RECTS * RECT_STRIDE),
            tiny_values: g.buffer_with(&vec![1u8; RECTS * RECT_STRIDE]),
            heavy_out: g.buffer(HEAVY_OUT_BYTES),
            heavy_in: g.buffer_with(&vec![0x3c; HEAVY_OUT_BYTES]),
            fill: g.buffer(4096),
        }
    }
}

/// What the encode workload changes between draws (and dispatches).
#[derive(Clone, Copy)]
enum Churn {
    /// Nothing: every draw the same.
    Static,
    /// The vertex buffer (another buffer and offset) and the fragment texture.
    Bind,
    /// As `Bind`, and the pipeline state switched between two.
    Pipeline,
    /// As `Pipeline`, in render passes of [`PASS_DRAWS`] draws each (the pass loads the target,
    /// the first clears it): a window server's few draws a pass, each binding its own.
    Passes,
}

/// The draws of each pass of [`Churn::Passes`].
const PASS_DRAWS: usize = 3;

impl Churn {
    fn name(self) -> &'static str {
        match self {
            Churn::Static => "static",
            Churn::Bind => "bind",
            Churn::Pipeline => "pipeline",
            Churn::Passes => "passes",
        }
    }
}

/// Encode one render pass of `draws` small draws with `churn` (module docs of `main.rs`).
fn encode_draws<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    command_buffer: &G::CommandBuffer,
    draws: usize,
    churn: Churn,
) {
    let mut encoder = g.render_encoder(command_buffer, &s.target, true);
    g.set_render_pipeline(&mut encoder, &s.textured[0]);
    g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
    g.set_vertex_buffer(&mut encoder, &s.rects[0], 0, 0);
    g.set_fragment_texture(&mut encoder, &s.textures[0], 0);
    for draw in 0..draws {
        if matches!(churn, Churn::Passes) && draw > 0 && draw % PASS_DRAWS == 0 {
            g.end_render(encoder);
            encoder = g.render_encoder(command_buffer, &s.target, false);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        }
        if !matches!(churn, Churn::Static) {
            let buffer = &s.rects[(draw / 7) % RECT_BUFFERS];
            g.set_vertex_buffer(&mut encoder, buffer, (draw % RECTS) * RECT_STRIDE, 0);
            g.set_fragment_texture(&mut encoder, &s.textures[draw % TEXTURES], 0);
        }
        if matches!(churn, Churn::Pipeline | Churn::Passes) {
            g.set_render_pipeline(&mut encoder, &s.textured[draw % 2]);
        }
        g.draw_strip(&mut encoder, 4);
    }
    g.end_render(encoder);
}

/// Encode one compute encoder of `dispatches` small dispatches, changing the buffers' offsets and
/// the pipeline state at each where `churn`.
fn encode_dispatches<G: Gpu>(
    g: &G,
    s: &Scene<G>,
    command_buffer: &G::CommandBuffer,
    dispatches: usize,
    churn: bool,
) {
    let mut encoder = g.compute_encoder(command_buffer);
    g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
    g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
    g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
    for dispatch in 0..dispatches {
        if churn {
            g.set_compute_pipeline(&mut encoder, &s.tiny[dispatch % 2]);
            let offset = (dispatch % RECTS) * RECT_STRIDE;
            g.set_buffer(&mut encoder, &s.tiny_out, offset, 0);
            g.set_buffer(&mut encoder, &s.tiny_values, offset, 1);
        }
        g.dispatch(&mut encoder, 1, 64);
    }
    g.end_compute(encoder);
}

/// The CPU cost of encoding: per draw and per dispatch, on the encoding thread (wall) and in the
/// whole process (CPU time of every thread, from the command buffer's creation until its wait
/// returned, which takes in work moved off the encoding thread to commit or to other threads).
pub fn encode<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes, variants: &[String]) {
    let n = sizes.draws;
    let runs = |name: String| variants.is_empty() || variants.contains(&name);
    for churn in [Churn::Static, Churn::Bind, Churn::Pipeline, Churn::Passes] {
        if !runs(format!("draw_{}", churn.name())) {
            continue;
        }
        let (mut wall, mut cpu, mut commit, mut total) = (vec![], vec![], vec![], vec![]);
        for run in 0..sizes.timed + 3 {
            g.pool(|| {
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                let command_buffer = g.command_buffer();
                encode_draws(g, s, &command_buffer, n, churn);
                let t1 = Instant::now();
                g.commit(&command_buffer);
                let t2 = Instant::now();
                g.wait(&command_buffer);
                let t3 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    wall.push(us(t1 - t0) / n as f64);
                    cpu.push(us(cpu1 - cpu0) / n as f64);
                    commit.push(us(t2 - t1));
                    total.push(ms(t3 - t0));
                }
            });
        }
        let name = churn.name();
        emit(
            "encode",
            &format!("draw_{name}_encode"),
            "us/draw",
            &mut wall,
        );
        emit(
            "encode",
            &format!("draw_{name}_process_cpu"),
            "us/draw",
            &mut cpu,
        );
        emit(
            "encode",
            &format!("draw_{name}_commit_call"),
            "us",
            &mut commit,
        );
        emit("encode", &format!("draw_{name}_cb_total"), "ms", &mut total);
    }
    for churn in [false, true] {
        if !runs(format!(
            "dispatch_{}",
            if churn { "churn" } else { "static" }
        )) {
            continue;
        }
        let (mut wall, mut cpu, mut total) = (vec![], vec![], vec![]);
        for run in 0..sizes.timed + 3 {
            g.pool(|| {
                let cpu0 = process_cpu();
                let t0 = Instant::now();
                let command_buffer = g.command_buffer();
                encode_dispatches(g, s, &command_buffer, n, churn);
                let t1 = Instant::now();
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                let t3 = Instant::now();
                let cpu1 = process_cpu();
                if run >= 3 {
                    wall.push(us(t1 - t0) / n as f64);
                    cpu.push(us(cpu1 - cpu0) / n as f64);
                    total.push(ms(t3 - t0));
                }
            });
        }
        let name = if churn { "churn" } else { "static" };
        emit(
            "encode",
            &format!("dispatch_{name}_encode"),
            "us/dispatch",
            &mut wall,
        );
        emit(
            "encode",
            &format!("dispatch_{name}_process_cpu"),
            "us/dispatch",
            &mut cpu,
        );
        emit(
            "encode",
            &format!("dispatch_{name}_cb_total"),
            "ms",
            &mut total,
        );
    }
}

/// The fixed costs: a command buffer's creation, an encoder of one command (opened, bound, one
/// command, ended), and the commit call.
pub fn fixed<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    const ENCODERS: usize = 10;
    let (mut create, mut render, mut compute, mut blit, mut commit) =
        (vec![], vec![], vec![], vec![], vec![]);
    for run in 0..sizes.timed + 3 {
        g.pool(|| {
            let t0 = Instant::now();
            let command_buffer = g.command_buffer();
            let t1 = Instant::now();
            let mut times = [vec![], vec![], vec![]];
            for _ in 0..ENCODERS {
                let a = Instant::now();
                let mut encoder = g.render_encoder(&command_buffer, &s.target, false);
                g.set_render_pipeline(&mut encoder, &s.textured[0]);
                g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
                g.set_vertex_buffer(&mut encoder, &s.rects[0], 0, 0);
                g.set_fragment_texture(&mut encoder, &s.textures[0], 0);
                g.draw_strip(&mut encoder, 4);
                g.end_render(encoder);
                let b = Instant::now();
                let mut encoder = g.compute_encoder(&command_buffer);
                g.set_compute_pipeline(&mut encoder, &s.tiny[0]);
                g.set_buffer(&mut encoder, &s.tiny_out, 0, 0);
                g.set_buffer(&mut encoder, &s.tiny_values, 0, 1);
                g.dispatch(&mut encoder, 1, 64);
                g.end_compute(encoder);
                let c = Instant::now();
                g.fill(&command_buffer, &s.fill, 256);
                let d = Instant::now();
                times[0].push(us(b - a));
                times[1].push(us(c - b));
                times[2].push(us(d - c));
            }
            let t2 = Instant::now();
            g.commit(&command_buffer);
            let t3 = Instant::now();
            g.wait(&command_buffer);
            let _ = t2;
            if run >= 3 {
                create.push(us(t1 - t0));
                render.extend(&times[0]);
                compute.extend(&times[1]);
                blit.extend(&times[2]);
                commit.push(us(t3 - t2));
            }
        });
    }
    emit("fixed", "command_buffer_create", "us", &mut create);
    emit("fixed", "render_encoder_1draw", "us", &mut render);
    emit("fixed", "compute_encoder_1dispatch", "us", &mut compute);
    emit("fixed", "blit_encoder_1fill", "us", &mut blit);
    emit("fixed", "commit_call_30_encoders", "us", &mut commit);
}

/// What a latency command buffer holds.
#[derive(Clone, Copy)]
enum Tiny {
    Empty,
    Fill,
    Dispatch,
    Draw,
}

fn encode_tiny<G: Gpu>(g: &G, s: &Scene<G>, command_buffer: &G::CommandBuffer, tiny: Tiny) {
    match tiny {
        Tiny::Empty => {}
        Tiny::Fill => g.fill(command_buffer, &s.fill, 256),
        Tiny::Dispatch => encode_dispatches(g, s, command_buffer, 1, false),
        Tiny::Draw => encode_draws(g, s, command_buffer, 1, Churn::Static),
    }
}

/// Submit-to-completion latency of a tiny command buffer, one in flight: from the commit call
/// until `waitUntilCompleted` returns, and until a completed handler runs.
pub fn latency<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    for (name, tiny) in [
        ("empty", Tiny::Empty),
        ("fill", Tiny::Fill),
        ("dispatch", Tiny::Dispatch),
        ("draw", Tiny::Draw),
    ] {
        let (mut wait, mut commit) = (vec![], vec![]);
        // The host-clock timeline: commit to kernel start, kernel start to end, kernel end to
        // GPU start, GPU start to end, GPU end to the wait's return.
        let mut split: [Vec<f64>; 5] = Default::default();
        for run in 0..sizes.latency + 10 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode_tiny(g, s, &command_buffer, tiny);
                let h0 = crate::host_seconds();
                let t0 = Instant::now();
                g.commit(&command_buffer);
                let t1 = Instant::now();
                g.wait(&command_buffer);
                let t2 = Instant::now();
                let h2 = crate::host_seconds();
                if run >= 10 {
                    commit.push(us(t1 - t0));
                    wait.push(us(t2 - t0));
                    let [ks, ke, gs, ge] = g.times(&command_buffer);
                    let points = [h0, ks, ke, gs, ge, h2];
                    for (k, part) in split.iter_mut().enumerate() {
                        part.push((points[k + 1] - points[k]) * 1e6);
                    }
                }
            });
        }
        for (part, values) in [
            "commit_to_kernel_start",
            "kernel_start_to_end",
            "kernel_end_to_gpu_start",
            "gpu_start_to_end",
            "gpu_end_to_wait_return",
        ]
        .iter()
        .zip(&mut split)
        {
            emit("latency_split", &format!("{name}_{part}"), "us", values);
        }
        emit(
            "latency",
            &format!("{name}_commit_to_wait_return"),
            "us",
            &mut wait,
        );
        emit("latency", &format!("{name}_commit_call"), "us", &mut commit);
        let mut handler = vec![];
        for run in 0..sizes.latency + 10 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode_tiny(g, s, &command_buffer, tiny);
                let (send, receive) = mpsc::channel();
                g.on_completed(
                    &command_buffer,
                    Box::new(move || {
                        let _ = send.send(Instant::now());
                    }),
                );
                let t0 = Instant::now();
                g.commit(&command_buffer);
                let at = receive.recv().expect("the completed handler ran");
                if run >= 10 {
                    handler.push(us(at - t0));
                }
            });
        }
        emit(
            "latency",
            &format!("{name}_commit_to_handler"),
            "us",
            &mut handler,
        );
    }
}

/// GPU time of the heavy shaders and the feature kernels, one command buffer in flight: from the
/// commit until the wait returned, and `GPUEndTime - GPUStartTime` where the side measures it on
/// the GPU. With `outputs`, each kernel's output after its last run goes to `<outputs>/<name>.bin`.
pub fn gpu<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes, outputs: Option<&Path>) {
    let run = |name: &str, encode: &dyn Fn(&G::CommandBuffer)| {
        let (mut wall, mut times) = (vec![], vec![]);
        for run in 0..sizes.heavy_runs + 3 {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                encode(&command_buffer);
                let t0 = Instant::now();
                g.commit(&command_buffer);
                g.wait(&command_buffer);
                let t1 = Instant::now();
                if run >= 3 {
                    wall.push(ms(t1 - t0));
                    if let Some(seconds) = g.gpu_seconds(&command_buffer) {
                        times.push(seconds * 1e3);
                    }
                }
            });
        }
        emit("gpu", &format!("{name}_commit_to_wait"), "ms", &mut wall);
        if !times.is_empty() {
            emit("gpu", &format!("{name}_gpu_time"), "ms", &mut times);
        }
    };
    let keep = |name: &str| {
        if let Some(dir) = outputs {
            let mut bytes = vec![0u8; HEAVY_OUT_BYTES];
            g.read_buffer(&s.heavy_out, &mut bytes);
            std::fs::write(dir.join(format!("{name}.bin")), bytes).expect("write an output");
        }
    };
    run("heavy_fragment", &|command_buffer| {
        let mut encoder = g.render_encoder(command_buffer, &s.heavy_target, true);
        g.set_render_pipeline(&mut encoder, &s.heavy_fragment);
        g.set_vertex_buffer(&mut encoder, &s.full, 0, 0);
        g.set_fragment_buffer(&mut encoder, &s.params, 0, 0);
        g.set_fragment_texture(&mut encoder, &s.heavy_texture, 0);
        g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        g.draw_strip(&mut encoder, 4);
        g.end_render(encoder);
    });
    run("heavy_kernel", &|command_buffer| {
        let mut encoder = g.compute_encoder(command_buffer);
        g.set_compute_pipeline(&mut encoder, &s.heavy_kernel);
        g.set_buffer(&mut encoder, &s.heavy_out, 0, 0);
        g.set_buffer(&mut encoder, &s.kernel_params, 0, 1);
        g.set_texture(&mut encoder, &s.kernel_texture, 0);
        g.dispatch(&mut encoder, HEAVY_GROUPS, HEAVY_THREADS);
        g.end_compute(encoder);
    });
    keep("heavy_kernel");
    for (name, state) in &s.features {
        run(name, &|command_buffer| {
            let mut encoder = g.compute_encoder(command_buffer);
            g.set_compute_pipeline(&mut encoder, state);
            g.set_buffer(&mut encoder, &s.heavy_out, 0, 0);
            g.set_buffer(&mut encoder, &s.kernel_params, 0, 1);
            g.set_buffer(&mut encoder, &s.heavy_in, 0, 2);
            g.set_texture(&mut encoder, &s.kernel_texture, 0);
            g.dispatch(&mut encoder, HEAVY_GROUPS, HEAVY_THREADS);
            g.end_compute(encoder);
        });
        keep(name);
    }
}

/// Pipeline creation, in a process whose caches are as the caller left them: the library, each
/// state's creation, its first use (a command buffer of one draw or dispatch, to completion), and
/// each state made again in the same process. Single samples.
pub fn pipeline<G: Gpu>(g: &G, metallib: &std::path::Path, sizes: &Sizes) {
    let t = Instant::now();
    let library = g.library(metallib);
    emit("pipeline", "library", "ms", &mut [ms(t.elapsed())]);
    let target = g.texture(Format::Bgra8, TARGET_SIDE, TARGET_SIDE, true);
    let heavy_target = g.texture(Format::Rgba16Float, 256, 256, true);
    let side = KERNEL_TEXTURE_SIDE;
    let texture = g.texture(Format::Rgba8, side, side, false);
    g.replace_region(&texture, (side, side), &texels(side, 1), side * 4);
    let sampler = g.sampler(true);
    let rect = g.buffer_with(&floats(&[-1.0, -1.0, 1.0, 1.0]));
    let params = g.buffer_with(&params_bytes(sizes.heavy_iterations.min(4)));
    let out = g.buffer(HEAVY_THREADS * 16);
    let values = g.buffer(HEAVY_THREADS * 4);
    for pass in ["first", "again"] {
        let timed = |name: &str, f: &mut dyn FnMut()| {
            let t = Instant::now();
            f();
            emit(
                "pipeline",
                &format!("{name}_{pass}"),
                "ms",
                &mut [ms(t.elapsed())],
            );
        };
        let draw = |state: &G::RenderPipeline, target: &G::Texture, uniforms: bool| {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                let mut encoder = g.render_encoder(&command_buffer, target, true);
                g.set_render_pipeline(&mut encoder, state);
                g.set_vertex_buffer(&mut encoder, &rect, 0, 0);
                if uniforms {
                    g.set_fragment_buffer(&mut encoder, &params, 0, 0);
                }
                g.set_fragment_texture(&mut encoder, &texture, 0);
                g.set_fragment_sampler(&mut encoder, &sampler, 0);
                g.draw_strip(&mut encoder, 4);
                g.end_render(encoder);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
            })
        };
        let dispatch = |state: &G::ComputePipeline, heavy: bool| {
            g.pool(|| {
                let command_buffer = g.command_buffer();
                let mut encoder = g.compute_encoder(&command_buffer);
                g.set_compute_pipeline(&mut encoder, state);
                g.set_buffer(&mut encoder, &out, 0, 0);
                if heavy {
                    g.set_buffer(&mut encoder, &params, 0, 1);
                    g.set_texture(&mut encoder, &texture, 0);
                } else {
                    g.set_buffer(&mut encoder, &values, 0, 1);
                }
                g.dispatch(&mut encoder, 1, HEAVY_THREADS);
                g.end_compute(encoder);
                g.commit(&command_buffer);
                g.wait(&command_buffer);
            })
        };
        let mut state = None;
        timed("textured_create", &mut || {
            state = Some(g.render_pipeline(
                &library,
                "quad_vertex",
                "textured_fragment",
                Format::Bgra8,
                false,
            ))
        });
        let textured = state.take().unwrap();
        timed("textured_first_draw", &mut || {
            draw(&textured, &target, false)
        });
        timed("heavy_fragment_create", &mut || {
            state = Some(g.render_pipeline(
                &library,
                "quad_vertex",
                "heavy_fragment",
                Format::Rgba16Float,
                false,
            ))
        });
        let heavy = state.take().unwrap();
        timed("heavy_fragment_first_draw", &mut || {
            draw(&heavy, &heavy_target, true)
        });
        let mut kernel = None;
        timed("heavy_kernel_create", &mut || {
            kernel = Some(g.compute_pipeline(&library, "heavy_kernel"))
        });
        let heavy_kernel = kernel.take().unwrap();
        timed("heavy_kernel_first_dispatch", &mut || {
            dispatch(&heavy_kernel, true)
        });
        timed("tiny_kernel_create", &mut || {
            kernel = Some(g.compute_pipeline(&library, "tiny_kernel"))
        });
        let tiny = kernel.take().unwrap();
        timed("tiny_kernel_first_dispatch", &mut || dispatch(&tiny, false));
    }
}

/// Memory: buffer creation (`Shared`, by length, by bytes, and over host memory without a copy)
/// and a `Shared` texture's `replaceRegion` and `getBytes`.
pub fn memory<G: Gpu>(g: &G, sizes: &Sizes) {
    const KIB: usize = 1024;
    const MIB: usize = 1024 * 1024;
    let repeats = |length: usize| {
        if length >= 16 * MIB {
            8
        } else {
            sizes.timed.max(20)
        }
    };
    for (name, length) in [("4k", 4 * KIB), ("1m", MIB), ("64m", 64 * MIB)] {
        let mut times = vec![];
        for _ in 0..repeats(length) {
            let t = Instant::now();
            let buffer = g.buffer(length);
            times.push(us(t.elapsed()));
            drop(buffer);
        }
        emit("memory", &format!("new_buffer_{name}"), "us", &mut times);
        let bytes = vec![0x33u8; length];
        let mut times = vec![];
        for _ in 0..repeats(length) {
            let t = Instant::now();
            let buffer = g.buffer_with(&bytes);
            times.push(us(t.elapsed()));
            drop(buffer);
        }
        emit(
            "memory",
            &format!("new_buffer_with_bytes_{name}"),
            "us",
            &mut times,
        );
    }
    for (name, length) in [("1m", MIB), ("64m", 64 * MIB)] {
        let mut times = vec![];
        let mut maps = vec![];
        for _ in 0..repeats(length) {
            // SAFETY: an anonymous private mapping of `length` bytes, unmapped below once every
            // buffer over it is gone.
            let pointer = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    length,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_ANON | libc::MAP_PRIVATE,
                    -1,
                    0,
                )
            };
            assert_ne!(pointer, libc::MAP_FAILED);
            // SAFETY: the pages are mapped read-write; touch them as a program's own data is.
            unsafe { std::ptr::write_bytes(pointer as *mut u8, 0x11, length) };
            let t = Instant::now();
            // SAFETY: page-aligned read-write memory that outlives the buffer (unmapped below).
            let buffer = unsafe { g.buffer_no_copy(pointer as *mut u8, length) };
            times.push(us(t.elapsed()));
            drop(buffer);
            maps.push(pointer);
        }
        emit(
            "memory",
            &format!("new_buffer_no_copy_{name}"),
            "us",
            &mut times,
        );
        // Let any release the side defers finish before the pages go.
        let command_buffer = g.command_buffer();
        g.commit(&command_buffer);
        g.wait(&command_buffer);
        std::thread::sleep(Duration::from_millis(50));
        for pointer in maps {
            // SAFETY: mapped above, no buffer over it remains.
            unsafe { libc::munmap(pointer, length) };
        }
    }
    let side = HEAVY_TEXTURE_SIDE;
    let texture = g.texture(Format::Rgba8, side, side, false);
    let bytes = texels(side, 3);
    let mut back = vec![0u8; bytes.len()];
    let (mut replace, mut get) = (vec![], vec![]);
    for run in 0..sizes.timed.max(20) + 2 {
        let t0 = Instant::now();
        g.replace_region(&texture, (side, side), &bytes, side * 4);
        let t1 = Instant::now();
        g.get_bytes(&texture, (side, side), &mut back, side * 4);
        let t2 = Instant::now();
        if run >= 2 {
            replace.push(us(t1 - t0));
            get.push(us(t2 - t1));
        }
    }
    assert!(back == bytes, "getBytes read back what replaceRegion wrote");
    emit("memory", "texture_replace_region_4m", "us", &mut replace);
    emit("memory", "texture_get_bytes_4m", "us", &mut get);
}

/// The `frame` workload's textures, their side, and the bytes between their offsets.
const FRAME_TEXTURES: usize = 8;
const FRAME_SIDE: usize = 256;
const FRAME_STRIDE: usize = 1 << 20;

/// One frame of the `frame` workload (module docs of `main.rs`) in `heap`: the textures made,
/// each rendered by a pass of three draws, each sampled into the target, committed, waited for
/// and dropped; the times of each step into `times` (`[encode, make, drop, process CPU]`, the CPU
/// per frame).
fn frame_once<G: Gpu>(g: &G, s: &Scene<G>, heap: &G::Heap, times: &mut [Vec<f64>; 4]) {
    g.pool(|| {
        let cpu0 = process_cpu();
        let t0 = Instant::now();
        let textures: Vec<G::Texture> = (0..FRAME_TEXTURES)
            .map(|i| g.heap_texture(heap, FRAME_SIDE, FRAME_SIDE, i * FRAME_STRIDE))
            .collect();
        let t1 = Instant::now();
        let command_buffer = g.command_buffer();
        for (i, texture) in textures.iter().enumerate() {
            let mut encoder = g.render_encoder(&command_buffer, texture, true);
            g.set_render_pipeline(&mut encoder, &s.textured[0]);
            g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
            for draw in 0..PASS_DRAWS {
                let rect = (i * PASS_DRAWS + draw) % RECTS;
                g.set_vertex_buffer(&mut encoder, &s.rects[0], rect * RECT_STRIDE, 0);
                g.set_fragment_texture(&mut encoder, &s.textures[(i + draw) % TEXTURES], 0);
                g.draw_strip(&mut encoder, 4);
            }
            g.end_render(encoder);
        }
        let mut encoder = g.render_encoder(&command_buffer, &s.target, true);
        g.set_render_pipeline(&mut encoder, &s.textured[1]);
        g.set_fragment_sampler(&mut encoder, &s.sampler, 0);
        for (i, texture) in textures.iter().enumerate() {
            g.set_vertex_buffer(&mut encoder, &s.rects[1], i * RECT_STRIDE, 0);
            g.set_fragment_texture(&mut encoder, texture, 0);
            g.draw_strip(&mut encoder, 4);
        }
        g.end_render(encoder);
        g.commit(&command_buffer);
        let t2 = Instant::now();
        g.wait(&command_buffer);
        let t3 = Instant::now();
        drop(textures);
        let t4 = Instant::now();
        let cpu1 = process_cpu();
        times[0].push(us(t2 - t0));
        times[1].push(us(t1 - t0));
        times[2].push(us(t4 - t3));
        times[3].push(us(cpu1 - cpu0));
    });
}

/// A window server's frame of heap textures (module docs of `main.rs`), in a heap of the device,
/// and the runtime's over host pages.
pub fn frame<G: Gpu>(g: &G, s: &Scene<G>, sizes: &Sizes) {
    let size = FRAME_TEXTURES * FRAME_STRIDE;
    let run = |name: &str, heap: &G::Heap| {
        let mut times: [Vec<f64>; 4] = Default::default();
        for run in 0..sizes.timed + 3 {
            let mut taken: [Vec<f64>; 4] = Default::default();
            frame_once(g, s, heap, &mut taken);
            if run >= 3 {
                for (all, one) in times.iter_mut().zip(taken) {
                    all.extend(one);
                }
            }
        }
        let [encode, make, dropped, cpu] = &mut times;
        emit("frame", &format!("{name}_encode"), "us", encode);
        emit("frame", &format!("{name}_make_textures"), "us", make);
        emit("frame", &format!("{name}_drop_textures"), "us", dropped);
        emit("frame", &format!("{name}_process_cpu"), "us", cpu);
    };
    run("frame", &g.heap(size));
    // SAFETY: an anonymous private mapping of `size` bytes, never unmapped: the process's to its
    // end, past the heap and every texture placed in it.
    let pointer = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            size,
            libc::PROT_READ | libc::PROT_WRITE,
            libc::MAP_ANON | libc::MAP_PRIVATE,
            -1,
            0,
        )
    };
    assert_ne!(pointer, libc::MAP_FAILED);
    // SAFETY: page-aligned read-write memory of `size` bytes that outlives the heap.
    if let Some(heap) = unsafe { g.heap_no_copy(pointer as *mut u8, size) } {
        run("frame_no_copy", &heap);
    }
}

/// The many-draws workload's target (1080p `BGRA8Unorm`, with a `Depth32Float` attachment), its
/// textures (16 `RGBA8Unorm` of 128x128) and its uniforms' stride.
const MANY_TARGET: (usize, usize) = (1920, 1080);
const MANY_TEXTURES: usize = 16;
const MANY_TEXTURE_SIDE: usize = 128;
const MANY_STRIDE: usize = 256;

/// What the many-draws workload draws with (module docs of `main.rs`).
struct ManyDraws<G: Gpu> {
    /// `draws_vertex` with each of `draws_fragment0` to `3`, the third blending.
    pipelines: [G::RenderPipeline; 4],
    /// `Less` with depth writes, `LessEqual` without.
    depth_states: [G::DepthStencil; 2],
    /// Linear and nearest.
    samplers: [G::Sampler; 2],
    textures: Vec<G::Texture>,
    /// The rectangle's corners, and its two triangles' indices.
    quad: G::Buffer,
    indices: G::Buffer,
    /// Four uniform buffers: draw `i` reads buffer `i % 4` at `(i / 4) * MANY_STRIDE`.
    uniforms: [G::Buffer; 4],
    target: G::Texture,
    depth: G::Texture,
}

impl<G: Gpu> ManyDraws<G> {
    fn new(g: &G, library: &G::Library, count: usize) -> ManyDraws<G> {
        let pipeline = |i: usize| {
            g.depth_pipeline(
                library,
                "draws_vertex",
                &format!("draws_fragment{i}"),
                i == 2,
            )
        };
        let textures = (0..MANY_TEXTURES)
            .map(|n| {
                let side = MANY_TEXTURE_SIDE;
                let texture = g.texture(Format::Rgba8, side, side, false);
                g.replace_region(
                    &texture,
                    (side, side),
                    &texels(side, n as u8 * 13 + 1),
                    side * 4,
                );
                texture
            })
            .collect();
        // A grid of rectangles over the target, each half again a cell's side, as m2vbench lays
        // them.
        let grid = (count as f64).sqrt().ceil() as usize;
        let cell = 2.0 / grid as f32;
        let per = count.div_ceil(4) * MANY_STRIDE;
        let mut bytes = [(); 4].map(|_| vec![0u8; per]);
        for i in 0..count {
            let (x, y) = ((i % grid) as f32, (i / grid) as f32);
            let at = (i / 4) * MANY_STRIDE;
            let uniforms = floats(&[
                -1.0 + x * cell,
                -1.0 + y * cell,
                cell * 1.5,
                cell * 1.5,
                0.5 + 0.5 * (i as f32 * 0.1).sin(),
                0.5 + 0.5 * (i as f32 * 0.07).cos(),
                (i % 7) as f32 / 7.0,
                1.0,
                i as f32 / count as f32,
                i as f32 * 0.001,
                0.0,
                0.0,
            ]);
            bytes[i % 4][at..at + uniforms.len()].copy_from_slice(&uniforms);
        }
        let indices: Vec<u8> = [0u16, 1, 2, 2, 1, 3]
            .iter()
            .flat_map(|i| i.to_le_bytes())
            .collect();
        ManyDraws {
            pipelines: [pipeline(0), pipeline(1), pipeline(2), pipeline(3)],
            depth_states: [g.depth_stencil(true), g.depth_stencil(false)],
            samplers: [g.sampler(true), g.sampler(false)],
            textures,
            quad: g.buffer_with(&floats(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0])),
            indices: g.buffer_with(&indices),
            uniforms: bytes.map(|bytes| g.buffer_with(&bytes)),
            target: g.texture(Format::Bgra8, MANY_TARGET.0, MANY_TARGET.1, true),
            depth: g.texture(Format::Depth32Float, MANY_TARGET.0, MANY_TARGET.1, true),
        }
    }

    /// Encode one frame's render pass of `count` draws into `command_buffer`, as m2vbench's draws
    /// mode does.
    fn encode(&self, g: &G, command_buffer: &G::CommandBuffer, count: usize) {
        let mut encoder = g.depth_render_encoder(command_buffer, &self.target, &self.depth);
        g.set_vertex_buffer(&mut encoder, &self.quad, 0, 0);
        g.set_counter_clockwise(&mut encoder);
        for i in 0..count {
            if i % 8 == 0 {
                g.set_render_pipeline(&mut encoder, &self.pipelines[(i / 8) % 4]);
            }
            if i % 16 == 0 {
                g.set_depth_stencil(&mut encoder, &self.depth_states[(i / 16) % 2]);
            }
            if i % 32 == 0 {
                g.set_fragment_sampler(&mut encoder, &self.samplers[(i / 32) % 2], 0);
            }
            if i % 64 == 0 {
                g.set_cull_back(&mut encoder, (i / 64) % 2 == 0);
            }
            let uniforms = &self.uniforms[i % 4];
            let at = (i / 4) * MANY_STRIDE;
            g.set_vertex_buffer(&mut encoder, uniforms, at, 1);
            g.set_fragment_buffer(&mut encoder, uniforms, at, 0);
            g.set_fragment_texture(&mut encoder, &self.textures[i % MANY_TEXTURES], 0);
            g.set_fragment_texture(&mut encoder, &self.textures[(i * 7 + 3) % MANY_TEXTURES], 1);
            let mut jitter = [0u8; 16];
            jitter[..4].copy_from_slice(&(0.002 * (i % 3) as f32).to_le_bytes());
            g.set_vertex_bytes(&mut encoder, &jitter, 2);
            if i % 2 == 1 {
                g.draw_indexed(&mut encoder, 6, &self.indices);
            } else {
                g.draw_strip(&mut encoder, 4);
            }
        }
        g.end_render(encoder);
    }
}

/// The many-draws workload (module docs of `main.rs`): per frame, the encoding thread's wall
/// time from the command buffer's creation through `endEncoding`, the commit call's, the commit's
/// return to the wait's, the whole frame's, the process's CPU time, and the GPU time where the
/// side measures it.
pub fn many_draws<G: Gpu>(g: &G, library: &G::Library, sizes: &Sizes) {
    let count = sizes.many_draws;
    let scene = ManyDraws::new(g, library, count);
    let mut times: [Vec<f64>; 6] = Default::default();
    for frame in 0..sizes.timed + 3 {
        g.pool(|| {
            let cpu = process_cpu();
            let start = Instant::now();
            let command_buffer = g.command_buffer();
            scene.encode(g, &command_buffer, count);
            let encoded = Instant::now();
            g.commit(&command_buffer);
            let committed = Instant::now();
            g.wait(&command_buffer);
            let done = Instant::now();
            let cpu = process_cpu() - cpu;
            if frame < 3 {
                return;
            }
            times[0].push(ms(encoded - start));
            times[1].push(ms(committed - encoded));
            times[2].push(ms(done - committed));
            times[3].push(ms(done - start));
            times[4].push(ms(cpu));
            if let Some(gpu) = g.gpu_seconds(&command_buffer) {
                times[5].push(gpu * 1e3);
            }
        });
    }
    let [encode, commit, wait, frame, cpu, gpu] = &mut times;
    emit("many_draws", "encode", "ms", encode);
    emit("many_draws", "commit", "ms", commit);
    emit("many_draws", "commit_to_done", "ms", wait);
    emit("many_draws", "frame", "ms", frame);
    emit("many_draws", "cpu", "ms", cpu);
    if !gpu.is_empty() {
        emit("many_draws", "gpu", "ms", gpu);
    }
}
