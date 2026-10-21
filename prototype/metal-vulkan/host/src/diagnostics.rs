//! Default-off, bounded numeric diagnostics. Snapshots take runtime locks and run at most once
//! per second on commit, plus the final device release, never on the draw/setter path.
use ash::vk::Handle;
use metal2vulkan_runtime::Device;
use serde_json::{json, Value};
use std::collections::{BTreeMap, VecDeque};
use std::io::Write;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;

const MAX_SAMPLES: usize = 512;
const MAX_TRACE: u64 = 16 << 20;
static TRACE_WRITE: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Phase {
    Library,
    RenderPipeline,
    ComputePipeline,
    Encoding,
    Acquire,
    Present,
    Commit,
    Wait,
    GpuSpan,
}

const NAMES: [&str; 9] = [
    "library",
    "render_pipeline",
    "compute_pipeline",
    "encoder_lifetime",
    "acquire",
    "present_call",
    "commit",
    "completion_wait",
    "command_buffer_gpu_span",
];

pub fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var("MACOBLOX_METAL_DIAGNOSTICS").as_deref() == Ok("1"))
}

pub fn gpu_timestamps() -> bool {
    enabled() && std::env::var("MACOBLOX_METAL_GPU_TIMESTAMPS").as_deref() == Ok("1")
}

struct Times {
    count: u64,
    nanos: u64,
    maximum: u64,
    recent: VecDeque<u64>,
}
impl Times {
    fn new() -> Self {
        Self {
            count: 0,
            nanos: 0,
            maximum: 0,
            recent: VecDeque::with_capacity(MAX_SAMPLES),
        }
    }
    fn push(&mut self, nanos: u64) {
        self.count += 1;
        self.nanos = self.nanos.saturating_add(nanos);
        self.maximum = self.maximum.max(nanos);
        if self.recent.len() == MAX_SAMPLES {
            self.recent.pop_front();
        }
        self.recent.push_back(nanos);
    }
    fn value(&self) -> Value {
        let mut recent: Vec<u64> = self.recent.iter().copied().collect();
        recent.sort_unstable();
        let percentile = |percent: usize| {
            recent
                .get((recent.len() * percent).div_ceil(100).saturating_sub(1))
                .copied()
                .unwrap_or(0)
        };
        json!({"count":self.count,"total_ns":self.nanos,"max_ns":self.maximum,"recent_samples":recent.len(),"p50_ns":percentile(50),"p95_ns":percentile(95),"p99_ns":percentile(99)})
    }
}

struct Trace {
    file: std::fs::File,
}
struct Stats {
    started: Instant,
    last_sample: AtomicU64,
    phases: [Mutex<Times>; 9],
    trace: Mutex<Option<Trace>>,
}

fn devices() -> &'static Mutex<BTreeMap<u64, Arc<Stats>>> {
    static DEVICES: OnceLock<Mutex<BTreeMap<u64, Arc<Stats>>>> = OnceLock::new();
    DEVICES.get_or_init(Mutex::default)
}
fn key(device: &Device) -> u64 {
    unsafe { device.raw() }.device.handle().as_raw()
}
fn stats(device: &Device) -> Option<Arc<Stats>> {
    devices()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .get(&key(device))
        .cloned()
}

pub fn register(device: &Device) {
    if !enabled() {
        return;
    }
    let trace = std::env::var_os("MACOBLOX_METAL_DIAGNOSTICS_FILE").and_then(|path| {
        let path = std::path::PathBuf::from(path);
        let opened = (|| {
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            let mut options = std::fs::OpenOptions::new();
            options.append(true).create(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt;
                options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
            }
            let file = options.open(path)?;
            Ok::<_, std::io::Error>(Trace { file })
        })();
        match opened {
            Ok(trace) => Some(trace),
            Err(error) => {
                crate::util::log_once(&format!(
                    "[Quadra Metal] performance trace unavailable: {error}"
                ));
                None
            }
        }
    });
    let value = Arc::new(Stats {
        started: Instant::now(),
        last_sample: AtomicU64::new(0),
        phases: std::array::from_fn(|_| Mutex::new(Times::new())),
        trace: Mutex::new(trace),
    });
    let mut devices = devices()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if devices.len() < 16 {
        devices.insert(key(device), value);
    }
}

pub struct Measurement {
    stats: Arc<Stats>,
    phase: Phase,
    started: Instant,
}
impl Drop for Measurement {
    fn drop(&mut self) {
        self.stats.phases[self.phase as usize]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(self.started.elapsed().as_nanos().min(u64::MAX as u128) as u64);
    }
}

pub fn timer(device: &Device, phase: Phase) -> Option<Measurement> {
    if !enabled() {
        return None;
    }
    stats(device).map(|stats| Measurement {
        stats,
        phase,
        started: Instant::now(),
    })
}

pub fn gpu_span(device: &Device, seconds: f64) {
    if !gpu_timestamps() || !seconds.is_finite() || seconds <= 0.0 {
        return;
    }
    if let Some(stats) = stats(device) {
        stats.phases[Phase::GpuSpan as usize]
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push((seconds * 1e9).min(u64::MAX as f64) as u64);
    }
}

pub fn snapshot(device: &Device) -> Value {
    let work = device.work_counts();
    let memory = device.memory_statistics();
    let cache = device.shader_cache_statistics();
    let phases: BTreeMap<_, _> = stats(device)
        .map(|stats| {
            NAMES
                .iter()
                .enumerate()
                .map(|(index, name)| {
                    (
                        *name,
                        stats.phases[index]
                            .lock()
                            .unwrap_or_else(std::sync::PoisonError::into_inner)
                            .value(),
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    json!({"schema":1,"device":device.name(),"timing_enabled":enabled(),"gpu_timestamp_queries_requested":gpu_timestamps(),"phases":phases,
        "memory":{"live_bytes":memory.live_bytes,"reserved_bytes":memory.reserved_bytes,"imported_bytes":memory.imported_bytes},
        "work":{"draws":work.draw_states_recorded,"draw_shapes_made":work.draw_shapes_made,"draw_shapes_found":work.draw_shapes_found,"table_reads":work.stage_tables_read,"pushed_draws":work.draws_pushing_set_zero,"pushed_writes":work.set_zero_writes_pushed,"submits":work.command_buffers_submitted,"submit_ns":work.batch_submit_nanos,"pipeline_jobs_busy":work.pipeline_pool_busy,"pipelines_made":work.variant_pipelines_made,"creation_pipelines_made":work.creation_pipelines_made,"creation_pipelines_drawn":work.creation_pipelines_drawn,"bc_cpu_decodes":work.cpu_block_decodes,"bc_gpu_decodes":work.block_decode_passes},
        "shader_cache":{"persistent":device.shader_cache_is_persistent(),"hits":cache.hits,"misses":cache.misses,"rejected":cache.rejected,"writes":cache.writes,"write_failures":cache.write_failures,"driver_loads":cache.driver_loads,"driver_hits":cache.driver_pipeline_hits,"driver_misses":cache.driver_pipeline_misses},
        "preload_process":{"recorded":crate::preload::RECORDED.load(Ordering::Relaxed),"warmed":crate::preload::WARMED.load(Ordering::Relaxed),"failed":crate::preload::FAILED.load(Ordering::Relaxed),"cache_rejections":crate::cache::REJECTED.load(Ordering::Relaxed)}})
}

pub fn sample(device: &Device, force: bool) {
    if !enabled() {
        return;
    }
    let Some(stats) = stats(device) else {
        return;
    };
    let now = stats.started.elapsed().as_nanos().min(u64::MAX as u128) as u64;
    let last = stats.last_sample.load(Ordering::Relaxed);
    if !force
        && (now.saturating_sub(last) < 1_000_000_000
            || stats
                .last_sample
                .compare_exchange(last, now, Ordering::Relaxed, Ordering::Relaxed)
                .is_err())
    {
        return;
    }
    let mut value = snapshot(device);
    value["elapsed_ns"] = json!(now);
    let Ok(mut line) = serde_json::to_vec(&value) else {
        return;
    };
    line.push(b'\n');
    let mut trace = stats
        .trace
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if let Some(trace) = trace.as_mut() {
        let _write = TRACE_WRITE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if trace
            .file
            .metadata()
            .is_ok_and(|metadata| metadata.len().saturating_add(line.len() as u64) <= MAX_TRACE)
        {
            let _ = trace.file.write_all(&line);
        }
    }
}

pub fn retire(device: &Device) {
    if !enabled() {
        return;
    }
    sample(device, true);
    devices()
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
        .remove(&key(device));
}
