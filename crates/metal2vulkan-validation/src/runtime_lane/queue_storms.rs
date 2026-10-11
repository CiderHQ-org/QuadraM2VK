//! The four-queue storm on the device the cfg selects and, on a Mac's GPU, on Metal in the same
//! test: what light command buffers wait while another queue keeps the GPU busy.
//!
//! One busy queue keeps two command buffers in flight, each one dispatch of one thread spinning
//! (the owned fixture `rfs_agx_yield`'s `agx_yield_lookback`, which takes ticket 1 and spins on a
//! flag no one raises until its cap of loads runs out), its cap grown until one alone takes about
//! [`BUSY`]. Three light queues each commit bursts of [`BURST`] command buffers, each one blit
//! filling 256 bytes of a `Private` buffer, and wait for a burst's last before the next, as a
//! guest rings its doorbells and then waits on a stamp. Two phases of [`PHASE`]: the busy queue
//! alone, then the busy queue and the light queues together. Every input is the lane's own.
//!
//! Metal runs command buffers of different queues side by side: a light burst waits at most for
//! the busy command buffer running when it was committed (on the Apple M4 another queue's work
//! does not start while one long dispatch of one threadgroup runs, on Metal and MoltenVK alike),
//! not behind the busy queue's next one too, as it does where every queue is one `VkQueue`.
//! Where Metal runs (a Mac's GPU), the light bursts' median and 99th percentile, each in units
//! of the busy command buffer's own length on its side (the spin's speed differs between runs
//! and between Metal's compiler and the translation), must stay within half a busy command buffer
//! and one busy command buffer of Metal's, and the busy queue must keep its throughput beside the
//! light queues as well as Metal's does, within a quarter of it (`metal::judge`). In our plain
//! Vulkan programs of the runtime's submissions on an M4 (EVs STL-brst2-m4 and STL-brst3-m4),
//! every queue on MoltenVK's one queue of family 0 (as the runtime once did) put the median 1 to
//! 1.4 busy command buffers above Metal's in the same run; a queue per family, 0.4 at most.
//!
//! On MoltenVK 1.4.1 the lane fails, for MoltenVK's own reason. With Metal argument buffers, which
//! the runtime needs, every encoder waits on the device's barrier fences, which the end of every
//! submission on any queue updates. So the light bursts wait for about two busy command buffers,
//! against Metal's half of one (Apple M4). With argument buffers off they took a fifth of one, and
//! the lane passed. No stock MoltenVK setting avoids the fences; the lane keeps judging against
//! Metal and measures the defect.
//!
//! The runtime's device opens without the validation layer, as Metal runs without its own: the
//! layer's cost on every call is no part of what the storm compares (the lanes that check the
//! device's queues under the layer are `handles`, `command_queues` and the rest).

use super::{validated_device_with, Messages, ScratchCache, VALIDATION_LAYER};
use crate::air_library::AirLibrary;
use crate::gpu::TEST_DEVICE;
use metal2vulkan::pipeline::{ComputePipelineDescriptor, StageSource};
use metal2vulkan_devtools::llvm_assemble;
use metal2vulkan_runtime::{
    Buffer, CommandBuffer, CommandBufferStatus, CommandQueue, ComputePipelineState, Device,
    PipelineOptions, ResourceOptions, Size,
};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};
use std::time::{Duration, Instant};

/// How long one busy command buffer alone takes, at least, once calibrated.
const BUSY: Duration = Duration::from_millis(6);

/// The light queues.
const LIGHTS: usize = 3;

/// The command buffers of a light burst.
const BURST: usize = 8;

/// Each phase's length.
const PHASE: Duration = Duration::from_secs(2);

/// The longest anything waits for a command buffer.
const WAIT: Duration = Duration::from_secs(30);

/// The tries of a storm whose busy work changed speed under it.
const TRIES: usize = 3;

/// The busy kernel's typed AIR text and its LLVM IR, the fixture's.
fn fixture(extension: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(format!("fixtures/public/engine/rfs_agx_yield.{extension}"))
}

/// What the storm needs of a side, the runtime or Metal: a busy command buffer, a light burst,
/// and a wait. Each light thread makes its own target, so a side's buffers stay on the thread
/// that made them.
trait Side: Sync {
    /// A committed command buffer, which the thread that committed it waits for.
    type Committed;
    /// A light queue's buffer.
    type Target;
    /// The side's name, for the report.
    fn name(&self) -> String;
    /// Commit one busy command buffer on the busy queue, its spin capped at `cap` loads.
    fn busy(&self, cap: u32) -> Result<Self::Committed, String>;
    /// A light queue's buffer of 256 bytes, `Private`.
    fn target(&self) -> Result<Self::Target, String>;
    /// Commit [`BURST`] light command buffers on light queue `light`, each filling `target` with
    /// `value`: the last.
    fn burst(
        &self,
        light: usize,
        target: &Self::Target,
        value: u8,
    ) -> Result<Self::Committed, String>;
    /// Wait for `committed`, which must complete.
    fn wait(&self, committed: Self::Committed) -> Result<(), String>;
}

/// A storm's measurements.
#[derive(Debug)]
struct Storm {
    /// One busy command buffer alone, the median of three.
    busy: Duration,
    /// Busy command buffers completed a second, the busy queue alone.
    alone: f64,
    /// Busy command buffers completed a second beside the light queues.
    beside: f64,
    /// Every light burst, from its first commit to its last's completion, sorted.
    bursts: Vec<Duration>,
}

impl Storm {
    /// The bursts' `p`th percentile in busy command buffers.
    fn percentile(&self, p: usize) -> f64 {
        let at = (self.bursts.len() * p / 100).min(self.bursts.len() - 1);
        self.bursts[at].as_secs_f64() / self.busy.as_secs_f64()
    }

    /// The fraction of its throughput alone the busy queue keeps beside the light queues.
    fn kept(&self) -> f64 {
        self.beside / self.alone
    }

    /// The report line.
    fn line(&self, side: &str) -> String {
        format!(
            "STORM {side}: busy {:.2} ms; bursts {} median {:.2} ms ({:.2} busy), p99 {:.2} ms \
             ({:.2} busy); busy {:.0}/s alone, {:.0}/s beside ({:.2} kept)",
            self.busy.as_secs_f64() * 1e3,
            self.bursts.len(),
            self.percentile(50) * self.busy.as_secs_f64() * 1e3,
            self.percentile(50),
            self.percentile(99) * self.busy.as_secs_f64() * 1e3,
            self.percentile(99),
            self.alone,
            self.beside,
            self.kept(),
        )
    }
}

/// One busy command buffer alone at `cap`: the median wall time of three.
fn busy_alone<S: Side>(side: &S, cap: u32) -> Result<Duration, String> {
    let mut times = Vec::new();
    for _ in 0..3 {
        let at = Instant::now();
        side.wait(side.busy(cap)?)?;
        times.push(at.elapsed());
    }
    times.sort();
    Ok(times[1])
}

/// A phase of [`PHASE`]: the busy queue, with the light queues where `lights`. The busy command
/// buffers completed a second, and every light burst's time.
fn phase<S: Side>(side: &S, cap: u32, lights: bool) -> Result<(f64, Vec<Duration>), String> {
    let stop = AtomicBool::new(false);
    let failure: Mutex<Option<String>> = Mutex::new(None);
    let fail = |error: String| {
        failure
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get_or_insert(error);
    };
    let bursts: Mutex<Vec<Duration>> = Mutex::new(Vec::new());
    let busy_rate = std::thread::scope(|scope| {
        let busy = scope.spawn(|| {
            let began = Instant::now();
            let mut flight = std::collections::VecDeque::new();
            let mut done = 0u32;
            while !stop.load(Ordering::SeqCst) {
                if flight.len() == 2 {
                    if let Err(error) = side.wait(flight.pop_front().expect("two in flight")) {
                        fail(error);
                        return 0.0;
                    }
                    done += 1;
                }
                match side.busy(cap) {
                    Ok(committed) => flight.push_back(committed),
                    Err(error) => {
                        fail(error);
                        return 0.0;
                    }
                }
            }
            let elapsed = began.elapsed();
            for committed in flight {
                if let Err(error) = side.wait(committed) {
                    fail(error);
                }
            }
            f64::from(done) / elapsed.as_secs_f64()
        });
        let light_threads: Vec<_> = (0..if lights { LIGHTS } else { 0 })
            .map(|light| {
                let (stop, bursts, fail) = (&stop, &bursts, &fail);
                scope.spawn(move || {
                    let run = || -> Result<Vec<Duration>, String> {
                        let target = side.target()?;
                        // The light queue's first burst, not timed.
                        side.wait(side.burst(light, &target, 0)?)?;
                        let mut times = Vec::new();
                        let mut value = 0u8;
                        while !stop.load(Ordering::SeqCst) {
                            value = value.wrapping_add(1);
                            let at = Instant::now();
                            side.wait(side.burst(light, &target, value)?)?;
                            times.push(at.elapsed());
                        }
                        Ok(times)
                    };
                    match run() {
                        Ok(times) => bursts
                            .lock()
                            .unwrap_or_else(PoisonError::into_inner)
                            .extend(times),
                        Err(error) => fail(error),
                    }
                })
            })
            .collect();
        std::thread::sleep(PHASE);
        stop.store(true, Ordering::SeqCst);
        for thread in light_threads {
            thread.join().expect("a light thread");
        }
        busy.join().expect("the busy thread")
    });
    if let Some(error) = failure.into_inner().unwrap_or_else(PoisonError::into_inner) {
        return Err(error);
    }
    let mut bursts = bursts.into_inner().unwrap_or_else(PoisonError::into_inner);
    bursts.sort();
    Ok((busy_rate, bursts))
}

/// The storm on `side`: the busy work calibrated, then the two phases, tried again (at most
/// [`TRIES`] times) where the busy work's length after the phases is not within half and twice
/// its length before them (the spin's speed changes between runs, and a storm measured across
/// such a change compares nothing).
fn storm<S: Side>(side: &S) -> Result<Storm, String> {
    let mut last = String::new();
    for _ in 0..TRIES {
        // The kernel's first use, then its cap grown until it takes `BUSY`.
        side.wait(side.busy(1 << 8)?)?;
        let mut cap: u32 = 1 << 8;
        let busy = loop {
            let took = busy_alone(side, cap)?;
            if took >= BUSY || cap >= 1 << 28 {
                break took;
            }
            cap = cap.saturating_mul(2);
        };
        let (alone, _) = phase(side, cap, false)?;
        let (beside, bursts) = phase(side, cap, true)?;
        let after = busy_alone(side, cap)?;
        if bursts.is_empty() {
            return Err(format!("{}: no light burst completed", side.name()));
        }
        if after * 2 >= busy && after <= busy * 2 {
            return Ok(Storm {
                busy,
                alone,
                beside,
                bursts,
            });
        }
        last = format!(
            "{}: the busy work took {busy:?} before the phases and {after:?} after",
            side.name()
        );
        eprintln!("STORM retry: {last}");
    }
    Err(last)
}

/// The runtime's side: its device, a busy queue and [`LIGHTS`] light queues, and the busy
/// kernel's state.
struct Runtime {
    device: Device,
    queues: Vec<CommandQueue>,
    state: ComputePipelineState,
}

impl Runtime {
    fn new(device: Device) -> Runtime {
        let library = device
            .new_library_with_air(&llvm_assemble(&fixture("ll")).unwrap())
            .unwrap();
        let function = library
            .new_function_with_name("agx_yield_lookback")
            .unwrap();
        let state = device
            .new_compute_pipeline_state(
                &ComputePipelineDescriptor::new(StageSource::Function(
                    function.function_descriptor(),
                )),
                &PipelineOptions::default(),
            )
            .unwrap();
        let queues = (0..=LIGHTS)
            .map(|_| device.new_command_queue().unwrap())
            .collect();
        Runtime {
            device,
            queues,
            state,
        }
    }
}

impl Side for Runtime {
    type Committed = CommandBuffer;
    type Target = Buffer;

    fn name(&self) -> String {
        format!("the runtime on {}", self.device.name())
    }

    fn busy(&self, cap: u32) -> Result<CommandBuffer, String> {
        let failed = |error: metal2vulkan_runtime::Error| format!("a busy command buffer: {error}");
        let with = |bytes: &[u8]| {
            self.device
                .new_buffer_with_bytes(bytes, ResourceOptions::shared())
                .map_err(failed)
        };
        let (ticket, words, cap) = (
            with(&1u32.to_le_bytes())?,
            with(&[0; 8])?,
            with(&cap.to_le_bytes())?,
        );
        let command_buffer = self.queues[0].command_buffer().map_err(failed)?;
        let mut encoder = command_buffer.compute_command_encoder().map_err(failed)?;
        encoder
            .set_compute_pipeline_state(&self.state)
            .map_err(failed)?;
        encoder.set_buffer(Some(&ticket), 0, 0).map_err(failed)?;
        encoder.set_buffer(Some(&words), 0, 1).map_err(failed)?;
        encoder.set_buffer(Some(&cap), 0, 2).map_err(failed)?;
        let one = Size::new(1, 1, 1);
        encoder.dispatch_threadgroups(one, one).map_err(failed)?;
        encoder.end_encoding().map_err(failed)?;
        command_buffer.commit().map_err(failed)?;
        Ok(command_buffer)
    }

    fn target(&self) -> Result<Buffer, String> {
        self.device
            .new_buffer(256, ResourceOptions::private())
            .map_err(|error| format!("a light buffer: {error}"))
    }

    fn burst(&self, light: usize, target: &Buffer, value: u8) -> Result<CommandBuffer, String> {
        let failed =
            |error: metal2vulkan_runtime::Error| format!("a light command buffer: {error}");
        let mut last = None;
        for _ in 0..BURST {
            let command_buffer = self.queues[1 + light].command_buffer().map_err(failed)?;
            let mut blit = command_buffer.blit_command_encoder().map_err(failed)?;
            blit.fill_buffer(target, 0..256, value).map_err(failed)?;
            blit.end_encoding().map_err(failed)?;
            command_buffer.commit().map_err(failed)?;
            last = Some(command_buffer);
        }
        last.ok_or_else(|| "an empty burst".to_owned())
    }

    fn wait(&self, command_buffer: CommandBuffer) -> Result<(), String> {
        command_buffer
            .wait_until_completed_for(WAIT)
            .map_err(|_| format!("a command buffer did not complete within {WAIT:?}"))?;
        match command_buffer.status() {
            CommandBufferStatus::Completed => Ok(()),
            status => Err(format!(
                "a command buffer ended {status:?}: {:?}",
                command_buffer.error()
            )),
        }
    }
}

/// Plain Metal's side on this Mac's GPU: a busy queue and [`LIGHTS`] light queues, and the busy
/// kernel's state, from the fixture's typed AIR text.
#[cfg(all(m2v_gpu, target_os = "macos"))]
mod metal {
    use super::{fixture, Side, Storm, BURST};
    use crate::gpu::compile;
    use crate::gpu::wait::metal::{settle_watched, watch, Watch};
    use metal2vulkan_devtools::metal_assemble;
    use objc2::rc::Retained;
    use objc2::runtime::ProtocolObject;
    use objc2_foundation::{NSRange, NSString};
    use objc2_metal::{
        MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
        MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLLibrary,
        MTLResourceOptions, MTLSize,
    };
    use std::ffi::c_void;
    use std::ptr::NonNull;

    /// How much longer than Metal's a light burst's median may be, in busy command buffers.
    const MEDIAN_MARGIN: f64 = 0.5;

    /// How much longer than Metal's a light burst's 99th percentile may be, in busy command
    /// buffers.
    const P99_MARGIN: f64 = 1.0;

    /// How much more of its throughput alone than Metal's the busy queue may lose beside the
    /// light queues, as a fraction of its throughput alone.
    const BUSY_MARGIN: f64 = 0.25;

    /// The runtime's storm `ours` against Metal's `metal` (the module docs).
    pub(super) fn judge(ours: &Storm, metal: &Storm) {
        let (median, metal_median) = (ours.percentile(50), metal.percentile(50));
        assert!(
            median <= metal_median + MEDIAN_MARGIN,
            "a light burst's median is {median:.2} busy command buffers, Metal's {metal_median:.2}"
        );
        let (p99, metal_p99) = (ours.percentile(99), metal.percentile(99));
        assert!(
            p99 <= metal_p99 + P99_MARGIN,
            "a light burst's 99th percentile is {p99:.2} busy command buffers, Metal's \
             {metal_p99:.2}"
        );
        assert!(
            ours.kept() >= metal.kept() - BUSY_MARGIN,
            "the busy queue keeps {:.2} of its throughput beside the light queues, Metal's {:.2}",
            ours.kept(),
            metal.kept()
        );
    }

    pub(super) struct Metal {
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        queues: Vec<Retained<ProtocolObject<dyn MTLCommandQueue>>>,
        state: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
    }

    impl Metal {
        pub(super) fn new() -> Metal {
            let device = crate::metal::system_device().unwrap();
            let air = metal_assemble(&fixture("air.ll")).unwrap();
            let library = crate::metal::load_library(&device, &air, None).unwrap();
            let function = library
                .newFunctionWithName(&NSString::from_str("agx_yield_lookback"))
                .expect("the busy kernel");
            // Under the compile deadline, as every Metal compilation of the crate
            // (`gpu/compile.rs`).
            let state = compile::bounded("the storm's busy kernel", || {
                device.newComputePipelineStateWithFunction_error(&function)
            })
            .unwrap();
            let queues = (0..=super::LIGHTS)
                .map(|_| device.newCommandQueue().expect("a Metal queue"))
                .collect();
            Metal {
                device,
                queues,
                state,
            }
        }

        /// A `Shared` buffer holding `bytes`.
        fn with(&self, bytes: &[u8]) -> Result<Retained<ProtocolObject<dyn MTLBuffer>>, String> {
            let pointer =
                NonNull::new(bytes.as_ptr().cast_mut().cast::<c_void>()).ok_or("a null slice")?;
            // SAFETY: `pointer` addresses `bytes.len()` live bytes, which Metal copies.
            unsafe {
                self.device.newBufferWithBytes_length_options(
                    pointer,
                    bytes.len(),
                    MTLResourceOptions::StorageModeShared,
                )
            }
            .ok_or_else(|| "a Metal buffer".to_owned())
        }
    }

    impl Side for Metal {
        type Committed = (Retained<ProtocolObject<dyn MTLCommandBuffer>>, Watch);
        type Target = Retained<ProtocolObject<dyn MTLBuffer>>;

        fn name(&self) -> String {
            format!("Metal on {}", self.device.name())
        }

        fn busy(&self, cap: u32) -> Result<Self::Committed, String> {
            let buffers = [
                self.with(&1u32.to_le_bytes())?,
                self.with(&[0; 8])?,
                self.with(&cap.to_le_bytes())?,
            ];
            let command_buffer = self.queues[0].commandBuffer().ok_or("a command buffer")?;
            let encoder = command_buffer
                .computeCommandEncoder()
                .ok_or("a compute encoder")?;
            encoder.setComputePipelineState(&self.state);
            for (index, buffer) in buffers.iter().enumerate() {
                // SAFETY: the command buffer retains the buffer, bound at the index the kernel
                // reads, at offset 0.
                unsafe { encoder.setBuffer_offset_atIndex(Some(buffer), 0, index) };
            }
            let one = MTLSize {
                width: 1,
                height: 1,
                depth: 1,
            };
            encoder.dispatchThreadgroups_threadsPerThreadgroup(one, one);
            encoder.endEncoding();
            let watched = watch(&command_buffer);
            command_buffer.commit();
            Ok((command_buffer, watched))
        }

        fn target(&self) -> Result<Self::Target, String> {
            self.device
                .newBufferWithLength_options(256, MTLResourceOptions::StorageModePrivate)
                .ok_or_else(|| "a Metal light buffer".to_owned())
        }

        fn burst(
            &self,
            light: usize,
            target: &Self::Target,
            value: u8,
        ) -> Result<Self::Committed, String> {
            let mut last = None;
            for _ in 0..BURST {
                let command_buffer = self.queues[1 + light]
                    .commandBuffer()
                    .ok_or("a command buffer")?;
                let blit = command_buffer
                    .blitCommandEncoder()
                    .ok_or("a blit encoder")?;
                blit.fillBuffer_range_value(target, NSRange::new(0, 256), value);
                blit.endEncoding();
                let watched = watch(&command_buffer);
                command_buffer.commit();
                last = Some((command_buffer, watched));
            }
            last.ok_or_else(|| "an empty burst".to_owned())
        }

        fn wait(&self, (command_buffer, watched): Self::Committed) -> Result<(), String> {
            settle_watched(&command_buffer, watched, "the queue storm")
        }
    }
}

/// The four-queue storm on the runtime's device: every command buffer completes, the busy queue
/// runs in both phases and the light queues beside it. Where Metal runs, Metal's storm runs
/// first in the same test, and the runtime's light bursts wait no more than Metal's by more than
/// the margins, and its busy queue keeps its throughput as Metal's does (the module docs).
#[test]
fn four_queues_storm_as_metal_s_do() {
    #[cfg(all(m2v_gpu, target_os = "macos"))]
    let metal = {
        let metal = storm(&metal::Metal::new()).unwrap();
        eprintln!("{}", metal.line("Metal"));
        metal
    };
    let cache = ScratchCache::new().unwrap();
    let messages = Messages::default();
    let ours = {
        // Without the validation layer (the module docs).
        let device = validated_device_with(TEST_DEVICE, &messages, cache.path(), |options| {
            options
                .layers
                .retain(|layer| layer.as_c_str() != VALIDATION_LAYER)
        })
        .unwrap();
        let runtime = Runtime::new(device);
        // SAFETY: read while the device lives.
        let families = unsafe { runtime.device.raw() }.queue_families.to_vec();
        let ours = storm(&runtime).unwrap();
        eprintln!(
            "{} (queue families {families:?})",
            ours.line(&runtime.name())
        );
        ours
    };
    assert!(ours.alone > 0.0 && ours.beside > 0.0, "{ours:?}");
    #[cfg(all(m2v_gpu, target_os = "macos"))]
    metal::judge(&ours, &metal);
    let messages = messages.take();
    assert!(messages.is_empty(), "the driver reported: {messages:#?}");
}
